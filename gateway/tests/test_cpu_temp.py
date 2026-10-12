"""host_status()'s cpu_temp_c: the chip that actually throttles.

apu-box-1's 2026-09-18 freeze (see hosts/apu-box-1/README.md) had a comfortable
amdgpu 'edge' reading -- the only temperature gpu_cards() exposes, and the
only one the dashboard renders -- while the CPU package sensor was 30-50 C
higher and climbing. `psutil.sensors_temperatures()` already carries that
number in host_status()'s generic `temps` dict; nothing surfaced it under a
name a client could rely on across a fleet of different chips. This is that
name.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

from types import SimpleNamespace

import app as gw


def _temps(monkeypatch, table: dict) -> None:
    """table: {"chip": [(label, current), ...]}, the shape
    psutil.sensors_temperatures() returns.

    raising=False: the attribute exists only on Linux/FreeBSD builds of
    psutil, and this suite has to pass on whatever OS runs it -- the same
    reason app.py's own call is wrapped in try/except Exception."""
    out = {chip: [SimpleNamespace(label=label, current=current)
                  for label, current in entries]
           for chip, entries in table.items()}
    monkeypatch.setattr(gw.psutil, "sensors_temperatures", lambda: out, raising=False)


class TestCpuPackageTemp:
    def test_k10temp_tctl_is_read_on_an_amd_box(self, monkeypatch):
        _temps(monkeypatch, {"k10temp": [("Tctl", 93.0), ("Tccd1", 71.0)],
                             "amdgpu": [("edge", 61.0)]})
        assert gw.host_status()["cpu_temp_c"] == 93.0

    def test_coretemp_package_id_0_is_read_on_an_intel_box(self, monkeypatch):
        _temps(monkeypatch, {"coretemp": [("Package id 0", 71.0),
                                         ("Core 0", 65.0)]})
        assert gw.host_status()["cpu_temp_c"] == 71.0

    def test_a_chip_with_neither_key_is_none_not_a_guess(self, monkeypatch):
        """Apple Silicon (no psutil sensor backend at all) and any chip this
        list has not met yet must not fall back to an unrelated per-core or
        per-chiplet reading."""
        _temps(monkeypatch, {})
        assert gw.host_status()["cpu_temp_c"] is None

        _temps(monkeypatch, {"nvme": [("Composite", 42.0)]})
        assert gw.host_status()["cpu_temp_c"] is None

    def test_sensors_temperatures_raising_still_leaves_a_usable_status(
            self, monkeypatch):
        """psutil.sensors_temperatures() raises NotImplementedError on
        platforms with no sensor backend at all -- host_status() already
        swallows that for `temps`; cpu_temp_c must come out None, not crash
        the whole endpoint the hub polls to decide 'online'."""
        def boom():
            raise NotImplementedError

        monkeypatch.setattr(gw.psutil, "sensors_temperatures", boom, raising=False)
        hs = gw.host_status()
        assert hs["cpu_temp_c"] is None
        assert hs["temps"] == {}
        assert hs["mem"]["total"] > 0

    def test_never_leaks_into_the_public_overview(self, client, fake_fleet,
                                                   monkeypatch):
        _temps(monkeypatch, {"k10temp": [("Tctl", 93.0)]})
        r = client.get("/public/api/overview")
        assert r.status_code == 200
        assert "cpu_temp_c" not in r.text
