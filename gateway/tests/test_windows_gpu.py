"""GPU telemetry on Windows, where there is no sysfs and no nvidia-smi.

gpu-desktop-1 is the reason this exists: a 16 GB Radeon whose fleet card rendered as
"n/a — no GPU telemetry" with no VRAM row at all, because both existing
backends are for other platforms.
"""
from __future__ import annotations

import json

import pytest

import app as gw


class FakePS:
    """Stands in for the PowerShell round trip."""

    def __init__(self, payload, rc: int = 0):
        self.payload = payload
        self.rc = rc
        self.calls = 0

    def __call__(self, cmd, **kwargs):
        self.calls += 1

        class P:
            returncode = self.rc
            stdout = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
            stderr = ""

        return P()


@pytest.fixture(autouse=True)
def _clear_cache():
    gw._wingpu_cache.update(t=0.0, gpus=[])
    yield
    gw._wingpu_cache.update(t=0.0, gpus=[])


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(gw.platform, "system", lambda: "Windows")


DISCRETE = {"card": "AMD Radeon RX 9060 XT", "vram_total": 17_112_170_496,
            "vram_used": 15_192_231_936, "busy_percent": 87.6}
IGPU = {"card": "AMD Radeon(TM) Graphics", "vram_total": 536_870_912,
        "vram_used": 201_326_592, "busy_percent": 3.0}


class TestOtherPlatforms:
    def test_linux_returns_nothing_and_never_shells_out(self, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        ps = FakePS([DISCRETE])
        monkeypatch.setattr(gw.subprocess, "run", ps)
        assert gw.windows_gpu_stats() == []
        assert ps.calls == 0


class TestWindows:
    def test_reports_the_card(self, on_windows, monkeypatch):
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        (g,) = gw.windows_gpu_stats()
        assert g["card"] == "AMD Radeon RX 9060 XT"
        assert g["vram_total"] == 17_112_170_496
        assert g["vram_used"] == 15_192_231_936
        assert g["busy_percent"] == 87.6

    def test_fields_the_counters_cannot_know_stay_none(self, on_windows, monkeypatch):
        """Better an honest blank than an invented temperature: the dashboard
        already omits what is None, and a plausible zero would read as a
        measurement."""
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        (g,) = gw.windows_gpu_stats()
        for k in ("temp_c", "power_w", "sclk_mhz", "gtt_used", "gtt_total"):
            assert g[k] is None

    def test_shape_matches_the_other_backends(self, on_windows, monkeypatch):
        """host_status() concatenates all three lists, so they have to agree
        on keys or the dashboard reads undefined off one of them."""
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        (g,) = gw.windows_gpu_stats()
        assert set(g) == {"card", "busy_percent", "vram_total", "vram_used",
                          "gtt_used", "gtt_total", "temp_c", "power_w", "sclk_mhz"}

    def test_the_igpu_is_left_out(self, on_windows, monkeypatch):
        """gpu-desktop-1 has two 'GPUs': the 9060 XT and the 9600X's integrated
        Radeon, which reports a few hundred MB carved out of system RAM.
        Nothing routes work to it, and counting it would make the fleet's
        pooled VRAM total wrong."""
        monkeypatch.setattr(gw.subprocess, "run", FakePS([IGPU, DISCRETE]))
        cards = [g["card"] for g in gw.windows_gpu_stats()]
        assert cards == ["AMD Radeon RX 9060 XT"]

    def test_two_discrete_cards_drop_the_unattributable_figure(self, on_windows, monkeypatch):
        """The dedicated-usage counter is keyed by LUID, not by card name, so
        with two real cards it cannot be attributed to either. A number that
        looks measured and is not is worse than a blank."""
        second = dict(DISCRETE, card="AMD Radeon RX 7900 XTX")
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE, second]))
        got = gw.windows_gpu_stats()
        assert len(got) == 2
        assert all(g["vram_used"] is None for g in got)
        assert all(g["vram_total"] for g in got), "totals are per-card and stay"

    def test_result_is_cached(self, on_windows, monkeypatch):
        ps = FakePS([DISCRETE])
        monkeypatch.setattr(gw.subprocess, "run", ps)
        gw.windows_gpu_stats()
        gw.windows_gpu_stats()
        assert ps.calls == 1, "a PowerShell start per dashboard poll is the cost this avoids"


class TestFailuresAreQuiet:
    """A box with no readable counters reports no GPU. It does not 500 the
    status endpoint that the whole dashboard hangs off."""

    def test_bad_json(self, on_windows, monkeypatch):
        monkeypatch.setattr(gw.subprocess, "run", FakePS("not json at all"))
        assert gw.windows_gpu_stats() == []

    def test_powershell_missing(self, on_windows, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("powershell")

        monkeypatch.setattr(gw.subprocess, "run", boom)
        assert gw.windows_gpu_stats() == []

    def test_empty_output(self, on_windows, monkeypatch):
        monkeypatch.setattr(gw.subprocess, "run", FakePS(""))
        assert gw.windows_gpu_stats() == []


class TestVramTotal:
    def test_measured_vram_beats_the_spec_sheet(self, on_windows, monkeypatch):
        """resolve_ctx() sizes context from this number, so a Windows box
        should now be sized by what the driver reports rather than by whatever
        DEFAULT_SPECS says it was fitted with."""
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        assert gw.vram_total_bytes() == 17_112_170_496

    def test_falls_back_to_the_spec_sheet_when_unreadable(self, on_windows, monkeypatch):
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakePS(""))
        monkeypatch.setattr(gw, "HOST_NAME", "gpu-desktop-1")
        monkeypatch.setattr(gw, "load_specs", lambda: {"gpu-desktop-1": {"vram_gb": 16}})
        assert gw.vram_total_bytes() == 16 * 1024 ** 3


class TestGhostAdapters:
    """The registry class key never forgets a card. gpu-laptop-2 still carries an
    "AMD Radeon RX 7800M" claiming 11.98 GiB from an eGPU that is long gone,
    and sized off the registry alone that ghost outranked the real 4070 --
    the fleet page showed the laptop as a 12 GB box and, worse, resolve_ctx()
    sized the context windows it PROMISES the fleet against memory that is
    not there. Same phantom that once had mini-pc-1 advertising 12 GiB.

    The gate lives in PowerShell, which cannot run here, so these pin the
    script's shape. Verified against the three real Windows boxes: gpu-laptop-2
    drops the 7800M and keeps the 4070, mini-pc-1 drops it and is left with
    only its iGPU, gpu-desktop-1's RX 9060 XT is untouched."""

    def test_the_registry_sizes_are_gated_on_a_present_adapter(self):
        assert "Get-CimInstance Win32_VideoController" in gw._WIN_GPU_PS
        assert "$present -contains" in gw._WIN_GPU_PS

    def test_an_unmatched_gate_falls_back_rather_than_reporting_no_gpu(self):
        # A missing VRAM row reads as "no GPU", which is gpu-desktop-1's whole reason
        # for having this backend -- so a naming convention this does not
        # recognise must degrade to the old behaviour, not to silence.
        assert "if ($live.Count -gt 0) { $total = $live }" in gw._WIN_GPU_PS

    def test_a_ghost_that_survives_is_still_bounded_by_the_igpu_filter(
            self, on_windows, monkeypatch):
        # The 2 GB floor cannot save us from this one -- the ghost is bigger
        # than the real card, which is exactly why the gate had to exist.
        ghost = {"card": "AMD Radeon RX 7800M", "vram_total": 12_868_124_672,
                 "vram_used": None, "busy_percent": 0}
        real = {"card": "NVIDIA GeForce RTX 4070 Laptop GPU",
                "vram_total": 8_585_740_288, "vram_used": None, "busy_percent": 0}
        monkeypatch.setattr(gw.subprocess, "run", FakePS([ghost, real]))
        cards = {g["card"]: g["vram_total"] for g in gw.windows_gpu_stats()}
        assert cards == {ghost["card"]: ghost["vram_total"],
                         real["card"]: real["vram_total"]}


class TestItIsAFallbackNotAThirdOpinion:
    """gpu-laptop-2 is a Windows box WITH nvidia-smi. Both backends describe its
    one RTX 4070, so listing both showed the laptop with two cards -- one
    carrying a temperature and no VRAM, one carrying VRAM and no temperature.
    """

    def test_the_vendor_backend_wins_when_it_answers(self, on_windows, monkeypatch):
        nv = dict(DISCRETE, card="NVIDIA GeForce RTX 4070 Laptop GPU", temp_c=52.0)
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [nv])
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        got = gw.gpu_cards()
        assert [g["card"] for g in got] == ["NVIDIA GeForce RTX 4070 Laptop GPU"]
        assert got[0]["temp_c"] == 52.0, "the richer reading is the one kept"

    def test_the_counters_answer_when_no_vendor_tool_does(self, on_windows, monkeypatch):
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakePS([DISCRETE]))
        assert [g["card"] for g in gw.gpu_cards()] == ["AMD Radeon RX 9060 XT"]

    def test_vram_is_not_counted_twice(self, on_windows, monkeypatch):
        """The number here sizes the context windows this box then PROMISES
        to the fleet, so a card counted twice is an OOM waiting to happen."""
        nv = dict(DISCRETE, card="NVIDIA GeForce RTX 4070 Laptop GPU",
                  vram_total=8_585_740_288)
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [nv])
        monkeypatch.setattr(gw.subprocess, "run", FakePS([dict(DISCRETE, vram_total=8_585_740_288)]))
        assert gw.vram_total_bytes() == 8_585_740_288
