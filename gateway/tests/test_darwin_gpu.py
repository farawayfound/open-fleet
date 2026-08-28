"""GPU telemetry on the Macs, where there is no sysfs, no nvidia-smi and no
Windows performance counter.

mac-laptop-1 is the reason this exists. It is the fleet's always-on primary inference
node, and its Overview page said "n/a -- no GPU telemetry" while `ioreg` on the
box itself reported the 32-core GPU at 76% with 20.7 GiB mapped. The fleet Home
page hid the gap rather than showing it: with no cards to aggregate it fell
back to the spec sheet for VRAM and multiplied a zero busy figure by the spec
sheet's TFLOPS, so it rendered plausible-looking numbers instead of an obvious
hole.

The plist below is the real `ioreg -r -d 1 -w 0 -a -c IOAccelerator` shape,
captured from mac-laptop-1 (macOS 26.6.2, Darwin 25.6.0, arm64 T6000).
"""
from __future__ import annotations

import plistlib

import pytest

import app as gw


def ioreg_plist(entries) -> bytes:
    return plistlib.dumps(entries, fmt=plistlib.FMT_XML)


# Trimmed to the keys the backend reads, in the shape ioreg really emits.
M1_MAX = {
    "IOClass": "AGXAcceleratorG13X",
    "model": "Apple M1 Max",
    "gpu-core-count": 32,
    "IONameMatched": "gpu,t6000",
    "PerformanceStatistics": {
        "Alloc system memory": 22_906_798_080,
        "Allocated PB Size": 9_175_040,
        "Device Utilization %": 76,
        "In use system memory": 22_186_737_664,
        "In use system memory (driver)": 0,
        "Renderer Utilization %": 3,
        "SplitSceneCount": 0,
        "TiledSceneBytes": 1_933_312,
        "Tiler Utilization %": 3,
        "lastRecoveryTime": 0,
        "recoveryCount": 0,
    },
}


class FakeRun:
    """Stands in for both shell-outs the backend makes: ioreg and sysctl."""

    def __init__(self, plist: bytes | None, wired_limit_mb: str = "43008"):
        self.plist = plist
        self.wired_limit_mb = wired_limit_mb
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.cmds.append(list(cmd))
        outer = self

        class P:
            returncode = 0
            if cmd[0] == "ioreg":
                stdout = outer.plist
                stderr = b""
            else:
                stdout = outer.wired_limit_mb
                stderr = ""

        return P()


@pytest.fixture(autouse=True)
def _clear_cache():
    gw._macgpu_cache.update(t=0.0, gpus=[])
    yield
    gw._macgpu_cache.update(t=0.0, gpus=[])


@pytest.fixture
def on_darwin(monkeypatch):
    monkeypatch.setattr(gw.platform, "system", lambda: "Darwin")


class TestOtherPlatforms:
    @pytest.mark.parametrize("system", ["Linux", "Windows"])
    def test_returns_nothing_and_never_shells_out(self, monkeypatch, system):
        monkeypatch.setattr(gw.platform, "system", lambda: system)
        run = FakeRun(ioreg_plist([M1_MAX]))
        monkeypatch.setattr(gw.subprocess, "run", run)
        assert gw.darwin_gpu_stats() == []
        assert run.cmds == []


class TestReadsTheAccelerator:
    def test_utilisation_and_mapped_memory_are_reported(self, monkeypatch, on_darwin):
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([M1_MAX])))
        (g,) = gw.darwin_gpu_stats()
        assert g["busy_percent"] == 76.0
        assert g["vram_used"] == 22_186_737_664
        assert g["card"] == "Apple M1 Max (32-core)"

    def test_the_device_figure_wins_over_the_renderer_and_tiler_parts(
            self, monkeypatch, on_darwin):
        """Renderer and Tiler are components of Device Utilization, not extra
        load on top of it -- summing them would report more than 100%."""
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([M1_MAX])))
        (g,) = gw.darwin_gpu_stats()
        assert g["busy_percent"] == 76.0

    def test_a_zero_busy_reading_is_kept_as_zero_not_dropped(
            self, monkeypatch, on_darwin):
        """An idle GPU honestly reports 0. That must not read as "unknown"."""
        idle = dict(M1_MAX, PerformanceStatistics=dict(
            M1_MAX["PerformanceStatistics"], **{"Device Utilization %": 0}))
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([idle])))
        (g,) = gw.darwin_gpu_stats()
        assert g["busy_percent"] == 0.0

    def test_no_gtt_temperature_power_or_clocks_are_invented(
            self, monkeypatch, on_darwin):
        """A GTT window is an amdgpu concept, and the rest needs root
        powermetrics. None, never a fabricated zero."""
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([M1_MAX])))
        (g,) = gw.darwin_gpu_stats()
        for k in ("gtt_used", "gtt_total", "temp_c", "power_w", "sclk_mhz"):
            assert g[k] is None

    def test_entries_without_performance_statistics_are_skipped(
            self, monkeypatch, on_darwin):
        bare = {"IOClass": "IOAccelerator", "model": "something else"}
        monkeypatch.setattr(gw.subprocess, "run",
                            FakeRun(ioreg_plist([bare, M1_MAX])))
        cards = gw.darwin_gpu_stats()
        assert len(cards) == 1
        assert cards[0]["busy_percent"] == 76.0


class TestUnifiedMemoryCeiling:
    def test_the_metal_wired_limit_is_the_vram_total(self, monkeypatch, on_darwin):
        """mac-laptop-1 pins iogpu.wired_limit_mb=43008 to stay a usable laptop, and
        42 GiB -- not the 64 GB fitted -- is what a model may occupy."""
        monkeypatch.setattr(gw.subprocess, "run",
                            FakeRun(ioreg_plist([M1_MAX]), wired_limit_mb="43008"))
        (g,) = gw.darwin_gpu_stats()
        assert g["vram_total"] == 42 * 1024 ** 3

    def test_without_a_cap_it_falls_back_to_the_spec_sheet(
            self, monkeypatch, on_darwin):
        """Which is exactly where vram_total_bytes() got the Macs' figure
        before this backend existed, so an uncapped box reports what it
        always did."""
        monkeypatch.setattr(gw, "HOST_NAME", "mac-desktop-1")
        monkeypatch.setattr(gw.subprocess, "run",
                            FakeRun(ioreg_plist([M1_MAX]), wired_limit_mb="0"))
        (g,) = gw.darwin_gpu_stats()
        assert g["vram_total"] == int(
            gw.load_specs()["mac-desktop-1"]["vram_gb"] * 1024 ** 3)

    def test_two_accelerators_get_no_vram_total_attributed(
            self, monkeypatch, on_darwin):
        """Neither the wired limit nor the spec sheet belongs to any one card
        on a two-GPU Mac, so nothing is claimed for either."""
        other = dict(M1_MAX, model="Radeon Pro 5500M", IOClass="AMDAccelerator")
        monkeypatch.setattr(gw.subprocess, "run",
                            FakeRun(ioreg_plist([M1_MAX, other])))
        cards = gw.darwin_gpu_stats()
        assert len(cards) == 2
        assert all(c["vram_total"] is None for c in cards)


class TestFailureIsSilent:
    def test_unparsable_output_is_no_telemetry_not_a_crash(
            self, monkeypatch, on_darwin):
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(b"not a plist"))
        assert gw.darwin_gpu_stats() == []

    def test_empty_output_is_no_telemetry(self, monkeypatch, on_darwin):
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(b""))
        assert gw.darwin_gpu_stats() == []

    def test_a_raising_ioreg_is_no_telemetry(self, monkeypatch, on_darwin):
        def boom(cmd, **kwargs):
            raise OSError("ioreg: command not found")

        monkeypatch.setattr(gw.subprocess, "run", boom)
        assert gw.darwin_gpu_stats() == []


class TestCaching:
    def test_a_second_call_inside_the_ttl_does_not_shell_out_again(
            self, monkeypatch, on_darwin):
        run = FakeRun(ioreg_plist([M1_MAX]))
        monkeypatch.setattr(gw.subprocess, "run", run)
        gw.darwin_gpu_stats()
        first = len(run.cmds)
        gw.darwin_gpu_stats()
        assert len(run.cmds) == first


class TestWiredIntoTheFleet:
    def test_gpu_cards_uses_it_when_no_other_backend_answers(
            self, monkeypatch, on_darwin):
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([M1_MAX])))
        (g,) = gw.gpu_cards()
        assert g["card"] == "Apple M1 Max (32-core)"
        assert g["busy_percent"] == 76.0

    def test_vram_total_bytes_reads_the_metal_ceiling(self, monkeypatch, on_darwin):
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(ioreg_plist([M1_MAX])))
        assert gw.vram_total_bytes() == 42 * 1024 ** 3

    def test_a_mac_with_no_ioreg_answer_still_reports_the_spec_sheet(
            self, monkeypatch, on_darwin):
        """The pre-existing fallback in vram_total_bytes() must survive: a box
        this backend cannot read is sized exactly as it was before."""
        monkeypatch.setattr(gw, "HOST_NAME", "mac-laptop-1")
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [])
        monkeypatch.setattr(gw.subprocess, "run", FakeRun(b""))
        assert gw.vram_total_bytes() == int(
            gw.load_specs()["mac-laptop-1"]["vram_gb"] * 1024 ** 3)
