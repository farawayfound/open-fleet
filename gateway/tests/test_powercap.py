"""gateway/bin/llmstack-powercap, exercised against a fake sysfs tree via its
own LLMSTACK_POWERCAP_ROOT/_CONF/_WAIT_SECS hooks -- no root, no real
amdgpu or cpufreq, the same trick test_review_fixes.py's
TestHelperOnASandboxTree plays against llmstack-gpuconf.

One honest limitation, worth saying up front rather than discovering by
reading a test and wondering why it looks incomplete: `pp_od_clk_voltage` on
a real box is a synthetic sysfs attribute -- a write is a command consumed by
the driver, and a read calls back into the driver for the CURRENT hardware
state, which is why the helper's `s 1 <MHz>` + `c` sequence changes what a
subsequent read reports. A plain file has no driver behind it, so writing to
one in a test tree only ever changes what is literally in the file; it can
never show a "the clock actually moved" readback. What IS tested here,
against that same plain file, is exactly what does not depend on driver
behaviour: that the right commands are sent, in order, only when a value
validates, and never when a knob is unset or a card cannot be found -- and,
separately, that `status`'s PARSER reads a driver's post-commit dump (typed
in by hand, the way a real card would render one) correctly.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parent.parent / "bin" / "llmstack-powercap"


def run(root: Path, *args: str, wait: str = "1") -> subprocess.CompletedProcess:
    env = {"LLMSTACK_POWERCAP_ROOT": str(root),
           "LLMSTACK_POWERCAP_WAIT_SECS": wait,
           "PATH": "/usr/bin:/bin"}
    return subprocess.run(["/bin/bash", str(HELPER), *args],
                          capture_output=True, text=True, env=env, timeout=30)


def status_json(p: subprocess.CompletedProcess) -> dict:
    return json.loads(p.stdout.strip().splitlines()[-1])


def _amd_card(root: Path, name: str = "card0", *, od_range: str | None = None) -> Path:
    dev = root / "sys/class/drm" / name / "device"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "vendor").write_text("0x1002\n")
    text = od_range if od_range is not None else (
        "OD_SCLK:\n0:        600Mhz\n1:        2900Mhz\n"
        "OD_RANGE:\nSCLK:     600Mhz       2900Mhz\n")
    (dev / "pp_od_clk_voltage").write_text(text)
    (dev / "power_dpm_force_performance_level").write_text("")
    return dev


def _cpu_policy(root: Path, name: str, *, minf: int = 400000, maxf: int = 5200000,
                cur: int = 5187500, boost: int | None = None) -> Path:
    p = root / "sys/devices/system/cpu/cpufreq" / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "cpuinfo_min_freq").write_text(f"{minf}\n")
    (p / "cpuinfo_max_freq").write_text(f"{maxf}\n")
    (p / "scaling_max_freq").write_text(f"{cur}\n")
    if boost is not None:
        (p / "boost").write_text(f"{boost}\n")
    return p


def _conf(root: Path, **kv: str) -> None:
    d = root / "etc/llmstack"
    d.mkdir(parents=True, exist_ok=True)
    (d / "powercap.conf").write_text(
        "".join(f"{k}={v}\n" for k, v in kv.items()))


# ---------------------------------------------------------------------------
# unset = no-op
# ---------------------------------------------------------------------------

class TestUnsetIsANoOp:
    def test_no_config_at_all_touches_nothing_and_exits_clean(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        d = status_json(p)
        assert d == {"gpu_sclk_max_mhz": None, "gpu_sclk_active_mhz": None,
                    "gpu_perf_level": None, "cpu_boost": None,
                    "cpu_boost_active": None, "cpu_max_mhz": None,
                    "cpu_max_active_mhz": None, "warnings": []}

    def test_a_card_present_but_not_asked_for_is_left_alone(self, tmp_path):
        dev = _amd_card(tmp_path)
        cf = tmp_path / "sys/devices/system/cpu/cpufreq"
        _cpu_policy(tmp_path, "policy0")
        _conf(tmp_path, CPU_MAX_MHZ="2000")  # gpu + boost left unset
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        # untouched: still the empty string this fixture seeded it with
        assert (dev / "power_dpm_force_performance_level").read_text() == ""
        assert (dev / "pp_od_clk_voltage").read_text().count("s 1 ") == 0
        assert not (cf / "boost").exists()


# ---------------------------------------------------------------------------
# GPU cap: card selection, validation, the write sequence
# ---------------------------------------------------------------------------

class TestGpuCap:
    def test_picks_the_amd_card_explicitly_not_card0_by_luck(self, tmp_path):
        # card0: present, vendor 0x8086 (Intel) -- must be skipped even
        # though it sorts first.
        other = tmp_path / "sys/class/drm/card0/device"
        other.mkdir(parents=True)
        (other / "vendor").write_text("0x8086\n")
        amd = _amd_card(tmp_path, "card1")
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        assert (amd / "power_dpm_force_performance_level").read_text().strip() == "manual"

    def test_an_amd_card_with_no_od_clk_voltage_is_not_selected(self, tmp_path):
        dev = tmp_path / "sys/class/drm/card0/device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x1002\n")   # AMD, but no OD file at all
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "apply", wait="1")
        assert p.returncode != 0
        assert "no AMD GPU" in p.stderr

    def test_writes_manual_then_the_cap_then_commits_in_order(self, tmp_path):
        dev = _amd_card(tmp_path)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        assert (dev / "power_dpm_force_performance_level").read_text().strip() == "manual"
        # Two discrete commands land, in order -- see the module docstring
        # for why a plain file even can observe this (>> not >).
        tail = (dev / "pp_od_clk_voltage").read_text().splitlines()[-2:]
        assert tail == ["s 1 2500", "c"]
        d = status_json(p)
        assert d["gpu_sclk_max_mhz"] == 2500

    def test_a_target_outside_od_range_is_rejected_and_nothing_is_written(
            self, tmp_path):
        dev = _amd_card(tmp_path)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="3200")   # OD_RANGE tops out at 2900
        p = run(tmp_path, "apply")
        assert p.returncode != 0
        assert "OD_RANGE" in p.stderr or "outside" in p.stderr
        assert (dev / "power_dpm_force_performance_level").read_text() == ""
        assert "s 1 " not in (dev / "pp_od_clk_voltage").read_text()

    def test_below_the_range_is_also_rejected(self, tmp_path):
        _amd_card(tmp_path)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="100")
        p = run(tmp_path, "apply")
        assert p.returncode != 0

    def test_no_card_at_all_fails_loudly_after_the_bounded_wait_and_does_not_hang(
            self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "apply", wait="1")
        assert p.returncode != 0
        assert "no AMD GPU" in p.stderr

    def test_two_amd_cards_with_pp_od_clk_voltage_refuses_to_guess(self, tmp_path):
        # gpu-laptop-1's shape: an AMD iGPU sitting beside an AMD dGPU, both
        # vendor 0x1002, both exposing pp_od_clk_voltage. Picking whichever
        # sorts first would silently cap the wrong one and still report the
        # configured target as "active" -- this must fail loudly instead and
        # touch NEITHER card.
        card0 = _amd_card(tmp_path, "card0")
        card1 = _amd_card(tmp_path, "card1")
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "apply", wait="1")
        assert p.returncode != 0
        assert "more than one AMD card" in p.stderr
        assert (card0 / "power_dpm_force_performance_level").read_text() == ""
        assert (card1 / "power_dpm_force_performance_level").read_text() == ""
        assert "s 1 " not in (card0 / "pp_od_clk_voltage").read_text()
        assert "s 1 " not in (card1 / "pp_od_clk_voltage").read_text()

    def test_two_amd_cards_makes_status_report_no_active_reading_either(
            self, tmp_path):
        # The dangerous half of the same hazard: `status` must not pick one
        # of the two cards either, or a `fleetctl check` reading it back
        # could show the configured target as active while the real compute
        # card sits uncapped.
        _amd_card(tmp_path, "card0")
        _amd_card(tmp_path, "card1")
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        p = run(tmp_path, "status")
        assert p.returncode == 0, p.stderr
        d = status_json(p)
        assert d["gpu_sclk_active_mhz"] is None
        assert d["gpu_perf_level"] is None


# ---------------------------------------------------------------------------
# CPU boost
# ---------------------------------------------------------------------------

class TestCpuBoost:
    def test_the_global_boost_file_is_preferred(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        cf = tmp_path / "sys/devices/system/cpu/cpufreq"
        cf.mkdir(parents=True)
        (cf / "boost").write_text("1\n")
        _cpu_policy(tmp_path, "policy0", boost=1)
        _conf(tmp_path, CPU_BOOST="0")
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        assert (cf / "boost").read_text().strip() == "0"
        # the per-policy fallback file is untouched when the global one exists
        assert (cf / "policy0" / "boost").read_text().strip() == "1"

    def test_falls_back_to_every_per_policy_file_when_the_global_one_is_absent(
            self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        _cpu_policy(tmp_path, "policy0", boost=1)
        _cpu_policy(tmp_path, "policy1", boost=1)
        _conf(tmp_path, CPU_BOOST="0")
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        cf = tmp_path / "sys/devices/system/cpu/cpufreq"
        assert (cf / "policy0" / "boost").read_text().strip() == "0"
        assert (cf / "policy1" / "boost").read_text().strip() == "0"

    def test_neither_file_present_fails_loudly(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, CPU_BOOST="0")
        p = run(tmp_path, "apply")
        assert p.returncode != 0
        assert "boost" in p.stderr


# ---------------------------------------------------------------------------
# CPU max frequency
# ---------------------------------------------------------------------------

class TestCpuMax:
    def test_writes_khz_to_every_policy_within_range(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        p0 = _cpu_policy(tmp_path, "policy0")
        p1 = _cpu_policy(tmp_path, "policy1")
        _conf(tmp_path, CPU_MAX_MHZ="2000")
        p = run(tmp_path, "apply")
        assert p.returncode == 0, p.stderr
        assert (p0 / "scaling_max_freq").read_text().strip() == "2000000"
        assert (p1 / "scaling_max_freq").read_text().strip() == "2000000"

    def test_validated_per_policy_not_against_a_single_shared_range(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        small = _cpu_policy(tmp_path, "policy0", minf=400000, maxf=3000000,
                            cur=3000000)
        big = _cpu_policy(tmp_path, "policy1", minf=400000, maxf=5200000,
                          cur=5187500)
        _conf(tmp_path, CPU_MAX_MHZ="4000")     # fits policy1, not policy0
        p = run(tmp_path, "apply")
        assert p.returncode != 0            # the box could not honour it fully
        assert "policy0" in p.stderr
        assert (small / "scaling_max_freq").read_text().strip() == "3000000"
        assert (big / "scaling_max_freq").read_text().strip() == "4000000"

    def test_no_cpufreq_policies_at_all_fails_loudly(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, CPU_MAX_MHZ="2000")
        p = run(tmp_path, "apply")
        assert p.returncode != 0
        assert "cpufreq policy" in p.stderr


# ---------------------------------------------------------------------------
# status: read-only, needs no root, parses a driver's own dump
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_never_writes_anything(self, tmp_path):
        dev = _amd_card(tmp_path)
        cf = tmp_path / "sys/devices/system/cpu/cpufreq"
        cf.mkdir(parents=True)
        (cf / "boost").write_text("1\n")
        _cpu_policy(tmp_path, "policy0")
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500", CPU_BOOST="0", CPU_MAX_MHZ="2000")
        before_od = (dev / "pp_od_clk_voltage").read_text()
        before_boost = (cf / "boost").read_text()
        p = run(tmp_path, "status")
        assert p.returncode == 0, p.stderr
        assert (dev / "pp_od_clk_voltage").read_text() == before_od
        assert (cf / "boost").read_text() == before_boost

    def test_reads_a_post_commit_dump_the_way_a_real_card_would_render_one(
            self, tmp_path):
        # Hand-typed as if the driver had already applied a 2500 MHz cap --
        # this is what `status` on a real, already-capped box actually reads.
        _amd_card(tmp_path, od_range=(
            "OD_SCLK:\n0:        600Mhz\n1:        2500Mhz\n"
            "OD_RANGE:\nSCLK:     600Mhz       2900Mhz\n"))
        (tmp_path / "sys/class/drm/card0/device/power_dpm_force_performance_level"
         ).write_text("manual\n")
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        d = status_json(run(tmp_path, "status"))
        assert d["gpu_sclk_active_mhz"] == 2500
        assert d["gpu_perf_level"] == "manual"

    def test_cpu_max_active_is_the_worst_case_not_the_best(self, tmp_path):
        """One under-capped core must show up as drift, not be hidden by an
        average or a minimum across cores."""
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        _cpu_policy(tmp_path, "policy0", cur=2000000)
        _cpu_policy(tmp_path, "policy1", cur=3000000)   # still uncapped
        _conf(tmp_path, CPU_MAX_MHZ="2000")
        d = status_json(run(tmp_path, "status"))
        assert d["cpu_max_active_mhz"] == 3000

    def test_no_card_and_nothing_requested_is_quiet(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        d = status_json(run(tmp_path, "status"))
        assert d["warnings"] == []

    def test_no_card_but_a_gpu_cap_is_configured_warns(self, tmp_path):
        (tmp_path / "sys/class/drm").mkdir(parents=True)
        (tmp_path / "sys/devices/system/cpu/cpufreq").mkdir(parents=True)
        _conf(tmp_path, GPU_SCLK_MAX_MHZ="2500")
        d = status_json(run(tmp_path, "status"))
        assert any("AMD GPU" in w for w in d["warnings"])
