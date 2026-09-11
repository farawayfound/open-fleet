"""GPU busy-ness on a box whose kernel will not say: apu-box-1.

The GMKtec EVO-X2 (Radeon 8060S, Strix Halo gfx1151, PCI 1002:1586) never
populates the SMU-derived activity counter. Measured on the box on 2026-09-10,
mid-inference: /sys/class/drm/card0/device/gpu_busy_percent read 0 while
llama-server's own DRM fdinfo carried drm-driver: amdgpu, the card's drm-pdev,
and a drm-engine-compute of 105422134447 ns that kept climbing. The fleet page
averages busy_percent over a host's cards, so the box serving the most work
showed 0% GPU while it was serving.

These tests build a fake /proc rather than reading the real one, because the
workstation this is written on is Windows and there is no amdgpu anywhere near
it. The one thing a fake cannot fake is a symlink on a machine that will not
create them -- so that is a skip with a reason, never a failure.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import hw

PDEV = "0000:c5:00.0"
OTHER_PDEV = "0000:03:00.0"


def fdinfo(pdev: str, client_id: int, *, gfx: int = 0, compute: int = 0,
           driver: str = "amdgpu") -> str:
    """One /proc/<pid>/fdinfo/<fd> as the DRM core really writes it: tab
    separators, the engine times carrying their unit, and the file's own
    generic fd lines above the drm- block."""
    return (
        "pos:\t0\n"
        "flags:\t02100002\n"
        "mnt_id:\t26\n"
        f"drm-driver:\t{driver}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-client-id:\t{client_id}\n"
        f"drm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-compute:\t{compute} ns\n"
        "drm-memory-vram:\t1048576 KiB\n"
    )


class FakeProc:
    """A /proc tree with DRM clients in it.

    The fds are real symlinks because that is the thing under test: the walk
    pre-filters on the link TARGET so it never opens the thousands of fdinfo
    files a busy box has. A fake /dev/dri under the temp directory is enough --
    the target still contains "/dev/dri/", which is what the filter matches.
    """

    def __init__(self, root: Path):
        self.root = root
        self.proc = root / "proc"
        self.proc.mkdir(parents=True, exist_ok=True)
        self.dri = root / "dev" / "dri"
        self.dri.mkdir(parents=True, exist_ok=True)
        # Directories, not files: the real targets are device nodes, and a
        # directory is the one shape BOTH os.symlink and a Windows junction
        # will point at. Only the path string matters to the walk.
        (self.dri / "renderD128").mkdir(exist_ok=True)
        (self.dri / "card0").mkdir(exist_ok=True)
        (root / "elsewhere" / "not-a-gpu").mkdir(parents=True, exist_ok=True)
        # /proc is not all pids: `self`, `cpuinfo`, `net`. The walk has to step
        # over them rather than trip on them.
        (self.proc / "self").mkdir(exist_ok=True)
        (self.proc / "cpuinfo").write_text("processor\t: 0\n")

    def add(self, pid: int, fd: int, text: str | None, *,
            node: str = "renderD128", dri: bool = True) -> None:
        d = self.proc / str(pid)
        (d / "fd").mkdir(parents=True, exist_ok=True)
        (d / "fdinfo").mkdir(parents=True, exist_ok=True)
        target = (self.dri / node) if dri else (self.root / "elsewhere" / "not-a-gpu")
        # Re-adding an fd is how a test advances a client's counters between
        # two samples; the link is already there and making it twice raises.
        if not os.path.lexists(d / "fd" / str(fd)):
            symlink(target, d / "fd" / str(fd))
        if text is None:
            # A directory where the file should be: read_text() raises, which is
            # what an unreadable fdinfo does on a real box and must be skipped
            # in silence rather than taking the whole reading down.
            (d / "fdinfo" / str(fd)).mkdir(exist_ok=True)
        else:
            (d / "fdinfo" / str(fd)).write_text(text)


def symlink(target: Path, link: Path) -> None:
    """A link the walk can readlink(), by whichever means this OS allows.

    os.symlink on Linux, where the fleet runs. On Windows it raises WinError
    1314 -- "a required privilege is not held" -- unless the shell is elevated
    or Developer Mode is on, which is not something a test may demand of the
    workstation it is written on. A directory junction needs no privilege,
    _winapi.CreateJunction makes one, and Path.readlink() reads it back (as
    \\\\?\\C:\\... , which is why the walk normalises separators before matching).
    If neither works this is a gap in the run, not a failing test.
    """
    try:
        os.symlink(target, link)
        return
    except (OSError, NotImplementedError, AttributeError) as exc:
        first = exc
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except Exception:  # noqa: BLE001 -- report the original failure, not this one
        pytest.skip("this platform will not create a link here: " + str(first))


@pytest.fixture(autouse=True)
def _clear_sample_cache():
    """The rate is a delta against the previous call, so a sample left behind
    by one test is a wrong percentage in the next."""
    hw._fdinfo_cache.clear()
    yield
    hw._fdinfo_cache.clear()


@pytest.fixture
def busy_box(tmp_path):
    """Two GPUs' worth of clients, one of them a dup, plus the noise a real
    /proc carries: another card's client, a foreign driver, an unreadable
    fdinfo and a process holding an fd that is not a DRM handle at all."""
    p = FakeProc(tmp_path)
    # llama-server on the card under test: 100 s of compute, a sliver of gfx.
    p.add(101, 3, fdinfo(PDEV, 12, gfx=1_000, compute=100_000_000_000))
    # The same client through a dup'd fd. Same counters, and summing them would
    # report one process as two.
    p.add(101, 4, fdinfo(PDEV, 12, gfx=1_000, compute=100_000_000_000))
    # A second client on the same card: the compositor, 5 s of gfx.
    p.add(101, 5, fdinfo(PDEV, 13, gfx=5_000_000_000))
    # A client on a DIFFERENT card. Counts only when no pdev is given.
    p.add(202, 3, fdinfo(OTHER_PDEV, 99, compute=7_000_000_000))
    # Not amdgpu.
    p.add(303, 3, fdinfo(PDEV, 77, compute=9_000_000_000, driver="i915"))
    # Unreadable fdinfo -- another user's process.
    p.add(404, 3, None)
    # A DRM-looking pid whose fd points at an ordinary file: never opened.
    p.add(505, 3, fdinfo(PDEV, 55, compute=3_000_000_000), dri=False)
    return p


class TestEngineTime:
    def test_sums_gfx_and_compute_for_the_matching_card(self, busy_box):
        total, clients = hw.amdgpu_fdinfo_engine_ns(PDEV, busy_box.proc)
        assert total == 1_000 + 100_000_000_000 + 5_000_000_000
        assert clients == 2, "the dup'd fd is the same client, not a third one"

    def test_without_a_pdev_every_amdgpu_client_counts(self, busy_box):
        """amdgpu_pdev() returns None when the sysfs card is not a symlink, and
        the honest fallback then is every amdgpu client on the box rather than
        none of them."""
        total, clients = hw.amdgpu_fdinfo_engine_ns(None, busy_box.proc)
        assert total == 1_000 + 100_000_000_000 + 5_000_000_000 + 7_000_000_000
        assert clients == 3

    def test_another_cards_client_is_left_out(self, busy_box):
        total, _ = hw.amdgpu_fdinfo_engine_ns(OTHER_PDEV, busy_box.proc)
        assert total == 7_000_000_000

    def test_a_foreign_driver_is_left_out(self, busy_box):
        """drm-engine-* is a DRM-core field, not an amdgpu one: an Intel iGPU
        on the same box reports it too, under its own driver name. pid 303 is
        holding 9 s of it on this very pdev."""
        total, clients = hw.amdgpu_fdinfo_engine_ns(PDEV, busy_box.proc)
        assert total == 105_000_001_000
        assert clients == 2

    def test_unreadable_and_non_drm_fds_are_skipped_in_silence(self, busy_box):
        """pid 404's fdinfo cannot be read and pid 505's fd is not a DRM
        handle. Neither may raise and neither may count -- /proc is mostly
        other people's processes, and that is the normal case, not a fault."""
        total, clients = hw.amdgpu_fdinfo_engine_ns(None, busy_box.proc)
        assert total == 112_000_001_000, "pid 505's 3 s never got opened"
        assert clients == 3

    def test_no_proc_at_all_is_zero_not_an_exception(self, tmp_path):
        assert hw.amdgpu_fdinfo_engine_ns(PDEV, tmp_path / "nope") == (0, 0)


class TestBusyPercent:
    def test_the_first_call_has_nothing_to_diff_against(self, busy_box):
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None

    def test_the_second_call_is_the_rate_between_the_two(self, busy_box):
        """Half a second of engine time in one wall-clock second is 50% busy --
        the same arithmetic amdgpu_top and nvtop do, and the reason this never
        sleeps: two dashboard polls are the two samples."""
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None
        busy_box.add(101, 3, fdinfo(PDEV, 12, gfx=1_000,
                                    compute=100_500_000_000))
        busy_box.add(101, 4, fdinfo(PDEV, 12, gfx=1_000,
                                    compute=100_500_000_000))
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=101.0) == 50.0

    def test_it_is_clamped_at_one_hundred(self, busy_box):
        """Several engines can each be busy for the whole interval, so the raw
        ratio goes over 100 legitimately. A card is still not 340% busy."""
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None
        busy_box.add(101, 5, fdinfo(PDEV, 13, gfx=9_000_000_000))
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=101.0) == 100.0

    def test_a_client_that_exited_does_not_report_negative(self, busy_box, tmp_path):
        """Engine time leaves with the process that earned it, so a
        llama-server restart between two polls is a falling total. "-40% busy"
        on the fleet page would be worse than "idle"."""
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None
        empty = FakeProc(tmp_path / "after")
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=empty.proc,
                                             now=101.0) == 0.0

    def test_two_calls_too_close_together_say_nothing(self, busy_box):
        """Under 0.2 s the ratio is mostly sampling jitter, and a number that
        looks measured and is not is worse than a blank."""
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.1) is None

    def test_cards_are_sampled_separately(self, busy_box):
        """One card's baseline must never be diffed against another's total."""
        assert hw.amdgpu_fdinfo_busy_percent(PDEV, proc_root=busy_box.proc,
                                             now=100.0) is None
        assert hw.amdgpu_fdinfo_busy_percent(OTHER_PDEV, proc_root=busy_box.proc,
                                             now=101.0) is None
        assert hw.amdgpu_fdinfo_busy_percent(OTHER_PDEV, proc_root=busy_box.proc,
                                             now=102.0) == 0.0


class TestPdev:
    def test_the_slot_comes_off_the_device_symlink(self, tmp_path):
        """/sys/class/drm/card0/device points into the PCI tree; the last
        component is what fdinfo's drm-pdev line carries.

        The slot is spelled without colons here because NTFS will not take one
        in a filename (WinError 123), and the reading is "the last component of
        the link target" either way -- the colons are not what is under test.
        """
        slot = PDEV if os.name != "nt" else "0000-c5-00.0"
        pci = tmp_path / "sys" / "devices" / "pci0000" / slot
        pci.mkdir(parents=True)
        card = tmp_path / "sys" / "class" / "drm" / "card0"
        card.mkdir(parents=True)
        symlink(pci, card / "device")
        assert hw.amdgpu_pdev(card / "device") == slot

    def test_a_plain_directory_is_no_slot_rather_than_an_error(self, tmp_path):
        dev = tmp_path / "device"
        dev.mkdir()
        assert hw.amdgpu_pdev(dev) is None


class FakeFdinfo:
    def __init__(self, value: float | None):
        self.value = value
        self.calls = 0

    def __call__(self, pdev, **kwargs):
        self.calls += 1
        return self.value


@pytest.fixture
def fake_card(tmp_path, monkeypatch):
    """One card's worth of sysfs, with _read_int replaced so no real file has
    to hold a plausible number. Returns a setter for whatever
    gpu_busy_percent should read on this run."""
    drm = tmp_path / "drm"
    (drm / "card0" / "device").mkdir(parents=True)
    (drm / "card0" / "device" / "gpu_busy_percent").write_text("0")
    values: dict[str, int | None] = {"mem_info_vram_total": 103_079_215_104,
                                     "mem_info_vram_used": 51_539_607_552,
                                     "mem_info_gtt_used": 1_048_576,
                                     "mem_info_gtt_total": 8_388_608}

    def _read_int(path: Path):
        return values.get(path.name)

    monkeypatch.setattr(hw, "_read_int", _read_int)

    def set_busy(v):
        values["gpu_busy_percent"] = v

    set_busy(0)
    return drm, set_busy


class TestWhichSourceWins:
    def test_a_positive_sysfs_reading_wins_and_costs_nothing(self, fake_card, monkeypatch):
        """gpu-laptop-1's RX 6700S reports honestly. Walking /proc for a figure
        sysfs already gave would be work for nothing, so the fallback must not
        even be consulted."""
        drm, set_busy = fake_card
        fake = FakeFdinfo(42.5)
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", fake)
        set_busy(87)
        (card,) = hw.amdgpu_stats(drm)
        assert card["busy_percent"] == 87
        assert card["busy_source"] == "sysfs"
        assert fake.calls == 0

    def test_a_zero_from_sysfs_falls_through_to_fdinfo(self, fake_card, monkeypatch):
        """This is apu-box-1: the counter reads 0 while the box is serving."""
        drm, set_busy = fake_card
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", FakeFdinfo(42.5))
        set_busy(0)
        (card,) = hw.amdgpu_stats(drm)
        assert card["busy_percent"] == 42.5
        assert card["busy_source"] == "fdinfo"

    def test_an_unreadable_counter_falls_through_too(self, fake_card, monkeypatch):
        drm, set_busy = fake_card
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", FakeFdinfo(42.5))
        set_busy(None)
        (card,) = hw.amdgpu_stats(drm)
        assert card["busy_percent"] == 42.5
        assert card["busy_source"] == "fdinfo"

    def test_a_genuine_idle_zero_stays_zero(self, fake_card, monkeypatch):
        """First poll after a restart: fdinfo has no previous sample to diff
        against, so sysfs's zero is the only answer there is -- and it is
        labelled as sysfs's, not as nothing."""
        drm, set_busy = fake_card
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", FakeFdinfo(None))
        set_busy(0)
        (card,) = hw.amdgpu_stats(drm)
        assert card["busy_percent"] == 0
        assert card["busy_source"] == "sysfs"

    def test_neither_source_answering_is_none_not_zero(self, fake_card, monkeypatch):
        """"Nobody would say" and "the card is idle" are different facts, and
        busy_source is the only place the difference survives."""
        drm, set_busy = fake_card
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", FakeFdinfo(None))
        set_busy(None)
        (card,) = hw.amdgpu_stats(drm)
        assert card["busy_percent"] is None
        assert card["busy_source"] == "none"

    def test_every_other_field_is_untouched(self, fake_card, monkeypatch):
        """busy_source is additive. The dashboard reads busy_percent and the
        three backends have to agree on the rest of the shape."""
        drm, set_busy = fake_card
        monkeypatch.setattr(hw, "amdgpu_fdinfo_busy_percent", FakeFdinfo(42.5))
        (card,) = hw.amdgpu_stats(drm)
        assert set(card) == {"card", "busy_percent", "busy_source", "vram_total",
                             "vram_used", "gtt_used", "gtt_total", "temp_c",
                             "power_w", "sclk_mhz"}
        assert card["card"] == "card0"
        assert card["vram_total"] == 103_079_215_104
        assert card["vram_used"] == 51_539_607_552
        assert card["gtt_used"] == 1_048_576
        assert card["gtt_total"] == 8_388_608
        assert card["temp_c"] is None and card["power_w"] is None
        assert card["sclk_mhz"] is None

    def test_the_default_root_is_still_sysfs(self):
        """The seam is for the tests; the gateway and fleetctl both call this
        with no arguments and must keep reading the real thing."""
        assert hw.amdgpu_stats.__defaults__ == (Path("/sys/class/drm"),)
