"""Restarting llama-swap in the local dialect.

Linux boxes answer to systemd. gpu-desktop-1 — the fleet's first Windows box running
llama.cpp rather than Ollama — answers to Scheduled Tasks, and gets the
config reload it needs when a model is saved.
"""
from __future__ import annotations

import pytest

import app as gw


class Recorder:
    """Stands in for subprocess.run and remembers every argv it was handed."""

    def __init__(self, rc: int = 0):
        self.calls: list[tuple[str, ...]] = []
        self.rc = rc

    def __call__(self, cmd, **kwargs):
        self.calls.append(tuple(cmd))

        class P:
            returncode = self.rc
            stdout = ""
            stderr = ""

        return P()


class Proc:
    """Stands in for one psutil.Process as process_iter() hands it back."""

    def __init__(self, pid, name, exe, ppid=4, cmdline=()):
        self.info = {"pid": pid, "ppid": ppid, "name": name, "exe": exe,
                     "cmdline": list(cmdline)}


FLEET = r"C:\llmstack"
# What is running on a laptop whose owner also runs a model of their own.
OURS_SWAP = Proc(900, "llama-swap.exe", FLEET + r"\bin\llama-swap.exe")
OURS_SERVER = Proc(901, "llama-server.exe",
                   FLEET + r"\bin\llama\llama-server.exe", ppid=900)
THEIRS = Proc(4242, "llama-server.exe",
              r"C:\Users\owner\llama.cpp\build\bin\llama-server.exe")
BYSTANDER = Proc(77, "python.exe", r"C:\Program Files\Python312\python.exe")


def processes(monkeypatch, *procs):
    monkeypatch.setattr(gw, "HOME", gw.Path(FLEET))
    monkeypatch.setattr(gw, "LLAMA_SERVER", FLEET + r"\bin\llama\llama-server.exe")
    monkeypatch.setattr(gw, "LLAMA_BENCH", FLEET + r"\bin\llama\llama-bench.exe")
    monkeypatch.setattr(gw, "SWAP_CONFIG", gw.Path(FLEET + r"\state\llama-swap.yaml"))
    monkeypatch.setattr(gw.psutil, "process_iter", lambda attrs=None: list(procs))


def killed(rec) -> list[str]:
    return [c[-1] for c in rec.calls if c[0] == "taskkill"]


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(gw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    rec = Recorder()
    monkeypatch.setattr(gw.subprocess, "run", rec)
    processes(monkeypatch, OURS_SWAP, OURS_SERVER)
    return rec


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
    rec = Recorder()
    monkeypatch.setattr(gw.subprocess, "run", rec)
    return rec


class TestWindows:
    def test_restart_ends_kills_the_survivors_then_runs(self, on_windows):
        rc, _out = gw.service_control("restart", "llama-swap")
        assert rc == 0
        assert on_windows.calls == [
            ("schtasks", "/End", "/TN", "llama-swap"),
            ("taskkill", "/F", "/T", "/PID", "900"),
            ("taskkill", "/F", "/T", "/PID", "901"),
            ("schtasks", "/Run", "/TN", "llama-swap"),
        ]

    def test_the_survivors_matter(self, on_windows):
        """/End stops the .cmd wrapper, not the llama-swap it spawned. Leave
        that one alive and the restart fails on a bound 8081 while the card
        stays full — the exact failure this kill list exists to prevent."""
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["900", "901"]

    def test_stop_does_not_start_anything(self, on_windows):
        gw.service_control("stop", "llama-swap")
        assert not any(c[:2] == ("schtasks", "/Run") for c in on_windows.calls)

    def test_start_does_not_kill_anything(self, on_windows):
        gw.service_control("start", "llama-swap")
        assert on_windows.calls == [("schtasks", "/Run", "/TN", "llama-swap")]

    def test_a_unit_with_no_known_children_is_only_a_task(self, on_windows):
        gw.service_control("restart", "cloudflared")
        assert not any(c[0] == "taskkill" for c in on_windows.calls)

    def test_never_kills_by_a_name_something_else_could_own(self):
        """python.exe is how the gpu-laptop-2 box once took its own unrelated
        interpreters down with it. Nothing in this table may be a name a
        bystander process could plausibly have."""
        for images in gw._WIN_TASK_CHILDREN.values():
            for image in images:
                assert image.startswith("llama-")


class TestALaptopItsOwnerAlsoUses:
    """gpu-laptop-5, 2026-09-27. The kill was `taskkill /IM llama-server.exe`:
    every llama-server on the machine, the owner's own included, ended by a
    registry save on a box the hub had been told to leave alone."""

    def test_nothing_is_ever_killed_by_name(self, on_windows, monkeypatch):
        processes(monkeypatch, OURS_SWAP, OURS_SERVER, THEIRS)
        gw.service_control("restart", "llama-swap")
        assert not any("/IM" in c for c in on_windows.calls)

    def test_the_owners_llama_server_is_left_running(self, on_windows, monkeypatch):
        processes(monkeypatch, OURS_SWAP, OURS_SERVER, THEIRS, BYSTANDER)
        _rc, out = gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["900", "901"]
        # ...and it says so, by pid and by path, in what the dashboard shows.
        assert "left alone: llama-server.exe pid 4242" in out
        assert "owner" in out

    def test_our_llama_swaps_child_is_ours_wherever_its_binary_is(
            self, on_windows, monkeypatch):
        """A llama-server our llama-swap started from a path nobody planned
        still holds our port and our VRAM."""
        stray = Proc(902, "llama-server.exe", r"D:\tools\llama-server.exe", ppid=900)
        processes(monkeypatch, OURS_SWAP, stray)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["900", "902"]

    def test_a_strangers_child_is_not_ours(self, on_windows, monkeypatch):
        """The parent rule must not reach the owner's own llama-swap."""
        their_swap = Proc(500, "llama-swap.exe", r"C:\Users\owner\bin\llama-swap.exe")
        their_child = Proc(501, "llama-server.exe",
                           r"C:\Users\owner\bin\llama-server.exe", ppid=500)
        processes(monkeypatch, OURS_SWAP, their_swap, their_child)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["900"]

    def test_the_same_name_under_a_lookalike_directory_is_not_ours(
            self, on_windows, monkeypatch):
        r"""C:\llmstack-old is not under C:\llmstack."""
        near = Proc(55, "llama-server.exe", r"C:\llmstack-old\bin\llama-server.exe")
        processes(monkeypatch, near)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == []

    def test_case_and_slashes_do_not_hide_our_own_process(
            self, on_windows, monkeypatch):
        odd = Proc(60, "LLAMA-SERVER.EXE", "c:/LLMSTACK/bin/llama/llama-server.exe")
        processes(monkeypatch, odd)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["60"]

    def test_a_process_whose_path_cannot_be_read_is_spared(
            self, on_windows, monkeypatch):
        """Only a gateway somebody started by hand cannot read them. Ending
        a stranger's model because we could not see whose it was is the
        worse mistake."""
        processes(monkeypatch, Proc(70, "llama-server.exe", None))
        _rc, out = gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == []
        assert "path unreadable" in out

    def test_an_engine_planned_outside_the_prefix_is_still_ours(
            self, on_windows, monkeypatch):
        processes(monkeypatch, Proc(80, "llama-server.exe", r"D:\engine\llama-server.exe"))
        monkeypatch.setattr(gw, "LLAMA_SERVER", r"D:\engine\llama-server.exe")
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["80"]

    def test_a_llama_swap_given_our_config_is_ours_wherever_it_was_installed(
            self, on_windows, monkeypatch):
        """The gateway is never told where llama-swap's binary is, only what
        it was given to read. One installed outside the prefix still holds
        our port."""
        swap = Proc(300, "llama-swap.exe", r"D:\tools\llama-swap.exe", cmdline=[
            r"D:\tools\llama-swap.exe", "--config",
            "c:/LLMSTACK/state/llama-swap.yaml", "--listen", "127.0.0.1:8081"])
        child = Proc(301, "llama-server.exe", r"D:\tools\llama-server.exe", ppid=300)
        processes(monkeypatch, swap, child, THEIRS)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == ["300", "301"]

    def test_a_strangers_llama_swap_with_its_own_config_is_not_ours(
            self, on_windows, monkeypatch):
        swap = Proc(310, "llama-swap.exe", r"C:\Users\owner\llama-swap.exe", cmdline=[
            "llama-swap.exe", "--config", r"C:\Users\owner\llama-swap.yaml"])
        processes(monkeypatch, swap)
        gw.service_control("restart", "llama-swap")
        assert killed(on_windows) == []

    def test_the_restart_still_happens_when_the_list_cannot_be_read(
            self, on_windows, monkeypatch):
        def boom(attrs=None):
            raise gw.psutil.AccessDenied(pid=0)
        monkeypatch.setattr(gw.psutil, "process_iter", boom)
        rc, out = gw.service_control("restart", "llama-swap")
        assert rc == 0
        assert on_windows.calls[-1] == ("schtasks", "/Run", "/TN", "llama-swap")
        assert "could not list processes" in out


class TestLinux:
    def test_still_speaks_systemd(self, on_linux):
        gw.service_control("restart", "llama-swap")
        assert on_linux.calls == [
            ("sudo", "-n", "/usr/bin/systemctl", "restart", "llama-swap.service")
        ]
