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


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(gw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    rec = Recorder()
    monkeypatch.setattr(gw.subprocess, "run", rec)
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
            ("taskkill", "/F", "/T", "/IM", "llama-swap.exe"),
            ("taskkill", "/F", "/T", "/IM", "llama-server.exe"),
            ("schtasks", "/Run", "/TN", "llama-swap"),
        ]

    def test_the_survivors_matter(self, on_windows):
        """/End stops the .cmd wrapper, not the llama-swap it spawned. Leave
        that one alive and the restart fails on a bound 8081 while the card
        stays full — the exact failure this kill list exists to prevent."""
        gw.service_control("restart", "llama-swap")
        killed = [c[-1] for c in on_windows.calls if c[0] == "taskkill"]
        assert killed == ["llama-swap.exe", "llama-server.exe"]

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


class TestLinux:
    def test_still_speaks_systemd(self, on_linux):
        gw.service_control("restart", "llama-swap")
        assert on_linux.calls == [
            ("sudo", "-n", "/usr/bin/systemctl", "restart", "llama-swap.service")
        ]
