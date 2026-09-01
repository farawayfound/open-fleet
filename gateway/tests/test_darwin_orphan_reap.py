"""On mac-laptop-1, a broken cron keepalive kept spawning doomed duplicate
llama-swaps -- each began its start-up preload before losing the race for
:8081, and one preload child outlived its own parent. Nothing was left to
route to that llama-server or ever unload it, so it sat there holding 21 GB
of unified memory for three days while every verify on the box answered
HTTP 500, because the llama-swap actually listening on :8081 could not fit
the model it was told to serve on top of the orphan's share.

`_darwin_service`'s start/restart path now treats this as routine: once
llama-swap itself is confirmed gone (`pgrep -f bin/llama-swap -config`
comes back empty), any surviving `llama-server` cannot belong to a live
llama-swap -- a live one stops its own children on the way out -- so it is
reaped with `pkill -f bin/llama-server --model` before the runner is
spawned. This file is about that reap step alone: that it fires only when
llama-swap is actually gone, that it happens before the relaunch (not after,
where it would just race a freshly-spawned server), and that a reap which
itself blows up (no `pkill` on the box, a permissions error) is swallowed
rather than sinking the whole restart."""
from __future__ import annotations

import pytest

import app as gw


class Recorder:
    """subprocess.run, recording argv into a shared `order` log (so calls
    can be interleaved with FakePopen's below and checked for ordering).
    `alive` scripts pgrep's exit code in call order -- 0 means "a process
    matched", 1 means "nothing did". `raises` maps one exact argv tuple to
    an exception `subprocess.run` should raise instead of returning, so the
    reap step's best-effort handling can be exercised."""

    def __init__(self, alive=(1,), raises=None, order=None):
        self.calls: list[tuple[str, ...]] = []
        self.alive = list(alive)
        self.raises = raises or {}
        self.order = order if order is not None else []

    def __call__(self, cmd, **kwargs):
        cmd = tuple(cmd)
        self.calls.append(cmd)
        self.order.append(cmd)
        if cmd in self.raises:
            raise self.raises[cmd]
        rc = 0
        if cmd[0] == "pgrep":
            rc = self.alive.pop(0) if self.alive else 1

        class P:
            returncode = rc
            stdout = ""
            stderr = ""

        return P()


class FakePopen:
    """subprocess.Popen, recording the runner it was asked to launch into
    the same shared `order` log as Recorder, so a test can prove a reap
    happened strictly before the relaunch rather than merely happening."""

    def __init__(self, order=None):
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []
        self.order = order if order is not None else []

    def __call__(self, cmd, **kwargs):
        cmd = tuple(cmd)
        self.calls.append(cmd)
        self.kwargs.append(kwargs)
        self.order.append(("Popen", cmd))

        class P:
            pid = 4242

        return P()


@pytest.fixture
def on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(gw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    runner = tmp_path / "run-llama-swap.sh"
    runner.write_text("#!/bin/sh\n")
    monkeypatch.setattr(gw, "_MAC_RUNNER", str(runner))
    monkeypatch.setattr(gw, "_MAC_SWAP_LOG", tmp_path / "llama-swap.log")
    return runner


REAP = ("pkill", "-f", gw._MAC_SERVER_PATTERN)
SWAP_PGREP = ("pgrep", "-f", gw._MAC_SWAP_PATTERN)
SWAP_PKILL = ("pkill", "-f", gw._MAC_SWAP_PATTERN)


class TestOrphanReapedBeforeRelaunch:
    def test_restart_reaps_a_dead_swaps_orphan_before_spawning_the_runner(
            self, monkeypatch, on_darwin):
        """Stop confirms llama-swap gone (pgrep rc 1); the start step checks
        again and gets the same empty answer, so whatever llama-server is
        still around belongs to nobody -- it gets pkilled, and the output
        says so, before run-llama-swap.sh is ever spawned."""
        order: list = []
        rec = Recorder(alive=[1, 1, 0], order=order)
        pop = FakePopen(order=order)
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", pop)
        rc, out = gw.service_control("restart", "llama-swap")
        assert rc == 0, out
        assert REAP in rec.calls
        assert "reaped an orphaned llama-server" in out
        assert order.index(REAP) < order.index(("Popen", (str(on_darwin),)))


class TestALiveSwapsChildrenAreNotOrphans:
    def test_start_leaves_llama_server_alone_when_swap_answers_pgrep(
            self, monkeypatch, on_darwin):
        """rc 0 at the start-step pgrep means a llama-swap is already up and
        holding its own children -- reaping llama-server here would kill a
        model out from under a live request for no reason."""
        rec = Recorder(alive=[0, 0])  # start-check: alive; final check: alive
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", FakePopen())
        rc, out = gw.service_control("start", "llama-swap")
        assert rc == 0, out
        assert REAP not in rec.calls


class TestPlainStartAlsoReaps:
    def test_start_with_no_swap_alive_reaps_same_as_restart_does(
            self, monkeypatch, on_darwin):
        """Gateway boot calls this with a plain 'start', not 'restart' -- an
        orphan left over from before the gateway itself came up must not
        survive just because there was no stop step this time."""
        rec = Recorder(alive=[1, 0])  # start-check: gone; final check: alive
        pop = FakePopen()
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", pop)
        rc, out = gw.service_control("start", "llama-swap")
        assert rc == 0, out
        assert REAP in rec.calls
        assert "reaped an orphaned llama-server" in out
        # 'start' never runs the stop-phase pkill against llama-swap itself.
        assert SWAP_PKILL not in rec.calls


class TestReapIsBestEffort:
    def test_a_reap_that_raises_does_not_fail_the_restart(
            self, monkeypatch, on_darwin):
        """subprocess.run can throw outright -- no pkill on the box, a
        permission error -- and the comment next to this code calls the reap
        best-effort for exactly that reason: it must not take the relaunch
        down with it."""
        order: list = []
        boom = OSError("no such file or directory: pkill")
        rec = Recorder(alive=[1, 1, 0], raises={REAP: boom}, order=order)
        pop = FakePopen(order=order)
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", pop)
        rc, out = gw.service_control("restart", "llama-swap")
        assert rc == 0, out
        assert REAP in rec.calls           # the reap really was attempted
        assert "reaped" not in out         # ...but never got to report success
        assert pop.calls == [(str(on_darwin),)]  # and the relaunch still ran
