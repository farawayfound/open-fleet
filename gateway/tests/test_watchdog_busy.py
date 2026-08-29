"""A gateway that is slow is not a gateway that is deaf.

apu-tablet-1, 2026-08-29: a 73 GB Flash-Next upload to the GPU had the box at
its commit ceiling, /health took longer than the watchdog's ten seconds three
times running, and the watchdog exited a process that was merely busy --
twice, since Task Scheduler brought it straight back into the same load. The
zombie the watchdog exists for (apu-tablet-2, WinError 64) REFUSES: its accept
loop is gone. A listener that accepts and then answers late is load. The
probe now tells the two apart, and only a refusal counts as a miss.
"""
from __future__ import annotations

import pytest

import app as gw


@pytest.fixture
def armed(monkeypatch):
    gw._watchdog_state.update(armed=True, misses=0)
    exits: list[int] = []
    monkeypatch.setattr(gw.os, "_exit", lambda code=1: exits.append(code))
    monkeypatch.setattr(gw, "_own_listener_pid", lambda: True)
    monkeypatch.setattr(gw, "draining", lambda: False)
    yield exits
    gw._watchdog_state.update(armed=False, misses=0)


class TestSlowIsNotDead:
    def test_slow_answers_never_add_up_to_an_exit(self, armed, monkeypatch):
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "slow")
        for _ in range(gw.WATCHDOG_MAX_MISSES * 3):
            gw._self_watchdog_tick()
        assert armed == []
        assert gw._watchdog_state["misses"] == 0

    def test_slow_does_not_forgive_misses_already_earned(self, armed, monkeypatch):
        """A gateway on its way down can be slow before it is gone; the count
        it has stands until an answer or a refusal settles it."""
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        gw._self_watchdog_tick()
        gw._self_watchdog_tick()
        assert gw._watchdog_state["misses"] == 2
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "slow")
        gw._self_watchdog_tick()
        assert gw._watchdog_state["misses"] == 2
        assert armed == []
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        gw._self_watchdog_tick()
        assert armed == [1]

    def test_a_refusal_still_kills(self, armed, monkeypatch):
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        for _ in range(gw.WATCHDOG_MAX_MISSES):
            gw._self_watchdog_tick()
        assert armed == [1]

    def test_an_answer_clears_the_count(self, armed, monkeypatch):
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        gw._self_watchdog_tick()
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "ok")
        gw._self_watchdog_tick()
        assert gw._watchdog_state["misses"] == 0


class TestTheProbeClassifies:
    def test_a_read_timeout_is_slow(self, monkeypatch):
        import httpx

        def get(*a, **k):
            raise httpx.ReadTimeout("late")
        monkeypatch.setattr(gw.httpx, "get", get)
        assert gw._loopback_health_probe() == "slow"

    def test_a_refusal_is_dead(self, monkeypatch):
        import httpx

        def get(*a, **k):
            raise httpx.ConnectError("refused")
        monkeypatch.setattr(gw.httpx, "get", get)
        assert gw._loopback_health_probe() == "dead"

    def test_a_200_is_ok(self, monkeypatch):
        class R:
            status_code = 200
        monkeypatch.setattr(gw.httpx, "get", lambda *a, **k: R())
        assert gw._loopback_health_probe() == "ok"
