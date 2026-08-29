"""A gateway that stops answering must die, not linger.

The regression these pin down is apu-tablet-2 on 2026-08-28. One failed accept
in uvicorn's proactor loop (OSError [WinError 64], arriving as an external-SSD
replug tore the hub's connections down) ended the accept loop without ending
the process. Everything background kept running -- llama-swap polls, LM Studio
sync, DB writes -- so Task Scheduler called the task Running, RestartCount
never fired, and the hub read connection refused as "offline" for as long as
the zombie lived. The box was on the whole time. The fix is not a better
accept loop; it is an honest one: once the gateway has proven it can answer,
consecutive failures end the process so the supervisor's restart-on-failure
(Task Scheduler's RestartCount, systemd's Restart=on-failure) can do the job
it was already configured for.

Every test here drives `_self_watchdog_tick` by hand, with the two probes
(listener, loopback /health) monkeypatched -- no thread timing, no real
sockets, and no test process ever in a position to be told to exit.
"""
from __future__ import annotations

import pytest

import app as gw


@pytest.fixture
def fresh(monkeypatch):
    """A disarmed watchdog, and an `os._exit` that records instead of kills.

    Resetting state matters more than it looks: `_watchdog_state` is module
    global, so an arming from one test would let the next one exit after a
    single miss."""
    gw._watchdog_state.update(armed=False, misses=0)
    exits: list[int] = []
    monkeypatch.setattr(gw.os, "_exit", lambda code=1: exits.append(code))
    yield exits
    gw._watchdog_state.update(armed=False, misses=0)


def drive(exits, listening, healthy, times=1):
    """Tick the watchdog `times` times under fixed probe readings."""
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(gw, "_own_listener_pid", lambda: listening)
        monkey.setattr(gw, "_loopback_health_probe",
                       lambda: "ok" if healthy else "dead")
        for _ in range(times):
            gw._self_watchdog_tick()
    finally:
        monkey.undo()
    return exits


# --------------------------------------------------------------------------
# arming: a probe that has never succeeded must never kill
# --------------------------------------------------------------------------


class TestArming:
    def test_failing_probes_leave_it_dormant(self, fresh):
        drive(fresh, listening=False, healthy=False, times=5)
        assert gw._watchdog_state["armed"] is False
        assert fresh == []

    def test_it_does_not_arm_on_someone_elses_server(self, fresh):
        """/health answering while THIS process owns no listener means some
        other process holds the port. Arming on that would put this
        watchdog's finger on a trigger another process controls."""
        drive(fresh, listening=False, healthy=True, times=5)
        assert gw._watchdog_state["armed"] is False
        assert fresh == []

    def test_a_listener_and_an_answer_arm_it(self, fresh):
        drive(fresh, listening=True, healthy=True)
        assert gw._watchdog_state["armed"] is True
        assert gw._watchdog_state["misses"] == 0

    def test_the_kill_switch_disarms_the_tick_completely(self, fresh,
                                                         monkeypatch):
        monkeypatch.setattr(gw, "_watchdog_enabled", lambda: False)
        drive(fresh, listening=True, healthy=True)
        assert gw._watchdog_state["armed"] is False


# --------------------------------------------------------------------------
# the zombie state itself: proven once, then gone
# --------------------------------------------------------------------------


class TestTheDeafGateway:
    def test_three_consecutive_failures_end_the_process(self, fresh):
        drive(fresh, listening=True, healthy=True)          # arm
        drive(fresh, listening=False, healthy=False, times=2)
        assert fresh == [], "below the threshold it must only complain"
        drive(fresh, listening=False, healthy=False)
        assert fresh == [1], "the third miss is the exit"
        assert gw._watchdog_state["misses"] >= gw.WATCHDOG_MAX_MISSES

    def test_the_living_listener_but_dead_http_case_is_a_miss_too(self, fresh):
        """The observed failure's uglier cousin: the socket exists, accepts
        never complete. Listening alone is not health."""
        drive(fresh, listening=True, healthy=True)          # arm
        drive(fresh, listening=True, healthy=False, times=gw.WATCHDOG_MAX_MISSES)
        assert fresh == [1]

    def test_one_good_answer_resets_the_count(self, fresh):
        drive(fresh, listening=True, healthy=True)          # arm
        drive(fresh, listening=False, healthy=False, times=2)
        drive(fresh, listening=True, healthy=True)          # recovered
        assert gw._watchdog_state["misses"] == 0
        drive(fresh, listening=False, healthy=False, times=2)
        assert fresh == [], "recovery must buy a full threshold again"
