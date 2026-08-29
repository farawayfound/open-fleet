"""A restart ENDS the answers in flight; it does not sever them.

The regression these pin down is apu-box-1 on 2026-08-28. The :30 reconcile in
deploy-gateway.yml restarted the gateway while two Copilot streams were open:

    19:51:38  Stopping llm-gateway.service...
    19:51:38  uvicorn: Waiting for connections to close.
    19:51:48  State 'stop-sigterm' timed out. Killing.
    19:51:48  Killing process 1586 (uvicorn) with signal SIGKILL

Ten seconds, because Mint ships DefaultTimeoutStopSec=10s and the unit never
said otherwise. SIGKILL leaves cloudflared holding origin sockets that die
mid-body with no HTTP framing, so the Cloudflare edge resets the HTTP/2
streams and the client is told ERR_HTTP2_PROTOCOL_ERROR -- a transport error
for what was really an orderly restart. Both requests failed in the same
instant, which is the signature of a process dying rather than a request
failing.

Three things had to be true for that to stop happening, and each has tests
here: the process must KNOW it is shutting down (draining), it must ask the
streams to end rather than let them be cut (drain_in_flight), and the
self-watchdog must not mistake an orderly shutdown for the deaf-gateway
zombie it hunts -- uvicorn closes the listening socket BEFORE it waits for
streams, so mid-drain the watchdog's own arming condition reads false.
"""
from __future__ import annotations

import asyncio

import pytest

import app as gw


@pytest.fixture
def undrained():
    """A process that is not shutting down, restored afterwards.

    `_draining` is a module global and a test that sets it would otherwise
    make every later test's proxy return 503."""
    gw._draining.clear()
    yield
    gw._draining.clear()


@pytest.fixture
def api_key(client):
    """A plain inference key -- the drain refusal happens before any budget
    or catalogue logic, so nothing about this key needs to be special."""
    raw, _meta = gw.mint_key("drain-test")
    return raw


@pytest.fixture
def jobs(undrained):
    """A clean job registry, restored afterwards."""
    with gw._active_lock:
        saved = dict(gw._active)
        gw._active.clear()
    yield gw._active
    with gw._active_lock:
        gw._active.clear()
        gw._active.update(saved)


def _job(kind="inference"):
    """A registered job with a real asyncio.Event, the way the proxy makes
    one -- the event is what _race_abort actually waits on."""
    return gw._job_open(kind, stop=asyncio.Event(), what="m", detail="d")


class TestDraining:
    def test_a_fresh_process_is_not_draining(self, undrained):
        assert gw.draining() is False

    def test_begin_drain_sets_the_flag(self, jobs):
        gw._begin_drain()
        assert gw.draining() is True

    def test_begin_drain_is_idempotent(self, jobs):
        """The handler is chained onto SIGTERM and SIGINT both, and a
        supervisor that sends one then the other must not start a second
        drain over the top of the first."""
        job = _job()
        gw._begin_drain()
        job["aborted"] = False          # pretend it somehow un-aborted
        gw._begin_drain()
        assert job["aborted"] is False  # the second call did nothing


class TestWindingUpTheWorkInFlight:
    def test_every_stream_is_asked_to_stop(self, jobs):
        a, b = _job(), _job()
        assert gw.drain_in_flight() == 2
        assert a["aborted"] and b["aborted"]
        assert a["stop"].is_set() and b["stop"].is_set()

    def test_a_drained_job_is_marked_as_such(self, jobs):
        """The mark is what lets the relay tell this apart from the
        dashboard's stop button: same lever, different meaning, different
        status code."""
        job = _job()
        gw.drain_in_flight()
        assert job["drained"] is True

    def test_downloads_are_left_alone(self, jobs):
        """A download resumes from the database after a restart (see
        active_jobs), so cutting it buys nothing and costs the bytes."""
        dl = gw._job_open("download", stop=asyncio.Event())
        stream = _job()
        assert gw.drain_in_flight() == 1
        assert stream["aborted"] is True
        assert dl["aborted"] is False

    def test_an_idle_box_drains_nothing(self, jobs):
        assert gw.drain_in_flight() == 0

    def test_inflight_work_counts_the_same_jobs(self, jobs):
        """/health and the deployer read this number; drain_in_flight acts on
        the same set. They must not be able to disagree."""
        _job("inference")
        _job("benchmark")
        gw._job_open("download", stop=asyncio.Event())
        assert gw.inflight_work() == 2
        assert gw.drain_in_flight() == 2


class TestTheStoppedResponse:
    def test_a_drain_is_a_retryable_503(self, jobs):
        job = _job()
        gw.drain_in_flight()
        r = gw._stopped_response(job)
        assert r.status_code == 503
        assert r.headers["retry-after"] == gw.DRAIN_RETRY_AFTER

    def test_the_dashboard_button_is_still_a_499(self, jobs):
        """499 says "whoever asked has gone away" and the budget predicate
        excludes it. A restart is not that: nobody asked, and the caller
        should try again."""
        job = _job()
        gw.job_abort(job)
        r = gw._stopped_response(job)
        assert r.status_code == 499


class TestHealthTellsTheDeployer:
    def test_health_reports_the_count(self, client, jobs):
        assert client.get("/health").json()["inflight"] == 0
        _job()
        assert client.get("/health").json()["inflight"] == 1

    def test_health_still_answers_200_while_draining(self, client, jobs):
        """The tunnel health check reads this endpoint, and a box winding
        down is still a box that is up. Only /v1 refuses."""
        gw._begin_drain()
        assert client.get("/health").status_code == 200


class TestTheProxyRefusesNewWork:
    def test_v1_answers_503_with_retry_after(self, client, api_key, jobs):
        gw._begin_drain()
        r = client.post("/v1/chat/completions",
                        headers={"Authorization": "Bearer " + api_key},
                        json={"model": "m", "messages": [{"role": "user",
                                                          "content": "hi"}]})
        assert r.status_code == 503
        assert r.headers["retry-after"] == gw.DRAIN_RETRY_AFTER
        assert r.json()["error"]["code"] == "gateway_restarting"

    def test_it_is_refused_before_any_upstream_call(self, client, api_key,
                                                   jobs, monkeypatch):
        """Refuse rather than start: a completion begun now has seconds to
        live and would be drained before its first token."""
        called: list[int] = []
        monkeypatch.setattr(gw, "model_routes",
                            lambda *a, **k: called.append(1))
        gw._begin_drain()
        client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer " + api_key},
                    json={"model": "m", "messages": []})
        assert called == []


class TestTheWatchdogStandsDownForAnOrderlyShutdown:
    def test_a_drain_is_not_a_miss(self, monkeypatch):
        """THE apu-box-1 regression. uvicorn closes the listener first and only
        then waits for streams, so mid-drain _own_listener_pid() is false --
        which, before this, scored "miss 1 of 3" against a process that was
        shutting down exactly as asked. Three of those and the watchdog would
        os._exit(1) in the middle of the drain, severing the very streams the
        drain exists to end cleanly."""
        gw._watchdog_state.update(armed=True, misses=0)
        exits: list[int] = []
        monkeypatch.setattr(gw.os, "_exit", lambda code=1: exits.append(code))
        monkeypatch.setattr(gw, "_own_listener_pid", lambda: False)
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        monkeypatch.setattr(gw, "draining", lambda: True)
        for _ in range(gw.WATCHDOG_MAX_MISSES + 2):
            gw._self_watchdog_tick()
        assert exits == []
        assert gw._watchdog_state["misses"] == 0
        gw._watchdog_state.update(armed=False, misses=0)

    def test_it_still_kills_a_deaf_gateway_that_is_not_draining(
            self, monkeypatch):
        """The stand-down must not disarm the thing entirely -- the apu-tablet-2
        zombie is still the case this watchdog exists for."""
        gw._watchdog_state.update(armed=True, misses=0)
        exits: list[int] = []
        monkeypatch.setattr(gw.os, "_exit", lambda code=1: exits.append(code))
        monkeypatch.setattr(gw, "_own_listener_pid", lambda: False)
        monkeypatch.setattr(gw, "_loopback_health_probe", lambda: "dead")
        monkeypatch.setattr(gw, "draining", lambda: False)
        for _ in range(gw.WATCHDOG_MAX_MISSES):
            gw._self_watchdog_tick()
        assert exits == [1]
        gw._watchdog_state.update(armed=False, misses=0)


class TestTheProbeBudgetsDoNotCollide:
    def test_the_watchdog_waits_longer_than_health_can_take(self):
        """Both were 5.0, which is why a slow llama-swap could score a miss
        against a healthy gateway: /health spent its whole upstream budget and
        answered at 5.0 s, the watchdog gave up at 5.0 s. The watchdog's
        timeout has to exceed this handler's worst case, not equal it."""
        assert gw.WATCHDOG_TIMEOUT_S > gw.UPSTREAM_HEALTH_BUDGET
