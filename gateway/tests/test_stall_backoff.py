"""The 2026-09-15 apu-tablet-1 case, in three parts.

Sixteen hub-hub requests went to apu-tablet-1 that day and every one was
metered 499: the hub's first-byte deadline (~130 s) was applied to
NON-streamed calls, whose first byte is the whole answer, so a box that
needed 135 s was cut off with the answer seconds away; the flat 120 s
cooldown then put it straight back in rotation, and the abort had left
llama-server busy for minutes, so the next request stalled too; and on the
box itself a Windows Update reboot had brought the model back "ready" with
its weights paged out, which nothing ever pulled in because no request was
allowed to finish.

  1. _answer_deadline: a non-streamed request's deadline covers generation.
  2. _stall_cooldown: consecutive stalls double the sit-out; an answer resets.
  3. warm-on-load: a model that turns up ready on an idle box gets one warm
     request; the Logs button reads the file a Windows box actually writes.

No real network I/O: httpx.AsyncClient.send is monkeypatched at the class
level, per the house style in test_proxy_failover.py.
Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio

CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


@pytest.fixture(autouse=True)
def _isolate_state():
    routes = dict(gw._routes_cache)   # the resolve test overwrites it wholesale
    gw._host_cooldown.clear()
    gw._stall_streak.clear()
    gw._inflight.clear()
    gw._host_last_used.clear()
    gw._warm_on_load_ready.clear()
    gw._warm_on_load_pending.clear()
    gw._warm_on_load_state.clear()
    gw._warm_on_load_done.clear()
    gw._warm_on_load_runs.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(routes)
    gw._host_cooldown.clear()
    gw._stall_streak.clear()
    gw._inflight.clear()
    gw._host_last_used.clear()
    gw._warm_on_load_ready.clear()
    gw._warm_on_load_pending.clear()
    gw._warm_on_load_state.clear()
    gw._warm_on_load_done.clear()
    gw._warm_on_load_runs.clear()


def _peers(names):
    return [{"name": n, "url": "http://%s:8080" % n, "token": "t"} for n in names]


def _wire_targets(monkeypatch, targets):
    async def _resolve(model, **kw):
        return targets

    async def _pik(cand):
        return "k"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers", lambda: _peers([t[0] for t in targets]))
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


# --------------------------------------------------------------------------
# 1. the deadline
# --------------------------------------------------------------------------

class TestAnswerDeadline:
    def test_streamed_is_the_first_byte_deadline_alone(self, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda p, r: 100.0)
        assert gw._answer_deadline(5000, 1024, True, stream=True) == 100.0
        assert gw._answer_deadline(5000, 1024, True, stream=True, tps=40.0) == 100.0

    def test_non_streamed_adds_generation_at_the_floor_rate(self, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda p, r: 100.0)
        # No history: 1024 tokens at GEN_RATE_FLOOR (3 tok/s) -- the same
        # pessimistic rate the fleet_chat/batch paths always used.
        assert gw._answer_deadline(5000, 1024, True, stream=False) == pytest.approx(100.0 + 1024 / 3.0)

    def test_non_streamed_uses_half_the_measured_decode_rate(self, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda p, r: 100.0)
        # apu-tablet-1's ~23 tok/s: generation budgeted at 11.5 tok/s.
        assert gw._answer_deadline(5000, 1024, True, stream=False, tps=23.0) \
            == pytest.approx(100.0 + 1024 / 11.5)
        # A measured rate below the floor never makes the deadline shorter
        # than the floor would.
        assert gw._answer_deadline(5000, 1024, True, stream=False, tps=2.0) \
            == pytest.approx(100.0 + 1024 / 3.0)

    def test_capped(self, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda p, r: 600.0)
        assert gw._answer_deadline(0, 1024, True, stream=False) == gw.ANSWER_DEADLINE_MAX

    def test_the_real_flowmaster_numbers(self):
        # 8k-token hub prompt, 1024 generated, resident box at 23 tok/s:
        # the old first-byte deadline was ~130 s; the box's honest answer
        # took ~135 s. The new deadline leaves room for it.
        old = gw._ttfb_deadline(8000, True)
        new = gw._answer_deadline(8000, 1024, True, stream=False, tps=23.0)
        assert old < 140
        assert new > 200


class TestProxyDeadline:
    def test_non_streamed_slow_answer_inside_the_generation_budget_is_awaited(
            self, client, monkeypatch):
        # First-byte budget 0.05 s, but the request is non-streamed and asks
        # for 256 tokens: the deadline covers generation, so p1's 0.3 s
        # answer arrives and p2 is never asked.
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda prompt_tokens, resident: 0.05)
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        seen: list[str] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(str(self.base_url))
            if "p1" in str(self.base_url):
                await asyncio.sleep(0.3)
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("stall-test-1")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert [s for s in seen if "p2" in s] == []
        assert gw.host_cooling("p1") is False
        assert gw.stall_streak("p1") == 0

    def test_streamed_request_still_fails_over_on_the_first_byte(self, client, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda prompt_tokens, resident: 0.05)
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                await asyncio.sleep(0.5)
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("stall-test-2")
        started = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "stream": True, "max_tokens": 256,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert time.time() - started < 0.45
        assert gw.host_cooling("p1") is True
        assert gw.stall_streak("p1") == 1


# --------------------------------------------------------------------------
# 2. the backoff
# --------------------------------------------------------------------------

class TestStallBackoff:
    def test_each_consecutive_stall_doubles_the_cooldown_up_to_the_cap(self):
        got = [gw._stall_cooldown("box") for _ in range(6)]
        assert got == [120.0, 240.0, 480.0, 960.0, 1800.0, 1800.0]
        assert gw.stall_streak("box") == 6

    def test_an_answer_resets_the_streak(self):
        gw._stall_cooldown("box")
        gw._stall_cooldown("box")
        gw._clear_stall("box")
        assert gw.stall_streak("box") == 0
        assert gw._stall_cooldown("box") == 120.0

    def test_mark_host_ok_does_not_reset_it(self):
        # The keep-warm loop marks a box ok after a /health touch, and a box
        # whose weights are paged out answers /health just fine.
        gw._stall_cooldown("box")
        gw._mark_host_ok("box")
        assert gw.stall_streak("box") == 1

    def test_streaks_are_per_host(self):
        gw._stall_cooldown("a")
        gw._stall_cooldown("a")
        assert gw._stall_cooldown("b") == 120.0

    def test_repeated_stalls_through_the_proxy_lengthen_the_sit_out(self, client, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda prompt_tokens, resident: 0.05)
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                await asyncio.sleep(0.4)
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("stall-test-3")
        body = {"model": "m", "stream": True, "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}]}
        for expect in (120.0, 240.0, 480.0):
            # The cooldown itself would keep p1 out of the second attempt's
            # candidate list in production; here routing is wired to a fixed
            # list, so each request reaches p1 again and stalls again.
            gw._host_cooldown.clear()
            r = client.post("/v1/chat/completions",
                            headers={"Authorization": "Bearer " + raw}, json=body)
            assert r.status_code == 200, r.text
            left = gw._host_cooldown["p1"] - time.time()
            assert expect - 5 < left <= expect

    def test_a_first_byte_from_the_box_clears_its_streak(self, client, monkeypatch):
        gw._stall_cooldown("p1")
        gw._stall_cooldown("p1")
        assert gw.stall_streak("p1") == 2
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("stall-test-4")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert gw.stall_streak("p1") == 0

    def test_a_streamed_first_chunk_clears_the_streak_too(self, client, monkeypatch):
        # The streaming relay clears at its first chunk (the other of the
        # /v1 proxy's two success paths).
        gw._stall_cooldown("p1")
        gw._stall_cooldown("p1")
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            resp = httpx.Response(200, request=request,
                                  headers={"content-type": "text/event-stream"})

            async def chunks():
                yield b'data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            resp.aiter_bytes = chunks
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("stall-test-5")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert r.text.rstrip().endswith("data: [DONE]")
        assert gw.stall_streak("p1") == 0

    async def test_fleet_chat_stalls_count_and_back_off_as_well(self, monkeypatch):
        # The fleet_chat (team / warm-up) path shares the streak: a stall
        # there is the same box not answering.
        calls: list[str] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(cand)
            if cand == "p1":
                assert read_timeout, "a deadline while another box remains"
                raise httpx.ReadTimeout("no answer")
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m"), ("p2", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        raw, meta = gw.mint_key("stall-fleetchat-1")
        key = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        for expect in (120.0, 240.0):
            gw._host_cooldown.clear()
            status, out, host, _granted = await gw.fleet_chat(key, body, "/v1/team-primary")
            assert status == 200 and host == "p2"
            left = gw._host_cooldown["p1"] - time.time()
            assert expect - 5 < left <= expect
        assert gw.stall_streak("p1") == 2
        # And an answer from p1 through the same path resets it.
        async def ok_post_chat(cand, payload, read_timeout=None):
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)
        monkeypatch.setattr(gw, "_post_chat", ok_post_chat)
        gw._host_cooldown.clear()
        status, out, host, _granted = await gw.fleet_chat(key, body, "/v1/team-primary")
        assert status == 200 and host == "p1"
        assert gw.stall_streak("p1") == 0

    def test_resolve_reports_the_streak(self, client, admin_headers, monkeypatch):
        gw._routes_cache.update(
            t=0.0, map={"m": "p1"}, cands={"m": ["p1"]}, cap={("p1", "m"): 1},
            running={"p1": {"m"}, gw.HOST_NAME: set()}, ctx={("p1", "m"): 32768},
            meta={("p1", "m"): {"fit": "unified", "bytes": 10_000_000_000}},
            alias={}, engine={}, warm={}, reachable={gw.HOST_NAME, "p1"},
        )

        async def _frozen(force: bool = False):
            return gw._routes_cache["map"]
        monkeypatch.setattr(gw, "model_routes", _frozen)
        for _ in range(2):   # what the proxy's stall branch does
            gw._mark_host_down("p1", gw._stall_cooldown("p1"), "test stall")
        r = client.get("/admin/api/resolve", params={"model": "m"}, headers=admin_headers)
        assert r.status_code == 200, r.text
        by_host = {t["host"]: t for t in r.json()["targets"]}
        assert by_host["p1"]["stalls"] == 2
        assert by_host["p1"]["cooling"] is True


# --------------------------------------------------------------------------
# 3. the box side
# --------------------------------------------------------------------------

class TestWarmOnLoad:
    def test_on_by_default_on_windows_only_and_env_overrides(self):
        assert gw.warm_on_load_enabled("Windows", "") is True
        assert gw.warm_on_load_enabled("Linux", "") is False
        assert gw.warm_on_load_enabled("Darwin", "") is False
        assert gw.warm_on_load_enabled("Linux", "1") is True
        assert gw.warm_on_load_enabled("Windows", "0") is False

    def test_warm_prompt_is_long_varied_and_deterministic(self):
        p = gw.warm_prompt()
        assert 5500 < len(p) < 8000
        assert p == gw.warm_prompt()
        # A different salt is a different walk: consecutive warm-ups of one
        # model must never share text llama-server's prompt cache could reuse.
        q = gw.warm_prompt(salt=1)
        assert q != p and q == gw.warm_prompt(salt=1)
        assert p.split("\n\n", 1)[1] != q.split("\n\n", 1)[1]
        # Several fragments present, not one repeated: the point is to route
        # through many experts, so the same handful of tokens will not do.
        hits = sum(1 for f in gw._WARM_FRAGMENTS if f in p)
        assert hits >= len(gw._WARM_FRAGMENTS) - 2

    async def test_a_fresh_ready_model_is_warmed_once(self, monkeypatch):
        ready = {"m"}
        runs: list[str] = []

        async def _ready():
            return set(ready)

        async def _run(mid):
            runs.append(mid)
            return True

        monkeypatch.setattr(gw, "_ready_model_ids", _ready)
        monkeypatch.setattr(gw, "_warm_on_load_run", _run)
        assert await gw._warm_on_load_tick() == ["m"]
        assert await gw._warm_on_load_tick() == []
        assert await gw._warm_on_load_tick() == []
        assert runs == ["m"]
        # Gone and back (a swap-out and a reload): warmed again.
        ready.clear()
        assert await gw._warm_on_load_tick() == []
        ready.add("m")
        assert await gw._warm_on_load_tick() == ["m"]
        assert runs == ["m", "m"]

    async def test_a_busy_box_waits_and_warms_when_idle(self, monkeypatch):
        runs: list[str] = []

        async def _ready():
            return {"m"}

        async def _run(mid):
            runs.append(mid)
            return True

        monkeypatch.setattr(gw, "_ready_model_ids", _ready)
        monkeypatch.setattr(gw, "_warm_on_load_run", _run)
        gw._inflight[gw.HOST_NAME] = 1
        assert await gw._warm_on_load_tick() == []
        assert gw._warm_on_load_state["m"]["status"] == "waiting"
        gw._inflight[gw.HOST_NAME] = 0
        gw._host_last_used[gw.HOST_NAME] = time.time()   # just answered one
        assert await gw._warm_on_load_tick() == []
        gw._host_last_used[gw.HOST_NAME] = time.time() - gw.WARM_ON_LOAD_IDLE - 1
        assert await gw._warm_on_load_tick() == ["m"]
        assert runs == ["m"]

    async def test_an_idle_resident_model_is_refreshed_after_the_refresh_interval(
            self, monkeypatch):
        runs: list[str] = []
        clock = {"t": 1_000_000.0}

        async def _ready():
            return {"m"}

        async def _run(mid):
            runs.append(mid)
            gw._warm_on_load_done[mid] = clock["t"]
            gw._host_last_used[gw.HOST_NAME] = clock["t"]
            return True

        monkeypatch.setattr(gw, "_ready_model_ids", _ready)
        monkeypatch.setattr(gw, "_warm_on_load_run", _run)
        monkeypatch.setattr(gw.time, "time", lambda: clock["t"])
        monkeypatch.setattr(gw, "WARM_ON_LOAD_REFRESH_S", 600.0)
        assert await gw._warm_on_load_tick() == ["m"]      # the arrival warm-up
        clock["t"] += 300
        assert await gw._warm_on_load_tick() == []         # not idle long enough
        clock["t"] += 301
        assert await gw._warm_on_load_tick() == ["m"]      # idle 601 s: refreshed
        assert runs == ["m", "m"]
        # A real request resets the idle clock: no refresh until it is idle again.
        clock["t"] += 500
        gw._host_last_used[gw.HOST_NAME] = clock["t"]
        clock["t"] += 200
        assert await gw._warm_on_load_tick() == []
        # Busy box: never.
        clock["t"] += 1000
        gw._inflight[gw.HOST_NAME] = 1
        assert await gw._warm_on_load_tick() == []
        gw._inflight[gw.HOST_NAME] = 0
        assert await gw._warm_on_load_tick() == ["m"]
        # Refresh off: only the arrival warm-up ever runs.
        monkeypatch.setattr(gw, "WARM_ON_LOAD_REFRESH_S", 0.0)
        clock["t"] += 5000
        assert await gw._warm_on_load_tick() == []
        # A model that went away is forgotten; back again, it is an arrival.
        monkeypatch.setattr(gw, "WARM_ON_LOAD_REFRESH_S", 600.0)
        gone = {"x"}

        async def _ready2():
            return set(gone)
        monkeypatch.setattr(gw, "_ready_model_ids", _ready2)
        clock["t"] += 5000
        assert await gw._warm_on_load_tick() == ["x"]
        assert "m" not in gw._warm_on_load_done

    async def test_a_poll_that_could_not_ask_keeps_the_bookkeeping(self, monkeypatch):
        # /running unreachable for one tick must not read as "everything
        # went away" -- that forgot every warm-up and redid them all on the
        # next tick, on the box's one slot, for a model that never left.
        answers = [{"m"}, None, {"m"}, {"m"}]
        runs: list[str] = []

        async def _ready():
            return answers.pop(0)

        async def _run(mid):
            runs.append(mid)
            gw._warm_on_load_done[mid] = time.time()
            return True

        monkeypatch.setattr(gw, "_ready_model_ids", _ready)
        monkeypatch.setattr(gw, "_warm_on_load_run", _run)
        assert await gw._warm_on_load_tick() == ["m"]
        assert await gw._warm_on_load_tick() == []    # could not ask
        assert "m" in gw._warm_on_load_done and "m" in gw._warm_on_load_ready
        assert await gw._warm_on_load_tick() == []    # back, unchanged: no re-warm
        assert await gw._warm_on_load_tick() == []
        assert runs == ["m"]

    async def test_ready_ids_is_none_when_the_engine_cannot_be_asked(self, monkeypatch):
        class _Boom:
            async def get(self, path, timeout=None):
                raise httpx.ConnectError("restarting")

        class _NotFound:
            async def get(self, path, timeout=None):
                class R:
                    status_code = 404

                    def json(self):
                        return {}
                return R()

        class _Empty:
            async def get(self, path, timeout=None):
                class R:
                    status_code = 200

                    def json(self):
                        return {"running": []}
                return R()

        monkeypatch.setattr(gw, "client", _Boom())
        assert await gw._ready_model_ids() is None
        monkeypatch.setattr(gw, "client", _NotFound())   # an Ollama box
        assert await gw._ready_model_ids() is None
        monkeypatch.setattr(gw, "client", _Empty())      # answered: nothing resident
        assert await gw._ready_model_ids() == set()
        monkeypatch.setattr(gw, "client", None)
        assert await gw._ready_model_ids() is None

    async def test_a_starting_model_is_not_ready(self, monkeypatch):
        class _Resp:
            status_code = 200

            def json(self):
                return {"running": [{"model": "a", "state": "starting"},
                                    {"model": "b", "state": "ready"},
                                    {"model": "c"}]}

        class _Client:
            async def get(self, path, timeout=None):
                assert path == "/running"
                return _Resp()

        monkeypatch.setattr(gw, "client", _Client())
        assert await gw._ready_model_ids() == {"b", "c"}

    async def test_the_warm_request_is_one_streamed_token_and_records_timings(self, monkeypatch):
        seen: dict = {}

        class _Stream:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"x"}}]}'
                yield ('data: {"choices":[],"timings":{"prompt_n":1500,'
                       '"prompt_ms":15000,"predicted_n":1,"predicted_ms":50}}')
                yield "data: [DONE]"

        class _Client:
            def stream(self, method, path, json=None, timeout=None):
                seen.update(method=method, path=path, body=json)
                return _Stream()

        monkeypatch.setattr(gw, "client", _Client())
        assert await gw._warm_on_load_run("m") is True
        assert seen["method"] == "POST" and seen["path"] == "/v1/chat/completions"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["stream"] is True
        assert seen["body"]["max_tokens"] == 1
        assert seen["body"]["cache_prompt"] is False
        first = seen["body"]["messages"][0]["content"]
        note = gw._warm_on_load_state["m"]
        assert note["status"] == "ok"
        assert note["detail"] == "prompt 1500 tok at 100.0 tok/s"
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0   # untracked afterwards
        # The next warm-up of the same model sends different text.
        assert await gw._warm_on_load_run("m") is True
        assert seen["body"]["messages"][0]["content"] != first
        assert gw._warm_on_load_runs["m"] == 2

    async def test_a_prompt_cache_hit_is_reported_as_a_failed_warm_up(self, monkeypatch):
        class _Stream:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def aiter_lines(self):
                yield ('data: {"choices":[],"timings":{"prompt_n":4,'
                       '"prompt_ms":24000,"predicted_n":1,"predicted_ms":50}}')
                yield "data: [DONE]"

        class _Client:
            def stream(self, method, path, json=None, timeout=None):
                return _Stream()

        monkeypatch.setattr(gw, "client", _Client())
        assert await gw._warm_on_load_run("m") is False
        note = gw._warm_on_load_state["m"]
        assert note["status"] == "failed"
        assert "SHORT" in note["detail"] and "prompt 4 tok" in note["detail"]

    async def test_a_failed_warm_request_is_a_note_not_a_crash(self, monkeypatch):
        class _Client:
            def stream(self, method, path, json=None, timeout=None):
                raise httpx.ConnectError("engine restarting")

        monkeypatch.setattr(gw, "client", _Client())
        assert await gw._warm_on_load_run("m") is False
        assert gw._warm_on_load_state["m"]["status"] == "failed"
        assert "ConnectError" in gw._warm_on_load_state["m"]["detail"]
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0

    def test_status_carries_the_report(self, client, admin_headers):
        gw._warm_on_load_state["m"] = {"model": "m", "at": "t", "status": "ok",
                                       "ms": 1234, "detail": ""}
        r = client.get("/admin/api/status", headers=admin_headers)
        assert r.status_code == 200, r.text
        rep = r.json()["warm_on_load"]
        assert rep["enabled"] in (True, False)
        assert "refresh_s" in rep
        assert [m["model"] for m in rep["models"]] == ["m"]


class TestWindowsLogs:
    def test_reads_the_file_the_run_scripts_write(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Windows")
        path = gw.STATE / "llama-swap.log"
        path.write_text("".join("line %d\n" % i for i in range(1, 501)), encoding="utf-8")
        try:
            r = client.get("/admin/api/logs", params={"unit": "llama-swap", "lines": 50},
                           headers=admin_headers)
            assert r.status_code == 200, r.text
            lines = r.text.splitlines()
            assert lines[0] == "line 451" and lines[-1] == "line 500" and len(lines) == 50
            r = client.get("/admin/api/logs", params={"unit": "cloudflared"}, headers=admin_headers)
            assert r.status_code == 501
            path.unlink()
            r = client.get("/admin/api/logs", params={"unit": "llama-swap"}, headers=admin_headers)
            assert r.status_code == 200 and r.text.startswith("no log yet")
            # The gateway's own log rides the same route under its unit name.
            gw_log = gw.STATE / "gateway.log"
            gw_log.write_text("INFO: started\nWARNING: self-watchdog\n", encoding="utf-8")
            try:
                r = client.get("/admin/api/logs", params={"unit": "llm-gateway", "lines": 10},
                               headers=admin_headers)
                assert r.status_code == 200, r.text
                assert r.text.splitlines() == ["INFO: started", "WARNING: self-watchdog"]
            finally:
                gw_log.unlink()
        finally:
            if path.exists():
                path.unlink()

    def test_tail_walks_back_in_blocks(self, tmp_path):
        p = tmp_path / "big.log"
        p.write_bytes(b"".join(b"row %06d\n" % i for i in range(200_000)))   # ~2 MB
        out = gw._tail_file(p, 3, block=4096).splitlines()
        assert out == ["row 199997", "row 199998", "row 199999"]
        assert gw._tail_file(p, 10_000, block=4096).splitlines()[0] == "row 190000"
