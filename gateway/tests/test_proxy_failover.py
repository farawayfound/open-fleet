"""The /v1 proxy's failover behaviour, and fleet_chat's (the team/warm-up
internal caller) matching semantics.

No real network I/O: httpx.AsyncClient.send is monkeypatched at the class
level (every peer attempt builds its own AsyncClient), and _post_chat is
monkeypatched directly for the fleet_chat tests, per the house style in
test_public.py.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import asyncio
import json
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
    gw._host_cooldown.clear()
    gw._inflight.clear()
    yield
    gw._host_cooldown.clear()
    gw._inflight.clear()


def _peers(names):
    return [{"name": n, "url": "http://%s:8080" % n, "token": "t"} for n in names]


def _wire_targets(monkeypatch, targets, peer_names=None):
    """Point the /v1 proxy's routing at a fixed candidate list, no matter
    what the real (network-touching) resolve_targets()/load_peers()/
    peer_inference_key() would have said."""
    async def _resolve(model, **kw):
        return targets

    async def _pik(cand):
        return "k"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers", lambda: _peers(peer_names or [t[0] for t in targets]))
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


def _assert_no_leftover_inference_work():
    assert not any(v.get("kind") == "inference" for v in gw._active.values())
    for host in ("p1", "p2"):
        assert gw._inflight.get(host, 0) == 0


# ---------------------------------------------------------------------------
# native_proxy(): in-flight tracking (step 4) -- a native call used to be
# invisible to _inflight[HOST_NAME], so a concurrent /v1 or fleet_chat
# request could rank this host as idle while it was actually busy here.
# ---------------------------------------------------------------------------

class TestNativeProxyInFlight:
    def test_tracked_while_in_flight_and_cleared_on_success(self, client, monkeypatch):
        raw, _meta = gw.mint_key("native-inflight-1")
        seen: list[int] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(gw._inflight.get(gw.HOST_NAME, 0))
            return httpx.Response(
                200, request=request,
                json={"message": {"role": "assistant", "content": "ok"}, "done": True,
                     "prompt_eval_count": 1, "eval_count": 1})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        assert seen == [1]   # counted from the moment it was asked
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0

    def test_cleared_on_connect_error(self, client, monkeypatch):
        raw, _meta = gw.mint_key("native-inflight-2")

        async def fake_send(self, request, stream=False, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0

    def test_cleared_on_a_broken_connection_after_it_was_accepted(
            self, client, monkeypatch):
        # A peer that accepts the TCP connection and then dies before a
        # response arrives (OOM, crash, restart) surfaces as httpx.ReadError
        # or httpx.RemoteProtocolError -- neither is a ConnectError, and the
        # except clause used to only catch ConnectError/ConnectTimeout. Left
        # uncaught, the in-flight count this box was charged the moment it
        # was asked would never be released, and it would sort as saturated
        # forever after.
        raw, _meta = gw.mint_key("native-inflight-5")

        async def fake_send(self, request, stream=False, **kw):
            raise httpx.RemoteProtocolError("peer closed connection without sending a response")

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0

    def test_cleared_after_a_streamed_reply(self, client, monkeypatch):
        raw, _meta = gw.mint_key("native-inflight-3")

        async def fake_send(self, request, stream=False, **kw):
            resp = httpx.Response(200, request=request,
                                  headers={"content-type": "application/x-ndjson"})

            async def it():
                yield (json.dumps({"message": {"role": "assistant", "content": "hi"},
                                   "done": True, "prompt_eval_count": 1,
                                   "eval_count": 1}) + "\n").encode()
            resp.aiter_bytes = it
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        list(r.iter_lines())   # drain the stream so the relay()'s finally runs
        assert gw._inflight.get(gw.HOST_NAME, 0) == 0

    def test_a_concurrent_pair_sees_the_host_as_saturated(self, client, monkeypatch):
        # The whole point: while this native call is "in flight", the shared
        # scorer predicate must see this box as occupied.
        raw, _meta = gw.mint_key("native-inflight-4")
        gw._routes_cache["cap"] = {("", "n"): 1}
        assert gw._host_saturated("", "n") is False

        async def fake_send(self, request, stream=False, **kw):
            assert gw._host_saturated("", "n") is True
            return httpx.Response(
                200, request=request,
                json={"message": {"role": "assistant", "content": "ok"}, "done": True,
                     "prompt_eval_count": 1, "eval_count": 1})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        assert gw._host_saturated("", "n") is False


# ---------------------------------------------------------------------------
# /v1 proxy failover
# ---------------------------------------------------------------------------

class TestOpenAIProxyFailover:
    def test_p1_503_then_p2_200_failover_and_cooldown(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                return httpx.Response(503, request=request, json={"error": {"message": "down"}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-1")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"] == "hi"
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_p1_remote_protocol_error_failover(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                raise httpx.RemoteProtocolError("connection reset")
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-2")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_p1_connect_error_failover(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                raise httpx.ConnectError("refused")
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-3")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_single_target_503_passes_through_and_cools_down(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(503, request=request, json={"error": {"message": "down"}})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-4")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        # No second box to try: the 503 is the caller's answer, not a 502.
        assert r.status_code == 503
        assert gw.host_cooling("p1") is True
        assert not any(v.get("kind") == "inference" for v in gw._active.values())
        assert gw._inflight.get("p1", 0) == 0

    def test_p1_429_then_p2_200_gets_the_short_busy_cooldown(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                return httpx.Response(429, request=request,
                                      json={"error": {"message": "rate limited"}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-busy-1")
        before = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert gw.host_cooling("p1") is True
        # 429 is 'busy' (requirement 3), not a real failure -- it earns the
        # short cooldown, not the 45s COOLDOWN_UPSTREAM_5XX a real 5xx gets.
        remaining = gw._host_cooldown["p1"] - before
        assert remaining <= gw.COOLDOWN_BUSY_DEFAULT + 2
        _assert_no_leftover_inference_work()

    def test_single_target_429_also_gets_the_short_busy_cooldown(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(429, request=request,
                                  json={"error": {"message": "rate limited"}})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-busy-2")
        before = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429
        assert gw.host_cooling("p1") is True
        remaining = gw._host_cooldown["p1"] - before
        assert remaining <= gw.COOLDOWN_BUSY_DEFAULT + 2

    def test_ctx_too_long_retries_same_box_and_succeeds(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m")])
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content or b"{}"))
            if len(seen) == 1:
                return httpx.Response(400, request=request, json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-ctx-retry-1")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert len(seen) == 2   # one same-box retry, not a failover
        assert gw.host_cooling("p1") is False   # the retry succeeded
        row = gw.db_query(
            "SELECT ctx FROM model_ctx WHERE model=? AND host=?", ("m", "p1"))
        assert row and row[0]["ctx"] == 8192 - gw.CTX_OVERFLOW_MARGIN
        _assert_no_leftover_inference_work()

    def test_ctx_retry_keeps_one_continuous_tracked_window(self, client, monkeypatch):
        # The failed attempt and its retry must share ONE _track() window --
        # closing and reopening it between them, even briefly, would let a
        # concurrent request see this host as idle mid-retry and pile on.
        _wire_targets(monkeypatch, [("p1", "m")])
        track_calls: list[str] = []
        real_track = gw._track

        def spy_track(host):
            track_calls.append(host)
            return real_track(host)

        monkeypatch.setattr(gw, "_track", spy_track)
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content or b"{}"))
            if len(seen) == 1:
                return httpx.Response(400, request=request, json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-ctx-retry-track")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert len(seen) == 2
        assert track_calls.count("p1") == 1

    def test_ctx_too_long_retry_that_also_fails_falls_through_to_next_box(
            self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        seen: dict[str, int] = {"p1": 0, "p2": 0}

        async def fake_send(self, request, stream=False, **kw):
            host = "p1" if "p1" in str(self.base_url) else "p2"
            seen[host] += 1
            if host == "p1":
                return httpx.Response(400, request=request, json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-ctx-retry-2")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert seen["p1"] == 2   # the one same-box retry, then gave up
        assert seen["p2"] == 1
        assert gw.host_cooling("p1") is True   # exhausted its retry
        _assert_no_leftover_inference_work()

    def test_single_target_ctx_too_long_exhausted_retry_returns_honest_413(
            self, client, monkeypatch):
        # No other candidate to fall over to: the only box declined on
        # length both times. That is a 413 about the prompt (the box's own
        # detail, named), not a 502 that reads as an outage and invites a
        # retry that cannot possibly succeed -- this reactive path used to
        # set upstream_failed=True and fall all the way through to the
        # generic "no reachable host" 502, discarding the real reason.
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(400, request=request, json={
                "error": {"code": 400,
                         "message": "the request exceeds the available "
                                    "context size",
                         "type": "exceed_context_size_error",
                         "n_prompt_tokens": 9000, "n_ctx": 8192}})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-ctx-retry-honest-413")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 413, r.text
        body = r.json()
        assert body["error"]["type"] == "context_limit"
        assert body["error"]["limit"] == 8192
        assert gw.host_cooling("p1") is True   # exhausted its retry
        _assert_no_leftover_inference_work()

    def test_ctx_retry_live_off_skips_the_same_box_retry_single_target(
            self, client, monkeypatch):
        # The owner's opt-off toggle must actually skip the reactive
        # same-box retry, not just default to on -- no second attempt on
        # this box. But the failure is still ctx_too_long, so with no other
        # candidate to try, the honest 413/context_limit comes back (same
        # as the exhausted-retry case above) rather than the box's raw
        # response, and the box still gets cooled down like any other
        # candidate that could not take this request -- ctx_retry_live
        # gates only the same-box retry ATTEMPT, not the classification or
        # the failover behaviour it drives.
        _wire_targets(monkeypatch, [("p1", "m")])
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content or b"{}"))
            return httpx.Response(400, request=request, json={
                "error": {"code": 400,
                         "message": "the request exceeds the available "
                                    "context size",
                         "type": "exceed_context_size_error",
                         "n_prompt_tokens": 9000, "n_ctx": 8192}})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        real_settings = gw.get_public_settings

        def off_settings():
            s = real_settings()
            s["ctx_retry_live"] = False
            return s

        monkeypatch.setattr(gw, "get_public_settings", off_settings)
        raw, _meta = gw.mint_key("proxy-ctx-retry-off-1")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 413, r.text
        assert r.json()["error"]["type"] == "context_limit"
        assert len(seen) == 1   # no same-box retry attempted
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_ctx_retry_live_off_still_fails_over_to_the_next_candidate(
            self, client, monkeypatch):
        # Two candidates: with the toggle off, p1 must still be skipped in
        # favour of p2 -- ctx_too_long is "not on THIS box", independent of
        # whether the gateway is allowed to try shrinking the prompt for it
        # first. Before this fix, disabling ctx_retry_live skipped the
        # WHOLE ctx_too_long branch (classification included), so p1's raw
        # 400 was returned as this box's final answer and p2 was never
        # tried at all.
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])
        seen = {"p1": 0, "p2": 0}

        async def fake_send(self, request, stream=False, **kw):
            host = "p1" if "p1" in str(self.base_url) else "p2"
            seen[host] += 1
            if host == "p1":
                return httpx.Response(400, request=request, json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        real_settings = gw.get_public_settings

        def off_settings():
            s = real_settings()
            s["ctx_retry_live"] = False
            return s

        monkeypatch.setattr(gw, "get_public_settings", off_settings)
        raw, _meta = gw.mint_key("proxy-ctx-retry-off-2")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "max_tokens": 256,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert seen == {"p1": 1, "p2": 1}   # no same-box retry, straight to p2
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_ordinary_400_is_never_ctx_retried(self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m")])
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content or b"{}"))
            return httpx.Response(400, request=request,
                                  json={"error": {"message": "invalid 'temperature'"}})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-ctx-retry-3")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400, r.text
        assert len(seen) == 1   # never retried -- an ordinary client error
        assert gw.host_cooling("p1") is False

    def test_ttfb_deadline_triggers_failover(self, client, monkeypatch):
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda prompt_tokens, resident: 0.05)
        _wire_targets(monkeypatch, [("p1", "m"), ("p2", "m")])

        async def fake_send(self, request, stream=False, **kw):
            if "p1" in str(self.base_url):
                await asyncio.sleep(1.0)
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-5")
        started = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert time.time() - started < 0.9, "p2 should have answered long before p1's 1s sleep"
        assert gw.host_cooling("p1") is True
        _assert_no_leftover_inference_work()

    def test_single_target_no_deadline_awaits_slow_answer(self, client, monkeypatch):
        # Same short deadline as above, but with nobody else to ask: the
        # deadline must not apply, and the slow answer is simply awaited.
        monkeypatch.setattr(gw, "_ttfb_deadline", lambda prompt_tokens, resident: 0.01)
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            await asyncio.sleep(0.2)
            return httpx.Response(200, request=request, json=CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-6")
        started = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert time.time() - started >= 0.2

    def test_streaming_read_error_midstream_emits_error_frame_and_502_usage(
            self, client, monkeypatch):
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            resp = httpx.Response(200, request=request,
                                  headers={"content-type": "text/event-stream"})

            async def bad_iter():
                yield (b'data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}'
                       b'\n\n')
                raise httpx.ReadError("connection dropped")

            resp.aiter_bytes = bad_iter
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, meta = gw.mint_key("proxy-test-7")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        body = r.text
        assert '"error"' in body
        assert body.rstrip().endswith("data: [DONE]")
        row = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC LIMIT 1", (meta["id"],),
        )[0]
        assert row["status"] == 502
        _assert_no_leftover_inference_work()

    def test_streaming_single_target_429_keeps_the_busy_cooldown(
            self, client, monkeypatch):
        # The non-streaming sibling of this narrowing already has
        # test_single_target_429_also_gets_the_short_busy_cooldown; the
        # streaming relay()'s own first-byte _mark_host_ok gate had zero
        # coverage. Reverting `status_out not in (429, 503)` back to just
        # `status_out < 500` on the streaming path must fail this test.
        _wire_targets(monkeypatch, [("p1", "m")])

        async def fake_send(self, request, stream=False, **kw):
            resp = httpx.Response(429, request=request,
                                  headers={"content-type": "application/json"})

            async def it():
                yield b'{"error": {"message": "rate limited"}}'
            resp.aiter_bytes = it
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, _meta = gw.mint_key("proxy-test-stream-busy")
        before = time.time()
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 429, r.text
        list(r.iter_lines())   # drain the stream so relay()'s mark_host_ok gate runs
        assert gw.host_cooling("p1") is True
        remaining = gw._host_cooldown["p1"] - before
        assert remaining <= gw.COOLDOWN_BUSY_DEFAULT + 2
        _assert_no_leftover_inference_work()

    def test_fleet_pass_502_hides_hostnames_shows_box_alias(
            self, client, monkeypatch, fake_fleet, captured_mail, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "proxy-hide@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200, r.text
        raw_key = next(
            line.split("key: ", 1)[1] for line in captured_mail[-1]["text"].splitlines()
            if line.startswith("key: ")
        )
        _wire_targets(monkeypatch,
                     [("secret-host-one", "gemma4-31b-qat"), ("secret-host-two", "gemma4-31b-qat")])

        async def fake_send(self, request, stream=False, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        resp = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw_key},
            json={"model": "gemma4-31b-qat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 502
        assert "secret-host-one" not in resp.text
        assert "secret-host-two" not in resp.text
        assert "Box " in resp.text


class TestPublicNotices:
    def test_alias_when_public_raw_host_otherwise(self):
        settings = gw.get_public_settings()
        fallback = {"requested": "a", "served": "b"}
        _text, xf_public, hdrs_public = gw.public_notices(
            fallback, None, "some-real-host", settings, public=True)
        assert xf_public["host"] == gw.public_alias("some-real-host")
        assert "some-real-host" not in hdrs_public["X-Fleet-Fallback"]

        _text, xf_raw, hdrs_raw = gw.public_notices(
            fallback, None, "some-real-host", settings, public=False)
        assert xf_raw["host"] == "some-real-host"
        assert "host=some-real-host" in hdrs_raw["X-Fleet-Fallback"]


# ---------------------------------------------------------------------------
# fleet_chat: the same failover semantics for the team/warm-up internal caller
# ---------------------------------------------------------------------------

class TestFleetChatFailover:
    async def _key(self, name):
        raw, meta = gw.mint_key(name)
        return gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]

    async def test_failover_503_then_200(self, monkeypatch):
        async def fake_post_chat(cand, payload, read_timeout=None):
            if cand == "p1":
                return httpx.Response(503, request=httpx.Request("POST", "http://x"),
                                      json={"error": {"message": "down"}})
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m"), ("p2", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-1")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 200
        assert host == "p2"
        assert gw.host_cooling("p1") is True

    async def test_single_host_503_passes_through(self, monkeypatch):
        async def fake_post_chat(cand, payload, read_timeout=None):
            return httpx.Response(503, request=httpx.Request("POST", "http://x"),
                                  json={"error": {"message": "down"}})

        async def fake_resolve(model, **kw):
            return [("p1", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-2")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 503
        assert host == "p1"
        assert gw.host_cooling("p1") is True

    async def test_connect_error_failover(self, monkeypatch):
        async def fake_post_chat(cand, payload, read_timeout=None):
            if cand == "p1":
                raise httpx.ConnectError("refused")
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m"), ("p2", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-3")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 200
        assert host == "p2"
        assert gw.host_cooling("p1") is True

    async def test_role_is_forwarded_to_resolve_targets(self, monkeypatch):
        captured: dict = {}

        async def fake_resolve(model, **kw):
            captured.update(kw)
            return [("", "m")]

        async def fake_post_chat(cand, payload, read_timeout=None):
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        key = await self._key("fleetchat-4")
        await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-worker", role="worker")
        assert captured.get("role") == "worker"

    async def test_ctx_too_long_retries_same_box_and_succeeds(self, monkeypatch):
        calls: list[bytes] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(payload)
            if len(calls) == 1:
                return httpx.Response(400, request=httpx.Request("POST", "http://x"), json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-ctx-1")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 200
        assert host == "p1"
        assert len(calls) == 2   # one same-box retry, not a failover
        assert gw.host_cooling("p1") is False
        row = gw.db_query(
            "SELECT ctx FROM model_ctx WHERE model=? AND host=?", ("m", "p1"))
        assert row and row[0]["ctx"] == 8192 - gw.CTX_OVERFLOW_MARGIN

    async def test_ctx_too_long_retry_that_also_fails_falls_through_to_next_box(
            self, monkeypatch):
        calls: dict[str, int] = {"p1": 0, "p2": 0}

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls[cand] += 1
            if cand == "p1":
                return httpx.Response(400, request=httpx.Request("POST", "http://x"), json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m"), ("p2", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-ctx-2")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 200
        assert host == "p2"
        assert calls == {"p1": 2, "p2": 1}
        assert gw.host_cooling("p1") is True

    async def test_single_host_ctx_too_long_exhausted_retry_returns_honest_413(
            self, monkeypatch):
        # No other candidate to fall over to: the honest answer is the
        # box's own 413/context_limit, not the generic 502 upstream_failed
        # used to force once it swallowed the reactive retry's failure.
        calls: list[bytes] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(payload)
            return httpx.Response(400, request=httpx.Request("POST", "http://x"), json={
                "error": {"code": 400,
                         "message": "the request exceeds the available "
                                    "context size",
                         "type": "exceed_context_size_error",
                         "n_prompt_tokens": 9000, "n_ctx": 8192}})

        async def fake_resolve(model, **kw):
            return [("p1", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        key = await self._key("fleetchat-ctx-3")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 413
        assert body["error"]["type"] == "context_limit"
        assert body["error"]["limit"] == 8192
        assert len(calls) == 2   # one same-box retry, then gave up honestly
        assert gw.host_cooling("p1") is True

    async def test_ctx_retry_live_off_skips_the_same_box_retry_single_target(
            self, monkeypatch):
        # ctx_retry_live gates only the same-box retry ATTEMPT: with no
        # other candidate, the honest 413/context_limit still comes back
        # (matching the exhausted-retry case above), and the box is still
        # cooled down like any other candidate that could not take this
        # request -- not the box's raw response passed through untouched.
        calls: list[bytes] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(payload)
            return httpx.Response(400, request=httpx.Request("POST", "http://x"), json={
                "error": {"code": 400,
                         "message": "the request exceeds the available "
                                    "context size",
                         "type": "exceed_context_size_error",
                         "n_prompt_tokens": 9000, "n_ctx": 8192}})

        async def fake_resolve(model, **kw):
            return [("p1", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        real_settings = gw.get_public_settings

        def off_settings():
            s = real_settings()
            s["ctx_retry_live"] = False
            return s

        monkeypatch.setattr(gw, "get_public_settings", off_settings)
        key = await self._key("fleetchat-ctx-4")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 413
        assert body["error"]["type"] == "context_limit"
        assert len(calls) == 1   # no same-box retry attempted
        assert gw.host_cooling("p1") is True

    async def test_ctx_retry_live_off_still_fails_over_to_the_next_candidate(
            self, monkeypatch):
        calls: dict[str, int] = {"p1": 0, "p2": 0}

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls[cand] += 1
            if cand == "p1":
                return httpx.Response(400, request=httpx.Request("POST", "http://x"), json={
                    "error": {"code": 400,
                             "message": "the request exceeds the available "
                                        "context size",
                             "type": "exceed_context_size_error",
                             "n_prompt_tokens": 9000, "n_ctx": 8192}})
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=CHAT_OK_BODY)

        async def fake_resolve(model, **kw):
            return [("p1", "m"), ("p2", "m")]

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
        real_settings = gw.get_public_settings

        def off_settings():
            s = real_settings()
            s["ctx_retry_live"] = False
            return s

        monkeypatch.setattr(gw, "get_public_settings", off_settings)
        key = await self._key("fleetchat-ctx-5")
        status, body, host, granted = await gw.fleet_chat(
            key, {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/team-primary")
        assert status == 200
        assert host == "p2"
        assert calls == {"p1": 1, "p2": 1}   # no same-box retry, straight to p2
        assert gw.host_cooling("p1") is True
