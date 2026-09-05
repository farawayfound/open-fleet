"""The /v1/batches dispatcher (requirements 1-3 as they apply to batches):
_batch_targets' host selection and model substitution, and _batch_run's
worker loop -- failure classification, cooldown-driven backoff instead of an
immediate resend to the box that just failed, and the reactive ctx_too_long
same-box retry.

No real network I/O: gw._post_chat is monkeypatched directly, per the house
style in test_proxy_failover.py's TestFleetChatFailover.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio

OK_BODY = {
    "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

CTX_OVERFLOW_BODY = {
    "error": {"code": 400,
             "message": "the request exceeds the available context size",
             "type": "exceed_context_size_error",
             "n_prompt_tokens": 9000, "n_ctx": 8192}}


@pytest.fixture(autouse=True)
def _isolate_state():
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()


def _wire_routes(monkeypatch, cands, cap, running=None, meta=None):
    gw._routes_cache.update(
        t=time.time(), map={}, cands=cands, cap=cap,
        running=running or {}, meta=meta or {})

    async def _routes(force: bool = False):
        return gw._routes_cache["map"]

    monkeypatch.setattr(gw, "model_routes", _routes)


_bid_counter = [90000]


def _new_batch(bodies: list[dict]) -> tuple[int, Path, Path]:
    _bid_counter[0] += 1
    bid = _bid_counter[0]
    in_path, out_path = gw._batch_paths(bid)
    gw.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    in_path.write_text("\n".join(json.dumps(b) for b in bodies) + "\n", "utf-8")
    if out_path.exists():
        out_path.unlink()
    return bid, in_path, out_path


def _read_output(out_path) -> list[dict]:
    if not out_path.exists():
        return []
    return [json.loads(line) for line in out_path.read_text("utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# _batch_targets: worker sizing (5d) and ctx-aware ranking (5e)
# ---------------------------------------------------------------------------

class TestBatchTargets:
    async def test_workers_sized_against_current_inflight(self, monkeypatch):
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 4}, {"h1": {"m"}})
        gw._inflight["h1"] = 2
        targets = await gw._batch_targets(["m"])
        assert targets[0]["workers"] == 4 - 2

    async def test_workers_never_drop_below_one(self, monkeypatch):
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 2}, {"h1": {"m"}})
        gw._inflight["h1"] = 9   # more in flight than the box's own capacity
        targets = await gw._batch_targets(["m"])
        assert targets[0]["workers"] == 1

    async def test_need_ctx_demotes_a_known_small_window(self, monkeypatch):
        _wire_routes(
            monkeypatch, {"m": ["small", "big"]},
            {("small", "m"): 1, ("big", "m"): 1}, {}, {})
        gw._routes_cache["ctx"] = {("small", "m"): 4096}   # known and small
        targets = await gw._batch_targets(["m"], need_ctx=8192)
        assert targets[0]["host"] == "big"

    def test_batch_need_ctx_takes_the_max_across_lines(self):
        short = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
        long = json.dumps({"messages": [{"role": "user", "content": "x" * 4000}]})
        lines = short + "\n" + long + "\n\n"   # a blank line must not blow this up
        assert gw._batch_need_ctx(lines) == max(
            gw.estimate_prompt_tokens(json.loads(short)),
            gw.estimate_prompt_tokens(json.loads(long)))


# ---------------------------------------------------------------------------
# _batch_targets: substitution (requirement 2 applied to batches)
# ---------------------------------------------------------------------------

class TestBatchSubstitution:
    async def test_a_model_with_no_candidate_is_substituted(self, fake_fleet):
        # nemotron3.5-lightning-30b has no candidate at all in fake_fleet;
        # qwen3.6-35b-a3b (test_public.py's TestFallback own substitute for
        # this row) does.
        targets = await gw._batch_targets(["nemotron3.5-lightning-30b"])
        assert targets
        assert all(t["model"] != "nemotron3.5-lightning-30b" for t in targets)
        assert any(t["model"] == "qwen3.6-35b" for t in targets)

    async def test_opt_out_suppresses_batch_substitution(self, fake_fleet):
        targets = await gw._batch_targets(
            ["nemotron3.5-lightning-30b"], no_fallback=True)
        assert targets == []   # nothing serves it, and no substitute was sought


# ---------------------------------------------------------------------------
# _batch_run's worker loop: failure classification, cooldown-driven backoff,
# and the reactive ctx_too_long same-box retry
# ---------------------------------------------------------------------------

class TestBatchWorkerFailureHandling:
    async def test_a_stall_marks_the_host_down_and_the_item_is_recorded_failed(
            self, monkeypatch):
        # A single-target batch: the sole worker retires on the stall (like
        # a connect failure), and the item it handed back has nowhere left
        # to run -- the drain loop after gather() records it, honestly, as
        # "no reachable host left".
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            raise httpx.TimeoutException("no answer")

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is False
        assert "no reachable host" in out[0]["body"]["error"]["message"]
        assert gw.host_cooling("h1") is True

    async def test_a_connect_error_marks_the_host_down(self, monkeypatch):
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert gw.host_cooling("h1") is True

    async def test_a_500_marks_the_host_down_too(self, monkeypatch):
        # Kept short so the requeue-with-backoff loop below does not burn
        # the real 45s COOLDOWN_UPSTREAM_5XX before this test can finish --
        # long enough, though, that the cooldown is still live when the
        # final assertion runs.
        monkeypatch.setattr(gw, "COOLDOWN_UPSTREAM_5XX", 1.0)
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            return httpx.Response(500, request=httpx.Request("POST", "http://x"),
                                  json={"error": {"message": "boom"}})

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert gw.host_cooling("h1") is True

    async def test_a_cooling_host_never_immediately_resends_a_requeued_item(
            self, monkeypatch):
        # h1 always fails and is marked down on its very first attempt --
        # host_cooling() then keeps its own next loop off the queue, so a
        # SECOND box (h2, which always succeeds) is what actually finishes
        # both items, whichever one h1 happened to grab first.
        _wire_routes(monkeypatch, {"m": ["h1", "h2"]},
                    {("h1", "m"): 1, ("h2", "m"): 1},
                    {"h1": {"m"}, "h2": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            if cand == "h1":
                raise httpx.ConnectError("refused")
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch([
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "there"}]},
        ])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 2
        assert all(row["ok"] for row in out)
        assert all(row["host"] == "h2" for row in out)
        assert gw.host_cooling("h1") is True

    async def test_a_busy_429_is_retried_not_recorded_as_an_immediate_failure(
            self, monkeypatch):
        monkeypatch.setattr(gw, "_busy_cooldown_seconds", lambda: 0.05)
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        calls = {"n": 0}

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, request=httpx.Request("POST", "http://x"),
                                      json={"error": {"message": "rate limited"}})
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is True
        assert calls["n"] == 3   # two 429s, requeued, then the third succeeds

    async def test_ctx_too_long_retries_the_same_box_once_and_remembers_it(
            self, monkeypatch):
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        calls: list[dict] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(json.loads(payload))
            if len(calls) == 1:
                return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                      json=CTX_OVERFLOW_BODY)
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is True
        assert len(calls) == 2   # one same-box retry, never resent as-is
        assert gw.host_cooling("h1") is False   # the retry succeeded
        row = gw.db_query(
            "SELECT ctx FROM model_ctx WHERE model=? AND host=?", ("m", "h1"))
        assert row and row[0]["ctx"] == 8192 - gw.CTX_OVERFLOW_MARGIN

    async def test_ctx_retry_keeps_one_continuous_tracked_window(self, monkeypatch):
        # Same invariant as the live proxy's: the failed attempt and its
        # retry must share one _track() window, not two -- closing it in
        # between would let a concurrent request see this host as idle
        # mid-retry.
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        track_calls: list[str] = []
        real_track = gw._track

        def spy_track(host):
            track_calls.append(host)
            return real_track(host)

        monkeypatch.setattr(gw, "_track", spy_track)
        calls: list[dict] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(json.loads(payload))
            if len(calls) == 1:
                return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                      json=CTX_OVERFLOW_BODY)
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert len(calls) == 2
        assert track_calls.count("h1") == 1

    async def test_ctx_too_long_retry_that_also_fails_is_recorded_failed(
            self, monkeypatch):
        # A short MIDSTREAM cooldown so this test does not have to wait out
        # the real 60s cooldown across the (up to 3) attempts a single-host
        # batch now spends -- requeuing on an exhausted ctx retry, exactly
        # like a busy/5xx failure -- before finally giving up.
        monkeypatch.setattr(gw, "COOLDOWN_MIDSTREAM", 0.05)
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        calls = {"n": 0}

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls["n"] += 1
            return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                  json=CTX_OVERFLOW_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is False
        assert gw.host_cooling("h1") is True   # exhausted its one retry
        # 3 attempts (the shared attempts[idx] cap), each spending its own
        # one same-box ctx-fit retry: 6 calls, not 1 -- the item WAS given
        # more chances, just never on a different (nonexistent) host.
        assert calls["n"] == 6

    async def test_ctx_too_long_exhausted_retry_fails_over_to_a_different_host(
            self, monkeypatch):
        # requirement 3: ctx_too_long is a decline on length, not a broken
        # box -- exactly like a busy/5xx failure, an exhausted same-box
        # retry must still give a DIFFERENT host serving the same model a
        # chance, rather than being recorded as a permanent failure while a
        # sibling worker on that other host sits idle.
        monkeypatch.setattr(gw, "COOLDOWN_MIDSTREAM", 0.05)
        _wire_routes(monkeypatch, {"m": ["h1", "h2"]},
                    {("h1", "m"): 1, ("h2", "m"): 1},
                    {"h1": {"m"}, "h2": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            if cand == "h1":
                return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                      json=CTX_OVERFLOW_BODY)
            return httpx.Response(200, request=httpx.Request("POST", "http://x"),
                                  json=OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch([
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256},
            {"messages": [{"role": "user", "content": "there"}], "max_tokens": 256},
        ])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 2
        assert all(row["ok"] for row in out)
        assert all(row["host"] == "h2" for row in out)
        assert gw.host_cooling("h1") is True

    async def test_ctx_retry_live_off_skips_the_same_box_retry(self, monkeypatch):
        # The toggle must skip only the same-box fitted retry -- the item
        # still gets requeued for a chance at a different host (or this one
        # again once cooled down) exactly like any other ctx_too_long
        # failure, so a sibling worker is never handed an identical resend
        # to a box that is still cooling down from just failing it.
        monkeypatch.setattr(gw, "COOLDOWN_MIDSTREAM", 0.05)
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        calls = {"n": 0}
        real_settings = gw.get_public_settings

        def off_settings():
            s = real_settings()
            s["ctx_retry_live"] = False
            return s

        monkeypatch.setattr(gw, "get_public_settings", off_settings)

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls["n"] += 1
            return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                  json=CTX_OVERFLOW_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is False
        # 3 attempts (the shared cap), ONE call each -- no fitted same-box
        # retry attempted on any of them, unlike the "on" test's 6 calls.
        assert calls["n"] == 3
        assert gw.host_cooling("h1") is True

    async def test_an_ordinary_client_error_is_never_ctx_retried(self, monkeypatch):
        _wire_routes(monkeypatch, {"m": ["h1"]}, {("h1", "m"): 1}, {"h1": {"m"}})
        calls: list[dict] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            calls.append(json.loads(payload))
            return httpx.Response(400, request=httpx.Request("POST", "http://x"),
                                  json={"error": {"message": "invalid 'temperature'"}})

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert len(calls) == 1   # never retried
        out = _read_output(out_path)
        assert out[0]["ok"] is False
        assert gw.host_cooling("h1") is False   # a client error is not the host's fault
