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
