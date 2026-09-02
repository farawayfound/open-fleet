"""The live demo behind example.org/fleet/tour: an unkeyed, per-address
budgeted chat box that answers from one configured model on the boxes the
owner allows.

Run with: gateway/.venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

import app as gw


DEMO_HEADERS = {"X-Intake-Token": "intake", "X-Forwarded-Client-IP": "203.0.113.42"}


@pytest.fixture(autouse=True)
def _isolate_routing():
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()


@pytest.fixture
def demo_fleet(monkeypatch, client):
    """A fleet where the demo model is served by four boxes: the preferred
    laptop (cold), a small always-on box (resident), and two big boxes the
    policy excludes (one of them resident, so it would win on the scorer's
    own terms). Every candidate has a context ceiling of its own."""
    fid = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
    zfid = "qwen3.8-9B"
    cands = {fid: ["mac-desktop-1", "apu-box-1", "mac-laptop-1"], zfid: ["gpu-laptop-1"]}
    running = {"mac-desktop-1": {fid}, "mac-laptop-1": {fid}, "apu-box-1": set(), "gpu-laptop-1": set()}
    ctx = {("mac-desktop-1", fid): 27648, ("apu-box-1", fid): 262144, ("mac-laptop-1", fid): 126976,
           ("gpu-laptop-1", zfid): 132768}
    cap = {(c, m): 1 for m, cs in cands.items() for c in cs}
    gw._routes_cache.update(
        t=time.time(), map={m: cs[0] for m, cs in cands.items()}, cands=cands, cap=cap,
        running=running, ctx=ctx, engine={"mac-desktop-1": "ollama", "gpu-laptop-1": "llama-swap"},
        reachable={"mac-desktop-1", "apu-box-1", "mac-laptop-1", "gpu-laptop-1"}, meta={})
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": n, "url": "http://%s:8080" % n, "token": "t"}
        for n in ("mac-desktop-1", "apu-box-1", "mac-laptop-1", "gpu-laptop-1")])

    async def _routes(force: bool = False):
        return gw._routes_cache["map"]

    async def _pik(cand):
        return "peer-key"

    monkeypatch.setattr(gw, "model_routes", _routes)
    monkeypatch.setattr(gw, "peer_inference_key", _pik)
    gw._known_ctx_cache.update(t=0.0, map={}, written=None)
    gw._public_catalogue_cache.update(t=0.0)
    return gw._routes_cache


def _sse_chunks(*texts: str, usage: dict | None = None) -> list[bytes]:
    out = []
    for t in texts:
        out.append(("data: " + json.dumps({"choices": [{"delta": {"content": t}}]}) + "\n\n").encode())
    if usage is not None:
        out.append(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
    out.append(b"data: [DONE]\n\n")
    return out


def _not_chat(request) -> bool:
    """The hub's own background loops (peer status polls, the preload tick)
    go through the same httpx.AsyncClient.send and would otherwise land in a
    test's call list at random; they are refused as if every peer were off."""
    return request.url.path != "/v1/chat/completions"


def _fake_upstream(monkeypatch, answers: dict, calls: list):
    """httpx.AsyncClient.send double keyed by the peer's host name. An entry
    may be a list of SSE byte chunks (a 200 stream), an int (that status with
    a JSON error body), or an exception instance (raised on send)."""
    async def fake_send(self, request, stream=False, **kw):
        if _not_chat(request):
            raise httpx.ConnectError("not a demo request", request=request)
        host = request.url.host
        calls.append((host, json.loads(request.content or b"{}")))
        spec = answers.get(host)
        if isinstance(spec, Exception):
            raise spec
        if isinstance(spec, int):
            return httpx.Response(spec, request=request, json={"error": {"message": "boom"}})
        resp = httpx.Response(200, request=request, headers={"content-type": "text/event-stream"})

        async def it():
            for c in spec:
                yield c
        resp.aiter_bytes = it
        return resp
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)


def _events(text: str) -> list[dict]:
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


# ---------------------------------------------------------------------------
# the host policy, as a pure function
# ---------------------------------------------------------------------------

class TestHostPolicy:
    def test_excluded_hosts_are_dropped_and_preferred_move_first(self):
        settings = {"demo_prefer_hosts": ["gpu-laptop-1"], "demo_exclude_hosts": ["apu-box-1", "mac-laptop-1"]}
        targets = [("mac-laptop-1", "a"), ("mac-desktop-1", "a"), ("gpu-laptop-1", "b"), ("apu-box-1", "a")]
        assert gw.demo_host_policy(targets, settings) == [("gpu-laptop-1", "b"), ("mac-desktop-1", "a")]

    def test_the_scorer_order_survives_within_a_band(self):
        settings = {"demo_prefer_hosts": [], "demo_exclude_hosts": []}
        targets = [("mac-desktop-1", "a"), ("cpu-box-1", "a"), ("mini-pc-1", "a")]
        assert gw.demo_host_policy(targets, settings) == targets

    def test_the_hub_itself_is_matched_by_its_own_name(self):
        settings = {"demo_prefer_hosts": [], "demo_exclude_hosts": [gw.HOST_NAME.lower()]}
        assert gw.demo_host_policy([("", "a"), ("p", "a")], settings) == [("p", "a")]

    def test_a_saturated_preferred_host_does_not_beat_a_free_one(self):
        # Requirement 1: preference is a tiebreak among equally-free (or
        # equally-saturated) boxes now, never an override of availability --
        # a busy gpu-laptop-1 must not jump ahead of an idle mac-desktop-1 just
        # because the owner listed it first.
        settings = {"demo_prefer_hosts": ["gpu-laptop-1"], "demo_exclude_hosts": []}
        gw._routes_cache["cap"] = {("gpu-laptop-1", "b"): 1, ("mac-desktop-1", "a"): 1}
        gw._inflight["gpu-laptop-1"] = 1  # busy >= slots(1): saturated
        targets = [("mac-desktop-1", "a"), ("gpu-laptop-1", "b")]
        assert gw.demo_host_policy(targets, settings) == [("mac-desktop-1", "a"), ("gpu-laptop-1", "b")]

    def test_preference_still_breaks_a_tie_among_equally_free_hosts(self):
        settings = {"demo_prefer_hosts": ["gpu-laptop-1"], "demo_exclude_hosts": []}
        gw._routes_cache["cap"] = {("gpu-laptop-1", "b"): 1, ("mac-desktop-1", "a"): 1}
        # Neither saturated: the free-vs-saturated key is 0/0 for both, so
        # preference still decides -- the feature is narrowed, not removed.
        targets = [("mac-desktop-1", "a"), ("gpu-laptop-1", "b")]
        assert gw.demo_host_policy(targets, settings) == [("gpu-laptop-1", "b"), ("mac-desktop-1", "a")]

    def test_host_lists_accept_a_comma_string(self):
        assert gw.clean_host_list(" gpu-laptop-1, mac-laptop-1 ,,") == ["gpu-laptop-1", "mac-laptop-1"]
        assert gw.clean_host_list(["a", 3, "A", ""]) == ["a"]
        assert gw.clean_host_list(None) == []

    def test_settings_round_trip_the_lists(self, client):
        gw.set_public_settings({"demo_prefer_hosts": "mac-desktop-1, mini-pc-1", "demo_exclude_hosts": []})
        s = gw.get_public_settings()
        assert s["demo_prefer_hosts"] == ["mac-desktop-1", "mini-pc-1"]
        assert s["demo_exclude_hosts"] == []
        assert s["demo_ip_rph"] == 5 and s["demo_max_tokens"] == 512


class TestThinkStripper:
    def test_a_whole_block_in_one_chunk(self):
        st = gw._ThinkStripper()
        assert st.feed("<think>plan plan</think>\n\nHello") == "Hello"

    def test_tags_split_across_chunks(self):
        st = gw._ThinkStripper()
        out = st.feed("<thi") + st.feed("nk>secret</th") + st.feed("ink>Answer") + st.flush()
        assert out == "Answer"

    def test_a_lone_angle_bracket_is_not_swallowed(self):
        st = gw._ThinkStripper()
        out = st.feed("a < b") + st.feed(" and c") + st.flush()
        assert out == "a < b and c"

    def test_unclosed_reasoning_never_leaks(self):
        st = gw._ThinkStripper()
        assert st.feed("<think>still thinking") == ""
        assert st.flush() == ""


# ---------------------------------------------------------------------------
# GET /public/api/demo
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_names_only_allowed_boxes_as_aliases(self, client, demo_fleet):
        r = client.get("/public/api/demo", headers=DEMO_HEADERS)
        assert r.status_code == 200
        d = r.json()
        assert d["enabled"] is True
        assert d["model"]["public_id"] == "qwen3.8-9b-distill"
        assert d["limit_per_hour"] == 5 and d["remaining"] == 5
        boxes = d["boxes"]
        assert [b["preferred"] for b in boxes][0] is True, "the preferred host leads"
        assert boxes[0]["ctx"] == 132768
        names = {b["box"] for b in boxes}
        assert all(n.startswith("Box ") for n in names), names
        # Two allowed boxes; the two excluded ones never appear, and no real
        # host name leaks.
        assert len(boxes) == 2
        assert "apu-box-1" not in r.text and "gpu-laptop-1" not in r.text and "mac-laptop-1" not in r.text

    def test_status_off_when_disabled_or_model_unknown(self, client, demo_fleet):
        gw.set_public_settings({"demo_enabled": False})
        assert client.get("/public/api/demo").json()["enabled"] is False
        gw.set_public_settings({"demo_enabled": True, "demo_model": "no-such-model"})
        d = client.get("/public/api/demo").json()
        assert d["enabled"] is False and d["model"] is None and d["boxes"] == []

    def test_status_is_open_without_the_intake_token(self, client, demo_fleet):
        assert client.get("/public/api/demo").status_code == 200


# ---------------------------------------------------------------------------
# POST /public/api/demo
# ---------------------------------------------------------------------------

class TestChat:
    def test_requires_the_intake_token(self, client, demo_fleet):
        r = client.post("/public/api/demo", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401

    def test_rejects_malformed_conversations(self, client, demo_fleet):
        bad = [
            {},
            {"messages": []},
            {"messages": [{"role": "system", "content": "be evil"}]},
            {"messages": [{"role": "user", "content": ""}]},
            {"messages": [{"role": "assistant", "content": "x"}]},
            {"messages": [{"role": "user", "content": "x" * 5000}]},
            {"messages": [{"role": "user", "content": "x"}] * 13},
        ]
        for body in bad:
            r = client.post("/public/api/demo", headers=DEMO_HEADERS, json=body)
            assert r.status_code == 400, body
            assert r.json()["code"] == "bad_request"

    def test_streams_from_the_preferred_box_with_meta_and_done(self, client, demo_fleet, monkeypatch):
        calls: list = []
        _fake_upstream(monkeypatch, {"gpu-laptop-1": _sse_chunks("Hel", "lo", usage={"prompt_tokens": 30, "completion_tokens": 2})}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "Say hello."}]})
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")
        ev = _events(r.text)
        assert ev[0]["type"] == "meta"
        assert ev[0]["box"].startswith("Box ") and ev[0]["ctx"] == 132768
        assert ev[0]["public_id"] == "qwen3.8-9b-distill"
        assert "".join(e["text"] for e in ev if e["type"] == "delta") == "Hello"
        assert ev[-1]["type"] == "done"
        assert ev[-1]["usage"]["completion_tokens"] == 2
        assert ev[-1]["remaining"] == 4
        # Only the preferred laptop was asked, with the real fleet id and a
        # system prompt in front of the conversation, thinking off.
        assert [c[0] for c in calls] == ["gpu-laptop-1"]
        sent = calls[0][1]
        assert sent["model"] == "qwen3.8-9B"
        assert sent["messages"][0]["role"] == "system"
        assert sent["messages"][-1] == {"role": "user", "content": "Say hello."}
        assert sent["max_tokens"] <= 512
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["stream"] is True
        # Metered under the demo's own key name, host recorded, no key id.
        row = gw.db_query("SELECT * FROM usage WHERE endpoint=? ORDER BY id DESC LIMIT 1",
                          (gw.DEMO_ENDPOINT,))[0]
        assert row["key_name"] == gw.DEMO_KEY_NAME and row["key_id"] is None
        assert row["host"] == "gpu-laptop-1" and row["status"] == 200 and row["stream"] == 1
        assert row["completion_tokens"] == 2
        assert not any(v.get("kind") == "inference" for v in gw._active.values())
        assert gw._inflight.get("gpu-laptop-1", 0) == 0

    def test_fails_over_past_a_dead_preferred_box_and_never_uses_an_excluded_one(
            self, client, demo_fleet, monkeypatch):
        calls: list = []
        _fake_upstream(monkeypatch, {
            "gpu-laptop-1": httpx.ConnectError("refused"),
            "mac-desktop-1": _sse_chunks("ok"),
            "apu-box-1": _sse_chunks("should not be asked"),
            "mac-laptop-1": _sse_chunks("should not be asked"),
        }, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert [c[0] for c in calls] == ["gpu-laptop-1", "mac-desktop-1"]
        assert ev[0]["type"] == "meta" and ev[0]["ctx"] == 27648
        assert calls[1][1]["model"] == "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
        assert calls[1][1]["think"] is False, "the Ollama box gets its own thinking switch"
        assert gw.host_cooling("gpu-laptop-1")
        row = gw.db_query("SELECT host, status FROM usage WHERE endpoint=? ORDER BY id DESC LIMIT 1",
                          (gw.DEMO_ENDPOINT,))[0]
        assert row["host"] == "mac-desktop-1" and row["status"] == 200

    def test_a_5xx_moves_on_a_4xx_does_not(self, client, demo_fleet, monkeypatch):
        calls: list = []
        _fake_upstream(monkeypatch, {"gpu-laptop-1": 503, "mac-desktop-1": _sse_chunks("fine")}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        assert [c[0] for c in calls] == ["gpu-laptop-1", "mac-desktop-1"]
        assert "".join(e["text"] for e in _events(r.text) if e["type"] == "delta") == "fine"

        calls.clear()
        gw._host_cooldown.clear()
        _fake_upstream(monkeypatch, {"gpu-laptop-1": 400, "mac-desktop-1": _sse_chunks("never")}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert [c[0] for c in calls] == ["gpu-laptop-1"]
        assert ev[-1]["type"] == "error" and ev[-1]["code"] == "upstream"

    def test_an_engine_s_error_text_never_reaches_the_visitor(self, client, demo_fleet, monkeypatch):
        """llama-server and Ollama put model paths in their error bodies;
        the public frame carries the box alias and the status, nothing else."""
        secret = "/var/lib/llmstack/models/empero-ai__Qwen3.8-9B-Distill-GGUF/x.gguf"

        async def fake_send(self, request, stream=False, **kw):
            if _not_chat(request):
                raise httpx.ConnectError("not a demo request", request=request)
            if request.url.host == "gpu-laptop-1":
                return httpx.Response(400, request=request,
                                      json={"error": {"message": "failed to load " + secret}})
            return httpx.Response(200, request=request, json={})
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert ev[-1]["type"] == "error" and ev[-1]["code"] == "upstream"
        assert secret not in r.text and "gpu-laptop-1" not in r.text and ".gguf" not in r.text
        assert ev[-1]["message"].startswith("Box ")

        # The same for an error object arriving mid-stream.
        async def mid(self, request, stream=False, **kw):
            if _not_chat(request):
                raise httpx.ConnectError("not a demo request", request=request)
            resp = httpx.Response(200, request=request, headers={"content-type": "text/event-stream"})

            async def it():
                yield _sse_chunks("Hel")[0]
                yield ("data: " + json.dumps({"error": {"message": "OOM at " + secret}}) + "\n\n").encode()
            resp.aiter_bytes = it
            return resp
        monkeypatch.setattr(httpx.AsyncClient, "send", mid)
        gw._host_cooldown.clear()
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert [e["type"] for e in ev] == ["meta", "delta", "error"]
        assert secret not in r.text and "gpu-laptop-1" not in r.text

    def test_the_owner_s_token_ceiling_beats_the_completion_floor(self, client, demo_fleet, monkeypatch):
        gw.set_public_settings({"demo_max_tokens": 128})
        calls: list = []
        _fake_upstream(monkeypatch, {"gpu-laptop-1": _sse_chunks("ok")}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert calls[0][1]["max_tokens"] == 128

    def test_every_box_down_is_an_error_frame_and_a_502_row(self, client, demo_fleet, monkeypatch):
        calls: list = []
        _fake_upstream(monkeypatch, {"gpu-laptop-1": httpx.ConnectError("x"),
                                     "mac-desktop-1": httpx.ConnectError("y")}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert ev == [ev[0]] and ev[0]["type"] == "error" and ev[0]["code"] == "demo_offline"
        row = gw.db_query("SELECT status FROM usage WHERE endpoint=? ORDER BY id DESC LIMIT 1",
                          (gw.DEMO_ENDPOINT,))[0]
        assert row["status"] == 502

    def test_reasoning_is_stripped_and_a_broken_stream_ends_cleanly(self, client, demo_fleet, monkeypatch):
        calls: list = []

        async def broken():
            yield _sse_chunks("<think>hmm</think>Real")[0]
            raise httpx.ReadError("dropped")

        async def fake_send(self, request, stream=False, **kw):
            if _not_chat(request):
                raise httpx.ConnectError("not a demo request", request=request)
            calls.append(request.url.host)
            resp = httpx.Response(200, request=request, headers={"content-type": "text/event-stream"})
            resp.aiter_bytes = broken
            return resp
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        ev = _events(r.text)
        assert "".join(e["text"] for e in ev if e["type"] == "delta") == "Real"
        assert ev[-1]["type"] == "error"
        assert calls == ["gpu-laptop-1"], "a box that started answering is never swapped"
        row = gw.db_query("SELECT status FROM usage WHERE endpoint=? ORDER BY id DESC LIMIT 1",
                          (gw.DEMO_ENDPOINT,))[0]
        assert row["status"] == 502

    def test_per_address_hourly_budget(self, client, demo_fleet, monkeypatch):
        gw.set_public_settings({"demo_ip_rph": 2})
        _fake_upstream(monkeypatch, {"gpu-laptop-1": _sse_chunks("a")}, [])
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert client.post("/public/api/demo", headers=DEMO_HEADERS, json=body).status_code == 200
        assert client.post("/public/api/demo", headers=DEMO_HEADERS, json=body).status_code == 200
        r = client.post("/public/api/demo", headers=DEMO_HEADERS, json=body)
        assert r.status_code == 429 and r.json()["code"] == "ip_limit"
        # Another address is unaffected, and the status endpoint agrees.
        other = dict(DEMO_HEADERS, **{"X-Forwarded-Client-IP": "198.51.100.7"})
        assert client.post("/public/api/demo", headers=other, json=body).status_code == 200
        assert client.get("/public/api/demo", headers=DEMO_HEADERS).json()["remaining"] == 0
        assert client.get("/public/api/demo", headers=other).json()["remaining"] == 1

    def test_disabled_is_a_503_before_anything_is_counted(self, client, demo_fleet):
        gw.set_public_settings({"demo_enabled": False})
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503 and r.json()["code"] == "demo_disabled"
        assert gw.demo_ip_used("203.0.113.42") == 0

    def test_folded_json_form(self, client, demo_fleet, monkeypatch):
        _fake_upstream(monkeypatch, {"gpu-laptop-1": _sse_chunks("one ", "two", usage={"completion_tokens": 2})}, [])
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["text"] == "one two" and d["box"].startswith("Box ") and d["remaining"] == 4

    def test_the_window_is_the_answering_box_s_own(self, client, demo_fleet, monkeypatch):
        """A conversation that fits the preferred laptop's 132k window but
        not a 512-token completion on a tiny one is fitted, not refused --
        and the meta frame says which window it got."""
        calls: list = []
        gw._routes_cache["ctx"][("gpu-laptop-1", "qwen3.8-9B")] = 1024
        _fake_upstream(monkeypatch, {"gpu-laptop-1": _sse_chunks("short")}, calls)
        r = client.post("/public/api/demo", headers=DEMO_HEADERS,
                        json={"messages": [{"role": "user", "content": "word " * 120}]})
        ev = _events(r.text)
        assert ev[0]["type"] == "meta" and ev[0]["ctx"] == 1024
        assert calls[0][1]["max_tokens"] == gw.PUBLIC_MIN_COMPLETION

    def test_admin_report_names_real_hosts_and_the_shut_out_big_boxes(self, client, demo_fleet, admin_headers):
        r = client.get("/admin/api/public/demo", headers=admin_headers)
        assert r.status_code == 200
        d = r.json()
        assert [c["host"] for c in d["candidates"]] == ["gpu-laptop-1", "mac-desktop-1"]
        assert d["excluded_serving"] == ["apu-box-1", "mac-laptop-1"]
        assert d["model_known"] is True and d["requests_day"] == 0

    def test_seed_claims_every_spelling_of_the_demo_model(self):
        seed = {r["public_id"]: r for r in gw.load_public_models_seed()}
        fids = seed["qwen3.8-9b-distill"]["fleet_ids"]
        assert "qwen3.8-9B" in fids and "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M" in fids
