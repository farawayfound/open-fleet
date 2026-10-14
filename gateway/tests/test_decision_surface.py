"""/v1/systemone -- a Jev-style decision model behind the same keys.

A decision request carries `state` and `questions` instead of `messages`;
nothing is generated, and the reply's usage is spelled input_/output_tokens.
This file pins what the gateway does differently for one, and that it is
the ROUTE that decides, never the body's shape: a single Fleet Pass key
reaches it under its model pin and its context cap (counted over the whole
body, so no extra field escapes), a team key does not reach it at all, no
chat-only shaping (system prompt, sampler defaults, max_tokens, the abuse
knobs, stream_options) is written into it, a chat body carrying a
`questions` field keeps every cap it had, and the usage row is metered in
the gateway's own spelling -- end to end through the proxy with keys minted
the way Fleet Pass mints them.

Run with: gateway/.venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

import app as gw

BODY = {
    "model": "tev1",
    "state": "The agent asked whether coverage is active. The member replied: my card stopped working.",
    "questions": {
        "label": {"type": "choice", "instructions": "Which meaning does the reply have?",
                  "criteria": {"active": "confirms active coverage", "inactive": "not active, or unsure", "UNCLEAR": None}},
        "asked": {"type": "noul", "instructions": "Is the member asking a question of their own?"},
    },
}
DECISION_OK_BODY = {
    "model": "tev1",
    "answers": {
        "label": {"type": "choice", "choice": "inactive", "probabilities": {"active": 0.05, "inactive": 0.93, "UNCLEAR": 0.02},
                  "confidence": 0.91},
        "asked": {"type": "noul", "noul": 0.03},
    },
    "usage": {"input_tokens": 300, "output_tokens": 0},
}
AGENT = {"allowed_models": '["tev1"]', "force_model": "", "system_prompt": "be brief", "rules": "no jokes",
         "param_overrides": '{"max_tokens": 64, "temperature": 0.2}'}


def body(**extra):
    return json.loads(json.dumps(BODY)) | extra


class TestPublicSurface:
    def test_a_single_key_reaches_a_decision_a_team_key_does_not_and_nothing_else_is_new(self):
        assert gw.public_surface_allowed("/v1/systemone", "POST", "single")
        assert not gw.public_surface_allowed("/v1/systemone", "POST", "team")
        assert not gw.public_surface_allowed("/v1/systemone", "POST", "")  # an unknown kind: deny by default
        for kind in ("single", "team", ""):
            assert gw.public_surface_allowed("/v1/chat/completions", "POST", kind)
            assert gw.public_surface_allowed("/v1/models", "GET", kind)
            assert not gw.public_surface_allowed("/v1/systemone", "GET", kind)
            for path in ("/v1/embeddings", "/v1/completions", "/v1/batches", "/v1/responses"):
                assert not gw.public_surface_allowed(path, "POST", kind)
        assert "POST /v1/systemone" in gw._public_surface_message("single")
        assert "systemone" not in gw._public_surface_message("team")


class TestTheRouteDecides:
    def test_a_decision_is_estimated_from_its_whole_body_but_the_model_id(self):
        assert gw._prompt_part(BODY, decision=True) == {k: v for k, v in BODY.items() if k != "model"}
        assert gw.estimate_prompt_tokens(BODY, decision=True) >= len(BODY["state"]) / 3.2
        assert gw.estimate_prompt_tokens({"model": "tev1"}, decision=True) <= 3  # nothing to read
        # no field a client adds escapes the count
        padded = gw.estimate_prompt_tokens(body(prompt="", notes="x" * 32000), decision=True)
        assert padded > gw.estimate_prompt_tokens(BODY, decision=True) + 9000

    def test_a_chat_body_with_a_questions_field_is_still_a_chat_body(self):
        chat = {"model": "m", "prompt": "hi", "questions": {}, "max_tokens": 100000}
        assert gw._prompt_part(chat) == "hi"
        capped = gw.apply_agent(AGENT | {"allowed_models": "[]"}, dict(chat))
        assert capped["max_tokens"] == 64 and capped["temperature"] == 0.2
        assert "max_tokens" in gw.apply_ctx_limit(dict(chat), 8192) and gw.apply_ctx_limit(dict(chat), 8192)["max_tokens"] <= 8192

    def test_the_context_cap_refuses_an_oversized_decision_and_writes_no_completion_budget(self):
        small = gw.apply_ctx_limit(body(), 8192, decision=True)
        assert small == BODY and "max_tokens" not in small
        with pytest.raises(HTTPException) as exc:
            gw.apply_ctx_limit(body(state="x" * 40000), 8192, decision=True)
        assert exc.value.status_code == 413 and exc.value.detail["error"]["type"] == "context_limit"
        assert exc.value.detail["error"]["message"].startswith("state is about ")
        for spoof in ({"prompt": ""}, {"messages": None}):  # the fields that used to turn a decision into "chat"
            with pytest.raises(HTTPException):
                gw.apply_ctx_limit(body(state="x" * 40000, **spoof), 8192, decision=True)


class TestAgentProfile:
    def test_the_allow_list_binds_and_nothing_chat_shaped_is_written_in(self):
        assert gw.apply_agent(AGENT, body(), decision=True) == BODY
        with pytest.raises(HTTPException) as exc:
            gw.apply_agent(AGENT, body(model="qwen3.8-27b"), decision=True)
        assert exc.value.status_code == 403

    def test_a_pinned_model_still_pins_a_decision(self):
        pinned = gw.apply_agent(AGENT | {"force_model": "tev1"}, body(model="whatever"), decision=True)
        assert pinned["model"] == "tev1" and "messages" not in pinned and "temperature" not in pinned

    def test_a_chat_body_is_shaped_exactly_as_before(self):
        chat = gw.apply_agent(AGENT, {"model": "tev1", "messages": [{"role": "user", "content": "hi"}]})
        assert chat["messages"][0]["role"] == "system" and chat["max_tokens"] == 64 and chat["temperature"] == 0.2


class TestUsage:
    def test_decision_usage_is_metered_in_the_gateways_spelling(self):
        assert gw._normalise_usage({"input_tokens": 300, "output_tokens": 0}) == {
            "input_tokens": 300, "output_tokens": 0, "prompt_tokens": 300, "completion_tokens": 0, "total_tokens": 300}
        chat = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        assert gw._normalise_usage(chat) is chat
        assert gw._normalise_usage(None) is None and gw._normalise_usage("x") is None
        odd = {"input_tokens": "many"}
        assert gw._normalise_usage(odd) is odd  # unreadable counts are left to record_usage's own int() guard


class TestCatalogue:
    def test_the_admin_can_re_save_the_seeded_decision_row_unchanged(self, client, admin_headers):
        items = client.get("/admin/api/public/models", headers=admin_headers).json()["items"]
        row = next(i for i in items if i["public_id"] == "tev1")
        assert row["enabled"] and not row["allow_primary"] and not row["allow_worker"]  # a single-key-only model
        r = client.put("/admin/api/public/models/tev1", headers=admin_headers, json=row)
        assert r.status_code == 200, r.text


@pytest.fixture(autouse=True)
def _isolate_routing():
    gw._host_cooldown.clear()
    gw._stall_streak.clear()
    gw._inflight.clear()
    yield
    gw._host_cooldown.clear()
    gw._stall_streak.clear()
    gw._inflight.clear()


def _wire(monkeypatch, host="p1", model="tev1"):
    """The /v1 proxy's routing, pointed at one fixed peer (the house pattern of test_proxy_failover.py)."""
    async def _resolve(requested, **kw):
        return [(host, model)]

    async def _pik(cand):
        return "k"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers", lambda: [{"name": host, "url": "http://%s:8080" % host, "token": "t"}])
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


def _fleet_pass_key(kind="single", ctx=4096):
    """A key minted the way Fleet Pass mints one: a public_keys row issued through issue_public_key(), so a single key
    has its agent row (model pin, allow-list, max_tokens override, ctx cap) and a team key its roster."""
    models = {"model": "tev1"} if kind == "single" else {"primary": "tev1", "workers": ["qwen3.5-4b"]}
    row_id = gw.db_exec(
        "INSERT INTO public_keys(created_at,email,domain,company,source,kind,models,ctx,status,ip,user_agent,note) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (gw.now(), "judge@example.com", "example.com", "", "manual", kind, json.dumps(models), ctx, "pending", "", "",
         "decision test"),
    )
    row = gw.db_query("SELECT * FROM public_keys WHERE id=?", (row_id,))[0]
    raw = gw.issue_public_key(row, decided_by="test")
    gw._public_key_cache.clear()
    gw._public_key_row_cache.clear()
    return raw, int(gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (row_id,))[0]["key_id"])


def _peer(monkeypatch, forwarded: list):
    async def fake_send(self, request, stream=False, **kw):
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, request=request, json=DECISION_OK_BODY)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)


def _rows(key_id: int) -> list[dict]:
    return gw.db_query("SELECT endpoint, model, status, prompt_tokens, completion_tokens, total_tokens FROM usage "
                       "WHERE key_id=? ORDER BY id DESC", (key_id,))


HEADERS = {"X-Fleet-No-Fallback": "1"}


class TestThroughTheProxy:
    def test_a_single_key_gets_a_decision_routed_untouched_and_metered(self, client, monkeypatch):
        _wire(monkeypatch)
        forwarded: list[dict] = []
        _peer(monkeypatch, forwarded)
        raw, key_id = _fleet_pass_key()
        assert gw.get_agent(key_id)["ctx_limit"] == 4096  # the profile is there, and it is applied below
        r = client.post("/v1/systemone", headers={"Authorization": "Bearer " + raw, **HEADERS}, json=BODY)
        assert r.status_code == 200, r.text
        assert r.json()["answers"]["label"]["choice"] == "inactive"
        assert forwarded == [BODY]  # no n=1, no max_tokens, no system prompt: the body the client wrote
        row = _rows(key_id)[0]
        assert row["endpoint"] == "/v1/systemone" and row["model"] == "tev1" and row["status"] == 200
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (300, 0, 300)

    def test_a_single_keys_context_cap_binds_and_counts_the_whole_body(self, client, monkeypatch):
        _wire(monkeypatch)
        forwarded: list[dict] = []
        _peer(monkeypatch, forwarded)
        raw, key_id = _fleet_pass_key(ctx=4096)
        big = "x" * 40000
        for spoof in ({}, {"prompt": ""}, {"messages": None}):
            r = client.post("/v1/systemone", headers={"Authorization": "Bearer " + raw, **HEADERS}, json=body(state=big, **spoof))
            assert r.status_code == 413, (spoof, r.text)
            assert r.json()["error"]["type"] == "context_limit"
        assert forwarded == [] and {row["status"] for row in _rows(key_id)} == {413}

    def test_a_team_key_is_refused_a_decision_before_anything_is_forwarded(self, client, monkeypatch):
        _wire(monkeypatch)
        forwarded: list[dict] = []
        _peer(monkeypatch, forwarded)
        raw, key_id = _fleet_pass_key(kind="team")
        r = client.post("/v1/systemone", headers={"Authorization": "Bearer " + raw, **HEADERS}, json=BODY)
        assert r.status_code == 403 and r.json()["error"]["message"] == "this key is limited to POST /v1/chat/completions and GET /v1/models"
        assert forwarded == [] and _rows(key_id)[0]["status"] == 403

    def test_a_streamed_decision_is_answered_whole_and_metered(self, client, monkeypatch):
        _wire(monkeypatch)
        forwarded: list[dict] = []
        _peer(monkeypatch, forwarded)
        raw, key_id = _fleet_pass_key()
        asked = body(stream=True, stream_options={"include_usage": True}, keep_alive=-1, runner="whatever")
        r = client.post("/v1/systemone", headers={"Authorization": "Bearer " + raw, **HEADERS}, json=asked)
        assert r.status_code == 200, r.text
        assert forwarded == [BODY]  # neither the stream flag nor a residency knob reaches the engine from a public key
        assert _rows(key_id)[0]["total_tokens"] == 300

    def test_a_single_key_whose_agent_profile_is_gone_is_refused_a_decision(self, client, monkeypatch):
        _wire(monkeypatch)
        forwarded: list[dict] = []
        _peer(monkeypatch, forwarded)
        raw, key_id = _fleet_pass_key()
        gw.db_exec("UPDATE agents SET enabled=0 WHERE key_id=?", (key_id,))
        r = client.post("/v1/systemone", headers={"Authorization": "Bearer " + raw, **HEADERS}, json=BODY)
        assert r.status_code == 403 and "agent profile" in r.json()["error"]["message"]
        assert forwarded == [] and _rows(key_id)[0]["status"] == 403

    def test_public_keys_are_still_refused_the_rest_of_v1_and_the_native_api(self, client, monkeypatch):
        raw, _key_id = _fleet_pass_key()
        r = client.post("/v1/embeddings", headers={"Authorization": "Bearer " + raw}, json={"model": "tev1", "input": "x"})
        assert r.status_code == 403 and "POST /v1/systemone" in r.json()["error"]["message"]
        r = client.post("/api/chat", headers={"Authorization": "Bearer " + raw},
                        json={"model": "tev1", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403 and "POST /v1/systemone" in r.json()["error"]["message"]
        team, _ = _fleet_pass_key(kind="team")
        r = client.post("/api/chat", headers={"Authorization": "Bearer " + team},
                        json={"model": "tev1", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403 and "systemone" not in r.json()["error"]["message"]
