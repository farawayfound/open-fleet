"""The 2026-08-25 usability pass: a Fleet Pass/team reply must complete the
caller's actual prompt. The trigger incident: a team key issued for 168960
tokens of context, served by a 126976 box while the big box was offline,
answered a Cline turn with the ctx notice, one sentence, and nothing else --
and every retry reproduced it. The notice is secondary; the answer is the
product. These tests pin every repair from that pass.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# the notice must never BE the reply, or corrupt one
# ---------------------------------------------------------------------------

class TestNoticeStaysSecondary:
    def test_an_empty_answer_is_not_replaced_by_the_notice(self):
        # A budget-starved thinking model returns "": prepending the notice
        # made the notice masquerade as the answer (the incident's retry).
        obj = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        gw.prepend_notice(obj, "[Fleet notice: reduced]\n\n")
        assert obj["choices"][0]["message"]["content"] == ""

    def test_a_whitespace_answer_counts_as_empty(self):
        obj = {"choices": [{"message": {"role": "assistant", "content": " \n"}}]}
        gw.prepend_notice(obj, "[Fleet notice: reduced]\n\n")
        assert obj["choices"][0]["message"]["content"] == " \n"

    def test_a_real_answer_still_gets_the_notice(self):
        obj = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        gw.prepend_notice(obj, "[Fleet notice: reduced]\n\n")
        assert obj["choices"][0]["message"]["content"].startswith("[Fleet notice:")

    def test_structured_output_is_detected(self):
        assert gw._wants_structured({"response_format": {"type": "json_object"}})
        assert gw._wants_structured({"response_format": {"type": "json_schema"}})
        assert not gw._wants_structured({"response_format": {"type": "text"}})
        assert not gw._wants_structured({})
        assert not gw._wants_structured(None)


class TestNoticeHistoryStripping:
    NOTICE = ("[Fleet notice: your key is set up for 168960 tokens of context, "
              "but the machine that can serve that much was unavailable, so "
              "this reply was answered with 126976. Your key is unchanged.]\n\n")

    def test_a_resent_banner_is_removed_from_assistant_history(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": self.NOTICE + "the answer"},
                {"role": "user", "content": "continue"}]
        assert gw.strip_fleet_notices(msgs) is True
        assert msgs[1]["content"] == "the answer"

    def test_both_banners_of_one_turn_are_removed(self):
        # A substitution and a reduction disclosed together, back to back.
        double = ("[Fleet notice: a was not available, so b answered.]\n\n"
                  + self.NOTICE + "the answer")
        msgs = [{"role": "assistant", "content": double}]
        gw.strip_fleet_notices(msgs)
        assert msgs[0]["content"] == "the answer"

    def test_user_text_quoting_a_notice_is_untouched(self):
        # Only assistant turns carry gateway-injected banners; a user PASTING
        # one into their own message is their content.
        msgs = [{"role": "user", "content": self.NOTICE + "what does this mean?"}]
        assert gw.strip_fleet_notices(msgs) is False
        assert msgs[0]["content"].startswith("[Fleet notice:")

    def test_clean_history_is_left_byte_identical(self):
        msgs = [{"role": "assistant", "content": "plain answer"}]
        assert gw.strip_fleet_notices(msgs) is False
        assert msgs[0]["content"] == "plain answer"

    def test_non_string_content_is_left_alone(self):
        msgs = [{"role": "assistant", "content": None},
                {"role": "assistant",
                 "content": [{"type": "text", "text": "part"}]}]
        assert gw.strip_fleet_notices(msgs) is False

    def test_double_application_is_a_no_op(self):
        # Both openai_proxy and team_orchestrate strip; the second pass over
        # already-clean history must change nothing and say so.
        msgs = [{"role": "assistant", "content": self.NOTICE + "the answer"}]
        assert gw.strip_fleet_notices(msgs) is True
        assert gw.strip_fleet_notices(msgs) is False
        assert msgs[0]["content"] == "the answer"


# ---------------------------------------------------------------------------
# fleet_chat: honest statuses, honest ordering
# ---------------------------------------------------------------------------

class TestFleetChatContextRejection:
    async def test_all_candidates_too_small_is_a_413_not_a_502(self, monkeypatch):
        # Every box declined the prompt on length and nothing broke: the old
        # 502 "no reachable host" read as an outage and invited a retry that
        # could never succeed. The 413 names the numbers.
        async def _targets(model, **kw):
            return [("boxa", "m1"), ("boxb", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("boxa", "m1"): 2048, ("boxb", "m1"): 2048})

        called = []

        async def _post(cand, payload, *a, **kw):  # pragma: no cover
            called.append(cand)
            raise AssertionError("no box should have been asked")
        monkeypatch.setattr(gw, "_post_chat", _post)

        big = "x" * 30000
        status, body, host, granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": big}]},
            "/v1/chat/completions", 32768)
        assert status == 413
        assert body["error"]["type"] == "context_limit"
        assert body["error"]["limit"] == 2048
        assert called == []
        rows = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC", (1,))
        assert rows and rows[0]["status"] == 413, "the 413 must be metered"

    async def test_a_broken_box_still_reads_as_a_502(self, monkeypatch):
        # One candidate rejected on length, the other actually failed: that
        # is a fleet problem, not a prompt problem.
        async def _targets(model, **kw):
            return [("small", "m1"), ("broken", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("small", "m1"): 2048,
                                     ("broken", "m1"): 131072})

        async def _post(cand, payload, *a, **kw):
            raise httpx.ConnectError("down")
        monkeypatch.setattr(gw, "_post_chat", _post)

        big = "x" * 30000
        status, body, _host, _granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": big}]},
            "/v1/chat/completions", 32768)
        assert status == 502


class TestCtxRank:
    def test_proven_sufficient_beats_unknown_beats_too_small(self):
        gw._routes_cache.update(ctx={("big", "m"): 131072, ("small", "m"): 8192})
        ranked = sorted([("small", "m"), ("mystery", "m"), ("big", "m")],
                        key=lambda t: gw._ctx_rank(t, 32768))
        assert [h for h, _m in ranked] == ["big", "mystery", "small"]

    async def test_fleet_chat_prefers_proven_over_unknown(self, monkeypatch):
        # The scorer liked the unverified box; the box PROVEN to hold the
        # whole window must still be asked first -- a wrong guess on the
        # unknown one is a silent server-side truncation.
        async def _targets(model, **kw):
            return [("mystery", "m1"), ("proven", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("proven", "m1"): 131072})
        seen = []

        async def _post(cand, payload, *a, **kw):
            seen.append(cand)
            return httpx.Response(200, json={"choices": []})
        monkeypatch.setattr(gw, "_post_chat", _post)

        await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions", 32768)
        assert seen == ["proven"]


# ---------------------------------------------------------------------------
# the team loop completes the caller's turn
# ---------------------------------------------------------------------------

async def _issue_team(client, intake_headers, email="usability@nasa.gov",
                      ctx=16384):
    r = client.post(
        "/public/api/request", headers=intake_headers,
        json={"email": email, "kind": "team", "primary": "gemma4-31b-qat",
              "workers": ["gemma4-26b-a4b"], "ctx": ctx, "accept_terms": True})
    assert r.status_code == 200, r.text
    row = gw.db_query("SELECT * FROM api_keys WHERE name=?",
                      ("fleet-pass:" + email,))[0]
    return row, gw.get_team(row["id"])


def _ok_round(content="done", finish="stop", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestTeamLoopCompletesTheTurn:
    async def test_a_length_cut_final_turn_is_retried_with_more_room(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "len@nasa.gov")
        budgets: list[int] = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            budgets.append(int(body.get("max_tokens") or 0))
            if len(budgets) == 1:
                return 200, _ok_round("I'll triage this ticket.", "length"), \
                    gw.HOST_NAME, 0
            return 200, _ok_round("the whole answer"), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "triage"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "the whole answer"
        assert len(budgets) == 2
        assert budgets[1] > budgets[0], "the retry must buy more completion room"

    async def test_a_second_length_cut_is_returned_not_looped(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "len2@nasa.gov")
        calls = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            calls.append(1)
            return 200, _ok_round("partial", "length"), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "triage"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert len(calls) == 2, "exactly one retry"
        assert body["choices"][0]["finish_reason"] == "length"

    async def test_a_mid_task_413_is_flat_and_metered(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        # The conversation outgrows the key's window: the caller must get the
        # numbered {"error": ...} shape, never FastAPI's {"detail": ...}.
        row, team = await _issue_team(client, intake_headers, "big@nasa.gov",
                                      ctx=8192)

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):  # pragma: no cover
            raise AssertionError("the primary must not be asked")
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        huge = "x" * 40000
        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": huge}]},
            False, time.time())
        assert resp.status_code == 413
        body = json.loads(resp.body)
        assert body["error"]["type"] == "context_limit"
        rows = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC", (row["id"],))
        assert rows and rows[0]["status"] == 413

    async def test_spawn_calls_never_leak_to_the_client(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        # A round mixing a client tool with spawn_subagents: the client is
        # handed only the tool it defined -- it has no handler for the
        # gateway's, and an unanswerable tool_call id corrupts its loop.
        row, team = await _issue_team(client, intake_headers, "mix@nasa.gov")
        client_call = {"id": "c1", "type": "function",
                       "function": {"name": "read_file", "arguments": "{}"}}
        spawn_call = {"id": "s1", "type": "function",
                      "function": {"name": "spawn_subagents",
                                   "arguments": '{"tasks": [{"prompt": "x"}]}'}}

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, _ok_round("mixed round", "tool_calls",
                                  [client_call, spawn_call]), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "go"}]},
            False, time.time())
        body = json.loads(resp.body)
        calls = body["choices"][0]["message"]["tool_calls"]
        assert [c["id"] for c in calls] == ["c1"]

    async def test_resent_notice_banners_are_stripped_before_estimating(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "strip@nasa.gov")
        seen_msgs = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            seen_msgs.extend(body["messages"])
            return 200, _ok_round(), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        banner = "[Fleet notice: reduced to 126976. Your key is unchanged.]\n\n"
        await gw.team_orchestrate(
            row, team, {"model": "team", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": banner + "earlier answer"},
                {"role": "user", "content": "continue"}]},
            False, time.time())
        assistant = [m for m in seen_msgs if m.get("role") == "assistant"]
        assert assistant and assistant[0]["content"] == "earlier answer"

    async def test_a_capped_retry_cannot_outgrow_the_window(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        # The doubled budget is re-fitted by apply_ctx_limit at the top of the
        # loop: with a small team window the retry still happens, but the
        # budget never exceeds what the window can actually hold.
        row, team = await _issue_team(client, intake_headers, "cap@nasa.gov",
                                      ctx=8192)
        budgets: list[int] = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            budgets.append(int(body.get("max_tokens") or 0))
            finish = "length" if len(budgets) == 1 else "stop"
            return 200, _ok_round("partial", finish), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "triage"}]},
            False, time.time())
        assert len(budgets) == 2
        assert budgets[1] <= 8192, "the window still caps the retry"
        assert budgets[1] < 2 * budgets[0]

    async def test_max_rounds_all_spawn_returns_a_finished_looking_turn(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        # A primary that fans out on every round until the cap: the final
        # reply must not say finish_reason="tool_calls" while carrying no
        # tool_calls -- a client branching on that goes looking for an array
        # that is not there.
        row, team = await _issue_team(client, intake_headers, "cap2@nasa.gov")
        spawn_round = _ok_round("", "tool_calls", [{
            "id": "s1", "type": "function",
            "function": {"name": "spawn_subagents",
                         "arguments": '{"tasks": [{"prompt": "sub"}]}'}}])

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, json.loads(json.dumps(spawn_round)), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "go"}]},
            False, time.time())
        body = json.loads(resp.body)
        msg = body["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert body["choices"][0]["finish_reason"] != "tool_calls"
        assert body["x_team"]["rounds"] >= 1, "the cap, not round zero, ended it"

    async def test_structured_output_never_gets_the_notice_in_content(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        # response_format json_object + a disclosed ctx reduction: the JSON
        # the client will parse stays pristine; headers and x_fleet disclose.
        row, team = await _issue_team(client, intake_headers, "json@nasa.gov")

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, _ok_round('{"verdict": "ok"}'), "peer1", 4096
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": "classify"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert json.loads(body["choices"][0]["message"]["content"]) == \
            {"verdict": "ok"}
        assert body["x_fleet"]["ctx"]["granted"] == 4096
        assert resp.headers.get("X-Fleet-Ctx")

    async def test_the_job_is_closed_when_the_loop_returns(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "job@nasa.gov")

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            assert any(v.get("kind") == "inference"
                       and str(v.get("what", "")).startswith("team:")
                       for v in gw._active.values()), \
                "orchestration must be visible as active work"
            return 200, _ok_round(), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "hi"}]},
            False, time.time())
        assert not any(v.get("kind") == "inference" for v in gw._active.values())


# ---------------------------------------------------------------------------
# workers: failures degrade one task, truncation is named
# ---------------------------------------------------------------------------

class TestWorkerRound:
    async def test_a_truncated_worker_says_so_in_its_result(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "wtrunc@nasa.gov")

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, _ok_round("cut off mid-", "length"), "peer1", 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        out = await gw._run_subagents(row, team, [{"prompt": "summarize"}])
        assert out[0]["ok"] is True
        assert out[0]["truncated"] is True

    async def test_one_crashing_worker_does_not_kill_the_round(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        row, team = await _issue_team(client, intake_headers, "wcrash@nasa.gov")
        n = {"i": 0}

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("boom")
            return 200, _ok_round("fine"), "peer1", 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        out = await gw._run_subagents(
            row, team, [{"prompt": "a"}, {"prompt": "b"}])
        assert len(out) == 2
        assert sorted(o["ok"] for o in out) == [False, True]

    def test_note_fallback_never_breaks_the_reply(self, client, monkeypatch):
        def _boom(*a, **kw):
            raise RuntimeError("sqlite hiccup")
        monkeypatch.setattr(gw, "db_exec", _boom)
        gw._note_fallback(1, "a", "b")  # must not raise

    async def test_a_tiny_team_window_still_runs_workers(self, client, monkeypatch):
        # ctx_limit 1024: half of it would sit at the completion floor, where
        # apply_ctx_limit rejects EVERYTHING -- every spawn call a 413. A tiny
        # team's workers share the whole window instead.
        team = {"worker_models": "[]", "max_workers": 2, "worker_prompt": "",
                "ctx_limit": 1024, "primary_model": "some-model"}
        seen = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            seen.append(int(body.get("max_tokens") or 0))
            return 200, _ok_round("hi"), gw.HOST_NAME, 0
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        out = await gw._run_subagents(
            {"id": 1, "name": "k"}, team, [{"prompt": "say hi"}])
        assert out[0]["ok"] is True, out[0]
        assert seen and 0 < seen[0] <= 1024


# ---------------------------------------------------------------------------
# protocol fidelity: errors and streams a client can actually read
# ---------------------------------------------------------------------------

class TestOpenAIErrorEnvelope:
    def test_a_bare_401_reaches_v1_clients_in_the_openai_shape(self, client):
        r = client.post("/v1/chat/completions", json={"model": "m"})
        assert r.status_code == 401
        body = r.json()
        assert body["error"]["type"] == "authentication_error"
        assert "bearer" in body["error"]["message"].lower()

    def test_native_api_routes_get_the_same_envelope(self, client):
        # The Ollama-compatible surface lives under /api/* and speaks to the
        # same class of SDK clients.
        r = client.get("/api/tags")
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"

    def test_admin_routes_keep_the_detail_shape(self, client):
        # static/index.html reads .detail; only the API surfaces re-shape.
        r = client.get("/admin/api/keys")
        assert r.status_code in (401, 403)
        assert "detail" in r.json()

    async def test_sse_once_carries_tool_calls_and_chunks_long_content(self):
        long = "y" * (gw._SSE_PIECE + 100)
        resp = gw._sse_once({
            "id": "c1", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": long,
                                     "tool_calls": [{
                                         "id": "t1", "type": "function",
                                         "function": {"name": "f",
                                                      "arguments": "{}"}}]}}],
            "usage": {"total_tokens": 2}})
        frames = []
        async for chunk in resp.body_iterator:
            frames.append(chunk.decode())
        datas = [json.loads(f[5:].strip()) for f in frames
                 if f.startswith("data:") and "[DONE]" not in f]
        first = datas[0]["choices"][0]["delta"]
        assert first["role"] == "assistant"
        assert first["tool_calls"][0]["index"] == 0
        assert first["tool_calls"][0]["id"] == "t1"
        content = "".join(d["choices"][0]["delta"].get("content", "")
                          for d in datas if d["choices"][0]["delta"])
        assert content == long
        assert all(len(f) < gw._SSE_PIECE + 1024 for f in frames), \
            "no single SSE frame may scale with the whole answer"
        assert datas[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert frames[-1].startswith("data: [DONE]")


# ---------------------------------------------------------------------------
# the /v1 proxy end to end for a ctx-limited agent key (no Fleet Pass row):
# the pub_row gate came off, so these keys get the same care
# ---------------------------------------------------------------------------

def _agent_key(name, ctx_limit):
    raw, meta = gw.mint_key(name)
    gw.db_exec(
        "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
        "param_overrides,ctx_limit,updated_at) VALUES (?,1,?,?,?,?,?,?)",
        (int(meta["id"]), name, "[]", "", "{}", ctx_limit, gw.now()))
    return raw, meta


def _wire_one_peer(monkeypatch, host, model):
    async def _resolve(m, **kw):
        return [(host, model)]

    async def _pik(cand):
        return "k"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": host, "url": "http://%s:8080" % host, "token": "t"}])
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


class TestAgentKeyProxyPath:
    def test_history_banners_are_stripped_before_forwarding(
            self, client, monkeypatch):
        raw, _meta = _agent_key("usability-agent-1", 8192)
        _wire_one_peer(monkeypatch, "p1", "m")
        sent = {}

        async def fake_send(self, request, stream=False, **kw):
            sent["body"] = json.loads(request.content)
            return httpx.Response(200, request=request, json={
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2}})
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        banner = "[Fleet notice: reduced. Your key is unchanged.]\n\n"
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": banner + "earlier"},
                {"role": "user", "content": "continue"}]})
        assert r.status_code == 200, r.text
        assistant = [m for m in sent["body"]["messages"]
                     if m.get("role") == "assistant"]
        assert assistant and assistant[0]["content"] == "earlier"

    def test_over_limit_is_a_flat_413_and_metered(self, client, monkeypatch):
        raw, meta = _agent_key("usability-agent-2", 2048)
        _wire_one_peer(monkeypatch, "p1", "m")

        async def fake_send(self, request, stream=False, **kw):  # pragma: no cover
            raise AssertionError("an over-limit prompt must never go upstream")
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m",
                 "messages": [{"role": "user", "content": "x" * 30000}]})
        assert r.status_code == 413
        body = r.json()
        assert body["error"]["type"] == "context_limit"
        rows = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC",
            (int(meta["id"]),))
        assert rows and rows[0]["status"] == 413

    def test_streaming_ctx_cut_notice_is_a_wellformed_first_chunk(
            self, client, monkeypatch):
        # The key was issued for 16384 but the only box holds 8192: the
        # streamed disclosure must be a protocol-clean chunk (role included)
        # in front of the upstream's own frames, plus the X-Fleet-Ctx header.
        raw, _meta = _agent_key("usability-agent-3", 16384)
        _wire_one_peer(monkeypatch, "p1", "m")
        gw._routes_cache.update(ctx={("p1", "m"): 8192})

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(
                200, request=request,
                headers={"content-type": "text/event-stream"},
                content=(b'data: {"id":"c","choices":[{"index":0,"delta":'
                         b'{"role":"assistant","content":"hi"}}]}\n\n'
                         b'data: [DONE]\n\n'))
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        assert r.headers.get("x-fleet-ctx")
        first = next(line for line in r.text.splitlines()
                     if line.startswith("data:"))
        chunk = json.loads(first[5:].strip())
        delta = chunk["choices"][0]["delta"]
        assert delta["role"] == "assistant"
        assert delta["content"].startswith("[Fleet notice:")

    def test_structured_ctx_cut_reply_content_stays_pristine(
            self, client, monkeypatch):
        # Same reduction, but the client asked for json_object: the notice
        # stays out of the content it will parse; the header still discloses.
        raw, _meta = _agent_key("usability-agent-4", 16384)
        _wire_one_peer(monkeypatch, "p1", "m")
        gw._routes_cache.update(ctx={("p1", "m"): 8192})

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json={
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": '{"a": 1}'}}]})
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "response_format": {"type": "json_object"},
                 "messages": [{"role": "user", "content": "classify"}]})
        assert r.status_code == 200, r.text
        assert r.headers.get("x-fleet-ctx")
        body = r.json()
        assert json.loads(body["choices"][0]["message"]["content"]) == {"a": 1}
        assert body["x_fleet"]["ctx"]["granted"] == 8192
