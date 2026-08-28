"""The 2026-08-26 follow-up to the usability pass: how OFTEN a Fleet Pass
caller is told something, and whether the numbers they are told are the ones
they are held to.

Two field reports, both from the same team key driven by Cline:

  1. The context-reduction banner was stapled in front of EVERY turn. An
     agentic client re-requests on every tool round, so a fleet that stays
     degraded repeated the same sentence a dozen times in one task -- read by
     the user as a fresh failure each time. It is a disclosure; it lands once.

  2. The settings tab said 18 requests/hour and the 429 said 2. A key is
     minted with the tab's numbers stamped onto its own row, enforcement reads
     that row, and nothing ever brought an already-issued key forward -- while
     the tab, the key-status endpoint and the welcome mail all went on quoting
     today's figure.

Run with: gateway/.venv/Scripts/python.exe -m pytest tests -q
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

import app as gw

NOTICE_PREFIX = "[Fleet notice:"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _agent_key(name: str, ctx_limit: int) -> tuple[str, dict]:
    """A key the gateway decorates: an agent profile pins its context window,
    which is what makes a smaller box a disclosed reduction."""
    raw, meta = gw.mint_key(name)
    gw.db_exec(
        "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
        "param_overrides,ctx_limit,updated_at) VALUES (?,1,?,?,?,?,?,?)",
        (int(meta["id"]), name, "[]", "", "{}", ctx_limit, gw.now()))
    return raw, meta


def _fleet_pass_key(kind: str, email: str, *, rpd: int, rph: int,
                    ctx: int = 16384, status: str = "issued",
                    archived: bool = False) -> tuple[str, dict]:
    """An issued Fleet Pass key, minted the way issue_key() mints one: the
    settings of the moment stamped onto the api_keys row, with a public_keys
    row naming its kind."""
    raw, meta = gw.mint_key("fleet-pass:" + email, max_rpd=rpd, max_rph=rph)
    kid = int(meta["id"])
    gw.db_exec(
        "INSERT INTO public_keys(created_at,email,domain,kind,models,ctx,"
        "key_id,status,archived_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (gw.now(), email, email.split("@")[-1], kind,
         json.dumps({"primary": "gemma4-31b-qat"} if kind == "team"
                    else {"model": "gemma4-31b-qat"}),
         ctx, kid, status, gw.now() if archived else None))
    if archived:
        gw.db_exec("UPDATE api_keys SET archived_at=? WHERE id=?",
                   (gw.now(), kid))
    return raw, gw.db_query("SELECT * FROM api_keys WHERE id=?", (kid,))[0]


def _team_for(key_row: dict, ctx: int = 16384) -> dict:
    gw.db_exec(
        "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,"
        "max_workers,max_rounds,system_prompt,worker_prompt,ctx_limit,"
        "updated_at) VALUES (?,1,?,?,?,?,?,?,?,?,?)",
        (int(key_row["id"]), "fleet-pass", "gemma4-31b-qat", "[]", 2, 3,
         "", "", ctx, gw.now()))
    return gw.get_team(int(key_row["id"]))


def _wire_one_peer(monkeypatch, host: str, model: str) -> None:
    async def _resolve(m, **kw):
        return [(host, model)]

    async def _pik(cand):
        return "k"

    monkeypatch.setattr(gw, "resolve_targets", _resolve)
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": host, "url": "http://%s:8080" % host, "token": "t"}])
    monkeypatch.setattr(gw, "peer_inference_key", _pik)


def _round(content: str = "the whole answer") -> dict:
    return {
        "id": "chatcmpl-x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# A conversation as an agentic client resends it on its second turn: the
# reply it was shown, banner and all, then the next instruction.
BANNER = ("[Fleet notice: your key is set up for 16384 tokens of context, but "
          "the machine that can serve that much was unavailable, so this reply "
          "was answered with 8192. Your key is unchanged.]\n\n")


# ---------------------------------------------------------------------------
# already_replied(): the "have they been told yet" question
# ---------------------------------------------------------------------------

class TestAlreadyReplied:
    def test_a_first_turn_has_had_no_reply(self):
        assert gw.already_replied([
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "explore this project"}]) is False

    def test_a_prior_prose_reply_counts(self):
        assert gw.already_replied([
            {"role": "user", "content": "explore this project"},
            {"role": "assistant", "content": "I'll start by listing the files."},
            {"role": "user", "content": "continue"}]) is True

    def test_a_resent_banner_counts_as_having_been_told(self):
        # The exact shape the incident produced: banner plus one sentence.
        assert gw.already_replied([
            {"role": "user", "content": "triage"},
            {"role": "assistant", "content": BANNER + "I'll triage this ticket."},
            {"role": "user", "content": "continue with triage"}]) is True

    def test_a_tool_call_only_turn_is_not_a_reply(self):
        # prepend_notice() declines to write on a turn with no text, so the
        # banner has not been shown yet -- this turn may still carry it.
        assert gw.already_replied([
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "ls", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"}]) is False

    def test_whitespace_is_not_a_reply(self):
        assert gw.already_replied([
            {"role": "assistant", "content": "  \n "}]) is False

    def test_non_string_content_is_not_a_reply(self):
        assert gw.already_replied([
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]) is False

    def test_a_missing_or_malformed_history_is_a_first_turn(self):
        assert gw.already_replied(None) is False
        assert gw.already_replied("not a list") is False
        assert gw.already_replied([]) is False
        assert gw.already_replied(["junk", None, 7]) is False


# ---------------------------------------------------------------------------
# the /v1 proxy path: banner on turn one, silence afterwards
# ---------------------------------------------------------------------------

class TestNoticeOnceOnTheProxyPath:
    def _degraded(self, monkeypatch, name: str):
        """A key issued for 16384 whose only box holds 8192."""
        raw, _meta = _agent_key(name, 16384)
        _wire_one_peer(monkeypatch, "p1", "m")
        gw._routes_cache.update(ctx={("p1", "m"): 8192})

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json=_round())
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        return raw

    def test_the_first_turn_is_told(self, client, monkeypatch):
        raw = self._degraded(monkeypatch, "cadence-agent-1")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [
                {"role": "user", "content": "explore this project"}]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"].startswith(NOTICE_PREFIX)
        assert body["x_fleet"]["ctx"]["granted"] == 8192

    def test_the_second_turn_is_not_told_again(self, client, monkeypatch):
        raw = self._degraded(monkeypatch, "cadence-agent-2")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [
                {"role": "user", "content": "explore this project"},
                {"role": "assistant", "content": BANNER + "I'll start by listing."},
                {"role": "user", "content": "now read the README"}]})
        assert r.status_code == 200, r.text
        content = r.json()["choices"][0]["message"]["content"]
        assert NOTICE_PREFIX not in content
        assert content == "the whole answer"

    def test_the_machine_readable_disclosure_survives_the_silence(
            self, client, monkeypatch):
        # Suppressing the prose must not suppress the CONTRACT: x_fleet and
        # the X-Fleet-* headers ride every reply, told or not.
        raw = self._degraded(monkeypatch, "cadence-agent-3")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [
                {"role": "user", "content": "explore"},
                {"role": "assistant", "content": "listing now"},
                {"role": "user", "content": "continue"}]})
        assert r.status_code == 200, r.text
        assert r.headers.get("x-fleet-ctx")
        assert r.json()["x_fleet"]["ctx"] == {
            "requested": 16384, "granted": 8192, "host": "p1"}

    def test_a_tool_round_still_gets_its_first_telling(self, client, monkeypatch):
        # Turn one was tool-calls-only, so nothing was ever written in front
        # of it; the first turn that HAS prose is still owed the disclosure.
        raw = self._degraded(monkeypatch, "cadence-agent-4")
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [
                {"role": "user", "content": "list the files"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "1", "type": "function",
                                 "function": {"name": "ls", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "a.py"}]})
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"].startswith(NOTICE_PREFIX)

    def test_a_continued_stream_gets_no_synthetic_notice_chunk(
            self, client, monkeypatch):
        raw, _meta = _agent_key("cadence-agent-5", 16384)
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
            json={"model": "m", "stream": True, "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": BANNER + "working on it"},
                {"role": "user", "content": "continue"}]})
        assert r.status_code == 200, r.text
        assert r.headers.get("x-fleet-ctx"), "the header is the standing channel"
        assert NOTICE_PREFIX not in r.text
        first = next(line for line in r.text.splitlines()
                     if line.startswith("data:"))
        assert json.loads(first[5:].strip())["id"] == "c", \
            "the upstream's own first chunk must lead"


# ---------------------------------------------------------------------------
# the team loop: same cadence, and the loop's own rounds must not fool it
# ---------------------------------------------------------------------------

class TestNoticeOnceOnTheTeamPath:
    @pytest.mark.asyncio
    async def test_the_first_team_turn_is_told(self, monkeypatch, fake_fleet):
        _raw, row = _fleet_pass_key("team", "cadence-a@nasa.gov", rpd=38, rph=18)
        team = _team_for(row)

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, _round(), gw.HOST_NAME, 8192
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team", "messages": [
                {"role": "user", "content": "triage this ticket"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"].startswith(NOTICE_PREFIX)
        assert body["x_fleet"]["ctx"]["granted"] == 8192

    @pytest.mark.asyncio
    async def test_a_continued_team_turn_is_not_told_again(
            self, monkeypatch, fake_fleet):
        _raw, row = _fleet_pass_key("team", "cadence-b@nasa.gov", rpd=38, rph=18)
        team = _team_for(row)

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            return 200, _round(), gw.HOST_NAME, 8192
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team", "messages": [
                {"role": "user", "content": "triage this ticket"},
                {"role": "assistant", "content": BANNER + "I'll triage this."},
                {"role": "user", "content": "continue with triage"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "the whole answer"
        assert NOTICE_PREFIX not in body["choices"][0]["message"]["content"]
        # Still disclosed where a program can read it.
        assert body["x_fleet"]["ctx"]["granted"] == 8192

    @pytest.mark.asyncio
    async def test_the_loops_own_appended_rounds_do_not_count_as_a_reply(
            self, monkeypatch, fake_fleet):
        # `payload` is a shallow copy, so `messages` is the caller's list and
        # the loop appends this turn's own assistant rounds to it. Asking
        # "have they been told?" after that would answer yes on a genuine
        # first turn and silence the disclosure forever.
        _raw, row = _fleet_pass_key("team", "cadence-c@nasa.gov", rpd=38, rph=18)
        team = _team_for(row)
        calls = {"n": 0}

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return 200, {
                    "id": "c", "object": "chat.completion", "model": "m",
                    "choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {
                                     "role": "assistant", "content": "",
                                     "tool_calls": [{
                                         "id": "t1", "type": "function",
                                         "function": {
                                             "name": "spawn_subagents",
                                             "arguments": json.dumps(
                                                 {"tasks": [{"prompt": "a"}]})}}]}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2},
                }, gw.HOST_NAME, 8192
            return 200, _round(), gw.HOST_NAME, 8192
        monkeypatch.setattr(gw, "fleet_chat", _fake)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team", "messages": [
                {"role": "user", "content": "triage this ticket"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert calls["n"] >= 2, "the loop must actually have run a second round"
        assert body["choices"][0]["message"]["content"].startswith(NOTICE_PREFIX)


# ---------------------------------------------------------------------------
# the numbers a caller is quoted must be the numbers that turn them away
# ---------------------------------------------------------------------------

class TestLiveKeysFollowTheSettings:
    def test_raising_the_limit_moves_the_keys_already_out_there(self, client):
        _raw, row = _fleet_pass_key("team", "sync-a@nasa.gov", rpd=3, rph=2)
        gw.set_public_settings({"team_rpd": 38, "team_rph": 18})
        after = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                            (row["id"],))[0]
        assert (after["max_rpd"], after["max_rph"]) == (38, 18)

    def test_each_kind_keeps_its_own_allowance(self, client):
        _r1, team_row = _fleet_pass_key("team", "sync-b@nasa.gov", rpd=3, rph=2)
        _r2, single_row = _fleet_pass_key("single", "sync-c@nasa.gov", rpd=3, rph=2)
        gw.set_public_settings({"team_rpd": 38, "team_rph": 18,
                                "single_rpd": 9, "single_rph": 4})
        t = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                        (team_row["id"],))[0]
        s = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                        (single_row["id"],))[0]
        assert (t["max_rpd"], t["max_rph"]) == (38, 18)
        assert (s["max_rpd"], s["max_rph"]) == (9, 4)

    def test_a_hand_minted_admin_key_keeps_its_bespoke_budget(self, client):
        # Nothing on the Fleet Pass tab describes an admin key; the sync must
        # not reach one.
        _raw, meta = gw.mint_key("ops-runner", max_rpd=1000, max_rph=500)
        gw.set_public_settings({"team_rpd": 38, "team_rph": 18,
                                "single_rpd": 9, "single_rph": 4})
        row = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                          (int(meta["id"]),))[0]
        assert (row["max_rpd"], row["max_rph"]) == (1000, 500)

    def test_a_revoked_key_is_not_quietly_re_armed(self, client):
        _raw, row = _fleet_pass_key("team", "sync-d@nasa.gov", rpd=3, rph=2,
                                    archived=True)
        gw.set_public_settings({"team_rpd": 38, "team_rph": 18})
        after = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                            (row["id"],))[0]
        assert (after["max_rpd"], after["max_rph"]) == (3, 2)

    def test_the_clamped_value_is_what_gets_stamped(self, client):
        # 9999/hour is past the bound the settings reader enforces; a key must
        # never carry a number that reader would refuse.
        _raw, row = _fleet_pass_key("team", "sync-e@nasa.gov", rpd=3, rph=2)
        gw.set_public_settings({"team_rph": 9999})
        after = gw.db_query("SELECT max_rph FROM api_keys WHERE id=?",
                            (row["id"],))[0]
        assert after["max_rph"] == gw._PUBLIC_SETTING_BOUNDS["team_rph"][1]

    def test_the_save_reports_what_it_moved(self, client, admin_headers):
        _fleet_pass_key("team", "sync-f@nasa.gov", rpd=3, rph=2)
        r = client.put("/admin/api/public/settings", headers=admin_headers,
                       json={"team_rpd": 38, "team_rph": 18})
        assert r.status_code == 200, r.text
        assert r.json()["keys_updated"] == 1

    def test_a_second_save_of_the_same_numbers_moves_nothing(
            self, client, admin_headers):
        _fleet_pass_key("team", "sync-g@nasa.gov", rpd=3, rph=2)
        client.put("/admin/api/public/settings", headers=admin_headers,
                   json={"team_rpd": 38, "team_rph": 18})
        again = client.put("/admin/api/public/settings", headers=admin_headers,
                           json={"team_rpd": 38, "team_rph": 18})
        assert again.json()["keys_updated"] == 0

    def test_key_status_quotes_what_the_429_will_quote(
            self, client, intake_headers):
        # The reported confusion: the status endpoint read today's settings
        # while require_api_key() read the key's own row.
        raw, _row = _fleet_pass_key("team", "sync-h@nasa.gov", rpd=3, rph=2)
        gw.db_exec("INSERT INTO settings(key,value) VALUES ('public',?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (json.dumps({"team_rpd": 38, "team_rph": 18}),))
        r = client.post("/public/api/key-status", headers=intake_headers,
                        json={"key": raw})
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["limit_day"], body["limit_hour"]) == (3, 2)

    def test_a_boot_brings_stale_keys_forward(self, client):
        # The startup reconcile: an operator who raised the tab last week and
        # restarted must not have to save it again to make it real.
        _raw, row = _fleet_pass_key("team", "sync-i@nasa.gov", rpd=3, rph=2)
        gw.db_exec("INSERT INTO settings(key,value) VALUES ('public',?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (json.dumps({"team_rpd": 38, "team_rph": 18}),))
        assert gw.sync_public_key_limits(gw.get_public_settings()) == 1
        after = gw.db_query("SELECT max_rpd,max_rph FROM api_keys WHERE id=?",
                            (row["id"],))[0]
        assert (after["max_rpd"], after["max_rph"]) == (38, 18)

    def test_the_budget_that_bites_is_the_synced_one(self, client, monkeypatch):
        # End to end: a key minted at 2/hour, the tab raised to 18, and a
        # third request in the same hour that must now go through.
        raw, row = _fleet_pass_key("single", "sync-j@nasa.gov", rpd=3, rph=2)
        for _ in range(2):
            gw.record_usage(row, "m", "/v1/chat/completions", False, 200,
                            None, None, 5)
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200, "/v1/models never spends an allowance"

        _wire_one_peer(monkeypatch, "p1", "m")

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json=_round())
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        blocked = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert blocked.status_code == 429
        assert "2/1h" in blocked.json()["error"]["message"]

        gw.set_public_settings({"single_rpd": 9, "single_rph": 4})
        allowed = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert allowed.status_code == 200, allowed.text
