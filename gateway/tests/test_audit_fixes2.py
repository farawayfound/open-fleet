"""Regressions for the 2026-09 fault-tolerance/security audit, second pass
(findings-A and findings-B verified items).

One test class per finding, in the order the pass listed them. See
test_audit_fixes.py for the first pass's findings.

Run with: $SP/venv/bin/python -m pytest gateway/tests/test_audit_fixes2.py -q
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. lmstudio_plan(): a `private` record's weights are never published to LM
#    Studio, and a copy already sitting there is never read back as public.
# ---------------------------------------------------------------------------

class TestLmStudioSyncSkipsPrivateModels:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        home = tmp_path / "lmstudio-home"
        lms = tmp_path / "lmstudio-models"
        fleet = tmp_path / "fleet-models"
        for d in (home, lms, fleet):
            d.mkdir(parents=True, exist_ok=True)
        (home / "settings.json").write_text(
            json.dumps({"downloadsFolder": str(lms)}))
        monkeypatch.setattr(gw, "LMSTUDIO_HOME_ENV", str(home))
        monkeypatch.setattr(gw, "MODELS_DIR", fleet)
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
        monkeypatch.setattr(gw, "_lmstudio_restart_swap", lambda: "test: not restarted")
        gw._lmstudio_root_cache.update(t=0.0, root=None, home=None)
        gw._lmstudio_last.clear()
        gw.db_exec("DELETE FROM settings WHERE key='lmstudio'")
        gw.save_models([])
        return home, lms, fleet

    def _private_record(self, path) -> dict:
        return dict(gw.DEFAULT_MODEL_RECORD, id="dirty-muse-9b",
                   path=str(path), private=True)

    def test_a_private_models_weights_are_never_published(self, store):
        home, lms, fleet = store
        priv = fleet / "priv" / "model.gguf"
        priv.parent.mkdir(parents=True, exist_ok=True)
        priv.write_bytes(b"GGUF" + b"a" * 4092)
        gw.save_models([self._private_record(priv)])

        plan = gw.lmstudio_plan()
        publishes = [a for a in plan["actions"] if a["kind"] == "publish"]
        assert len(publishes) == 1
        assert publishes[0]["status"] == "skipped"
        assert publishes[0]["reason"] == "private model"
        assert not any(lms.rglob("*.gguf")), \
            "the private weights must never be linked into LM Studio's tree"

        # A real sync pass must agree with the plan.
        res = gw.lmstudio_sync()
        assert res["published"] == 0
        assert not any(lms.rglob("*.gguf"))

    def test_a_copy_already_in_lmstudios_tree_is_never_read_back_as_public(
            self, store):
        # Simulates the bug this closes: a hard link into LM Studio's
        # directory made before this fix existed (or a manual copy), which
        # the import loop must not treat as a new, unregistered model.
        home, lms, fleet = store
        priv = fleet / "priv" / "model.gguf"
        priv.parent.mkdir(parents=True, exist_ok=True)
        priv.write_bytes(b"GGUF" + b"a" * 4092)
        gw.save_models([self._private_record(priv)])

        leaked = lms / "fleet" / "priv" / "model.gguf"
        leaked.parent.mkdir(parents=True, exist_ok=True)
        os.link(priv, leaked)

        plan = gw.lmstudio_plan()
        imports = [a for a in plan["actions"] if a["kind"] == "import"]
        assert len(imports) == 1
        assert imports[0]["status"] == "skipped"
        assert imports[0]["reason"] == "private model"

        res = gw.lmstudio_sync()
        assert res["imported"] == 0
        ids = {m["id"] for m in gw.load_models()}
        assert ids == {"dirty-muse-9b"}, \
            "the leaked copy must never become a second, public record"

    def test_an_ordinary_public_model_is_unaffected(self, store):
        home, lms, fleet = store
        pub = fleet / "pub" / "model.gguf"
        pub.parent.mkdir(parents=True, exist_ok=True)
        pub.write_bytes(b"GGUF" + b"a" * 4092)
        gw.save_models([dict(gw.DEFAULT_MODEL_RECORD, id="public-one",
                             path=str(pub), private=False)])

        plan = gw.lmstudio_plan()
        publishes = [a for a in plan["actions"] if a["kind"] == "publish"]
        assert len(publishes) == 1
        assert publishes[0]["status"] == "todo"


# ---------------------------------------------------------------------------
# 2. POST /admin/api/reboot: a hub (any registered peers) needs {"confirm":
#    "hub"}, not the ordinary worker {"confirm": true}.
# ---------------------------------------------------------------------------

class TestRebootHubExclusion:
    def _stub(self, monkeypatch):
        monkeypatch.setattr(gw, "reboot_support", lambda: {
            "supported": True, "method": "systemd", "reason": None,
            "host": gw.HOST_NAME})
        monkeypatch.setattr(gw.subprocess, "Popen", lambda *a, **k: None)

    def test_a_hub_with_registered_peers_refuses_a_plain_confirm(
            self, client, admin_headers, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "peer1", "url": "http://peer1:8080", "token": "t"}])
        self._stub(monkeypatch)
        r = client.post("/admin/api/reboot", headers=admin_headers,
                        json={"confirm": True})
        assert r.status_code == 409, r.text
        assert "hub" in r.json()["detail"].lower()

    def test_a_hub_reboots_with_the_hub_confirm_literal(
            self, client, admin_headers, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "peer1", "url": "http://peer1:8080", "token": "t"}])
        self._stub(monkeypatch)
        r = client.post("/admin/api/reboot", headers=admin_headers,
                        json={"confirm": "hub"})
        assert r.status_code == 200, r.text

    def test_a_worker_with_no_registered_peers_still_accepts_a_plain_confirm(
            self, client, admin_headers, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")  # never written
        self._stub(monkeypatch)
        r = client.post("/admin/api/reboot", headers=admin_headers,
                        json={"confirm": True})
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 4. Batch cancellation survives a restart: the DB row, not just the
#    process-local set, carries the cancel.
# ---------------------------------------------------------------------------

class TestBatchCancelPersists:
    def _insert_running_batch(self, cancel_requested: int = 0) -> int:
        return gw.db_exec(
            "INSERT INTO batches(created_at,updated_at,key_id,key_name,label,"
            "models,status,total,cancel_requested) VALUES (?,?,?,?,?,?,?,?,?)",
            (gw.now(), gw.now(), None, "k", "", "[]", "running", 1,
             cancel_requested),
        )

    async def test_the_client_facing_cancel_route_persists_the_flag(self):
        raw, meta = gw.mint_key("batch-persist-cancel-1")
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        bid = self._insert_running_batch()
        gw.db_exec("UPDATE batches SET key_id=? WHERE id=?", (key_row["id"], bid))

        class _FakeRequest:
            method = "POST"

        out = await gw.v1_batches("batches/" + str(bid) + "/cancel",
                                  _FakeRequest(), key_row)
        assert json.loads(out.body)["cancelling"] == bid
        row = gw.db_query("SELECT * FROM batches WHERE id=?", (bid,))[0]
        assert row["cancel_requested"] == 1

    def test_the_admin_cancel_route_persists_the_flag(self, client, admin_headers):
        bid = self._insert_running_batch()
        r = client.post("/admin/api/batches/" + str(bid) + "/cancel",
                        headers=admin_headers)
        assert r.status_code == 200, r.text
        row = gw.db_query("SELECT * FROM batches WHERE id=?", (bid,))[0]
        assert row["cancel_requested"] == 1

    async def test_resume_orphaned_batches_skips_a_cancelled_one(self, monkeypatch):
        bid = self._insert_running_batch(cancel_requested=1)
        resumed: list[int] = []

        async def fake_batch_run(bid_, *a, **k):
            resumed.append(bid_)

        monkeypatch.setattr(gw, "_batch_run", fake_batch_run)
        await gw.resume_orphaned_batches()
        row = gw.db_query("SELECT * FROM batches WHERE id=?", (bid,))[0]
        assert row["status"] == "cancelled"
        assert bid not in resumed
        assert bid not in gw._batch_tasks

    async def test_resume_orphaned_batches_still_resumes_an_uncancelled_one(
            self, monkeypatch, tmp_path):
        bid = self._insert_running_batch(cancel_requested=0)
        in_path, _out_path = gw._batch_paths(bid)
        gw.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
        in_path.write_text(json.dumps({"messages": []}) + "\n", "utf-8")
        resumed: list[int] = []

        async def fake_batch_run(bid_, *a, **k):
            resumed.append(bid_)

        monkeypatch.setattr(gw, "_batch_run", fake_batch_run)
        await gw.resume_orphaned_batches()
        # resume_orphaned_batches() only SCHEDULES the dispatcher task; give
        # the loop a turn to actually run it before checking.
        for _ in range(5):
            if bid in resumed:
                break
            await asyncio.sleep(0)
        assert bid in resumed
        row = gw.db_query("SELECT * FROM batches WHERE id=?", (bid,))[0]
        assert row["status"] == "running"  # unchanged -- the fake never flushed


# ---------------------------------------------------------------------------
# 5. team_orchestrate(): TEAM_ROUND_MAX_TASKS caps the round total, not just
#    one spawn_subagents call's 32 tasks.
# ---------------------------------------------------------------------------

async def _issue_team(client, intake_headers, email, ctx=16384):
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


class TestTeamRoundTaskCap:
    async def test_a_multi_call_round_is_capped_across_all_spawn_calls(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        monkeypatch.setattr(gw, "TEAM_ROUND_MAX_TASKS", 3)
        row, team = await _issue_team(client, intake_headers, "cap3@nasa.gov")
        spawn_a = {"id": "call-a", "type": "function", "function": {
            "name": "spawn_subagents",
            "arguments": json.dumps({"tasks": [{"prompt": "1"}, {"prompt": "2"}]})}}
        spawn_b = {"id": "call-b", "type": "function", "function": {
            "name": "spawn_subagents",
            "arguments": json.dumps({"tasks": [{"prompt": "3"}, {"prompt": "4"}]})}}
        worker_prompts: list[str] = []
        primary_bodies: list[dict] = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            if endpoint == "/v1/team-primary":
                primary_bodies.append(body)
                if len(primary_bodies) == 1:
                    return 200, _ok_round("", "tool_calls",
                                          [spawn_a, spawn_b]), gw.HOST_NAME, 0
                return 200, _ok_round("done"), gw.HOST_NAME, 0
            worker_prompts.append(body["messages"][-1]["content"])
            return 200, _ok_round("worker ok"), gw.HOST_NAME, 0

        monkeypatch.setattr(gw, "fleet_chat", _fake)
        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                       "messages": [{"role": "user", "content": "go"}]},
            False, time.time())
        assert resp.status_code == 200, resp.body

        # Only 3 of the 4 requested tasks -- the round's cap -- actually ran.
        assert worker_prompts == ["1", "2", "3"]

        # Both tool_call ids still get an answer (an unanswered one corrupts
        # the primary's own tool-calling loop), and the capped call's answer
        # says why its last task never ran.
        second_round = primary_bodies[1]["messages"]
        tool_msgs = {m["tool_call_id"]: m for m in second_round
                    if m.get("role") == "tool"}
        assert set(tool_msgs) == {"call-a", "call-b"}
        b_results = json.loads(tool_msgs["call-b"]["content"])["results"]
        assert any("cap" in str(r.get("error", "")) for r in b_results)

    async def test_a_call_entirely_beyond_the_cap_gets_no_tasks_at_all(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        monkeypatch.setattr(gw, "TEAM_ROUND_MAX_TASKS", 1)
        row, team = await _issue_team(client, intake_headers, "cap4@nasa.gov")
        spawn_a = {"id": "call-a", "type": "function", "function": {
            "name": "spawn_subagents",
            "arguments": json.dumps({"tasks": [{"prompt": "1"}]})}}
        spawn_b = {"id": "call-b", "type": "function", "function": {
            "name": "spawn_subagents",
            "arguments": json.dumps({"tasks": [{"prompt": "2"}]})}}
        worker_prompts: list[str] = []
        primary_bodies: list[dict] = []

        async def _fake(key, body, endpoint, ctx_limit=None, **kw):
            if endpoint == "/v1/team-primary":
                primary_bodies.append(body)
                if len(primary_bodies) == 1:
                    return 200, _ok_round("", "tool_calls",
                                          [spawn_a, spawn_b]), gw.HOST_NAME, 0
                return 200, _ok_round("done"), gw.HOST_NAME, 0
            worker_prompts.append(body["messages"][-1]["content"])
            return 200, _ok_round("worker ok"), gw.HOST_NAME, 0

        monkeypatch.setattr(gw, "fleet_chat", _fake)
        await gw.team_orchestrate(
            row, team, {"model": "team",
                       "messages": [{"role": "user", "content": "go"}]},
            False, time.time())
        assert worker_prompts == ["1"]
        second_round = primary_bodies[1]["messages"]
        tool_msgs = {m["tool_call_id"]: m for m in second_round
                    if m.get("role") == "tool"}
        b_results = json.loads(tool_msgs["call-b"]["content"])["results"]
        assert len(b_results) == 1 and "cap" in str(b_results[0].get("error", ""))


# ---------------------------------------------------------------------------
# 6. Batch worker loop retires off a peer whose killswitch flips mid-batch.
# ---------------------------------------------------------------------------

_bid_counter = [990000]


def _new_batch(bodies: list[dict]):
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


def _wire_routes(monkeypatch, cands, cap, running=None):
    gw._routes_cache.update(t=time.time(), map={}, cands=cands, cap=cap,
                            running=running or {})

    async def _routes(force: bool = False):
        return gw._routes_cache["map"]

    monkeypatch.setattr(gw, "model_routes", _routes)


class TestBatchKillswitchRetiresWorker:
    @pytest.fixture(autouse=True)
    def _isolate(self):
        snapshot = dict(gw._routes_cache)
        gw._inflight.clear()
        gw._host_cooldown.clear()
        yield
        gw._routes_cache.clear()
        gw._routes_cache.update(snapshot)
        gw._inflight.clear()
        gw._host_cooldown.clear()

    async def test_a_peer_killed_before_the_batch_starts_never_gets_a_request(
            self, monkeypatch, tmp_path):
        # The killswitch is checked on a worker's very first loop iteration
        # (the throttle only limits RE-checks within the same worker), so a
        # peer that is already off by the time the batch's targets resolve
        # is the simplest reliable repro of "must never be dispatched to".
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "p1", "url": "http://p1:8080", "token": "t",
                       "routed": False}])
        _wire_routes(monkeypatch, {"m": ["p1"]}, {("p1", "m"): 1}, {"p1": {"m"}})

        async def fake_post_chat(cand, payload, read_timeout=None):
            raise AssertionError("a killed peer must never be dispatched to")

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        out = _read_output(out_path)
        assert len(out) == 1
        assert out[0]["ok"] is False
        assert "no reachable host" in out[0]["body"]["error"]["message"]

    async def test_the_local_host_is_never_treated_as_a_killed_peer(
            self, monkeypatch, tmp_path):
        # `cand` is "" for the local host -- there is no peers.json record
        # for it at all, so the killswitch check must never even ask.
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        _wire_routes(monkeypatch, {"m": [""]}, {("", "m"): 1}, {gw.HOST_NAME: {"m"}})
        called: list[str] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            called.append(cand)
            return httpx.Response(
                200, request=httpx.Request("POST", "http://x"),
                json={"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "ok"}}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                               "total_tokens": 2}})

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert called == [""]
        out = _read_output(out_path)
        assert out[0]["ok"] is True

    async def test_a_still_routed_peer_is_unaffected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "p1", "url": "http://p1:8080", "token": "t"}])
        _wire_routes(monkeypatch, {"m": ["p1"]}, {("p1", "m"): 1}, {"p1": {"m"}})
        called: list[str] = []

        async def fake_post_chat(cand, payload, read_timeout=None):
            called.append(cand)
            return httpx.Response(
                200, request=httpx.Request("POST", "http://x"),
                json={"choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "ok"}}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                               "total_tokens": 2}})

        monkeypatch.setattr(gw, "_post_chat", fake_post_chat)
        bid, _in, out_path = _new_batch(
            [{"messages": [{"role": "user", "content": "hi"}]}])
        await gw._batch_run(bid, ["m"], {"id": 1, "name": "k"})
        assert called == ["p1"]


# ---------------------------------------------------------------------------
# 7. public_request(): an auto-issued key still reports "issued" (so an
#    existing intake integration keeps working) but flags delivery failure.
# ---------------------------------------------------------------------------

class TestFleetPassAutoIssueMailFailure:
    def test_a_failed_delivery_is_still_issued_but_reported_undelivered(
            self, client, intake_headers, monkeypatch, fake_fleet):
        async def fake_send_key_email(row, raw_key):
            return False, "smtp: connection refused"

        monkeypatch.setattr(gw, "send_key_email", fake_send_key_email)
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "engineer@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "issued"
        assert body["delivered"] is False
        assert "could not email" in body["message"]

        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("engineer@nasa.gov",))[0]
        # Still counts as issued -- the key is real and live either way, and
        # the admin Public tab's resend filter (`status='issued'`) must keep
        # finding this row with no change of its own.
        assert row["status"] == "issued"
        assert row["email_error"] == "smtp: connection refused"
        assert row["emailed_at"] is None

        events = gw.db_query(
            "SELECT * FROM public_events WHERE kind='mail_error' "
            "ORDER BY id DESC LIMIT 1")
        assert events and events[0]["email"] == "engineer@nasa.gov"

    def test_a_successful_delivery_reports_delivered_true(
            self, client, intake_headers, captured_mail, fake_fleet):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "engineer2@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True})
        assert r.status_code == 200, r.text
        assert r.json()["delivered"] is True

    def test_the_admin_resend_path_still_applies_to_an_undelivered_row(
            self, client, intake_headers, admin_headers, monkeypatch, fake_fleet):
        async def fake_fail(row, raw_key):
            return False, "boom"

        monkeypatch.setattr(gw, "send_key_email", fake_fail)
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "engineer3@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True})
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("engineer3@nasa.gov",))[0]

        async def fake_ok(row, raw_key):
            return True, ""

        monkeypatch.setattr(gw, "send_key_email", fake_ok)
        r = client.post("/admin/api/public/keys/" + str(row["id"]) + "/resend",
                        headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["email_error"] == ""


# ---------------------------------------------------------------------------
# 9. relay()'s usage-token scan reassembles an SSE frame split across two
#    aiter_bytes() reads, without altering what is forwarded to the client.
# ---------------------------------------------------------------------------

class TestStreamingUsageAcrossSplitChunks:
    async def test_a_usage_frame_split_mid_line_is_still_captured(
            self, client, monkeypatch, fake_fleet):
        raw, meta = gw.mint_key("usage-split-1")
        full = ('data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
               'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
               '"usage":{"prompt_tokens":11,"completion_tokens":22,'
               '"total_tokens":33}}\n\ndata: [DONE]\n\n')
        split_at = full.index('"usage"') + 4     # inside the usage object's key
        part1, part2 = full[:split_at].encode(), full[split_at:].encode()

        async def fake_send(self, request, stream=False, **kw):
            resp = httpx.Response(200, request=request,
                                  headers={"content-type": "text/event-stream"})

            async def it():
                yield part1
                yield part2
            resp.aiter_bytes = it
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "gemma4-31b-qat", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        # The client still gets every byte, unmodified, whole line split or not.
        assert r.content == part1 + part2

        row = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC LIMIT 1",
            (meta["id"],))[0]
        assert row["total_tokens"] == 33
        assert row["prompt_tokens"] == 11
        assert row["completion_tokens"] == 22

    async def test_an_ordinary_unsplit_usage_frame_still_works(
            self, client, monkeypatch, fake_fleet):
        raw, meta = gw.mint_key("usage-split-2")
        full = ('data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
               '"usage":{"prompt_tokens":5,"completion_tokens":6,'
               '"total_tokens":11}}\n\ndata: [DONE]\n\n').encode()

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request,
                                  headers={"content-type": "text/event-stream"},
                                  content=full)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "gemma4-31b-qat", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        row = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC LIMIT 1",
            (meta["id"],))[0]
        assert row["total_tokens"] == 11
