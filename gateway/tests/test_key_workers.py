"""Max parallel workers on a key: sub-agents switched on from the key form.

Setting max_worker_nodes on a key gives it a teams row of its own (blank
primary = the client's model, one worker class as the roster), so POST
/v1/chat/completions for that key runs through the team loop. The number caps
how many OTHER nodes one request's workers may engage: null is off (a plain
key, today's behaviour), 0 is this node only, N is at most N other nodes.
Work on the key's own node never counts, the primary is never capped, and the
set of engaged nodes spans every round of one request. An ordinary Teams-tab
team (max_worker_nodes NULL) has no gate at all.

The worker class the admin picked on the form is allowed implicitly; every
other model -- the primary, and any roster entry added by hand -- is still
held to the key's own model list exactly as part 1 holds it.

Style follows test_key_access.py (the echo-checked _mint helper, the crew
fleet_chat fake, the scratch-DB migration test) and test_audit_fixes2.py
(TestTeamRoundTaskCap: team_orchestrate driven directly by a primary that
emits spawn_subagents calls). The node-cap tests replace only resolve_targets
and _post_chat, so fleet_chat's own candidate loop -- where the gate sits --
really runs.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from collections.abc import Callable

import httpx
import pytest

import app as gw

_MSGS = [{"role": "user", "content": "hi"}]
_NO_FALLBACK = {"X-Fleet-No-Fallback": "1"}
_PRIMARY = "qwen3.8-27b"   # the client's model; a form team's primary is blank
_FAST_ID = "qwen3.6-35b"   # a member of the shipped `fast` ladder
_SMALL_ID = "qwen3.5-4b"   # a member of both `fast` and `small`

_CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


# ---- fixtures and helpers ------------------------------------------------------

@pytest.fixture(autouse=True)
def _routes_cache_restored():
    """Peer aliases cleared and the routing table restored afterwards, as in
    test_key_access.py: a stale (host, name) -> canonical row would widen what
    a restricted key's list reaches, and the exemption tests pin exactly that."""
    saved = {k: (dict(v) if isinstance(v, dict) else set(v) if isinstance(v, set) else v)
             for k, v in gw._routes_cache.items()}
    gw._routes_cache["alias"] = {}
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)


@pytest.fixture
def upstream(monkeypatch):
    """Every request that leaves the process, by path, answered in-process."""
    sent: list[str] = []

    async def fake_send(self, request, stream=False, **kw):
        sent.append(request.url.path)
        return httpx.Response(200, request=request, json=_CHAT_OK_BODY)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return sent


@pytest.fixture
def routed(monkeypatch):
    """resolve_targets sending every plain-proxy request to this box."""
    calls: list[str] = []

    async def fake_resolve(model, **kw):
        calls.append(model)
        return [("", model)]

    monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
    return calls


@pytest.fixture
def orchestrated(monkeypatch):
    """team_orchestrate replaced by a recorder: the key ids it was handed. It
    answers 200 with a marker body, so a test can also tell which path served
    a request from the reply alone."""
    calls: list[int] = []

    async def spy(key, team, payload, stream, started, request=None):
        calls.append(int(key["id"]))
        return gw.JSONResponse({"spy": "team_orchestrate"})

    monkeypatch.setattr(gw, "team_orchestrate", spy)
    return calls


@pytest.fixture
def crew(monkeypatch):
    """fleet_chat replaced: the primary answers in one round, every call is
    recorded as (endpoint, body)."""
    calls: list[tuple[str, dict]] = []

    async def fake_chat(key, body, endpoint, ctx_limit=None, **kw):
        calls.append((endpoint, body))
        return 200, _round("done"), gw.HOST_NAME, 0

    monkeypatch.setattr(gw, "fleet_chat", fake_chat)
    return calls


def _round(content="done", finish="stop", tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"id": "chatcmpl-x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _spawn(call_id: str, tasks: list[dict]) -> dict:
    """One primary turn that calls spawn_subagents with `tasks`."""
    call = {"id": call_id, "type": "function", "function": {
        "name": "spawn_subagents", "arguments": json.dumps({"tasks": tasks})}}
    return _round("", "tool_calls", [call])


def _auth(raw: str, **extra: str) -> dict:
    return {"Authorization": "Bearer " + raw, **extra}


def _mint_workers(client, admin_headers, cap: int, cls: str | None = None,
                  name: str = "workers-key", **extra) -> tuple[str, int]:
    """A key minted through the admin API with sub-agents on.

    The echo is asserted here because it is the precondition of every test
    that builds on it: a gateway that ignored the field would mint a plain
    key, and a test of the team loop against that key would test nothing."""
    body = {"name": name, "max_worker_nodes": cap, **extra}
    if cls is not None:
        body["worker_class"] = cls
    r = client.post("/admin/api/keys", headers=admin_headers, json=body)
    assert r.status_code == 200, r.status_code
    got = r.json()
    # Compared through locals, never the whole body: assertion introspection
    # would otherwise print the raw key into the test log.
    echo, fields = got.get("max_worker_nodes", "<missing>"), ", ".join(sorted(got))
    assert echo == cap, "POST /admin/api/keys did not echo max_worker_nodes; fields: " + fields
    return got["key"], int(got["id"])


def _listed(client, admin_headers, kid: int) -> dict:
    items = client.get("/admin/api/keys", headers=admin_headers,
                       params={"limit": 200}).json()["items"]
    return next(k for k in items if k["id"] == kid)


def _team_of(kid: int) -> dict | None:
    rows = gw.db_query("SELECT * FROM teams WHERE key_id=?", (kid,))
    return rows[0] if rows else None


def _key_count() -> int:
    return gw.db_query("SELECT COUNT(*) c FROM api_keys")[0]["c"]


def _team_count() -> int:
    return gw.db_query("SELECT COUNT(*) c FROM teams")[0]["c"]


def _tab_team(kid: int, primary: str = "qwen3.5-4b") -> None:
    """An ordinary Teams-tab team, written with only the columns that tab
    knows: worker_class and max_worker_nodes are left to their defaults."""
    gw.db_exec(
        "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,"
        "max_workers,max_rounds,system_prompt,worker_prompt,updated_at) "
        "VALUES (?,1,?,?,?,?,?,?,?,?)",
        (kid, "crew", primary, json.dumps([primary]), 2, 3, "", "", gw.now()))


def _key_row(name: str, allowed: list[str] | None = None) -> dict:
    raw, meta = gw.mint_key(name, allowed_models=allowed)
    return gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]


def _form_team(kid: int, cap: int | None, cls: str = "fast",
               roster: list[str] | None = None, primary: str = "") -> dict:
    """The teams row as get_team() returns it for a key-form team: blank
    primary, the worker class as the roster, the column defaults for
    max_workers and max_rounds. Built in Python rather than read back from
    the table so the engine is pinned on its own, apart from the API that
    writes the row. cap=None, cls="" is an ordinary Teams-tab team."""
    return {"id": 1, "key_id": kid, "enabled": 1, "name": "workers",
            "primary_model": primary,
            "worker_models": json.dumps(roster if roster is not None else [cls]),
            "max_workers": 4, "max_rounds": 6, "system_prompt": "",
            "worker_prompt": "", "updated_at": gw.now(), "archived_at": None,
            "ctx_limit": None, "max_worker_nodes": cap, "worker_class": cls}


class _Wire:
    """resolve_targets and _post_chat stood in for; fleet_chat itself runs, so
    its candidate loop -- where the host gate sits -- is really exercised.

    Routes are per requested name: a fixed candidate list, or one rotated per
    call so successive workers prefer different boxes, the way the live
    scorer spreads load. `script` is the primary's turns in order, the last
    one repeated. Every call that reaches a box is recorded as (host, fleet
    id, is-primary); a candidate the gate refuses never shows up here."""

    def __init__(self) -> None:
        self.routes: dict[str, Callable[[int], list[tuple[str, str]]]] = {}
        self.asked: dict[str, int] = {}
        self.script: list[dict] = [_round("done")]
        self.sent: list[tuple[str, str, bool]] = []
        self.primary_bodies: list[dict] = []
        # The read timeout each call was given: None is "wait as long as it
        # takes", a number is the failover deadline fleet_chat sets while it
        # still has another box to try.
        self.timeouts: list[tuple[str, float | None]] = []
        # Hosts that answer every call with this HTTP status instead.
        self.fail: dict[str, int] = {}

    def route(self, model: str, pairs: list[tuple[str, str]]) -> None:
        self.routes[model] = lambda n: list(pairs)

    def rotate(self, model: str, hosts: list[str], fid: str) -> None:
        pairs = [(h, fid) for h in hosts]
        self.routes[model] = lambda n: pairs[n % len(pairs):] + pairs[:n % len(pairs)]

    async def resolve(self, model, **kw):
        n = self.asked.get(model, 0)
        self.asked[model] = n + 1
        pick = self.routes.get(model)
        return pick(n) if pick else []

    async def post(self, cand, payload, read_timeout=None):
        body = json.loads(payload)
        primary = any((t.get("function") or {}).get("name") == "spawn_subagents"
                      for t in body.get("tools") or [])
        self.sent.append((cand or gw.HOST_NAME, body["model"], primary))
        self.timeouts.append((cand or gw.HOST_NAME, read_timeout))
        # A scheduler yield, not a wait: every worker in flight reaches its
        # host pick before any answer lands -- the moment a gate that counted
        # hosts on the answer rather than on the pick would overrun its cap.
        await asyncio.sleep(0)
        if (cand or gw.HOST_NAME) in self.fail:
            return httpx.Response(self.fail[cand or gw.HOST_NAME],
                                  json={"error": {"message": "busy"}})
        if primary:
            self.primary_bodies.append(body)
            turn = self.script[min(len(self.primary_bodies), len(self.script)) - 1]
            return httpx.Response(200, json=turn)
        return httpx.Response(200, json=_round("worker ok"))

    def worker_hosts(self) -> list[str]:
        return [h for h, _fid, primary in self.sent if not primary]

    def worker_ids(self) -> list[str]:
        return [fid for _h, fid, primary in self.sent if not primary]

    def primary_hosts(self) -> list[str]:
        return [h for h, _fid, primary in self.sent if primary]

    def results(self) -> list[dict]:
        """Every sub-agent result the primary was handed, across all rounds."""
        msgs = self.primary_bodies[-1]["messages"] if self.primary_bodies else []
        return [r for m in msgs if m.get("role") == "tool"
                for r in json.loads(m["content"]).get("results", [])]


@pytest.fixture
def wire(monkeypatch, fake_fleet):
    w = _Wire()
    monkeypatch.setattr(gw, "resolve_targets", w.resolve)
    monkeypatch.setattr(gw, "_post_chat", w.post)
    return w


async def _orchestrate(key: dict, team: dict, model: str = _PRIMARY):
    return await gw.team_orchestrate(
        key, team, {"model": model, "messages": list(_MSGS), "fleet_no_fallback": 1},
        False, time.time())


def _remote(hosts: list[str]) -> set[str]:
    return {h for h in hosts if h != gw.HOST_NAME}


# ---- the columns ---------------------------------------------------------------

class TestMigration:
    def _pre_migration_schema(self, path) -> None:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                "CREATE TABLE teams ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, key_id INTEGER NOT NULL UNIQUE,"
                "enabled INTEGER NOT NULL DEFAULT 1, name TEXT NOT NULL DEFAULT '',"
                "primary_model TEXT NOT NULL DEFAULT '',"
                "worker_models TEXT NOT NULL DEFAULT '[]',"
                "max_workers INTEGER NOT NULL DEFAULT 4,"
                "max_rounds INTEGER NOT NULL DEFAULT 6,"
                "system_prompt TEXT NOT NULL DEFAULT '',"
                "worker_prompt TEXT NOT NULL DEFAULT '',"
                "updated_at TEXT, archived_at TEXT)")
            conn.execute(
                "INSERT INTO teams(key_id,name,primary_model,worker_models) "
                "VALUES (7,'old-crew','qwen3.5-4b','[\"qwen3.5-4b\"]')")
            conn.commit()
        finally:
            conn.close()

    def test_a_pre_migration_teams_table_gains_both_columns_and_old_teams_stay_uncapped(
            self, tmp_path, monkeypatch):
        scratch = tmp_path / "pre-migration.db"
        self._pre_migration_schema(scratch)
        monkeypatch.setattr(gw, "DB_PATH", scratch)

        gw.db_init()

        cols = {r["name"]: r for r in gw.db_query("PRAGMA table_info(teams)")}
        assert "max_worker_nodes" in cols, "MIGRATIONS must add teams.max_worker_nodes"
        assert "worker_class" in cols, "MIGRATIONS must add teams.worker_class"
        assert cols["max_worker_nodes"]["notnull"] == 0
        assert cols["max_worker_nodes"]["dflt_value"] is None
        assert cols["worker_class"]["notnull"] == 1
        assert cols["worker_class"]["dflt_value"] == "''"
        conn = sqlite3.connect(str(scratch))
        try:
            row = conn.execute("SELECT max_worker_nodes, worker_class FROM teams "
                               "WHERE name='old-crew'").fetchone()
        finally:
            conn.close()
        assert row == (None, "")


# ---- minting -------------------------------------------------------------------

_OFF = [
    pytest.param({}, id="field-absent"),
    pytest.param({"max_worker_nodes": None}, id="null"),
    pytest.param({"max_worker_nodes": ""}, id="empty-string"),
]


class TestMint:
    @pytest.mark.parametrize("extra", _OFF)
    def test_a_key_without_workers_echoes_off_and_gets_no_team(
            self, client, admin_headers, extra):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "plain", **extra})
        assert r.status_code == 200, r.status_code
        got = r.json()
        assert got.get("max_worker_nodes", "<missing>") is None
        assert got.get("worker_class", "<missing>") == ""
        assert _team_of(int(got["id"])) is None

    def test_workers_on_echoes_the_cap_and_fast_and_writes_the_form_team(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2, name="two-nodes")
        team = _team_of(kid)
        assert team is not None, "no teams row was written for a key with workers on"
        assert team["enabled"] == 1 and team["archived_at"] is None
        assert team["name"] == "two-nodes"
        assert team["primary_model"] == ""
        assert json.loads(team["worker_models"]) == ["fast"]
        assert team.get("worker_class") == "fast"
        assert team.get("max_worker_nodes") == 2
        assert (team["max_workers"], team["max_rounds"]) == (4, 6)

    def test_the_echo_names_the_default_class(self, client, admin_headers):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "default-class", "max_worker_nodes": 1})
        assert r.status_code == 200, r.status_code
        assert r.json().get("worker_class", "<missing>") == "fast"

    def test_the_worker_echo_rides_beside_the_model_list_echo(self, client, admin_headers):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "both", "max_worker_nodes": 1,
                              "allowed_models": [_PRIMARY]})
        assert r.status_code == 200, r.status_code
        got = r.json()
        assert got.get("allowed_models") == [_PRIMARY]
        assert got.get("max_worker_nodes", "<missing>") == 1
        assert got.get("worker_class", "<missing>") == "fast"

    def test_a_chosen_shipped_class_is_stored(self, client, admin_headers):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "small-class", "max_worker_nodes": 1,
                              "worker_class": "small"})
        assert r.status_code == 200, r.status_code
        assert r.json().get("worker_class", "<missing>") == "small"
        team = _team_of(int(r.json()["id"])) or {}
        assert team.get("worker_class") == "small"
        assert json.loads(team.get("worker_models") or "[]") == ["small"]

    def test_a_class_saved_on_the_routing_tab_is_accepted(self, client, admin_headers):
        gw.set_role_config({"classes": {"notes": {"models": [_SMALL_ID], "nodes": []}}})
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "custom-class", "max_worker_nodes": 1,
                              "worker_class": "notes"})
        assert r.status_code == 200, r.text
        assert r.json().get("worker_class", "<missing>") == "notes"

    @pytest.mark.parametrize("cap", [
        pytest.param(0, id="zero-is-this-node-only"),
        pytest.param(32, id="thirty-two-is-the-ceiling"),
    ])
    def test_a_boundary_cap_is_accepted_and_stored_as_given(
            self, client, admin_headers, cap):
        raw, kid = _mint_workers(client, admin_headers, cap)
        assert (_team_of(kid) or {}).get("max_worker_nodes") == cap

    @pytest.mark.parametrize("cap", [
        pytest.param(-1, id="negative"),
        pytest.param(33, id="over-32"),
        pytest.param("x", id="not-a-number"),
        pytest.param(1.5, id="not-whole"),
    ])
    def test_a_bad_cap_is_a_400_and_mints_nothing(self, client, admin_headers, cap):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "bad-cap", "max_worker_nodes": cap})
        assert r.status_code == 400, r.status_code
        assert _key_count() == 0, "a refused cap must not leave a key behind"
        assert _team_count() == 0

    def test_an_unknown_class_is_a_400_naming_the_known_ones_and_mints_nothing(
            self, client, admin_headers):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "bad-class", "max_worker_nodes": 1,
                              "worker_class": "turbo"})
        assert r.status_code == 400, r.status_code
        assert "fast" in str(r.json().get("detail")), r.text
        assert _key_count() == 0, "a refused class must not leave a key behind"
        assert _team_count() == 0

    def test_a_failed_team_insert_fails_the_request_and_removes_the_new_key(
            self, client, admin_headers, monkeypatch):
        real = gw.db_exec
        fired: list[str] = []

        def failing(sql, args=()):
            if re.match(r"\s*(INSERT|REPLACE)\b.*\bINTO\s+teams\b", sql, re.I | re.S):
                fired.append(sql)
                raise sqlite3.OperationalError("database is locked (injected)")
            return real(sql, args)

        monkeypatch.setattr(gw, "db_exec", failing)
        try:
            status = client.post("/admin/api/keys", headers=admin_headers,
                                 json={"name": "half-made", "max_worker_nodes": 1}
                                 ).status_code
        except sqlite3.OperationalError:
            status = 500   # re-raised past the handler: the server answers 500
        assert status >= 400, "a key whose team was not written must not be handed out"
        assert fired, "the teams row was never written through db_exec"
        assert _key_count() == 0, "the just-minted key must be deleted again"


# ---- listing -------------------------------------------------------------------

class TestList:
    def test_a_form_key_lists_its_cap_and_class(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2, cls="small")
        item = _listed(client, admin_headers, kid)
        assert item.get("max_worker_nodes", "<missing>") == 2
        assert item.get("worker_class", "<missing>") == "small"

    def test_a_plain_key_lists_workers_off(self, client, admin_headers):
        raw, meta = gw.mint_key("plain")
        item = _listed(client, admin_headers, int(meta["id"]))
        assert item.get("max_worker_nodes", "<missing>") is None
        assert item.get("worker_class", "<missing>") == ""

    def test_a_teams_tab_team_lists_a_blank_class(self, client, admin_headers):
        raw, meta = gw.mint_key("tab-team")
        _tab_team(int(meta["id"]))
        item = _listed(client, admin_headers, int(meta["id"]))
        assert item.get("max_worker_nodes", "<missing>") is None
        assert item.get("worker_class", "<missing>") == ""


# ---- editing -------------------------------------------------------------------

def _patch(client, admin_headers, kid: int, body: dict):
    return client.patch("/admin/api/keys/" + str(kid), headers=admin_headers, json=body)


class TestPatch:
    def test_a_cap_on_a_key_without_workers_creates_its_form_team(
            self, client, admin_headers):
        raw, meta = gw.mint_key("patch-on")
        kid = int(meta["id"])
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 3})
        assert r.status_code == 200, r.text
        assert r.json().get("max_worker_nodes", "<missing>") == 3
        assert r.json().get("worker_class", "<missing>") == "fast"
        team = _team_of(kid)
        assert team is not None and team["enabled"] == 1
        assert team.get("max_worker_nodes") == 3 and team.get("worker_class") == "fast"
        assert team["primary_model"] == ""
        assert json.loads(team["worker_models"]) == ["fast"]

    def test_null_switches_the_form_team_off_and_the_key_serves_plain_chat_again(
            self, client, admin_headers, fake_fleet, routed, upstream, orchestrated):
        raw, kid = _mint_workers(client, admin_headers, 2)
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": _PRIMARY, "messages": _MSGS})
        assert orchestrated == [kid], "the key was not served by the team loop to begin with"

        r = _patch(client, admin_headers, kid, {"max_worker_nodes": None})
        assert r.status_code == 200, r.text
        assert r.json().get("max_worker_nodes", "<missing>") is None
        assert (_team_of(kid) or {}).get("enabled") == 0
        item = _listed(client, admin_headers, kid)
        assert (item.get("max_worker_nodes", "<missing>"), item.get("worker_class")) == (None, "")

        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": _PRIMARY, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert orchestrated == [kid], "a switched-off key still went to the team loop"
        assert "/v1/chat/completions" in upstream

    def test_a_cap_after_null_re_enables_the_same_row_with_the_new_class(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": None}).status_code == 200
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 1, "worker_class": "small"})
        assert r.status_code == 200, r.text
        assert r.json().get("worker_class", "<missing>") == "small"
        rows = gw.db_query("SELECT * FROM teams WHERE key_id=?", (kid,))
        assert len(rows) == 1
        assert rows[0]["enabled"] == 1
        assert (rows[0].get("max_worker_nodes"), rows[0].get("worker_class")) == (1, "small")
        assert json.loads(rows[0]["worker_models"]) == ["small"]

    def test_a_key_fronting_a_teams_tab_team_is_a_409_and_the_team_is_untouched(
            self, client, admin_headers):
        raw, meta = gw.mint_key("tab-team")
        kid = int(meta["id"])
        _tab_team(kid, primary="qwen3.5-4b")
        before = _team_of(kid)
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 3})
        assert r.status_code == 409, r.text
        assert "Teams tab" in str(r.json().get("detail")), r.text
        assert _team_of(kid) == before

    def test_an_archived_key_is_a_404_and_gets_no_team(self, client, admin_headers):
        raw, meta = gw.mint_key("archived")
        kid = int(meta["id"])
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 3})
        assert r.status_code == 404, r.text
        assert _team_of(kid) is None

    @pytest.mark.parametrize("bad", [
        pytest.param({"max_worker_nodes": 33}, id="cap-over-32"),
        pytest.param({"max_worker_nodes": 1, "worker_class": "turbo"}, id="unknown-class"),
    ])
    def test_a_bad_worker_field_is_a_400_and_saves_none_of_the_patch(
            self, client, admin_headers, bad):
        raw, meta = gw.mint_key("patch-bad")
        kid = int(meta["id"])
        r = _patch(client, admin_headers, kid, {"name": "renamed", **bad})
        assert r.status_code == 400, r.text
        assert gw.db_query("SELECT name FROM api_keys WHERE id=?", (kid,))[0]["name"] == "patch-bad"
        assert _team_of(kid) is None


# ---- archive and restore -------------------------------------------------------

class TestArchive:
    def test_archiving_a_workers_key_archives_its_form_team(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert (_team_of(kid) or {}).get("archived_at") is not None
        assert gw.get_team(kid) is None

    def test_restoring_a_key_does_not_switch_its_workers_back_on(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": None}).status_code == 200
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        r = client.post("/admin/api/keys/" + str(kid) + "/restore", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert (_team_of(kid) or {}).get("enabled") == 0
        assert gw.get_team(kid) is None

    def test_a_team_removed_on_the_teams_tab_stays_removed_across_archive_and_restore(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        # Removed earlier, under a stamp of its own (the DELETE route would
        # stamp "now", the same second as the key's archive below).
        removed = "2020-01-01T00:00:00+00:00"
        gw.db_exec("UPDATE teams SET archived_at=? WHERE key_id=?", (removed, kid))
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert client.post("/admin/api/keys/" + str(kid) + "/restore",
                           headers=admin_headers).status_code == 200
        assert gw.get_team(kid) is None, "restoring the key woke up a team that was already removed"
        assert (_team_of(kid) or {}).get("archived_at") == removed
        assert _listed(client, admin_headers, kid).get("max_worker_nodes", "<missing>") is None

    def test_the_team_a_key_archive_took_down_comes_back_with_the_key(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2, cls="small")
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert gw.get_team(kid) is None
        assert client.post("/admin/api/keys/" + str(kid) + "/restore",
                           headers=admin_headers).status_code == 200
        team = gw.get_team(kid)
        assert team is not None and (team["max_worker_nodes"], team["worker_class"]) == (2, "small")


def _listed_archived(client, admin_headers, kid: int) -> dict:
    items = client.get("/admin/api/keys", headers=admin_headers,
                       params={"limit": 200, "archived": "true"}).json()["items"]
    return next(k for k in items if k["id"] == kid)


class TestArchivedListing:
    def test_an_archived_key_lists_the_setting_its_restore_brings_back(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2, cls="small")
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        item = _listed_archived(client, admin_headers, kid)
        assert (item.get("max_worker_nodes", "<missing>"), item.get("worker_class")) == (2, "small")
        assert client.post("/admin/api/keys/" + str(kid) + "/restore",
                           headers=admin_headers).status_code == 200
        item = _listed(client, admin_headers, kid)
        assert (item.get("max_worker_nodes", "<missing>"), item.get("worker_class")) == (2, "small")

    def test_an_archived_key_whose_team_was_removed_earlier_reads_off(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        gw.db_exec("UPDATE teams SET archived_at=? WHERE key_id=?",
                   ("2020-01-01T00:00:00+00:00", kid))
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        item = _listed_archived(client, admin_headers, kid)
        assert (item.get("max_worker_nodes", "<missing>"), item.get("worker_class")) == (None, "")

    def test_an_archived_key_whose_workers_were_off_reads_off(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": None}).status_code == 200
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        item = _listed_archived(client, admin_headers, kid)
        assert (item.get("max_worker_nodes", "<missing>"), item.get("worker_class")) == (None, "")


# ---- editing keeps what the Teams tab set --------------------------------------

class TestPatchKeepsTeamEdits:
    def test_a_cap_sent_alone_keeps_the_keys_class(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2, cls="small")
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 3})
        assert r.status_code == 200, r.text
        assert r.json().get("worker_class", "<missing>") == "small"
        team = _team_of(kid) or {}
        assert (team.get("max_worker_nodes"), team.get("worker_class")) == (3, "small")
        assert json.loads(team["worker_models"]) == ["small"]

    def test_a_key_that_never_had_a_class_gets_fast(self, client, admin_headers):
        raw, meta = gw.mint_key("never-had-one")
        r = _patch(client, admin_headers, int(meta["id"]), {"max_worker_nodes": 1})
        assert r.status_code == 200, r.text
        assert r.json().get("worker_class", "<missing>") == "fast"

    def test_moving_the_cap_leaves_a_roster_edited_on_the_teams_tab(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={"name": "crew", "primary_model": "",
                             "worker_models": ["fast", "small"]})
        assert r.status_code == 200, r.text
        # What the dashboard sends on a save: the cap and the class it shows.
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 3, "worker_class": "fast"})
        assert r.status_code == 200, r.text
        team = _team_of(kid) or {}
        assert team["max_worker_nodes"] == 3
        assert json.loads(team["worker_models"]) == ["fast", "small"]

    def test_changing_the_class_sets_the_roster_to_the_new_class(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={"name": "crew", "primary_model": "",
                             "worker_models": ["fast", "small"]})
        assert r.status_code == 200, r.text
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 2, "worker_class": "small"})
        assert r.status_code == 200, r.text
        team = _team_of(kid) or {}
        assert team["worker_class"] == "small"
        assert json.loads(team["worker_models"]) == ["small"]

    def test_a_cap_alone_with_a_stored_class_routing_forgot_is_a_400_and_saves_nothing(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        gw.db_exec("UPDATE teams SET worker_class='gone' WHERE key_id=?", (kid,))
        r = _patch(client, admin_headers, kid, {"name": "renamed", "max_worker_nodes": 3})
        assert r.status_code == 400, r.text
        assert "gone" in str(r.json().get("detail")) and "fast" in str(r.json().get("detail"))
        assert gw.db_query("SELECT name FROM api_keys WHERE id=?", (kid,))[0]["name"] == "workers-key"
        assert (_team_of(kid) or {}).get("max_worker_nodes") == 2

    def test_switching_off_does_not_need_the_stored_class_to_exist(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 2)
        gw.db_exec("UPDATE teams SET worker_class='gone' WHERE key_id=?", (kid,))
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": None})
        assert r.status_code == 200, r.text
        assert (_team_of(kid) or {}).get("enabled") == 0


# ---- an agent profile and sub-agents are never both live on one key -------------

def _put_agent(client, admin_headers, kid: int, enabled: bool = True):
    return client.put("/admin/api/agents/" + str(kid), headers=admin_headers,
                      json={"name": "pinned", "force_model": _PRIMARY,
                            "rules": "be brief", "enabled": enabled})


def _agent_rows(kid: int) -> list[dict]:
    return gw.db_query("SELECT * FROM agents WHERE key_id=?", (kid,))


class TestAgentProfileAndWorkers:
    """The team loop is taken before an agent profile is read, so workers on a
    key with a live profile would silently stop its model pin, rules and
    limits. The keys form refuses that order, and the Agents routes refuse the
    other."""

    def test_workers_over_a_live_profile_are_a_409_and_save_none_of_the_patch(
            self, client, admin_headers):
        raw, meta = gw.mint_key("pinned")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid).status_code == 200
        r = _patch(client, admin_headers, kid, {"name": "renamed", "max_worker_nodes": 1})
        assert r.status_code == 409, r.text
        assert "agent profile" in str(r.json().get("detail")), r.text
        assert _team_of(kid) is None
        assert gw.db_query("SELECT name FROM api_keys WHERE id=?", (kid,))[0]["name"] == "pinned"

    def test_a_profile_does_not_stop_workers_being_switched_off(self, client, admin_headers):
        raw, meta = gw.mint_key("pinned-off")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid).status_code == 200
        r = _patch(client, admin_headers, kid, {"name": "renamed", "max_worker_nodes": None})
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "renamed"

    def test_a_profile_switched_off_does_not_block_workers(self, client, admin_headers):
        raw, meta = gw.mint_key("pinned-idle")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid, enabled=False).status_code == 200
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 1})
        assert r.status_code == 200, r.text
        assert gw.get_team(kid) is not None

    def test_a_profile_over_live_workers_is_a_409_and_writes_no_profile(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        r = _put_agent(client, admin_headers, kid)
        assert r.status_code == 409, r.text
        assert "sub-agents" in str(r.json().get("detail")), r.text
        assert _agent_rows(kid) == []

    def test_a_profile_saved_switched_off_is_allowed_over_workers(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        assert _put_agent(client, admin_headers, kid, enabled=False).status_code == 200
        assert gw.get_agent(kid) is None and gw.get_team(kid) is not None

    def test_restoring_a_profile_over_live_workers_is_a_409(self, client, admin_headers):
        raw, meta = gw.mint_key("pinned-restore")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid).status_code == 200
        assert client.delete("/admin/api/agents/" + str(kid), headers=admin_headers).status_code == 200
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": 1}).status_code == 200
        r = client.post("/admin/api/agents/" + str(kid) + "/restore", headers=admin_headers)
        assert r.status_code == 409, r.text
        assert gw.get_agent(kid) is None

    def test_a_teams_tab_team_and_a_profile_are_left_as_they_were(self, client, admin_headers):
        # Not widened: team-beats-agent on a Teams-tab team is the Teams tab's
        # own, visible order (test_public_auto_issue pins it).
        raw, meta = gw.mint_key("tab-and-agent")
        kid = int(meta["id"])
        _tab_team(kid)
        assert _put_agent(client, admin_headers, kid).status_code == 200

    def test_restoring_a_form_team_over_a_live_profile_is_a_409(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        assert client.delete("/admin/api/teams/" + str(kid), headers=admin_headers).status_code == 200
        assert _put_agent(client, admin_headers, kid).status_code == 200   # no live workers now
        r = client.post("/admin/api/teams/" + str(kid) + "/restore", headers=admin_headers)
        assert r.status_code == 409, r.text
        assert "agent profile" in str(r.json().get("detail")), r.text
        assert gw.get_team(kid) is None and gw.get_agent(kid) is not None

    def test_rearming_a_form_team_on_the_teams_tab_over_a_live_profile_is_a_409(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": None}).status_code == 200
        assert _put_agent(client, admin_headers, kid).status_code == 200
        team = {"name": "crew", "primary_model": "", "worker_models": ["fast"]}
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={**team, "enabled": True})
        assert r.status_code == 409, r.text
        assert "agent profile" in str(r.json().get("detail")), r.text
        assert gw.get_team(kid) is None
        # Saved switched off it arms nothing, so it is allowed.
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={**team, "enabled": False})
        assert r.status_code == 200, r.text

    def test_restoring_a_key_does_not_bring_back_a_profile_archived_before_it(
            self, client, admin_headers):
        raw, meta = gw.mint_key("profile-then-workers")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid).status_code == 200
        assert client.delete("/admin/api/agents/" + str(kid), headers=admin_headers).status_code == 200
        # Under a stamp of its own (the DELETE route would stamp "now", the same
        # second as the key's archive below).
        gw.db_exec("UPDATE agents SET archived_at=? WHERE key_id=?",
                   ("2020-01-01T00:00:00+00:00", kid))
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": 1}).status_code == 200
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert client.post("/admin/api/keys/" + str(kid) + "/restore",
                           headers=admin_headers).status_code == 200
        assert gw.get_team(kid) is not None, "the workers come back with the key"
        assert gw.get_agent(kid) is None, "a profile removed earlier came back beside live workers"

    def test_restoring_a_key_brings_back_the_profile_its_own_archive_took_down(
            self, client, admin_headers):
        raw, meta = gw.mint_key("pinned-archived")
        kid = int(meta["id"])
        assert _put_agent(client, admin_headers, kid).status_code == 200
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert gw.get_agent(kid) is None
        assert client.post("/admin/api/keys/" + str(kid) + "/restore",
                           headers=admin_headers).status_code == 200
        assert gw.get_agent(kid) is not None


# ---- a crew with no roster -----------------------------------------------------

class TestEmptyRoster:
    @pytest.mark.asyncio
    async def test_workers_of_a_blank_lead_with_no_roster_run_on_the_clients_model(
            self, client, wire):
        # A Teams-tab team saved with "client's model" as its lead and nothing
        # ticked for the workers: the primary is told the workers are that
        # model, so that is where they must go -- not to model ''.
        key = _key_row("no-roster")
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.script = [_spawn("call-1", [{"prompt": "a"}, {"prompt": "b"}]), _round("done")]
        team = _form_team(int(key["id"]), cap=None, cls="", roster=[], primary="")
        resp = await _orchestrate(key, team)
        assert resp.status_code == 200, resp.body
        assert wire.worker_ids() == [_PRIMARY, _PRIMARY], wire.sent
        assert all(r.get("ok") for r in wire.results()), wire.results()

    @pytest.mark.asyncio
    async def test_a_pinned_lead_with_no_roster_still_runs_the_workers_on_that_lead(
            self, client, wire):
        key = _key_row("no-roster-pinned")
        wire.route(_SMALL_ID, [("", _SMALL_ID)])
        wire.script = [_spawn("call-1", [{"prompt": "a"}]), _round("done")]
        team = _form_team(int(key["id"]), cap=None, cls="", roster=[], primary=_SMALL_ID)
        resp = await _orchestrate(key, team)
        assert resp.status_code == 200, resp.body
        assert wire.worker_ids() == [_SMALL_ID], wire.sent


# ---- which path serves chat ----------------------------------------------------

class TestChatPath:
    @pytest.mark.parametrize("extra", _OFF)
    def test_a_key_without_workers_is_served_by_the_plain_proxy(
            self, client, admin_headers, fake_fleet, routed, upstream, orchestrated, extra):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "plain", **extra})
        assert r.status_code == 200, r.status_code
        raw = r.json()["key"]
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": _PRIMARY, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert orchestrated == []
        assert routed == [_PRIMARY] and "/v1/chat/completions" in upstream

    def test_a_key_with_workers_is_orchestrated_on_the_clients_model(
            self, client, admin_headers, fake_fleet, crew):
        raw, kid = _mint_workers(client, admin_headers, 1)
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": _PRIMARY, "messages": _MSGS, "fleet_no_fallback": 1})
        assert r.status_code == 200, r.text
        assert "x_team" in r.json()
        (body,) = [b for e, b in crew if e == "/v1/team-primary"]
        assert body["model"] == _PRIMARY
        assert any((t.get("function") or {}).get("name") == "spawn_subagents"
                   for t in body.get("tools") or [])


# ---- the node cap --------------------------------------------------------------

class TestNodeCap:
    @pytest.mark.asyncio
    async def test_cap_one_puts_every_remote_worker_on_one_peer_and_still_uses_this_node(
            self, client, wire):
        key = _key_row("cap-one")
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.rotate("fast", ["peer1", "peer2", "peer3", ""], _FAST_ID)
        wire.script = [_spawn("call-1", [{"prompt": "t%d" % i} for i in range(6)]),
                       _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=1))
        assert resp.status_code == 200, resp.body
        hosts = wire.worker_hosts()
        assert len(hosts) == 6, hosts
        assert len(_remote(hosts)) == 1, hosts
        assert gw.HOST_NAME in hosts, hosts
        assert all(r.get("ok") for r in wire.results()), wire.results()

    @pytest.mark.asyncio
    async def test_cap_zero_keeps_workers_local_and_fails_only_the_task_no_local_box_holds(
            self, client, wire):
        key = _key_row("cap-zero")
        wire.route(_PRIMARY, [("", _PRIMARY)])
        # The remote box is ranked first, so a local pick means the gate moved
        # past it -- not that routing happened to prefer this node.
        wire.route("fast", [("peer1", _FAST_ID), ("", _FAST_ID)])
        wire.route("small", [("peer1", _SMALL_ID), ("peer2", _SMALL_ID)])
        # A second class added to the roster by hand (the Teams tab can), so
        # one task asks for a class that no box on this node holds.
        team = _form_team(int(key["id"]), cap=0, roster=["fast", "small"])
        wire.script = [_spawn("call-1", [{"prompt": "a", "model": "fast"},
                                         {"prompt": "b", "model": "small"},
                                         {"prompt": "c", "model": "fast"}]),
                       _round("done")]
        resp = await _orchestrate(key, team)
        assert resp.status_code == 200, resp.body
        out = json.loads(resp.body)
        assert out["choices"][0]["message"]["content"] == "done"
        assert wire.worker_hosts() == [gw.HOST_NAME, gw.HOST_NAME], wire.sent
        assert _SMALL_ID not in wire.worker_ids(), "a refused task must reach no box"
        by_task = {r["task"]: r for r in wire.results()}
        assert by_task[0]["ok"] and by_task[2]["ok"], by_task
        assert not by_task[1]["ok"], by_task
        assert "cap" in str(by_task[1]["result"]), by_task[1]
        assert (out["x_team"]["subagents"], out["x_team"]["subagent_failures"]) == (3, 1)

    @pytest.mark.asyncio
    async def test_the_engaged_nodes_are_counted_across_every_round_of_one_request(
            self, client, wire):
        key = _key_row("cap-two")
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.rotate("fast", ["peer1", "peer2", "peer3", "peer4"], _FAST_ID)
        wire.script = [_spawn("call-1", [{"prompt": "r1a"}, {"prompt": "r1b"}]),
                       _spawn("call-2", [{"prompt": "r2a"}, {"prompt": "r2b"}]),
                       _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=2))
        assert resp.status_code == 200, resp.body
        hosts = wire.worker_hosts()
        assert len(hosts) == 4, hosts
        assert len(_remote(hosts)) <= 2, hosts
        assert all(r.get("ok") for r in wire.results()), wire.results()

    @pytest.mark.asyncio
    async def test_an_ordinary_teams_tab_team_gets_no_gate_and_spreads_as_before(
            self, client, wire, monkeypatch):
        key = _key_row("tab-team")
        real = gw.fleet_chat
        worker_kw: list[dict] = []

        async def spy(k, body, endpoint, ctx_limit=None, **kw):
            if endpoint == "/v1/team-worker":
                worker_kw.append(dict(kw))
            return await real(k, body, endpoint, ctx_limit, **kw)

        monkeypatch.setattr(gw, "fleet_chat", spy)
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.rotate("fast", ["peer1", "peer2", "peer3", ""], _FAST_ID)
        wire.script = [_spawn("call-1", [{"prompt": "t%d" % i} for i in range(4)]),
                       _round("done")]
        team = _form_team(int(key["id"]), cap=None, cls="", roster=["fast"], primary=_PRIMARY)
        resp = await _orchestrate(key, team)
        assert resp.status_code == 200, resp.body
        assert set(wire.worker_hosts()) == {"peer1", "peer2", "peer3", gw.HOST_NAME}
        assert len(worker_kw) == 4
        assert not any(callable(v) for kw in worker_kw for v in kw.values()), worker_kw

    @pytest.mark.asyncio
    async def test_the_primary_is_not_capped_even_at_zero(self, client, wire):
        key = _key_row("primary-free")
        wire.route(_PRIMARY, [("peer1", _PRIMARY)])
        wire.route("fast", [("", _FAST_ID)])
        wire.script = [_spawn("call-1", [{"prompt": "a"}]), _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=0))
        assert resp.status_code == 200, resp.body
        assert wire.primary_hosts() == ["peer1", "peer1"]
        assert json.loads(resp.body)["x_team"]["primary_host"] == "peer1"


# ---- the failover deadline follows the gate -------------------------------------

async def _worker_call(key: dict, gate):
    """One worker completion for the class `fast`, straight through fleet_chat."""
    return await gw.fleet_chat(
        key, {"model": "fast", "messages": list(_MSGS)}, "/v1/team-worker", None,
        role="worker", host_gate=gate)


class TestFailoverDeadline:
    """fleet_chat gives a box a stall deadline only while ANOTHER box could take
    the request. A box the gate will refuse is not one: timing out the only
    usable box of a capped worker cools it down for every other key too."""

    @pytest.mark.asyncio
    async def test_cap_zero_with_a_peer_listed_after_this_node_sets_no_deadline(
            self, client, wire):
        key = _key_row("deadline-zero")
        wire.route("fast", [("", _FAST_ID), ("peer1", _FAST_ID)])
        status, _body, host, _granted = await _worker_call(key, gw._worker_node_gate(0, set()))
        assert (status, host) == (200, gw.HOST_NAME)
        assert wire.timeouts == [(gw.HOST_NAME, None)], wire.timeouts

    @pytest.mark.asyncio
    async def test_the_same_candidates_without_a_gate_still_get_the_deadline(
            self, client, wire):
        # The control: it is the gate, not the candidate list, that removes it.
        key = _key_row("deadline-ungated")
        wire.route("fast", [("", _FAST_ID), ("peer1", _FAST_ID)])
        await _worker_call(key, None)
        (host, deadline), = wire.timeouts
        assert host == gw.HOST_NAME and deadline is not None and deadline > 0

    @pytest.mark.asyncio
    async def test_a_cap_whose_slot_this_box_takes_leaves_the_next_box_no_room(
            self, client, wire):
        key = _key_row("deadline-one")
        wire.route("fast", [("peer1", _FAST_ID), ("peer2", _FAST_ID)])
        await _worker_call(key, gw._worker_node_gate(1, set()))
        assert wire.timeouts == [("peer1", None)], wire.timeouts

    @pytest.mark.asyncio
    async def test_a_cap_with_a_slot_to_spare_keeps_the_deadline(self, client, wire):
        key = _key_row("deadline-two")
        wire.route("fast", [("peer1", _FAST_ID), ("peer2", _FAST_ID)])
        await _worker_call(key, gw._worker_node_gate(2, set()))
        (host, deadline), = wire.timeouts
        assert host == "peer1" and deadline is not None

    @pytest.mark.asyncio
    async def test_looking_at_the_next_box_takes_nothing_from_the_cap(self, client, wire):
        key = _key_row("deadline-peek")
        engaged: set[str] = set()
        wire.route("fast", [("peer1", _FAST_ID), ("peer2", _FAST_ID)])
        await _worker_call(key, gw._worker_node_gate(2, engaged))
        assert engaged == {"peer1"}, engaged

    @pytest.mark.asyncio
    async def test_a_5xx_from_the_only_admitted_box_is_its_own_answer_not_a_502(
            self, client, wire, monkeypatch):
        monkeypatch.setattr(gw, "_host_cooldown", {})   # the 503 cools the box down
        key = _key_row("busy-local")
        wire.route("fast", [("", _FAST_ID), ("peer1", _FAST_ID)])
        wire.fail[gw.HOST_NAME] = 503
        status, body, host, _granted = await _worker_call(key, gw._worker_node_gate(0, set()))
        assert (status, host) == (503, gw.HOST_NAME), (status, body)
        assert [h for h, _t in wire.timeouts] == [gw.HOST_NAME], "the refused peer was called"


# ---- the worker class and the key's model list ---------------------------------

class TestWorkerClassAndModelList:
    @pytest.mark.asyncio
    async def test_the_forms_class_runs_although_the_model_list_does_not_name_it(
            self, client, wire):
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.route("fast", [("", _SMALL_ID)])
        wire.script = [_spawn("call-1", [{"prompt": "a"}, {"prompt": "b"}]), _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=2))
        assert resp.status_code == 200, resp.body
        assert wire.worker_ids() == [_SMALL_ID, _SMALL_ID]
        assert all(r.get("ok") for r in wire.results()), wire.results()

    @pytest.mark.asyncio
    async def test_a_primary_off_the_list_is_still_refused_before_any_round(
            self, client, wire):
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route("fast", [("", _SMALL_ID)])
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=2), model=_FAST_ID)
        assert resp.status_code == 403, resp.body
        assert json.loads(resp.body)["error"]["code"] == "model_not_allowed"
        assert wire.sent == []

    @pytest.mark.asyncio
    async def test_a_roster_entry_other_than_the_class_is_still_dropped(
            self, client, wire):
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.route("fast", [("", _SMALL_ID)])
        wire.route("gemma4-31b-qat", [("", "gemma4-31b-qat")])
        # Added to the form team's roster by hand; the key's list never named it.
        team = _form_team(int(key["id"]), cap=2, roster=["fast", "gemma4-31b-qat"])
        wire.script = [_spawn("call-1", [{"prompt": "a", "model": "gemma4-31b-qat"},
                                         {"prompt": "b"}]),
                       _round("done")]
        resp = await _orchestrate(key, team)
        assert resp.status_code == 200, resp.body
        sysline = wire.primary_bodies[0]["messages"][0]["content"]
        assert "gemma4-31b-qat" not in sysline
        assert wire.worker_ids() == [_SMALL_ID, _SMALL_ID]
        assert all(r.get("ok") for r in wire.results()), wire.results()

    @pytest.mark.asyncio
    async def test_a_class_of_the_operators_own_is_exempt_while_it_exists(
            self, client, wire):
        gw.set_role_config({"classes": {"writers": {"models": [_SMALL_ID], "nodes": []}}})
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.route("writers", [("", _SMALL_ID)])
        wire.script = [_spawn("call-1", [{"prompt": "a"}]), _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=2, cls="writers"))
        assert resp.status_code == 200, resp.body
        assert wire.worker_ids() == [_SMALL_ID]

    @pytest.mark.asyncio
    async def test_a_class_deleted_since_is_no_longer_exempt_from_the_list(
            self, client, wire):
        # The word is no class any more, so it is just a name -- and a box
        # serving a model or alias of that name would be handed the work.
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route(_PRIMARY, [("", _PRIMARY)])
        wire.route("gone-class", [("peer1", "gone-class")])
        wire.script = [_spawn("call-1", [{"prompt": "a"}]), _round("done")]
        resp = await _orchestrate(key, _form_team(int(key["id"]), cap=2, cls="gone-class"))
        assert resp.status_code == 403, resp.body
        assert json.loads(resp.body)["error"]["code"] == "model_not_allowed"
        assert wire.sent == []

    @pytest.mark.asyncio
    async def test_a_worker_asked_to_run_as_a_deleted_class_is_refused_at_the_call(
            self, client, wire):
        # A roster team_orchestrate never filtered: the check at the point of
        # the side effect.
        key = _key_row("listed", allowed=[_PRIMARY])
        wire.route("gone-class", [("peer1", "gone-class")])
        team = _form_team(int(key["id"]), cap=2, cls="gone-class")
        (res,) = await gw._run_subagents(key, team, [{"prompt": "a"}])
        assert res["ok"] is False and "not allowed" in res["result"], res
        assert wire.sent == []

    @pytest.mark.asyncio
    async def test_a_routing_failure_for_the_class_worker_is_a_404_not_this_boxs_alias_row(
            self, client, wire, monkeypatch):
        # The exemption skips the key's list for what the class routes to. If
        # routing itself fails, falling back to "this box, by the word" would
        # hand the word to a local alias row of that name -- a model the list
        # never named -- with nothing left to filter it.
        async def broken(model, **kw):
            raise RuntimeError("scorer down")

        monkeypatch.setattr(gw, "resolve_targets", broken)
        key = _key_row("listed", allowed=[_PRIMARY])
        status, resp, _host, _granted = await gw.fleet_chat(
            key, {"model": "fast", "messages": list(_MSGS)}, "/v1/team-worker",
            role="worker", list_exempt=True)
        assert status == 404 and "no host" in resp["error"]["message"], (status, resp)
        (res,) = await gw._run_subagents(key, _form_team(int(key["id"]), cap=2),
                                         [{"prompt": "a"}])
        assert res["ok"] is False, res
        assert wire.sent == []

    @pytest.mark.asyncio
    async def test_a_routing_failure_for_a_plain_model_still_falls_back_to_this_box_through_the_list(
            self, client, wire, monkeypatch):
        async def broken(model, **kw):
            raise RuntimeError("scorer down")

        monkeypatch.setattr(gw, "resolve_targets", broken)
        key = _key_row("listed", allowed=[_PRIMARY])
        status, _resp, host, _granted = await gw.fleet_chat(
            key, {"model": _PRIMARY, "messages": list(_MSGS)}, "/v1/team-worker",
            role="worker")
        assert status == 200 and host == gw.HOST_NAME
        assert wire.sent == [(gw.HOST_NAME, _PRIMARY, False)]
        wire.sent.clear()
        status, resp, _host, _granted = await gw.fleet_chat(
            key, {"model": _SMALL_ID, "messages": list(_MSGS)}, "/v1/team-worker",
            role="worker")
        assert status == 403 and resp["error"]["code"] == "model_not_allowed"
        assert wire.sent == []


# ---- the Teams tab -------------------------------------------------------------

class TestTeamsTabBlankPrimary:
    """Only a team the keys form made may lead with the client's own model. On
    any other team (Fleet Pass included) a blank primary would let the key's
    holder pick any fleet model, so it stays a 400."""

    def test_a_keys_form_team_may_lead_with_the_clients_model(self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={"name": "crew", "primary_model": "", "worker_models": ["fast"]})
        assert r.status_code == 200, r.text
        assert r.json()["primary_model"] == ""

    def test_a_blank_primary_on_a_keys_form_team_keeps_every_other_check(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={"name": "crew", "primary_model": "", "worker_models": "fast"})
        assert r.status_code == 400, r.text

    def test_a_teams_tab_team_still_needs_a_primary_model(self, client, admin_headers):
        raw, meta = gw.mint_key("tab-blank")
        kid = int(meta["id"])
        _tab_team(kid)
        r = client.put("/admin/api/teams/" + str(kid), headers=admin_headers,
                       json={"name": "crew", "primary_model": "", "worker_models": ["fast"]})
        assert r.status_code == 400, r.text
        assert "primary model" in str(r.json().get("detail")), r.text
        assert (_team_of(kid) or {}).get("primary_model") == "qwen3.5-4b"

    def test_a_key_with_no_team_yet_cannot_be_given_a_blank_one(self, client, admin_headers):
        raw, meta = gw.mint_key("tab-new-blank")
        r = client.put("/admin/api/teams/" + str(meta["id"]), headers=admin_headers,
                       json={"name": "crew", "primary_model": "", "worker_models": ["fast"]})
        assert r.status_code == 400, r.text
        assert _team_of(int(meta["id"])) is None

    def test_a_new_team_needs_a_primary_model_and_mints_no_key(self, client, admin_headers):
        before = _key_count()
        r = client.post("/admin/api/teams", headers=admin_headers,
                        json={"name": "crew", "primary_model": "", "worker_models": ["fast"]})
        assert r.status_code == 400, r.text
        assert _key_count() == before


# ---- a class this node is not part of -------------------------------------------

def _fast_only_on_peer1() -> None:
    """The Routing tab limits `fast` to peer1: this node may not play it."""
    gw.set_role_config({"classes": {"fast": {"models": [], "nodes": ["peer1"]}}})
    assert gw.HOST_NAME.lower() not in gw.role_nodes("fast")


class TestCapZeroNeedsThisNode:
    """Cap 0 is "this node only". With the class limited to other nodes that is
    a key whose every worker task is refused, so it is refused at save."""

    def _post(self, client, admin_headers, cap, cls="fast"):
        return client.post("/admin/api/keys", headers=admin_headers, json={
            "name": "capped", "max_worker_nodes": cap, "worker_class": cls})

    def test_a_new_key_capped_at_zero_on_a_class_that_leaves_this_node_out_is_a_400(
            self, client, admin_headers):
        _fast_only_on_peer1()
        before = _key_count()
        r = self._post(client, admin_headers, 0)
        assert r.status_code == 400, r.text
        detail = str(r.json().get("detail"))
        assert "fast" in detail and "this node" in detail, detail
        assert _key_count() == before and _team_count() == 0

    def test_a_cap_that_allows_other_nodes_or_a_class_that_includes_this_one_is_saved(
            self, client, admin_headers):
        _fast_only_on_peer1()
        assert self._post(client, admin_headers, 1).status_code == 200
        gw.set_role_config({"classes": {"fast": {"models": [], "nodes": [
            "peer1", gw.HOST_NAME.lower()]}}})
        assert self._post(client, admin_headers, 0).status_code == 200

    def test_patching_a_key_to_cap_zero_on_such_a_class_is_a_400_and_changes_nothing(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        _fast_only_on_peer1()
        r = _patch(client, admin_headers, kid, {"name": "renamed", "max_worker_nodes": 0})
        assert r.status_code == 400, r.text
        assert "this node" in str(r.json().get("detail")), r.text
        assert (_team_of(kid) or {}).get("max_worker_nodes") == 1
        assert gw.db_query("SELECT name FROM api_keys WHERE id=?", (kid,))[0]["name"] != "renamed"

    def test_switching_workers_off_or_moving_class_is_not_held_to_it(
            self, client, admin_headers):
        raw, kid = _mint_workers(client, admin_headers, 1)
        _fast_only_on_peer1()
        assert _patch(client, admin_headers, kid, {"max_worker_nodes": None}).status_code == 200
        r = _patch(client, admin_headers, kid, {"max_worker_nodes": 0, "worker_class": "small"})
        assert r.status_code == 200, r.text
