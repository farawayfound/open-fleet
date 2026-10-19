"""Per-key model access: api_keys.allowed_models.

A key can be limited to a list of models and routing classes. [] means every
model, which is what every key minted before the column existed reads as. A
class word on the list grants its whole ladder; a class word NOT on the list
is still callable when its ladder overlaps the list, but routes only to the
allowed members. A stored list nobody can read refuses everything, and so
does a POST whose model cannot be read off the body: an access list that
fails open is not an access list.

Enforcement is on the gateway that authenticated the client -- keys live in
each node's own database, and a peer only ever sees the hub's own key -- so
every surface that takes a model is pinned here: /v1 (team keys included),
the native /api, batches (and their resume at boot), the listings, and the
fallback substitution that could otherwise swap an allowed model for one off
the list.

Style follows test_audit_fixes.py (TestFallbackRespectsAllowedModels,
TestNativeProxyAllowList), test_public.py (the scratch-DB migration test),
test_fleet_roles.py (_routes) and test_audit_fixes2.py (team fleet_chat
fakes).
"""
from __future__ import annotations

import inspect
import json
import logging
import sqlite3
import threading

import httpx
import pytest

import app as gw

_MSGS = [{"role": "user", "content": "hi"}]
_NO_FALLBACK = {"X-Fleet-No-Fallback": "1"}

_CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
_NATIVE_OK = {"message": {"role": "assistant", "content": "ok"},
              "done": True, "prompt_eval_count": 1, "eval_count": 1}
# What this box's Ollama would list: one tag the test keys may use (the
# engine's own spelling of qwen3.5-4b) and one they may not.
_NATIVE_LISTING = {"models": [{"name": "qwen3.5:4b", "model": "qwen3.5:4b"},
                              {"name": "qwen3.8-27b", "model": "qwen3.8-27b"}]}


# ---- fixtures and helpers ------------------------------------------------------

@pytest.fixture(autouse=True)
def _routes_cache_restored():
    """Several tests install routing tables by hand (t=9e9), which would
    otherwise outlive them. Peer aliases are cleared for every test: a stale
    (host, name) -> canonical row left by the live lifespan would widen what
    a list entry matches, and these tests pin exactly what it matches."""
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
        path = request.url.path
        sent.append(path)
        if path in ("/api/tags", "/api/ps"):
            return httpx.Response(200, request=request, json=_NATIVE_LISTING)
        if path.startswith("/api/"):
            return httpx.Response(200, request=request, json=_NATIVE_OK)
        return httpx.Response(200, request=request, json=_CHAT_OK_BODY)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return sent


@pytest.fixture
def routed(monkeypatch):
    """resolve_targets replaced by a recorder that sends everything to this
    box: (model, keyword arguments) per call. A class word is answered the way
    the real resolver answers it -- by a member of its ladder (the limited one
    when a key's list reaches only part of it), never by the word itself, which
    no box serves."""
    calls: list[tuple[str, dict]] = []

    async def fake_resolve(model, **kw):
        calls.append((model, kw))
        if model in gw.FLEET_ROLES:
            return [("", (kw.get("ladder") or gw.FLEET_ROLES[model])[0])]
        return [("", model)]

    monkeypatch.setattr(gw, "resolve_targets", fake_resolve)
    return calls


@pytest.fixture
def sent_requests(monkeypatch):
    """Every request that leaves the process, answered in-process: (host, path,
    body bytes) in the order sent."""
    sent: list[tuple[str, str, bytes]] = []

    async def fake_send(self, request, stream=False, **kw):
        sent.append((request.url.host, request.url.path, request.content))
        if request.url.path.startswith("/api/"):
            return httpx.Response(200, request=request, json=_NATIVE_OK)
        return httpx.Response(200, request=request, json=_CHAT_OK_BODY)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return sent


@pytest.fixture
def fleet_wire(monkeypatch, sent_requests):
    """sent_requests, plus the hub's own key on every peer: what a routed
    request needs to leave for a peer. Returns the same list."""
    async def fake_key(peer):
        return "hub-key-on-" + peer

    monkeypatch.setattr(gw, "peer_inference_key", fake_key)
    return sent_requests


def _chat_hosts(sent) -> list[str]:
    return [host for host, path, _body in sent if path == "/v1/chat/completions"]


@pytest.fixture
def paired(client):
    """A catalogue row serving two DIFFERENT weights under one public id (the
    catalogue is admin-editable, so nothing stops that): pair-pub -> pair-a,
    pair-b. A request for either fleet id is routed over both."""
    gw.db_exec("INSERT INTO public_models(public_id,name,fleet_ids,enabled) "
               "VALUES (?,?,?,1)", ("pair-pub", "Pair", json.dumps(["pair-a", "pair-b"])))
    gw._public_catalogue_cache.update(t=0.0)
    yield
    gw.db_exec("DELETE FROM public_models WHERE public_id=?", ("pair-pub",))
    gw._public_catalogue_cache.update(t=0.0)


@pytest.fixture
def batch_runs(monkeypatch):
    """The batch dispatcher replaced by a recorder, so nothing a test queues
    ever dispatches. Recorded when the coroutine is CREATED, which is the
    moment batch_submit/resume_orphaned_batches decide to run a batch."""
    started: list[int] = []

    async def _idle():
        return None

    def fake_run(bid, *a, **kw):
        started.append(int(bid))
        return _idle()

    monkeypatch.setattr(gw, "_batch_run", fake_run)
    return started


def _auth(raw: str, **extra: str) -> dict:
    return {"Authorization": "Bearer " + raw, **extra}


def _mint(client, admin_headers, models: list[str], name: str = "key-access") -> tuple[str, int]:
    """A key minted through the admin API with `models` as its list.

    The echo is asserted here because it is the precondition of every gate
    test: a gateway that ignored the field would mint an unrestricted key,
    and a refusal test against that key would be testing nothing."""
    r = client.post("/admin/api/keys", headers=admin_headers,
                    json={"name": name, "allowed_models": models})
    assert r.status_code == 200, r.status_code
    body = r.json()
    # Compared through a local, never the whole body: assertion introspection
    # would otherwise print the raw key into the test log.
    echo, fields = body.get("allowed_models"), ", ".join(sorted(body))
    assert echo == models, "POST /admin/api/keys did not echo allowed_models; fields: " + fields
    return body["key"], int(body["id"])


def _listed(client, admin_headers, kid: int) -> dict:
    items = client.get("/admin/api/keys", headers=admin_headers,
                       params={"limit": 200}).json()["items"]
    return next(k for k in items if k["id"] == kid)


def _set_stored(kid: int, raw_value: str) -> None:
    cols = {r["name"] for r in gw.db_query("PRAGMA table_info(api_keys)")}
    assert "allowed_models" in cols, "api_keys has no allowed_models column"
    gw.db_exec("UPDATE api_keys SET allowed_models=? WHERE id=?", (raw_value, kid))


def _usage(kid: int) -> list[dict]:
    return gw.db_query("SELECT model, endpoint, status FROM usage WHERE key_id=?", (kid,))


def _key_count() -> int:
    return gw.db_query("SELECT COUNT(*) c FROM api_keys")[0]["c"]


def _batches_for(kid: int) -> list[dict]:
    return gw.db_query("SELECT id, status FROM batches WHERE key_id=?", (kid,))


def _need(name: str):
    fn = getattr(gw, name, None)
    assert fn is not None, "app has no " + name + " yet"
    return fn


def _accepts(fn, kw: str) -> None:
    assert kw in inspect.signature(fn).parameters, (
        fn.__name__ + " takes no " + kw + "= keyword yet")


def _assert_refused(r, model: str | None) -> None:
    """The gate's 403: the flat OpenAI error shape an SDK can read, naming
    the refused model when there is one model to name."""
    assert r.status_code == 403, r.text
    err = r.json().get("error") or {}
    assert err.get("code") == "model_not_allowed", r.text
    assert err.get("type") == "invalid_request_error", r.text
    if model is not None:
        assert model in str(err.get("message")), r.text


def _roles_reaching(*models: str) -> set[str]:
    return {role for role, ladder in gw.FLEET_ROLES.items()
            if any(m in ladder for m in models)}


def _routes(cands, alias, meta, running=None):
    gw._routes_cache.update(t=9e9, map={}, cands=cands, alias=alias, meta=meta,
                            cap={}, running=running or {},
                            reachable={h for hs in cands.values() for h in hs if h})


def _ident(*pairs):
    return {(h, m): m for h, m in pairs}


def _team(kid: int, primary: str, workers: list[str]) -> None:
    gw.db_exec(
        "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,"
        "max_workers,max_rounds,system_prompt,worker_prompt,updated_at) "
        "VALUES (?,1,?,?,?,?,?,?,?,?)",
        (kid, "crew", primary, json.dumps(workers), 2, 3, "", "", gw.now()))


def _agent(kid: int, allowed: list[str]) -> None:
    gw.db_exec(
        "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
        "param_overrides,updated_at) VALUES (?,1,?,?,?,?,?)",
        (kid, "restricted", json.dumps(allowed), "", "{}", gw.now()))


def _round(content="done", finish="stop", tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"id": "chatcmpl-x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


# ---- the column ----------------------------------------------------------------

class TestMigration:
    def _pre_migration_schema(self, path) -> None:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                "CREATE TABLE api_keys ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,"
                "key_hash TEXT NOT NULL UNIQUE, prefix TEXT NOT NULL,"
                "created_at TEXT NOT NULL, last_used_at TEXT,"
                "disabled INTEGER NOT NULL DEFAULT 0)")
            conn.execute(
                "INSERT INTO api_keys(name,key_hash,prefix,created_at) "
                "VALUES ('old-key','h-old','p-old','2026-01-01T00:00:00Z')")
            conn.commit()
        finally:
            conn.close()

    def test_a_pre_migration_table_gains_the_column_and_old_keys_read_as_every_model(
            self, tmp_path, monkeypatch):
        scratch = tmp_path / "pre-migration.db"
        self._pre_migration_schema(scratch)
        monkeypatch.setattr(gw, "DB_PATH", scratch)

        gw.db_init()

        cols = {r["name"]: r for r in gw.db_query("PRAGMA table_info(api_keys)")}
        assert "allowed_models" in cols, "MIGRATIONS must add allowed_models to an old DB"
        assert cols["allowed_models"]["notnull"] == 1
        assert cols["allowed_models"]["dflt_value"] == "'[]'"
        conn = sqlite3.connect(str(scratch))
        try:
            row = conn.execute(
                "SELECT allowed_models FROM api_keys WHERE name='old-key'").fetchone()
        finally:
            conn.close()
        assert row == ("[]",)


# ---- minting, listing, editing -------------------------------------------------

class TestMintAndList:
    def test_a_key_minted_without_a_list_echoes_an_empty_one_and_routes_any_model(
            self, client, admin_headers, fake_fleet, routed, upstream):
        r = client.post("/admin/api/keys", headers=admin_headers, json={"name": "plain"})
        assert r.status_code == 200
        raw, echo = r.json()["key"], r.json().get("allowed_models", "<missing>")
        assert echo == []

        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert [m for m, _ in routed] == ["qwen3.8-27b"]

    def test_a_minted_list_is_echoed_and_listed_back_as_a_list(self, client, admin_headers):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b", "fast"])
        item = _listed(client, admin_headers, kid)
        assert item.get("allowed_models") == ["qwen3.5-4b", "fast"]
        assert not item.get("allowed_models_invalid")

    def test_the_hub_routing_key_a_peer_mints_stays_unrestricted(
            self, client, admin_headers, fake_fleet, routed, upstream):
        # peer_inference_key's exact request: a name and nothing else.
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "hub-" + gw.HOST_NAME})
        assert r.status_code == 200
        raw, echo = r.json()["key"], r.json().get("allowed_models", "<missing>")
        assert echo == []
        for model in ("qwen3.8-27b", "fast", "gemma4-31b-qat"):
            r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                            json={"model": model, "messages": _MSGS})
            assert r.status_code == 200, (model, r.text)
        assert [m for m, _ in routed] == ["qwen3.8-27b", "fast", "gemma4-31b-qat"]


class TestValidation:
    @pytest.mark.parametrize("value", [
        pytest.param("qwen3.5-4b", id="a-string-not-a-list"),
        pytest.param({"qwen3.5-4b": True}, id="an-object"),
        pytest.param(["qwen3.5-4b", 7], id="a-non-string-entry"),
        pytest.param(["qwen 3.5"], id="an-entry-with-a-space"),
        pytest.param(["qwen\t3.5"], id="an-entry-with-a-tab"),
        pytest.param(["qwen\x073.5"], id="an-entry-with-a-control-char"),
        pytest.param(["m" * 129], id="an-entry-over-128-chars"),
        pytest.param(["m%d" % i for i in range(129)], id="129-entries"),
    ])
    def test_a_bad_list_is_a_400_and_mints_nothing(self, client, admin_headers, value):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "bad-list", "allowed_models": value})
        assert r.status_code == 400, r.status_code
        assert _key_count() == 0, "a refused list must not leave a key behind"

    @pytest.mark.parametrize("value, stored", [
        pytest.param(["a", "a", " a "], ["a"], id="duplicates-and-padding-stored-once"),
        pytest.param(["", "  ", "b"], ["b"], id="empty-entries-dropped"),
        pytest.param(None, [], id="null-means-every-model"),
        pytest.param([], [], id="empty-means-every-model"),
        pytest.param(["m" * 128], ["m" * 128], id="a-128-char-entry-is-accepted"),
        pytest.param(["m%d" % i for i in range(128)], ["m%d" % i for i in range(128)],
                     id="128-entries-are-accepted"),
    ])
    def test_an_acceptable_list_is_stored_normalised_and_echoed(
            self, client, admin_headers, value, stored):
        r = client.post("/admin/api/keys", headers=admin_headers,
                        json={"name": "norm-list", "allowed_models": value})
        assert r.status_code == 200, r.status_code
        kid, echo = int(r.json()["id"]), r.json().get("allowed_models", "<missing>")
        assert echo == stored
        assert _listed(client, admin_headers, kid).get("allowed_models") == stored


class TestPatch:
    def test_patching_a_live_key_sets_its_list_and_it_reads_back(self, client, admin_headers):
        raw, meta = gw.mint_key("patch-live")
        r = client.patch("/admin/api/keys/" + str(meta["id"]), headers=admin_headers,
                         json={"allowed_models": ["qwen3.5-4b"]})
        assert r.status_code == 200, r.text
        assert r.json().get("allowed_models") == ["qwen3.5-4b"]
        assert _listed(client, admin_headers, int(meta["id"])).get("allowed_models") == ["qwen3.5-4b"]

    def test_a_patched_list_is_enforced_on_the_next_request(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, meta = gw.mint_key("patch-enforced")
        r = client.patch("/admin/api/keys/" + str(meta["id"]), headers=admin_headers,
                         json={"allowed_models": ["qwen3.5-4b"]})
        assert r.status_code == 200, r.text
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        _assert_refused(r, "qwen3.8-27b")
        assert routed == []

    def test_patching_an_empty_list_lifts_the_restriction(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.patch("/admin/api/keys/" + str(kid), headers=admin_headers,
                         json={"allowed_models": []})
        assert r.status_code == 200, r.text
        assert r.json().get("allowed_models") == []
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        assert r.status_code == 200, r.text

    def test_patching_an_archived_key_is_a_404_and_changes_nothing(self, client, admin_headers):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        r = client.patch("/admin/api/keys/" + str(kid), headers=admin_headers,
                         json={"allowed_models": []})
        assert r.status_code == 404, r.text
        row = gw.db_query("SELECT allowed_models FROM api_keys WHERE id=?", (kid,))[0]
        assert json.loads(row["allowed_models"]) == ["qwen3.5-4b"]

    def test_a_bad_list_in_a_patch_is_a_400_and_saves_none_of_the_patch(
            self, client, admin_headers):
        raw, meta = gw.mint_key("patch-bad")
        r = client.patch("/admin/api/keys/" + str(meta["id"]), headers=admin_headers,
                         json={"name": "renamed", "allowed_models": "qwen3.5-4b"})
        assert r.status_code == 400, r.text
        row = gw.db_query("SELECT name FROM api_keys WHERE id=?", (meta["id"],))[0]
        assert row["name"] == "patch-bad"


class TestUnreadableList:
    @pytest.mark.parametrize("model", ["qwen3.5-4b", "fast", "gemma4-31b-qat"])
    def test_a_corrupt_stored_list_refuses_every_model(
            self, client, admin_headers, fake_fleet, routed, upstream, model):
        raw, meta = gw.mint_key("corrupt-list")
        _set_stored(int(meta["id"]), "{bad")
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": model, "messages": _MSGS})
        _assert_refused(r, model)
        assert routed == []
        assert "/v1/chat/completions" not in upstream

    def test_the_key_list_reports_a_corrupt_list_as_blocked(self, client, admin_headers):
        raw, meta = gw.mint_key("corrupt-listed")
        _set_stored(int(meta["id"]), "{bad")
        item = _listed(client, admin_headers, int(meta["id"]))
        assert item.get("allowed_models") == []
        assert item.get("allowed_models_invalid") is True

    def test_a_request_that_spells_the_unreadable_marker_is_refused_too(
            self, client, admin_headers, fake_fleet, routed, upstream):
        # The marker stands in for the list; a model named like it must not
        # find itself on that list.
        raw, meta = gw.mint_key("corrupt-marker")
        _set_stored(int(meta["id"]), "{bad")
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": gw.KEY_MODELS_UNREADABLE, "messages": _MSGS})
        assert r.status_code == 403, r.text
        assert (r.json().get("error") or {}).get("code") == "model_not_allowed", r.text
        assert routed == []
        assert "/v1/chat/completions" not in upstream

    @pytest.mark.asyncio
    async def test_the_marker_passes_no_gate_listing_or_routing_check(self, client, fake_fleet):
        key = {"id": 0, "name": "unit", "allowed_models": "{bad"}
        ok, _ladder, _why = gw.key_model_gate(key, gw.KEY_MODELS_UNREADABLE)
        assert ok is False
        assert gw.key_filter_models(key, [gw.KEY_MODELS_UNREADABLE, "qwen3.5-4b"]) == []
        assert gw.key_filter_targets(key, [("", gw.KEY_MODELS_UNREADABLE),
                                           ("", "qwen3.5-4b")]) == []

    def test_an_unreadable_list_is_logged_once_per_key_not_once_per_check(self, caplog):
        caplog.set_level(logging.ERROR, logger=gw.log.name)

        def reports(kid: int) -> int:
            return sum(1 for r in caplog.records
                       if r.levelno >= logging.ERROR and "key %d:" % kid in r.getMessage())

        key = {"id": 987001, "name": "unit", "allowed_models": "{bad"}
        for _ in range(4):
            gw.key_model_gate(key, "qwen3.5-4b")
            gw.key_filter_models(key, ["qwen3.5-4b"])
            gw.key_filter_targets(key, [("", "qwen3.5-4b")])
            gw.pin_model_field(key, {"model": "x"})
        assert reports(987001) == 1
        # Another key is reported on its own account ...
        gw.key_model_gate({"id": 987002, "name": "unit", "allowed_models": "{bad"}, "x")
        assert reports(987002) == 1
        # ... and one that was saved and then broke again is reported again.
        gw.key_model_gate({**key, "allowed_models": '["qwen3.5-4b"]'}, "qwen3.5-4b")
        gw.key_model_gate(key, "qwen3.5-4b")
        assert reports(987001) == 2


# ---- what the gate costs -------------------------------------------------------
#
# models.json is read from disk and parsed on every load_models() call, and the
# gate resolves a local alias for every name it compares. It runs on the event
# loop for each chat request, each worker of a team and each /v1/models
# listing, so a read per name is a read per ladder member per check.

@pytest.fixture
def models_reads(monkeypatch):
    """The models.json reads this thread makes, counted (the session client's
    background loops read it on their own schedule, from other threads)."""
    me = threading.get_ident()
    reads: list[int] = []
    real = gw.load_models

    def counting():
        if threading.get_ident() == me:
            reads.append(1)
        return real()

    monkeypatch.setattr(gw, "load_models", counting)
    return reads


class TestGateCost:
    @pytest.mark.parametrize("model", ["fast", "quality", "qwen3.8-27b", "deep"])
    def test_a_gate_decision_reads_the_registry_at_most_once(self, client, models_reads, model):
        key = _key_row(["fast", "deep"])
        gw.key_model_gate(key, model)
        assert len(models_reads) <= 1, "%d reads for one decision" % len(models_reads)

    def test_a_listing_reads_the_registry_at_most_once(self, client, models_reads):
        key = _key_row(["fast", "deep"])
        names = sorted({m for ladder in gw.FLEET_ROLES.values() for m in ladder}
                       | set(gw.FLEET_ROLES))
        assert len(names) > 30
        kept = gw.key_filter_models(key, names)
        assert "fast" in kept and "qwen3.5-4b" in kept     # it still filters
        assert len(models_reads) <= 1, "%d reads for one listing" % len(models_reads)

    def test_filtering_routes_reads_the_registry_at_most_once(self, client, models_reads):
        key = _key_row(["fast", "deep"])
        pairs = [(h, m) for h in ("", "peer1", "peer2")
                 for m in gw.FLEET_ROLES["fast"] + gw.FLEET_ROLES["quality"]]
        assert gw.key_filter_targets(key, pairs)
        assert len(models_reads) <= 1, "%d reads for one routing decision" % len(models_reads)

    def test_the_cheaper_reading_decides_the_same_way(self, client, local_rows):
        # A local alias is still resolved, from the one read: this box's row
        # "house-pick" is the 4b, so a list naming the 4b reaches it.
        local_rows(("qwen3.5-4b", "house-pick"))
        key = _key_row(["qwen3.5-4b"])
        assert gw.key_model_gate(key, "house-pick")[0] is True
        assert gw.key_model_gate(_key_row(["qwen3.8-27b"]), "house-pick")[0] is False
        assert gw.key_filter_targets(key, [("", "house-pick")]) == [("", "house-pick")]


# ---- the gate itself -----------------------------------------------------------

_GATE_TABLE = [
    # (stored list, requested model, allowed?, ladder routing is limited to)
    pytest.param([], "qwen3.8-27b", True, None, id="empty-list-allows-anything"),
    pytest.param(["qwen3.5-4b"], "qwen3.5-4b", True, None, id="listed-model"),
    pytest.param(["qwen3.5-4b"], "qwen3.8-27b", False, None, id="unlisted-model"),
    pytest.param(["qwen3.8-9b-distill"], "qwen3.8-9B", True, None, id="old-spelling-of-a-listed-id"),
    pytest.param(["qwen3.8-9B"], "qwen3.8-9b-distill", True, None, id="canonical-id-of-a-listed-spelling"),
    pytest.param(["qwen3.5-4b"], "qwen3.5:4b", True, None, id="ollama-tag-of-a-listed-id"),
    pytest.param(["fast"], "fast", True, None, id="listed-class-gets-its-whole-ladder"),
    pytest.param(["qwen3.5-4b"], "fast", True, ("qwen3.5-4b",), id="unlisted-class-limited-to-overlap"),
    pytest.param(["qwen3.5-4b", "qwen3.6-35b"], "fast", True, ("qwen3.6-35b", "qwen3.5-4b"),
                 id="limited-ladder-keeps-the-class-order"),
    pytest.param(["qwen3.8-27b"], "small", False, None, id="class-with-no-overlap"),
    pytest.param(["fast"], "qwen3.5-4b", True, None, id="listed-class-grants-a-member-by-name"),
    pytest.param(["fast"], "qwen3.8-27b", False, None, id="listed-class-grants-nothing-off-its-ladder"),
    pytest.param(["qwen3.6-35b"], "qwen3.6-35b-a3b", True, None, id="catalogue-id-whose-fleet-ids-are-listed"),
    pytest.param(["qwen3.6-35b-a3b"], "qwen3.6-35b-a3b", True, None, id="listed-catalogue-id"),
    pytest.param(["qwen3.8-27b"], "qwen3.6-35b-a3b", False, None, id="catalogue-id-whose-fleet-ids-are-not"),
    pytest.param(["qwen3.5-4b"], "", False, None, id="no-model-named"),
]


async def _gate(stored: str, model: str):
    gate = _need("key_model_gate")
    out = gate({"id": 0, "name": "unit", "allowed_models": stored}, model)
    if inspect.isawaitable(out):
        out = await out
    return out


class TestKeyModelGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("allowed, model, ok, ladder", _GATE_TABLE)
    async def test_the_gate_decides_by_canonical_names_and_ladders(
            self, client, fake_fleet, allowed, model, ok, ladder):
        got_ok, got_ladder, reason = await _gate(json.dumps(allowed), model)
        assert got_ok is ok, (allowed, model, reason)
        if ok:
            assert got_ladder == ladder
        else:
            assert reason, "a refusal must say why"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stored", [
        pytest.param("{bad", id="not-json"),
        pytest.param('"qwen3.5-4b"', id="json-but-not-a-list"),
        pytest.param('["qwen3.5-4b", 7]', id="a-list-with-a-non-string"),
    ])
    async def test_an_unreadable_stored_list_refuses_even_a_model_it_seems_to_name(
            self, client, fake_fleet, stored):
        got_ok, _ladder, reason = await _gate(stored, "qwen3.5-4b")
        assert got_ok is False
        assert reason


class TestALimitedLadder:
    """Routing a class through only part of its ladder. The limit has to be
    applied BEFORE role_pairs picks one model per box: filtering afterwards
    would drop a box whose best ladder model is off the list even though it
    holds an allowed lower one."""

    def _two_boxes(self):
        _routes(
            {"qwen3.6-35b": ["gpu-laptop-1"], "qwen3.5-4b": ["gpu-laptop-1", "gpu-desktop-1"]},
            _ident(("gpu-laptop-1", "qwen3.6-35b"), ("gpu-laptop-1", "qwen3.5-4b"),
                   ("gpu-desktop-1", "qwen3.5-4b")),
            {("gpu-laptop-1", "qwen3.6-35b"): {"fit": "vram"},
             ("gpu-laptop-1", "qwen3.5-4b"): {"fit": "vram"},
             ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
        )

    def test_role_pairs_offers_each_box_its_best_allowed_member(self):
        self._two_boxes()
        assert dict(gw.role_pairs("fast")) == {"gpu-laptop-1": "qwen3.6-35b", "gpu-desktop-1": "qwen3.5-4b"}
        _accepts(gw.role_pairs, "ladder")
        assert dict(gw.role_pairs("fast", ladder=("qwen3.5-4b",))) == {
            "gpu-laptop-1": "qwen3.5-4b", "gpu-desktop-1": "qwen3.5-4b"}

    def test_role_pairs_with_a_ladder_no_box_holds_offers_nothing(self):
        self._two_boxes()
        _accepts(gw.role_pairs, "ladder")
        assert gw.role_pairs("fast", ladder=("nemotron-mini:4b",)) == []

    @pytest.mark.asyncio
    async def test_resolve_targets_hands_the_ladder_to_the_class_branch(self):
        self._two_boxes()
        _accepts(gw.resolve_targets, "ladder")
        got = await gw.resolve_targets("fast", ladder=("qwen3.5-4b",))
        assert set(got) == {("gpu-laptop-1", "qwen3.5-4b"), ("gpu-desktop-1", "qwen3.5-4b")}


# ---- /v1 -----------------------------------------------------------------------

class TestChatGate:
    def test_a_listed_model_passes_the_gate(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.5-4b", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert [m for m, _ in routed] == ["qwen3.5-4b"]

    def test_an_unlisted_model_is_refused_metered_and_never_routed(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        _assert_refused(r, "qwen3.8-27b")
        assert routed == []
        assert "/v1/chat/completions" not in upstream
        assert {"model": "qwen3.8-27b", "endpoint": "/v1/chat/completions",
                "status": 403} in _usage(kid)

    @pytest.mark.parametrize("allowed, model", [
        pytest.param(["qwen3.8-9b-distill"], "qwen3.8-9B", id="old-spelling-of-a-listed-id"),
        pytest.param(["qwen3.8-9B"], "qwen3.8-9b-distill", id="canonical-id-of-a-listed-spelling"),
        pytest.param(["qwen3.5-4b"], "qwen3.5:4b", id="ollama-tag-of-a-listed-id"),
    ])
    def test_another_spelling_of_a_listed_model_passes(
            self, client, admin_headers, fake_fleet, routed, upstream, allowed, model):
        raw, kid = _mint(client, admin_headers, allowed)
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": model, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert [m for m, _ in routed] == [model]

    def test_a_listed_class_routes_over_its_whole_ladder(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["fast"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "fast", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert len(routed) == 1 and routed[0][0] == "fast"
        assert routed[0][1].get("ladder") is None

    def test_an_unlisted_class_routes_only_to_its_allowed_members(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "fast", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert len(routed) == 1 and routed[0][0] == "fast"
        assert routed[0][1].get("ladder") == ("qwen3.5-4b",)

    def test_a_class_with_no_allowed_member_is_refused(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.8-27b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "small", "messages": _MSGS})
        _assert_refused(r, "small")
        assert routed == []

    @pytest.mark.parametrize("path, kwargs", [
        pytest.param("/v1/audio/transcriptions",
                     {"files": {"file": ("a.wav", b"RIFF0000", "audio/wav")},
                      "data": {"model": "qwen3.5-4b"}}, id="multipart-form"),
        pytest.param("/v1/chat/completions",
                     {"content": b'{"model": "qwen3.5-4b", "messages": [',
                      "headers": {"content-type": "application/json"}}, id="malformed-json"),
        pytest.param("/v1/chat/completions",
                     {"json": [{"model": "qwen3.5-4b", "messages": _MSGS}]}, id="json-array"),
        pytest.param("/v1/chat/completions", {"json": {"messages": _MSGS}}, id="no-model-field"),
        pytest.param("/v1/chat/completions", {"json": {"model": "", "messages": _MSGS}},
                     id="empty-model"),
        pytest.param("/v1/chat/completions", {}, id="empty-body"),
    ])
    def test_a_post_whose_model_cannot_be_read_is_refused_for_a_restricted_key(
            self, client, admin_headers, fake_fleet, routed, upstream, path, kwargs):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        kwargs = dict(kwargs)
        headers = {**_auth(raw), **kwargs.pop("headers", {})}
        r = client.post(path, headers=headers, **kwargs)
        assert r.status_code == 403, r.text
        assert (r.json().get("error") or {}).get("code") == "model_not_allowed", r.text
        assert routed == []
        assert path not in upstream

    def test_an_unrestricted_key_still_forwards_a_body_with_no_readable_model(
            self, client, fake_fleet, routed, upstream):
        raw, meta = gw.mint_key("form-upload")
        r = client.post("/v1/audio/transcriptions", headers=_auth(raw),
                        files={"file": ("a.wav", b"RIFF0000", "audio/wav")},
                        data={"model": "whisper"})
        assert r.status_code == 200, r.text
        assert "/v1/audio/transcriptions" in upstream


class TestAgentAndKeyListsIntersect:
    def test_a_model_the_agent_allows_but_the_key_does_not_is_refused(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _agent(kid, ["qwen3.8-27b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        _assert_refused(r, "qwen3.8-27b")
        assert routed == []

    def test_a_model_the_key_allows_but_the_agent_does_not_is_still_refused(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _agent(kid, ["qwen3.8-27b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.5-4b", "messages": _MSGS})
        assert r.status_code == 403, r.text
        assert routed == []

    def test_a_model_both_lists_allow_is_routed(
            self, client, admin_headers, fake_fleet, routed, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b", "qwen3.8-27b"])
        _agent(kid, ["qwen3.8-27b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.8-27b", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert [m for m, _ in routed] == ["qwen3.8-27b"]


class TestFallbackRespectsTheKeyList:
    """nemotron3.5-lightning-30b has no candidate in the fake fleet, and
    pick_fallback's substitute for it is qwen3.6-35b-a3b (pinned for agent
    lists in test_audit_fixes.py). Routing is the real resolve_targets here."""

    def test_a_substitute_off_the_key_list_is_dropped(
            self, client, admin_headers, fake_fleet, upstream):
        raw, kid = _mint(client, admin_headers, ["nemotron3.5-lightning-30b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "nemotron3.5-lightning-30b", "messages": _MSGS})
        assert r.status_code == 404, r.text
        assert "qwen3.6-35b-a3b" not in r.json()["error"]["message"]
        assert "/v1/chat/completions" not in upstream

    def test_a_substitute_on_the_key_list_still_applies(
            self, client, admin_headers, fake_fleet, upstream):
        raw, kid = _mint(client, admin_headers,
                         ["nemotron3.5-lightning-30b", "qwen3.6-35b-a3b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "nemotron3.5-lightning-30b", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert r.json()["x_fleet"]["served"] == "qwen3.6-35b-a3b"


class TestListings:
    def test_an_unrestricted_key_sees_the_whole_fleet(self, client, fake_fleet):
        # Guard for the test below: these are the names a restricted key must
        # NOT see, so they have to be listed in this fake fleet to begin with.
        raw, meta = gw.mint_key("list-all")
        ids = {m["id"] for m in client.get("/v1/models", headers=_auth(raw)).json()["data"]}
        assert {"qwen3.8-27b", "qwen3.8-flash-next", "fast", "gemma", "default"} <= ids

    def test_v1_models_lists_only_what_the_key_may_call(self, client, admin_headers, fake_fleet):
        raw, kid = _mint(client, admin_headers, ["qwen3.8-27b"])
        r = client.get("/v1/models", headers=_auth(raw))
        assert r.status_code == 200, r.text
        ids = {m["id"] for m in r.json()["data"]}
        may = {"qwen3.8-27b"} | _roles_reaching("qwen3.8-27b")
        assert ids <= may, sorted(ids - may)
        assert {"qwen3.8-27b", "default"} <= ids

    def test_the_404_for_an_unknown_model_lists_only_allowed_ids(
            self, client, admin_headers, fake_fleet, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.8-27b", "ghost-model"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "ghost-model", "messages": _MSGS})
        assert r.status_code == 404, r.text
        available = set(r.json()["error"]["available"])
        may = {"qwen3.8-27b", "ghost-model"} | _roles_reaching("qwen3.8-27b")
        assert available <= may, sorted(available - may)
        assert "qwen3.8-27b" in available


# ---- native /api ---------------------------------------------------------------

class TestNative:
    @pytest.mark.parametrize("path", ["/api/chat", "/api/generate", "/api/embed",
                                      "/api/embeddings", "/api/show"])
    def test_a_call_naming_an_unlisted_model_is_refused(
            self, client, admin_headers, upstream, path):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post(path, headers=_auth(raw),
                        json={"model": "qwen3.8-27b", "stream": False, "messages": _MSGS})
        assert r.status_code == 403, r.text
        assert path not in upstream

    def test_a_listed_model_reaches_the_engine(self, client, admin_headers, upstream):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/api/chat", headers=_auth(raw),
                        json={"model": "qwen3.5-4b", "stream": False, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert "/api/chat" in upstream

    @pytest.mark.parametrize("path", ["/api/tags", "/api/ps"])
    def test_a_listing_shows_only_allowed_models_or_is_refused(
            self, client, admin_headers, upstream, path):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.get(path, headers=_auth(raw))
        if r.status_code == 403:
            return
        assert r.status_code == 200, r.text
        assert [m["name"] for m in r.json()["models"]] == ["qwen3.5:4b"]

    @pytest.mark.parametrize("path", ["/api/tags", "/api/ps"])
    def test_an_unrestricted_key_sees_the_whole_listing(self, client, upstream, path):
        raw, meta = gw.mint_key("native-list-all")
        r = client.get(path, headers=_auth(raw))
        assert r.status_code == 200, r.text
        assert [m["name"] for m in r.json()["models"]] == ["qwen3.5:4b", "qwen3.8-27b"]


# The gate judges the NAME a client asked for against every box's meaning of it
# (a peer's alias included). A native request is never routed: it runs on THIS
# box's engine, so it is this box's meaning of the name that has to be listed.

@pytest.fixture
def local_rows(monkeypatch, tmp_path):
    """This box's models.json, replaced by `put((id, *aliases), ...)`."""
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")

    def put(*rows: tuple[str, ...]) -> None:
        gw.save_models([{"id": rid, "aliases": list(aliases)} for rid, *aliases in rows])

    return put


class TestNativeRunsOnThisBox:
    def test_a_name_only_a_peer_aliases_to_a_listed_model_is_refused(
            self, client, admin_headers, upstream):
        _routes({"x": ["peer1"]}, {("peer1", "x"): "small-model"}, {})
        assert gw.key_model_gate(_key_row(["small-model"]), "x")[0] is True   # the control
        raw, kid = _mint(client, admin_headers, ["small-model"])
        r = client.post("/api/chat", headers=_auth(raw), json={
            "model": "x", "stream": False, "messages": _MSGS})
        _assert_refused(r, "x")
        assert "/api/chat" not in upstream
        assert {"model": "x", "endpoint": "/api/chat", "status": 403} in _usage(kid)

    def test_a_listed_class_word_this_box_has_a_row_for_off_the_list_is_refused(
            self, client, admin_headers, upstream, local_rows):
        local_rows(("qwen3.8-27b", "fast"))   # the word "fast" means the 27B here
        raw, kid = _mint(client, admin_headers, ["fast"])
        r = client.post("/api/chat", headers=_auth(raw), json={
            "model": "fast", "stream": False, "messages": _MSGS})
        _assert_refused(r, "fast")
        assert "/api/chat" not in upstream

    def test_a_listed_models_own_alias_on_this_box_still_reaches_the_engine(
            self, client, admin_headers, upstream, local_rows):
        local_rows(("qwen3.5-4b", "house-pick"))
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/api/chat", headers=_auth(raw), json={
            "model": "house-pick", "stream": False, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert "/api/chat" in upstream

    @pytest.mark.parametrize("path", ["/api/tags", "/api/ps"])
    def test_a_listing_is_cut_by_this_boxs_own_meaning_of_each_name(
            self, client, admin_headers, monkeypatch, path):
        # peer1 calls "x" the small model; this box's engine lists its own "x",
        # which it means something else by.
        _routes({"x": ["peer1"]}, {("peer1", "x"): "small-model"}, {})
        listing = {"models": [{"name": "x", "model": "x"},
                              {"name": "small-model", "model": "small-model"}]}

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json=listing)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        raw, kid = _mint(client, admin_headers, ["small-model"])
        r = client.get(path, headers=_auth(raw))
        assert r.status_code == 200, r.text
        assert [m["name"] for m in r.json()["models"]] == ["small-model"]


# ---- batches -------------------------------------------------------------------

class TestBatches:
    @pytest.mark.parametrize("pick", [
        pytest.param({"model": "qwen3.8-27b"}, id="one-unlisted-model"),
        pytest.param({"models": ["qwen3.5-4b", "qwen3.8-27b"]}, id="one-unlisted-of-two"),
    ])
    def test_a_batch_naming_an_unlisted_model_is_refused_and_nothing_is_queued(
            self, client, admin_headers, fake_fleet, batch_runs, pick):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        r = client.post("/v1/batches", headers=_auth(raw),
                        json={**pick, "requests": [{"messages": _MSGS}]})
        assert r.status_code == 403, r.text
        assert _batches_for(kid) == []
        assert batch_runs == []

    def test_a_class_with_only_partial_overlap_is_refused_for_a_batch(
            self, client, admin_headers, fake_fleet, batch_runs):
        # A batch fans a class out over each box's best member up front; a
        # limited ladder is a live-routing concept, so a batch takes the class
        # whole or not at all.
        raw, kid = _mint(client, admin_headers, ["qwen3.6-35b"])
        r = client.post("/v1/batches", headers=_auth(raw),
                        json={"model": "fast", "requests": [{"messages": _MSGS}]})
        assert r.status_code == 403, r.text
        assert _batches_for(kid) == []
        assert batch_runs == []

    def test_a_listed_class_is_batchable(self, client, admin_headers, fake_fleet, batch_runs):
        raw, kid = _mint(client, admin_headers, ["fast"])
        r = client.post("/v1/batches", headers=_auth(raw),
                        json={"model": "fast", "requests": [{"messages": _MSGS}]})
        assert r.status_code == 200, r.text
        assert r.json()["id"] in batch_runs

    def test_a_fallback_substitute_off_the_list_is_not_batched(
            self, client, admin_headers, fake_fleet, batch_runs):
        raw, kid = _mint(client, admin_headers, ["nemotron3.5-lightning-30b"])
        r = client.post("/v1/batches", headers=_auth(raw),
                        json={"model": "nemotron3.5-lightning-30b",
                              "requests": [{"messages": _MSGS}]})
        assert r.status_code == 404, r.text
        assert _batches_for(kid) == []
        assert batch_runs == []

    def _orphan(self, kid: int, models: list[str]) -> int:
        bid = gw.db_exec(
            "INSERT INTO batches(created_at,updated_at,key_id,key_name,label,"
            "models,status,total) VALUES (?,?,?,?,?,?,?,?)",
            (gw.now(), gw.now(), kid, "key-access", "", json.dumps(models), "running", 1))
        in_path, out_path = gw._batch_paths(bid)
        gw.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
        in_path.write_bytes(json.dumps({"messages": _MSGS}).encode() + b"\n")
        if out_path.exists():
            out_path.unlink()
        return int(bid)

    @pytest.mark.asyncio
    async def test_a_resumed_batch_whose_model_the_key_no_longer_allows_fails_with_a_reason(
            self, client, admin_headers, fake_fleet, batch_runs):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        bid = self._orphan(kid, ["qwen3.8-27b"])
        await gw.resume_orphaned_batches()
        row = gw.db_query("SELECT status, message FROM batches WHERE id=?", (bid,))[0]
        assert bid not in batch_runs
        assert row["status"] != "running"
        assert row["message"]

    @pytest.mark.asyncio
    async def test_a_resumed_batch_the_key_still_allows_resumes(
            self, client, admin_headers, fake_fleet, batch_runs):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        bid = self._orphan(kid, ["qwen3.5-4b"])
        await gw.resume_orphaned_batches()
        assert bid in batch_runs

    @pytest.fixture
    def boot_sweep(self, monkeypatch):
        """Boot, as resume_orphaned_batches meets it: no routing sweep has run,
        so no peer alias is known. model_routes() is replaced by a sweep that
        learns one -- peer1 serves `alias-a` as qwen3.5-4b -- and counts."""
        gw._routes_cache.update(t=0.0, map={}, cands={}, alias={}, meta={}, cap={},
                                running={}, reachable=set())
        sweeps: list[int] = []

        async def sweep(force: bool = False):
            sweeps.append(1)
            _routes({"alias-a": ["peer1"]}, {("peer1", "alias-a"): "qwen3.5-4b"}, {})
            return {}

        monkeypatch.setattr(gw, "model_routes", sweep)
        return sweeps

    @pytest.mark.asyncio
    async def test_a_resume_at_boot_learns_the_peer_aliases_before_failing_a_batch(
            self, client, admin_headers, batch_runs, boot_sweep):
        # The batch was accepted while the table was warm: the list names
        # qwen3.5-4b and the batch asked for the peer's alias of it. Nothing
        # about the list changed, so a cold table must not be what fails it.
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        bid = self._orphan(kid, ["alias-a"])
        await gw.resume_orphaned_batches()
        assert bid in batch_runs
        assert boot_sweep == [1]

    @pytest.mark.asyncio
    async def test_a_resume_at_boot_still_fails_what_the_swept_table_does_not_reach(
            self, client, admin_headers, batch_runs, boot_sweep):
        raw, kid = _mint(client, admin_headers, ["qwen3.8-27b"])
        bid = self._orphan(kid, ["alias-a"])
        await gw.resume_orphaned_batches()
        row = gw.db_query("SELECT status, message FROM batches WHERE id=?", (bid,))[0]
        assert bid not in batch_runs
        assert row["status"] == "error" and row["message"]

    @pytest.mark.asyncio
    async def test_a_warm_table_is_not_swept_again_to_fail_a_batch(
            self, client, admin_headers, batch_runs, boot_sweep):
        gw._routes_cache["t"] = 9e9   # warm, as a running gateway's is
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        bid = self._orphan(kid, ["qwen3.8-27b"])
        await gw.resume_orphaned_batches()
        assert bid not in batch_runs
        assert boot_sweep == []

    @pytest.mark.asyncio
    async def test_a_resumed_batch_whose_key_was_purged_fails_with_a_reason(
            self, client, admin_headers, fake_fleet, batch_runs):
        # No key row means no list to read, which must not read as "no list":
        # a key that was restricted would run its batch on any model.
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        bid = self._orphan(kid, ["qwen3.8-27b"])
        assert client.delete("/admin/api/keys/" + str(kid), headers=admin_headers).status_code == 200
        assert client.delete("/admin/api/keys/" + str(kid) + "/purge",
                             headers=admin_headers).status_code == 200
        assert gw.db_query("SELECT id FROM api_keys WHERE id=?", (kid,)) == []
        await gw.resume_orphaned_batches()
        row = gw.db_query("SELECT status, message FROM batches WHERE id=?", (bid,))[0]
        assert bid not in batch_runs
        assert row["status"] == "error"
        assert "key" in row["message"], row["message"]


# ---- team keys -----------------------------------------------------------------

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


class TestTeams:
    @pytest.mark.parametrize("path, body", [
        pytest.param("/v1/embeddings", {"model": "qwen3.8-27b", "input": "hi"}, id="embeddings"),
        pytest.param("/v1/completions", {"model": "qwen3.8-27b", "prompt": "hi"}, id="completions"),
    ])
    def test_a_team_key_off_the_chat_path_is_gated_like_any_key(
            self, client, admin_headers, fake_fleet, routed, upstream, crew, path, body):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, "qwen3.5-4b", [])
        r = client.post(path, headers=_auth(raw, **_NO_FALLBACK), json=body)
        _assert_refused(r, "qwen3.8-27b")
        assert routed == []
        assert path not in upstream

    @pytest.mark.parametrize("team_primary, client_model", [
        pytest.param("qwen3.8-27b", "team", id="the-teams-own-primary"),
        pytest.param("", "qwen3.8-27b", id="the-clients-model-when-the-team-names-none"),
    ])
    def test_a_refused_effective_primary_is_a_403_before_any_round(
            self, client, admin_headers, fake_fleet, crew, team_primary, client_model):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, team_primary, [])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": client_model, "messages": _MSGS,
                              "fleet_no_fallback": 1})
        _assert_refused(r, "qwen3.8-27b")
        assert crew == []

    def test_the_clients_placeholder_model_is_not_gated_when_the_team_has_a_primary(
            self, client, admin_headers, fake_fleet, crew):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, "qwen3.5-4b", [])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "team", "messages": _MSGS, "fleet_no_fallback": 1})
        assert r.status_code == 200, r.text
        assert [b["model"] for e, b in crew if e == "/v1/team-primary"] == ["qwen3.5-4b"]

    def test_every_worker_refused_leaves_no_crew_and_is_a_403(
            self, client, admin_headers, fake_fleet, crew):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, "qwen3.5-4b", ["qwen3.8-27b"])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "team", "messages": _MSGS, "fleet_no_fallback": 1})
        _assert_refused(r, None)
        assert crew == []

    def _spawning_crew(self, monkeypatch):
        primary: list[dict] = []
        workers: list[str] = []
        spawn = {"id": "call-1", "type": "function", "function": {
            "name": "spawn_subagents",
            "arguments": json.dumps({"tasks": [
                {"prompt": "a", "model": "qwen3.8-27b"},
                {"prompt": "b", "model": "qwen3.5-4b"}]})}}

        async def fake_chat(key, body, endpoint, ctx_limit=None, **kw):
            if endpoint == "/v1/team-primary":
                primary.append(body)
                if len(primary) == 1:
                    return 200, _round("", "tool_calls", [spawn]), gw.HOST_NAME, 0
                return 200, _round("done"), gw.HOST_NAME, 0
            workers.append(body["model"])
            return 200, _round("worker ok"), gw.HOST_NAME, 0

        monkeypatch.setattr(gw, "fleet_chat", fake_chat)
        return primary, workers

    def test_a_refused_worker_never_runs_even_when_a_task_asks_for_it(
            self, client, admin_headers, fake_fleet, monkeypatch):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, "qwen3.5-4b", ["qwen3.8-27b", "qwen3.5-4b"])
        primary, workers = self._spawning_crew(monkeypatch)
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "team", "messages": _MSGS, "fleet_no_fallback": 1})
        assert r.status_code == 200, r.text
        assert workers == ["qwen3.5-4b", "qwen3.5-4b"]

    def test_the_primary_is_not_offered_a_refused_worker(
            self, client, admin_headers, fake_fleet, monkeypatch):
        raw, kid = _mint(client, admin_headers, ["qwen3.5-4b"])
        _team(kid, "qwen3.5-4b", ["qwen3.8-27b", "qwen3.5-4b"])
        primary, workers = self._spawning_crew(monkeypatch)
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "team", "messages": _MSGS, "fleet_no_fallback": 1})
        assert r.status_code == 200, r.text
        sysline = primary[0]["messages"][0]["content"]
        assert "qwen3.8-27b" not in sysline


# ---- one model field -----------------------------------------------------------
# The gate reads a body with Python's json (the LAST of a repeated key wins, and
# names match exactly); the engines behind the gateway read the same bytes their
# own way (a first-wins gjson, a case-folding Go decoder). So a restricted key's
# request is rebuilt from what the gate read, holding one "model", on every
# surface that forwards a body.

_ALLOWED, _FORBIDDEN = "qwen3.5-4b", "qwen3.6-35b"
_TAIL = ', "messages": [{"role": "user", "content": "hi"}], "stream": false}'
_REPEAT_FORBIDDEN_FIRST = '{"model": "%s", "model": "%s"' % (_FORBIDDEN, _ALLOWED) + _TAIL
_SPLIT_BODIES = [
    pytest.param(_REPEAT_FORBIDDEN_FIRST, id="forbidden-then-allowed-repeat"),
    pytest.param('{"model": "%s", "Model": "%s"' % (_ALLOWED, _FORBIDDEN) + _TAIL,
                 id="allowed-then-a-case-variant"),
    pytest.param('{"MODEL": "%s", "model": "%s"' % (_FORBIDDEN, _ALLOWED) + _TAIL,
                 id="a-case-variant-then-allowed"),
]
_JSON = {"content-type": "application/json"}


def _assert_single_model(forwarded: bytes) -> None:
    """Whichever parser reads the body, it finds the allowed model and no other."""
    text = forwarded.decode()
    assert _FORBIDDEN not in text, text
    assert text.lower().count('"model"') == 1, text
    assert json.loads(text)["model"] == _ALLOWED


class TestOneModelField:
    @pytest.mark.parametrize("body", _SPLIT_BODIES)
    def test_v1_forwards_exactly_the_model_the_gate_read(
            self, client, admin_headers, fake_fleet, routed, sent_requests, body):
        raw, kid = _mint(client, admin_headers, [_ALLOWED])
        r = client.post("/v1/chat/completions",
                        headers={**_auth(raw, **_NO_FALLBACK), **_JSON}, content=body)
        assert r.status_code == 200, r.text
        (forwarded,) = [b for _h, p, b in sent_requests if p == "/v1/chat/completions"]
        _assert_single_model(forwarded)

    def test_a_repeat_whose_last_value_is_forbidden_is_refused(
            self, client, admin_headers, fake_fleet, routed, sent_requests):
        raw, kid = _mint(client, admin_headers, [_ALLOWED])
        body = '{"model": "%s", "model": "%s"' % (_ALLOWED, _FORBIDDEN) + _TAIL
        r = client.post("/v1/chat/completions",
                        headers={**_auth(raw, **_NO_FALLBACK), **_JSON}, content=body)
        _assert_refused(r, _FORBIDDEN)
        assert sent_requests == []

    def test_an_unrestricted_keys_body_is_forwarded_as_it_arrived(
            self, client, fake_fleet, routed, sent_requests):
        raw, meta = gw.mint_key("unrestricted-bytes")
        r = client.post("/v1/chat/completions",
                        headers={**_auth(raw, **_NO_FALLBACK), **_JSON},
                        content=_REPEAT_FORBIDDEN_FIRST)
        assert r.status_code == 200, r.text
        (forwarded,) = [b for _h, p, b in sent_requests if p == "/v1/chat/completions"]
        assert forwarded == _REPEAT_FORBIDDEN_FIRST.encode()

    @pytest.mark.parametrize("body", _SPLIT_BODIES)
    def test_native_forwards_exactly_the_model_the_gate_read(
            self, client, admin_headers, sent_requests, body):
        raw, kid = _mint(client, admin_headers, [_ALLOWED])
        r = client.post("/api/chat", headers={**_auth(raw), **_JSON}, content=body)
        assert r.status_code == 200, r.text
        (forwarded,) = [b for _h, p, b in sent_requests if p == "/api/chat"]
        _assert_single_model(forwarded)

    def test_a_batch_request_line_keeps_no_case_variant_of_model(
            self, client, admin_headers, fake_fleet, batch_runs):
        # The dispatcher sets "model" on each line; a "Model" riding along must
        # not be left for a case-folding engine to prefer to it.
        raw, kid = _mint(client, admin_headers, ["qwen3.6-35b"])
        r = client.post("/v1/batches", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "qwen3.6-35b",
                              "requests": [{"messages": _MSGS, "Model": "qwen3.8-27b"}]})
        assert r.status_code == 200, r.text
        in_path, _out_path = gw._batch_paths(int(r.json()["id"]))
        (line,) = in_path.read_text(encoding="utf-8").splitlines()
        assert [k for k in json.loads(line) if k.lower() == "model"] == []

    def test_a_teams_rounds_carry_no_case_variant_of_model(
            self, client, admin_headers, fake_fleet, crew):
        raw, kid = _mint(client, admin_headers, [_ALLOWED])
        _team(kid, _ALLOWED, [])
        r = client.post("/v1/chat/completions", headers=_auth(raw),
                        json={"model": "team", "Model": _FORBIDDEN, "messages": _MSGS,
                              "fleet_no_fallback": 1})
        assert r.status_code == 200, r.text
        (body,) = [b for e, b in crew if e == "/v1/team-primary"]
        assert [k for k in body if k.lower() == "model"] == ["model"]
        assert body["model"] == _ALLOWED


# ---- routing stays inside the list ----------------------------------------------
# The gate passes the NAME a client asked for. Routing then fans a name out --
# over every fleet id a catalogue row claims, and over every box that serves the
# word, though two boxes may mean different weights by it -- so each (box, id)
# it picks is held to the list on its own.

def _key_row(models: list[str]) -> dict:
    return {"id": 0, "name": "unit", "allowed_models": json.dumps(models)}


_DISAGREE = {("peer1", "x"): "small-model", ("bigbox", "x"): "big-model"}


class TestTargetsStayInsideTheList:
    def test_a_sibling_id_of_a_catalogue_row_the_list_does_not_reach_is_dropped(
            self, client, paired):
        pairs = [("p1", "pair-a"), ("p2", "pair-b")]
        assert gw.key_filter_targets(_key_row(["pair-a"]), pairs) == [("p1", "pair-a")]
        assert gw.key_filter_targets(_key_row(["pair-b"]), pairs) == [("p2", "pair-b")]
        assert gw.key_filter_targets(_key_row(["pair-a", "pair-b"]), pairs) == pairs
        assert gw.key_filter_targets(_key_row(["pair-pub"]), pairs) == pairs
        assert gw.key_filter_targets(_key_row([]), pairs) == pairs

    def test_a_name_two_boxes_mean_differently_keeps_the_box_whose_meaning_is_listed(
            self, client):
        _routes({"x": ["peer1", "bigbox"]}, dict(_DISAGREE), {})
        pairs = [("peer1", "x"), ("bigbox", "x")]
        assert gw.key_model_gate(_key_row(["small-model"]), "x")[0] is True
        assert gw.key_filter_targets(_key_row(["small-model"]), pairs) == [("peer1", "x")]
        assert gw.key_filter_targets(_key_row(["big-model"]), pairs) == [("bigbox", "x")]
        assert gw.key_filter_targets(_key_row(["small-model", "big-model"]), pairs) == pairs
        assert gw.key_filter_targets(_key_row(["other-model"]), pairs) == []

    # A peer's alias row translates the NAME a client asked for into the model
    # it means there; it must not translate a LIST ENTRY into more models, or a
    # row any peer reports would widen what the admin granted.
    _WIDE = ({"writers": ["peer1"], "premium-x": ["peer1"]},
             {("peer1", "writers"): "premium-x", ("peer1", "premium-x"): "premium-x"})

    def test_a_peers_alias_row_for_a_listed_word_does_not_list_its_target(self, client):
        _routes(*self._WIDE, {})
        key = _key_row(["writers"])   # a word, not a class: nothing on the list names premium-x
        assert gw.key_model_gate(key, "premium-x")[0] is False
        assert gw.key_filter_targets(key, [("peer1", "premium-x")]) == []
        assert gw.key_filter_models(key, ["premium-x", "writers"]) == ["writers"]

    def test_the_listed_word_itself_is_still_served_by_the_box_that_answers_it(self, client):
        _routes(*self._WIDE, {})
        key = _key_row(["writers"])
        assert gw.key_model_gate(key, "writers")[0] is True
        assert gw.key_filter_targets(key, [("peer1", "writers")]) == [("peer1", "writers")]

    def test_this_boxs_own_alias_row_still_lists_its_target(self, client, local_rows):
        local_rows(("qwen3.5-4b", "house-pick"))
        key = _key_row(["house-pick"])
        assert gw.key_model_gate(key, "qwen3.5-4b")[0] is True
        assert gw.key_filter_targets(key, [("", "qwen3.5-4b")]) == [("", "qwen3.5-4b")]

    def test_a_model_only_a_peers_alias_of_a_listed_word_names_is_a_metered_403(
            self, client, admin_headers, fake_fleet, fleet_wire):
        _routes(*self._WIDE, {})
        raw, kid = _mint(client, admin_headers, ["writers"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "premium-x", "messages": _MSGS})
        _assert_refused(r, "premium-x")
        assert fleet_wire == []
        assert {"model": "premium-x", "endpoint": "/v1/chat/completions",
                "status": 403} in _usage(kid)

    def test_a_peer_serving_an_old_spelling_is_still_reached_by_the_canonical_id(self, client):
        # No canonical reported (an older peer): the box serves the tag as itself.
        _routes({"qwen3.5:4b": ["peer1"]}, {}, {})
        assert gw.key_filter_targets(
            _key_row(["qwen3.5-4b"]), [("peer1", "qwen3.5:4b")]) == [("peer1", "qwen3.5:4b")]

    def test_a_listed_class_keeps_every_box_that_plays_it_whichever_member_it_holds(self, client, fake_fleet):
        pairs = [("peer1", "qwen3.8-9b-distill"), ("bigbox", "qwen3.5-4b")]
        assert gw.key_filter_targets(_key_row(["fast"]), pairs) == pairs

    @pytest.mark.parametrize("listed, host", [(["small-model"], "peer1"), (["big-model"], "bigbox")])
    def test_a_name_the_boxes_disagree_about_is_served_only_by_a_listed_meaning(
            self, client, admin_headers, fake_fleet, fleet_wire, listed, host):
        _routes({"x": ["peer1", "bigbox"]}, dict(_DISAGREE), {})
        raw, kid = _mint(client, admin_headers, listed)
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "x", "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert _chat_hosts(fleet_wire) == [host]

    @pytest.mark.parametrize("asked, host", [("pair-a", "peer1"), ("pair-b", "bigbox")])
    def test_a_fleet_id_a_catalogue_row_claims_is_not_served_by_the_rows_other_ids(
            self, client, admin_headers, fake_fleet, fleet_wire, paired, asked, host):
        _routes({"pair-a": ["peer1"], "pair-b": ["bigbox"]},
                _ident(("peer1", "pair-a"), ("bigbox", "pair-b")), {})
        raw, kid = _mint(client, admin_headers, [asked])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": asked, "messages": _MSGS})
        assert r.status_code == 200, r.text
        assert _chat_hosts(fleet_wire) == [host]
        (sent,) = [b for _h, p, b in fleet_wire if p == "/v1/chat/completions"]
        assert json.loads(sent)["model"] == asked

    def test_a_name_only_served_off_the_list_is_a_metered_403_and_nothing_is_sent(
            self, client, admin_headers, fake_fleet, fleet_wire, paired):
        _routes({"pair-b": ["bigbox"]}, _ident(("bigbox", "pair-b")), {})
        raw, kid = _mint(client, admin_headers, ["pair-a"])
        r = client.post("/v1/chat/completions", headers=_auth(raw, **_NO_FALLBACK),
                        json={"model": "pair-a", "messages": _MSGS})
        _assert_refused(r, "pair-a")
        assert fleet_wire == []
        assert {"model": "pair-a", "endpoint": "/v1/chat/completions",
                "status": 403} in _usage(kid)

    @pytest.mark.asyncio
    async def test_fleet_chat_keeps_a_teams_calls_inside_the_list(
            self, client, fake_fleet, paired, monkeypatch):
        sent: list[tuple[str, str]] = []

        async def fake_post(cand, payload, deadline=None):
            sent.append((cand, json.loads(payload)["model"]))
            return httpx.Response(200, json=_CHAT_OK_BODY)

        monkeypatch.setattr(gw, "_post_chat", fake_post)
        body = {"model": "pair-a", "messages": _MSGS}
        _routes({"pair-a": ["peer1"], "pair-b": ["bigbox"]},
                _ident(("peer1", "pair-a"), ("bigbox", "pair-b")), {})
        status, _resp, _host, _granted = await gw.fleet_chat(
            _key_row(["pair-a"]), body, "/v1/team-worker", role="worker")
        assert status == 200 and sent == [("peer1", "pair-a")]

        sent.clear()
        _routes({"pair-b": ["bigbox"]}, _ident(("bigbox", "pair-b")), {})
        status, resp, _host, _granted = await gw.fleet_chat(
            _key_row(["pair-a"]), body, "/v1/team-worker", role="worker")
        assert status == 403 and sent == []
        assert resp["error"]["code"] == "model_not_allowed"

    @pytest.mark.asyncio
    async def test_a_batch_is_spread_only_over_boxes_the_list_reaches(
            self, client, fake_fleet, paired):
        _routes({"pair-a": ["peer1"], "pair-b": ["bigbox"]},
                _ident(("peer1", "pair-a"), ("bigbox", "pair-b")), {})
        free = await gw._batch_targets(["pair-a"], no_fallback=True)
        assert {t["host"] for t in free} == {"peer1", "bigbox"}   # the control
        held = await gw._batch_targets(["pair-a"], no_fallback=True,
                                       key=_key_row(["pair-a"]))
        assert [(t["host"], t["model"]) for t in held] == [("peer1", "pair-a")]


# ---- identity ------------------------------------------------------------------

class TestNoIdentity:
    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat", "/v1/batches"])
    @pytest.mark.parametrize("headers", [
        pytest.param({}, id="no-bearer"),
        pytest.param({"Authorization": "Bearer sk-not-a-key-of-this-node"}, id="unknown-bearer"),
    ])
    def test_no_valid_key_is_a_401_before_any_gate_or_side_effect(
            self, client, fake_fleet, routed, upstream, batch_runs, path, headers):
        r = client.post(path, headers=headers,
                        json={"model": "qwen3.8-27b", "messages": _MSGS,
                              "requests": [{"messages": _MSGS}]})
        assert r.status_code == 401, r.text
        assert routed == [] and path not in upstream and batch_runs == []
        assert gw.db_query("SELECT COUNT(*) c FROM usage")[0]["c"] == 0
