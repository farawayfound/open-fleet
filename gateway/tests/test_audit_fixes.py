"""Regressions for the 2026-09 fault-tolerance/security audit (findings-A).

One file, one test class per finding, in the order the audit listed them.
Each of these was a real gap: an ordinary key deleting a model off disk, a
filename injecting CLI flags into a launch command, one bad line in
peers.json 404ing every request, and so on -- none of them raised where they
happened, which is exactly what makes them worth pinning down here.

Run with: $SP/venv/bin/python -m pytest gateway/tests/test_audit_fixes.py -q
"""
from __future__ import annotations

import json
import sqlite3
import time

import httpx
import pytest
import yaml

import app as gw

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. native_proxy(): the Ollama /api/* allow-list
# ---------------------------------------------------------------------------

class TestNativeProxyAllowList:
    def test_delete_pull_create_push_copy_are_refused_for_an_ordinary_key(
            self, client, monkeypatch):
        raw, _meta = gw.mint_key("native-allowlist-1")
        reached: list[str] = []

        async def fake_send(self, request, stream=False, **kw):
            reached.append(request.url.path)
            return httpx.Response(200, request=request, json={})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        headers = {"Authorization": "Bearer " + raw}

        r = client.request("DELETE", "/api/delete", headers=headers,
                           json={"model": "victim"})
        assert r.status_code == 403, r.text

        r = client.post("/api/pull", headers=headers, json={"model": "x"})
        assert r.status_code == 403, r.text

        r = client.post("/api/create", headers=headers,
                        json={"model": "x", "modelfile": "FROM x"})
        assert r.status_code == 403, r.text

        r = client.post("/api/push", headers=headers, json={"model": "x"})
        assert r.status_code == 403, r.text

        r = client.post("/api/copy", headers=headers,
                        json={"source": "a", "destination": "b"})
        assert r.status_code == 403, r.text

        assert reached == [], "none of these may ever reach the upstream"

    def test_the_documented_inference_and_probe_paths_still_pass_through(
            self, client, monkeypatch):
        raw, _meta = gw.mint_key("native-allowlist-2")
        reached: list[str] = []

        async def fake_send(self, request, stream=False, **kw):
            reached.append(request.url.path)
            return httpx.Response(
                200, request=request,
                json={"message": {"role": "assistant", "content": "ok"},
                     "done": True, "prompt_eval_count": 1, "eval_count": 1})

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        headers = {"Authorization": "Bearer " + raw}

        r = client.post("/api/generate", headers=headers,
                        json={"model": "m", "stream": False, "prompt": "hi"})
        assert r.status_code == 200, r.text

        for path in ("/api/ps", "/api/tags", "/api/show", "/api/version"):
            r = client.get(path, headers=headers)
            assert r.status_code == 200, (path, r.text)

        assert reached == ["/api/generate", "/api/ps", "/api/tags",
                           "/api/show", "/api/version"]


# ---------------------------------------------------------------------------
# 2. build_cmd() argv injection: SAFE_PATH / unsafe_path()
# ---------------------------------------------------------------------------

class TestUnsafePath:
    @pytest.mark.parametrize("value", [
        "/models/plain.gguf",
        "/models/sub-dir/model.name-v2.gguf",
        "",
        None,
    ])
    def test_ordinary_paths_are_safe(self, value):
        assert gw.unsafe_path(value) is None

    @pytest.mark.parametrize("value", [
        "/models/has space.gguf",           # medium finding: silently breaks
        "/models/x.gguf; rm -rf /",
        "/models/x.gguf' --foo bar",
        "/models/x.gguf\" --foo",
        "/models/x.gguf`whoami`",
        "-ngl 999",                          # a leading '-' reads as a flag
    ])
    def test_dangerous_values_are_rejected(self, value):
        assert gw.unsafe_path(value) is not None


class TestModelPutValidatesPathFields:
    REC = dict(gw.DEFAULT_MODEL_RECORD, id="m-safety", path="/models/ok.gguf")

    def test_a_path_with_a_space_is_rejected(self, client, admin_headers):
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [dict(self.REC,
                                             path="/models/has space.gguf")]})
        assert r.status_code == 400
        assert "path" in r.json()["detail"]

    def test_an_mmproj_with_a_shell_metacharacter_is_rejected(self, client, admin_headers):
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [dict(self.REC,
                                             mmproj="/models/p.gguf; touch pwned")]})
        assert r.status_code == 400
        assert "mmproj" in r.json()["detail"]

    def test_a_cache_type_with_a_quote_is_rejected(self, client, admin_headers):
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [dict(self.REC, cache_type_k="q4_0' -x")]})
        assert r.status_code == 400
        assert "cache_type_k" in r.json()["detail"]

    def test_an_ordinary_record_is_still_accepted(self, client, admin_headers):
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [self.REC]})
        assert r.status_code == 200, r.text


class TestRenderSwapConfigSkipsUnsafeRecords:
    def test_an_unsafe_path_is_skipped_not_launched(self):
        safe = dict(gw.DEFAULT_MODEL_RECORD, id="safe-one", path="/models/ok.gguf")
        unsafe = dict(gw.DEFAULT_MODEL_RECORD, id="unsafe-one",
                     path="/models/bad one.gguf")
        cfg = yaml.safe_load(gw.render_swap_config([safe, unsafe]))
        assert "safe-one" in cfg["models"]
        assert "unsafe-one" not in cfg["models"]


class TestLmStudioImportSkipsUnsafePaths:
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
        return lms

    def test_a_maliciously_named_gguf_is_listed_skipped_not_todo(self, store):
        bad = store / "pub" / "repo-GGUF" / "model with space.gguf"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"GGUF" + b"a" * 4092)

        plan = gw.lmstudio_plan()
        actions = [a for a in plan["actions"] if a["kind"] == "import"]
        assert len(actions) == 1
        assert actions[0]["status"] == "skipped"
        assert "whitespace" in actions[0]["reason"] or "shell" in actions[0]["reason"]

    def test_it_is_never_registered_by_a_sync_pass(self, store, monkeypatch):
        bad = store / "pub" / "repo-GGUF" / "model with space.gguf"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"GGUF" + b"a" * 4092)
        monkeypatch.setattr(gw, "_upstream_listening", lambda: True)

        res = gw.lmstudio_sync()
        assert res["imported"] == 0
        assert gw.load_models() == []


# ---------------------------------------------------------------------------
# 3. load_peers(): tolerate a corrupt/unreadable peers.json
# ---------------------------------------------------------------------------

class TestLoadPeersTolerant:
    def test_non_list_json_degrades_to_no_peers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "_peers_load_warned", False)
        p = tmp_path / "peers.json"
        p.write_text(json.dumps({"not": "a list"}))
        monkeypatch.setattr(gw, "PEERS_PATH", p)
        assert gw.load_peers() == []

    def test_an_unreadable_file_degrades_to_no_peers(self, monkeypatch):
        monkeypatch.setattr(gw, "_peers_load_warned", False)

        class _Unreadable:
            def exists(self):
                return True

            def read_text(self):
                raise OSError("permission denied")

        monkeypatch.setattr(gw, "PEERS_PATH", _Unreadable())
        assert gw.load_peers() == []

    def test_bad_encoding_degrades_to_no_peers(self, monkeypatch):
        monkeypatch.setattr(gw, "_peers_load_warned", False)

        class _BadEncoding:
            def exists(self):
                return True

            def read_text(self):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        monkeypatch.setattr(gw, "PEERS_PATH", _BadEncoding())
        assert gw.load_peers() == []

    def test_a_single_line_json_error_still_returns_no_peers(self, tmp_path, monkeypatch):
        # The pre-existing behaviour (JSONDecodeError) must survive untouched.
        p = tmp_path / "peers.json"
        p.write_text("{not json")
        monkeypatch.setattr(gw, "PEERS_PATH", p)
        assert gw.load_peers() == []

    def test_the_warning_is_logged_once_not_per_request(self, monkeypatch):
        monkeypatch.setattr(gw, "_peers_load_warned", False)

        class _Unreadable:
            def exists(self):
                return True

            def read_text(self):
                raise OSError("permission denied")

        monkeypatch.setattr(gw, "PEERS_PATH", _Unreadable())
        calls = []
        monkeypatch.setattr(gw.log, "warning", lambda *a, **k: calls.append(a))
        gw.load_peers()
        gw.load_peers()
        gw.load_peers()
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 4. team keys: n/best_of/logprobs strip
# ---------------------------------------------------------------------------

class TestStripAbuseKnobs:
    def test_strip_abuse_knobs_pins_n_and_drops_best_of_logprobs(self):
        payload = {"n": 5, "best_of": 4, "logprobs": 5, "model": "m"}
        mutated = gw._strip_abuse_knobs(payload)
        assert mutated is True
        assert payload["n"] == 1
        assert "best_of" not in payload
        assert "logprobs" not in payload

    def test_an_already_clean_payload_is_reported_unmutated(self):
        payload = {"n": 1, "model": "m"}
        assert gw._strip_abuse_knobs(payload) is False

    async def test_a_team_keys_primary_round_is_stripped_even_off_fleet_pass(
            self, fake_fleet, monkeypatch):
        raw, meta = gw.mint_key("team-strip-1")
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        team = {"primary_model": "gemma4-31b-qat", "worker_models": "[]",
               "max_rounds": 1, "max_workers": 2, "ctx_limit": None}
        seen: list[dict] = []

        async def fake_fleet_chat(key, body, endpoint, ctx_limit=None, **kw):
            seen.append(dict(body))
            return 200, {
                "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }, gw.HOST_NAME, 0

        monkeypatch.setattr(gw, "fleet_chat", fake_fleet_chat)
        payload = {"model": "team", "n": 8, "best_of": 8, "logprobs": 5,
                  "messages": [{"role": "user", "content": "hi"}]}
        await gw.team_orchestrate(key_row, team, payload, False, time.time())

        assert len(seen) == 1
        assert seen[0]["n"] == 1
        assert "best_of" not in seen[0]
        assert "logprobs" not in seen[0]


# ---------------------------------------------------------------------------
# 5. fallback substitution must respect an agent's allowed_models
# ---------------------------------------------------------------------------

_CHAT_OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


class TestFallbackRespectsAllowedModels:
    def _insert_agent(self, kid: int, allowed: list[str]) -> None:
        gw.db_exec(
            "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
            "param_overrides,updated_at) VALUES (?,1,?,?,?,?,?)",
            (kid, "restricted", json.dumps(allowed), "", "{}", gw.now()),
        )

    def test_a_substitute_outside_the_allow_list_is_dropped(
            self, client, fake_fleet, monkeypatch):
        # nemotron3.5-lightning-30b has no candidate in the fake fleet, and
        # pick_fallback's own substitute for it (proven in test_public.py) is
        # qwen3.6-35b-a3b -- which is deliberately NOT on this key's
        # allow-list. If the substitution were not dropped, routing would
        # succeed against qwen and reach the upstream below.
        raw, meta = gw.mint_key("agent-allowlist-1")
        self._insert_agent(int(meta["id"]), ["nemotron3.5-lightning-30b"])

        async def fake_send(self, request, stream=False, **kw):
            raise AssertionError(
                "must not reach the upstream: the only candidate for this "
                "request is a model outside the key's allow-list")

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "nemotron3.5-lightning-30b",
                 "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 404, r.text
        msg = r.json()["error"]["message"]
        assert "nemotron3.5-lightning-30b" in msg
        assert "qwen3.6-35b-a3b" not in msg

    def test_a_substitute_inside_the_allow_list_still_applies(
            self, client, fake_fleet, monkeypatch):
        raw, meta = gw.mint_key("agent-allowlist-2")
        self._insert_agent(int(meta["id"]),
                          ["nemotron3.5-lightning-30b", "qwen3.6-35b-a3b"])

        async def fake_send(self, request, stream=False, **kw):
            return httpx.Response(200, request=request, json=_CHAT_OK_BODY)

        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "nemotron3.5-lightning-30b",
                 "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 200, r.text
        assert r.json()["x_fleet"]["served"] == "qwen3.6-35b-a3b"


# ---------------------------------------------------------------------------
# 6. /v1/batches: request-count and total-byte caps
# ---------------------------------------------------------------------------

class TestBatchCaps:
    async def test_too_many_requests_is_refused_before_touching_the_fleet(
            self, monkeypatch):
        monkeypatch.setattr(gw, "MAX_BATCH_REQUESTS", 3)
        raw, meta = gw.mint_key("batch-cap-count")
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        payload = {"model": "m", "requests": [
            {"messages": [{"role": "user", "content": "hi"}]} for _ in range(4)]}
        with pytest.raises(gw.HTTPException) as exc:
            await gw.batch_submit(key_row, payload)
        assert exc.value.status_code == 413
        assert "3" in str(exc.value.detail)

    async def test_total_spool_bytes_over_the_cap_is_refused(
            self, fake_fleet, monkeypatch):
        monkeypatch.setattr(gw, "MAX_BATCH_BYTES", 300)
        raw, meta = gw.mint_key("batch-cap-bytes")
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        big = "x" * 100
        payload = {"model": "gemma4-31b-qat", "requests": [
            {"messages": [{"role": "user", "content": big}]} for _ in range(6)]}
        with pytest.raises(gw.HTTPException) as exc:
            await gw.batch_submit(key_row, payload)
        assert exc.value.status_code == 413
        assert "MiB" in str(exc.value.detail)

    async def test_an_ordinary_small_batch_still_submits(self, fake_fleet):
        raw, meta = gw.mint_key("batch-cap-ok")
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        payload = {"model": "gemma4-31b-qat", "requests": [
            {"messages": [{"role": "user", "content": "hi"}]}]}
        out = await gw.batch_submit(key_row, payload)
        assert out["status"] == "running"
        assert out["total"] == 1


# ---------------------------------------------------------------------------
# 7. CSRF defense on the cookie-authenticated admin surface
# ---------------------------------------------------------------------------

class _FakeSigningKey:
    key = "unused"


class _FakeJwks:
    def get_signing_key_from_jwt(self, assertion):
        return _FakeSigningKey()


@pytest.fixture
def cf_cookie_auth(monkeypatch):
    """Makes require_admin's Cloudflare Access branch succeed for ANY
    assertion string, with no real JWT or network call -- so these tests
    exercise only the CSRF gate layered in front of it, not Access itself."""
    monkeypatch.setattr(gw, "CF_AUD", "test-aud")
    monkeypatch.setattr(gw, "CF_TEAM_DOMAIN", "test.cloudflareaccess.com")
    monkeypatch.setattr(gw, "jwks", lambda: _FakeJwks())
    monkeypatch.setattr(gw.jwt, "decode", lambda *a, **k: {"email": "admin@example.test"})


class TestAdminCSRFGuard:
    def _peers_body(self):
        return {"peers": []}

    def test_cookie_auth_cross_site_sec_fetch_site_is_refused(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       headers={"Sec-Fetch-Site": "cross-site"},
                       json=self._peers_body())
        assert r.status_code == 403
        assert "sec-fetch-site" in r.json()["detail"].lower()

    def test_cookie_auth_same_origin_sec_fetch_site_passes(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       headers={"Sec-Fetch-Site": "same-origin"},
                       json=self._peers_body())
        assert r.status_code == 200, r.text

    def test_cookie_auth_mismatched_origin_is_refused(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       headers={"Origin": "http://evil.example"},
                       json=self._peers_body())
        assert r.status_code == 403
        assert "origin" in r.json()["detail"].lower()

    def test_cookie_auth_matching_origin_passes(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       headers={"Origin": "http://testserver"},
                       json=self._peers_body())
        assert r.status_code == 200, r.text

    def test_cookie_auth_with_neither_header_passes_a_non_browser_client(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       json=self._peers_body())
        assert r.status_code == 200, r.text

    def test_a_safe_get_is_never_csrf_checked(self, client, cf_cookie_auth):
        r = client.get("/admin/api/peers", cookies={"CF_Authorization": "fake"},
                       headers={"Origin": "http://evil.example"})
        assert r.status_code == 200

    def test_the_header_assertion_path_is_never_csrf_checked(
            self, client, cf_cookie_auth, tmp_path, monkeypatch):
        # A browser cannot attach this header to a cross-site request the
        # way it rides along on a cookie -- the header path is exempt.
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        r = client.put("/admin/api/peers",
                       headers={"cf-access-jwt-assertion": "fake",
                               "Origin": "http://evil.example"},
                       json=self._peers_body())
        assert r.status_code == 200, r.text

    def test_the_admin_bearer_token_path_is_never_csrf_checked(
            self, client, admin_headers, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        headers = dict(admin_headers, Origin="http://evil.example")
        r = client.put("/admin/api/peers", headers=headers,
                       json=self._peers_body())
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# 8. write_atomic(): fsync the temp file and the directory
# ---------------------------------------------------------------------------

class TestWriteAtomicFsyncs:
    def test_fsyncs_the_file_and_the_directory(self, tmp_path, monkeypatch):
        calls: list[int] = []
        real_fsync = gw.os.fsync

        def _counting_fsync(fd):
            calls.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(gw.os, "fsync", _counting_fsync)
        target = tmp_path / "sub" / "file.txt"
        gw.write_atomic(target, "hello")
        assert target.read_text() == "hello"
        assert len(calls) >= 2  # the temp file, and the directory

    def test_a_directory_that_refuses_fsync_does_not_break_the_write(
            self, tmp_path, monkeypatch):
        real_open = gw.os.open

        def _flaky_open(path, flags, *a, **kw):
            if flags == gw.os.O_RDONLY:
                raise OSError("this filesystem refuses to open a directory")
            return real_open(path, flags, *a, **kw)

        monkeypatch.setattr(gw.os, "open", _flaky_open)
        target = tmp_path / "file.txt"
        gw.write_atomic(target, "data")  # must not raise
        assert target.read_text() == "data"


# ---------------------------------------------------------------------------
# 9. db_init(): only swallow the two expected migration errors
# ---------------------------------------------------------------------------

class TestDbInitMigrationErrors:
    def test_every_real_migration_stays_silent_on_a_second_run(self, client):
        # Every migration has already applied once (the session-scoped
        # `client` fixture's lifespan ran db_init() at import) -- running it
        # again must re-hit every "duplicate column"/"already exists" branch
        # without raising.
        gw.db_init()

    def test_an_unrelated_operational_error_is_not_swallowed(self, client, monkeypatch):
        bad = gw.MIGRATIONS + ("ALTER TABLE no_such_table ADD COLUMN x INTEGER",)
        monkeypatch.setattr(gw, "MIGRATIONS", bad)
        with pytest.raises(sqlite3.OperationalError):
            gw.db_init()


# ---------------------------------------------------------------------------
# 10. apply_ctx_limit(): cap max_completion_tokens too
# ---------------------------------------------------------------------------

class TestApplyCtxLimitCapsBothFieldNames:
    def test_max_completion_tokens_is_clipped_to_the_context_cap(self):
        payload = {"messages": [{"role": "user", "content": "hi"}],
                  "max_completion_tokens": 99999}
        out = gw.apply_ctx_limit(payload, 4096)
        assert out["max_completion_tokens"] < 4096
        assert out["max_completion_tokens"] == out["max_tokens"]

    def test_a_tiny_max_completion_tokens_is_not_read_as_unset(self):
        # Before the fix, only max_tokens was inspected -- an explicit tiny
        # max_completion_tokens with no max_tokens field fell through to
        # PUBLIC_DEFAULT_COMPLETION instead of being honoured (then raised
        # to the floor).
        payload = {"messages": [{"role": "user", "content": "hi"}],
                  "max_completion_tokens": 12}
        out = gw.apply_ctx_limit(payload, 8192)
        assert out["max_tokens"] == gw.PUBLIC_MIN_COMPLETION
        assert out["max_completion_tokens"] == gw.PUBLIC_MIN_COMPLETION

    def test_max_tokens_only_leaves_max_completion_tokens_untouched(self):
        # No spurious field is invented when the client never sent one.
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000}
        out = gw.apply_ctx_limit(payload, 8192)
        assert out["max_tokens"] == 1000
        assert "max_completion_tokens" not in out


# ---------------------------------------------------------------------------
# 11. PUT /admin/api/peers: a url change must not carry the old token over
# ---------------------------------------------------------------------------

class TestPeerUrlChangeRequiresFreshToken:
    def _seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        gw.save_peers([{"name": "peer1", "url": "http://peer1:8080",
                       "token": "real-secret-token", "api_url": "",
                       "inference_key": "", "routed": True}])

    def test_changing_url_without_a_new_token_is_refused(
            self, client, admin_headers, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        r = client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "peer1", "url": "http://attacker.example:8080"}]})
        assert r.status_code == 400
        assert "token" in r.json()["detail"].lower()
        stored = gw.load_peers()
        assert stored[0]["url"] == "http://peer1:8080"
        assert stored[0]["token"] == "real-secret-token"

    def test_changing_url_with_a_fresh_token_is_allowed(
            self, client, admin_headers, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        r = client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "peer1", "url": "http://peer1-new:8080",
                      "token": "new-token"}]})
        assert r.status_code == 200, r.text
        stored = gw.load_peers()
        assert stored[0]["url"] == "http://peer1-new:8080"
        assert stored[0]["token"] == "new-token"

    def test_an_ordinary_edit_with_no_url_change_still_keeps_the_token(
            self, client, admin_headers, tmp_path, monkeypatch):
        self._seed(tmp_path, monkeypatch)
        r = client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "peer1", "url": "http://peer1:8080"}]})
        assert r.status_code == 200, r.text
        stored = gw.load_peers()
        assert stored[0]["token"] == "real-secret-token"
