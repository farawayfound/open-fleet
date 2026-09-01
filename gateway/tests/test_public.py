"""Fleet Pass: public, auto-approved, rate-limited demo API keys.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# eligibility (pure function)
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_free_mail_rejected(self):
        assert gw.public_eligibility("person@gmail.com") == \
            {"verdict": "reject", "code": "free_mail"}

    def test_free_mail_family_suffix(self):
        # yahoo operates many country TLDs; the family match must catch them.
        assert gw.public_eligibility("person@mail.yahoo.co.uk")["code"] == "free_mail"

    def test_gov_suffix_allowed(self):
        out = gw.public_eligibility("person@nasa.gov")
        assert out["verdict"] == "allow"
        assert out["source"] == "gov"

    def test_mil_suffix_allowed(self):
        out = gw.public_eligibility("person@army.mil")
        assert out["verdict"] == "allow"
        assert out["source"] == "gov"

    def test_exact_domain_allowed(self):
        out = gw.public_eligibility("person@amazon.com")
        assert out["verdict"] == "allow"
        assert out["source"] == "fortune500"
        assert out["company"] == "Amazon"

    def test_subdomain_of_allowed_domain_matches(self):
        out = gw.public_eligibility("person@mail.amazon.com")
        assert out["verdict"] == "allow"
        assert out["company"] == "Amazon"

    def test_blocked_domain_rejected(self):
        gw._public_domain_upsert({"domain": "notwelcome.example", "mode": "block",
                                  "source": "custom"})
        try:
            out = gw.public_eligibility("person@notwelcome.example")
            assert out == {"verdict": "reject", "code": "blocked"}
        finally:
            gw.db_exec("DELETE FROM public_domains WHERE domain=?",
                      ("notwelcome.example",))

    def test_unlisted_domain_reviews_by_default(self):
        out = gw.public_eligibility("person@some-unlisted-company.example")
        assert out == {"verdict": "review"}

    def test_unlisted_domain_rejected_when_review_disabled(self):
        gw.set_public_settings({"review_unlisted": False})
        try:
            out = gw.public_eligibility("person@some-unlisted-company.example")
            assert out == {"verdict": "reject", "code": "not_listed"}
        finally:
            gw.set_public_settings({"review_unlisted": True})

    def test_invalid_email_syntax(self):
        assert gw.public_eligibility("not-an-email")["code"] == "invalid_email"


# ---------------------------------------------------------------------------
# the Public tab's issued-keys view
# ---------------------------------------------------------------------------

class TestIssuedKeysListing:
    def _issue(self, client, intake_headers, email):
        r = client.post("/public/api/request", headers=intake_headers,
                        json={"email": email, "kind": "single",
                              "model": "gemma4-31b-qat", "ctx": 8192,
                              "accept_terms": True})
        assert r.status_code == 200

    def test_listing_carries_expiry_and_the_minted_limits(
            self, client, admin_headers, intake_headers, captured_mail, fake_fleet):
        self._issue(client, intake_headers, "listing@nasa.gov")
        row = [r for r in client.get("/admin/api/public/keys",
                                     headers=admin_headers).json()["items"]
               if r["email"] == "listing@nasa.gov"][0]
        assert row["expires_at"], "the tab cannot show an expiry it never selected"
        assert row["limit_day"] == 5 and row["limit_hour"] == 2

    def test_a_key_past_its_expiry_shows_as_expired(
            self, client, admin_headers, intake_headers, captured_mail, fake_fleet):
        self._issue(client, intake_headers, "expired@nasa.gov")
        kid = [r for r in client.get("/admin/api/public/keys",
                                     headers=admin_headers).json()["items"]
               if r["email"] == "expired@nasa.gov"][0]["key_id"]
        gw.db_exec("UPDATE api_keys SET expires_at=? WHERE id=?",
                   ("2020-01-01T00:00:00+00:00", kid))
        row = [r for r in client.get("/admin/api/public/keys",
                                     headers=admin_headers).json()["items"]
               if r["email"] == "expired@nasa.gov"][0]
        assert row["status"] == "expired"


# ---------------------------------------------------------------------------
# what an unauthenticated caller can see
# ---------------------------------------------------------------------------

class TestUnauthenticatedSurface:
    def test_openapi_schema_is_not_served(self, client):
        """api.example.com is on the public internet with no Access
        policy. FastAPI's schema named every admin route and its parameters
        to anyone who asked -- 59 KB of it, measured against the live host."""
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

    def test_health_and_public_endpoints_still_open(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/public/api/models").status_code == 200


# ---------------------------------------------------------------------------
# the agents admin API and a public key's context cap
# ---------------------------------------------------------------------------

class TestAgentCtxLimit:
    def test_editing_a_profile_does_not_wipe_the_context_cap(self, client,
                                                             admin_headers):
        """A Fleet Pass key is an ordinary key with an agent profile. Someone
        editing its prompt on the Agents tab must not silently uncap its
        context: that tab does not send ctx_limit at all."""
        raw, meta = gw.mint_key("agent-ctx-test")
        kid = int(meta["id"])
        gw.db_exec(
            "INSERT INTO agents(key_id,enabled,name,allowed_models,force_model,"
            "param_overrides,ctx_limit,updated_at) VALUES (?,1,?,?,?,?,?,?)",
            (kid, "fleet-pass", '["gemma4-12b"]', "gemma4-12b", "{}", 8192,
             gw.now()),
        )
        r = client.put("/admin/api/agents/" + str(kid), headers=admin_headers,
                       json={"enabled": True, "name": "edited",
                             "system_prompt": "hello", "allowed_models": ["gemma4-12b"],
                             "force_model": "gemma4-12b"})
        assert r.status_code == 200
        assert r.json()["ctx_limit"] == 8192
        assert r.json()["system_prompt"] == "hello"

    def test_ctx_limit_can_be_set_and_cleared_explicitly(self, client, admin_headers):
        raw, meta = gw.mint_key("agent-ctx-test-2")
        kid = int(meta["id"])
        body = {"enabled": True, "name": "x", "allowed_models": [],
                "force_model": "", "ctx_limit": 4096}
        r = client.put("/admin/api/agents/" + str(kid), headers=admin_headers, json=body)
        assert r.json()["ctx_limit"] == 4096
        r = client.put("/admin/api/agents/" + str(kid), headers=admin_headers,
                       json={**body, "ctx_limit": None})
        assert r.json()["ctx_limit"] in (None, 0)


# ---------------------------------------------------------------------------
# client_ip: Cloudflare's header wins, the proxy header only counts off-edge
# ---------------------------------------------------------------------------

class _HdrRequest:
    def __init__(self, headers: dict, host: str = "127.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


class TestClientIp:
    def test_cloudflare_header_beats_forwarded_header(self):
        # A public caller holding a leaked intake token cannot pick its own
        # address: through the edge, CF-Connecting-IP is the only truth.
        req = _HdrRequest({"cf-connecting-ip": "198.51.100.7",
                           "x-forwarded-client-ip": "203.0.113.99"})
        assert gw.client_ip(req) == "198.51.100.7"

    def test_forwarded_header_honoured_off_edge(self):
        req = _HdrRequest({"x-forwarded-client-ip": "203.0.113.99"})
        assert gw.client_ip(req) == "203.0.113.99"

    def test_socket_peer_last(self):
        assert gw.client_ip(_HdrRequest({}, host="100.64.0.85")) == "100.64.0.85"


# ---------------------------------------------------------------------------
# apply_ctx_limit (pure function)
# ---------------------------------------------------------------------------

class TestApplyCtxLimit:
    def test_small_prompt_sets_max_tokens(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = gw.apply_ctx_limit(payload, 8192)
        assert out["max_tokens"] <= 8192
        assert out["max_tokens"] > 0

    def test_max_tokens_is_capped_never_inflated(self):
        """A client's own ceiling is honoured whenever it is a real answer
        budget: only values under the floor (below) are lifted, and nothing is
        ever raised beyond what the client asked for."""
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000}
        out = gw.apply_ctx_limit(payload, 8192)
        assert out["max_tokens"] == 1000

    def test_max_tokens_is_clipped_to_the_context_cap(self):
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}
        out = gw.apply_ctx_limit(payload, 4096)
        assert out["max_tokens"] < 4096

    def test_oversized_prompt_raises_413(self):
        big = "x" * 100_000
        payload = {"messages": [{"role": "user", "content": big}]}
        with pytest.raises(gw.HTTPException) as exc:
            gw.apply_ctx_limit(payload, 2048)
        assert exc.value.status_code == 413
        detail = exc.value.detail
        assert detail["error"]["type"] == "context_limit"
        assert detail["error"]["limit"] == 2048

    def test_tiny_max_tokens_is_raised_to_the_floor(self):
        """A thinking model spends the same budget on its preamble: gemma4-12b
        with max_tokens=12 came back finish_reason=length, content "". A demo
        key must not answer a recruiter with an empty string."""
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12}
        out = gw.apply_ctx_limit(payload, 8192)
        assert out["max_tokens"] == gw.PUBLIC_MIN_COMPLETION

    def test_too_little_room_for_a_real_answer_is_rejected(self):
        # A prompt that leaves less than the floor used to be squeezed into
        # whatever was left -- and a thinking model spends that sliver on
        # reasoning and answers with "", a 200 that completes nothing. The
        # honest reply is the 413, which names the numbers a caller (or an
        # agentic client trimming its history) can act on.
        big = "x" * 3000
        payload = {"messages": [{"role": "user", "content": big}], "max_tokens": 8}
        with pytest.raises(gw.HTTPException) as exc:
            gw.apply_ctx_limit(payload, 1200)
        assert exc.value.status_code == 413
        assert exc.value.detail["error"]["limit"] == 1200

    def test_omitted_max_tokens_gets_an_agentic_budget(self):
        # The 2026-08-25 incident: Cline omits max_tokens, the gateway used to
        # invent 1024, and a thinking model burned it all on reasoning -- one
        # visible sentence, finish_reason=length, task dead. No stated budget
        # now buys PUBLIC_DEFAULT_COMPLETION (room permitting).
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = gw.apply_ctx_limit(payload, 131072)
        assert out["max_tokens"] == gw.PUBLIC_DEFAULT_COMPLETION

    def test_nonpositive_max_tokens_means_unset(self):
        # Cline sends -1 for "model default"; some SDKs send 0. Both used to
        # collapse into the floor via min(-1, room).
        for sentinel in (-1, 0, None):
            payload = {"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": sentinel}
            out = gw.apply_ctx_limit(payload, 131072)
            assert out["max_tokens"] == gw.PUBLIC_DEFAULT_COMPLETION, sentinel

    def test_default_budget_is_still_bounded_by_the_room(self):
        big = "x" * 20000  # ~6900 estimated tokens of an 8192 window
        payload = {"messages": [{"role": "user", "content": big}]}
        out = gw.apply_ctx_limit(payload, 8192)
        assert gw.PUBLIC_MIN_COMPLETION <= out["max_tokens"] < gw.PUBLIC_DEFAULT_COMPLETION

    def test_uses_prompt_when_no_messages(self):
        payload = {"prompt": "hello there"}
        out = gw.apply_ctx_limit(payload, 4096)
        assert "max_tokens" in out


# ---------------------------------------------------------------------------
# require_api_key: hourly budget + the /v1/team-% exclusion
# ---------------------------------------------------------------------------

class _FakeRequest:
    """Just enough of Starlette's Request for require_api_key: it reads
    .headers, .query_params, and .url.path -- the last because the endpoints a
    request budget does not COUNT are also the ones it must not BLOCK."""

    def __init__(self, bearer: str, path: str = "/v1/chat/completions"):
        self.headers = {"authorization": "Bearer " + bearer}
        self.query_params: dict = {}
        self.url = SimpleNamespace(path=path)


class TestRequireApiKeyBudgets:
    def test_hourly_budget_counts_client_facing_only(self):
        raw, meta = gw.mint_key("budget-test")
        gw.db_exec("UPDATE api_keys SET max_rph=2 WHERE id=?", (meta["id"],))
        req = _FakeRequest(raw)

        # Two ordinary client-facing requests: allowed through, and each one
        # recorded the way the proxy would after the fact.
        gw.require_api_key(req)
        gw.record_usage({"id": meta["id"], "name": "budget-test"}, "m",
                        "/v1/chat/completions", False, 200, None, None, 5)
        gw.require_api_key(req)
        gw.record_usage({"id": meta["id"], "name": "budget-test"}, "m",
                        "/v1/chat/completions", False, 200, None, None, 5)

        # A third, over budget, is refused before any work happens.
        with pytest.raises(gw.HTTPException) as exc:
            gw.require_api_key(req)
        assert exc.value.status_code == 429

    def test_a_spent_budget_still_lets_the_client_list_models(self):
        """The endpoints a request budget does not COUNT are also the ones it
        must not BLOCK. A client that cannot list models reports itself broken
        ("no models available") rather than rate limited -- which is the one
        thing the caller needed to be told."""
        raw, meta = gw.mint_key("budget-test-5")
        gw.db_exec("UPDATE api_keys SET max_rph=1 WHERE id=?", (meta["id"],))
        gw.record_usage({"id": meta["id"], "name": "budget-test-5"}, "m",
                        "/v1/chat/completions", False, 200, None, None, 5)

        with pytest.raises(gw.HTTPException) as exc:
            gw.require_api_key(_FakeRequest(raw, "/v1/chat/completions"))
        assert exc.value.status_code == 429

        # Same key, same spent hour -- these two still answer.
        assert gw.require_api_key(_FakeRequest(raw, "/v1/models"))["id"] == meta["id"]
        assert gw.require_api_key(_FakeRequest(raw, "/v1/warm"))["id"] == meta["id"]

    def test_team_rounds_and_dead_backends_do_not_count(self):
        raw, meta = gw.mint_key("budget-test-2")
        gw.db_exec("UPDATE api_keys SET max_rph=1 WHERE id=?", (meta["id"],))
        key = {"id": meta["id"], "name": "budget-test-2"}
        req = _FakeRequest(raw)

        # A team's internal rounds/workers, and a request a dead backend
        # never served, must not eat the budget.
        gw.record_usage(key, "m", "/v1/team-primary", False, 200, None, None, 5)
        gw.record_usage(key, "m", "/v1/team-worker", False, 200, None, None, 5)
        gw.record_usage(key, "m", "/v1/chat/completions", False, 503, None, None, 5)
        gw.require_api_key(req)  # still under budget

        gw.record_usage(key, "m", "/v1/chat/completions", False, 200, None, None, 5)
        with pytest.raises(gw.HTTPException) as exc:
            gw.require_api_key(req)
        assert exc.value.status_code == 429

    def test_an_abandoned_request_does_not_count(self):
        """Every OpenAI SDK retries by default, so a slow first call can be
        served twice while the caller sees one reply. The proxy meters the
        abandoned attempt as 499; the budget must ignore it (the tokens are
        still recorded -- the box did the work)."""
        raw, meta = gw.mint_key("budget-test-4")
        gw.db_exec("UPDATE api_keys SET max_rph=1 WHERE id=?", (meta["id"],))
        key = {"id": meta["id"], "name": "budget-test-4"}
        req = _FakeRequest(raw)

        gw.record_usage(key, "m", "/v1/chat/completions", False, 499,
                        {"total_tokens": 400}, None, 5)
        gw.require_api_key(req)  # the retry that the client never received

        gw.record_usage(key, "m", "/v1/chat/completions", False, 200, None, None, 5)
        with pytest.raises(gw.HTTPException) as exc:
            gw.require_api_key(req)
        assert exc.value.status_code == 429

    def test_model_listing_and_failed_requests_do_not_count(self):
        """Every chat client calls /v1/models on startup, and a typo'd model
        or an over-long prompt costs the fleet nothing -- none of those may
        spend a 2/hour allowance. Found on the local smoke run: listing the
        models ate one of the two requests."""
        raw, meta = gw.mint_key("budget-test-3")
        gw.db_exec("UPDATE api_keys SET max_rph=1 WHERE id=?", (meta["id"],))
        key = {"id": meta["id"], "name": "budget-test-3"}
        req = _FakeRequest(raw)

        gw.record_usage(key, "", "/v1/models", False, 200, None, None, 5)
        gw.record_usage(key, "m", "/v1/chat/completions", False, 404, None, None, 5)
        gw.record_usage(key, "m", "/v1/chat/completions", False, 403, None, None, 5)
        gw.require_api_key(req)  # nothing above counted

        gw.record_usage(key, "m", "/v1/chat/completions", True, 200, None, None, 5)
        with pytest.raises(gw.HTTPException) as exc:
            gw.require_api_key(req)
        assert exc.value.status_code == 429


# ---------------------------------------------------------------------------
# fallback selection
# ---------------------------------------------------------------------------

class TestFallback:
    async def test_unreachable_picks_same_arch_closest_active(self, fake_fleet):
        cat = gw.public_catalogue(force=True)["by_public"]
        req_row = cat["nemotron3.5-lightning-30b"]  # moe, active_b=3, unreachable
        settings = gw.get_public_settings()
        chosen = await gw.pick_fallback(req_row, "primary", settings)
        assert chosen is not None
        # Both gemma4-26b-a4b (diff 1) and qwen3.6-35b-a3b (diff 0) are
        # online-but-not-resident candidates in the fake fleet -- the
        # smaller active-params distance wins.
        assert chosen["public_id"] == "qwen3.6-35b-a3b"

    async def test_resident_beats_closer_active_when_both_qualify(self, monkeypatch):
        # Requested active_b = 4: gemma4-26b-a4b (active_b 4, diff 0) is the
        # closer match by parameters but is NOT resident; qwen3.6-35b-a3b
        # (active_b 3, diff 1) IS resident. Residency must win the tie-break
        # ahead of the smaller active-params distance.
        gw._routes_cache.update(
            t=time.time(),
            map={},
            cands={"gemma4-31b-qat": [""], "gemma-4-26b": [""],
                  "qwen3.6-35b": [""]},
            cap={("", "gemma4-31b-qat"): 1, ("", "gemma-4-26b"): 1,
                ("", "qwen3.6-35b"): 1},
            running={gw.HOST_NAME: {"qwen3.6-35b"}},
        )

        async def _routes(force: bool = False):
            return gw._routes_cache["map"]

        monkeypatch.setattr(gw, "model_routes", _routes)
        cat = gw.public_catalogue(force=True)["by_public"]
        req_row = dict(cat["nemotron3-super-120b"])
        req_row["arch"] = "moe"
        req_row["active_b"] = 4.0
        req_row["fleet_ids"] = '["totally-unreachable-fleet-id"]'
        chosen = await gw.pick_fallback(req_row, "primary", gw.get_public_settings())
        assert chosen is not None
        assert chosen["public_id"] == "qwen3.6-35b-a3b"

    async def test_no_swap_when_the_substitute_is_also_cold(self, monkeypatch):
        """Under not_resident the swap exists to skip a cold load. If the best
        candidate would also have to be loaded, the caller keeps the model they
        chose -- same wait either way. Seen live on the first team run:
        qwen3.5-4b was replaced by nemotron-3-nano-4b, neither resident."""
        gw._routes_cache.update(
            t=time.time(), map={},
            cands={"qwen3.5:4b": [""], "nemotron-3-nano:4b": [""]},
            cap={("", "qwen3.5:4b"): 1, ("", "nemotron-3-nano:4b"): 1},
            running={gw.HOST_NAME: set()},   # nothing resident anywhere
        )

        async def _routes(force: bool = False):
            return gw._routes_cache["map"]

        monkeypatch.setattr(gw, "model_routes", _routes)
        cat = gw.public_catalogue(force=True)["by_public"]
        chosen = await gw.pick_fallback(cat["qwen3.5-4b"], "worker",
                                        gw.get_public_settings())
        assert chosen is None

    async def test_swap_still_happens_when_the_substitute_is_resident(self, monkeypatch):
        gw._routes_cache.update(
            t=time.time(), map={},
            cands={"qwen3.5:4b": [""], "nemotron-3-nano:4b": [""]},
            cap={("", "qwen3.5:4b"): 1, ("", "nemotron-3-nano:4b"): 1},
            running={gw.HOST_NAME: {"nemotron-3-nano:4b"}},
        )

        async def _routes(force: bool = False):
            return gw._routes_cache["map"]

        monkeypatch.setattr(gw, "model_routes", _routes)
        cat = gw.public_catalogue(force=True)["by_public"]
        chosen = await gw.pick_fallback(cat["qwen3.5-4b"], "worker",
                                        gw.get_public_settings())
        assert chosen is not None and chosen["public_id"] == "nemotron-3-nano-4b"

    async def test_never_disables_fallback(self, fake_fleet):
        cat = gw.public_catalogue(force=True)["by_public"]
        req_row = cat["nemotron3.5-lightning-30b"]
        settings = dict(gw.get_public_settings())
        settings["fallback_when"] = "never"
        chosen = await gw.pick_fallback(req_row, "primary", settings)
        assert chosen is None

    async def test_resident_original_needs_no_fallback(self, fake_fleet):
        cat = gw.public_catalogue(force=True)["by_public"]
        req_row = cat["gemma4-31b-qat"]  # resident in the fake fleet
        settings = gw.get_public_settings()  # default not_resident
        chosen = await gw.pick_fallback(req_row, "single", settings)
        assert chosen is None


# ---------------------------------------------------------------------------
# team fallback (contract 1.9c): a team key's primary/worker selection must
# get the same substitution a single key's does -- pick_fallback's own
# role='primary'/'worker' branches were previously never invoked.
# ---------------------------------------------------------------------------

class TestTeamFallback:
    async def test_team_primary_fallback_applies(self, client, intake_headers,
                                                  captured_mail, fake_fleet, monkeypatch):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "crew2@nasa.gov", "kind": "team",
                 # unreachable in the fake fleet -- same row TestFallback
                 # confirms falls back to qwen3.6-35b-a3b.
                 "primary": "nemotron3.5-lightning-30b", "workers": ["gemma4-e4b"],
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200, r.text
        row = gw.db_query(
            "SELECT * FROM api_keys WHERE name=?", ("fleet-pass:crew2@nasa.gov",)
        )[0]
        team = gw.get_team(row["id"])
        assert team is not None

        seen_models: list[str] = []

        async def _fake_fleet_chat(key, body, endpoint, ctx_limit=None, **kw):
            seen_models.append(str(body.get("model")))
            return 200, {
                "id": "chatcmpl-x", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }, gw.HOST_NAME, 0

        monkeypatch.setattr(gw, "fleet_chat", _fake_fleet_chat)

        payload = {"model": "team", "messages": [{"role": "user", "content": "hi"}]}
        resp = await gw.team_orchestrate(row, team, payload, False, time.time())
        assert seen_models == ["qwen3.6-35b-a3b"]  # routed against the served model
        body = json.loads(resp.body)
        assert body["x_fleet"]["requested"] == "nemotron3.5-lightning-30b"
        assert body["x_fleet"]["served"] == "qwen3.6-35b-a3b"
        assert "Fleet notice" in body["choices"][0]["message"]["content"]
        assert resp.headers.get("x-fleet-fallback") is not None

        pub_row = gw.db_query(
            "SELECT fallbacks FROM public_keys WHERE key_id=?", (row["id"],)
        )[0]
        assert pub_row["fallbacks"] == 1


# ---------------------------------------------------------------------------
# public request flow
# ---------------------------------------------------------------------------

class TestPublicRequestFlow:
    def test_issued_flow_mails_the_key(self, client, intake_headers, captured_mail,
                                       fake_fleet):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "engineer@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "issued"
        assert "nasa.gov" in body["message"] or "engineer@nasa.gov" in body["message"]

        assert len(captured_mail) == 1
        assert "sk-ffa-" in captured_mail[0]["text"]

        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("engineer@nasa.gov",))[0]
        assert row["status"] == "issued"
        assert row["key_id"]

    def test_pending_flow_for_unlisted_domain(self, client, intake_headers,
                                              captured_mail):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "person@some-unlisted-co.example", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 202
        assert r.json()["status"] == "pending"
        row = gw.db_query(
            "SELECT * FROM public_keys WHERE email=?",
            ("person@some-unlisted-co.example",),
        )[0]
        assert row["status"] == "pending"
        assert row["key_id"] is None

    def test_missing_terms_rejected(self, client, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "x@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": False},
        )
        assert r.status_code == 400
        assert r.json() == {"error": "the terms checkbox is required", "code": "terms"}

    def test_bad_kind_rejected(self, client, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "x@nasa.gov", "kind": "solo", "ctx": 8192,
                 "accept_terms": True},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "bad_kind"

    def test_bad_model_rejected(self, client, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "x@nasa.gov", "kind": "single", "model": "not-a-real-model",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "bad_model"

    def test_bad_ctx_rejected(self, client, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "x@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 999999, "accept_terms": True},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "bad_ctx"

    def test_free_mail_rejected_with_403(self, client, intake_headers):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "x@gmail.com", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 403
        assert r.json()["code"] == "free_mail"

    def test_team_request_issues_two_agent_free_key(self, client, intake_headers,
                                                     captured_mail, fake_fleet):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "engineer@army.mil", "kind": "team",
                 "primary": "qwen3.6-35b-a3b", "workers": ["gemma4-e4b"],
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200, r.text
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("engineer@army.mil",))[0]
        assert row["kind"] == "team"
        team = gw.db_query("SELECT * FROM teams WHERE key_id=?", (row["key_id"],))[0]
        assert team["primary_model"] == "qwen3.6-35b-a3b"
        assert "gemma4-e4b" in team["worker_models"]

    def test_ip_limit(self, client, intake_headers, captured_mail, fake_fleet):
        gw.set_public_settings({"ip_requests_per_day": 1})
        headers = dict(intake_headers)
        headers["X-Forwarded-Client-IP"] = "203.0.113.9"
        r1 = client.post(
            "/public/api/request", headers=headers,
            json={"email": "one@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r1.status_code == 200
        r2 = client.post(
            "/public/api/request", headers=headers,
            json={"email": "two@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r2.status_code == 429
        assert r2.json()["code"] == "ip_limit"

    def test_max_keys_per_email_defaults_to_three(self):
        """An employer who mislaid the mail asks again rather than waiting out
        the expiry, so the allowance is three live keys, not one."""
        assert gw.DEFAULT_PUBLIC_SETTINGS["max_keys_per_email"] == 3
        assert gw.get_public_settings()["max_keys_per_email"] == 3

    def test_max_keys_per_email_allows_three_then_refuses(
            self, client, intake_headers, captured_mail, fake_fleet):
        # The per-IP intake limit would fire first at its own default of 3.
        gw.set_public_settings({"ip_requests_per_day": 100})
        body = {"email": "dup@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
               "ctx": 8192, "accept_terms": True}
        for _ in range(3):
            assert client.post("/public/api/request", headers=intake_headers,
                              json=body).status_code == 200
        r4 = client.post("/public/api/request", headers=intake_headers, json=body)
        assert r4.status_code == 409
        assert r4.json()["code"] == "already_active"
        assert "3 live keys" in r4.json()["error"]

    def test_revoking_frees_a_slot(self, client, intake_headers, admin_headers,
                                   captured_mail, fake_fleet):
        """The whole point of the count: a revoked key is not a live key, so
        the address it belonged to may ask for another one immediately."""
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_email": 2})
        body = {"email": "lost@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
               "ctx": 8192, "accept_terms": True}
        for _ in range(2):
            assert client.post("/public/api/request", headers=intake_headers,
                              json=body).status_code == 200
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 409

        first = gw.db_query("SELECT id FROM public_keys WHERE status='issued' "
                            "ORDER BY id LIMIT 1")[0]["id"]
        assert client.post("/admin/api/public/keys/%d/revoke" % first,
                          headers=admin_headers).status_code == 200
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 200

    def test_expired_keys_do_not_count(self, client, intake_headers,
                                       captured_mail, fake_fleet):
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_email": 1})
        body = {"email": "old@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
               "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 200
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 409
        gw.db_exec("UPDATE api_keys SET expires_at='2000-01-01T00:00:00+00:00'")
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 200

    def test_max_keys_per_email_zero_means_no_limit(
            self, client, intake_headers, captured_mail, fake_fleet):
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_email": 0})
        body = {"email": "many@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
               "ctx": 8192, "accept_terms": True}
        for _ in range(5):
            assert client.post("/public/api/request", headers=intake_headers,
                              json=body).status_code == 200

    def test_one_key_per_email_false_migrates_to_no_limit(self):
        """The old switch is gone, but a hub that had deliberately turned it
        OFF must not silently acquire today's default of 3."""
        gw.db_exec(
            "INSERT INTO settings(key,value) VALUES ('public',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps({"one_key_per_email": False}),),
        )
        assert gw.get_public_settings()["max_keys_per_email"] == 0
        # A stored `true` carried no number -- one was all the switch could
        # mean -- so it yields to the new default rather than pinning at 1.
        gw.db_exec(
            "INSERT INTO settings(key,value) VALUES ('public',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps({"one_key_per_email": True}),),
        )
        assert gw.get_public_settings()["max_keys_per_email"] == 3
        # An explicit save wins over both.
        gw.set_public_settings({"max_keys_per_email": 7})
        assert gw.get_public_settings()["max_keys_per_email"] == 7

    def test_plus_addressing_is_the_same_person(self, client, intake_headers,
                                                 captured_mail, fake_fleet):
        """jane+2@ must not buy a spare key for jane@ -- the duplicate check
        compares canonical addresses (local part with any +tag dropped)."""
        gw.set_public_settings({"max_keys_per_email": 1})
        one = {"email": "jane@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
               "ctx": 8192, "accept_terms": True}
        two = dict(one, email="Jane+fleet2@nasa.gov")
        assert client.post("/public/api/request", headers=intake_headers,
                          json=one).status_code == 200
        r2 = client.post("/public/api/request", headers=intake_headers, json=two)
        assert r2.status_code == 409
        assert r2.json()["code"] == "already_active"

    def test_pending_requests_count_against_the_allowance(
            self, client, intake_headers, captured_mail):
        """A pending row is a key waiting to be minted, so it is budgeted with
        the live ones -- otherwise three approvals could land on an address
        that already holds three keys."""
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_email": 2})
        body = {"email": "dup2@some-unlisted-co.example", "kind": "single",
               "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True}
        for _ in range(2):
            assert client.post("/public/api/request", headers=intake_headers,
                              json=body).status_code == 202
        r3 = client.post("/public/api/request", headers=intake_headers, json=body)
        assert r3.status_code == 409
        assert r3.json()["code"] == "already_pending"

    def test_domain_cap(self, client, intake_headers, captured_mail, fake_fleet):
        gw.set_public_settings({"max_keys_per_domain": 1})
        one = {"email": "a@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        two = {"email": "b@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=one).status_code == 200
        r2 = client.post("/public/api/request", headers=intake_headers, json=two)
        assert r2.status_code == 429
        assert r2.json()["code"] == "domain_cap"

    def test_global_cap(self, client, intake_headers, captured_mail, fake_fleet):
        gw.set_public_settings({"max_live_keys": 1})
        one = {"email": "a@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        two = {"email": "b@army.mil", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=one).status_code == 200
        r2 = client.post("/public/api/request", headers=intake_headers, json=two)
        assert r2.status_code == 429
        assert r2.json()["code"] == "global_cap"

    def test_expired_keys_do_not_count_toward_domain_cap(
            self, client, intake_headers, captured_mail, fake_fleet):
        """An expired key's row keeps status='issued' (nothing archives it),
        so the domain cap must count key liveness, not row status -- otherwise
        every key a domain ever held occupies its allowance forever."""
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_domain": 1})
        one = {"email": "a@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        two = {"email": "b@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=one).status_code == 200
        assert client.post("/public/api/request", headers=intake_headers,
                          json=two).status_code == 429
        gw.db_exec("UPDATE api_keys SET expires_at='2000-01-01T00:00:00+00:00'")
        assert client.post("/public/api/request", headers=intake_headers,
                          json=two).status_code == 200

    def test_expired_keys_do_not_count_toward_global_cap(
            self, client, intake_headers, captured_mail, fake_fleet):
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_live_keys": 1})
        one = {"email": "a@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        two = {"email": "b@army.mil", "kind": "single", "model": "gemma4-31b-qat",
              "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=one).status_code == 200
        assert client.post("/public/api/request", headers=intake_headers,
                          json=two).status_code == 429
        gw.db_exec("UPDATE api_keys SET expires_at='2000-01-01T00:00:00+00:00'")
        assert client.post("/public/api/request", headers=intake_headers,
                          json=two).status_code == 200

    def test_date_only_expiry_is_live_until_end_of_day(
            self, client, intake_headers, captured_mail, fake_fleet):
        """A date-only expires_at means the end of that day -- the same
        reading /public/api/key-status gives it. A key expiring today still
        counts as live, so a duplicate request is still refused."""
        from datetime import datetime, timezone
        gw.set_public_settings({"ip_requests_per_day": 100,
                                "max_keys_per_email": 1})
        body = {"email": "today@nasa.gov", "kind": "single",
               "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True}
        assert client.post("/public/api/request", headers=intake_headers,
                          json=body).status_code == 200
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        gw.db_exec("UPDATE api_keys SET expires_at=?", (today,))
        r = client.post("/public/api/request", headers=intake_headers, json=body)
        assert r.status_code == 409
        assert r.json()["code"] == "already_active"

    def test_missing_intake_token_rejected(self, client):
        r = client.post(
            "/public/api/request",
            json={"email": "x@nasa.gov", "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 401

    def test_key_status_unknown_key(self, client, intake_headers):
        r = client.post("/public/api/key-status", headers=intake_headers,
                        json={"key": "sk-ffa-doesnotexist"})
        assert r.status_code == 404
        assert r.json() == {"error": "unknown key"}

    def test_key_status_requires_intake_token(self, client):
        # Same guard as every other /public/api/* POST -- this one used to
        # be the exception, letting anyone skip the the public site-only gate.
        r = client.post("/public/api/key-status", json={"key": "sk-ffa-doesnotexist"})
        assert r.status_code == 401

    def test_key_status_issued_key(self, client, intake_headers, captured_mail,
                                   fake_fleet):
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "status@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        # Pull the raw key back out of the captured email body.
        text = captured_mail[-1]["text"]
        raw = next(line.split("key: ", 1)[1].strip() for line in text.splitlines()
                   if line.startswith("key: "))
        r = client.post("/public/api/key-status", headers=intake_headers, json={"key": raw})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "active"
        assert body["kind"] == "single"
        assert body["limit_day"] == gw.get_public_settings()["single_rpd"]


# ---------------------------------------------------------------------------
# public models / overview
# ---------------------------------------------------------------------------

class TestPublicCatalogueEndpoints:
    def test_models_endpoint_lists_enabled_catalogue(self, client, fake_fleet):
        r = client.get("/public/api/models")
        assert r.status_code == 200
        body = r.json()
        assert len(body["models"]) == 17
        ids = {m["public_id"] for m in body["models"]}
        assert "gemma4-31b-qat" in ids
        assert "qwen3.8-9b-distill" in ids
        resident = next(m for m in body["models"] if m["public_id"] == "gemma4-31b-qat")
        assert resident["availability"] == "resident"
        offline = next(m for m in body["models"]
                      if m["public_id"] == "nemotron3-super-120b")
        assert offline["availability"] == "offline"
        assert "single_rpd" in body["limits"]

    def _walk(self, node, forbidden, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                assert k.lower() not in forbidden, "forbidden key %r at %s" % (k, path)
                self._walk(v, forbidden, path + "." + str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                self._walk(v, forbidden, path + "[%d]" % i)

    def test_overview_never_leaks_forbidden_fields(self, client, fake_fleet):
        r = client.get("/public/api/overview")
        assert r.status_code == 200
        forbidden = {"api_url", "url", "ip", "net", "storage", "disk", "services",
                    "temps", "error", "swap", "token", "tokens", "mounts", "peers", "os"}
        self._walk(r.json(), forbidden)

    def test_overview_uses_box_aliases_never_real_names_or_os(
        self, client, fake_fleet, monkeypatch,
    ):
        """The owner requirement added after the first draft (contract 1.6):
        public surfaces show 'Box N'/'Hub', never the real host name or any
        OS detail -- verified here against a fake host_status() carrying
        deliberately distinctive real-looking values that must not survive
        sanitization."""
        real_host_status = gw.host_status

        def _fake_host_status():
            d = real_host_status()
            d["os"] = {"name": "TotallyRealOS 9", "arch": "sparc64"}
            d["engine"] = {"kind": "llama-swap",
                           "name": "llama.cpp (llama-swap v9.9.9-secret)"}
            return d

        monkeypatch.setattr(gw, "host_status", _fake_host_status)
        gw._public_overview_cache.update(t=0.0, data=None)  # bypass the 10s cache

        r = client.get("/public/api/overview")
        assert r.status_code == 200
        body = r.json()
        raw = r.text

        assert "hub" not in body  # no top-level raw-HOST_NAME field either
        assert gw.HOST_NAME not in raw
        assert "TotallyRealOS" not in raw
        assert "sparc64" not in raw
        assert "secret" not in raw

        hub_host = next(h for h in body["hosts"] if h["online"])
        assert hub_host["name"] == "Hub"
        assert hub_host["engine"] == "llama.cpp"  # kind only, no version/name
        assert "os" not in hub_host


# ---------------------------------------------------------------------------
# /v1/models filtering for a Fleet Pass key
# ---------------------------------------------------------------------------

class TestV1ModelsFiltering:
    def test_single_key_sees_only_its_model(self, client, intake_headers,
                                            captured_mail, fake_fleet):
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "solo@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        text = captured_mail[-1]["text"]
        raw = next(line.split("key: ", 1)[1].strip() for line in text.splitlines()
                   if line.startswith("key: "))
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        data = r.json()["data"]
        assert [m["id"] for m in data] == ["gemma4-31b-qat"]
        # The one field a client can read the key's window from: without it,
        # a stale context figure in the client's provider profile outlives
        # every new key.
        assert [m["context_length"] for m in data] == [8192]

    def test_team_key_sees_team_and_primary(self, client, intake_headers,
                                            captured_mail, fake_fleet):
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "crew@nasa.gov", "kind": "team",
                 "primary": "qwen3.6-35b-a3b", "workers": ["gemma4-e4b"],
                 "ctx": 8192, "accept_terms": True},
        )
        text = captured_mail[-1]["text"]
        raw = next(line.split("key: ", 1)[1].strip() for line in text.splitlines()
                   if line.startswith("key: "))
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        data = r.json()["data"]
        assert [m["id"] for m in data] == ["team", "qwen3.6-35b-a3b"]
        assert [m["context_length"] for m in data] == [8192, 8192]

    def test_key_email_setup_tells_the_client_its_window(
            self, client, intake_headers, captured_mail, fake_fleet):
        """The default setup text now names the granted window next to the
        Cline instruction, so the number the key enforces is the number the
        user configures -- not whatever their provider profile last held."""
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "window@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        text = captured_mail[-1]["text"]
        assert "set Context Window Size to 8192" in text


# ---------------------------------------------------------------------------
# surface restriction (contract 1.9h): a public key reaches exactly two
# routes -- POST /v1/chat/completions and GET /v1/models. Everything else,
# including the native /api/* proxy, is 403.
# ---------------------------------------------------------------------------

class TestPublicSurfaceRestriction:
    def _issued_key(self, client, intake_headers, captured_mail,
                    email="surface@nasa.gov"):
        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": email, "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        text = captured_mail[-1]["text"]
        return next(line.split("key: ", 1)[1].strip() for line in text.splitlines()
                    if line.startswith("key: "))

    def _fake_chat_send(self, seen: dict | None = None):
        async def _send(request, stream=False, **kwargs):
            if seen is not None:
                seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-abc", "object": "chat.completion",
                    "model": "gemma4-31b-qat",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                             "total_tokens": 2},
                },
                request=request,
            )
        return _send

    def test_batches_forbidden(self, client, intake_headers, captured_mail, fake_fleet):
        raw = self._issued_key(client, intake_headers, captured_mail)
        r = client.post("/v1/batches", headers={"Authorization": "Bearer " + raw},
                        json={"model": "gemma4-31b-qat", "requests": []})
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "permission_error"

    def test_embeddings_forbidden(self, client, intake_headers, captured_mail, fake_fleet):
        raw = self._issued_key(client, intake_headers, captured_mail)
        r = client.post("/v1/embeddings", headers={"Authorization": "Bearer " + raw},
                        json={"model": "gemma4-31b-qat", "input": "hi"})
        assert r.status_code == 403

    def test_native_api_forbidden(self, client, intake_headers, captured_mail, fake_fleet):
        raw = self._issued_key(client, intake_headers, captured_mail)
        r = client.post("/api/chat", headers={"Authorization": "Bearer " + raw},
                        json={"model": "gemma4-31b-qat", "messages": []})
        assert r.status_code == 403

    def test_chat_completions_allowed(self, client, intake_headers, captured_mail,
                                      fake_fleet, monkeypatch):
        raw = self._issued_key(client, intake_headers, captured_mail)
        monkeypatch.setattr(gw.client, "send", self._fake_chat_send())
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "gemma4-31b-qat",
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text

    def test_models_allowed(self, client, intake_headers, captured_mail, fake_fleet):
        raw = self._issued_key(client, intake_headers, captured_mail)
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200

    def test_plain_key_batches_still_works(self, client):
        # A public row (any status) restricts a key permanently; an ordinary
        # bearer key must see no change at all.
        raw, meta = gw.mint_key("plain-key-batches")
        r = client.get("/v1/batches", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        assert r.json() == {"batches": []}

    def test_n_pinned_and_sampler_knobs_stripped(self, client, intake_headers,
                                                  captured_mail, fake_fleet, monkeypatch):
        raw = self._issued_key(client, intake_headers, captured_mail)
        seen: dict = {}
        monkeypatch.setattr(gw.client, "send", self._fake_chat_send(seen))
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer " + raw},
            json={"model": "gemma4-31b-qat", "n": 5, "best_of": 3, "logprobs": 5,
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert seen["body"]["n"] == 1
        assert "best_of" not in seen["body"]
        assert "logprobs" not in seen["body"]


# ---------------------------------------------------------------------------
# admin: box aliases (contract 1.6/1.10 §8) -- the mapping the admin tab
# shows so an operator can tell which real box is 'Box N'.
# ---------------------------------------------------------------------------

class TestAdminAliases:
    def test_lists_hub_and_assigns_boxes_on_first_sight(self, client, admin_headers):
        # Other tests may already have assigned box numbers to other hosts
        # (e.g. the fake fleet's 'peer1'), so this only checks the properties
        # public_alias promises -- not a specific number -- for two hosts
        # unique to this test.
        assert gw.public_alias(gw.HOST_NAME) == "Hub"
        first = gw.public_alias("some-peer-xyz")
        assert first.startswith("Box ")
        assert gw.public_alias("some-peer-xyz") == first  # stable on repeat lookups
        second = gw.public_alias("another-peer-xyz")
        assert second.startswith("Box ") and second != first

        r = client.get("/admin/api/public/aliases", headers=admin_headers)
        assert r.status_code == 200
        items = {i["host"]: i["label"] for i in r.json()["items"]}
        assert items[gw.HOST_NAME] == "Hub"
        assert items["some-peer-xyz"] == first
        assert items["another-peer-xyz"] == second

    def test_requires_auth(self, client):
        r = client.get("/admin/api/public/aliases")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# plain (non-public) key: the existing /v1 path must not regress
# ---------------------------------------------------------------------------

class TestPlainKeyUnaffected:
    def test_plain_key_with_no_public_row_proxies_unchanged(self, client, monkeypatch):
        async def _fake_model_hosts(model, **kw):
            return [""]

        async def _fake_send(request, stream=False, **kwargs):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-abc", "object": "chat.completion",
                    "model": "some-private-fleet-model",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                             "total_tokens": 5},
                },
                request=request,
            )

        monkeypatch.setattr(gw, "model_hosts", _fake_model_hosts)
        monkeypatch.setattr(gw.client, "send", _fake_send)

        raw, meta = gw.mint_key("plain-key")
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + raw},
            json={"model": "some-private-fleet-model",
                 "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "hello"
        # No Fleet Pass machinery engaged for an ordinary key.
        assert "x_fleet" not in body
        assert "X-Fleet-Fallback" not in r.headers

        row = gw.db_query(
            "SELECT * FROM usage WHERE key_id=? ORDER BY id DESC LIMIT 1",
            (meta["id"],),
        )[0]
        assert row["model"] == "some-private-fleet-model"
        assert row["fallback_from"] is None
        assert row["status"] == 200


# ---------------------------------------------------------------------------
# admin: approve / deny / revoke / extend / resend
# ---------------------------------------------------------------------------

class TestAdminKeyActions:
    def _pending_request(self, client, intake_headers, email="review@some-unlisted-co.example"):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": email, "kind": "single", "model": "gemma4-31b-qat",
                 "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 202
        return gw.db_query("SELECT * FROM public_keys WHERE email=?", (email,))[0]

    def test_admin_endpoints_require_auth(self, client):
        r = client.get("/admin/api/public/keys")
        assert r.status_code == 401

    def test_approve_issues_and_mails_key(self, client, intake_headers, admin_headers,
                                          captured_mail):
        row = self._pending_request(client, intake_headers)
        r = client.post("/admin/api/public/keys/%d/approve" % row["id"],
                       headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "issued"
        assert len(captured_mail) == 1
        assert "sk-ffa-" in captured_mail[0]["text"]

    def test_deny_mails_a_reason(self, client, intake_headers, admin_headers,
                                 captured_mail):
        row = self._pending_request(client, intake_headers, email="nope@some-unlisted-co.example")
        r = client.post("/admin/api/public/keys/%d/deny" % row["id"],
                       headers=admin_headers, json={"reason": "not a match for this program"})
        assert r.status_code == 200
        assert r.json()["status"] == "denied"
        assert len(captured_mail) == 1
        assert "not a match" in captured_mail[0]["text"]

    def test_revoke_disables_the_api_key(self, client, intake_headers, admin_headers,
                                         captured_mail):
        row = self._pending_request(client, intake_headers, email="rev@some-unlisted-co.example")
        client.post("/admin/api/public/keys/%d/approve" % row["id"], headers=admin_headers)
        approved = gw.db_query("SELECT * FROM public_keys WHERE id=?", (row["id"],))[0]
        key_id = approved["key_id"]

        r = client.post("/admin/api/public/keys/%d/revoke" % row["id"], headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["status"] == "revoked"
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (key_id,))[0]
        assert key_row["archived_at"] is not None
        assert key_row["disabled"] == 1

    def test_extend_pushes_expiry_forward(self, client, intake_headers, admin_headers,
                                          captured_mail):
        row = self._pending_request(client, intake_headers, email="ext@some-unlisted-co.example")
        client.post("/admin/api/public/keys/%d/approve" % row["id"], headers=admin_headers)
        before = gw.db_query(
            "SELECT k.expires_at exp FROM public_keys pk JOIN api_keys k "
            "ON k.id=pk.key_id WHERE pk.id=?", (row["id"],),
        )[0]["exp"]
        r = client.post("/admin/api/public/keys/%d/extend" % row["id"], headers=admin_headers)
        assert r.status_code == 200
        after = gw.db_query(
            "SELECT k.expires_at exp FROM public_keys pk JOIN api_keys k "
            "ON k.id=pk.key_id WHERE pk.id=?", (row["id"],),
        )[0]["exp"]
        assert after > before

    def test_resend_mints_a_replacement_key(self, client, intake_headers, admin_headers,
                                            captured_mail):
        row = self._pending_request(client, intake_headers, email="resend@some-unlisted-co.example")
        client.post("/admin/api/public/keys/%d/approve" % row["id"], headers=admin_headers)
        first = gw.db_query("SELECT * FROM public_keys WHERE id=?", (row["id"],))[0]
        old_key_id = first["key_id"]

        r = client.post("/admin/api/public/keys/%d/resend" % row["id"], headers=admin_headers)
        assert r.status_code == 200
        second = r.json()
        assert second["status"] == "issued"
        assert second["key_id"] != old_key_id

        old_key = gw.db_query("SELECT * FROM api_keys WHERE id=?", (old_key_id,))[0]
        assert old_key["archived_at"] is not None
        assert len(captured_mail) == 2
        assert captured_mail[0]["text"] != captured_mail[1]["text"]

    def test_manual_issue_skips_eligibility(self, client, admin_headers, captured_mail):
        # A gmail.com address would be a hard `free_mail` rejection on the
        # public intake -- the owner vouching for someone by hand (contract
        # 1.10) must not run any of that eligibility/cap/free-mail machinery.
        r = client.post(
            "/admin/api/public/keys", headers=admin_headers,
            json={"email": "friend@gmail.com", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "note": "vouched for"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "issued"
        assert body["source"] == "manual"
        assert body["decided_by"] == "admin-token"
        assert len(captured_mail) == 1
        assert "sk-ffa-" in captured_mail[0]["text"]

        row = gw.db_query("SELECT * FROM public_keys WHERE id=?", (body["id"],))[0]
        assert row["email"] == "friend@gmail.com"
        assert row["key_id"] is not None

    def test_manual_issue_still_validates_model(self, client, admin_headers):
        r = client.post(
            "/admin/api/public/keys", headers=admin_headers,
            json={"email": "friend@gmail.com", "kind": "single",
                 "model": "not-a-real-model", "ctx": 8192},
        )
        assert r.status_code == 400

    def test_manual_issue_requires_auth(self, client):
        r = client.post(
            "/admin/api/public/keys",
            json={"email": "friend@gmail.com", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# hardware-aware context ceilings
#
# The catalogue used to hand every model the same 32768. These cover the three
# things that replaced it: what each BOX reports it can serve, how the hub
# turns those reports into one advertised ceiling per model, and what happens
# at call time when the box that could honour a key's window is not around.
# ---------------------------------------------------------------------------

class TestPerBoxCtxReporting:
    """What a single box reports about itself."""

    def test_ollama_kv_meta_flattens_namespaced_geometry(self):
        meta = gw._ollama_kv_meta({
            "qwen3moe.block_count": 48,
            "qwen3moe.attention.head_count_kv": 4,
            "qwen3moe.attention.head_count": 32,
            "qwen3moe.attention.key_length": 128,
            "qwen3moe.attention.value_length": 128,
            "qwen3moe.embedding_length": 2048,
            "qwen3moe.context_length": 262144,
            "tokenizer.ggml.model": "gpt2",
        })
        assert meta["block_count"] == 48
        assert meta["head_count_kv"] == 4
        assert meta["key_length"] == 128
        assert meta["n_ctx_train"] == 262144
        # Namespaced keys that are not geometry must not survive the flatten.
        assert "model" not in meta
        # ...and the flattened shape has to be what the KV sizer actually eats.
        # q8_0 is 8.5 bits per weight, not 8: 48 layers x 4 KV heads x
        # (128 + 128) x 8.5 / 8 bytes.
        assert gw.kv_bytes_per_token(meta, "q8_0", "q8_0") == 48 * 4 * 272

    def test_ollama_kv_meta_ignores_junk(self):
        assert gw._ollama_kv_meta(None) == {}
        assert gw._ollama_kv_meta({"no_dot": 5}) == {}
        assert gw._ollama_kv_meta({"a.block_count": "not a number"}) == {}

    def test_local_model_ctx_reports_the_pinned_window(self, monkeypatch):
        # A pinned record is a promise: resolve_ctx honours it exactly, and
        # that promise is what the box advertises.
        monkeypatch.setattr(gw, "load_models", lambda: [
            {"id": "big", "path": "/nope.gguf", "enabled": True, "ctx": 131072,
             "aliases": ["huge"]},
            {"id": "small", "path": "/nope.gguf", "enabled": True, "ctx": 8192},
            {"id": "off", "path": "/nope.gguf", "enabled": False, "ctx": 65536},
        ])
        out = gw.local_model_ctx()
        assert out == {"big": 131072, "huge": 131072, "small": 8192}

    def test_resolve_ctx_credits_expert_tensors_pushed_to_ram(self, monkeypatch):
        """--n-cpu-moe is the whole reason an 8 GB laptop can serve a 20 GB MoE.

        Charging the GPU for weights that are sitting in DRAM sized this box
        down to the 4096 floor, which is a number it would then have promised
        the fleet."""
        meta = {"block_count": 48, "head_count_kv": 4, "key_length": 128,
                "value_length": 128, "n_ctx_train": 262144, "arch": "qwen3moe"}
        monkeypatch.setattr(gw, "gguf_meta", lambda p: meta)
        monkeypatch.setattr(gw, "model_bytes", lambda p: 20 * 1024 ** 3 if p else 0)
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: 8 * 1024 ** 3)
        rec = {"id": "m", "path": "/x.gguf", "ctx": 0,
               "cache_type_k": "q8_0", "cache_type_v": "q8_0"}

        no_offload, _ = gw.resolve_ctx(rec)
        assert no_offload == 4096, "20 GB of weights really do not fit in 8 GB"

        rec["n_cpu_moe"] = 46
        offloaded, detail = gw.resolve_ctx(rec)
        assert offloaded > 32768, offloaded
        assert detail["gpu_weights"] < detail["weights"]
        assert detail["n_cpu_moe"] == 46

    async def test_served_models_endpoint_publishes_ctx(self, client, admin_headers,
                                                        monkeypatch):
        monkeypatch.setattr(gw, "load_models", lambda: [
            {"id": "pinned", "path": "/nope.gguf", "enabled": True, "ctx": 49152},
        ])
        r = client.get("/admin/api/served-models", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["ctx"]["pinned"] == 49152


class TestCatalogueCtx:
    """How the hub turns per-box reports into one ceiling per model."""

    async def test_ceiling_differs_per_model_and_follows_the_hardware(
            self, client, fake_fleet):
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        # Straight from the fake fleet's per-(host, model) ceilings.
        assert rows["gemma4-31b-qat"]["ctx_max"] == 32768
        assert rows["gemma4-26b-a4b"]["ctx_max"] == 16384
        assert rows["qwen3.6-35b-a3b"]["ctx_max"] == 8192
        assert rows["qwen3.8-27b"]["ctx_max"] == 131072
        # The point of the whole exercise: not one number for everything.
        assert len({r["ctx_max"] for r in rows.values()}) > 1

    async def test_unreported_model_holds_the_conservative_default(
            self, client, fake_fleet):
        # nemotron3-super-120b is in the catalogue with a 131072 row ceiling
        # but no box in the fake fleet serves it. Advertising the row ceiling
        # on nothing but a spec sheet would promise a window nothing can hold.
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        assert rows["nemotron3-super-120b"]["ctx_max"] == 32768

    async def test_offline_box_raises_the_ceiling_but_not_the_online_one(
            self, client, fake_fleet):
        # A big box that is switched off right now still counts toward what a
        # week-long key may be issued for -- and the gap is exactly what the
        # call-time notice exists to disclose.
        gw.remember_model_ctx({("bigbox", "gemma4-31b-qat"): 131072})
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        row = rows["gemma4-31b-qat"]
        assert row["ctx_max"] == 131072
        assert row["ctx_max_online"] == 32768
        assert {h["host"] for h in row["ctx_hosts"]} == {
            gw.public_alias("bigbox"), gw.public_alias(gw.HOST_NAME)}

    async def test_advertised_ceiling_is_always_selectable(self, client, fake_fleet):
        # apu-box-1 really does pin qwen3.8-27b at 262114 -- four short of 256K,
        # and not a multiple of 1024, so the intake would reject its own
        # advertised maximum.
        gw.remember_model_ctx({("bigbox", "qwen3.8-27b"): 262114})
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        assert rows["qwen3.8-27b"]["ctx_max"] == 261120

    async def test_row_ceiling_still_caps_the_hardware(self, client, admin_headers,
                                                       fake_fleet):
        listed = client.get("/admin/api/public/models",
                            headers=admin_headers).json()["items"]
        row = next(r for r in listed if r["public_id"] == "qwen3.8-27b")
        original = dict(row)
        capped = dict(row, ctx_max=16384, ctx_default=8192)
        assert client.put("/admin/api/public/models/qwen3.8-27b",
                          headers=admin_headers, json=capped).status_code == 200
        try:
            gw._public_catalogue_cache.update(t=0.0)
            rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
            # peer1 would give it 131072; the owner said 16384.
            assert rows["qwen3.8-27b"]["ctx_max"] == 16384
        finally:
            client.put("/admin/api/public/models/qwen3.8-27b",
                       headers=admin_headers, json=original)
            gw._public_catalogue_cache.update(t=0.0)

    async def test_program_ceiling_caps_everything(self, client, admin_headers,
                                                   fake_fleet):
        client.put("/admin/api/public/settings", headers=admin_headers,
                   json={"ctx_ceiling": 12288})
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        assert max(r["ctx_max"] for r in rows.values()) == 12288

    async def test_ctx_default_never_exceeds_the_ceiling(self, client, fake_fleet):
        rows = await gw.public_models_payload()
        assert all(r["ctx_default"] <= r["ctx_max"] for r in rows)


class TestIntakeCtxBounds:
    def test_rejects_above_the_hardware_ceiling(self, client, intake_headers,
                                                fake_fleet):
        # qwen3.6-35b-a3b is served only by a box that can give it 8192.
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "a@nasa.gov", "kind": "single",
                 "model": "qwen3.6-35b-a3b", "ctx": 32768, "accept_terms": True})
        assert r.status_code == 400
        assert r.json()["code"] == "bad_ctx"
        assert "8192" in r.json()["error"]

    def test_accepts_exactly_the_hardware_ceiling(self, client, intake_headers,
                                                  captured_mail, fake_fleet):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "b@nasa.gov", "kind": "single",
                 "model": "qwen3.6-35b-a3b", "ctx": 8192, "accept_terms": True})
        assert r.status_code == 200, r.text
        row = gw.db_query("SELECT ctx FROM public_keys WHERE email=?",
                          ("b@nasa.gov",))[0]
        assert row["ctx"] == 8192

    def test_a_team_is_bound_by_its_smallest_member(self, client, intake_headers,
                                                    fake_fleet):
        # primary gemma4-26b-a4b (16384) with worker qwen3.6-35b-a3b (8192).
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "c@nasa.gov", "kind": "team",
                 "primary": "gemma4-26b-a4b", "workers": ["qwen3.6-35b-a3b"],
                 "ctx": 16384, "accept_terms": True})
        assert r.status_code == 400
        assert "8192" in r.json()["error"]


class TestCtxReduction:
    """Call time: the window is fitted to the box, and the caller is told."""

    async def test_fleet_chat_fits_the_request_to_the_box(self, monkeypatch):
        async def _targets(model, **kw):
            return [("smallbox", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("smallbox", "m1"): 8192})
        sent = {}

        async def _post(cand, payload, *a, **kw):
            sent.update(json.loads(payload))
            return httpx.Response(
                200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant",
                                                    "content": "ok"}}]})
        monkeypatch.setattr(gw, "_post_chat", _post)

        status, body, host, granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 100000},
            "/v1/chat/completions", 32768)
        assert status == 200
        assert host == "smallbox"
        assert granted == 8192
        # Fitted, not merely reported: the body that went out is bounded by
        # what that box can actually hold.
        assert sent["max_tokens"] <= 8192

    async def test_fleet_chat_leaves_an_honoured_window_alone(self, monkeypatch):
        async def _targets(model, **kw):
            return [("bigbox", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("bigbox", "m1"): 131072})

        async def _post(cand, payload, *a, **kw):
            return httpx.Response(200, json={"choices": []})
        monkeypatch.setattr(gw, "_post_chat", _post)

        _s, _b, _h, granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions", 32768)
        assert granted == 0

    async def test_fleet_chat_prefers_a_box_that_can_honour_the_window(
            self, monkeypatch):
        async def _targets(model, **kw):
            # The scorer put the small box first (it is resident, say).
            return [("smallbox", "m1"), ("bigbox", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={("smallbox", "m1"): 8192,
                                     ("bigbox", "m1"): 131072})
        seen = []

        async def _post(cand, payload, *a, **kw):
            seen.append(cand)
            return httpx.Response(200, json={"choices": []})
        monkeypatch.setattr(gw, "_post_chat", _post)

        _s, _b, host, granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions", 32768)
        assert seen == ["bigbox"], seen
        assert host == "bigbox"
        assert granted == 0

    async def test_unknown_ceiling_is_not_treated_as_zero(self, monkeypatch):
        # A peer on an older gateway reports no ctx at all. It must be served
        # exactly as it always was, not demoted or reduced.
        async def _targets(model, **kw):
            return [("oldpeer", "m1")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        gw._routes_cache.update(ctx={})

        async def _post(cand, payload, *a, **kw):
            return httpx.Response(200, json={"choices": []})
        monkeypatch.setattr(gw, "_post_chat", _post)

        _s, _b, host, granted = await gw.fleet_chat(
            {"id": 1, "name": "k"},
            {"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions", 32768)
        assert host == "oldpeer"
        assert granted == 0


class TestNotices:
    def test_ctx_notice_names_both_numbers(self):
        s = gw.get_public_settings()
        text = gw.render_ctx_notice(32768, 8192, "peer1", s)
        assert "32768" in text and "8192" in text

    def test_ctx_notice_text_is_configurable(self, client, admin_headers):
        client.put("/admin/api/public/settings", headers=admin_headers,
                   json={"ctx_notice_text": "shrunk {requested}->{granted}\n"})
        s = gw.get_public_settings()
        assert gw.render_ctx_notice(100, 50, "", s) == "shrunk 100->50\n"

    def test_a_broken_template_still_produces_a_notice(self):
        s = dict(gw.get_public_settings())
        s["ctx_notice_text"] = "{nope} is not a field"
        assert gw.render_ctx_notice(1, 2, "", s) == "{nope} is not a field"

    def test_substitution_and_reduction_are_disclosed_together(self):
        s = gw.get_public_settings()
        notice, xf, headers = gw.public_notices(
            {"requested": "nemotron3-super-120b", "served": "qwen3.6-35b-a3b"},
            {"requested": 32768, "granted": 8192}, "peer1", s)
        assert "nemotron3-super-120b" in notice and "8192" in notice
        assert xf["served"] == "qwen3.6-35b-a3b"
        assert xf["ctx"] == {"requested": 32768, "granted": 8192, "host": "peer1"}
        assert "X-Fleet-Fallback" in headers and "X-Fleet-Ctx" in headers

    def test_notices_can_be_switched_off_independently(self):
        s = dict(gw.get_public_settings())
        s["ctx_notice"] = False
        notice, xf, headers = gw.public_notices(
            None, {"requested": 32768, "granted": 8192}, "peer1", s)
        assert notice == ""
        # Silent to the reader, still machine-readable to a client that looks.
        assert xf["ctx"]["granted"] == 8192
        assert headers["X-Fleet-Ctx"]

    def test_prepend_notice_leaves_a_tool_call_turn_alone(self):
        obj = {"choices": [{"message": {"role": "assistant", "content": None,
                                        "tool_calls": [{"id": "1"}]}}]}
        gw.prepend_notice(obj, "notice: ")
        assert obj["choices"][0]["message"]["content"] is None

    def test_prepend_notice_survives_an_empty_choice_list(self):
        obj = {"choices": []}
        gw.prepend_notice(obj, "notice: ")  # must not raise
        assert obj == {"choices": []}


class TestTeamCtxReduction:
    async def test_team_primary_reduction_is_disclosed(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "crew3@nasa.gov", "kind": "team",
                 "primary": "gemma4-31b-qat", "workers": ["gemma4-26b-a4b"],
                 "ctx": 16384, "accept_terms": True})
        assert r.status_code == 200, r.text
        row = gw.db_query("SELECT * FROM api_keys WHERE name=?",
                          ("fleet-pass:crew3@nasa.gov",))[0]
        team = gw.get_team(row["id"])

        async def _fake_fleet_chat(key, body, endpoint, ctx_limit=None, **kw):
            assert ctx_limit == 16384
            return 200, {
                "id": "chatcmpl-x", "object": "chat.completion",
                "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }, "peer1", 4096
        monkeypatch.setattr(gw, "fleet_chat", _fake_fleet_chat)

        resp = await gw.team_orchestrate(
            row, team, {"model": "team",
                        "messages": [{"role": "user", "content": "hi"}]},
            False, time.time())
        body = json.loads(resp.body)
        assert body["x_fleet"]["ctx"] == {"requested": 16384, "granted": 4096,
                                          "host": "peer1"}
        assert "16384" in body["choices"][0]["message"]["content"]
        assert resp.headers.get("X-Fleet-Ctx")


# ---------------------------------------------------------------------------
# one physical machine, one entry
# ---------------------------------------------------------------------------

class TestTwinCollapse:
    SPECS = {
        "dualboot-win": {"ram_gb": 32, "vram_gb": 8},
        "dualboot-linux": {"ram_gb": 32, "vram_gb": 8, "replaces": "dualboot-win"},
        "cpu-box-1": {"ram_gb": 32},
    }

    def _hosts(self, linux_online: bool):
        return [
            {"name": "cpu-box-1", "online": True},
            {"name": "dualboot-win", "online": False},
            {"name": "dualboot-linux", "online": linux_online},
        ]

    def test_linux_replaces_windows_when_it_is_up(self):
        out = gw.collapse_twins(self._hosts(True), self.SPECS)
        names = [h["name"] for h in out]
        assert names == ["cpu-box-1", "dualboot-linux"]
        assert out[-1]["replaces"] == "dualboot-win"

    def test_windows_holds_the_slot_when_linux_is_down(self):
        out = gw.collapse_twins(self._hosts(False), self.SPECS)
        assert [h["name"] for h in out] == ["cpu-box-1", "dualboot-win"]

    def test_the_shared_hardware_is_only_listed_once(self):
        for linux_up in (True, False):
            out = gw.collapse_twins(self._hosts(linux_up), self.SPECS)
            assert len(out) == 2, "32 GB of RAM must not be counted twice"

    def test_a_replacement_with_no_original_is_left_alone(self):
        out = gw.collapse_twins(
            [{"name": "dualboot-linux", "online": True}], self.SPECS)
        assert [h["name"] for h in out] == ["dualboot-linux"]

    def test_a_fleet_with_no_twins_is_untouched(self):
        hosts = [{"name": "cpu-box-1", "online": True}]
        assert gw.collapse_twins(hosts, self.SPECS) is hosts

    def test_both_halves_share_one_public_box_number(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: self.SPECS)
        assert gw.public_alias("dualboot-linux") == gw.public_alias("dualboot-win")
        assert gw.public_alias("dualboot-linux") != gw.public_alias("cpu-box-1")

    def test_the_public_card_never_leaks_the_replaces_wiring(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: self.SPECS)
        card = gw._sanitize_public_host(
            {"name": "dualboot-linux", "online": True, "replaces": "dualboot-win",
             "status": {}}, self.SPECS)
        assert "replaces" not in card
        assert card["name"] == gw.public_alias("dualboot-win")


class TestCtxCeilingMigration:
    """The reseed refuses to touch an existing row, which would have left the
    whole catalogue pinned at the flat ceiling this feature replaced."""

    def _set(self, pid: str, value: int) -> None:
        gw.db_update("UPDATE public_models SET ctx_max=? WHERE public_id=?",
                     (value, pid))
        gw.public_catalogue(force=True)

    def _get(self, pid: str) -> int:
        return int(gw.db_query(
            "SELECT ctx_max FROM public_models WHERE public_id=?", (pid,))[0]["ctx_max"])

    def test_an_untouched_row_is_lifted_to_the_trained_context(self):
        original = self._get("qwen3.6-35b-a3b")
        try:
            self._set("qwen3.6-35b-a3b", 32768)   # exactly what the old seed shipped
            assert gw.raise_stale_ctx_ceilings() >= 1
            assert self._get("qwen3.6-35b-a3b") == 262144
        finally:
            self._set("qwen3.6-35b-a3b", original)

    def test_a_hand_edited_row_is_left_alone(self):
        original = self._get("qwen3.6-35b-a3b")
        try:
            self._set("qwen3.6-35b-a3b", 20480)   # nobody's default; an admin chose it
            gw.raise_stale_ctx_ceilings()
            assert self._get("qwen3.6-35b-a3b") == 20480
        finally:
            self._set("qwen3.6-35b-a3b", original)

    def test_running_it_twice_changes_nothing_the_second_time(self):
        original = self._get("muse-glimmer-30b")
        try:
            self._set("muse-glimmer-30b", 32768)
            assert gw.raise_stale_ctx_ceilings() >= 1
            assert gw.raise_stale_ctx_ceilings() == 0
        finally:
            self._set("muse-glimmer-30b", original)

    def test_every_seed_row_has_a_pre_migration_value(self):
        # A row added to the seed later without an entry here would silently
        # never migrate, which is the failure mode that hides for months.
        seeded = {r["public_id"] for r in gw.load_public_models_seed()}
        assert seeded == set(gw._PRE_HARDWARE_CTX_MAX)


class TestSlidingWindowBudget:
    """Gemma 4 E4B publishes a 512-token window and half-width local keys.
    Charging its 42 layers as if they all cached everything is what sized a
    16 GB Mac down to a 2048-token ceiling for that model in production."""

    GEMMA = {"block_count": 42, "head_count_kv": 2, "head_count": 8,
             "key_length": 512, "value_length": 512, "key_length_swa": 256,
             "value_length_swa": 256, "sliding_window": 512,
             "n_ctx_train": 131072}
    DENSE = {"block_count": 36, "head_count_kv": 8, "key_length": 128,
             "value_length": 128, "n_ctx_train": 262144}

    def test_full_attention_is_just_budget_over_cost(self):
        per_tok = gw.kv_bytes_per_token(self.DENSE, "f16", "f16")
        assert gw.ctx_for_kv_budget(self.DENSE, int(per_tok * 5000), "f16",
                                    "f16") == 5000

    def test_a_sliding_window_buys_far_more_than_full_attention(self):
        per_tok = gw.kv_bytes_per_token(self.GEMMA, "f16", "f16")
        budget = int(per_tok * 4096)
        naive = 4096
        assert gw.ctx_for_kv_budget(self.GEMMA, budget, "f16", "f16") > naive * 4

    def test_below_the_window_nothing_has_slid_yet(self):
        per_tok = gw.kv_bytes_per_token(self.GEMMA, "f16", "f16")
        # A budget that only covers 100 tokens must not be credited with
        # savings from a window it never reaches.
        assert gw.ctx_for_kv_budget(self.GEMMA, int(per_tok * 100), "f16",
                                    "f16") == 100

    def test_a_window_never_makes_the_answer_smaller(self):
        per_tok = gw.kv_bytes_per_token(self.GEMMA, "f16", "f16")
        for mult in (50, 512, 4096, 100000):
            budget = int(per_tok * mult)
            assert gw.ctx_for_kv_budget(self.GEMMA, budget, "f16", "f16") >= mult

    def test_no_geometry_and_no_budget_are_zero(self):
        assert gw.ctx_for_kv_budget({}, 1 << 30, "f16", "f16") == 0
        assert gw.ctx_for_kv_budget(self.DENSE, 0, "f16", "f16") == 0
        assert gw.ctx_for_kv_budget(self.DENSE, -1, "f16", "f16") == 0


class TestUpstreamCtxRefresh:
    """The hub polls each peer's /admin/api/served-models on a 6 s timeout,
    and a peer that times out drops out of the routing table altogether -- not
    just out of the context numbers. Building the Ollama map inline would have
    made a box with six models do exactly that."""

    def setup_method(self):
        gw._upstream_ctx_cache.update(t=0.0, ctx={})
        gw._upstream_ctx_task = None

    async def test_the_first_caller_is_not_made_to_wait(self, monkeypatch):
        started = asyncio.Event()

        async def _slow():
            started.set()
            await asyncio.sleep(30)
        monkeypatch.setattr(gw, "_refresh_upstream_ctx", _slow)

        t0 = time.monotonic()
        out = await gw.upstream_model_ctx()
        assert time.monotonic() - t0 < 0.5
        assert out == {}
        await asyncio.wait_for(started.wait(), 1.0)
        gw._upstream_ctx_task.cancel()

    async def test_a_burst_of_callers_spawns_one_refresh(self, monkeypatch):
        calls = []

        async def _count():
            calls.append(1)
            await asyncio.sleep(0.2)
        monkeypatch.setattr(gw, "_refresh_upstream_ctx", _count)

        await asyncio.gather(*(gw.upstream_model_ctx() for _ in range(8)))
        assert len(calls) == 1

    async def test_a_stale_map_is_served_while_the_refresh_runs(self, monkeypatch):
        gw._upstream_ctx_cache.update(t=0.0, ctx={"gemma4:e4b": 22528})

        async def _slow():
            await asyncio.sleep(30)
        monkeypatch.setattr(gw, "_refresh_upstream_ctx", _slow)

        assert await gw.upstream_model_ctx() == {"gemma4:e4b": 22528}
        gw._upstream_ctx_task.cancel()

    async def test_an_empty_result_retries_in_a_minute_not_ten(self, monkeypatch):
        class _NoOllama:
            async def get(self, *a, **k):
                raise RuntimeError("connection refused")
        monkeypatch.setattr(gw, "client", _NoOllama())
        await gw._refresh_upstream_ctx()
        assert gw._upstream_ctx_cache["ctx"] == {}
        age = time.time() - gw._upstream_ctx_cache["t"]
        assert gw.UPSTREAM_CTX_TTL - age <= gw.UPSTREAM_CTX_RETRY + 1

    async def test_routing_survives_a_hung_report(self, monkeypatch):
        async def _hang():
            await asyncio.sleep(30)
        monkeypatch.setattr(gw, "served_model_ctx", _hang)
        monkeypatch.setattr(gw, "CTX_REPORT_BUDGET", 0.05)
        monkeypatch.setattr(gw, "load_peers", lambda: [])
        gw._routes_cache["t"] = 0.0
        t0 = time.monotonic()
        await gw.model_routes(force=True)
        assert time.monotonic() - t0 < 5, "a hung engine must not stall routing"


class TestRetiredPeerCeilings:
    async def test_an_offline_peer_keeps_counting(self, client, fake_fleet):
        # peer1 is in the peer list but serves nothing in `cands` for this
        # model -- exactly the asleep-big-box case the table exists for.
        gw.remember_model_ctx({("peer1", "gemma4-31b-qat"): 131072})
        gw._known_ctx_cache.update(t=0.0)
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        assert rows["gemma4-31b-qat"]["ctx_max"] == 131072

    async def test_a_retired_peer_stops_counting_at_once(self, client, fake_fleet,
                                                         monkeypatch):
        gw.remember_model_ctx({("retired-box", "gemma4-31b-qat"): 131072})
        gw._known_ctx_cache.update(t=0.0)
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        # Never in peers.json, so its remembered ceiling must not advertise a
        # window nothing in the fleet can actually serve.
        assert rows["gemma4-31b-qat"]["ctx_max"] == 32768


class TestEclipsedTwin:
    """A dual-boot machine is one machine. collapse_twins() already keeps it
    to one card; these are the numbers printed on that card."""

    SPECS = {
        "dualboot-win": {"ram_gb": 32, "vram_gb": 8},
        "dualboot-linux": {"ram_gb": 32, "vram_gb": 8, "replaces": "dualboot-win"},
        "peer1": {"ram_gb": 64},
    }

    def _reachable(self, monkeypatch, *hosts):
        monkeypatch.setattr(gw, "load_specs", lambda: self.SPECS)
        gw._routes_cache.update(reachable=set(hosts))
        gw._known_ctx_cache.update(t=0.0)

    def test_the_booted_half_eclipses_the_dormant_one(self, monkeypatch):
        self._reachable(monkeypatch, "dualboot-win")
        assert gw.eclipsed_hosts() == {"dualboot-linux"}

    def test_it_works_in_both_directions(self, monkeypatch):
        self._reachable(monkeypatch, "dualboot-linux")
        assert gw.eclipsed_hosts() == {"dualboot-win"}

    def test_a_machine_that_is_off_eclipses_nobody(self, monkeypatch):
        # Either OS could be the one that boots next, so both halves keep
        # counting -- ctx_max has always meant "awake or not".
        self._reachable(monkeypatch, "peer1")
        assert gw.eclipsed_hosts() == set()

    def test_a_cold_gateway_eclipses_nobody(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: self.SPECS)
        gw._routes_cache.update(reachable=set())
        assert gw.eclipsed_hosts() == set()

    def test_a_fleet_with_no_twins_is_unaffected(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {"peer1": {"ram_gb": 64}})
        gw._routes_cache.update(reachable={"peer1"})
        assert gw.eclipsed_hosts() == set()

    async def test_the_dormant_half_stops_advertising_its_ceiling(
            self, client, fake_fleet, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [
            {"name": "dualboot-win", "url": "http://s:8080", "token": "t"},
            {"name": "dualboot-linux", "url": "http://f:8080", "token": "t"},
        ])
        gw.remember_model_ctx({("dualboot-linux", "gemma4-31b-qat"): 131072})
        self._reachable(monkeypatch, "dualboot-win")
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        # Windows is up. Fedora's 131072 needs a REBOOT, not a wake-up, and
        # advertising it against the shared Box alias said "this online box
        # offers 131072" about a window nothing can currently serve.
        assert rows["gemma4-31b-qat"]["ctx_max"] == 32768

    async def test_it_counts_again_once_the_machine_is_off(
            self, client, fake_fleet, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [
            {"name": "dualboot-win", "url": "http://s:8080", "token": "t"},
            {"name": "dualboot-linux", "url": "http://f:8080", "token": "t"},
        ])
        gw.remember_model_ctx({("dualboot-linux", "gemma4-31b-qat"): 131072})
        self._reachable(monkeypatch, "peer1")
        rows = {m["public_id"]: m for m in (await gw.public_models_payload())}
        assert rows["gemma4-31b-qat"]["ctx_max"] == 131072

    def test_a_reachable_peer_is_recorded_from_the_routing_probe(self):
        # The whole mechanism rests on _peer_served distinguishing "answered
        # with nothing" from "did not answer", so pin that shape down.
        assert gw._peer_served.__doc__
        import inspect
        src = inspect.getsource(gw._peer_served)
        assert '"online": False' in src and 'out["online"] = True' in src


class TestIdleGpuReportsZero:
    """0 bytes in use is a measurement. gpu-laptop-2's 4070 sits at exactly 0 MiB
    -- nothing loaded, and the panel runs off the Radeon iGPU -- and the
    gateway turned that into None, so the dashboard printed "VRAM –" as
    though the card had no telemetry at all."""

    @staticmethod
    def _smi(monkeypatch, used_field: str):
        line = ("NVIDIA GeForce RTX 4070 Laptop GPU, 0, 8188, "
                + used_field + ", 41, 12.5, 210")
        monkeypatch.setattr(gw, "_nvsmi", "/usr/bin/nvidia-smi")
        monkeypatch.setattr(gw.subprocess, "run", lambda *a, **k: type(
            "P", (), {"returncode": 0, "stdout": line, "stderr": ""})())
        return gw.nvidia_stats()

    def test_an_idle_card_reports_zero_not_unknown(self, monkeypatch):
        assert self._smi(monkeypatch, "0")[0]["vram_used"] == 0

    def test_a_loaded_card_still_reports_its_bytes(self, monkeypatch):
        assert self._smi(monkeypatch, "5129")[0]["vram_used"] == 5129 * 1024 ** 2

    def test_an_unreadable_field_is_still_unknown(self, monkeypatch):
        assert self._smi(monkeypatch, "[N/A]")[0]["vram_used"] is None

    def test_the_card_that_is_fitted_is_unaffected(self, monkeypatch):
        assert self._smi(monkeypatch, "0")[0]["vram_total"] == 8188 * 1024 ** 2

    def test_windows_counters_keep_the_same_distinction(self, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Windows")
        gw._wingpu_cache.update(t=0.0, gpus=[])

        def run(payload):
            monkeypatch.setattr(gw.subprocess, "run", lambda *a, **k: type(
                "P", (), {"returncode": 0, "stdout": payload, "stderr": ""})())
            gw._wingpu_cache.update(t=0.0, gpus=[])
            return gw.windows_gpu_stats()

        card = '"card":"RX 9060 XT","vram_total":17179869184,"busy_percent":3'
        assert run("[{" + card + ',"vram_used":0}]')[0]["vram_used"] == 0
        assert run("[{" + card + ',"vram_used":null}]')[0]["vram_used"] is None


class TestDownloadStallDetection:
    """A download can fail without ever raising. Measured on a fleet box:
    a model crawled at 1 MB/s for half an hour while a fresh connection to the
    same file pulled 9 MB/s. Bytes kept arriving, so the read timeout never
    fired and the retry loop -- which only catches exceptions -- never ran."""

    @staticmethod
    def _serve(monkeypatch, chunk: bytes, count: int, secs_per_chunk: float):
        """A response that hands over `count` chunks, advancing a fake clock
        by `secs_per_chunk` for each one -- so the test dictates throughput."""
        clock = [1000.0]

        def gen():
            for _ in range(count):
                clock[0] += secs_per_chunk
                yield chunk

        monkeypatch.setattr(gw.time, "time", lambda: clock[0])
        monkeypatch.setattr(gw, "job_update", lambda *a, **k: None)
        monkeypatch.setattr(gw.httpx, "stream", lambda *a, **k: type(
            "R", (), {
                "status_code": 200,
                "headers": {"content-length": str(len(chunk) * count)},
                "iter_bytes": lambda self, chunk_size=0: gen(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *a: False,
            })())

    def test_a_crawling_connection_is_abandoned(self, monkeypatch, tmp_path):
        # 1 KB/s, far under the 512 KB/s floor, for well over the window.
        self._serve(monkeypatch, b"x" * 1024, 300, 1.0)
        with pytest.raises(gw.DownloadStalled) as exc:
            gw._download_attempt(1, "http://x/y", tmp_path / "m.part",
                                 tmp_path / "m", stall_abort=True)
        assert "KB/s" in str(exc.value)

    def test_the_same_crawl_is_tolerated_once_reconnecting_has_not_helped(
            self, monkeypatch, tmp_path):
        # stall_abort=False is what the later attempts pass: a genuinely slow
        # link must be allowed to finish rather than be retried to death.
        self._serve(monkeypatch, b"x" * 1024, 300, 1.0)
        assert gw._download_attempt(1, "http://x/y", tmp_path / "m.part",
                                    tmp_path / "m", stall_abort=False) is True

    def test_a_healthy_download_is_never_interrupted(self, monkeypatch, tmp_path):
        # 4 MB/s, comfortably over the floor, for longer than the window.
        self._serve(monkeypatch, b"y" * (4 * 1024 * 1024), 120, 1.0)
        assert gw._download_attempt(1, "http://x/y", tmp_path / "m.part",
                                    tmp_path / "m", stall_abort=True) is True

    def test_a_short_slow_burst_does_not_trip_it(self, monkeypatch, tmp_path):
        # Slow, but the transfer ends before the window elapses -- judged over
        # a whole window, never on one unlucky chunk.
        self._serve(monkeypatch, b"x" * 1024, 10, 1.0)
        assert gw._download_attempt(1, "http://x/y", tmp_path / "m.part",
                                    tmp_path / "m", stall_abort=True) is True

    def test_stall_is_retryable_like_any_other_failure(self):
        # The worker's except clause must catch it, or the whole point is lost.
        assert issubclass(gw.DownloadStalled, Exception)


class TestCompletionFloor:
    """Measured on a fleet box: asked one short question,
    nemotron3.5-lightning-30b spent 220 tokens reasoning and returned an empty
    string. A floor below what the fleet's thinking models actually spend is
    not a floor."""

    def test_a_tiny_max_tokens_is_raised_to_the_floor(self):
        p = gw.apply_ctx_limit(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12},
            8192)
        assert p["max_tokens"] == gw.PUBLIC_MIN_COMPLETION
        assert gw.PUBLIC_MIN_COMPLETION >= 437, "must clear a real thinking budget"

    def test_less_room_than_the_floor_is_a_413_not_a_stub_reply(self):
        # Only ~150 tokens of room: squeezing the floor under the cap used to
        # hand a thinking model a budget it spends entirely on reasoning --
        # an empty 200. The cap still wins, by refusing with the numbers.
        big = "x" * 3000            # ~1030 estimated tokens
        with pytest.raises(gw.HTTPException) as exc:
            gw.apply_ctx_limit(
                {"messages": [{"role": "user", "content": big}],
                 "max_tokens": 8},
                1200)
        assert exc.value.status_code == 413
        assert exc.value.detail["error"]["type"] == "context_limit"

    def test_a_generous_request_is_left_alone(self):
        p = gw.apply_ctx_limit(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4000},
            8192)
        assert p["max_tokens"] == 4000


class TestHybridGraphicsVram:
    """A hybrid laptop enumerates a 512 MB amdgpu carve-out before its 8 GB
    RTX 4070, which made the fleet page advertise the 4070 as a 512 MB card."""

    IGPU = {"vram_total": 536870912, "vram_used": 458706944, "busy_percent": 0}
    DGPU = {"vram_total": 8585740288, "vram_used": 5378146304, "busy_percent": 12}

    def test_the_public_card_shows_the_discrete_gpu(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {})
        card = gw._sanitize_public_host(
            {"name": "hybrid-box", "online": True,
             "status": {"host": {"gpu": [self.IGPU, self.DGPU]}}}, {})
        assert card["vram"]["total"] == self.DGPU["vram_total"]

    def test_order_does_not_matter(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {})
        card = gw._sanitize_public_host(
            {"name": "hybrid-box", "online": True,
             "status": {"host": {"gpu": [self.DGPU, self.IGPU]}}}, {})
        assert card["vram"]["total"] == self.DGPU["vram_total"]

    def test_a_box_with_no_gpu_still_renders(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {})
        card = gw._sanitize_public_host(
            {"name": "cpu-box", "online": True, "status": {"host": {"gpu": []}}}, {})
        assert card["vram"] is None

    def test_vram_budget_takes_the_largest_pool_not_the_sum(self, monkeypatch):
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [self.IGPU])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [self.DGPU])
        monkeypatch.setattr(gw, "windows_gpu_stats", lambda: [])
        assert gw.vram_total_bytes() == self.DGPU["vram_total"]

    def test_two_cards_of_one_vendor_are_still_one_pool(self, monkeypatch):
        # gpu-desktop-2's pair of 3090s: llama.cpp really does split across them.
        monkeypatch.setattr(gw, "amdgpu_stats", lambda: [])
        monkeypatch.setattr(gw, "nvidia_stats", lambda: [self.DGPU, self.DGPU])
        monkeypatch.setattr(gw, "windows_gpu_stats", lambda: [])
        assert gw.vram_total_bytes() == self.DGPU["vram_total"] * 2
