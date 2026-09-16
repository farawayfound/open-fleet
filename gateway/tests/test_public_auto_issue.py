"""Auto-issued Fleet Pass keys for connected services.

POST /admin/api/public/keys/auto mints a single-worker key on the Public
tab's `auto_issue_model` with the tab's lifetime and budgets, records it
under source `auto:<service>`, and -- the one place the gateway does this --
returns the raw key in the response, because the service is going to write
it into a document itself rather than read an inbox.
"""
from __future__ import annotations

import app as gw  # conftest has set the env and sys.path by the time this loads

AUTO = "/admin/api/public/keys/auto"


def _mint(client, admin_headers, **body):
    payload = {"service": "demo-app", "ref": "042-acme", "company": "Acme"}
    payload.update(body)
    return client.post(AUTO, headers=admin_headers, json=payload)


class TestMint:
    def test_returns_raw_key_and_bundle_once(self, client, admin_headers, fake_fleet):
        r = _mint(client, admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "issued"
        assert body["key"].startswith(gw.KEY_PREFIX)
        assert body["key_prefix"] == body["key"][: len(gw.KEY_PREFIX) + 6]
        assert body["service"] == "demo-app" and body["ref"] == "042-acme"
        # The tab's defaults, not anything the caller said.
        s = gw.get_public_settings()
        assert body["model"] == s["auto_issue_model"] == "qwen3.8-27b"
        assert body["model_name"] == "Qwen 3.8 27B"
        assert body["limit_day"] == s["single_rpd"] and body["limit_hour"] == s["single_rph"]
        assert body["base_url"] == "https://api.example.test/v1"
        assert body["expires_date"] and body["expires_at"].startswith(body["expires_date"])
        assert body["warm_url"].startswith("https://api.example.test/public/warm/")
        # The setup text is the tab's template rendered with this key's facts.
        assert body["key"] in body["setup_text"]
        assert "qwen3.8-27b" in body["setup_text"]
        assert body["expires_date"] in body["setup_text"]

        row = gw.db_query("SELECT * FROM public_keys WHERE id=?", (body["id"],))[0]
        assert row["status"] == "issued"
        assert row["source"] == "auto:demo-app"
        assert row["email"] == "" and row["domain"] == "demo-app"
        assert row["company"] == "Acme" and row["note"] == "042-acme"
        assert row["decided_by"] == "auto:demo-app"
        # A real Fleet Pass key underneath: pinned to the model, budgeted, expiring.
        agent = gw.db_query("SELECT * FROM agents WHERE key_id=?", (row["key_id"],))[0]
        assert agent["force_model"] == "qwen3.8-27b"
        key = gw.db_query("SELECT * FROM api_keys WHERE id=?", (row["key_id"],))[0]
        assert key["max_rpd"] == s["single_rpd"] and key["max_rph"] == s["single_rph"]
        assert key["expires_at"] == body["expires_at"]
        assert key["name"] == "fleet-pass:auto:demo-app:042-acme"
        assert gw.hash_key(body["key"]) == key["key_hash"]
        # The listing never shows the key again.
        listed = client.get("/admin/api/public/keys?source=auto", headers=admin_headers).json()
        assert listed["total"] == 1
        assert "key" not in listed["items"][0] and "key_hash" not in listed["items"][0]

    def test_each_call_is_a_distinct_key(self, client, admin_headers, fake_fleet):
        a = _mint(client, admin_headers, ref="042-acme").json()
        b = _mint(client, admin_headers, ref="043-globex", company="Globex").json()
        assert a["key"] != b["key"]
        assert a["id"] != b["id"]
        rows = gw.db_query("SELECT note, company FROM public_keys ORDER BY id")
        assert [(r["note"], r["company"]) for r in rows] == [("042-acme", "Acme"), ("043-globex", "Globex")]

    def test_ctx_is_clamped_to_the_model_ceiling_not_refused(self, client, admin_headers, fake_fleet):
        r = _mint(client, admin_headers, ctx=99999999)
        assert r.status_code == 200, r.text
        assert r.json()["ctx"] <= 131072  # peer1's ceiling for the 27B in the fake fleet
        assert r.json()["ctx"] % 1024 == 0
        r = _mint(client, admin_headers, ctx=100)
        assert r.status_code == 200 and r.json()["ctx"] == 1024

    def test_uses_the_tab_ctx_by_default(self, client, admin_headers, fake_fleet):
        client.put("/admin/api/public/settings", headers=admin_headers, json={"auto_issue_ctx": 8192})
        assert _mint(client, admin_headers).json()["ctx"] == 8192

    def test_setup_text_states_the_context_window(self, client, admin_headers, fake_fleet):
        """The setup block a service pastes into a document carries the
        granted window, so the recipient can configure their client to it."""
        body = _mint(client, admin_headers, ctx=16384).json()
        assert "Context window: " + str(body["ctx"]) + " tokens" in body["setup_text"]

    def test_caller_may_pick_another_enabled_model_but_not_a_disabled_one(
        self, client, admin_headers, fake_fleet
    ):
        r = _mint(client, admin_headers, model="qwen3.8-9b-distill")
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "qwen3.8-9b-distill"
        r = _mint(client, admin_headers, model="no-such-model")
        assert r.status_code == 400

    def test_optional_email_also_gets_the_mail(self, client, admin_headers, fake_fleet, captured_mail):
        r = _mint(client, admin_headers, email="jane@bigco.example")
        assert r.status_code == 200, r.text
        assert len(captured_mail) == 1
        assert captured_mail[0]["to"] == "jane@bigco.example"
        assert r.json()["key"] in captured_mail[0]["text"]
        row = gw.db_query("SELECT email, domain, emailed_at FROM public_keys WHERE id=?", (r.json()["id"],))[0]
        assert row["email"] == "jane@bigco.example" and row["domain"] == "bigco.example"
        assert row["emailed_at"]

    def test_no_mail_without_an_address(self, client, admin_headers, fake_fleet, captured_mail):
        _mint(client, admin_headers)
        assert captured_mail == []

    def test_the_key_works_like_any_fleet_pass_key(self, client, admin_headers, fake_fleet):
        raw = _mint(client, admin_headers).json()["key"]
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200, r.text
        assert [m["id"] for m in r.json()["data"]] == ["qwen3.8-27b"]
        # Restricted to the public surface, like every Fleet Pass key.
        r = client.get("/api/tags", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 403


class TestGuards:
    def test_requires_the_admin_token(self, client, fake_fleet):
        r = client.post(AUTO, json={"service": "demo-app", "ref": "x"})
        assert r.status_code == 401
        r = client.post(AUTO, headers={"Authorization": "Bearer nope"}, json={"service": "demo-app", "ref": "x"})
        assert r.status_code == 401

    def test_switched_off_on_the_tab(self, client, admin_headers, fake_fleet):
        client.put("/admin/api/public/settings", headers=admin_headers, json={"auto_issue_enabled": False})
        r = _mint(client, admin_headers)
        assert r.status_code == 403
        assert gw.db_query("SELECT COUNT(*) c FROM public_keys")[0]["c"] == 0

    def test_service_and_ref_are_required_and_sane(self, client, admin_headers, fake_fleet):
        assert _mint(client, admin_headers, service="").status_code == 400
        assert _mint(client, admin_headers, service="Not A Slug").status_code == 400
        assert _mint(client, admin_headers, ref="").status_code == 400
        assert _mint(client, admin_headers, email="not-an-address").status_code == 400

    def test_daily_cap_across_services(self, client, admin_headers, fake_fleet):
        client.put("/admin/api/public/settings", headers=admin_headers, json={"auto_issue_daily_cap": 2})
        assert _mint(client, admin_headers, ref="1").status_code == 200
        assert _mint(client, admin_headers, service="other-app", ref="2").status_code == 200
        r = _mint(client, admin_headers, ref="3")
        assert r.status_code == 429
        kinds = [e["detail"] for e in gw.db_query("SELECT detail FROM public_events WHERE kind='rate_limited'")]
        assert kinds == ["auto_issue_cap:demo-app"]
        # A hand-issued key is not an auto-issued one and does not count.
        r = client.post(
            "/admin/api/public/keys", headers=admin_headers,
            json={"email": "friend@gmail.com", "kind": "single", "model": "gemma4-31b-qat", "ctx": 8192},
        )
        assert r.status_code == 200
        assert _mint(client, admin_headers, ref="4").status_code == 429

    def test_global_live_cap_still_applies(self, client, admin_headers, fake_fleet):
        client.put("/admin/api/public/settings", headers=admin_headers, json={"max_live_keys": 1})
        assert _mint(client, admin_headers, ref="1").status_code == 200
        assert _mint(client, admin_headers, ref="2").status_code == 429


class TestSettingsAndStatus:
    def test_defaults_and_round_trip(self, client, admin_headers):
        s = client.get("/admin/api/public/settings", headers=admin_headers).json()
        assert s["auto_issue_enabled"] is True
        assert s["auto_issue_model"] == "qwen3.8-27b"
        # As large as the model is served with, clamped at mint -- not a
        # cautious middle that runs out inside one file-reading turn.
        assert s["auto_issue_ctx"] == 262144 and s["auto_issue_daily_cap"] == 20
        assert "{key}" in s["auto_issue_setup_text"] and "{base_url}" in s["auto_issue_setup_text"]
        r = client.put(
            "/admin/api/public/settings", headers=admin_headers,
            json={"auto_issue_model": "gemma4-31b-qat", "auto_issue_daily_cap": 5000,
                  "auto_issue_setup_text": "key {key} on {base_url}"},
        )
        s = r.json()
        assert s["auto_issue_model"] == "gemma4-31b-qat"
        assert s["auto_issue_daily_cap"] == 1000  # bounds-clamped
        assert s["auto_issue_setup_text"] == "key {key} on {base_url}"

    def test_setup_text_with_a_stray_brace_falls_back_to_the_template(
        self, client, admin_headers, fake_fleet
    ):
        client.put("/admin/api/public/settings", headers=admin_headers,
                   json={"auto_issue_setup_text": "use {key} with {unknown}"})
        body = _mint(client, admin_headers).json()
        assert body["setup_text"] == "use {key} with {unknown}"

    def test_setup_text_with_attribute_traversal_falls_back_too(
        self, client, admin_headers, fake_fleet
    ):
        client.put("/admin/api/public/settings", headers=admin_headers,
                   json={"auto_issue_setup_text": "model {model.name} until {expires.year}"})
        r = _mint(client, admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["setup_text"] == "model {model.name} until {expires.year}"

    def test_resend_is_refused_without_an_address(self, client, admin_headers, fake_fleet, captured_mail):
        body = _mint(client, admin_headers).json()
        r = client.post(f"/admin/api/public/keys/{body['id']}/resend", headers=admin_headers)
        assert r.status_code == 400
        # The working key is untouched and nothing was mailed.
        key = gw.db_query("SELECT archived_at, disabled FROM api_keys WHERE key_hash=?", (gw.hash_key(body["key"]),))[0]
        assert key["archived_at"] is None and key["disabled"] == 0
        assert captured_mail == []
        # Revoke and extend still work on it.
        assert client.post(f"/admin/api/public/keys/{body['id']}/extend", headers=admin_headers).status_code == 200
        assert client.post(f"/admin/api/public/keys/{body['id']}/revoke", headers=admin_headers).status_code == 200

    def test_status_reports_counts_and_the_tab(self, client, admin_headers, fake_fleet):
        a = client.get("/admin/api/public/auto", headers=admin_headers).json()
        assert a["enabled"] is True and a["model"] == "qwen3.8-27b" and a["model_name"] == "Qwen 3.8 27B"
        assert a["model_enabled"] is True
        assert a["issued_today"] == 0 and a["live"] == 0 and a["by_service"] == []
        assert a["endpoint"] == "/admin/api/public/keys/auto"
        assert a["base_url"] == "https://api.example.test/v1"
        _mint(client, admin_headers, ref="1")
        _mint(client, admin_headers, ref="2", service="other-app")
        a = client.get("/admin/api/public/auto", headers=admin_headers).json()
        assert a["issued_today"] == 2 and a["live"] == 2
        assert sorted(x["service"] for x in a["by_service"]) == ["demo-app", "other-app"]
        # Revoking one drops it from live but not from today's count.
        rid = gw.db_query("SELECT id FROM public_keys ORDER BY id LIMIT 1")[0]["id"]
        client.post(f"/admin/api/public/keys/{rid}/revoke", headers=admin_headers)
        a = client.get("/admin/api/public/auto", headers=admin_headers).json()
        assert a["issued_today"] == 2 and a["live"] == 1

    def test_list_filters_by_source_and_searches_the_ref(self, client, admin_headers, fake_fleet):
        _mint(client, admin_headers, ref="042-acme")
        client.post(
            "/admin/api/public/keys", headers=admin_headers,
            json={"email": "friend@gmail.com", "kind": "single", "model": "gemma4-31b-qat", "ctx": 8192},
        )
        assert client.get("/admin/api/public/keys", headers=admin_headers).json()["total"] == 2
        assert client.get("/admin/api/public/keys?source=auto", headers=admin_headers).json()["total"] == 1
        assert client.get("/admin/api/public/keys?source=manual", headers=admin_headers).json()["total"] == 1
        assert client.get("/admin/api/public/keys?source=auto:demo-app", headers=admin_headers).json()["total"] == 1
        found = client.get("/admin/api/public/keys?q=042-acme", headers=admin_headers).json()
        assert found["total"] == 1 and found["items"][0]["source"] == "auto:demo-app"


RESIZE = "/admin/api/public/keys/resize-ctx"


class TestResizeCtx:
    """Raising the window on keys that are already out there.

    The tab's `auto_issue_ctx` decides the NEXT key; this decides the ones
    already sitting in somebody's cover letter. Raise-only, clamped per key
    to its own model's ceiling.
    """

    def _ctx_of(self, body):
        row = gw.db_query("SELECT ctx, key_id, kind FROM public_keys WHERE id=?", (body["id"],))[0]
        agent = gw.db_query("SELECT ctx_limit FROM agents WHERE key_id=?", (row["key_id"],))[0]
        return int(row["ctx"]), int(agent["ctx_limit"])

    def test_raises_row_and_enforced_limit_together(self, client, admin_headers, fake_fleet):
        body = _mint(client, admin_headers, ctx=8192).json()
        assert self._ctx_of(body) == (8192, 8192)
        r = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["updated"] == 1 and out["scanned"] == 1 and out["target"] == 65536
        assert out["keys"][0]["from"] == 8192 and out["keys"][0]["to"] == 65536
        # public_keys.ctx is what /v1/models reports; agents.ctx_limit is what
        # apply_ctx_limit() actually holds a request to. Both, or neither.
        assert self._ctx_of(body) == (65536, 65536)

    def test_clamped_per_key_to_its_own_model(self, client, admin_headers, fake_fleet):
        """peer1 serves the 27B at 131072 in the fake fleet, so an ask above
        that is capped rather than promised."""
        body = _mint(client, admin_headers, ctx=8192).json()
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 1048576}).json()
        assert out["keys"][0]["to"] == 131072
        assert self._ctx_of(body) == (131072, 131072)

    def test_never_lowers(self, client, admin_headers, fake_fleet):
        body = _mint(client, admin_headers, ctx=65536).json()
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 8192}).json()
        assert out["updated"] == 0 and out["skipped"] == 1 and out["keys"] == []
        assert self._ctx_of(body) == (65536, 65536)

    def test_dry_run_reports_without_writing(self, client, admin_headers, fake_fleet):
        body = _mint(client, admin_headers, ctx=8192).json()
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536, "dry_run": True}).json()
        assert out["dry_run"] is True and out["updated"] == 1
        assert self._ctx_of(body) == (8192, 8192)

    def test_defaults_to_the_tab_setting(self, client, admin_headers, fake_fleet):
        body = _mint(client, admin_headers, ctx=8192).json()
        client.put("/admin/api/public/settings", headers=admin_headers, json={"auto_issue_ctx": 32768})
        out = client.post(RESIZE, headers=admin_headers, json={}).json()
        assert out["target"] == 32768
        assert self._ctx_of(body) == (32768, 32768)

    def test_skips_revoked_and_expired_keys(self, client, admin_headers, fake_fleet):
        live = _mint(client, admin_headers, ref="live", ctx=8192).json()
        dead = _mint(client, admin_headers, ref="dead", ctx=8192).json()
        client.post(f"/admin/api/public/keys/{dead['id']}/revoke", headers=admin_headers)
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["scanned"] == 1 and [k["id"] for k in out["keys"]] == [live["id"]]
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (dead["id"],))[0]["ctx"] == 8192

    def test_scope_auto_leaves_self_requested_keys_alone(self, client, admin_headers, fake_fleet):
        """Somebody who picked their own window on the public form did choose
        it; only the figure this gateway chose on a service's behalf is ours
        to widen unasked."""
        auto = _mint(client, admin_headers, ctx=8192).json()
        manual = client.post(
            "/admin/api/public/keys", headers=admin_headers,
            json={"email": "friend@gmail.com", "kind": "single", "model": "qwen3.8-27b", "ctx": 8192},
        ).json()
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["scanned"] == 1 and [k["id"] for k in out["keys"]] == [auto["id"]]
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (manual["id"],))[0]["ctx"] == 8192
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536, "scope": "all"}).json()
        assert out["updated"] == 1 and out["keys"][0]["id"] == manual["id"]
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (manual["id"],))[0]["ctx"] == 65536

    def test_raises_the_derived_max_tokens_floor_with_it(self, client, admin_headers, fake_fleet):
        """max_tokens is min(4096, ctx // 2) at mint, so a key minted small
        carries a small completion cap that has to move too."""
        import json as _json
        body = _mint(client, admin_headers, ctx=4096).json()
        row = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]
        before = _json.loads(gw.db_query(
            "SELECT param_overrides FROM agents WHERE key_id=?", (row["key_id"],))[0]["param_overrides"])
        assert before["max_tokens"] == 2048
        client.post(RESIZE, headers=admin_headers, json={"ctx": 65536})
        after = _json.loads(gw.db_query(
            "SELECT param_overrides FROM agents WHERE key_id=?", (row["key_id"],))[0]["param_overrides"])
        assert after["max_tokens"] == 4096

    def test_rejects_an_out_of_bounds_window(self, client, admin_headers, fake_fleet):
        assert client.post(RESIZE, headers=admin_headers, json={"ctx": 12}).status_code == 400
        assert client.post(RESIZE, headers=admin_headers, json={"ctx": 99999999}).status_code == 400
        assert client.post(RESIZE, headers=admin_headers, json={"ctx": "wide"}).status_code == 400

    def test_needs_the_admin_token(self, client, fake_fleet):
        assert client.post(RESIZE, json={"ctx": 65536}).status_code in (401, 403)

    def test_a_key_whose_profile_is_archived_is_left_alone_and_reported(self, client, admin_headers, fake_fleet):
        """DELETE /admin/api/agents/{key_id} is an ordinary admin action. Once
        the profile is archived get_agent() returns nothing and the key
        enforces NO window, so raising its advertised number would be
        advertising a window nothing holds. Left exactly as it was, and said
        so -- not counted as moved, and not silently skipped either."""
        body = _mint(client, admin_headers, ctx=8192).json()
        row = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]
        assert client.delete(f"/admin/api/agents/{row['key_id']}", headers=admin_headers).status_code == 200
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 0 and out["keys"] == []
        assert [u["id"] for u in out["unsynced"]] == [body["id"]]
        assert out["unsynced"][0]["reason"] == "no_live_profile" and out["unsynced"][0]["ctx"] == 8192
        # ...and says which kind of profile to restore: the tab differs.
        assert out["unsynced"][0]["kind"] == "single"
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (body["id"],))[0]["ctx"] == 8192
        # Restoring the profile brings it back onto the sweep's path, and THEN
        # both sides move together.
        assert client.post(f"/admin/api/agents/{row['key_id']}/restore", headers=admin_headers).status_code == 200
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 1 and out["unsynced"] == []
        assert self._ctx_of(body) == (65536, 65536)

    def test_a_profile_archived_between_the_read_and_the_write_is_still_left_alone(
        self, client, admin_headers, fake_fleet, monkeypatch
    ):
        """The TOCTOU the review demonstrated: the profile is read before the
        lock, and an archive can land in the gap. With the enforced side
        already ahead (so only the advertised number would move, and the
        ctx_limit UPDATE's rowcount guard never runs), the liveness check
        inside the lock is the only thing standing between the sweep and a
        raised advertised window on a key that enforces nothing."""
        body = _mint(client, admin_headers, ctx=8192).json()
        key_id = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]["key_id"]
        gw.db_update("UPDATE agents SET ctx_limit=65536 WHERE key_id=?", (key_id,))
        real_db_query = gw.db_query
        fired = []

        def spy(sql, args=()):
            out = real_db_query(sql, args)
            if sql.startswith("SELECT ctx_limit, param_overrides FROM agents") and tuple(args) == (key_id,) and not fired:
                fired.append(True)
                # The ordinary archive route, landing right after the read.
                assert gw.db_update(
                    "UPDATE agents SET archived_at=? WHERE key_id=? AND archived_at IS NULL",
                    (gw.now(), key_id)) == 1
            return out

        monkeypatch.setattr(gw, "db_query", spy)
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 32768}).json()
        assert fired, "the interleaving did not happen -- the probe missed the read"
        assert out["updated"] == 0 and [u["id"] for u in out["unsynced"]] == [body["id"]]
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (body["id"],))[0]["ctx"] == 8192

    def _disable_agent_via_the_tab(self, client, admin_headers, key_id):
        """What the Agents tab's enabled switch does: PUT the profile back
        with enabled=false. The route clears archived_at on every save, so
        disabled-but-not-archived is an ordinary, reachable state."""
        row = gw.db_query("SELECT * FROM agents WHERE key_id=?", (key_id,))[0]
        import json as _json
        r = client.put(f"/admin/api/agents/{key_id}", headers=admin_headers, json={
            "enabled": False, "name": row["name"], "system_prompt": row["system_prompt"] or "",
            "rules": row["rules"] or "", "allowed_models": _json.loads(row["allowed_models"] or "[]"),
            "force_model": row["force_model"] or "", "param_overrides": _json.loads(row["param_overrides"] or "{}"),
        })
        assert r.status_code == 200, r.text
        assert gw.get_agent(key_id) is None, "the request path must not see a disabled profile"

    def test_a_disabled_profile_is_no_live_profile_to_the_sweep_either(self, client, admin_headers, fake_fleet):
        """The request path's "live" is enabled=1 AND archived_at IS NULL. A
        profile the tab has disabled enforces nothing, exactly like an
        archived one, so the sweep must not size the key by it."""
        body = _mint(client, admin_headers, ctx=8192).json()
        key_id = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]["key_id"]
        self._disable_agent_via_the_tab(client, admin_headers, key_id)
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 0 and out["unsynced"][0]["reason"] == "no_live_profile"
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (body["id"],))[0]["ctx"] == 8192
        assert gw.db_query("SELECT ctx_limit FROM agents WHERE key_id=?", (key_id,))[0]["ctx_limit"] == 8192

    def test_an_enabled_team_beside_a_disabled_agent_is_sized_as_the_team(self, client, admin_headers, fake_fleet):
        """Request-path precedence, exactly: get_team() wins when it is
        live, and a disabled agent row beside it is not a second live
        profile -- it is nothing. Sized as a team, not refused as dual."""
        body = _mint(client, admin_headers, ctx=8192).json()
        key_id = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]["key_id"]
        self._disable_agent_via_the_tab(client, admin_headers, key_id)
        gw.db_exec(
            "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,max_workers,"
            "max_rounds,system_prompt,worker_prompt,ctx_limit,updated_at) VALUES (?,1,?,?,?,?,?,?,?,?,?)",
            (key_id, "crew", "qwen3.8-27b", "[]", 1, 2, "", "", 8192, gw.now()),
        )
        assert gw.get_team(key_id) is not None
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 1 and out["unsynced"] == [] and out["keys"][0]["profile"] == "teams"
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (body["id"],))[0]["ctx"] == 65536
        assert gw.db_query("SELECT ctx_limit FROM teams WHERE key_id=?", (key_id,))[0]["ctx_limit"] == 65536
        # The disabled agent row is nobody's business here.
        assert gw.db_query("SELECT ctx_limit FROM agents WHERE key_id=?", (key_id,))[0]["ctx_limit"] == 8192

    def test_a_key_with_two_live_profiles_is_reported_not_guessed(self, client, admin_headers, fake_fleet):
        """public_keys.kind is frozen at mint and nothing keeps it in step
        with the profile tables. A key that somehow carries a live agents
        AND a live teams row is a state the sweep cannot size honestly
        (the request path would take the team), so it is named and left."""
        body = _mint(client, admin_headers, ctx=8192).json()
        key_id = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]["key_id"]
        gw.db_exec(
            "INSERT INTO teams(key_id,enabled,name,primary_model,worker_models,max_workers,"
            "max_rounds,system_prompt,worker_prompt,ctx_limit,updated_at) VALUES (?,1,?,?,?,?,?,?,?,?,?)",
            (key_id, "stray", "qwen3.8-27b", "[]", 1, 2, "", "", 8192, gw.now()),
        )
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 0 and out["unsynced"][0]["reason"] == "dual_profile"
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (body["id"],))[0]["ctx"] == 8192
        assert gw.db_query("SELECT ctx_limit FROM agents WHERE key_id=?", (key_id,))[0]["ctx_limit"] == 8192
        assert gw.db_query("SELECT ctx_limit FROM teams WHERE key_id=?", (key_id,))[0]["ctx_limit"] == 8192

    def test_a_desynced_pair_converges_upward(self, client, admin_headers, fake_fleet):
        """Each side is raise-only on its own. A profile someone hand-lowered
        is raised with the rest; an advertised number that is already ahead
        stays put while the enforced side catches up -- never one side cut
        down to meet the other."""
        low = _mint(client, admin_headers, ref="low-profile", ctx=8192).json()
        krow = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (low["id"],))[0]
        gw.db_update("UPDATE agents SET ctx_limit=4096 WHERE key_id=?", (krow["key_id"],))
        ahead = _mint(client, admin_headers, ref="pub-ahead", ctx=8192).json()
        gw.db_update("UPDATE public_keys SET ctx=65536 WHERE id=?", (ahead["id"],))
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert out["updated"] == 2
        by_id = {k["id"]: k for k in out["keys"]}
        assert by_id[low["id"]]["enforced_from"] == 4096 and by_id[low["id"]]["enforced_to"] == 65536
        assert by_id[ahead["id"]]["from"] == 65536 and by_id[ahead["id"]]["enforced_from"] == 8192
        assert self._ctx_of(low) == (65536, 65536)
        assert self._ctx_of(ahead) == (65536, 65536)

    def test_a_date_only_expiry_counts_as_live_through_that_day(self, client, admin_headers, fake_fleet):
        """The Keys tab stores a date-only expiry (YYYY-MM-DD) that every
        other liveness check in this file reads as the end of that day. The
        sweep uses the same definition, so a key expiring tonight is still
        moved this afternoon, and one that expired yesterday is not."""
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        live = _mint(client, admin_headers, ref="expires-tonight", ctx=8192).json()
        gone = _mint(client, admin_headers, ref="expired-yesterday", ctx=8192).json()
        for body, day in ((live, today), (gone, today - timedelta(days=1))):
            kid = gw.db_query("SELECT key_id FROM public_keys WHERE id=?", (body["id"],))[0]["key_id"]
            gw.db_update("UPDATE api_keys SET expires_at=? WHERE id=?", (day.isoformat(), kid))
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert [k["id"] for k in out["keys"]] == [live["id"]]
        assert gw.db_query("SELECT ctx FROM public_keys WHERE id=?", (gone["id"],))[0]["ctx"] == 8192

    def test_response_shape_is_stable(self, client, admin_headers, fake_fleet):
        out = client.post(RESIZE, headers=admin_headers, json={"ctx": 65536}).json()
        assert set(out) == {"target", "scope", "dry_run", "scanned", "updated", "skipped", "keys", "unsynced"}
