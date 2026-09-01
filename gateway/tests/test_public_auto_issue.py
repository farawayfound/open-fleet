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
        assert s["auto_issue_ctx"] == 16384 and s["auto_issue_daily_cap"] == 20
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
