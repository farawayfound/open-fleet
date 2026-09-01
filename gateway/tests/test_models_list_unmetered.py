"""GET /v1/models is a listing, not usage.

Every OpenAI-style client lists models on startup, and the downstream-app
autopilot lists them every 30 s as a health probe. Recording each listing as
a usage row made that key's ledger read as 26,562 requests with zero
completions (2026-08-30) and buried the real ones. Budget queries already
excluded these rows (BUDGET_REQ_SQL); now they are simply not written. The
key's last_used_at is still stamped by require_api_key, so "when was this
key last seen" keeps its meaning.
"""
from __future__ import annotations

import app as gw  # conftest has set the env and sys.path by the time this loads


class TestModelsListUnmetered:
    def test_listing_models_writes_no_usage_row(self, client):
        raw, meta = gw.mint_key("probe@box")
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        assert gw.db_query("SELECT COUNT(*) c FROM usage")[0]["c"] == 0

    def test_listing_models_still_marks_the_key_as_used(self, client):
        raw, meta = gw.mint_key("probe2@box")
        assert gw.db_query("SELECT last_used_at FROM api_keys WHERE id=?", (meta["id"],))[0]["last_used_at"] is None
        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        assert gw.db_query("SELECT last_used_at FROM api_keys WHERE id=?", (meta["id"],))[0]["last_used_at"]

    def test_a_completion_is_still_metered(self, client):
        """The change is scoped to the listing: record_usage itself is
        untouched, so a chat row written the way the proxy writes one lands."""
        raw, meta = gw.mint_key("probe3@box")
        gw.record_usage({"id": meta["id"], "name": meta["name"]}, "qwen3.8-9b-distill",
                        "/v1/chat/completions", False, 200,
                        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 50, 100)
        assert gw.db_query("SELECT COUNT(*) c FROM usage")[0]["c"] == 1
