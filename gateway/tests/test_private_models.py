"""A model can be `private`: served by this box's own llama-swap, reachable
only by a bearer key minted on this box against this box's own /v1, and
invisible everywhere else in the fleet.

DEFAULT_MODEL_RECORD["private"] is the flag; private_model_ids() reads it.
Four things are pinned here:

1. render_swap_config() does not care -- a private record loads exactly like
   any other, because llama-swap's job (serve it locally) is unaffected by
   who else gets to hear about it.
2. GET /admin/api/served-models -- the one surface the hub's routing table
   is built from (model_routes -> _peer_served) -- cuts every private id and
   alias out of every field it answers, and reports how many it withheld
   without saying which.
3. PUT /admin/api/models round-trips the flag: the merge-over-
   DEFAULT_MODEL_RECORD in both load_models() and the PUT handler does not
   drop a field neither of them special-cases.
4. A Fleet Pass key -- which already only ever sees its own granted model --
   and the anonymous public overview never name a private id, even one that
   is actually resident right now.

Run with: python -m pytest gateway/tests -q
"""
from __future__ import annotations

import yaml

import app as gw


def rec(mid: str, **kw) -> dict:
    r = dict(gw.DEFAULT_MODEL_RECORD)
    r.update(id=mid, path="/models/" + mid + ".gguf")
    r.update(kw)
    return r


# ---------------------------------------------------------------------------
# render_swap_config() -- a private model still loads
# ---------------------------------------------------------------------------

class TestRenderSwapConfig:
    def test_a_private_model_still_renders(self):
        cfg = yaml.safe_load(gw.render_swap_config(
            [rec("secret-chat", private=True), rec("public-chat")]))
        assert set(cfg["models"]) == {"secret-chat", "public-chat"}

    def test_a_private_persistent_model_still_gets_its_warm_group(self):
        """private only changes who else finds out; the swap groups llama-swap
        itself needs (persistent -> its own unevictable group) are untouched."""
        cfg = yaml.safe_load(gw.render_swap_config(
            [rec("secret-chat", private=True, persistent=True), rec("other")]))
        assert cfg["groups"]["warm"]["members"] == ["secret-chat"]
        assert cfg["hooks"]["on_startup"]["preload"] == ["secret-chat"]


# ---------------------------------------------------------------------------
# private_model_ids()
# ---------------------------------------------------------------------------

class TestPrivateModelIds:
    def test_ids_and_aliases_of_private_records_only(self):
        models = [
            rec("secret-chat", private=True, aliases=["secret-alt", "sc"]),
            rec("public-chat"),
        ]
        assert gw.private_model_ids(models) == {"secret-chat", "secret-alt", "sc"}

    def test_a_disabled_private_record_does_not_count(self):
        models = [rec("secret-chat", private=True, enabled=False)]
        assert gw.private_model_ids(models) == set()

    def test_truthy_like_every_other_bool_field_on_the_record(self):
        """Coerced by truthiness (`rec.get("private")`), the same as
        `preload`/`persistent` already are -- not an `is True` identity
        check -- so a hand-edited "private": 1 behaves like the checkbox."""
        assert gw.private_model_ids([rec("a", private=1)]) == {"a"}
        assert gw.private_model_ids([rec("a", private=0)]) == set()
        assert gw.private_model_ids([rec("a", private=False)]) == set()
        assert gw.private_model_ids([rec("a")]) == set()  # the default

    def test_defaults_to_load_models_when_called_with_no_argument(self, monkeypatch):
        monkeypatch.setattr(gw, "load_models",
                            lambda: [rec("secret-chat", private=True)])
        assert gw.private_model_ids() == {"secret-chat"}


# ---------------------------------------------------------------------------
# GET /admin/api/served-models
# ---------------------------------------------------------------------------

class TestServedModelsEndpoint:
    def test_private_ids_and_aliases_are_cut_from_every_field(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        gw.save_models([
            rec("public-chat", ctx=8192),
            rec("secret-chat", ctx=8192, private=True, persistent=True,
                aliases=["secret-alt"]),
        ])

        async def fake_running():
            return {"secret-chat", "public-chat"}

        monkeypatch.setattr(gw, "upstream_running_ids", fake_running)

        body = client.get("/admin/api/served-models", headers=admin_headers).json()

        assert "public-chat" in body["models"]
        assert "secret-chat" not in body["models"]
        assert "secret-alt" not in body["models"]

        assert body["running"] == ["public-chat"]

        assert body["capacity"].get("public-chat") == 1
        assert "secret-chat" not in body["capacity"]

        assert body["ctx"].get("public-chat") == 8192
        assert "secret-chat" not in body["ctx"]

        assert "public-chat" in body["meta"]
        assert "secret-chat" not in body["meta"]

        # secret-chat is persistent (implies preload), so without the filter
        # it would be the one and only entry here.
        assert "secret-chat" not in body["warm"]
        assert "secret-alt" not in body["warm"]

        assert "secret-chat" not in body["canonical"]
        assert "secret-alt" not in body["canonical"]
        assert "secret-chat" not in body["canonical"].values()
        assert body["canonical"].get("public-chat") == "public-chat"

        # One id + one alias withheld -- reported as a count, never a name.
        assert body["private_count"] == 2

    def test_a_model_with_no_private_flag_reports_zero(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        gw.save_models([rec("public-chat", ctx=8192)])
        body = client.get("/admin/api/served-models", headers=admin_headers).json()
        assert body["private_count"] == 0
        assert "public-chat" in body["models"]

    def test_the_unavailable_branch_also_reports_zero(
            self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(
            gw, "availability",
            lambda: {"gated": True, "available": False,
                     "reason": "owner at the keyboard", "age_s": 1.0})
        body = client.get("/admin/api/served-models", headers=admin_headers).json()
        assert body["unavailable"] == "owner at the keyboard"
        assert body["private_count"] == 0
        assert body["models"] == []


# ---------------------------------------------------------------------------
# PUT /admin/api/models -- the flag round-trips
# ---------------------------------------------------------------------------

class TestPutRoundTrip:
    def test_private_true_round_trips_through_a_put(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        monkeypatch.setattr(gw, "service_control", lambda a, u: (0, "ok"))
        r = client.put(
            "/admin/api/models", headers=admin_headers,
            json={"models": [rec("secret-chat", private=True)], "verify": False},
        )
        assert r.status_code == 200, r.text

        (saved,) = gw.load_models()
        assert saved["id"] == "secret-chat"
        assert saved["private"] is True

        # Still rendered -- the flag never reaches llama-swap's own config.
        cfg = yaml.safe_load(gw.SWAP_CONFIG.read_text())
        assert "secret-chat" in cfg["models"]

    def test_private_false_is_the_quiet_default(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        monkeypatch.setattr(gw, "service_control", lambda a, u: (0, "ok"))
        r = client.put(
            "/admin/api/models", headers=admin_headers,
            json={"models": [rec("public-chat")], "verify": False},
        )
        assert r.status_code == 200, r.text
        (saved,) = gw.load_models()
        assert saved["private"] is False


# ---------------------------------------------------------------------------
# A Fleet Pass key never sees a private model (contract 1.9h's isolation,
# extended): it already reaches only its own granted model(s), sourced from
# public_keys, never from fleet_model_list() -- this pins that a private
# model sitting resident on the answering box changes nothing about that.
# ---------------------------------------------------------------------------

class TestFleetPassNeverSeesPrivate:
    def test_a_resident_private_model_is_absent_from_a_single_keys_listing(
            self, client, intake_headers, captured_mail, fake_fleet, monkeypatch):
        monkeypatch.setattr(
            gw, "load_models", lambda: [rec("secret-chat", private=True)])
        gw._routes_cache["running"].setdefault(gw.HOST_NAME, set()).add("secret-chat")

        client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "priv@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        text = captured_mail[-1]["text"]
        raw = next(line.split("key: ", 1)[1].strip() for line in text.splitlines()
                   if line.startswith("key: "))

        r = client.get("/v1/models", headers={"Authorization": "Bearer " + raw})
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert ids == ["gemma4-31b-qat"]
        assert "secret-chat" not in ids


# ---------------------------------------------------------------------------
# The anonymous public overview never names a private model
# ---------------------------------------------------------------------------

class TestPublicOverviewNeverNamesPrivate:
    def test_a_running_private_model_is_absent_from_the_overview(
            self, client, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [])
        monkeypatch.setattr(
            gw, "load_models",
            lambda: [rec("secret-chat", private=True), rec("public-chat")])

        async def fake_swap_running():
            return True, [
                {"id": "secret-chat", "state": "ready", "n_ctx": 8192},
                {"id": "public-chat", "state": "ready", "n_ctx": 8192},
            ]

        monkeypatch.setattr(gw, "swap_running", fake_swap_running)
        gw._public_overview_cache.update(t=0.0, data=None)

        r = client.get("/public/api/overview")
        assert r.status_code == 200
        assert "secret-chat" not in r.text

        body = r.json()
        hub_host = next(h for h in body["hosts"] if h["name"] == "Hub")
        served = {row["model"] for row in hub_host["serving"]}
        assert "public-chat" in served
        assert "secret-chat" not in served


# ---------------------------------------------------------------------------
# A PEER's private model: its /admin/api/status names its private ids, and
# the hub's public card for that peer drops them - resident or in a snapshot.
# ---------------------------------------------------------------------------

class TestPeerPrivateModelsNeverReachThePublicCard:
    def test_status_reports_this_boxes_private_ids(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(
            gw, "load_models",
            lambda: [rec("secret-chat", private=True, aliases=["sc"]), rec("public-chat")])

        async def fake_swap_running():
            return True, [{"id": "secret-chat", "state": "ready", "n_ctx": 8192}]

        monkeypatch.setattr(gw, "swap_running", fake_swap_running)
        monkeypatch.setattr(gw, "host_status", lambda: {})
        monkeypatch.setattr(gw, "service_states", lambda: {})
        r = client.get("/admin/api/status", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        # The box's own dashboard still sees its full residency...
        assert [m["id"] for m in body["models_running"]] == ["secret-chat"]
        # ...and tells a hub which of those ids are private.
        assert body["models_private"] == ["sc", "secret-chat"]

    def test_a_peers_reported_private_model_is_absent_from_its_card(self, client):
        import json
        h = {"name": "peer-a", "online": True, "last_seen": 0, "status": {
            "host": {},
            "models_running": [{"id": "secret-peer", "n_ctx": 8192},
                               {"id": "public-peer", "n_ctx": 4096}],
            "models_private": ["secret-peer"]}}
        card = gw._sanitize_public_host(h, {})
        assert {row["model"] for row in card["serving"]} == {"public-peer"}
        assert "secret-peer" not in json.dumps(card)

    def test_an_old_peer_without_the_field_is_unaffected(self, client):
        h = {"name": "peer-b", "online": True, "last_seen": 0, "status": {
            "host": {}, "models_running": [{"id": "public-peer", "n_ctx": 4096}]}}
        card = gw._sanitize_public_host(h, {})
        assert {row["model"] for row in card["serving"]} == {"public-peer"}
