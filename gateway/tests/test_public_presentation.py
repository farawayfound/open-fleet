"""How the public catalogue is presented, and how the suggested model is kept
resident.

Two features, one subject: the order a visitor reads the model list in
(families, then each row's sort, with one model suggested ahead of all of
them), and the loop that keeps that suggested model loaded on the boxes big
enough to hold it.

Run with: gateway/.venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import time

import pytest

import app as gw

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolate_routing():
    """The routing cache and the preload bookkeeping are module state; a test
    that leaves either dirty poisons the next one."""
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()
    gw._preload_state.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()
    gw._preload_state.clear()


def rows(*specs) -> list[dict]:
    """Catalogue-shaped rows, as terse as the assertions need."""
    return [{"public_id": p, "name": n, "family": f, "sort": s, "enabled": 1}
            for p, n, f, s in specs]


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

class TestFamily:
    def test_the_column_wins_when_it_is_set(self):
        assert gw.public_family(
            {"family": "Qwen", "name": "Qwopus 3.6 35B-A3B"}) == "Qwen"

    def test_a_blank_column_is_derived_from_the_name(self):
        assert gw.public_family({"family": "", "name": "Muse Glimmer 30B"}) == "Muse"

    def test_short_names_read_as_acronyms(self):
        # "GLM-4.7-Flash" is GLM, not Glm.
        assert gw.derive_family({"name": "GLM-4.7-Flash"}) == "GLM"

    def test_internal_capitals_survive(self):
        assert gw.derive_family({"name": "DeepSeek V4 Flash"}) == "DeepSeek"

    def test_falls_back_to_the_public_id(self):
        assert gw.derive_family({"name": "", "public_id": "gemma4-31b-qat"}) == "Gemma"

    def test_a_row_with_nothing_readable_is_not_a_crash(self):
        assert gw.derive_family({"name": "4", "public_id": "42"}) == "Other"

    def test_the_seed_files_a_fine_tune_under_its_base_model(self):
        """The whole reason family is not vendor: these two are vendored to
        the community and belong in the Qwen list a visitor is scanning."""
        seed = {r["public_id"]: r for r in gw.load_public_models_seed()}
        assert seed["qwopus3.6-35b"]["family"] == "Qwen"
        assert seed["qwen3.8-9b-distill"]["family"] == "Qwen"
        assert "community" in seed["qwen3.8-9b-distill"]["vendor"]


class TestBackfill:
    def test_it_fills_blanks_from_the_seed_and_then_does_nothing(self):
        gw.db_exec("UPDATE public_models SET family=''")
        assert gw.backfill_public_families() > 0
        assert gw.backfill_public_families() == 0
        got = gw.db_query(
            "SELECT family FROM public_models WHERE public_id='qwen3.8-27b'")
        assert got[0]["family"] == "Qwen"

    def test_it_never_overwrites_a_family_somebody_set(self):
        gw.db_exec("UPDATE public_models SET family='Hand-picked' "
                   "WHERE public_id='gemma4-e4b'")
        gw.backfill_public_families()
        got = gw.db_query(
            "SELECT family FROM public_models WHERE public_id='gemma4-e4b'")
        assert got[0]["family"] == "Hand-picked"
        gw.db_exec("UPDATE public_models SET family='Gemma' WHERE public_id='gemma4-e4b'")


class TestCleanFamilyOrder:
    def test_trims_drops_blanks_and_dedupes_case_insensitively(self):
        assert gw.clean_family_order(
            ["  Qwen ", "", "qwen", None, "Gemma"]) == ["Qwen", "Gemma"]

    def test_a_non_list_is_no_order_at_all(self):
        assert gw.clean_family_order("Qwen, Gemma") == []

    def test_it_is_bounded(self):
        assert len(gw.clean_family_order([str(i) for i in range(500)])) == 64


class TestPresentationSettings:
    def test_the_shipped_order_leads_with_qwen(self):
        s = gw.get_public_settings()
        assert s["model_family_order"][0] == "Qwen"
        assert s["featured_model"] == "qwen3.8-27b"
        assert s["preload_featured"] is True

    def test_an_empty_order_is_a_real_choice_and_survives_a_save(self):
        gw.set_public_settings({"model_family_order": []})
        assert gw.get_public_settings()["model_family_order"] == []

    def test_garbage_falls_back_to_the_shipped_order(self):
        gw.set_public_settings({"model_family_order": "Qwen,Gemma"})
        assert gw.get_public_settings()["model_family_order"] == \
            gw.DEFAULT_PUBLIC_SETTINGS["model_family_order"]

    def test_a_featured_model_that_no_longer_exists_suggests_nothing(self):
        gw.set_public_settings({"featured_model": "not-a-model"})
        assert gw.featured_public_id() == ""

    def test_a_disabled_featured_model_suggests_nothing(self):
        gw.db_exec("UPDATE public_models SET enabled=0 WHERE public_id='qwen3.8-27b'")
        gw.public_catalogue(force=True)
        try:
            assert gw.featured_public_id() == ""
        finally:
            gw.db_exec("UPDATE public_models SET enabled=1 "
                       "WHERE public_id='qwen3.8-27b'")
            gw.public_catalogue(force=True)


# ---------------------------------------------------------------------------
# the order itself
# ---------------------------------------------------------------------------

class TestOrder:
    CATALOGUE = rows(
        ("g-big", "Gemma 4 31B", "Gemma", 10),
        ("g-small", "Gemma 4 E4B", "Gemma", 40),
        ("q-big", "Qwen 3.8 27B", "Qwen", 50),
        ("q-small", "Qwen 3.5 4B", "Qwen", 70),
        ("n-one", "Nemotron 3 Super", "Nemotron", 90),
    )

    def order(self, **settings) -> list[str]:
        base = {"model_family_order": [], "featured_model": ""}
        base.update(settings)
        return [r["public_id"] for r in gw.order_public_models(self.CATALOGUE, base)]

    def test_no_family_order_is_the_sort_only_listing_this_shipped_as(self):
        assert self.order() == ["g-big", "g-small", "q-big", "q-small", "n-one"]

    def test_families_lead_in_the_configured_order(self):
        assert self.order(model_family_order=["Nemotron", "Qwen", "Gemma"]) == \
            ["n-one", "q-big", "q-small", "g-big", "g-small"]

    def test_sort_still_decides_inside_a_family(self):
        got = self.order(model_family_order=["Qwen"])
        assert got.index("q-big") < got.index("q-small")

    def test_the_match_is_case_insensitive(self):
        assert self.order(model_family_order=["qwen"])[0] == "q-big"

    def test_unlisted_families_keep_their_catalogue_order_behind_the_listed_ones(self):
        # Only Nemotron is named, so Gemma (sort 10) and Qwen (sort 50) fall in
        # behind it -- Gemma first, because its best-placed row sorts first.
        assert self.order(model_family_order=["Nemotron"]) == \
            ["n-one", "g-big", "g-small", "q-big", "q-small"]

    def test_the_featured_model_leads_whatever_its_family_and_sort(self):
        got = self.order(model_family_order=["Gemma", "Qwen", "Nemotron"],
                         featured_model="q-small")
        assert got[0] == "q-small"
        # ...and the rest is untouched by the promotion.
        assert got[1:] == ["g-big", "g-small", "q-big", "n-one"]

    def test_a_row_with_no_family_is_placed_by_its_derived_one(self):
        mixed = self.CATALOGUE + rows(("glm", "GLM-4.7-Flash", "", 5))
        got = [r["public_id"] for r in gw.order_public_models(
            mixed, {"model_family_order": ["GLM"], "featured_model": ""})]
        assert got[0] == "glm"

    def test_ordering_is_total_so_the_listing_never_wobbles(self):
        # Same family, same sort: the tiebreak has to be deterministic, or the
        # public page reshuffles itself between two refreshes.
        tied = rows(("b", "B model", "Fam", 10), ("a", "A model", "Fam", 10))
        first = gw.order_public_models(tied, {"model_family_order": [],
                                              "featured_model": ""})
        second = gw.order_public_models(list(reversed(tied)),
                                        {"model_family_order": [],
                                         "featured_model": ""})
        assert [r["public_id"] for r in first] == [r["public_id"] for r in second]


# ---------------------------------------------------------------------------
# what the portal and the dashboard actually receive
# ---------------------------------------------------------------------------

class TestPublicModelsRoute:
    async def test_the_suggested_model_leads_and_is_badged(self, client, fake_fleet):
        gw.set_public_settings({"featured_model": "gemma4-e4b"})
        body = client.get("/public/api/models").json()
        assert body["featured"] == "gemma4-e4b"
        assert body["models"][0]["public_id"] == "gemma4-e4b"
        assert body["models"][0]["featured"] is True
        assert sum(1 for m in body["models"] if m["featured"]) == 1

    async def test_every_row_carries_its_family_and_the_order_is_echoed(
            self, client, fake_fleet):
        body = client.get("/public/api/models").json()
        assert body["family_order"][0] == "Qwen"
        by_id = {m["public_id"]: m for m in body["models"]}
        assert by_id["qwen3.8-27b"]["family"] == "Qwen"
        assert by_id["qwopus3.6-35b"]["family"] == "Qwen"

    async def test_the_default_settings_lead_with_qwen_3_8_27b(
            self, client, fake_fleet):
        body = client.get("/public/api/models").json()
        assert body["models"][0]["public_id"] == "qwen3.8-27b"

    async def test_families_are_contiguous_in_the_listing(self, client, fake_fleet):
        """A page that just renders the list in order gets grouping for free,
        which is the point of doing this in the hub rather than the portal."""
        body = client.get("/public/api/models").json()
        seen, runs = set(), []
        for m in body["models"]:
            if not runs or runs[-1] != m["family"]:
                runs.append(m["family"])
        # The featured row is promoted out of its family, so its family may
        # legitimately appear twice; nothing else may.
        featured_family = next(m["family"] for m in body["models"] if m["featured"])
        for fam in runs:
            assert fam not in seen or fam == featured_family, fam
            seen.add(fam)

    async def test_a_disabled_row_is_still_absent(self, client, fake_fleet):
        gw.db_exec("UPDATE public_models SET enabled=0 WHERE public_id='gemma4-e4b'")
        gw.public_catalogue(force=True)
        try:
            body = client.get("/public/api/models").json()
            assert "gemma4-e4b" not in {m["public_id"] for m in body["models"]}
        finally:
            gw.db_exec("UPDATE public_models SET enabled=1 "
                       "WHERE public_id='gemma4-e4b'")
            gw.public_catalogue(force=True)


class TestAdminCatalogue:
    async def test_the_tab_sees_the_order_the_public_page_will_use(
            self, client, admin_headers, fake_fleet):
        body = client.get("/admin/api/public/models", headers=admin_headers).json()
        assert body["items"][0]["public_id"] == "qwen3.8-27b"
        assert body["items"][0]["featured"] is True
        assert body["featured"] == "qwen3.8-27b"
        assert body["family_order"][0] == "Qwen"
        assert "preload" in body

    async def test_family_round_trips_through_a_put(
            self, client, admin_headers, fake_fleet):
        row = client.get("/admin/api/public/models",
                         headers=admin_headers).json()["items"]
        m = next(r for r in row if r["public_id"] == "gemma4-e4b")
        body = {k: m[k] for k in ("name", "vendor", "description", "fleet_ids",
                                  "arch", "params_b", "active_b", "allow_primary",
                                  "allow_worker", "ctx_max", "ctx_default",
                                  "enabled", "sort")}
        body["family"] = "Gemma Nano"
        r = client.put("/admin/api/public/models/gemma4-e4b",
                       json=body, headers=admin_headers)
        assert r.status_code == 200, r.text
        assert r.json()["family"] == "Gemma Nano"
        # ...and a body that never mentions family leaves it alone, so an old
        # script does not silently blank the column.
        body.pop("family")
        r2 = client.put("/admin/api/public/models/gemma4-e4b",
                        json=body, headers=admin_headers)
        assert r2.json()["family"] == "Gemma Nano"
        gw.db_exec("UPDATE public_models SET family='Gemma' WHERE public_id='gemma4-e4b'")
        gw.public_catalogue(force=True)


# ---------------------------------------------------------------------------
# keeping the suggested model resident
# ---------------------------------------------------------------------------

FID = "qwen3.8-27b"


def fleet(monkeypatch, *, running=(), fits=None, engine=None, reachable=None):
    """A fake fleet serving the suggested model on three boxes: one that holds
    it in VRAM, one whose experts would spill, and one big unified-memory box.
    `fits` overrides the per-host fit report. gpu-laptop-1 and mac-laptop-1 rather than
    gpu-desktop-2: the real spec sheet marks gpu-desktop-2 (and gpu-desktop-1) `reserve` --
    somebody's personal machine -- and the loop refuses to warm those at all,
    which is pinned separately below and in test_reserve_boxes.py."""
    fits = fits or {"gpu-laptop-1": "vram", "gpu-desktop-1": "spill", "mac-laptop-1": "unified"}
    hosts = list(fits)
    gw._routes_cache.update(
        t=time.time(),
        cands={FID: hosts},
        running={h: ({FID} if h in running else set()) for h in hosts},
        meta={(h, FID): {"bytes": 17 * gw.GIB, "fit": fits[h], "moe": False,
                         "source": {}} for h in hosts},
        engine=engine or {h: "llama-swap" for h in hosts},
        reachable=set(hosts) if reachable is None else set(reachable),
        warm={},
    )
    monkeypatch.setattr(gw, "load_peers", lambda: [
        {"name": h, "url": "http://" + h + ":8080", "token": "t"} for h in hosts])
    # This process is "test-hub", which the spec sheet does not name as the
    # hub; the loop is hub-only now (see test_preload_dedicated), so say so
    # the way a fleet with an unlisted hub would.
    monkeypatch.setenv("LLMSTACK_PRELOAD", "on")

    async def _routes(force: bool = False):
        return {}

    monkeypatch.setattr(gw, "model_routes", _routes)
    return gw._routes_cache


class TestPreloadCapable:
    def test_a_box_holding_it_in_vram_qualifies(self, monkeypatch):
        # Capability, not policy: gpu-desktop-2 is a reserve box the PLAN skips,
        # but it can hold the model, and this function only answers that.
        fleet(monkeypatch, fits={"gpu-desktop-2": "vram"})
        assert gw.preload_capable("gpu-desktop-2", FID) is True

    def test_unified_memory_qualifies(self, monkeypatch):
        fleet(monkeypatch)
        assert gw.preload_capable("mac-laptop-1", FID) is True

    def test_a_box_that_would_spill_does_not(self, monkeypatch):
        """Pinning a model a box can only run half in RAM buys a slow answer
        AND costs the swap slot -- worse than the fallback on both counts."""
        fleet(monkeypatch)
        assert gw.preload_capable("gpu-desktop-1", FID) is False

    def test_a_spilling_moe_qualifies_where_the_spec_sheet_allows_it(self, monkeypatch):
        fleet(monkeypatch, fits={"gpu-desktop-2": "spill"})
        gw._routes_cache["meta"][("gpu-desktop-2", FID)]["moe"] = True
        assert gw.load_specs()["gpu-desktop-2"]["moe_spill_ok"] is True
        assert gw.preload_capable("gpu-desktop-2", FID) is True

    def test_a_box_the_report_says_nothing_about_does_not_qualify(self, monkeypatch):
        fleet(monkeypatch)
        assert gw.preload_capable("someone-else", FID) is False


class TestPreloadPlan:
    def test_it_plans_the_capable_boxes_and_leaves_the_rest_alone(self, monkeypatch):
        fleet(monkeypatch)
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"gpu-laptop-1", "mac-laptop-1"}
        assert all(p["public_id"] == "qwen3.8-27b" for p in plan)
        # gpu-desktop-1 cannot hold it, so it is not even reported as skipped.
        assert "gpu-desktop-1" not in gw._preload_state

    def test_a_box_already_serving_a_request_is_not_touched(self, monkeypatch):
        fleet(monkeypatch)
        gw._inflight["gpu-laptop-1"] = 1
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["gpu-laptop-1"]["phase"] == "busy"

    def test_a_box_that_just_answered_keeps_what_it_has(self, monkeypatch):
        """The next turn of that conversation outranks a conversation nobody
        has started yet."""
        fleet(monkeypatch)
        gw._host_last_used["gpu-laptop-1"] = time.time()
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["gpu-laptop-1"]["phase"] == "in use"

    def test_a_box_that_answered_long_enough_ago_is_fair_game(self, monkeypatch):
        fleet(monkeypatch)
        gw._host_last_used["gpu-laptop-1"] = time.time() - gw.PRELOAD_IDLE_GRACE - 1
        assert "gpu-laptop-1" in {p["host"] for p in gw.preload_plan()}

    def test_a_cooling_box_is_left_to_recover(self, monkeypatch):
        fleet(monkeypatch)
        gw._mark_host_down("gpu-laptop-1", 60.0, "test")
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["gpu-laptop-1"]["phase"] == "cooling"

    def test_an_unreachable_box_is_not_planned(self, monkeypatch):
        fleet(monkeypatch, reachable=["mac-laptop-1"])
        assert {p["host"] for p in gw.preload_plan()} == {"mac-laptop-1"}

    def test_a_resident_box_inside_its_ttl_is_left_alone(self, monkeypatch):
        fleet(monkeypatch, running=["gpu-laptop-1"])
        gw._preload_state["gpu-laptop-1"] = {"touched": time.time()}
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["gpu-laptop-1"]["phase"] == "resident"

    def test_a_resident_box_is_re_touched_before_its_ttl_runs_out(self, monkeypatch):
        """llama-swap counts the ttl from the last request, so residency is
        held by asking again -- not by having asked once."""
        assert gw.PRELOAD_REFRESH < gw.DEFAULT_MODEL_RECORD["ttl"]
        fleet(monkeypatch, running=["gpu-laptop-1"])
        gw._preload_state["gpu-laptop-1"] = {"touched": time.time() - gw.PRELOAD_REFRESH - 1}
        assert "gpu-laptop-1" in {p["host"] for p in gw.preload_plan()}

    def test_a_reserve_box_is_never_warmed(self, monkeypatch):
        """gpu-desktop-2 can hold the featured model in VRAM outright, and the
        plan still refuses it: it is somebody's personal machine (spec-sheet
        `reserve`), routed to only as a last resort, so keeping the featured
        model warm on it would cost its owner memory for traffic it almost
        never takes."""
        fleet(monkeypatch, fits={"gpu-desktop-2": "vram", "mac-laptop-1": "unified"})
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["gpu-desktop-2"]["phase"] == "reserve"

    def test_turning_it_off_plans_nothing(self, monkeypatch):
        fleet(monkeypatch)
        gw.set_public_settings({"preload_featured": False})
        assert gw.preload_plan() == []
        assert gw._preload_state == {}

    def test_no_suggested_model_means_nothing_to_pin(self, monkeypatch):
        fleet(monkeypatch)
        gw.set_public_settings({"featured_model": ""})
        assert gw.preload_plan() == []


class TestPreloadPass:
    def _capture(self, monkeypatch, status=200):
        seen = []

        async def _post(cand, payload, read_timeout=None):
            seen.append((cand, payload))
            return type("R", (), {"status_code": status, "json": lambda self: {}})()

        monkeypatch.setattr(gw, "_post_chat", _post)
        return seen

    async def test_it_asks_each_capable_box_for_one_token(self, monkeypatch):
        fleet(monkeypatch)
        seen = self._capture(monkeypatch)
        touched = await gw.preload_pass()
        assert {t["host"] for t in touched} == {"gpu-laptop-1", "mac-laptop-1"}
        assert {c for c, _ in seen} == {"gpu-laptop-1", "mac-laptop-1"}
        import json as _json
        body = _json.loads(seen[0][1])
        assert body["model"] == FID and body["max_tokens"] == 1
        assert gw._preload_state["mac-laptop-1"]["phase"] == "resident"

    async def test_it_is_never_metered_against_anybody(self, monkeypatch):
        """These rows would show up in somebody's 'requests today' and count
        against a rate limit that has nothing to do with them."""
        fleet(monkeypatch)
        self._capture(monkeypatch)
        before = gw.db_query("SELECT COUNT(*) c FROM usage")[0]["c"]
        await gw.preload_pass()
        assert gw.db_query("SELECT COUNT(*) c FROM usage")[0]["c"] == before

    async def test_a_box_that_refuses_is_backed_off_not_hammered(self, monkeypatch):
        fleet(monkeypatch)
        self._capture(monkeypatch, status=500)
        await gw.preload_pass()
        assert gw._preload_state["gpu-laptop-1"]["phase"] == "failed"
        assert gw._preload_state["gpu-laptop-1"]["retry_at"] > time.time()
        # The very next pass plans nothing: cooled down and inside its backoff.
        assert gw.preload_plan() == []

    async def test_a_box_with_no_peers_is_not_driven_by_this_loop(self, monkeypatch):
        """Every box runs this app; only the one fronting the fleet pins."""
        fleet(monkeypatch)
        monkeypatch.setattr(gw, "load_peers", lambda: [])
        self._capture(monkeypatch)
        assert await gw.preload_pass() == []
        assert gw.preload_orchestrator() is False

    async def test_the_admin_trigger_runs_it_anyway(self, monkeypatch):
        fleet(monkeypatch)
        monkeypatch.setattr(gw, "load_peers", lambda: [])
        self._capture(monkeypatch)
        assert {t["host"] for t in await gw.preload_pass(force=True)} == \
            {"gpu-laptop-1", "mac-laptop-1"}

    async def test_an_ollama_box_is_asked_in_its_own_language(self, monkeypatch):
        """/v1 cannot express a keep_alive, and keep_alive is the whole point."""
        fleet(monkeypatch, engine={"gpu-laptop-1": "ollama", "gpu-desktop-1": "ollama",
                                   "mac-laptop-1": "ollama"})
        native = []

        async def _peer_native(cand, path, body, timeout=600.0):
            native.append((cand, path, body))
            return type("R", (), {"status_code": 200})()

        monkeypatch.setattr(gw, "_peer_native", _peer_native)
        await gw.preload_pass()
        assert {c for c, _, _ in native} == {"gpu-laptop-1", "mac-laptop-1"}
        assert all(p == "/api/generate" for _, p, _ in native)
        assert all(b["keep_alive"] == "30m" for _, _, b in native)


class TestPreloadReport:
    async def test_the_admin_route_says_what_is_pinned_where(
            self, client, admin_headers, monkeypatch):
        fleet(monkeypatch)
        gw._preload_state["gpu-desktop-2"] = {"host": "gpu-desktop-2", "phase": "resident",
                                          "detail": "kept warm"}
        body = client.get("/admin/api/public/preload", headers=admin_headers).json()
        assert body["enabled"] is True
        assert body["model"] == "qwen3.8-27b"
        assert any(h["phase"] == "resident" for h in body["hosts"])

    async def test_it_needs_an_admin(self, client):
        assert client.get("/admin/api/public/preload").status_code == 401
