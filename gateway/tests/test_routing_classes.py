"""Routing classes: the FLEET_ROLES ladders made editable on the hub, and the
Routing tab's endpoint family that edits them.

FLEET_ROLES used to be the only word on what `fast` or `deep` meant, and
changing a ladder meant a code deploy. The hub now keeps its edits in one
settings-table row (`fleet_roles`, shaped like the Configurations tab's
`fleet_routing` blob): a re-ordered ladder, an optional node list, or a
whole new class. fleet_roles() merges that row onto the shipped ladders,
and FLEET_ROLES itself stays what was shipped. The tests below pin the
seams that make this safe to hand to an operator:

  * one filter (role_pairs) applies a class's node list, so
    resolve_targets(), /v1/batches, the /v1/models listing and
    /admin/api/roles cannot disagree about where a class may run;
  * a PUT validates every class before it writes any of them, so a refused
    PUT stores nothing;
  * a new class name may not shadow a catalogue row or a model name, whether
    or not a box is serving it right now (asleep, gated, disabled, or only an
    old spelling), because resolve_targets() answers those first;
  * the scorer reads the merged ladders from memory: role_index() runs once
    per candidate inside a sort, so a database read there would put SQLite
    on every request's hot path;
  * an unreadable row is logged, never silently swapped for the defaults;
  * GET /admin/api/roles stays a superset of what the hub scripts read
    (`roles[role]["order"]`, `name in roles`).

Run with (from the repository root):
  timeout 900 uv run --no-sync --with pytest --with pyyaml \
    --with-requirements gateway/requirements.txt \
    --with-requirements gateway/requirements-dev.txt \
    python -m pytest gateway/tests/test_routing_classes.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

import app as gw

URL = "/admin/api/routing-classes"
EDITOR_CHECK = Path(__file__).with_name("routing_editor_check.mjs")
# The settings row the design names. Spelled out rather than read from
# gw.ROLES_KEY so the isolation below works on its own, and pinned by a test.
ROLES_KEY = "fleet_roles"
PEER1 = {"name": "peer1", "url": "http://127.0.0.1:9"}
PEER2 = {"name": "peer2", "url": "http://127.0.0.1:9"}
NOT_ADMIN = [
    pytest.param({}, id="no-credentials"),
    pytest.param({"Authorization": "Bearer not-the-admin-token"}, id="wrong-bearer"),
    pytest.param({"X-Intake-Token": "intake"}, id="intake-token"),
]


@pytest.fixture
def route_reads(monkeypatch):
    """Every routing-table read a test causes, and never a real sweep.

    The PUT invalidates the routing table on purpose, and the real
    model_routes() would answer the next read by polling every peer in
    peers.json. Serving the hand-installed table instead keeps the tests
    offline, and recording how it was asked lets a test prove the tab never
    forced a refresh (a sweep waits on every sleeping box's connect timeout)."""
    reads: list[str] = []

    async def routes(force: bool = False):
        reads.append("force" if force else "read")
        return gw._routes_cache["map"]

    async def refresh():
        reads.append("refresh")
        return gw._routes_cache["map"]

    monkeypatch.setattr(gw, "model_routes", routes)
    monkeypatch.setattr(gw, "_refresh_routes", refresh)
    return reads


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path, route_reads):
    """Routing-class state lives in three places, none of which may leak
    across tests: peers.json (node names are validated against it), the
    settings-table `fleet_roles` row, and the routing table these tests
    install by hand with t=9e9. conftest drops the merged fleet_roles()
    cache before each test; the row is deleted here as well so this file
    does not depend on test order."""
    monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
    gw.save_peers([dict(PEER1), dict(PEER2)])
    saved = {k: (dict(v) if isinstance(v, dict) else set(v) if isinstance(v, set) else v)
             for k, v in gw._routes_cache.items()}
    gw.db_exec("DELETE FROM settings WHERE key=?", (ROLES_KEY,))
    # Remembered ceilings are cached for 30 s and the name-clash check reads
    # them, so one test's sleeping box must not be another test's clash.
    gw._known_ctx_cache.update(t=0.0, written=None)
    yield
    gw.db_exec("DELETE FROM settings WHERE key=?", (ROLES_KEY,))
    gw._known_ctx_cache.update(t=0.0, written=None)
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)


@pytest.fixture
def no_local_models(monkeypatch, tmp_path):
    """fleet_model_list() also lists this box's own models.json and engine;
    both are emptied so the listing is exactly what the routing table says."""
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    gw.save_models([])


def _routes(cands, alias=None, meta=None, running=None, reachable=None):
    """Install a routing table. Unless told otherwise every box reports each
    name as its own canonical id, holds it whole in VRAM, and is online."""
    pairs = [(h, n) for n, hs in cands.items() for h in hs]
    gw._routes_cache.update(
        t=9e9, map={n: hs[0] for n, hs in cands.items()}, cands=cands, cap={},
        alias=alias if alias is not None else {(h, n): n for h, n in pairs},
        meta=meta if meta is not None else {(h, n): {"fit": "vram"} for h, n in pairs},
        running=running or {},
        reachable=reachable if reachable is not None else {h for h, _ in pairs if h})


def _two_boxes_hold_the_4b():
    """peer1 and peer2 both hold qwen3.5-4b (on the `fast` and `small`
    ladders); peer1 also holds qwen3.8-27b."""
    _routes({"qwen3.5-4b": ["peer1", "peer2"], "qwen3.8-27b": ["peer1"]})


def _remember(ceilings):
    """Context ceilings the hub remembers from earlier sweeps -- what it still
    knows about a box that is asleep or holding back. conftest wipes the table
    between tests, and remember_model_ctx skips a write equal to its last one,
    so the cache is told it has written nothing."""
    gw._known_ctx_cache.update(t=0.0, written=None)
    gw.remember_model_ctx(ceilings)


def _put(client, headers, classes):
    return client.put(URL, headers=headers, json={"classes": classes})


def _classes(body):
    return {c["name"]: c for c in body["classes"]}


def _stored_row() -> list[dict]:
    return gw.db_query("SELECT value FROM settings WHERE key=?", (ROLES_KEY,))


# ---------------------------------------------------------------------------
# Nothing saved: the shipped ladders, exactly
# ---------------------------------------------------------------------------


class TestShippedDefaults:
    def test_the_settings_row_is_named_fleet_roles(self):
        assert gw.ROLES_KEY == ROLES_KEY

    def test_with_nothing_saved_fleet_roles_is_exactly_the_shipped_ladders(self):
        assert gw.fleet_roles() == dict(gw.FLEET_ROLES)

    def test_with_nothing_saved_no_class_is_limited_to_any_node(self):
        assert {r: gw.role_nodes(r) for r in gw.FLEET_ROLES} == {
            r: None for r in gw.FLEET_ROLES}
        assert gw.role_nodes("not-a-class") is None

    def test_the_stored_config_is_empty_before_the_first_save(self):
        assert gw.get_role_config() == {}

    def test_a_save_never_rewrites_the_shipped_constant(self, client, admin_headers):
        # Tests and the reset-to-default button both read FLEET_ROLES as "what
        # was shipped"; an override merged into it in place would make a
        # reset restore the override.
        shipped = {k: tuple(v) for k, v in gw.FLEET_ROLES.items()}
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": []}})
        assert r.status_code == 200, r.text
        assert gw.FLEET_ROLES == shipped


# ---------------------------------------------------------------------------
# GET: what the Routing tab renders
# ---------------------------------------------------------------------------


class TestGetRoutingClasses:
    def test_every_shipped_class_is_listed_as_an_unedited_builtin(
            self, client, admin_headers):
        _routes({"qwen3.5-4b": ["peer1"]})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert {"classes", "holders", "models", "nodes"} <= set(body)
        classes = _classes(body)
        assert set(classes) == set(gw.FLEET_ROLES)
        for name, ladder in gw.FLEET_ROLES.items():
            c = classes[name]
            assert (c["builtin"], c["overridden"], c["models"], c["default_models"],
                    c["nodes"]) == (True, False, list(ladder), list(ladder), []), name

    def test_nodes_are_the_configured_peers_plus_this_hub(self, client, admin_headers):
        _routes({"qwen3.5-4b": ["peer1"]})   # peer1 answered; peer2 did not
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        nodes = {n["name"]: n for n in r.json()["nodes"]}
        assert set(nodes) == {"peer1", "peer2", gw.HOST_NAME}
        assert nodes[gw.HOST_NAME]["self"] is True
        assert (nodes["peer1"]["self"], nodes["peer1"]["online"]) == (False, True)
        assert (nodes["peer2"]["self"], nodes["peer2"]["online"]) == (False, False)

    def test_a_holder_known_by_an_old_spelling_is_listed_under_the_canonical_id(
            self, client, admin_headers):
        # mac-laptop-2 before its start-up rename: the weights answer only to the
        # LM Studio id. The tab offers the model by its canonical id, the
        # same reading role_pairs() makes, and still shows the name the box
        # answers to.
        _routes({"empero-ai-qwen3.8-9b-distill": ["peer1"]})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "empero-ai-qwen3.8-9b-distill" not in body["holders"]
        assert "empero-ai-qwen3.8-9b-distill" not in body["models"]
        [row] = body["holders"]["qwen3.8-9b-distill"]
        assert (row["host"], row["name"], row["online"]) == (
            "peer1", "empero-ai-qwen3.8-9b-distill", True)

    def test_a_holder_row_says_where_how_well_and_whether_it_is_loaded(
            self, client, admin_headers):
        # The hub is "" in the routing table but HOST_NAME everywhere a node
        # can be named -- the node list a save sends is built from these
        # rows, and "" is not a node name a PUT would accept.
        _routes({"qwen3.6-35b": [""], "qwen3.5-4b": ["peer1"]},
                meta={("", "qwen3.6-35b"): {"fit": "unified"},
                      ("peer1", "qwen3.5-4b"): {"fit": "spill"}},
                running={gw.HOST_NAME: {"qwen3.6-35b"}})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        holders = r.json()["holders"]
        [hub] = holders["qwen3.6-35b"]
        [peer] = holders["qwen3.5-4b"]
        fields = {"host", "name", "fit", "resident", "online", "reserve"}
        assert fields <= set(hub) and fields <= set(peer)
        assert (hub["host"], hub["fit"], hub["resident"], hub["online"]) == (
            gw.HOST_NAME, "unified", True, True)
        assert (peer["host"], peer["fit"], peer["resident"]) == ("peer1", "spill", False)

    def test_a_sleeping_holder_from_the_remembered_ceilings_is_listed_offline(
            self, client, admin_headers):
        # peer2 is asleep, so it is in no live table -- but it is the box the
        # operator may want to name, and model_ctx remembers what it holds.
        _routes({"qwen3.8-27b": ["peer1"]})
        gw.remember_model_ctx({("peer1", "qwen3.8-27b"): 65536,
                               ("peer2", "qwen3.8-27b"): 131072})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        rows = r.json()["holders"]["qwen3.8-27b"]
        assert sorted((h["host"], h["online"]) for h in rows) == [
            ("peer1", True), ("peer2", False)]

    def test_a_box_that_answers_but_lists_nothing_is_still_a_remembered_holder(
            self, client, admin_headers):
        # An availability-gated box (its owner is at the keyboard) answers the
        # sweep with an empty catalogue: reachable, so not "asleep", yet it
        # holds everything it held before. Left out of the checklist it could
        # not be named for a class, and a saved restriction to it would read
        # "holds none of the members".
        _routes({"qwen3.5-4b": ["peer1"]}, reachable={"peer1", "peer2"})
        _remember({("peer2", "qwen3.8-27b"): 65536})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        assert [(h["host"], h["online"]) for h in r.json()["holders"]["qwen3.8-27b"]] == [
            ("peer2", False)]

    def test_a_box_that_lists_models_but_not_this_one_no_longer_holds_it(
            self, client, admin_headers):
        _routes({"qwen3.5-4b": ["peer1"]}, reachable={"peer1"})
        _remember({("peer1", "qwen3.8-27b"): 65536})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        assert "qwen3.8-27b" not in r.json()["holders"]

    def test_a_sleeping_boxs_alias_is_not_offered_as_a_model(self, client, admin_headers):
        # model_ctx keeps a box's alias beside its id. peer1 is up and says
        # `house-pick` is its word for the 4B, so the same word remembered
        # from sleeping peer2 is no model id: a class member by that name
        # would match no box, under a pill that said "1 node". peer2's real
        # rows -- the id, and the old spelling the table canonicalises --
        # still count.
        _routes({"house-pick": ["peer1"], "qwen3.5-4b": ["peer1"], "qwen3.5:4b": ["peer1"]},
                alias={("peer1", "house-pick"): "qwen3.5-4b",
                       ("peer1", "qwen3.5-4b"): "qwen3.5-4b",
                       ("peer1", "qwen3.5:4b"): "qwen3.5-4b"},
                reachable={"peer1"})
        _remember({("peer2", "house-pick"): 8192, ("peer2", "qwen3.5:4b"): 8192})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "house-pick" not in body["holders"] and "house-pick" not in body["models"]
        assert {h["host"] for h in body["holders"]["qwen3.5-4b"]} == {"peer1", "peer2"}

    def test_models_is_every_ladder_entry_plus_every_holder_sorted_once(
            self, client, admin_headers):
        _routes({"homebrew-7b": ["peer1"]})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        models = r.json()["models"]
        assert models == sorted(set(models))
        ladders = {m for ladder in gw.FLEET_ROLES.values() for m in ladder}
        assert ladders | {"homebrew-7b"} <= set(models)

    def test_reading_the_tab_never_forces_a_routing_sweep(
            self, client, admin_headers, route_reads):
        _routes({"qwen3.5-4b": ["peer1"]})
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200, r.text
        assert "force" not in route_reads and "refresh" not in route_reads


# ---------------------------------------------------------------------------
# PUT: editing a built-in class
# ---------------------------------------------------------------------------


class TestPutOverridesABuiltin:
    def test_a_saved_ladder_and_node_list_show_as_an_edited_builtin(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        # The response is the GET shape, so the tab needs no second read.
        for body in (r.json(), client.get(URL, headers=admin_headers).json()):
            fast = _classes(body)["fast"]
            assert (fast["builtin"], fast["overridden"], fast["models"], fast["nodes"]) == (
                True, True, ["qwen3.5-4b"], ["peer1"])
            assert fast["default_models"] == list(gw.FLEET_ROLES["fast"])

    def test_the_row_is_stored_in_the_documented_shape(self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert gw.get_role_config() == {
            "classes": {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}}}

    def test_the_roles_endpoint_shows_the_saved_ladder_and_picks_only_the_listed_node(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        fast = client.get("/admin/api/roles", headers=admin_headers).json()["roles"]["fast"]
        assert fast["ladder"] == ["qwen3.5-4b"]
        assert fast["picks"] == {"peer1": "qwen3.5-4b"}
        assert [(o["host"], o["model"]) for o in fast["order"]] == [("peer1", "qwen3.5-4b")]
        assert fast["nodes"] == ["peer1"]

    @pytest.mark.asyncio
    async def test_resolve_targets_for_the_class_goes_only_to_the_listed_node(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert await gw.resolve_targets("fast") == [("peer1", "qwen3.5-4b")]

    def test_a_builtin_saved_with_no_models_keeps_its_shipped_ladder_but_not_its_reach(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": [], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert gw.fleet_roles()["fast"] == gw.FLEET_ROLES["fast"]
        assert gw.role_nodes("fast") == frozenset({"peer1"})
        fast = _classes(r.json())["fast"]
        assert (fast["models"], fast["nodes"]) == (list(gw.FLEET_ROLES["fast"]), ["peer1"])

    def test_models_and_nodes_are_deduplicated_keeping_the_first_position(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {
            "models": ["qwen3.5-4b", "qwen3.6-35b", "qwen3.5-4b"],
            "nodes": ["peer2", "peer1", "peer2"]}})
        assert r.status_code == 200, r.text
        stored = gw.get_role_config()["classes"]["fast"]
        assert stored["models"] == ["qwen3.5-4b", "qwen3.6-35b"]
        assert sorted(stored["nodes"]) == ["peer1", "peer2"] and len(stored["nodes"]) == 2

    def test_a_builtin_may_be_edited_while_a_box_still_has_a_leftover_row_of_its_name(
            self, client, admin_headers):
        # The clash rule is for NEW names. Boxes still carry models.json rows
        # called `fast` (/admin/api/roles lists them as local_rows), and the
        # built-in has always answered first.
        _routes({"fast": ["peer1"], "qwen3.5-4b": ["peer1"]},
                alias={("peer1", "fast"): "qwen3.5-4b", ("peer1", "qwen3.5-4b"): "qwen3.5-4b"})
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": []}})
        assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# The node list: one filter in role_pairs, seen by every role path
# ---------------------------------------------------------------------------


class TestNodeFilter:
    def test_an_empty_node_list_leaves_every_holder_a_candidate(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": []}})
        assert r.status_code == 200, r.text
        assert gw.role_nodes("fast") is None
        assert sorted(gw.role_pairs("fast")) == [("peer1", "qwen3.5-4b"),
                                                 ("peer2", "qwen3.5-4b")]

    def test_a_node_list_drops_every_other_holder(self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert gw.role_nodes("fast") == frozenset({"peer1"})
        assert gw.role_pairs("fast") == [("peer1", "qwen3.5-4b")]

    def test_this_hub_is_named_by_its_host_name_and_routed_as_blank(
            self, client, admin_headers):
        _routes({"qwen3.5-4b": ["", "peer1"]})
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"],
                                                  "nodes": [gw.HOST_NAME]}})
        assert r.status_code == 200, r.text
        assert gw.role_pairs("fast") == [("", "qwen3.5-4b")]

    def test_a_node_matches_the_box_whatever_case_the_box_reports(
            self, client, admin_headers):
        gw.save_peers([dict(PEER1), dict(PEER2),
                       {"name": "Box-C", "url": "http://127.0.0.1:9"}])
        _routes({"qwen3.5-4b": ["Box-C", "peer1"]})
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["Box-C"]}})
        assert r.status_code == 200, r.text
        assert gw.get_role_config()["classes"]["fast"]["nodes"] == ["box-c"]
        assert gw.role_pairs("fast") == [("Box-C", "qwen3.5-4b")]

    def test_a_sleeping_node_can_be_saved_and_until_it_wakes_nothing_plays_the_class(
            self, client, admin_headers):
        # Validated against peers.json, not against who answered just now:
        # the big box is exactly the one that is asleep when the operator
        # edits the tab.
        _routes({"qwen3.5-4b": ["peer1"]})
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer2"]}})
        assert r.status_code == 200, r.text
        assert gw.role_pairs("fast") == []

    @pytest.mark.asyncio
    async def test_a_batch_for_the_class_fans_out_over_the_listed_nodes_only(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        targets = await gw._batch_targets(["fast"])
        assert {(t["host"], t["model"]) for t in targets} == {("peer1", "qwen3.5-4b")}

    @pytest.mark.asyncio
    async def test_the_models_listing_drops_a_class_its_nodes_cannot_play(
            self, client, admin_headers, no_local_models):
        _routes({"qwen3.5-4b": ["peer1"]})
        r = _put(client, admin_headers, {"small": {"models": [], "nodes": ["peer2"]}})
        assert r.status_code == 200, r.text
        owner = {m["id"]: m["owned_by"] for m in await gw.fleet_model_list()}
        assert "small" not in owner
        # `fast` has no node list, and peer1 holds a member.
        assert owner["fast"] == "fleet-role"


# ---------------------------------------------------------------------------
# A custom class: a new word, routed like a built-in
# ---------------------------------------------------------------------------


class TestCustomClass:
    def test_a_new_class_is_listed_and_shown_by_the_roles_endpoint(
            self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"]})
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        writers = _classes(r.json())["writers"]
        assert (writers["builtin"], writers["models"], writers["nodes"]) == (
            False, ["qwen3.8-27b"], [])
        assert gw.fleet_roles()["writers"] == ("qwen3.8-27b",)
        roles = client.get("/admin/api/roles", headers=admin_headers).json()["roles"]
        assert set(roles) == set(gw.FLEET_ROLES) | {"writers"}
        assert roles["writers"]["ladder"] == ["qwen3.8-27b"]
        assert roles["writers"]["picks"] == {"peer1": "qwen3.8-27b"}

    @pytest.mark.asyncio
    async def test_resolve_targets_routes_the_class_to_its_holder(
            self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"], "qwen3.5-4b": ["peer2"]})
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        assert await gw.resolve_targets("writers") == [("peer1", "qwen3.8-27b")]

    @pytest.mark.asyncio
    async def test_a_batch_for_the_class_targets_the_model_never_the_class_word(
            self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"]})
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        targets = await gw._batch_targets(["writers"])
        assert {(t["host"], t["model"]) for t in targets} == {("peer1", "qwen3.8-27b")}

    @pytest.mark.asyncio
    async def test_v1_models_lists_the_class_only_while_a_holder_exists(
            self, client, admin_headers, no_local_models):
        _routes({"qwen3.8-27b": ["peer1"]})
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        owner = {m["id"]: m["owned_by"] for m in await gw.fleet_model_list()}
        assert owner["writers"] == "fleet-role"
        _routes({"glm-4.7-flash": ["peer1"]})
        owner = {m["id"]: m["owned_by"] for m in await gw.fleet_model_list()}
        assert "writers" not in owner

    @pytest.mark.asyncio
    async def test_a_boxs_own_row_named_like_the_class_is_not_listed_as_its_model(
            self, client, admin_headers, no_local_models):
        # resolve_targets() answers the class before any box's row of that
        # name, so listing peer1's `writers` would promise what a request
        # for it cannot get -- the same rule the built-ins have always had.
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        _routes({"writers": ["peer1"]})
        owner = {m["id"]: m["owned_by"] for m in await gw.fleet_model_list()}
        assert "writers" not in owner

    def test_a_box_rows_named_like_the_class_are_no_alias_conflict(
            self, client, admin_headers):
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        _routes({"writers": ["peer1", "peer2"]},
                alias={("peer1", "writers"): "qwen3.8-27b", ("peer2", "writers"): "qwen3.5-4b"})
        assert gw.alias_conflicts() == []

    @pytest.mark.parametrize("public_id, fleet_ids", [
        pytest.param("writers", ["qwen3.8-27b"], id="row-named-like-the-class"),
        pytest.param("rt-test-row", ["writers"], id="row-claiming-the-class-as-a-fleet-id"),
    ])
    def test_a_catalogue_row_may_not_take_a_class_name(
            self, client, admin_headers, public_id, fleet_ids):
        # The catalogue is answered before a class, so either row would
        # silently take the class word over.
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        try:
            r = client.put(f"/admin/api/public/models/{public_id}", headers=admin_headers,
                           json={"fleet_ids": fleet_ids, "name": "Writers"})
            assert r.status_code == 400, r.text
            assert gw.db_query("SELECT 1 FROM public_models WHERE public_id=?",
                               (public_id,)) == []
        finally:
            gw.db_exec("DELETE FROM public_models WHERE public_id=?", (public_id,))
            gw._public_catalogue_cache.update(t=0.0)


# ---------------------------------------------------------------------------
# PUT: validation -- every refusal is a 400 that stores nothing
# ---------------------------------------------------------------------------


class TestPutValidation:
    @pytest.mark.parametrize("classes", [
        pytest.param({"fast": {"models": ["qwen3.5-4b"], "nodes": ["ghost"]}},
                     id="unknown-node"),
        pytest.param({"fast": {"models": [], "nodes": ["bad host"]}}, id="malformed-node"),
        pytest.param({"fast": {"models": [], "nodes": "peer1"}}, id="nodes-not-a-list"),
        pytest.param({"bad name": {"models": ["qwen3.5-4b"]}}, id="unsafe-name"),
        pytest.param({"../fast": {"models": ["qwen3.5-4b"]}}, id="path-like-name"),
        pytest.param({"a" * 49: {"models": ["qwen3.5-4b"]}}, id="name-over-48-chars"),
        pytest.param({"writers": {"models": []}}, id="custom-with-empty-models"),
        pytest.param({"writers": {"nodes": []}}, id="custom-with-no-models-key"),
        pytest.param({"fast": {"models": "qwen3.5-4b"}}, id="models-not-a-list"),
        pytest.param({"fast": {"models": [7]}}, id="model-not-a-string"),
        pytest.param({"fast": {"models": ["bad id"]}}, id="unsafe-model-id"),
        # SAFE_ID's `$` lets a trailing newline through, and a name of dots is
        # a path step the DELETE route can never address again.
        pytest.param({"fast\n": {"models": ["qwen3.5-4b"]}}, id="name-with-trailing-newline"),
        pytest.param({".": {"models": ["qwen3.5-4b"]}}, id="name-of-one-dot"),
        pytest.param({"..": {"models": ["qwen3.5-4b"]}}, id="name-of-two-dots"),
        pytest.param({"fast": {"models": ["qwen3.5-4b\n"]}}, id="model-id-with-trailing-newline"),
        pytest.param({"writers": {"models": [f"m-{i}" for i in range(65)]}},
                     id="more-than-64-models"),
        pytest.param({"fast": ["qwen3.5-4b"]}, id="entry-not-an-object"),
        pytest.param({"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
                      "writers": {"models": []}},
                     id="valid-builtin-beside-invalid-custom"),
        pytest.param({"writers": {"models": ["qwen3.8-27b"]},
                      "small": {"models": [], "nodes": ["ghost"]}},
                     id="valid-custom-beside-invalid-builtin"),
    ])
    def test_a_bad_class_is_refused_and_nothing_is_stored(
            self, client, admin_headers, classes):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, classes)
        assert r.status_code == 400, r.text
        assert _stored_row() == []

    @pytest.mark.parametrize("body", [
        pytest.param([], id="list-body"),
        pytest.param({}, id="no-classes-key"),
        pytest.param({"classes": None}, id="classes-null"),
        pytest.param({"classes": "fast"}, id="classes-a-string"),
        pytest.param({"classes": ["fast"]}, id="classes-a-list"),
    ])
    def test_a_badly_shaped_body_is_refused_and_nothing_is_stored(
            self, client, admin_headers, body):
        r = client.put(URL, headers=admin_headers, json=body)
        assert r.status_code == 400, r.text
        assert _stored_row() == []

    @pytest.mark.parametrize("raw", [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"", id="empty-body"),
        pytest.param(b'{"classes": {"fast": ', id="truncated-json"),
        pytest.param(b"\xff\xfe\x00", id="not-utf8"),
    ])
    def test_a_body_that_is_not_json_is_a_400_that_says_so_not_a_500(
            self, client, admin_headers, raw):
        r = client.put(URL, headers={**admin_headers, "Content-Type": "application/json"},
                       content=raw)
        assert r.status_code == 400, r.text
        assert "JSON" in r.json()["detail"]
        assert _stored_row() == []

    @pytest.mark.parametrize("name, why, setup", [
        pytest.param(
            "gemma-sleepy", "a model a box served before it went quiet",
            lambda: _remember({("peer2", "gemma-sleepy"): 8192}),
            id="a-model-only-a-sleeping-box-holds"),
        pytest.param(
            "cold-weights", "a model name or alias a box serves",
            lambda: _routes({"x": ["peer1"]}, alias={("peer1", "x"): "cold-weights"}),
            id="an-id-a-box-serves-only-under-another-name"),
        pytest.param(
            "unplugged-7b", "an id or alias in this box's models.json",
            lambda: gw.save_models([{"id": "unplugged-7b", "enabled": False}]),
            id="a-disabled-models-json-record"),
        pytest.param(
            "house-pick", "an id or alias in this box's models.json",
            lambda: gw.save_models([{"id": "m", "aliases": ["house-pick"]}]),
            id="a-models-json-alias"),
        pytest.param(
            "ornith-1.5-9b", "a fleet model id or an old spelling of one",
            lambda: None, id="a-canonical-id-nothing-holds-now"),
        pytest.param(
            "Qwopus3.6-A3B", "a fleet model id or an old spelling of one",
            lambda: None, id="an-old-spelling-of-a-canonical-id"),
        pytest.param(
            "scratch-model", "a model on a class ladder",
            lambda: gw.set_role_config({"classes": {"writers": {
                "models": ["scratch-model"], "nodes": []}}}),
            id="a-member-of-another-classs-ladder"),
    ])
    def test_a_new_class_may_not_take_the_name_of_a_model_that_is_merely_away(
            self, client, admin_headers, no_local_models, monkeypatch, name, why, setup):
        # resolve_targets() answers a class before a plain model name, so a
        # name that only LOOKS free -- its box is asleep, gated, or between
        # engines -- would hand the model's traffic to the class the moment
        # the box came back.
        #
        # The shipped catalogue claims most canonical ids already, which would
        # answer for the check under test; with it empty, each case can only
        # be refused by its own rule.
        monkeypatch.setattr(gw, "public_catalogue",
                            lambda force=False: {"by_public": {}, "by_fleet": {}})
        _routes({})
        setup()
        before = _stored_row()
        r = _put(client, admin_headers, {name: {"models": ["qwen3.5-4b"]}})
        assert r.status_code == 400, r.text
        assert name in r.text and why in r.text
        assert _stored_row() == before

    def test_an_old_spelling_is_stored_as_the_id_it_stands_for(self, client, admin_headers):
        # peer1 still knows the weights only by the old tag. role_pairs()
        # compares the ladder with canonical ids, so a ladder that kept the
        # tag as typed would be accepted and then match no box.
        _routes({"qwen3.5:4b": ["peer1"]})
        r = _put(client, admin_headers, {"fast": {
            "models": ["qwen3.5:4b", "qwen3.8-9B", "qwen3.5-4b"]}})
        assert r.status_code == 200, r.text
        assert gw.get_role_config()["classes"]["fast"]["models"] == [
            "qwen3.5-4b", "qwen3.8-9b-distill"]
        assert _classes(r.json())["fast"]["models"] == ["qwen3.5-4b", "qwen3.8-9b-distill"]
        assert gw.role_pairs("fast") == [("peer1", "qwen3.5:4b")]

    @pytest.mark.parametrize("name, routes", [
        pytest.param("homebrew-7b", {"cands": {"homebrew-7b": ["peer1"]}},
                     id="a-served-model-id"),
        pytest.param("house-pick", {"cands": {"house-pick": ["peer1"]},
                                    "alias": {("peer1", "house-pick"): "qwen3.5-4b"}},
                     id="a-served-alias"),
        pytest.param("tev1", {"cands": {}}, id="a-catalogue-public-id"),
        pytest.param("qwen3.6-35b", {"cands": {}}, id="a-catalogue-fleet-id"),
    ])
    def test_a_new_class_may_not_shadow_a_name_routing_answers_first(
            self, client, admin_headers, name, routes):
        _routes(routes["cands"], alias=routes.get("alias"))
        r = _put(client, admin_headers, {name: {"models": ["qwen3.5-4b"]}})
        assert r.status_code == 400, r.text
        assert name in r.text   # the refusal names the clash
        assert _stored_row() == []

    @pytest.mark.parametrize("name, models", [
        pytest.param("a" * 48, ["qwen3.5-4b"], id="a-48-char-name"),
        pytest.param("wide", [f"m-{i}" for i in range(64)], id="exactly-64-models"),
    ])
    def test_the_limits_themselves_are_accepted(self, client, admin_headers, name, models):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {name: {"models": models}})
        assert r.status_code == 200, r.text
        assert gw.fleet_roles()[name] == tuple(models)


# ---------------------------------------------------------------------------
# DELETE: a built-in resets, a custom class goes
# ---------------------------------------------------------------------------


class TestDelete:
    def test_deleting_a_builtin_restores_its_shipped_ladder_and_reach(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        r = client.delete(URL + "/fast", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert gw.fleet_roles()["fast"] == gw.FLEET_ROLES["fast"]
        assert gw.role_nodes("fast") is None
        fast = _classes(client.get(URL, headers=admin_headers).json())["fast"]
        assert (fast["builtin"], fast["overridden"], fast["models"], fast["nodes"]) == (
            True, False, list(gw.FLEET_ROLES["fast"]), [])

    def test_deleting_a_custom_class_removes_it_everywhere(self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"]})
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        r = client.delete(URL + "/writers", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert "writers" not in gw.fleet_roles()
        assert "writers" not in _classes(client.get(URL, headers=admin_headers).json())
        assert "writers" not in gw.get_role_config().get("classes", {})
        assert "writers" not in client.get("/admin/api/roles",
                                           headers=admin_headers).json()["roles"]

    def test_deleting_one_class_keeps_the_others_as_saved(self, client, admin_headers):
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
            "writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        assert client.delete(URL + "/writers", headers=admin_headers).status_code == 200
        assert gw.get_role_config()["classes"] == {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}}

    def test_deleting_an_unknown_class_is_404_and_stores_nothing(
            self, client, admin_headers):
        r = client.delete(URL + "/nope", headers=admin_headers)
        assert r.status_code == 404, r.text
        assert _stored_row() == []


def _writers_class(client, admin_headers) -> None:
    _routes({"qwen3.8-27b": ["peer1"]})
    r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
    assert r.status_code == 200, r.text


def _delete_refused(client, admin_headers, *key_names: str) -> None:
    r = client.delete(URL + "/writers", headers=admin_headers)
    assert r.status_code == 409, r.text
    for name in key_names:
        assert name in str(r.json().get("detail")), r.text
    assert "writers" in gw.fleet_roles(), "a refused delete removed the class"
    assert "writers" in gw.get_role_config()["classes"]


class TestDeleteKeepsWhatKeysDependOn:
    """A class word on a key is not a model name: delete the class and the word
    is just a name any box may serve as whatever it likes, with the key's
    exemption or list entry still pointing at it."""

    def test_a_class_a_keys_workers_run_as_is_a_409_naming_the_key(
            self, client, admin_headers):
        _writers_class(client, admin_headers)
        r = client.post("/admin/api/keys", headers=admin_headers, json={
            "name": "crew-key", "max_worker_nodes": 1, "worker_class": "writers"})
        assert r.status_code == 200, r.text
        _delete_refused(client, admin_headers, "crew-key")

    def test_a_class_on_a_live_keys_model_list_is_a_409_naming_the_key(
            self, client, admin_headers):
        _writers_class(client, admin_headers)
        gw.mint_key("listed-key", allowed_models=["qwen3.5-4b", "writers"])
        gw.mint_key("other-key", allowed_models=["qwen3.5-4b"])
        _delete_refused(client, admin_headers, "listed-key")
        r = client.delete(URL + "/writers", headers=admin_headers)
        assert "other-key" not in str(r.json().get("detail")), r.text

    def test_a_key_whose_workers_are_off_does_not_hold_it(self, client, admin_headers):
        _writers_class(client, admin_headers)
        r = client.post("/admin/api/keys", headers=admin_headers, json={
            "name": "workers-off", "max_worker_nodes": 1, "worker_class": "writers"})
        assert client.patch("/admin/api/keys/" + str(r.json()["id"]), headers=admin_headers,
                            json={"max_worker_nodes": None}).status_code == 200
        r = client.delete(URL + "/writers", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert "writers" not in gw.fleet_roles()

    # An archived key is restorable until its retention ends. Restoring it
    # hands back the list and the workers it was archived with, so the word
    # must still mean what it did then -- not be a class that is gone, which
    # a box may answer with any model of that name.
    def test_an_archived_keys_list_holds_the_class_until_the_key_is_gone_for_good(
            self, client, admin_headers):
        _writers_class(client, admin_headers)
        _raw, meta = gw.mint_key("archived-key", allowed_models=["writers"])
        assert client.delete("/admin/api/keys/" + str(meta["id"]),
                             headers=admin_headers).status_code == 200
        _delete_refused(client, admin_headers, "archived-key")
        assert "archived" in client.delete(URL + "/writers",
                                           headers=admin_headers).json()["detail"]
        assert client.delete("/admin/api/keys/" + str(meta["id"]) + "/purge",
                             headers=admin_headers).status_code == 200
        r = client.delete(URL + "/writers", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert "writers" not in gw.fleet_roles()

    def test_an_archived_keys_workers_hold_the_class_too(self, client, admin_headers):
        _writers_class(client, admin_headers)
        r = client.post("/admin/api/keys", headers=admin_headers, json={
            "name": "archived-crew", "max_worker_nodes": 1, "worker_class": "writers"})
        assert client.delete("/admin/api/keys/" + str(r.json()["id"]),
                             headers=admin_headers).status_code == 200
        _delete_refused(client, admin_headers, "archived-crew")

    def test_restoring_a_key_after_a_refused_delete_leaves_its_list_meaning_what_it_did(
            self, client, admin_headers):
        _writers_class(client, admin_headers)
        _raw, meta = gw.mint_key("archived-key", allowed_models=["writers"])
        key_url = "/admin/api/keys/" + str(meta["id"])
        assert client.delete(key_url, headers=admin_headers).status_code == 200
        _delete_refused(client, admin_headers, "archived-key")
        assert client.post(key_url + "/restore", headers=admin_headers).status_code == 200
        key = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        assert gw.key_model_gate(key, "writers") == (True, None, "")

    def test_resetting_a_builtin_is_not_a_delete_and_is_allowed_while_keys_use_it(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        assert _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": []}}).status_code == 200
        gw.mint_key("fast-key", allowed_models=["fast"])
        r = client.post("/admin/api/keys", headers=admin_headers, json={
            "name": "fast-crew", "max_worker_nodes": 1, "worker_class": "fast"})
        assert r.status_code == 200, r.text
        r = client.delete(URL + "/fast", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert gw.fleet_roles()["fast"] == gw.FLEET_ROLES["fast"]


def _workers_key(client, admin_headers, name: str, cap: int, cls: str = "fast") -> int:
    r = client.post("/admin/api/keys", headers=admin_headers, json={
        "name": name, "max_worker_nodes": cap, "worker_class": cls})
    assert r.status_code == 200, r.text
    return int(r.json()["id"])


class TestPutWarnsAboutStrandedWorkers:
    """A node list that leaves this node out strands the workers of a key
    capped at 0 ("this node only"): every task they spawn is refused. The save
    is allowed -- the admin may mean it -- but says which keys it hits."""

    def test_a_node_list_without_this_node_names_the_keys_it_strands(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        _workers_key(client, admin_headers, "stranded-key", 0)
        _workers_key(client, admin_headers, "roaming-key", 1)
        _workers_key(client, admin_headers, "small-key", 0, "small")
        off = _workers_key(client, admin_headers, "off-key", 0)
        assert client.patch("/admin/api/keys/" + str(off), headers=admin_headers,
                            json={"max_worker_nodes": None}).status_code == 200
        r = _put(client, admin_headers, {"fast": {"models": [], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert gw.role_nodes("fast") == frozenset({"peer1"}), "the save was refused"
        warnings = r.json()["warnings"]
        assert len(warnings) == 1 and "stranded-key" in warnings[0], warnings
        assert "fast" in warnings[0], warnings

    def test_a_node_list_that_keeps_this_node_or_no_node_list_warns_of_nothing(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        _workers_key(client, admin_headers, "stranded-key", 0)
        r = _put(client, admin_headers, {"fast": {"models": [], "nodes": [
            "peer1", gw.HOST_NAME.lower()]}})
        assert r.status_code == 200, r.text
        assert r.json()["warnings"] == []
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        assert r.json()["warnings"] == []

    def test_only_the_classes_in_the_put_are_reported(self, client, admin_headers):
        _two_boxes_hold_the_4b()
        _workers_key(client, admin_headers, "fast-key", 0)
        r = _put(client, admin_headers, {"small": {"models": [], "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        assert r.json()["warnings"] == []


class TestNewClassNeedsAWordNoKeyListsYet:
    """A key's model list may hold a word that names nothing yet. A class
    created under that word would hand the key its whole ladder, and nobody
    chose that for the key."""

    def test_a_word_on_a_live_keys_list_cannot_become_a_class(self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"]})
        gw.mint_key("waiting-key", allowed_models=["qwen3.5-4b", "mycorp-writers"])
        r = _put(client, admin_headers, {"mycorp-writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 400, r.text
        assert "model list" in str(r.json().get("detail")), r.text
        assert _stored_row() == []

    def test_a_word_on_an_archived_keys_list_cannot_become_a_class(self, client, admin_headers):
        # Restoring the key would hand it the class's whole ladder.
        _routes({"qwen3.8-27b": ["peer1"]})
        _raw, meta = gw.mint_key("gone-key", allowed_models=["legacyword"])
        key_url = "/admin/api/keys/" + str(meta["id"])
        assert client.delete(key_url, headers=admin_headers).status_code == 200
        r = _put(client, admin_headers, {"legacyword": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 400, r.text
        assert "model list" in str(r.json().get("detail")), r.text
        assert _stored_row() == []
        assert client.post(key_url + "/restore", headers=admin_headers).status_code == 200
        key = gw.db_query("SELECT * FROM api_keys WHERE id=?", (meta["id"],))[0]
        assert gw.key_model_gate(key, "qwen3.8-27b")[0] is False

    def test_a_word_on_a_purged_keys_list_is_a_fine_class_name(self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"]})
        _raw, meta = gw.mint_key("gone-key", allowed_models=["old-word"])
        key_url = "/admin/api/keys/" + str(meta["id"])
        assert client.delete(key_url, headers=admin_headers).status_code == 200
        assert client.delete(key_url + "/purge", headers=admin_headers).status_code == 200
        assert _put(client, admin_headers,
                    {"old-word": {"models": ["qwen3.8-27b"]}}).status_code == 200

    def test_a_class_a_key_already_lists_can_still_be_edited(self, client, admin_headers):
        _routes({"qwen3.8-27b": ["peer1"], "qwen3.5-4b": ["peer1"]})
        assert _put(client, admin_headers,
                    {"writers": {"models": ["qwen3.8-27b"]}}).status_code == 200
        gw.mint_key("writers-key", allowed_models=["writers"])
        r = _put(client, admin_headers, {"writers": {"models": ["qwen3.8-27b", "qwen3.5-4b"]}})
        assert r.status_code == 200, r.text


class TestRoutingEditsLeaveATrace:
    """Editing a class changes what restricted keys may use, so each edit is
    one INFO line: who, which class, and what it routed with before and after."""

    @staticmethod
    def _lines(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records
                if r.name == gw.log.name and r.levelno == logging.INFO
                and "routing class" in r.getMessage()]

    def test_a_put_logs_one_line_per_class_with_the_admin_and_before_and_after(
            self, client, admin_headers, caplog):
        _two_boxes_hold_the_4b()
        caplog.set_level(logging.INFO, logger=gw.log.name)
        r = _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
            "writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        lines = self._lines(caplog)
        assert len(lines) == 2, lines
        fast = next(ln for ln in lines if "'fast'" in ln)
        assert "admin-token" in fast, fast
        assert "Bearer" not in fast
        assert gw.FLEET_ROLES["fast"][0] in fast and "'qwen3.5-4b']" in fast, fast   # before -> after
        assert "'peer1'" in fast, fast
        writers = next(ln for ln in lines if "'writers'" in ln)
        assert "'qwen3.8-27b'" in writers, writers

    def test_a_delete_and_a_reset_log_too(self, client, admin_headers, caplog):
        _two_boxes_hold_the_4b()
        assert _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
            "writers": {"models": ["qwen3.8-27b"]}}).status_code == 200
        caplog.set_level(logging.INFO, logger=gw.log.name)
        assert client.delete(URL + "/writers", headers=admin_headers).status_code == 200
        assert client.delete(URL + "/fast", headers=admin_headers).status_code == 200
        lines = self._lines(caplog)
        assert len(lines) == 2, lines
        assert "'writers'" in lines[0] and "admin-token" in lines[0] and "'qwen3.8-27b'" in lines[0]
        assert "'fast'" in lines[1] and "'peer1'" in lines[1], lines[1]

    def test_a_refused_put_logs_nothing(self, client, admin_headers, caplog):
        _two_boxes_hold_the_4b()
        caplog.set_level(logging.INFO, logger=gw.log.name)
        r = _put(client, admin_headers, {"fast": {"models": [], "nodes": ["ghost"]}})
        assert r.status_code == 400, r.text
        assert self._lines(caplog) == []


# ---------------------------------------------------------------------------
# Auth: admin only, and a refused call changes nothing
# ---------------------------------------------------------------------------


class TestNeedsTheAdminToken:
    @pytest.mark.parametrize("headers", NOT_ADMIN)
    def test_get_is_refused(self, client, headers):
        assert client.get(URL, headers=headers).status_code in (401, 403)

    @pytest.mark.parametrize("headers", NOT_ADMIN)
    def test_put_is_refused_and_nothing_is_stored(self, client, headers):
        _two_boxes_hold_the_4b()
        r = _put(client, headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}})
        assert r.status_code in (401, 403)
        assert _stored_row() == []

    @pytest.mark.parametrize("headers", NOT_ADMIN)
    def test_delete_is_refused_and_the_saved_class_survives(self, client, headers):
        saved = json.dumps({"classes": {"writers": {"models": ["qwen3.8-27b"], "nodes": []}}})
        gw.db_exec("INSERT INTO settings(key, value) VALUES(?, ?)", (ROLES_KEY, saved))
        r = client.delete(URL + "/writers", headers=headers)
        assert r.status_code in (401, 403)
        assert _stored_row() == [{"value": saved}]


# ---------------------------------------------------------------------------
# Writes: one per request, partial, and the caches follow
# ---------------------------------------------------------------------------


class TestWrites:
    def test_one_put_saves_several_classes_in_one_write(
            self, client, admin_headers, monkeypatch):
        writes: list[dict] = []
        real = gw.set_role_config

        def counted(cfg):
            writes.append(cfg)
            real(cfg)
        monkeypatch.setattr(gw, "set_role_config", counted)
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
            "small": {"models": [], "nodes": ["peer2"]},
            "writers": {"models": ["qwen3.8-27b"]}})
        assert r.status_code == 200, r.text
        assert len(writes) == 1
        assert set(gw.get_role_config()["classes"]) == {"fast", "small", "writers"}
        assert gw.fleet_roles()["fast"] == ("qwen3.5-4b",)
        assert gw.role_nodes("small") == frozenset({"peer2"})
        assert gw.fleet_roles()["writers"] == ("qwen3.8-27b",)

    def test_a_put_naming_one_class_leaves_the_others_as_saved(
            self, client, admin_headers):
        _two_boxes_hold_the_4b()
        assert _put(client, admin_headers, {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]}}).status_code == 200
        assert _put(client, admin_headers, {
            "writers": {"models": ["qwen3.8-27b"]}}).status_code == 200
        assert gw.get_role_config()["classes"] == {
            "fast": {"models": ["qwen3.5-4b"], "nodes": ["peer1"]},
            "writers": {"models": ["qwen3.8-27b"], "nodes": []}}

    def test_a_put_invalidates_the_routing_table(self, client, admin_headers):
        _two_boxes_hold_the_4b()
        gw._routes_cache["t"] = time.time()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b"], "nodes": []}})
        assert r.status_code == 200, r.text
        assert gw._routes_cache["t"] == 0.0

    def test_a_save_drops_both_caches_and_is_visible_at_once(self):
        assert gw.fleet_roles()["fast"] == gw.FLEET_ROLES["fast"]   # warm the cache
        gw._routes_cache["t"] = time.time()
        gw._known_ctx_cache["t"] = time.time()
        gw.set_role_config({"classes": {"fast": {"models": ["qwen3.5-4b"], "nodes": []}}})
        assert gw._routes_cache["t"] == 0.0
        assert gw._known_ctx_cache["t"] == 0.0
        assert gw.fleet_roles()["fast"] == ("qwen3.5-4b",)


# ---------------------------------------------------------------------------
# An unreadable row is loud, and routing falls back to what was shipped
# ---------------------------------------------------------------------------


def _corrupt_the_row():
    # Through the setter first, so the merged cache is dropped exactly as a
    # real save drops it; then the row is overwritten behind its back.
    gw.set_role_config({"classes": {}})
    gw.db_exec("UPDATE settings SET value=? WHERE key=?", ("{not json", ROLES_KEY))


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == gw.log.name and r.levelno >= logging.WARNING]


class TestUnreadableRow:
    def test_get_role_config_logs_and_returns_nothing(self, caplog):
        _corrupt_the_row()
        caplog.set_level(logging.WARNING, logger=gw.log.name)
        assert gw.get_role_config() == {}
        assert _warnings(caplog), "an unreadable fleet_roles row was swallowed silently"

    def test_routing_uses_the_shipped_ladders_and_says_so(self, caplog):
        _corrupt_the_row()
        caplog.set_level(logging.WARNING, logger=gw.log.name)
        assert gw.fleet_roles() == dict(gw.FLEET_ROLES)
        assert _warnings(caplog), "fell back to the shipped ladders without a word"


# ---------------------------------------------------------------------------
# The scorer reads the merged ladders from memory
# ---------------------------------------------------------------------------


class TestScorerStaysInMemory:
    def test_role_index_role_pairs_and_role_nodes_read_no_database_after_warm_up(
            self, client, admin_headers, monkeypatch):
        """role_index() runs once per candidate inside the scorer's sort. A
        SQLite read there would cost every routed request a query per box.
        Only this thread's reads are counted: the session client's
        background loops use the database on their own schedule."""
        _two_boxes_hold_the_4b()
        r = _put(client, admin_headers, {"fast": {"models": ["qwen3.5-4b", "qwen3.8-27b"],
                                                  "nodes": ["peer1"]}})
        assert r.status_code == 200, r.text
        gw.fleet_roles()   # the one read a generation is allowed
        me = threading.get_ident()
        reads: list[str] = []
        real_query, real_db = gw.db_query, gw.db

        def query(sql, args=()):
            if threading.get_ident() == me:
                reads.append(sql)
            return real_query(sql, args)

        def connect():
            if threading.get_ident() == me:
                reads.append("db()")
            return real_db()
        monkeypatch.setattr(gw, "db_query", query)
        monkeypatch.setattr(gw, "db", connect)

        assert gw.role_index("fast", "peer1", "qwen3.5-4b") == 0
        assert gw.role_index("fast", "peer1", "qwen3.8-27b") == 1
        assert gw.role_pairs("fast") == [("peer1", "qwen3.5-4b")]
        assert gw.role_nodes("fast") == frozenset({"peer1"})
        assert reads == []


# ---------------------------------------------------------------------------
# /admin/api/roles: additive only -- hub scripts read it
# ---------------------------------------------------------------------------


def test_with_nothing_saved_every_shipped_role_keeps_its_shape_and_gains_nodes(
        client, admin_headers):
    _routes({"qwen3.5-4b": ["peer1"]})
    r = client.get("/admin/api/roles", headers=admin_headers)
    assert r.status_code == 200, r.text
    roles = r.json()["roles"]
    assert set(roles) == set(gw.FLEET_ROLES)
    for name, ladder in gw.FLEET_ROLES.items():
        assert {"ladder", "picks", "order", "local_rows", "nodes"} <= set(roles[name]), name
        assert roles[name]["ladder"] == list(ladder), name
        # No node list means every holder; [] and null both read that way.
        assert not roles[name]["nodes"], name


# ---------------------------------------------------------------------------
# The tab's own JavaScript, run by node
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_class_editor_cannot_overwrite_a_class_and_can_be_driven_from_the_keyboard():
    """routing_editor_check.mjs runs the page's script against a DOM stub: a
    "+ New class" draft named like an existing class (any case) sends no PUT,
    the class rows are real buttons, and every redraw hands the focus to its
    twin. Skipped, not failed, where node is missing."""
    r = subprocess.run(["node", str(EDITOR_CHECK)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    assert "all assertions passed" in r.stdout
