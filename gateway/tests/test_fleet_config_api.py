"""The Configurations tab's one PUT: drag-and-drop routing order, boxes taken
out of use, and per-box reserve flags, all landing in the settings-table
`fleet_routing` blob and re-merged into the spec sheet on the next
`load_specs()`.

Before this endpoint existed, reordering the fleet meant hand-editing
`specs.json` (a file deploy) or flipping the peers.json killswitch one box at
a time -- two mechanisms an operator had to know were related. This is the
one form that drives both, so the tests below are mostly about that seam: the
order becomes `rank` through the THIRD spec-sheet override layer (see
`load_specs()`'s docstring -- DEFAULT_SPECS, then specs.json, then this),
`not_in_use` still flips the same `peers.json` `routed` flag the killswitch
toggle used to, and a save that touches nothing about a peer must not rewrite
it (the same promise TestEditingPeersDoesNotDisturbTheKillswitch makes for
the peer editor).

Run with:
  cd gateway && ./.venv/Scripts/python.exe -m pytest tests/test_fleet_config_api.py -q
"""
from __future__ import annotations

import time

import pytest

import app as gw

PEER1 = {"name": "peer1", "url": "http://127.0.0.1:9"}
PEER2 = {"name": "peer2", "url": "http://127.0.0.1:9"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Fleet-config state lives in three places, none of which may leak
    across tests: peers.json (routing order self-registers there too, via
    the killswitch flag), the settings-table `fleet_routing` row, and the
    process-wide `_specs_cache` keyed on `_fleet_routing_gen`.

    The cache reset on entry matters as much as the one on exit: a prior
    test file may have left `_specs_cache` holding a hit whose `gen` still
    matches the live `_fleet_routing_gen` (we clear the settings row below
    by DELETE, not through `set_fleet_routing()`, so that counter does not
    move) -- an unfixed cache would then serve THAT stale blob instead of
    the clean slate this test expects.
    """
    monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
    gw.save_peers([dict(PEER1), dict(PEER2)])
    gen_before = gw._fleet_routing_gen
    specs_before = dict(gw._specs_cache)
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.FLEET_ROUTING_KEY,))
    gw._specs_cache.update(stamp=None, specs=None, gen=-1)
    yield
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.FLEET_ROUTING_KEY,))
    gw._fleet_routing_gen = gen_before
    gw._specs_cache.clear()
    gw._specs_cache.update(specs_before)


def _put(client, admin_headers, order, not_in_use=None, reserve=None):
    body = {"order": order}
    if not_in_use is not None:
        body["not_in_use"] = not_in_use
    if reserve is not None:
        body["reserve"] = reserve
    return client.put("/admin/api/fleet-config", headers=admin_headers, json=body)


# ---------------------------------------------------------------------------
# GET: the merged state a blank config renders
# ---------------------------------------------------------------------------


class TestGetFleetConfig:
    def test_hosts_cover_exactly_peers_and_self_with_the_full_field_set(
            self, client, admin_headers):
        r = client.get("/admin/api/fleet-config", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert set(body["hosts"]) == {"peer1", "peer2", gw.HOST_NAME}
        fields = {"name", "self", "klass", "reserve", "rank", "routed",
                  "cpu", "gpu", "ram_gb", "vram_gb"}
        for name, h in body["hosts"].items():
            assert set(h) == fields, name
            assert h["name"] == name
        assert body["hosts"][gw.HOST_NAME]["self"] is True
        assert body["hosts"]["peer1"]["self"] is False

    def test_an_untouched_config_routes_everyone_unranked_hosts_alphabetically(
            self, client, admin_headers):
        # Nothing has ever been saved, so no host carries a `rank`; the tie
        # is broken by name among the PEERS -- this box itself is pinned last
        # unconditionally (see order_key), it just happens to also sort last
        # by name here.
        r = client.get("/admin/api/fleet-config", headers=admin_headers)
        body = r.json()
        assert body["order"] == sorted(["peer1", "peer2", gw.HOST_NAME])
        assert body["not_in_use"] == []
        assert body["saved"] == {}

    def test_this_box_sorts_last_whatever_its_name_says(
            self, client, admin_headers, monkeypatch):
        """The tab pins the self row to the bottom of the order list and the
        scorer already treats the hub as the last resort (tier 6) -- so an
        alphabetical accident ('hub' < 'mac-laptop-2') must not render this box
        mid-list, where it would strand every drag-to-bottom above it."""
        monkeypatch.setattr(gw, "HOST_NAME", "aaa-first-by-name")
        body = client.get("/admin/api/fleet-config", headers=admin_headers).json()
        assert body["order"][-1] == "aaa-first-by-name"
        assert body["order"][:-1] == ["peer1", "peer2"]

    def test_needs_the_admin_token(self, client):
        assert client.get("/admin/api/fleet-config").status_code in (401, 403)


# ---------------------------------------------------------------------------
# PUT: input validation
# ---------------------------------------------------------------------------


class TestPutValidation:
    @pytest.mark.parametrize("order", ["peer1", None, {"peer1": 0}, [1, 2]])
    def test_order_must_be_a_list_of_strings(self, client, admin_headers, order):
        r = client.put("/admin/api/fleet-config", headers=admin_headers,
                       json={"order": order})
        assert r.status_code == 400

    @pytest.mark.parametrize("niu", ["peer1", [1, 2]])
    def test_not_in_use_must_be_a_list_of_strings_when_given(
            self, client, admin_headers, niu):
        r = _put(client, admin_headers, order=[], not_in_use=niu)
        assert r.status_code == 400

    @pytest.mark.parametrize("reserve", [["peer1"], "peer1"])
    def test_reserve_must_be_a_dict_when_given(self, client, admin_headers, reserve):
        r = _put(client, admin_headers, order=[], reserve=reserve)
        assert r.status_code == 400

    def test_an_unknown_host_in_order_is_refused(self, client, admin_headers):
        r = _put(client, admin_headers, order=["ghost"])
        assert r.status_code == 400
        assert "unknown host" in r.json()["detail"]
        # Neither the blob nor peers.json moved.
        assert gw.get_fleet_routing() == {}

    def test_an_unknown_host_in_not_in_use_or_reserve_is_also_refused(
            self, client, admin_headers):
        assert _put(client, admin_headers, order=[],
                    not_in_use=["ghost"]).status_code == 400
        assert _put(client, admin_headers, order=[],
                    reserve={"ghost": True}).status_code == 400

    def test_a_malformed_host_name_is_refused_before_the_unknown_check(
            self, client, admin_headers):
        # SAFE_PEER has no room for a space; "bad host" was never going to be
        # `known` either, but the name check has to fire first so the error
        # names the actual problem (a bad name), not a misleading "unknown".
        r = _put(client, admin_headers, order=["bad host"])
        assert r.status_code == 400
        assert "bad host name" in r.json()["detail"]

    def test_the_hub_cannot_take_itself_out_of_its_own_routing(
            self, client, admin_headers):
        r = _put(client, admin_headers, order=[], not_in_use=[gw.HOST_NAME])
        assert r.status_code == 400
        assert "cannot be taken out" in r.json()["detail"]

    def test_a_name_in_both_lists_is_refused(self, client, admin_headers):
        r = _put(client, admin_headers, order=["peer1"], not_in_use=["peer1"])
        assert r.status_code == 400
        assert "peer1" in r.json()["detail"]


# ---------------------------------------------------------------------------
# PUT: the order becomes rank, reserve reaches load_specs(), the blob persists
# ---------------------------------------------------------------------------


class TestPutPersistsRouting:
    def test_the_order_becomes_rank_visible_through_load_specs(
            self, client, admin_headers):
        r = _put(client, admin_headers, order=["peer2", "peer1"])
        assert r.status_code == 200
        specs = gw.load_specs()
        assert specs["peer2"]["rank"] == 0
        assert specs["peer1"]["rank"] == 1
        # ...and the response body already reflects it -- no second GET
        # needed to see a save take effect.
        assert r.json()["order"][:2] == ["peer2", "peer1"]

    def test_a_second_put_reordering_changes_ranks(self, client, admin_headers):
        _put(client, admin_headers, order=["peer1", "peer2"])
        assert gw.load_specs()["peer1"]["rank"] == 0
        assert gw.load_specs()["peer2"]["rank"] == 1

        r = _put(client, admin_headers, order=["peer2", "peer1"])
        assert r.status_code == 200
        assert gw.load_specs()["peer2"]["rank"] == 0
        assert gw.load_specs()["peer1"]["rank"] == 1

    def test_a_host_left_out_of_the_order_keeps_no_rank_and_still_routes(
            self, client, admin_headers):
        # Only peer2 is ranked; peer1 and the hub are untouched by the save
        # and fall to the back, sorted by name -- they are not `not_in_use`.
        r = _put(client, admin_headers, order=["peer2"])
        assert r.status_code == 200
        body = r.json()
        assert gw.load_specs().get("peer1", {}).get("rank") is None
        assert body["order"] == ["peer2", "peer1", gw.HOST_NAME]
        assert body["not_in_use"] == []

    def test_reserve_overrides_reach_load_specs(self, client, admin_headers):
        # peer1 carries no DEFAULT_SPECS / specs.json entry at all, so its
        # baseline reserve is simply absent -- the override is the only
        # source of truth here.
        r = _put(client, admin_headers, order=[], reserve={"peer1": True})
        assert r.status_code == 200
        assert gw.load_specs()["peer1"]["reserve"] is True
        assert r.json()["hosts"]["peer1"]["reserve"] is True

        # The blob is replaced wholesale on every save, not merged: an empty
        # reserve map on the next PUT removes the override entirely rather
        # than leaving peer1 pinned.
        r = _put(client, admin_headers, order=[], reserve={})
        assert r.status_code == 200
        assert not gw.load_specs().get("peer1", {}).get("reserve")
        assert r.json()["hosts"]["peer1"]["reserve"] is False

    def test_a_snapshot_save_does_not_freeze_the_spec_sheets_reserve_flags(
            self, client, admin_headers, monkeypatch):
        """The dashboard sends the FULL reserve map on every save. Only the
        entries that differ from the shipped spec sheet are stored, so a
        later DEFAULT_SPECS or specs.json change still reaches the fleet
        after a routine drag-reorder resaved everything."""
        monkeypatch.setitem(gw.DEFAULT_SPECS, "peer2", {"reserve": True})
        r = _put(client, admin_headers, order=[],
                 reserve={"peer1": False, "peer2": True})
        assert r.status_code == 200
        # Both values match the sheet (peer1 absent = falsy, peer2 True), so
        # neither is an override worth remembering.
        assert gw.get_fleet_routing()["reserve"] == {}
        # A value that DIFFERS is remembered, and wins.
        r = _put(client, admin_headers, order=[], reserve={"peer2": False})
        assert r.status_code == 200
        assert gw.get_fleet_routing()["reserve"] == {"peer2": False}
        assert not gw.load_specs()["peer2"].get("reserve")

    def test_the_saved_blob_round_trips_through_get_fleet_routing(
            self, client, admin_headers):
        r = _put(client, admin_headers, order=["peer1", "peer2"],
                 reserve={"peer2": True})
        assert r.status_code == 200
        assert gw.get_fleet_routing() == {
            "order": ["peer1", "peer2"],
            "reserve": {"peer2": True},
        }
        # And the GET endpoint's `saved` field is exactly that, not a
        # recomputation of it.
        got = client.get("/admin/api/fleet-config", headers=admin_headers).json()
        assert got["saved"] == gw.get_fleet_routing()


# ---------------------------------------------------------------------------
# PUT: not_in_use flips the peers.json killswitch flag
# ---------------------------------------------------------------------------


class TestPutFlipsPeerRouting:
    def test_not_in_use_marks_a_peer_unrouted_and_the_order_excludes_it(
            self, client, admin_headers):
        r = _put(client, admin_headers, order=["peer2"], not_in_use=["peer1"])
        assert r.status_code == 200
        body = r.json()
        assert body["not_in_use"] == ["peer1"]
        assert "peer1" not in body["order"]
        assert body["hosts"]["peer1"]["routed"] is False

        peers = {p["name"]: p for p in gw.load_peers()}
        assert gw.peer_routed(peers["peer1"]) is False
        assert gw.peer_routed(peers["peer2"]) is True

    def test_flipping_it_back_restores_routing(self, client, admin_headers):
        _put(client, admin_headers, order=["peer2"], not_in_use=["peer1"])
        r = _put(client, admin_headers, order=["peer1", "peer2"], not_in_use=[])
        assert r.status_code == 200
        assert r.json()["not_in_use"] == []
        peers = {p["name"]: p for p in gw.load_peers()}
        assert gw.peer_routed(peers["peer1"]) is True

    def test_a_save_that_changes_no_peers_routed_state_never_rewrites_peers_json(
            self, client, admin_headers):
        # Same shape as TestEditingPeersDoesNotDisturbTheKillswitch: a form
        # submit with nothing to flip must not add a `routed` key nobody
        # asked for, or a hand-diff of peers.json would show noise on every
        # unrelated reorder.
        before = gw.PEERS_PATH.read_text()
        r = _put(client, admin_headers, order=["peer1", "peer2"], not_in_use=[])
        assert r.status_code == 200
        assert gw.PEERS_PATH.read_text() == before

    def test_a_put_invalidates_the_routing_and_ctx_caches(
            self, client, admin_headers):
        gw._routes_cache["t"] = time.time()
        gw._known_ctx_cache["t"] = time.time()
        r = _put(client, admin_headers, order=["peer1", "peer2"])
        assert r.status_code == 200
        assert gw._routes_cache["t"] == 0.0
        assert gw._known_ctx_cache["t"] == 0.0

    def test_needs_the_admin_token(self, client):
        r = client.put("/admin/api/fleet-config", json={"order": []})
        assert r.status_code in (401, 403)
