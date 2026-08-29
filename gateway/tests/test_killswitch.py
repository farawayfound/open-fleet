"""The per-machine routing killswitch.

One flag on a peer record, flipped from the home page, that takes a box out
of the hub's routing without unregistering it. The box keeps its card, its
last-known telemetry and its admin proxy; what it loses is every path that
would send it work.

The tests below are mostly about the boundary between those two halves,
because that is the whole design: too wide a filter and killing a box hides
it from the operator who killed it, too narrow and the toggle is decorative.

Run with: cd gateway && ./.venv/Scripts/python.exe -m pytest tests -q
"""
from __future__ import annotations

import json
import time

import pytest

import app as gw

pytestmark = pytest.mark.asyncio


# Port 9 on the loopback is the discard port with nothing bound to it, so the
# handful of tests below that go through a real route (the fleet ping, the
# public overview) get an immediate connection refused. A made-up HOSTNAME
# would instead put a DNS lookup and a full connect timeout inside the test,
# and leave that in flight when the session-scoped client shuts down.
LIVE = {"name": "live", "url": "http://127.0.0.1:9", "token": "t"}
KILLED = {"name": "killed", "url": "http://127.0.0.1:9", "token": "t",
          "routed": False}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """peers.json per test, and a routing cache that never leaks forward."""
    monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
    snapshot = dict(gw._routes_cache)
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._known_ctx_cache.update(t=0.0, map={}, written=None)


# ---------------------------------------------------------------------------
# the flag itself
# ---------------------------------------------------------------------------


class TestPeerRouted:
    """`routed` postdates every peers.json in the fleet, so absence has to
    mean routed -- a hub that reads an older file must behave exactly as it
    did before the flag existed, not quietly stop routing to everything."""

    def test_a_record_without_the_flag_routes(self):
        assert gw.peer_routed({"name": "old", "url": "http://old:8080"}) is True

    def test_true_routes_and_false_does_not(self):
        assert gw.peer_routed({"routed": True}) is True
        assert gw.peer_routed({"routed": False}) is False

    @pytest.mark.parametrize("v", ["false", "False", " off ", "0", "no"])
    def test_a_stringly_typed_false_is_still_off(self, v):
        # peers.json is hand-editable, and someone reaching for the file
        # rather than the toggle will write a string. Reading it as truthy
        # would route to a box its owner believes is switched out.
        assert gw.peer_routed({"routed": v}) is False

    @pytest.mark.parametrize("v", ["true", "on", "yes", 1])
    def test_anything_else_routes(self, v):
        assert gw.peer_routed({"routed": v}) is True

    def test_routeable_peers_is_load_peers_minus_the_killed(self, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        assert [p["name"] for p in gw.routeable_peers()] == ["live"]


# ---------------------------------------------------------------------------
# what the flag takes away
# ---------------------------------------------------------------------------


class TestKilledPeerLeavesTheRoutingTable:

    @staticmethod
    def _quiet_local(monkeypatch):
        """Silence everything model_routes() asks about THIS box, so the test
        is only about peers."""
        async def _empty_set():
            return set()

        async def _empty_dict():
            return {}

        monkeypatch.setattr(gw, "served_model_ids", _empty_set)
        monkeypatch.setattr(gw, "upstream_running_ids", _empty_set)
        monkeypatch.setattr(gw, "served_model_ctx", _empty_dict)
        monkeypatch.setattr(gw, "served_model_meta", _empty_dict)
        monkeypatch.setattr(gw, "local_capacity", dict)
        monkeypatch.setattr(gw, "engine_info", lambda: {"kind": "llama-swap"})
        monkeypatch.setattr(gw, "remember_model_ctx", lambda ctx: None)

    async def test_it_is_never_even_probed(self, monkeypatch):
        # Not just filtered out of the result: a killed box must not be asked.
        # It may be off, and a hub that still opened a connection to every
        # killed box would pay their connect timeouts on every refresh.
        probed: list[str] = []

        async def _served(p):
            probed.append(p["name"])
            return {"models": {"m-" + p["name"]}, "running": set(),
                    "capacity": {}, "ctx": {("m-" + p["name"]): 4096},
                    "meta": {}, "engine": "llama-swap", "online": True}

        self._quiet_local(monkeypatch)
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        monkeypatch.setattr(gw, "_peer_served", _served)

        routes = await gw.model_routes(force=True)

        assert probed == ["live"]
        assert routes == {"m-live": "live"}
        assert gw._routes_cache["cands"] == {"m-live": ["live"]}
        # And it is not reachable, which is what the preload loop and the
        # eclipse rule both read.
        assert gw._routes_cache["reachable"] == {gw.HOST_NAME, "live"}

    async def test_flipping_it_back_returns_the_box(self, monkeypatch):
        self._quiet_local(monkeypatch)
        peers = [dict(LIVE), dict(KILLED)]
        monkeypatch.setattr(gw, "load_peers", lambda: peers)

        async def _served(p):
            return {"models": {"m-" + p["name"]}, "running": set(),
                    "capacity": {}, "ctx": {}, "meta": {}, "engine": "",
                    "online": True}

        monkeypatch.setattr(gw, "_peer_served", _served)

        assert "m-killed" not in await gw.model_routes(force=True)
        peers[1]["routed"] = True
        assert "m-killed" in await gw.model_routes(force=True)

    async def test_its_remembered_context_ceiling_stops_counting(self, monkeypatch):
        # known_model_ctx() is what the public catalogue advertises a window
        # from. A killed box's ceiling is a window this fleet will not serve,
        # so leaving it in would publish a number and then reduce every
        # request down from it -- the same failure a REMOVED peer causes.
        gw.remember_model_ctx({("live", "shared-model"): 32768,
                               ("killed", "shared-model"): 262144})
        gw._known_ctx_cache.update(t=0.0, map={}, written=None)

        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, dict(KILLED, routed=True)])
        assert gw.known_model_ctx()["shared-model"]["killed"] == 262144

        gw._known_ctx_cache.update(t=0.0, map={}, written=None)
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        by_host = gw.known_model_ctx()["shared-model"]
        assert "killed" not in by_host
        assert by_host["live"] == 32768


# ---------------------------------------------------------------------------
# what the flag deliberately leaves alone
# ---------------------------------------------------------------------------


class TestAKilledPeerIsStillPartOfTheFleet:
    """A killswitch is not a delete. Everything an operator needs in order to
    see the box, diagnose it and turn it back on has to keep working -- most
    of all on a box that is BOTH killed and offline, which is exactly when
    someone goes looking for it."""

    def test_it_still_appears_on_the_fleet_page(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        rows = {h["name"]: h for h in client.get(
            "/admin/api/fleet", headers=admin_headers).json()["hosts"]}
        assert set(rows) >= {"live", "killed"}
        assert rows["killed"]["routed"] is False
        assert rows["live"]["routed"] is True

    def test_the_peers_list_reports_its_state(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        rows = {p["name"]: p for p in client.get(
            "/admin/api/peers", headers=admin_headers).json()}
        assert rows["killed"]["routed"] is False

    def test_the_admin_proxy_still_reaches_it(self, monkeypatch):
        # _peer_admin() is how the dashboard drives a peer's own settings.
        # It looks the peer up in load_peers(), not routeable_peers(): you
        # have to be able to administer a box you have taken out of routing,
        # not least to work out why you took it out.
        import inspect
        src = inspect.getsource(gw._peer_admin)
        assert "load_peers()" in src and "routeable_peers" not in src

    async def test_the_public_page_never_learns_about_it(self, monkeypatch):
        # The killswitch is an internal operational fact. The public overview
        # builds its host rows from an allow-list, so this is really a guard
        # against someone later switching it to a passthrough.
        monkeypatch.setattr(gw, "load_peers", lambda: [LIVE, KILLED])
        payload = await gw.public_overview_payload()
        assert payload["hosts"]
        for h in payload["hosts"]:
            assert "routed" not in h


# ---------------------------------------------------------------------------
# the toggle
# ---------------------------------------------------------------------------


class TestTheToggleEndpoint:

    @staticmethod
    def _write(peers):
        gw.save_peers(peers)

    def test_it_flips_the_flag_and_persists_it(self, client, admin_headers):
        self._write([dict(LIVE)])
        r = client.put("/admin/api/peers/live/routed",
                       headers=admin_headers, json={"routed": False})
        assert r.status_code == 200
        assert r.json() == {"name": "live", "routed": False}
        assert json.loads(gw.PEERS_PATH.read_text())[0]["routed"] is False

        r = client.put("/admin/api/peers/live/routed",
                       headers=admin_headers, json={"routed": True})
        assert r.status_code == 200
        assert json.loads(gw.PEERS_PATH.read_text())[0]["routed"] is True

    def test_it_invalidates_the_caches_that_would_outlive_it(self, client, admin_headers):
        # A killswitch that takes half a minute to bite is not a killswitch.
        self._write([dict(LIVE)])
        gw._routes_cache["t"] = time.time()
        gw._known_ctx_cache["t"] = time.time()
        client.put("/admin/api/peers/live/routed",
                   headers=admin_headers, json={"routed": False})
        assert gw._routes_cache["t"] == 0.0
        assert gw._known_ctx_cache["t"] == 0.0

    def test_an_unknown_peer_is_a_404_not_a_new_record(self, client, admin_headers):
        self._write([dict(LIVE)])
        r = client.put("/admin/api/peers/ghost/routed",
                       headers=admin_headers, json={"routed": False})
        assert r.status_code == 404
        assert [p["name"] for p in json.loads(gw.PEERS_PATH.read_text())] == ["live"]

    @pytest.mark.parametrize("body", [{}, {"routed": "false"}, {"routed": 0}])
    def test_it_refuses_anything_but_a_boolean(self, client, admin_headers, body):
        # "false" is the string that would arrive from a form, and it is
        # truthy. Taking it would report success and route on regardless.
        self._write([dict(LIVE)])
        r = client.put("/admin/api/peers/live/routed",
                       headers=admin_headers, json=body)
        assert r.status_code == 400

    def test_it_needs_the_admin_token(self, client):
        self._write([dict(LIVE)])
        assert client.put("/admin/api/peers/live/routed",
                          json={"routed": False}).status_code in (401, 403)


class TestEditingPeersDoesNotDisturbTheKillswitch:
    """The peer editor's form has no killswitch field, so its PUT sends the
    whole list back without one. Read naively that would revive every killed
    box the moment anyone saved an unrelated URL change."""

    def test_an_edit_leaves_a_killed_box_killed(self, client, admin_headers):
        gw.save_peers([dict(KILLED)])
        r = client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "killed", "url": "http://killed:9090", "token": "t"}]})
        assert r.status_code == 200
        saved = json.loads(gw.PEERS_PATH.read_text())[0]
        assert saved["url"] == "http://killed:9090"   # the edit landed
        assert saved["routed"] is False               # the killswitch held

    def test_an_edit_cannot_kill_a_box_either(self, client, admin_headers):
        gw.save_peers([dict(LIVE)])
        client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "live", "url": "http://live:8080",
                       "token": "t", "routed": False}]})
        assert json.loads(gw.PEERS_PATH.read_text())[0]["routed"] is True

    def test_a_newly_added_peer_routes(self, client, admin_headers):
        gw.save_peers([])
        client.put("/admin/api/peers", headers=admin_headers, json={
            "peers": [{"name": "fresh", "url": "http://fresh:8080", "token": "t"}]})
        assert json.loads(gw.PEERS_PATH.read_text())[0]["routed"] is True
