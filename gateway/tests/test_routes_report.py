"""The routes table reports the scorer's pick, and /admin/api/resolve says
where a request would land and why.

model_routes() records who SERVES each name in poll order (this box, then
peers in peers.json order). For an evening on 2026-09-09 /admin/api/routes
reported that map as "where a request goes": qwen3.8-27b -> server-1, the
first peer polled and a 4 tok/s CPU box, while mac-laptop-1 sat awake at tier 0
with the model resident and the proxy was in fact sending every request to
mac-laptop-1. Two sessions read the table as the fleet's decision. These tests
pin the table to the scorer, and give the scorer a readable surface.
"""
import pytest

import app as gw


@pytest.fixture(autouse=True)
def _isolate_routes_cache():
    snapshot = dict(gw._routes_cache)
    gw._host_cooldown.clear()
    gw._inflight.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._host_cooldown.clear()
    gw._inflight.clear()


@pytest.fixture
def two_boxes(monkeypatch):
    """server-1 polled first (so it owns the map), mac-laptop-1 holding the model
    resident in unified memory: the exact shape of the 2026-09-09 case."""
    gw._routes_cache.update(
        t=0.0,
        map={"m": "server-1"},
        cands={"m": ["server-1", "mac-laptop-1"]},
        cap={("server-1", "m"): 1, ("mac-laptop-1", "m"): 2},
        running={"server-1": set(), "mac-laptop-1": {"m"}, gw.HOST_NAME: set()},
        ctx={("server-1", "m"): 32768, ("mac-laptop-1", "m"): 131072},
        meta={("server-1", "m"): {"fit": "spill", "bytes": 15_705_861_088},
              ("mac-laptop-1", "m"): {"fit": "unified", "bytes": 20_699_117_152}},
        alias={}, engine={}, warm={}, reachable={gw.HOST_NAME, "server-1", "mac-laptop-1"},
    )

    async def _frozen(force: bool = False):
        return gw._routes_cache["map"]
    monkeypatch.setattr(gw, "model_routes", _frozen)


def test_routes_table_reports_the_scorer_not_the_poll_order(client, admin_headers, two_boxes):
    r = client.get("/admin/api/routes", headers=admin_headers)
    assert r.status_code == 200
    row = {m["model"]: m for m in r.json()["models"]}["m"]
    assert row["host"] == "mac-laptop-1"                      # tier 0, resident -- not "polled first"
    assert row["serving"] == ["mac-laptop-1", "server-1"]     # both could answer; the reader sees that
    assert gw._routes_cache["map"]["m"] == "server-1"   # the map itself is untouched


def test_resolve_ranks_like_the_proxy_and_shows_why(client, admin_headers, two_boxes):
    r = client.get("/admin/api/resolve", params={"model": "m", "prompt_tokens": 10000,
                                                  "gen_tokens": 900}, headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["served"] is True and body["prompt_tokens"] == 10000
    first, second = body["targets"]
    assert first["host"] == "mac-laptop-1" and first["tier"] == 0 and first["resident"] is True
    assert first["fit"] == "unified" and first["ctx"] == 131072 and first["slots"] == 2
    assert first["saturated"] is False and first["cooling"] is False and first["reserved"] is False
    assert second["host"] == "server-1" and second["tier"] == 2 and second["resident"] is False
    assert second["est_s"] > first["est_s"]


def test_resolve_shows_the_state_that_demotes_a_box(client, admin_headers, two_boxes):
    """The two things that can put a tier-0 box behind a CPU box are both
    in-process state nothing else exposes: a full slot count and a cooldown.
    Both are named in the answer, so the next such evening is a GET."""
    gw._inflight["mac-laptop-1"] = 2                          # both slots busy
    r = client.get("/admin/api/resolve", params={"model": "m"}, headers=admin_headers)
    t = {x["host"]: x for x in r.json()["targets"]}
    assert r.json()["targets"][0]["host"] == "server-1"
    assert t["mac-laptop-1"]["saturated"] is True and t["mac-laptop-1"]["inflight"] == 2
    gw._inflight.clear()
    gw._mark_host_down("mac-laptop-1", 120, "test")
    r = client.get("/admin/api/resolve", params={"model": "m"}, headers=admin_headers)
    t = {x["host"]: x for x in r.json()["targets"]}
    assert r.json()["targets"][0]["host"] == "server-1" and t["mac-laptop-1"]["cooling"] is True


def test_resolve_for_a_name_nobody_serves(client, admin_headers, two_boxes):
    r = client.get("/admin/api/resolve", params={"model": "nope"}, headers=admin_headers)
    assert r.status_code == 200
    assert r.json() == {**r.json(), "served": False, "targets": []}


def test_resolve_needs_the_admin_token(client, two_boxes):
    assert client.get("/admin/api/resolve", params={"model": "m"}).status_code in (401, 403)
