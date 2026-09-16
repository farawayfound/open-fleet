"""model_routes() serves a stale table while it refreshes in the background.

Before 2026-09-16 the first request to find the routing table older than
ROUTE_TTL rebuilt it inline, awaiting a sweep of every routed peer -- ~6 s
with three boxes offline -- and the downstream app's lane planner (4 s timeout) read
that as "the hub did not answer" once every 30 s."""
from __future__ import annotations

import asyncio
import time

import pytest

import app as gw


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    saved = dict(gw._routes_cache)
    gw._routes_refresh_task = None
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)
    gw._routes_refresh_task = None


def _slow_refresh(delay: float, table: dict, calls: list):
    async def refresh():
        calls.append(time.time())
        await asyncio.sleep(delay)
        gw._routes_cache.update(t=time.time(), map=dict(table))
        return dict(table)
    return refresh


@pytest.mark.asyncio
async def test_a_stale_table_is_served_at_once_and_refreshed_behind_the_caller(monkeypatch):
    gw._routes_cache.update(t=time.time() - gw.ROUTE_TTL - 1, map={"old": "peer-a"})
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.2, {"new": "peer-b"}, calls))
    t0 = time.time()
    assert await gw.model_routes() == {"old": "peer-a"}
    assert time.time() - t0 < 0.1, "a stale read must not wait for the peer sweep"
    await asyncio.sleep(0)               # let the background task take its first step
    assert len(calls) == 1, "the refresh started behind the caller"
    await asyncio.sleep(0.3)
    assert gw._routes_cache["map"] == {"new": "peer-b"}
    assert await gw.model_routes() == {"new": "peer-b"}
    assert len(calls) == 1, "a fresh table is not refreshed again"


@pytest.mark.asyncio
async def test_concurrent_stale_reads_share_one_refresh(monkeypatch):
    gw._routes_cache.update(t=time.time() - gw.ROUTE_TTL - 1, map={"old": "peer-a"})
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.2, {"new": "peer-b"}, calls))
    results = await asyncio.gather(*(gw.model_routes() for _ in range(5)))
    assert all(r == {"old": "peer-a"} for r in results)
    assert len(calls) == 1
    await asyncio.sleep(0.3)


@pytest.mark.asyncio
async def test_the_first_fill_waits_for_the_sweep(monkeypatch):
    gw._routes_cache.update(t=0.0, map={})
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.1, {"m": "peer-b"}, calls))
    assert await gw.model_routes() == {"m": "peer-b"}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_invalidation_waits_for_a_fresh_sweep_like_before(monkeypatch):
    gw._routes_cache.update(t=time.time(), map={"old": "peer-a"})
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.1, {"new": "peer-b"}, calls))
    gw._routes_cache["t"] = 0.0          # what the killswitch / models editor do
    assert await gw.model_routes() == {"new": "peer-b"}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_force_waits_and_joins_an_in_flight_refresh(monkeypatch):
    gw._routes_cache.update(t=time.time() - gw.ROUTE_TTL - 1, map={"old": "peer-a"})
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.2, {"new": "peer-b"}, calls))
    assert await gw.model_routes() == {"old": "peer-a"}       # kicks the background refresh
    assert await gw.model_routes(force=True) == {"new": "peer-b"}
    assert len(calls) == 1, "force joined the refresh already in flight rather than starting a second"


@pytest.mark.asyncio
async def test_a_failed_background_refresh_keeps_the_previous_table(monkeypatch):
    gw._routes_cache.update(t=time.time() - gw.ROUTE_TTL - 1, map={"old": "peer-a"})

    async def boom():
        await asyncio.sleep(0.05)
        raise RuntimeError("peer sweep exploded")
    monkeypatch.setattr(gw, "_refresh_routes", boom)
    assert await gw.model_routes() == {"old": "peer-a"}
    await asyncio.sleep(0.15)
    assert gw._routes_cache["map"] == {"old": "peer-a"}
    # The next stale read tries again (the failed task is done).
    calls: list = []
    monkeypatch.setattr(gw, "_refresh_routes", _slow_refresh(0.05, {"new": "peer-b"}, calls))
    assert await gw.model_routes() == {"old": "peer-a"}
    await asyncio.sleep(0.15)
    assert calls and gw._routes_cache["map"] == {"new": "peer-b"}
