# -*- coding: utf-8 -*-
"""measured_tps()/measured_pp() are stale-while-revalidate.

The per-model usage aggregation behind them took ~6 s for the busiest model
on hub (2026-09-15) and was paid inline by whichever request arrived after
the 300 s cache lapsed -- /admin/api/resolve for the lane planner, which
gives up at 4 s. A stale estimate is returned at once and refreshed in the
background; only a model with no entry at all waits for the query."""
from __future__ import annotations

import threading
import time

import app as gw


def _wait_refresh(cache, model, before, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hit = cache.get(model)
        if hit and hit[0] > before:
            return hit
        time.sleep(0.01)
    raise AssertionError("background refresh never landed")


class TestMeasuredStaleWhileRevalidate:
    def test_cold_model_pays_the_query_inline_once(self):
        cache: dict = {}
        calls = []
        out = gw._measured_cached(cache, "m1", lambda m: (calls.append(m), {"mac-laptop-1": 12.0})[1])
        assert out == {"mac-laptop-1": 12.0}
        assert calls == ["m1"]
        again = gw._measured_cached(cache, "m1", lambda m: (calls.append(m), {"mac-laptop-1": 99.0})[1])
        assert again == {"mac-laptop-1": 12.0}, "a fresh entry is served without a query"
        assert calls == ["m1"]

    def test_stale_entry_is_returned_at_once_and_refreshed_in_the_background(self):
        cache: dict = {"m2": (time.time() - gw.MEASURED_TTL - 1, {"mac-laptop-1": 12.0})}
        gate = threading.Event()
        calls = []

        def slow_query(m):
            calls.append(m)
            gate.wait(5)
            return {"mac-laptop-1": 20.0}

        before = time.time()
        t0 = time.time()
        out = gw._measured_cached(cache, "m2", slow_query)
        assert time.time() - t0 < 1.0, "a stale read must not wait for the query"
        assert out == {"mac-laptop-1": 12.0}, "the stale value is what the caller gets"
        # a second stale read while the refresh is in flight starts no second query
        out2 = gw._measured_cached(cache, "m2", slow_query)
        assert out2 == {"mac-laptop-1": 12.0}
        assert calls == ["m2"]
        gate.set()
        hit = _wait_refresh(cache, "m2", before)
        assert hit[1] == {"mac-laptop-1": 20.0}
        assert gw._measured_cached(cache, "m2", slow_query) == {"mac-laptop-1": 20.0}
        assert calls == ["m2"], "the refreshed entry is fresh: no further query"

    def test_failed_refresh_keeps_the_stale_entry_and_clears_the_mark(self):
        cache: dict = {"m3": (time.time() - gw.MEASURED_TTL - 1, {"mac-laptop-1": 12.0})}
        calls = []

        def broken(m):
            calls.append(m)
            raise RuntimeError("db down")

        assert gw._measured_cached(cache, "m3", broken) == {"mac-laptop-1": 12.0}
        t0 = time.time()
        while (id(cache), "m3") in gw._measured_refreshing and time.time() - t0 < 5:
            time.sleep(0.01)
        assert (id(cache), "m3") not in gw._measured_refreshing
        assert cache["m3"][1] == {"mac-laptop-1": 12.0}, "the stale entry survives a failed refresh"
        # the next stale read tries again rather than being stuck behind a dead mark
        assert gw._measured_cached(cache, "m3", broken) == {"mac-laptop-1": 12.0}
        t0 = time.time()
        while len(calls) < 2 and time.time() - t0 < 5:
            time.sleep(0.01)
        assert len(calls) == 2

    def test_measured_tps_and_pp_share_the_policy(self, monkeypatch):
        gw._tps_cache.clear()
        gw._pp_cache.clear()
        monkeypatch.setattr(gw, "_query_tps", lambda m: {"mac-laptop-1": 30.0})
        monkeypatch.setattr(gw, "_query_pp", lambda m: {"mac-laptop-1": 300.0})
        assert gw.measured_tps("mx") == {"mac-laptop-1": 30.0}
        assert gw.measured_pp("mx") == {"mac-laptop-1": 300.0}
        gw._tps_cache["mx"] = (time.time() - gw.MEASURED_TTL - 1, {"mac-laptop-1": 1.0})
        monkeypatch.setattr(gw, "_query_tps", lambda m: {"mac-laptop-1": 31.0})
        before = time.time()
        assert gw.measured_tps("mx") == {"mac-laptop-1": 1.0}, "stale served immediately"
        assert _wait_refresh(gw._tps_cache, "mx", before)[1] == {"mac-laptop-1": 31.0}
        gw._tps_cache.clear()
        gw._pp_cache.clear()


class TestUsageModelIndex:
    def test_migration_creates_the_model_ts_index(self):
        assert any("usage_model_ts" in m for m in gw.MIGRATIONS)
        gw.db_init()
        rows = gw.db_query("SELECT name FROM sqlite_master WHERE type='index' AND name='usage_model_ts'")
        assert rows and rows[0]["name"] == "usage_model_ts"
