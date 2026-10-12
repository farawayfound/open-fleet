"""Pressure-warm (A1-A9): sustained demand for a model whose best candidate
keeps coming up saturated gets its own, separate preload entry -- fired
every tick alongside (never instead of) the featured/demo preload loop, but
driven entirely off resolve_targets()'s own choke point (_note_pressure),
zero LLM calls, zero new HTTP surface.

Covers, in order: the pure demand bookkeeping (_record_pressure/
_pressured_models), pressure_warm_plan()'s own cascade (capability,
dedication, busy, idle-grace, cooling, retry backoff, reserve-gating, the
cooldown re-touch rule), preload_pass()'s integration of the two loops, the
resolve_targets()/_note_pressure() choke point itself (direct call and the
GET /admin/api/resolve preview must record identically), and the new public
settings' defaults/bounds.

Run with: gateway/.venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import pytest

import app as gw

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolate():
    """Every test here pokes _routes_cache and the pressure/preload module
    globals directly, so the next test (in this file or elsewhere in the
    session -- these are module singletons) must not see what one left
    behind."""
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()
    gw._preload_state.clear()
    gw._pressure_events.clear()
    gw._overflow_demoted.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    gw._host_last_used.clear()
    gw._preload_state.clear()
    gw._pressure_events.clear()
    gw._overflow_demoted.clear()


# ---------------------------------------------------------------------------
# _record_pressure / _pressured_models: the pure demand bookkeeping
# ---------------------------------------------------------------------------

class TestRecordPressureAndPressuredModels:
    def test_window_pruning_drops_events_older_than_pressure_window(self, monkeypatch):
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-a")
        t[0] += gw.PRESSURE_WINDOW + 1
        gw._record_pressure("fid-a")  # the first entry is now outside the window
        # _record_pressure() itself prunes on write, so only the most recent
        # observation survives -- which looks like a single, brand-new blip.
        assert gw._pressure_events["fid-a"] == [t[0]]
        assert gw._pressured_models(0.0) == [("fid-a", 1, t[0])]

    def test_a_blip_is_not_sustained_until_after_s_elapses(self, monkeypatch):
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-a")
        t[0] += 10.0  # only 10s old: not sustained at after_s=45
        assert gw._pressured_models(45.0) == []
        t[0] = 1_000_000.0 + 50.0  # 50s after the FIRST (only) observation
        out = gw._pressured_models(45.0)
        assert len(out) == 1 and out[0][0] == "fid-a"

    def test_most_and_longest_pressured_sorts_first(self, monkeypatch):
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-a")           # 1 observation, oldest
        t[0] += 10.0
        gw._record_pressure("fid-b")
        t[0] += 1.0
        gw._record_pressure("fid-b")
        t[0] += 1.0
        gw._record_pressure("fid-b")           # fid-b: 3 observations
        t[0] += 100.0                          # well past after_s for both
        out = gw._pressured_models(45.0)
        # Count wins the sort before earliest-timestamp: fid-b first.
        assert [fid for fid, _, _ in out] == ["fid-b", "fid-a"]

    def test_self_cleaning_evicts_an_empty_window_fid_on_read(self, monkeypatch):
        # Correction (7): eviction happens on READ (_pressured_models), the
        # same idiom silent_answer_cooling() uses -- not only the next
        # _record_pressure() write.
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-a")
        assert "fid-a" in gw._pressure_events
        t[0] += gw.PRESSURE_WINDOW + 1
        assert gw._pressured_models(0.0) == []
        assert "fid-a" not in gw._pressure_events

    def test_a_threshold_above_the_old_fixed_120s_window_still_detects_pressure(
            self, monkeypatch):
        # Regression: PRESSURE_WINDOW used to be a hardcoded 120s, so both
        # _record_pressure()'s write-side prune and _pressured_models()'s
        # own cutoff threw away any event older than that -- meaning `now -
        # earliest` could never reach an after_s configured above 120s, and
        # _pressured_models() permanently returned [] no matter how long
        # the pressure lasted. pressure_warm_after_s is admin-configurable
        # up to 600s (_PUBLIC_SETTING_BOUNDS) and the dashboard exposes the
        # whole range -- 300s is comfortably inside it and well past the
        # old fixed window.
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-x")
        t[0] += 300.0  # sustained for 300s -- the old code pruned this away
        out = gw._pressured_models(300.0)
        assert out, "300s of sustained saturation must be visible at after_s=300"
        assert out[0][0] == "fid-x"

    def test_after_s_at_the_very_top_of_its_allowed_bound_still_works(
            self, monkeypatch):
        lo, hi = gw._PUBLIC_SETTING_BOUNDS["pressure_warm_after_s"]
        assert hi == 600  # pin the bound this test (and the fix) size against
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._record_pressure("fid-x")
        t[0] += hi
        out = gw._pressured_models(hi)
        assert out and out[0][0] == "fid-x"


# ---------------------------------------------------------------------------
# pressure_warm_plan(): built on a stubbed _pressured_models() (covered on
# its own above), so these tests are purely about the cascade -- capability,
# dedication, busy, idle-grace, cooling, retry backoff, reserve-gating, and
# the cooldown re-touch rule -- mirroring preload_plan()'s own tests.
# ---------------------------------------------------------------------------

class TestPressureWarmPlan:
    def _fleet(self, monkeypatch, *, cands, running=None, reachable=None,
               warm=None, specs=None):
        monkeypatch.setattr(gw, "eclipsed_hosts", lambda: set())
        monkeypatch.setattr(gw, "preload_capable", lambda c, f: True)
        monkeypatch.setattr(gw, "load_specs", lambda: specs or {})
        all_hosts = {h for hs in cands.values() for h in hs}
        gw._routes_cache.update(
            cands=cands,
            running=running if running is not None else {h: set() for h in all_hosts},
            reachable=reachable if reachable is not None else set(all_hosts),
            warm=warm or {},
        )

    def _pressured(self, monkeypatch, t, *fids):
        """Freeze the clock at t[0] and make _pressured_models() report every
        given fid as sustained-pressured, whatever after_s is asked for."""
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        monkeypatch.setattr(
            gw, "_pressured_models",
            lambda after_s: [(fid, 5, t[0] - 50.0) for fid in fids])

    async def test_disabled_returns_none_even_with_sustained_pressure(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        self._pressured(monkeypatch, t, "m")
        gw.set_public_settings({"pressure_warm_enabled": False})
        assert gw.pressure_warm_plan() is None

    async def test_no_sustained_pressure_returns_none(self, monkeypatch):
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        monkeypatch.setattr(gw, "_pressured_models", lambda after_s: [])
        assert gw.pressure_warm_plan() is None

    async def test_reserve_host_used_only_when_pressure_warm_reserve_is_on(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["reservebox"]},
                    specs={"reservebox": {"reserve": True}})
        self._pressured(monkeypatch, t, "m")
        gw.set_public_settings({"pressure_warm_reserve": False})
        assert gw.pressure_warm_plan() is None, \
            "preload_plan() always skips a reserve box; pressure-warm must too when the flag is off"
        gw.set_public_settings({"pressure_warm_reserve": True})
        plan = gw.pressure_warm_plan()
        assert plan is not None
        assert plan["host"] == "reservebox" and plan["reason"] == "pressure"

    async def test_skips_a_host_dedicated_to_a_different_model(self, monkeypatch):
        # Mirrors test_preload_dedicated.py's TestTheDedicatedBoxIsLeftAlone
        # fixture: a box's own configured preload/persistent model is not
        # evicted for anything else, exclusive-swap-group safety included.
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["apu-box-1"]}, warm={"apu-box-1": {"other-model"}})
        self._pressured(monkeypatch, t, "m")
        assert gw.pressure_warm_plan() is None

    async def test_skips_a_busy_host(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        self._pressured(monkeypatch, t, "m")
        gw._inflight["boxA"] = 1
        assert gw.pressure_warm_plan() is None

    async def test_skips_a_host_inside_idle_grace(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        self._pressured(monkeypatch, t, "m")
        gw._host_last_used["boxA"] = t[0] - 1.0  # served 1s ago
        assert gw.pressure_warm_plan() is None

    async def test_skips_a_cooling_host(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        self._pressured(monkeypatch, t, "m")
        gw._host_cooldown["boxA"] = t[0] + 60.0
        assert gw.pressure_warm_plan() is None

    async def test_skips_a_host_still_inside_its_retry_backoff(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]})
        self._pressured(monkeypatch, t, "m")
        gw._preload_state["boxA"] = {"retry_at": t[0] + 60.0}
        assert gw.pressure_warm_plan() is None

    async def test_resident_pressure_warmed_host_skipped_until_cooldown_then_offered_again(
            self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m": ["boxA"]}, running={"boxA": {"m"}})
        self._pressured(monkeypatch, t, "m")
        gw._preload_state["boxA"] = {"touched": t[0]}  # just touched
        gw.set_public_settings({"pressure_warm_cooldown_s": 600})
        assert gw.pressure_warm_plan() is None, \
            "still inside the cooldown -- skipped cheaply, not re-touched"
        t[0] += 601
        plan = gw.pressure_warm_plan()
        assert plan is not None
        assert plan["host"] == "boxA" and plan["resident"] is True
        assert plan["reason"] == "pressure"

    async def test_never_returns_more_than_one_entry_for_two_qualifying_fids(self, monkeypatch):
        t = [1_000_000.0]
        self._fleet(monkeypatch, cands={"m1": ["boxA"], "m2": ["boxB"]})
        self._pressured(monkeypatch, t, "m1", "m2")
        plan = gw.pressure_warm_plan()
        assert plan is not None
        # The function's own return type already caps this at one dict;
        # this pins WHICH one -- the first (most/longest pressured) fid.
        assert plan["fleet_id"] == "m1"


# ---------------------------------------------------------------------------
# preload_pass(): the pressure entry rides the SAME tick and the SAME
# asyncio.gather as the featured/demo plan -- added even when the featured
# loop itself found nothing to do (correction (5)).
# ---------------------------------------------------------------------------

class TestPreloadPassIntegration:
    async def test_pressure_entry_is_touched_alongside_the_featured_plan(self, monkeypatch):
        monkeypatch.setattr(gw, "preload_orchestrator", lambda: True)

        async def _routes(force: bool = False):
            return {}
        monkeypatch.setattr(gw, "model_routes", _routes)

        featured_item = {"host": "featuredbox", "cand": "featuredbox",
                         "fleet_id": "featured-m", "public_id": "featured-m",
                         "resident": False}
        monkeypatch.setattr(gw, "preload_plan", lambda: [featured_item])

        pressure_item = {"host": "pressurebox", "cand": "pressurebox",
                         "fleet_id": "pressured-m", "public_id": "pressured-m",
                         "resident": False, "reason": "pressure"}
        monkeypatch.setattr(gw, "pressure_warm_plan", lambda: pressure_item)

        touched: list[dict] = []

        async def _fake_touch(item):
            touched.append(item)
        monkeypatch.setattr(gw, "_preload_touch", _fake_touch)

        plan = await gw.preload_pass()

        assert plan == [featured_item, pressure_item]
        assert {(t["host"], t.get("reason")) for t in touched} == {
            ("featuredbox", None), ("pressurebox", "pressure"),
        }
        # _preload_note's **item splat carries the 'reason' key straight
        # into _preload_state with no code change needed there.
        assert gw._preload_state["pressurebox"]["reason"] == "pressure"

    async def test_a_pressure_entry_fires_even_when_the_featured_plan_is_empty(
            self, monkeypatch):
        # correction (5): pressure-warm must not wait for the featured
        # preload to have something to do -- the common steady state.
        monkeypatch.setattr(gw, "preload_orchestrator", lambda: True)

        async def _routes(force: bool = False):
            return {}
        monkeypatch.setattr(gw, "model_routes", _routes)
        monkeypatch.setattr(gw, "preload_plan", lambda: [])

        pressure_item = {"host": "pressurebox", "cand": "pressurebox",
                         "fleet_id": "pressured-m", "public_id": "pressured-m",
                         "resident": False, "reason": "pressure"}
        monkeypatch.setattr(gw, "pressure_warm_plan", lambda: pressure_item)

        touched: list[dict] = []

        async def _fake_touch(item):
            touched.append(item)
        monkeypatch.setattr(gw, "_preload_touch", _fake_touch)

        plan = await gw.preload_pass()
        assert plan == [pressure_item]
        assert touched == [pressure_item]


# ---------------------------------------------------------------------------
# resolve_targets() / _note_pressure(): the one choke point every caller
# shares -- the real /v1 dispatch and the admin resolve preview alike.
# ---------------------------------------------------------------------------

class TestNotePressureChokePoint:
    async def test_saturated_only_candidate_records_pressure(self, monkeypatch):
        gw._routes_cache.update(cands={"m": ["hostA"]}, cap={("hostA", "m"): 1},
                                running={}, meta={})

        async def _routes(force: bool = False):
            return {}
        monkeypatch.setattr(gw, "model_routes", _routes)
        gw._inflight["hostA"] = 1  # busy >= slots(1): saturated
        out = await gw.resolve_targets("m", role="primary")
        assert out and out[0][0] == "hostA"
        assert "m" in gw._pressure_events and len(gw._pressure_events["m"]) == 1

    async def test_free_only_candidate_does_not_record_pressure(self, monkeypatch):
        gw._routes_cache.update(cands={"m": ["hostA"]}, cap={("hostA", "m"): 1},
                                running={}, meta={})

        async def _routes(force: bool = False):
            return {}
        monkeypatch.setattr(gw, "model_routes", _routes)
        out = await gw.resolve_targets("m", role="primary")
        assert out and out[0][0] == "hostA"
        assert "m" not in gw._pressure_events

    async def test_admin_resolve_preview_records_the_same_as_a_direct_call(
            self, client, monkeypatch, admin_headers):
        gw._routes_cache.update(cands={"m": ["hostA"]}, cap={("hostA", "m"): 1},
                                running={}, meta={}, reachable={"hostA"})

        async def _routes(force: bool = False):
            return {}
        monkeypatch.setattr(gw, "model_routes", _routes)
        gw._inflight["hostA"] = 1
        r = client.get("/admin/api/resolve", params={"model": "m"}, headers=admin_headers)
        assert r.status_code == 200, r.text
        # The preview call went through the SAME resolve_targets() choke
        # point as a real /v1 dispatch would -- one observation recorded,
        # not zero and not two.
        assert "m" in gw._pressure_events and len(gw._pressure_events["m"]) == 1


# ---------------------------------------------------------------------------
# public settings: defaults and bounds for the new A/B knobs
# ---------------------------------------------------------------------------

class TestPressureAndOverflowSettingsDefaultsAndBounds:
    def test_defaults_when_unset(self):
        s = gw.get_public_settings()
        assert s["pressure_warm_enabled"] is True
        assert s["pressure_warm_reserve"] is True
        assert s["pressure_warm_after_s"] == 45
        assert s["pressure_warm_cooldown_s"] == 600
        assert s["overflow_max_slowdown"] == 2.0

    def test_bounds_clamp_out_of_range_values(self):
        gw.set_public_settings({
            "pressure_warm_after_s": 1,        # below (5, 600)
            "pressure_warm_cooldown_s": 5000,  # above (60, 3600)
            "overflow_max_slowdown": 99.0,     # above (0.0, 10.0)
        })
        s = gw.get_public_settings()
        assert s["pressure_warm_after_s"] == 5
        assert s["pressure_warm_cooldown_s"] == 3600
        assert s["overflow_max_slowdown"] == 10.0

    def test_overflow_max_slowdown_zero_is_a_real_off_not_the_default(self):
        # Unlike busy_cooldown_seconds (0 falls back to the default), a
        # stored 0 here is a deliberate "off" -- ctx_ceiling's own idiom.
        gw.set_public_settings({"overflow_max_slowdown": 0})
        assert gw.get_public_settings()["overflow_max_slowdown"] == 0.0
        assert gw._overflow_max_slowdown() == 0.0

    def test_get_public_settings_and_set_public_settings_both_seed_the_new_keys(self):
        # DEFAULT_PUBLIC_SETTINGS is the source both get_public_settings()
        # (via dict(DEFAULT_PUBLIC_SETTINGS)) and set_public_settings() (via
        # its `k in DEFAULT_PUBLIC_SETTINGS` filter) key off of -- a setting
        # missing from it is silently dropped by both. Pin all seven here.
        for key in ("overflow_max_slowdown", "pressure_warm_enabled",
                    "pressure_warm_reserve", "pressure_warm_after_s",
                    "pressure_warm_cooldown_s"):
            assert key in gw.DEFAULT_PUBLIC_SETTINGS, key
        for key in ("overflow_max_slowdown", "pressure_warm_after_s",
                    "pressure_warm_cooldown_s"):
            assert key in gw._PUBLIC_SETTING_BOUNDS, key
