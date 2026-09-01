"""Personal machines -- gpu-desktop-2 under somebody's desk, gpu-desktop-1 when its
owner is at the keyboard, gpu-laptop-2's gaming rig, apu-tablet-1/apu-tablet-2 (the
two Z13s), mac-laptop-2 (the M4 that sleeps) -- stay full members of the fleet:
registered, warm with whatever their owner is running, answering when
asked. What they must never do is get PLANNED for: the router should not
route a stranger's request onto gpu-desktop-2 just because its two 3090s beat
everything else on paper, and the preload loop should not evict whatever
its owner left resident to keep the featured model warm on a timer nobody
asked for. `host_reserved()` reads the spec sheet's `reserve` flag that
marks these boxes, and this file pins down the two places that act on it:
`_score_host_model_pairs` demotes a reserved box to "only when every other
candidate is saturated or cooling" rather than dropping it out of routing
altogether, and `preload_plan()` skips it outright, the same way it already
leaves alone a box with a warm model of its own (test_preload_dedicated.py).

Run with: cd gateway && ./.venv/Scripts/python.exe -m pytest tests/test_reserve_boxes.py -q
"""
from __future__ import annotations

import pytest

import app as gw


def _set_meta(**by_host_fid: dict) -> None:
    """by_host_fid keys are 'host|fid' strings, mapped to a meta dict (see
    test_routing_policy.py, which this file borrows the shape from)."""
    gw._routes_cache["meta"] = {
        tuple(k.split("|", 1)): v for k, v in by_host_fid.items()
    }


@pytest.fixture(autouse=True)
def _isolate_routes_cache():
    """Every test here pokes _routes_cache directly (cap/running/meta)
    rather than going through a real routing refresh, so the next test file
    must not see what this one left behind."""
    snapshot = dict(gw._routes_cache)
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)


@pytest.fixture(autouse=True)
def _isolate_cooldown_and_inflight():
    snapshot_cooldown = dict(gw._host_cooldown)
    snapshot_inflight = dict(gw._inflight)
    gw._host_cooldown.clear()
    gw._inflight.clear()
    yield
    gw._host_cooldown.clear()
    gw._host_cooldown.update(snapshot_cooldown)
    gw._inflight.clear()
    gw._inflight.update(snapshot_inflight)


@pytest.fixture(autouse=True)
def _isolate_preload_state():
    snapshot = dict(gw._preload_state)
    yield
    gw._preload_state.clear()
    gw._preload_state.update(snapshot)


# ---------------------------------------------------------------------------
# host_reserved / DEFAULT_SPECS: exactly the six personal machines
# ---------------------------------------------------------------------------

class TestHostReservedDefaultSpecs:
    def test_default_specs_marks_exactly_the_personal_machines(self):
        reserved = {"apu-tablet-1", "apu-tablet-2", "gpu-desktop-2", "gpu-laptop-2", "gpu-desktop-1", "mac-laptop-2"}
        not_reserved = {"apu-box-1", "mac-laptop-1", "gpu-laptop-1", "mac-desktop-1", "cpu-box-1",
                         "server-1", "mini-pc-1", "hub"}
        assert reserved | not_reserved == set(gw.DEFAULT_SPECS)
        for host in reserved:
            assert gw.DEFAULT_SPECS[host].get("reserve") is True, host
        for host in not_reserved:
            assert not gw.DEFAULT_SPECS[host].get("reserve"), host

    def test_host_reserved_reads_the_spec_sheets_flag(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: {
            "gpu-desktop-2": {"reserve": True}, "apu-box-1": {"klass": "big"},
        })
        assert gw.host_reserved("gpu-desktop-2") is True
        assert gw.host_reserved("apu-box-1") is False
        # A host the sheet has never heard of is simply not reserved.
        assert gw.host_reserved("brand-new-peer") is False


# ---------------------------------------------------------------------------
# _score_host_model_pairs: a reserve box is last resort, not last place --
# it loses to an idle ordinary box but beats one that is saturated or cooling
# ---------------------------------------------------------------------------

class TestScoreHostModelPairsReserveOrdering:
    """A synthetic two-box fleet, 'normalgpu' and 'reservedgpu', both bare
    GPU-klass boxes with no rank/always_on field. Reporting fit='vram' for
    model 'm' ties them at host_tier() == (0, 0) -- the same construction
    test_routing_policy.py uses for gpu-desktop-1 vs. mac-laptop-1 -- so only the reserved
    bit (and whatever saturation/cooldown a test adds) can move the order."""

    SPECS = {
        "normalgpu": {"klass": "gpu"},
        "reservedgpu": {"klass": "gpu", "reserve": True},
    }

    def _tie(self, monkeypatch):
        monkeypatch.setattr(gw, "load_specs", lambda: self.SPECS)
        gw._routes_cache["cap"] = {("normalgpu", "m"): 1, ("reservedgpu", "m"): 1}
        gw._routes_cache["running"] = {}
        _set_meta(**{"normalgpu|m": {"fit": "vram"}, "reservedgpu|m": {"fit": "vram"}})
        assert gw.host_tier("normalgpu", "m", "primary") == gw.host_tier("reservedgpu", "m", "primary")

    @pytest.mark.asyncio
    async def test_idle_reserve_box_loses_to_an_idle_normal_box(self, monkeypatch):
        self._tie(monkeypatch)
        out = await gw._score_host_model_pairs(
            [("reservedgpu", "m"), ("normalgpu", "m")], role="primary")
        assert out == [("normalgpu", "m"), ("reservedgpu", "m")]

    @pytest.mark.asyncio
    async def test_idle_reserve_box_beats_a_saturated_normal_box(self, monkeypatch):
        self._tie(monkeypatch)
        gw._inflight["normalgpu"] = 1  # cap is 1 slot -> busy >= slots: saturated
        out = await gw._score_host_model_pairs(
            [("normalgpu", "m"), ("reservedgpu", "m")], role="primary")
        assert out == [("reservedgpu", "m"), ("normalgpu", "m")]

    @pytest.mark.asyncio
    async def test_idle_reserve_box_beats_a_cooling_normal_box(self, monkeypatch):
        self._tie(monkeypatch)
        gw._mark_host_down("normalgpu", 60.0, "test")
        out = await gw._score_host_model_pairs(
            [("normalgpu", "m"), ("reservedgpu", "m")], role="primary")
        assert out == [("reservedgpu", "m"), ("normalgpu", "m")]


# ---------------------------------------------------------------------------
# preload_plan: a reserve box is skipped outright, never pinned
# ---------------------------------------------------------------------------

class TestPreloadPlanSkipsReserveBoxes:
    """preload_plan() against a hand-built routing cache, following
    test_preload_dedicated.py's fixture style, with one of the two
    candidates flagged `reserve` on the spec sheet instead of `warm` with
    something else."""

    def _fleet(self, monkeypatch, specs: dict) -> None:
        monkeypatch.setattr(gw, "get_public_settings",
                            lambda: {"featured_model": "qwen3.8-27b",
                                     "preload_featured": True})
        monkeypatch.setattr(gw, "featured_public_id", lambda s=None: "qwen3.8-27b")
        row = {"public_id": "qwen3.8-27b", "enabled": 1,
               "fleet_ids": ["qwen3.8-27b"]}
        monkeypatch.setattr(gw, "public_catalogue",
                            lambda: {"by_public": {"qwen3.8-27b": row}})
        monkeypatch.setattr(gw, "_row_fleet_ids", lambda r: ["qwen3.8-27b"])
        monkeypatch.setattr(gw, "eclipsed_hosts", lambda: set())
        monkeypatch.setattr(gw, "preload_capable", lambda c, f: True)
        monkeypatch.setattr(gw, "host_cooling", lambda h: False)
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        gw._inflight.clear()
        gw._host_last_used.clear()
        gw._preload_state.clear()
        gw._routes_cache.update(
            cands={"qwen3.8-27b": ["gpu-desktop-2", "mac-laptop-1"]},
            running={"gpu-desktop-2": set(), "mac-laptop-1": set()},
            reachable={"gpu-desktop-2", "mac-laptop-1"},
            warm={},
        )

    def test_reserve_host_is_skipped_and_noted_normal_host_still_plans(self, monkeypatch):
        specs = {"gpu-desktop-2": {"reserve": True}, "mac-laptop-1": {}}
        self._fleet(monkeypatch, specs)
        plan = gw.preload_plan()
        planned = {p["host"] for p in plan}
        assert planned == {"mac-laptop-1"}
        assert "gpu-desktop-2" not in planned
        assert gw._preload_state["gpu-desktop-2"]["phase"] == "reserve"
        assert "personal box" in gw._preload_state["gpu-desktop-2"]["detail"]

    def test_neither_host_reserved_both_still_plan(self, monkeypatch):
        # Control case: with the reserve flag off the same two candidates
        # both land in the plan, confirming the skip above is the `reserve`
        # flag's doing and not some other quirk of this fixture shape.
        specs = {"gpu-desktop-2": {}, "mac-laptop-1": {}}
        self._fleet(monkeypatch, specs)
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"gpu-desktop-2", "mac-laptop-1"}
        assert "gpu-desktop-2" not in gw._preload_state or \
            gw._preload_state["gpu-desktop-2"].get("phase") != "reserve"
