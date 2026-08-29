"""The featured-model preload loop runs on the hub only, and leaves a box
that has a job of its own alone.

What this pins down, from 2026-08-29: preload_orchestrator() used to be
"has peers", on the theory that only the hub has any. Three boxes did that
day -- hub, gpu-laptop-1 (given apu-box-1 as a peer for the public tour) and
apu-box-1 (given three so its worker traffic could leave) -- and every one of
them ran the loop, each pinning the featured 27B onto every capable box every
ten minutes. On apu-box-1 that evicted the 125B it exists to hold, on a timer,
with nobody asking for anything. Two rules close it: the spec sheet's
`role: hub` decides who orchestrates, and a box whose own registry preloads
or keeps a model resident (reported as `warm` in served-models) is not
touched for anything else.
"""
from __future__ import annotations

import pytest

import app as gw


@pytest.fixture
def specs(monkeypatch):
    sheet = {"hub": {"role": "hub"}, "apu-box-1": {"klass": "big"},
             "gpu-laptop-1": {"klass": "gpu"}}
    monkeypatch.setattr(gw, "load_specs", lambda: sheet)
    monkeypatch.setattr(gw, "load_peers", lambda: [{"name": "x", "url": "http://127.0.0.1:9"}])
    monkeypatch.delenv("LLMSTACK_PRELOAD", raising=False)
    return sheet


class TestWhoOrchestrates:
    def test_the_hub_with_peers_does(self, specs, monkeypatch):
        monkeypatch.setattr(gw, "HOST_NAME", "hub")
        assert gw.preload_orchestrator() is True

    def test_a_peer_with_peers_of_its_own_does_not(self, specs, monkeypatch):
        """gpu-laptop-1 and apu-box-1 both had peers registered; neither is the hub."""
        for box in ("gpu-laptop-1", "apu-box-1"):
            monkeypatch.setattr(gw, "HOST_NAME", box)
            assert gw.preload_orchestrator() is False, box

    def test_the_hub_without_peers_has_nothing_to_drive(self, specs, monkeypatch):
        monkeypatch.setattr(gw, "HOST_NAME", "hub")
        monkeypatch.setattr(gw, "load_peers", lambda: [])
        assert gw.preload_orchestrator() is False

    def test_off_disarms_even_the_hub(self, specs, monkeypatch):
        monkeypatch.setattr(gw, "HOST_NAME", "hub")
        monkeypatch.setenv("LLMSTACK_PRELOAD", "off")
        assert gw.preload_orchestrator() is False

    def test_on_forces_it_for_a_hub_the_sheet_does_not_name(self, specs, monkeypatch):
        monkeypatch.setattr(gw, "HOST_NAME", "some-new-hub")
        monkeypatch.setenv("LLMSTACK_PRELOAD", "on")
        assert gw.preload_orchestrator() is True


class TestLocalWarmIds:
    def test_preload_and_persistent_count_disabled_does_not(self, monkeypatch):
        rows = [
            {"id": "big", "enabled": True, "preload": True},
            {"id": "small", "enabled": True, "persistent": True},
            {"id": "off", "enabled": False, "preload": True},
            {"id": "plain", "enabled": True},
        ]
        monkeypatch.setattr(gw, "load_models", lambda: rows)
        assert gw.local_warm_ids() == {"big", "small"}


class TestTheDedicatedBoxIsLeftAlone:
    """preload_plan() against a hand-built routing cache, the way
    test_public_presentation drives it, with one box declaring a warm model."""

    def _fleet(self, monkeypatch, warm):
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
        gw._inflight.clear()
        gw._host_last_used.clear()
        gw._preload_state.clear()
        gw._routes_cache.update(
            cands={"qwen3.8-27b": ["apu-box-1", "mac-laptop-1"]},
            running={"apu-box-1": {"qwen3.8-flash-next"}, "mac-laptop-1": set()},
            reachable={"apu-box-1", "mac-laptop-1"},
            warm=warm,
        )

    def test_a_box_warm_with_something_else_is_skipped_and_says_why(self, monkeypatch):
        self._fleet(monkeypatch, {"apu-box-1": {"qwen3.8-flash-next"}, "mac-laptop-1": set()})
        plan = gw.preload_plan()
        assert {p["host"] for p in plan} == {"mac-laptop-1"}
        assert gw._preload_state["apu-box-1"]["phase"] == "dedicated"
        assert "qwen3.8-flash-next" in gw._preload_state["apu-box-1"]["detail"]

    def test_a_box_warm_with_the_featured_model_is_still_touched(self, monkeypatch):
        self._fleet(monkeypatch, {"apu-box-1": {"qwen3.8-27b"}, "mac-laptop-1": set()})
        assert {p["host"] for p in gw.preload_plan()} == {"apu-box-1", "mac-laptop-1"}

    def test_a_box_that_says_nothing_is_treated_as_before(self, monkeypatch):
        """An older gateway omits `warm`; that must not read as 'dedicated'."""
        self._fleet(monkeypatch, {})
        assert {p["host"] for p in gw.preload_plan()} == {"apu-box-1", "mac-laptop-1"}
