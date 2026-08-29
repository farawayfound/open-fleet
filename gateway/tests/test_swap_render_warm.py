"""Warm-on-boot and never-evicted models in the rendered llama-swap config.

Two record fields, added 2026-08-29 when the fleet's boxes were given jobs:

  preload     -- load at llama-swap start-up (hooks.on_startup.preload), so a
                 box dedicated to one model pays its cold load at boot, not on
                 the first request after every reboot. ttl 0 beside it is
                 what makes the model stay.
  persistent  -- its own llama-swap group that nothing can evict, next to the
                 exclusive swap group. For a box that holds a small always-on
                 model and still serves big ones on demand.

The shapes here are llama-swap's own (config.example.yaml): `persistent: true`
means other groups cannot unload the group's members, `exclusive: true` on
the main group evicts every other non-persistent group when a member loads.
"""
from __future__ import annotations

import pytest
import yaml
from fastapi import HTTPException

import app as gw


def rec(mid: str, **kw) -> dict:
    r = dict(gw.DEFAULT_MODEL_RECORD)
    r.update(id=mid, path="/models/" + mid + ".gguf")
    r.update(kw)
    return r


def render(models):
    return yaml.safe_load(gw.render_swap_config(models))


class TestPreload:
    def test_no_flags_means_no_hooks_and_one_exclusive_group(self):
        cfg = render([rec("a"), rec("b")])
        assert "hooks" not in cfg
        assert set(cfg["groups"]) == {"main"}
        assert cfg["groups"]["main"] == {"swap": True, "exclusive": True,
                                         "members": ["a", "b"]}

    def test_preload_renders_the_startup_hook(self):
        cfg = render([rec("a", preload=True), rec("b")])
        assert cfg["hooks"] == {"on_startup": {"preload": ["a"]}}
        # still in the swap group: a request for b evicts it, by design
        assert cfg["groups"]["main"]["members"] == ["a", "b"]

    def test_a_disabled_model_is_not_preloaded(self):
        cfg = render([rec("a", preload=True, enabled=False), rec("b")])
        assert "hooks" not in cfg
        assert cfg["groups"]["main"]["members"] == ["b"]

    def test_ttl_zero_survives_as_never_unload(self):
        """llama-swap reads ttl 0 as 'never unload'. It must not be turned
        back into the 900 default on the way through."""
        cfg = render([rec("a", ttl=0)])
        assert cfg["models"]["a"]["ttl"] == 0

    def test_two_preloads_in_the_swap_group_are_refused(self):
        with pytest.raises(HTTPException) as exc:
            gw.check_preload_count([rec("a", preload=True), rec("b", preload=True)])
        assert exc.value.status_code == 400
        assert "a, b" in str(exc.value.detail)

    def test_a_persistent_model_does_not_count_against_the_limit(self):
        gw.check_preload_count([rec("small", persistent=True),
                                rec("big", preload=True)])


class TestPersistent:
    def test_persistent_gets_its_own_unevictable_group(self):
        cfg = render([rec("small", persistent=True), rec("big"), rec("other")])
        assert cfg["groups"]["warm"] == {"swap": False, "exclusive": False,
                                         "persistent": True,
                                         "members": ["small"]}
        assert cfg["groups"]["main"] == {"swap": True, "exclusive": True,
                                         "members": ["big", "other"]}

    def test_persistent_implies_preload(self):
        """Resident forever and cold until asked is a contradiction."""
        cfg = render([rec("small", persistent=True), rec("big")])
        assert cfg["hooks"]["on_startup"]["preload"] == ["small"]

    def test_persistent_beside_a_preload_lists_both(self):
        cfg = render([rec("small", persistent=True), rec("big", preload=True)])
        assert sorted(cfg["hooks"]["on_startup"]["preload"]) == ["big", "small"]

    def test_only_persistent_models_means_no_main_group(self):
        cfg = render([rec("small", persistent=True)])
        assert set(cfg["groups"]) == {"warm"}

    def test_every_enabled_model_is_in_exactly_one_group(self):
        models = [rec("small", persistent=True), rec("a"), rec("b", preload=True),
                  rec("off", enabled=False)]
        cfg = render(models)
        members = [m for g in cfg["groups"].values() for m in g["members"]]
        assert sorted(members) == ["a", "b", "small"]
        assert len(members) == len(set(members))
