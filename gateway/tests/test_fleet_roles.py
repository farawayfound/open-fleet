"""A fleet role is a policy, not a name any box serves.

`fast` used to be a row in each box's models.json meaning a different model
on every box, so a request for it got whichever box the scorer liked. Now
FLEET_ROLES says which models can play each role, best first, and each box
offers the best one it holds -- these tests pin that per-box choice, that the
scorer ranks the result like any other candidates, that roles are listed and
callable, and that a box's leftover row of the same name is no longer a
conflict.
"""
from __future__ import annotations

import pytest

import app as gw


@pytest.fixture(autouse=True)
def _routes_cache_restored():
    """These tests install routing tables by hand with t=9e9, which would
    otherwise outlive them: a later test's fleet_model_list() would find
    these hosts and list roles for them."""
    saved = {k: (dict(v) if isinstance(v, dict) else set(v) if isinstance(v, set) else v)
             for k, v in gw._routes_cache.items()}
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)


def _routes(cands, alias, meta, running=None):
    gw._routes_cache.update(t=9e9, map={}, cands=cands, alias=alias, meta=meta,
                            cap={}, running=running or {},
                            reachable={h for hs in cands.values() for h in hs if h})


def _ident(*pairs):
    return {(h, m): m for h, m in pairs}


# ---- the ladders ------------------------------------------------------------

def test_ladders_name_canonical_ids_and_roles_are_not_models():
    models = ({f for r in gw.PUBLIC_MODELS_SEED for f in r["fleet_ids"]}
              | {r["public_id"] for r in gw.PUBLIC_MODELS_SEED}
              | set(gw.FLEET_MODEL_NAMES) | set(gw.SPELLING_TO_CANONICAL))
    for role, ladder in gw.FLEET_ROLES.items():
        assert gw.SAFE_ID.match(role)
        # resolve_targets() answers the catalogue first, so a role that was
        # also a public id would never reach the policy.
        assert role not in models, role
        assert len(set(ladder)) == len(ladder), role
        for m in ladder:
            assert gw.SAFE_ID.match(m), (role, m)
            assert m not in gw.SPELLING_TO_CANONICAL, (role, m, "use the canonical id")


# ---- each box's best model for the role ---------------------------------------

class TestRolePairs:
    def test_each_box_offers_the_best_model_it_can_hold(self):
        # gpu-laptop-1 holds both the 35B MoE and the 4B: its own `fast` row said
        # 4B, the policy says the MoE. gpu-desktop-1 holds only the 4B. server-1 is a
        # CPU box that holds nothing "in memory" -- it still plays the role
        # with the first ladder entry it serves, as its owner had chosen.
        _routes(
            {"qwen3.6-35b": ["gpu-laptop-1", "server-1"], "qwen3.5-4b": ["gpu-laptop-1", "gpu-desktop-1"],
             "fast": ["gpu-laptop-1", "gpu-desktop-1", "server-1"]},
            {**_ident(("gpu-laptop-1", "qwen3.6-35b"), ("server-1", "qwen3.6-35b"),
                      ("gpu-laptop-1", "qwen3.5-4b"), ("gpu-desktop-1", "qwen3.5-4b")),
             ("gpu-laptop-1", "fast"): "qwen3.5-4b", ("gpu-desktop-1", "fast"): "qwen3.5-4b",
             ("server-1", "fast"): "qwen3.6-35b"},
            {("gpu-laptop-1", "qwen3.6-35b"): {"fit": "vram"}, ("gpu-laptop-1", "qwen3.5-4b"): {"fit": "vram"},
             ("server-1", "qwen3.6-35b"): {"fit": "cpu"}, ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
        )
        assert dict(gw.role_pairs("fast")) == {
            "gpu-laptop-1": "qwen3.6-35b", "gpu-desktop-1": "qwen3.5-4b", "server-1": "qwen3.6-35b"}

    def test_a_model_the_box_would_spill_yields_to_one_it_holds(self):
        _routes(
            {"qwen3.6-35b": ["gpu-desktop-1"], "qwen3.5-4b": ["gpu-desktop-1"]},
            _ident(("gpu-desktop-1", "qwen3.6-35b"), ("gpu-desktop-1", "qwen3.5-4b")),
            {("gpu-desktop-1", "qwen3.6-35b"): {"fit": "spill"}, ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
        )
        assert gw.role_pairs("fast") == [("gpu-desktop-1", "qwen3.5-4b")]

    def test_one_pair_per_box_never_two(self):
        _routes(
            {"qwen3.8-27b": ["apu-box-1"], "qwen3.6-35b": ["apu-box-1"], "qwen3.5-4b": ["apu-box-1"]},
            _ident(("apu-box-1", "qwen3.8-27b"), ("apu-box-1", "qwen3.6-35b"), ("apu-box-1", "qwen3.5-4b")),
            {("apu-box-1", m): {"fit": "vram"} for m in ("qwen3.8-27b", "qwen3.6-35b", "qwen3.5-4b")},
        )
        assert gw.role_pairs("default") == [("apu-box-1", "qwen3.8-27b")]
        assert gw.role_pairs("fast") == [("apu-box-1", "qwen3.6-35b")]

    def test_a_peer_that_only_knows_an_old_spelling_is_addressed_by_it(self):
        # mac-laptop-2 before its start-up rename: the weights are registered as
        # empero-ai-qwen3.8-9b-distill and the peer reports that as canonical.
        # It still plays `small`, and the proxy must send the name it answers to.
        _routes(
            {"empero-ai-qwen3.8-9b-distill": ["mac-laptop-2"]},
            _ident(("mac-laptop-2", "empero-ai-qwen3.8-9b-distill")),
            {("mac-laptop-2", "empero-ai-qwen3.8-9b-distill"): {"fit": "unified"}},
        )
        assert gw.role_pairs("small") == [("mac-laptop-2", "empero-ai-qwen3.8-9b-distill")]

    def test_the_canonical_name_wins_over_an_alias_on_the_same_box(self):
        _routes(
            {"qwen3.8-9b-distill": ["apu-tablet-2"], "qwen3.8-9B": ["apu-tablet-2"], "small": ["apu-tablet-2"]},
            {("apu-tablet-2", "qwen3.8-9b-distill"): "qwen3.8-9b-distill",
             ("apu-tablet-2", "qwen3.8-9B"): "qwen3.8-9b-distill",
             ("apu-tablet-2", "small"): "qwen3.8-9b-distill"},
            {("apu-tablet-2", "qwen3.8-9b-distill"): {"fit": "vram"}},
        )
        assert gw.role_pairs("small") == [("apu-tablet-2", "qwen3.8-9b-distill")]

    def test_the_name_that_is_loaded_beats_the_canonical_spelling(self):
        # Two RECORDS for one file on one box -- the start-up rename stepped
        # aside because the canonical id was already a variant's -- and only
        # the old spelling is resident. Sending the canonical name would pay
        # a reload of the same weights.
        _routes(
            {"Qwen3.8-9B": ["mac-laptop-1"], "qwen3.8-9b-distill": ["mac-laptop-1"]},
            _ident(("mac-laptop-1", "Qwen3.8-9B"), ("mac-laptop-1", "qwen3.8-9b-distill")),
            {("mac-laptop-1", "Qwen3.8-9B"): {"fit": "unified"},
             ("mac-laptop-1", "qwen3.8-9b-distill"): {"fit": "unified"}},
            running={"mac-laptop-1": {"Qwen3.8-9B"}},
        )
        assert gw.role_pairs("small") == [("mac-laptop-1", "Qwen3.8-9B")]
        # Neither loaded: the canonical spelling wins, whichever came first.
        gw._routes_cache["running"] = {}
        assert gw.role_pairs("small") == [("mac-laptop-1", "qwen3.8-9b-distill")]

    def test_the_local_host_is_blank_like_everywhere_else(self):
        _routes({"qwen3.8-27b": [""]}, _ident(("", "qwen3.8-27b")),
                {("", "qwen3.8-27b"): {"fit": "vram"}})
        assert gw.role_pairs("default") == [("", "qwen3.8-27b")]

    def test_nothing_to_offer(self):
        _routes({"glm-4.7-flash": ["gpu-laptop-1"]}, _ident(("gpu-laptop-1", "glm-4.7-flash")), {})
        assert gw.role_pairs("coder") == []
        assert gw.role_pairs("not-a-role") == []


# ---- resolved and ranked like any other request --------------------------------

@pytest.mark.asyncio
async def test_resolve_targets_answers_a_role_with_each_boxes_best_model():
    _routes(
        {"qwen3.6-35b": ["gpu-laptop-1", "server-1"], "qwen3.5-4b": ["gpu-laptop-1", "gpu-desktop-1"]},
        _ident(("gpu-laptop-1", "qwen3.6-35b"), ("server-1", "qwen3.6-35b"),
               ("gpu-laptop-1", "qwen3.5-4b"), ("gpu-desktop-1", "qwen3.5-4b")),
        {("gpu-laptop-1", "qwen3.6-35b"): {"fit": "vram"}, ("gpu-laptop-1", "qwen3.5-4b"): {"fit": "vram"},
         ("server-1", "qwen3.6-35b"): {"fit": "cpu"}, ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
    )
    got = await gw.resolve_targets("fast")
    assert set(got) == {("gpu-laptop-1", "qwen3.6-35b"), ("gpu-desktop-1", "qwen3.5-4b"),
                        ("server-1", "qwen3.6-35b")}
    assert len(got) == 3


@pytest.mark.asyncio
async def test_among_boxes_that_hold_it_the_ladder_decides_not_residency():
    """`deep` used to go to a warm 27B on mac-laptop-1 while the 120B sat idle on
    apu-box-1: the scorer knew tiers and wall time, not the ladder. With the
    role named, the box offering the model higher on the ladder wins."""
    _routes(
        {"qwen3.8-27b": ["mac-laptop-1"], "nemotron3-super-120b": ["apu-box-1"]},
        _ident(("mac-laptop-1", "qwen3.8-27b"), ("apu-box-1", "nemotron3-super-120b")),
        {("mac-laptop-1", "qwen3.8-27b"): {"fit": "unified"},
         ("apu-box-1", "nemotron3-super-120b"): {"fit": "vram"}},
        running={"mac-laptop-1": {"qwen3.8-27b"}},
    )
    pairs = [("mac-laptop-1", "qwen3.8-27b"), ("apu-box-1", "nemotron3-super-120b")]
    assert (await gw._score_host_model_pairs(pairs, fleet_role="deep"))[0] == (
        "apu-box-1", "nemotron3-super-120b")
    # ...and for `quality`, whose ladder leads with the 27B, the other way.
    assert (await gw._score_host_model_pairs(pairs, fleet_role="quality"))[0] == (
        "mac-laptop-1", "qwen3.8-27b")
    assert (await gw.resolve_targets("deep"))[0] == ("apu-box-1", "nemotron3-super-120b")


@pytest.mark.asyncio
async def test_a_box_that_cannot_hold_the_better_model_sits_behind_one_that_holds_a_lesser():
    # The ladder only orders boxes that HOLD their model; a window too small
    # for the request (tier 5) puts the ladder's favourite behind the rest.
    _routes(
        {"nemotron3-super-120b": ["apu-box-1"], "qwen3.8-27b": ["mac-laptop-1"]},
        _ident(("apu-box-1", "nemotron3-super-120b"), ("mac-laptop-1", "qwen3.8-27b")),
        {("apu-box-1", "nemotron3-super-120b"): {"fit": "vram"},
         ("mac-laptop-1", "qwen3.8-27b"): {"fit": "unified"}},
    )
    gw._routes_cache["ctx"] = {("apu-box-1", "nemotron3-super-120b"): 4096}
    pairs = [("apu-box-1", "nemotron3-super-120b"), ("mac-laptop-1", "qwen3.8-27b")]
    got = await gw._score_host_model_pairs(pairs, fleet_role="deep", need_ctx=16000)
    assert got[0] == ("mac-laptop-1", "qwen3.8-27b")


def test_role_index_reads_through_aliases_and_spellings():
    gw._routes_cache.update(alias={("mac-laptop-2", "empero-ai-qwen3.8-9b-distill"): "empero-ai-qwen3.8-9b-distill",
                                   ("apu-tablet-2", "small"): "qwen3.8-9b-distill"})
    assert gw.role_index("small", "mac-laptop-2", "empero-ai-qwen3.8-9b-distill") == 0
    assert gw.role_index("small", "apu-tablet-2", "small") == 0
    assert gw.role_index("small", "gpu-desktop-1", "qwen3.5-4b") == 1
    assert gw.role_index("small", "apu-box-1", "nemotron3-super-120b") == len(gw.FLEET_ROLES["small"])


@pytest.mark.asyncio
async def test_a_role_nobody_can_play_resolves_to_nothing():
    _routes({"glm-4.7-flash": ["gpu-laptop-1"]}, _ident(("gpu-laptop-1", "glm-4.7-flash")), {})
    assert await gw.resolve_targets("coder") == []


@pytest.mark.asyncio
async def test_v1_models_lists_a_role_only_while_something_plays_it(monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    gw.save_models([])
    _routes({"qwen3.8-27b": ["apu-tablet-2"]}, _ident(("apu-tablet-2", "qwen3.8-27b")),
            {("apu-tablet-2", "qwen3.8-27b"): {"fit": "vram"}})
    gw._routes_cache["map"] = {"qwen3.8-27b": "apu-tablet-2"}
    owner = {m["id"]: m["owned_by"] for m in await gw.fleet_model_list()}
    assert owner["qwen3.8-27b"] == "apu-tablet-2"
    for role in ("default", "quality", "deep", "vision", "qwen"):
        assert owner[role] == "fleet-role", role
    for role in ("coder", "gemma", "nemotron", "small", "fast"):
        assert role not in owner, role


@pytest.mark.asyncio
async def test_a_batch_for_a_role_fans_out_over_each_boxes_best_model(monkeypatch):
    """/v1/batches builds its own target list; it has to resolve a role the
    same way the live proxy does, and every target must carry the id the box
    really serves -- a worker posting the literal word `fast` to five boxes
    would get five different models and record all of them as `fast`."""
    _routes(
        {"qwen3.6-35b": ["gpu-laptop-1", "server-1"], "qwen3.5-4b": ["gpu-laptop-1", "gpu-desktop-1"],
         "fast": ["gpu-laptop-1", "gpu-desktop-1", "server-1"]},
        {**_ident(("gpu-laptop-1", "qwen3.6-35b"), ("server-1", "qwen3.6-35b"),
                  ("gpu-laptop-1", "qwen3.5-4b"), ("gpu-desktop-1", "qwen3.5-4b")),
         ("gpu-laptop-1", "fast"): "qwen3.5-4b", ("gpu-desktop-1", "fast"): "qwen3.5-4b",
         ("server-1", "fast"): "qwen3.6-35b"},
        {("gpu-laptop-1", "qwen3.6-35b"): {"fit": "vram"}, ("gpu-laptop-1", "qwen3.5-4b"): {"fit": "vram"},
         ("server-1", "qwen3.6-35b"): {"fit": "cpu"}, ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
    )

    async def keep(force: bool = False):
        return gw._routes_cache["map"]
    monkeypatch.setattr(gw, "model_routes", keep)   # _batch_targets forces a rebuild

    targets = await gw._batch_targets(["fast"])
    assert {(t["host"], t["model"]) for t in targets} == {
        ("gpu-laptop-1", "qwen3.6-35b"), ("gpu-desktop-1", "qwen3.5-4b"), ("server-1", "qwen3.6-35b")}
    assert all(t["model"] != "fast" for t in targets)
    # A plain id is unchanged by the same path.
    plain = await gw._batch_targets(["qwen3.5-4b"])
    assert {(t["host"], t["model"]) for t in plain} == {
        ("gpu-laptop-1", "qwen3.5-4b"), ("gpu-desktop-1", "qwen3.5-4b")}


def test_a_catalogue_row_may_not_take_a_role_name(client, admin_headers):
    """resolve_targets() answers the catalogue before a role, so a public row
    named `fast` -- or claiming `fast` as a fleet id -- would silently take
    over live routing for the word. Refused at the one place a person can
    create it."""
    r = client.put("/admin/api/public/models/fast", headers=admin_headers,
                   json={"fleet_ids": ["qwen3.5-4b"], "name": "Fast"})
    assert r.status_code == 400 and "fleet role" in r.json()["detail"]
    r = client.put("/admin/api/public/models/some-new-row", headers=admin_headers,
                   json={"fleet_ids": ["small"], "name": "Small"})
    assert r.status_code == 400 and "fleet role" in r.json()["detail"]


def test_a_role_name_is_no_longer_an_alias_conflict():
    _routes({"fast": ["apu-box-1", "mac-desktop-1"]},
            {("apu-box-1", "fast"): "qwen3.6-35b", ("mac-desktop-1", "fast"): "qwen3.5:9b"}, {})
    assert gw.alias_conflicts() == []


def test_roles_endpoint_shows_picks_order_and_leftover_rows(client, admin_headers):
    _routes(
        {"qwen3.6-35b": ["gpu-laptop-1"], "qwen3.5-4b": ["gpu-desktop-1"], "fast": ["gpu-desktop-1"]},
        {**_ident(("gpu-laptop-1", "qwen3.6-35b"), ("gpu-desktop-1", "qwen3.5-4b")),
         ("gpu-desktop-1", "fast"): "qwen3.5-4b"},
        {("gpu-laptop-1", "qwen3.6-35b"): {"fit": "vram"}, ("gpu-desktop-1", "qwen3.5-4b"): {"fit": "vram"}},
    )
    r = client.get("/admin/api/roles", headers=admin_headers)
    assert r.status_code == 200
    roles = r.json()["roles"]
    assert set(roles) == set(gw.FLEET_ROLES)
    fast = roles["fast"]
    assert fast["ladder"] == list(gw.FLEET_ROLES["fast"])
    assert fast["picks"] == {"gpu-laptop-1": "qwen3.6-35b", "gpu-desktop-1": "qwen3.5-4b"}
    assert {(o["host"], o["model"]) for o in fast["order"]} == {
        ("gpu-laptop-1", "qwen3.6-35b"), ("gpu-desktop-1", "qwen3.5-4b")}
    assert fast["local_rows"] == {"gpu-desktop-1": "qwen3.5-4b"}
    # coder falls back to the 35B MoE as a last resort, so it IS playable
    # here; gemma has nothing at all.
    assert roles["coder"]["picks"] == {"gpu-laptop-1": "qwen3.6-35b"}
    assert roles["gemma"]["order"] == [] and roles["gemma"]["picks"] == {}
