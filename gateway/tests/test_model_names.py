"""One id per model, fleet-wide.

Candidates for a request are grouped by name, so the same weights under two
names on two boxes are never each other's alternatives. FLEET_MODEL_NAMES is
the fleet's answer: the canonical id per model and every spelling it has
been seen under. These tests pin what is done with it -- the start-up rename
on a llama.cpp box, the alias an Ollama box answers to and maps back to its
tag, the one-entry-per-model listing -- and that the map and the public
catalogue never disagree about what is one model.
"""
from __future__ import annotations

import time

import pytest

import app as gw

TAG = "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"
CANON = "qwen3.8-9b-distill"


def _rec(mid: str, aliases=(), **kw) -> dict:
    return dict(gw.DEFAULT_MODEL_RECORD, id=mid, path="/w/" + mid + ".gguf",
                aliases=list(aliases), **kw)


@pytest.fixture(autouse=True)
def _routes_cache_restored():
    """The listing tests below install a routing table by hand (t=9e9); put
    back whatever was there so it does not leak into the next test file."""
    saved = {k: (dict(v) if isinstance(v, dict) else set(v) if isinstance(v, set) else v)
             for k, v in gw._routes_cache.items()}
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)


# ---- the map itself ---------------------------------------------------------

def test_every_name_in_the_map_is_claimed_by_one_catalogue_row():
    """The catalogue (resolve_targets) and the routing layer (model_routes)
    must agree on what is one model, or a request by public id could reach a
    box a request by fleet id cannot."""
    for canon, spellings in gw.FLEET_MODEL_NAMES.items():
        owners = [r["public_id"] for r in gw.PUBLIC_MODELS_SEED
                  if canon in r["fleet_ids"]]
        assert len(owners) == 1, (canon, owners)
        row = next(r for r in gw.PUBLIC_MODELS_SEED if r["public_id"] == owners[0])
        for s in spellings:
            assert s in row["fleet_ids"], (s, "is not claimed by", owners[0])
        assert gw.SAFE_ID.match(canon), canon


def test_a_spelling_belongs_to_exactly_one_model():
    seen: dict[str, str] = {}
    for canon, spellings in gw.FLEET_MODEL_NAMES.items():
        assert canon not in gw.SPELLING_TO_CANONICAL, canon
        for s in spellings:
            assert s != canon
            assert seen.setdefault(s, canon) == canon, s


def test_the_public_row_claims_the_canonical_id_itself():
    """Without this, the LM Studio sync on mac-laptop-2 dodges the canonical id as
    a collision with the public id and derives a fourth spelling."""
    for canon in gw.FLEET_MODEL_NAMES:
        row = next((r for r in gw.PUBLIC_MODELS_SEED if canon in r["fleet_ids"]), None)
        assert row is not None
        if row["public_id"] == canon:
            assert canon not in gw._public_ids()


# ---- the rename (llama.cpp box) ---------------------------------------------

class TestConvergeModelNames:
    def test_zephyrus_as_found(self):
        # Registered as qwen3.8-9B by hand, with the canonical id later added
        # as an alias. The two swap places; everything else travels with it.
        (r,), changes = gw.converge_model_names(
            [_rec("qwen3.8-9B", [CANON], preload=True, ctx=65536)])
        assert r["id"] == CANON
        assert r["aliases"] == ["qwen3.8-9B"]
        assert r["preload"] is True and r["ctx"] == 65536
        assert changes == [{"id": "qwen3.8-9B", "to": CANON}]
        gw.check_name_collisions([r])

    def test_flowstate_is_already_right(self):
        models = [_rec(CANON, ["qwen3.8-9B", "small", "fast"])]
        out, changes = gw.converge_model_names(models)
        assert out == models and changes == []

    def test_a_record_with_nothing_to_say_passes_through_untouched(self):
        models = [_rec("qwen3.6-35b", ["qwen"]), _rec("Qwopus3.6-A3B", ["qwopus"])]
        out, changes = gw.converge_model_names(models)
        assert out[0] is models[0]
        assert out[1]["id"] == "qwopus3.6-35b-coder"
        assert out[1]["aliases"] == ["qwopus", "Qwopus3.6-A3B"]
        assert changes == [{"id": "Qwopus3.6-A3B", "to": "qwopus3.6-35b-coder"}]

    def test_idempotent(self):
        once, c1 = gw.converge_model_names([_rec("qwen3.8-9B"), _rec("Qwopus3.6-A3B")])
        twice, c2 = gw.converge_model_names(once)
        assert len(c1) == 2 and c2 == [] and twice == once

    def test_a_variant_that_owns_the_canonical_id_is_left_alone(self):
        # Two records for one model is a variant the owner built on purpose.
        models = [_rec(CANON), _rec("Qwen3.8-9B", ctx=4096)]
        out, changes = gw.converge_model_names(models)
        assert [r["id"] for r in out] == [CANON, "Qwen3.8-9B"]
        (c,) = changes
        assert c["id"] == "Qwen3.8-9B" and CANON in c["skipped"]
        gw.check_name_collisions(out)

    def test_an_alias_on_another_record_blocks_the_rename(self):
        models = [_rec("qwen3.5-4b", ["gemma-4-26b"]), _rec("gemma4:26b")]
        out, changes = gw.converge_model_names(models)
        assert out[1]["id"] == "gemma4:26b"
        assert changes[0]["skipped"] and "'qwen3.5-4b'" in changes[0]["skipped"]

    def test_a_disabled_record_is_neither_renamed_nor_overwritten(self):
        out, changes = gw.converge_model_names([_rec("qwen3.8-9B", enabled=False)])
        assert out[0]["id"] == "qwen3.8-9B" and changes == []
        # ...and a disabled record holding the canonical id still owns it:
        # a rename must never leave two records with one id.
        out, changes = gw.converge_model_names(
            [_rec(CANON, enabled=False), _rec("qwen3.8-9B")])
        assert out[1]["id"] == "qwen3.8-9B" and changes[0]["skipped"]


class TestStartupPass:
    @pytest.fixture(autouse=True)
    def _registry(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
        self.calls: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: (self.calls.append((a, u)) or (0, "ok")))
        gw._converged.update(at="", changes=[], restart_rc=0)

    def test_renames_writes_and_restarts_once(self):
        gw.save_models([_rec("qwen3.8-9B", [CANON], preload=True), _rec("qwen3.6-35b")])
        changes = gw.apply_model_name_convergence()
        assert changes == [{"id": "qwen3.8-9B", "to": CANON}]
        by_id = {r["id"]: r for r in gw.load_models()}
        assert set(by_id) == {CANON, "qwen3.6-35b"}
        assert by_id[CANON]["aliases"] == ["qwen3.8-9B"]
        assert by_id[CANON]["preload"] is True
        assert self.calls == [("restart", "llama-swap")]
        text = gw.SWAP_CONFIG.read_text()
        assert "  " + CANON + ":" in text and "- qwen3.8-9B" in text
        assert gw._converged["changes"] == changes
        # A second start finds nothing to do.
        assert gw.apply_model_name_convergence() == []
        assert self.calls == [("restart", "llama-swap")]

    def test_nothing_to_do_touches_nothing(self):
        gw.save_models([_rec(CANON, ["qwen3.8-9B"])])
        before = gw.MODELS_JSON.read_text()
        assert gw.apply_model_name_convergence() == []
        assert gw.MODELS_JSON.read_text() == before
        assert self.calls == [] and not gw.SWAP_CONFIG.exists()

    def test_an_ollama_box_is_left_alone(self, monkeypatch):
        # mac-desktop-1: a Library download registered a llama.cpp record on a box
        # whose engine is Ollama. Renaming cannot make it servable.
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        gw.save_models([_rec("Qwen3.8-9B")])
        assert gw.apply_model_name_convergence() == []
        assert gw.load_models()[0]["id"] == "Qwen3.8-9B"
        assert self.calls == []

    def test_a_skip_is_recorded_for_the_page_without_a_restart(self):
        gw.save_models([_rec(CANON), _rec("Qwen3.8-9B")])
        changes = gw.apply_model_name_convergence()
        assert changes[0]["skipped"]
        assert self.calls == []
        assert gw._converged["changes"] == changes


# ---- the alias (Ollama box) --------------------------------------------------

class TestOllamaAliases:
    @pytest.fixture(autouse=True)
    def _ollama(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        gw.save_models([])
        gw._upstream_cache.update(t=time.time(), ids={
            TAG, "gemma4:26b", "qwen3.5:4b", "nemotron-mini:4b"})
        yield
        gw._upstream_cache.update(t=0.0, ids=set())

    @pytest.mark.asyncio
    async def test_the_canonical_id_is_answered_to_and_mapped_back(self):
        assert await gw.upstream_alias_pairs() == {
            CANON: TAG, "gemma-4-26b": "gemma4:26b", "qwen3.5-4b": "qwen3.5:4b"}
        ids = await gw.served_model_ids()
        assert {CANON, TAG, "gemma-4-26b", "nemotron-mini:4b"} <= ids
        can = await gw.served_canonical_map()
        assert can[TAG] == CANON == can[CANON]
        assert can["nemotron-mini:4b"] == "nemotron-mini:4b"

    @pytest.mark.asyncio
    async def test_a_canonical_id_that_is_already_a_tag_is_not_aliased(self):
        gw._upstream_cache["ids"].add("gemma-4-26b")
        assert "gemma-4-26b" not in await gw.upstream_alias_pairs()

    @pytest.mark.asyncio
    async def test_a_canonical_id_that_is_a_local_record_is_not_aliased(self):
        gw.save_models([_rec(CANON)])
        assert CANON not in await gw.upstream_alias_pairs()

    @pytest.mark.asyncio
    async def test_not_on_a_llama_cpp_box(self, monkeypatch):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
        assert await gw.upstream_alias_pairs() == {}
        assert CANON not in await gw.served_model_ids()

    def test_the_hub_is_told(self, client, admin_headers):
        r = client.get("/admin/api/served-models", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert CANON in body["models"] and TAG in body["models"]
        assert body["canonical"][TAG] == CANON
        assert body["canonical"][CANON] == CANON


# ---- one entry per model ----------------------------------------------------

@pytest.mark.asyncio
async def test_v1_models_lists_each_model_once(monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    gw.save_models([])
    gw._routes_cache.update(
        t=9e9, cands={}, meta={},
        map={CANON: "apu-tablet-2", "qwen3.8-9B": "apu-tablet-2", "fast": "apu-tablet-2",
             TAG: "mini-pc-1", "gemma-4-26b": "gpu-laptop-1", "gemma": "gpu-laptop-1"},
        alias={("apu-tablet-2", CANON): CANON, ("apu-tablet-2", "qwen3.8-9B"): CANON,
               ("apu-tablet-2", "fast"): CANON, ("mini-pc-1", TAG): CANON,
               ("gpu-laptop-1", "gemma-4-26b"): "gemma-4-26b",
               ("gpu-laptop-1", "gemma"): "gemma-4-26b"},
    )
    listed = await gw.fleet_model_list()
    assert [m["id"] for m in listed] == ["gemma-4-26b", CANON]
    assert {m["owned_by"] for m in listed} == {"gpu-laptop-1", "apu-tablet-2"}


@pytest.mark.asyncio
async def test_an_old_peer_that_reports_no_canonical_is_listed_as_is(monkeypatch, tmp_path):
    # No `canonical` from a peer means "cannot check", never "hide it".
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    gw.save_models([])
    gw._routes_cache.update(t=9e9, cands={}, meta={},
                            map={"speedy": "oldbox", "m": "oldbox", "fast": "oldbox"},
                            alias={})
    # ...except a fleet-role word: the peer's `fast` row is not what a request
    # for `fast` would be routed by, so it is not listed on the peer's say-so.
    assert [m["id"] for m in await gw.fleet_model_list()] == ["m", "speedy"]


def test_routes_marks_a_local_alias_and_names_its_model(client, admin_headers,
                                                        monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    gw.save_models([_rec("qwen3.6-35b", ["fast"])])
    r = client.get("/admin/api/routes", headers=admin_headers)
    assert r.status_code == 200
    rows = {m["model"]: m for m in r.json()["models"]}
    assert rows["qwen3.6-35b"]["alias"] is False
    assert rows["qwen3.6-35b"]["canonical"] == "qwen3.6-35b"
    assert rows["fast"]["alias"] is True
    assert rows["fast"]["canonical"] == "qwen3.6-35b"
