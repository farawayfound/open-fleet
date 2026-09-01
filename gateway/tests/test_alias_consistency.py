"""A role alias has to mean one model across the fleet, or say that it does not.

The scorer assumes the candidates for a name are interchangeable. That holds
for a real model id and is only an assumption for an alias: `fast`, `triage`
and `deep` are rows in each box's own models.json, and two boxes can point the
same word at different weights without anything noticing. These tests pin the
detection, and - just as importantly - pin that a peer too old to answer the
question is reported as unknown rather than counted as agreement.
"""
from __future__ import annotations

import pytest

import app as gw


@pytest.fixture(autouse=True)
def _routes_cache_restored():
    """Tables installed by hand below (t=9e9) must not outlive the test: a
    later file's fleet_model_list() would otherwise find these hosts."""
    saved = {k: (dict(v) if isinstance(v, dict) else set(v) if isinstance(v, set) else v)
             for k, v in gw._routes_cache.items()}
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(saved)


def _routes(cands, alias):
    """Install a routing table by hand. alias_conflicts() reads the cache the
    same way the scorer does, so this is the whole input."""
    gw._routes_cache.update(t=9e9, cands=cands, alias=alias)


# The names below are deliberately NOT fleet roles (`fast`, `deep`, `triage`
# are answered by FLEET_ROLES policy before any box's row is consulted, and
# are skipped here for that reason -- see test_fleet_roles.py). A box-local
# alias that is not a fleet role is still a per-box fact the fleet can
# disagree about.

def test_agreement_is_not_a_conflict():
    _routes(
        {"speedy": ["", "mac-desktop-1"]},
        {("", "speedy"): "qwen3.6-35b", ("mac-desktop-1", "speedy"): "qwen3.6-35b"},
    )
    assert gw.alias_conflicts() == []


def test_one_host_is_never_a_conflict():
    # Nothing to disagree with. The common case, and it must stay cheap.
    _routes({"heavy": ["apu-box-1"]}, {("apu-box-1", "heavy"): "nemotron3-super-120b"})
    assert gw.alias_conflicts() == []


def test_two_hosts_two_meanings():
    _routes(
        {"speedy": ["apu-box-1", "mac-desktop-1"]},
        {("apu-box-1", "speedy"): "qwen3.6-35b", ("mac-desktop-1", "speedy"): "qwen3.5:9b"},
    )
    (c,) = gw.alias_conflicts()
    assert c["name"] == "speedy"
    assert c["targets"] == {"qwen3.5:9b": ["mac-desktop-1"],
                            "qwen3.6-35b": ["apu-box-1"]}
    assert c["unknown"] == []
    # "speedy" is not itself a model id here, so this is a pure alias clash.
    assert c["is_model_id"] is False


def test_alias_shadowing_a_real_model_id_is_flagged_as_such():
    # The nastier shape read the other way round: one box serves the real
    # model, another points the same word somewhere else.
    _routes(
        {"qwen3.6-35b": ["apu-box-1", "gpu-laptop-1"]},
        {("apu-box-1", "qwen3.6-35b"): "qwen3.6-35b",
         ("gpu-laptop-1", "qwen3.6-35b"): "qwen3.5-4b"},
    )
    (c,) = gw.alias_conflicts()
    assert c["is_model_id"] is True
    assert c["targets"]["qwen3.6-35b"] == ["apu-box-1"]


def test_the_local_host_is_named_not_blank():
    # '' means this host inside the routing table; a report that said '' would
    # be unreadable on the dashboard.
    _routes(
        {"sorter": ["", "gpu-desktop-1"]},
        {("", "sorter"): "a", ("gpu-desktop-1", "sorter"): "b"},
    )
    (c,) = gw.alias_conflicts()
    assert c["targets"] == {"a": [gw.HOST_NAME], "b": ["gpu-desktop-1"]}
    assert all(h for hs in c["targets"].values() for h in hs)


def test_an_old_peer_is_unknown_not_agreement():
    # A gateway that predates the `canonical` field reports nothing. Counting
    # that as "agrees" would make the check answer "all clear" on exactly the
    # fleet that cannot be checked.
    _routes(
        {"speedy": ["apu-box-1", "gpu-laptop-2"]},
        {("apu-box-1", "speedy"): "qwen3.6-35b"},          # gpu-laptop-2: silent
    )
    assert gw.alias_conflicts() == []                    # one known target only

    _routes(
        {"speedy": ["apu-box-1", "gpu-laptop-2", "mac-desktop-1"]},
        {("apu-box-1", "speedy"): "qwen3.6-35b",
         ("mac-desktop-1", "speedy"): "qwen3.5:9b"},           # gpu-laptop-2: still silent
    )
    (c,) = gw.alias_conflicts()
    assert c["unknown"] == ["gpu-laptop-2"]


def test_local_alias_map_resolves_aliases_to_their_id(monkeypatch):
    monkeypatch.setattr(gw, "load_models", lambda: [
        {"id": "qwen3.6-35b", "aliases": ["fast", "triage"], "enabled": True},
        {"id": "nemotron3-super-120b", "aliases": ["deep"], "enabled": True},
        {"id": "retired-model", "aliases": ["old"], "enabled": False},
        {"id": "", "aliases": ["junk"], "enabled": True},
    ])
    m = gw.local_alias_map()
    assert m["fast"] == "qwen3.6-35b"
    assert m["triage"] == "qwen3.6-35b"
    assert m["qwen3.6-35b"] == "qwen3.6-35b"    # an id maps to itself
    assert m["deep"] == "nemotron3-super-120b"
    assert "old" not in m and "retired-model" not in m   # disabled
    assert "junk" not in m                                # no id, no entry


class TestSplitModels:
    """One model under two names: the boxes never become each other's
    alternatives, so the better one is invisible for that request."""

    def _routes(self, cands, meta, alias=None):
        gw._routes_cache.update(t=9e9, cands=cands, meta=meta,
                                alias=alias or {})

    def test_the_real_case_from_this_fleet(self):
        # gpu-laptop-1 (always-on, meant to hold it warm) served the 9B distill as
        # qwen3.8-9B; apu-tablet-2 served the same file as qwen3.8-9b-distill,
        # which is the id the hub publishes. gpu-laptop-1 was never a candidate.
        src = {"repo": "empero-ai/Qwen3.8-9B-Distill-GGUF",
               "file": "Qwen3.8-9B-Q4_K_M.gguf"}
        self._routes(
            {"qwen3.8-9B": ["gpu-laptop-1"], "qwen3.8-9b-distill": ["apu-tablet-2"]},
            {("gpu-laptop-1", "qwen3.8-9B"): {"source": src},
             ("apu-tablet-2", "qwen3.8-9b-distill"): {"source": src}},
        )
        (s,) = gw.split_models()
        assert s["repo"] == "empero-ai/Qwen3.8-9B-Distill-GGUF"
        assert s["names"] == {"qwen3.8-9B": ["gpu-laptop-1"],
                              "qwen3.8-9b-distill": ["apu-tablet-2"]}

    def test_the_same_name_everywhere_is_not_a_split(self):
        src = {"repo": "org/repo", "file": "m.gguf"}
        self._routes(
            {"m": ["apu-box-1", "gpu-laptop-1"]},
            {("apu-box-1", "m"): {"source": src}, ("gpu-laptop-1", "m"): {"source": src}},
        )
        assert gw.split_models() == []

    def test_an_alias_beside_its_own_id_is_not_a_split(self):
        # One box serving the model as both `fast` and its real id is normal.
        src = {"repo": "org/repo", "file": "m.gguf"}
        self._routes(
            {"m": ["apu-box-1"], "fast": ["apu-box-1"]},
            {("apu-box-1", "m"): {"source": src}, ("apu-box-1", "fast"): {"source": src}},
            {("apu-box-1", "m"): "m", ("apu-box-1", "fast"): "m"},
        )
        assert gw.split_models() == []

    def test_different_repos_are_different_models(self):
        self._routes(
            {"a": ["h1"], "b": ["h2"]},
            {("h1", "a"): {"source": {"repo": "org/r", "file": "m.gguf"}},
             ("h2", "b"): {"source": {"repo": "org/other", "file": "m.gguf"}}},
        )
        assert gw.split_models() == []

    def test_a_different_quant_of_one_repo_is_the_same_model(self):
        # The fleet already serves qwen3.8-27b as Q5_K_M on apu-box-1 and
        # UD-Q5_K_M on mac-laptop-1 under one id; gpu-laptop-1's Q3_K_M Qwopus under
        # its own name is a split, not a second model.
        self._routes(
            {"qwopus3.6-35b-coder": ["apu-box-1"], "Qwopus3.6-A3B": ["gpu-laptop-1"]},
            {("apu-box-1", "qwopus3.6-35b-coder"): {"source": {
                "repo": "Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF", "file": "q4.gguf"}},
             ("gpu-laptop-1", "Qwopus3.6-A3B"): {"source": {
                "repo": "Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF", "file": "q3.gguf"}}},
        )
        (s,) = gw.split_models()
        assert set(s["names"]) == {"qwopus3.6-35b-coder", "Qwopus3.6-A3B"}

    def test_an_ollama_tag_from_the_same_repo_joins_the_check(self):
        # mini-pc-1 serves the distill by its hf.co tag; the tag names the repo.
        self._routes(
            {"qwen3.8-9B": ["gpu-laptop-1"],
             "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M": ["mini-pc-1"]},
            {("gpu-laptop-1", "qwen3.8-9B"): {"source": {
                "repo": "empero-ai/Qwen3.8-9B-Distill-GGUF", "file": "Qwen3.8-9B-Q4_K_M.gguf"}},
             ("mini-pc-1", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"): {"source": {
                "tag": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"}}},
        )
        (s,) = gw.split_models()
        assert s["repo"] == "empero-ai/Qwen3.8-9B-Distill-GGUF"
        assert s["names"]["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"] == ["mini-pc-1"]

    def test_converged_names_are_not_a_split(self):
        # After the start-up rename on gpu-laptop-1 and the alias on mini-pc-1, every
        # box reports the same canonical id for the weights, whatever it
        # calls them locally.
        canon = "qwen3.8-9b-distill"
        self._routes(
            {"qwen3.8-9B": ["gpu-laptop-1"], canon: ["gpu-laptop-1", "mini-pc-1"],
             "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M": ["mini-pc-1"]},
            {("gpu-laptop-1", "qwen3.8-9B"): {"source": {"repo": "empero-ai/Qwen3.8-9B-Distill-GGUF", "file": "f"}},
             ("gpu-laptop-1", canon): {"source": {"repo": "empero-ai/Qwen3.8-9B-Distill-GGUF", "file": "f"}},
             ("mini-pc-1", canon): {"source": {"tag": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"}},
             ("mini-pc-1", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"): {"source": {
                 "tag": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"}}},
            {("gpu-laptop-1", "qwen3.8-9B"): canon, ("gpu-laptop-1", canon): canon,
             ("mini-pc-1", canon): canon,
             ("mini-pc-1", "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"): canon},
        )
        assert gw.split_models() == []

    def test_a_variant_on_one_box_is_not_a_split(self):
        # apu-box-1 serves the VL model as qwen3-vl-30b and, with its own
        # window, as qwen3-vl-30b-classify. Both are reachable everywhere the
        # weights are, so no box is invisible to either name.
        src = {"repo": "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF", "file": "m.gguf"}
        self._routes(
            {"qwen3-vl-30b": ["apu-box-1"], "qwen3-vl-30b-classify": ["apu-box-1"]},
            {("apu-box-1", "qwen3-vl-30b"): {"source": src},
             ("apu-box-1", "qwen3-vl-30b-classify"): {"source": src}},
        )
        assert gw.split_models() == []

    def test_a_variant_missing_from_a_second_box_is_a_split(self):
        # ...but a box that holds the weights under only one of the names
        # cannot stand in for the other, and that is exactly the finding.
        src = {"repo": "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF", "file": "m.gguf"}
        self._routes(
            {"qwen3-vl-30b": ["apu-box-1", "server-1"], "qwen3-vl-30b-classify": ["apu-box-1"]},
            {("apu-box-1", "qwen3-vl-30b"): {"source": src},
             ("server-1", "qwen3-vl-30b"): {"source": src},
             ("apu-box-1", "qwen3-vl-30b-classify"): {"source": src}},
        )
        (s,) = gw.split_models()
        assert s["names"]["qwen3-vl-30b-classify"] == ["apu-box-1"]

    def test_a_box_with_no_source_is_skipped_not_guessed(self):
        # An Ollama library tag (gemma4:26b) names no repo, and a hand-placed
        # file has none. Two different models can share a byte count, so an
        # unsourced entry contributes nothing rather than a guess.
        src = {"repo": "org/repo", "file": "m.gguf"}
        self._routes(
            {"m": ["apu-box-1"], "other": ["mac-desktop-1"], "gemma4:26b": ["cpu-box-1"]},
            {("apu-box-1", "m"): {"source": src},
             ("mac-desktop-1", "other"): {"source": None, "bytes": 123},
             ("cpu-box-1", "gemma4:26b"): {"source": {"tag": "gemma4:26b"}}},
        )
        assert gw.split_models() == []

    def test_model_repo_reads_both_engines(self):
        assert gw._model_repo({"repo": "org/r", "file": "f"}) == "org/r"
        assert gw._model_repo({"tag": "hf.co/org/r:Q4_K_M"}) == "org/r"
        assert gw._model_repo({"tag": "hf.co/org/r"}) == "org/r"
        assert gw._model_repo({"tag": "gemma4:26b"}) == ""
        assert gw._model_repo({"tag": "hf.co/r"}) == ""
        assert gw._model_repo({}) == ""

    def test_the_local_host_is_named(self):
        src = {"repo": "org/repo", "file": "m.gguf"}
        self._routes(
            {"m": [""], "m-alt": ["gpu-desktop-1"]},
            {("", "m"): {"source": src}, ("gpu-desktop-1", "m-alt"): {"source": src}},
        )
        (s,) = gw.split_models()
        assert s["names"]["m"] == [gw.HOST_NAME]


def test_routes_endpoint_carries_both_checks(client, admin_headers):
    r = client.get("/admin/api/routes", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["alias_conflicts"], list)
    assert isinstance(body["split_models"], list)


def test_served_models_reports_canonical(client, admin_headers):
    """The field the hub reads. Without it on the wire, nothing above works."""
    r = client.get("/admin/api/served-models", headers=admin_headers)
    assert r.status_code == 200
    assert isinstance(r.json().get("canonical"), dict)
