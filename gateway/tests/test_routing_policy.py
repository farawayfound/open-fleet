"""Host policy: which box in the fleet a model should land on.

Covers host_class/host_tier (the owner's ranking rules), _est_wall and
_score_host_model_pairs (how a tier is broken), resolve_targets (the public
catalogue's fan-out across fleet ids), measured_pp, estimate_prompt_tokens,
and the fallback-substitution helpers that ride on top of host_tier.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations


import pytest

import app as gw

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolate_routes_cache():
    """Every test here pokes _routes_cache directly (meta/cands/cap/running/
    ctx) rather than going through a real routing refresh, so the next test
    must not see what this one left behind."""
    snapshot = dict(gw._routes_cache)
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)


@pytest.fixture(autouse=True)
def _isolate_cooldown_and_inflight():
    gw._host_cooldown.clear()
    gw._inflight.clear()
    yield
    gw._host_cooldown.clear()
    gw._inflight.clear()


@pytest.fixture
def frozen_routes(monkeypatch):
    """Freeze model_routes() so resolve_targets()/pick_fallback()/model_hosts()
    -- which all call it before doing anything else -- read exactly the
    _routes_cache a test set up, instead of the real (network-touching)
    refresh clobbering it."""
    async def _fake(force: bool = False):
        return gw._routes_cache.get("map", {})
    monkeypatch.setattr(gw, "model_routes", _fake)


def _set_meta(**by_host_fid: dict) -> None:
    """by_host_fid keys are 'host|fid' strings, mapped to a meta dict."""
    gw._routes_cache["meta"] = {
        tuple(k.split("|", 1)): v for k, v in by_host_fid.items()
    }


# ---------------------------------------------------------------------------
# host_class
# ---------------------------------------------------------------------------

class TestHostClass:
    def test_class_matrix(self, monkeypatch):
        assert gw.host_class("gpu-laptop-1") == "gpu"
        assert gw.host_class("apu-box-1") == "big"
        assert gw.host_class("server-1") == "small"
        assert gw.host_class("cpu-box-1") == "fallback"
        assert gw.host_class("hub") == "hub"
        assert gw.host_class("some-unlisted-box") == "unknown"

        # A peer the spec sheet has an override for, but no explicit klass --
        # host_class() has to fall back to guessing from vram_gb rather than
        # calling it 'unknown', which would sink a real GPU box to the big
        # box's tier.
        fake_specs = {
            "big-guess": {"vram_gb": 80},
            "gpu-guess": {"vram_gb": 10},
            "small-guess": {"vram_gb": 0},
        }
        monkeypatch.setattr(gw, "load_specs", lambda: fake_specs)
        assert gw.host_class("big-guess") == "big"
        assert gw.host_class("gpu-guess") == "gpu"
        assert gw.host_class("small-guess") == "small"


# ---------------------------------------------------------------------------
# host_tier: the owner's ranking rules, each its own test
# ---------------------------------------------------------------------------

class TestHostTierOwnerRules:
    def test_zephyrus_vram_fit_beats_resident_ai_max(self):
        # A small model that fits gpu-laptop-1's 8 GB outright beats the big box
        # even when apu-box-1 already holds it resident -- residency is a
        # tiebreak inside a tier, not a way to jump one.
        _set_meta(**{"gpu-laptop-1|m": {"fit": "vram"}, "apu-box-1|m": {"fit": "spill"}})
        assert gw.host_tier("gpu-laptop-1", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_pipedream_vram_beats_ai_max(self):
        _set_meta(**{"gpu-desktop-2|m": {"fit": "vram"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-desktop-2", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_pipedream_spill_with_moe_beats_ai_max(self):
        # moe_spill_ok: gpu-desktop-2 keeps its GPU tier for a MoE whose experts
        # spill into its 192 GB of RAM.
        _set_meta(**{"gpu-desktop-2|m": {"fit": "spill", "moe": True}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-desktop-2", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_pipedream_spill_without_moe_does_not_beat_ai_max(self):
        # Without moe=True, moe_spill_ok does not apply -- an ordinary dense
        # model spilling off a 48 GB card is not a fast path.
        _set_meta(**{"gpu-desktop-2|m": {"fit": "spill", "moe": False}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-desktop-2", "m", "primary") > gw.host_tier("apu-box-1", "m", "primary")

    def test_terra_vram_beats_ai_max(self):
        _set_meta(**{"gpu-desktop-1|m": {"fit": "vram"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-desktop-1", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_terra_spill_behind_ai_max(self):
        _set_meta(**{"gpu-desktop-1|m": {"fit": "spill"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-desktop-1", "m", "primary") > gw.host_tier("apu-box-1", "m", "primary")

    def test_mb_pro_vram_beats_ai_max(self):
        _set_meta(**{"mac-laptop-1|m": {"fit": "vram"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("mac-laptop-1", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_mb_pro_spill_behind_ai_max(self):
        _set_meta(**{"mac-laptop-1|m": {"fit": "spill"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("mac-laptop-1", "m", "primary") > gw.host_tier("apu-box-1", "m", "primary")

    def test_stealth_vram_beats_ai_max(self):
        _set_meta(**{"gpu-laptop-2|m": {"fit": "vram"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-laptop-2", "m", "primary") < gw.host_tier("apu-box-1", "m", "primary")

    def test_stealth_spill_behind_ai_max(self):
        _set_meta(**{"gpu-laptop-2|m": {"fit": "spill"}, "apu-box-1|m": {"fit": ""}})
        assert gw.host_tier("gpu-laptop-2", "m", "primary") > gw.host_tier("apu-box-1", "m", "primary")

    def test_primary_role_ai_max_ahead_of_always_on_small_boxes_nanobot_last(self):
        # For a primary request the big box stays the first fallback behind
        # the fast cards, ahead of the always-on small boxes, which stay
        # ahead of the CPU backstop.
        ai_max = gw.host_tier("apu-box-1", "m", "primary")
        small_box = gw.host_tier("cpu-box-1", "m", "primary")
        for h in ("server-1", "mac-desktop-1", "mini-pc-1"):
            t = gw.host_tier(h, "m", "primary")
            assert ai_max < t < small_box

    def test_nanobot_last_in_every_role(self):
        for role in ("primary", "worker"):
            small_box = gw.host_tier("cpu-box-1", "m", role)
            for h in ("apu-box-1", "server-1", "mac-desktop-1", "mini-pc-1", "gpu-laptop-1"):
                assert gw.host_tier(h, "m", role) < small_box

    def test_unlisted_host_is_unknown_class_tier1_for_primary(self):
        assert gw.host_class("brand-new-peer") == "unknown"
        assert gw.host_tier("brand-new-peer", "m", "primary") == (1, 5)

    def test_worker_role_always_on_order_ahead_of_ai_max(self):
        """Sub-agents go to the always-on small boxes in the owner's order, so
        the big box stays free for the primary -- but cpu-box-1 is not one of
        them. It is the declared *fallback* ("a fallback for anything served
        by any machine except apu-box-1"), so it sits behind apu-box-1 even here:
        a CPU-only DDR4 box preempting 96 GB of VRAM would be the one place
        this policy made a request slower on purpose."""
        order = ["gpu-laptop-1", "server-1", "mac-desktop-1", "mini-pc-1"]
        _set_meta(**{"gpu-laptop-1|m": {"fit": "spill", "moe": False}})
        tiers = [gw.host_tier(h, "m", "worker") for h in order]
        assert tiers == sorted(tiers)
        ai_max = gw.host_tier("apu-box-1", "m", "worker")
        assert all(t < ai_max for t in tiers)
        assert gw.host_tier("cpu-box-1", "m", "worker") > ai_max


# ---------------------------------------------------------------------------
# saturation and cooldown -- the keys _score_host_model_pairs sorts on before
# the owner's tier even gets consulted
# ---------------------------------------------------------------------------

class TestSaturationAndCooldown:
    async def test_saturated_host_sorts_last(self):
        gw._routes_cache["cap"] = {("hostA", "m"): 1, ("hostB", "m"): 1}
        gw._routes_cache["running"] = {}
        gw._routes_cache["meta"] = {}
        gw._inflight["hostA"] = 1
        out = await gw._score_host_model_pairs(
            [("hostA", "m"), ("hostB", "m")], role="primary")
        assert out == [("hostB", "m"), ("hostA", "m")]

    async def test_cooldown_host_sorts_after_healthy_then_mark_ok_clears_it(self):
        gw._routes_cache["cap"] = {}
        gw._routes_cache["running"] = {}
        gw._routes_cache["meta"] = {}
        gw._mark_host_down("hostA", 30.0, "test")
        out = await gw._score_host_model_pairs(
            [("hostA", "m"), ("hostB", "m")], role="primary")
        assert out == [("hostB", "m"), ("hostA", "m")]

        # _mark_host_ok() clears the cooldown -- the same tie between two
        # equally-ranked unknown hosts is then broken by the order given.
        gw._mark_host_ok("hostA")
        assert gw.host_cooling("hostA") is False
        out2 = await gw._score_host_model_pairs(
            [("hostA", "m"), ("hostB", "m")], role="primary")
        assert out2 == [("hostA", "m"), ("hostB", "m")]

    def test_host_cooling_expires_on_its_own(self, monkeypatch):
        t = [1_000_000.0]
        monkeypatch.setattr(gw.time, "time", lambda: t[0])
        gw._mark_host_down("flakyhost", 5.0, "test")
        assert gw.host_cooling("flakyhost") is True
        t[0] += 10.0
        assert gw.host_cooling("flakyhost") is False
        assert "flakyhost" not in gw._host_cooldown


# ---------------------------------------------------------------------------
# need_ctx: a request that will not fit demotes a host with a KNOWN small
# ceiling, but never one whose ceiling is simply unreported (0 == unknown).
# ---------------------------------------------------------------------------

class TestNeedCtx:
    async def test_need_ctx_demotes_known_small_ceiling_not_unknown(self):
        _set_meta(**{"gpu-laptop-1|m": {"fit": "vram"}, "gpu-desktop-2|m": {"fit": "vram"}})
        gw._routes_cache["cap"] = {}
        gw._routes_cache["running"] = {}
        gw._routes_cache["ctx"] = {("gpu-laptop-1", "m"): 4096}  # known and small
        # gpu-desktop-2 reports nothing for this pair -- host_model_ctx() reads
        # that as "unknown", not as zero, and must not be demoted for it.
        out = await gw._score_host_model_pairs(
            [("gpu-laptop-1", "m"), ("gpu-desktop-2", "m")], role="primary", need_ctx=8192)
        assert out == [("gpu-desktop-2", "m"), ("gpu-laptop-1", "m")]


# ---------------------------------------------------------------------------
# within a tier: residency vs. measured speed (the qwen3.6-35b-on-gpu-laptop-1
# incident -- a 42k-token prompt sent to an 8 GB laptop because it was warm)
# ---------------------------------------------------------------------------

class TestResidencyVsSpeed:
    def _tie(self, monkeypatch):
        # gpu-desktop-1 and mac-laptop-1 are both bare GPU boxes with no rank/always_on
        # field, so a shared fit='vram' ties them at (tier, rank) == (0, 0);
        # only _est_wall can break the tie. The real sheet marks gpu-desktop-1
        # `reserve` now (somebody's personal machine), which would decide
        # these before speed is even consulted -- that ordering is pinned in
        # test_reserve_boxes.py; here the flag is stripped so the subject
        # stays residency vs. measured speed.
        specs = {h: dict(s) for h, s in gw.DEFAULT_SPECS.items()}
        specs["gpu-desktop-1"].pop("reserve", None)
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        gw._routes_cache["cap"] = {("gpu-desktop-1", "m"): 1, ("mac-laptop-1", "m"): 1}
        _set_meta(**{"gpu-desktop-1|m": {"fit": "vram"}, "mac-laptop-1|m": {"fit": "vram"}})
        assert gw.host_tier("gpu-desktop-1", "m", "primary") == gw.host_tier("mac-laptop-1", "m", "primary")

    async def test_resident_beats_cold_for_small_prompt(self, monkeypatch):
        self._tie(monkeypatch)
        gw._routes_cache["running"] = {"gpu-desktop-1": {"m"}, "mac-laptop-1": set()}
        monkeypatch.setattr(gw, "measured_tps", lambda model: {})
        monkeypatch.setattr(gw, "measured_pp", lambda model: {})
        out = await gw._score_host_model_pairs(
            [("gpu-desktop-1", "m"), ("mac-laptop-1", "m")], role="primary",
            prompt_tokens=50, gen_tokens=50)
        assert out[0] == ("gpu-desktop-1", "m")

    async def test_fast_cold_beats_resident_slow_for_40k_prompt(self, monkeypatch):
        self._tie(monkeypatch)
        gw._routes_cache["running"] = {"gpu-desktop-1": {"m"}, "mac-laptop-1": set()}
        # mac-laptop-1 is cold, but ~20x faster at both decode and prompt reading
        # than gpu-desktop-1's un-measured class defaults.
        monkeypatch.setattr(gw, "measured_tps", lambda model: {"mac-laptop-1": 900.0})
        monkeypatch.setattr(gw, "measured_pp", lambda model: {"mac-laptop-1": 30000.0})
        out = await gw._score_host_model_pairs(
            [("gpu-desktop-1", "m"), ("mac-laptop-1", "m")], role="primary",
            prompt_tokens=40_000, gen_tokens=256)
        assert out[0] == ("mac-laptop-1", "m")


# ---------------------------------------------------------------------------
# _catalogue_row_resident_well and pick_fallback
# ---------------------------------------------------------------------------

class TestResidentWellAndFallback:
    def _setup_nanobot_vs_ai_max(self):
        # qwen3.6-35b-a3b (moe, active_b=3) is only warm on cpu-box-1, the CPU
        # backstop; gemma4-26b-a4b (moe, active_b=4 -- within the default 0.5
        # tolerance) is warm on apu-box-1, a real home for it.
        gw._routes_cache["cands"] = {
            "qwen3.6-35b": ["cpu-box-1"],
            "gemma-4-26b": ["apu-box-1"],
            "gemma4:26b": ["apu-box-1"],
        }
        gw._routes_cache["running"] = {
            "cpu-box-1": {"qwen3.6-35b"}, "apu-box-1": {"gemma-4-26b"},
        }
        gw._routes_cache["cap"] = {}
        gw._routes_cache["meta"] = {}

    def test_resident_only_on_nanobot_is_not_well(self):
        self._setup_nanobot_vs_ai_max()
        assert gw._catalogue_row_resident_well(["qwen3.6-35b"], "primary") is False

    def test_resident_on_ai_max_is_well(self):
        self._setup_nanobot_vs_ai_max()
        assert gw._catalogue_row_resident_well(["gemma-4-26b", "gemma4:26b"], "primary") is True

    async def test_pick_fallback_swaps_away_from_nanobot_only_resident(self, frozen_routes):
        self._setup_nanobot_vs_ai_max()
        cat = gw.public_catalogue(force=True)["by_public"]
        req_row = cat["qwen3.6-35b-a3b"]
        settings = gw.get_public_settings()
        sub = await gw.pick_fallback(req_row, "primary", settings)
        assert sub is not None
        assert sub["public_id"] == "gemma4-26b-a4b"


# ---------------------------------------------------------------------------
# measured_pp: prompt tokens/sec from the usage log
# ---------------------------------------------------------------------------

class TestMeasuredPP:
    def test_reads_and_filters_the_usage_log(self):
        model = "test-pp-model-alpha"
        # Counts: enough prompt tokens, a real ttft, status < 400.
        gw.db_exec(
            "INSERT INTO usage(ts,model,endpoint,stream,status,prompt_tokens,"
            "ttft_ms,host) VALUES (?,?,?,?,?,?,?,?)",
            (gw.now(), model, "/v1/chat/completions", 1, 200, 5000, 4000, "hostA"),
        )
        # Too small a prompt to say anything about read speed -- excluded.
        gw.db_exec(
            "INSERT INTO usage(ts,model,endpoint,stream,status,prompt_tokens,"
            "ttft_ms,host) VALUES (?,?,?,?,?,?,?,?)",
            (gw.now(), model, "/v1/chat/completions", 1, 200, 100, 500, "hostB"),
        )
        # No ttft recorded (non-streamed) -- excluded.
        gw.db_exec(
            "INSERT INTO usage(ts,model,endpoint,stream,status,prompt_tokens,"
            "ttft_ms,host) VALUES (?,?,?,?,?,?,?,?)",
            (gw.now(), model, "/v1/chat/completions", 0, 200, 5000, 0, "hostC"),
        )
        out = gw.measured_pp(model)
        assert out == {"hostA": 5000 / 4.0}


# ---------------------------------------------------------------------------
# estimate_prompt_tokens: a pure function, guarded the same way apply_ctx_limit
# guards its own char-count estimate
# ---------------------------------------------------------------------------

class TestEstimatePromptTokens:
    def test_estimate_prompt_tokens(self):
        payload = {
            "messages": [{"role": "user", "content": "x" * 320}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }
        without_tools = gw.estimate_prompt_tokens({"messages": payload["messages"]})
        with_tools = gw.estimate_prompt_tokens(payload)
        assert without_tools > 0
        assert with_tools > without_tools

        # falls back to the 'prompt' field when there are no messages
        assert gw.estimate_prompt_tokens({"prompt": "hello there"}) > 0

        # a non-dict payload (or none at all) is simply zero, not an error
        assert gw.estimate_prompt_tokens("not-a-dict") == 0
        assert gw.estimate_prompt_tokens(None) == 0


# ---------------------------------------------------------------------------
# resolve_targets: public catalogue id -> every (host, fleet id) pair across
# ALL the row's fleet ids, ranked together; falls back to a plain fleet id.
# ---------------------------------------------------------------------------

class TestResolveTargets:
    async def test_ranks_catalogue_row_then_falls_back_outside_it(self, frozen_routes):
        # gemma4-26b-a4b's two fleet ids are served by two different hosts;
        # resolve_targets has to rank them against each other in one list.
        gw._routes_cache["cands"] = {
            "gemma-4-26b": ["gpu-laptop-1"], "gemma4:26b": ["cpu-box-1"],
        }
        gw._routes_cache["running"] = {}
        gw._routes_cache["cap"] = {}
        _set_meta(**{"gpu-laptop-1|gemma-4-26b": {"fit": "vram"}})
        out = await gw.resolve_targets("gemma4-26b-a4b", role="primary")
        assert out == [("gpu-laptop-1", "gemma-4-26b"), ("cpu-box-1", "gemma4:26b")]

        # A plain fleet id outside the public catalogue falls back to
        # model_hosts()'s own candidate list instead.
        gw._routes_cache["cands"] = {"some-private-model": ["gpu-laptop-1", "cpu-box-1"]}
        _set_meta(**{"gpu-laptop-1|some-private-model": {"fit": "vram"}})
        out2 = await gw.resolve_targets("some-private-model", role="primary")
        assert out2 == [("gpu-laptop-1", "some-private-model"), ("cpu-box-1", "some-private-model")]
