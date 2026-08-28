"""The dashboards must agree about models -- with each other and with disk.

Two sync surfaces are pinned here:

1. /admin/api/models carries `upstream`: on an Ollama-backed box
   (LLMSTACK_MODELS_FROM_UPSTREAM=1) the Models/Library tabs render the
   engine's own live catalogue instead of an empty models.json view, so this
   dashboard and any external admin page (downstream-app's Ollama tab) are two
   views of one daemon and cannot drift.

2. PUBLIC_MODELS_SEED and gateway/public_seed.json are two copies of the same
   catalogue with no shared import -- the only thing keeping them identical
   is this test.

Run with: python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

import app as gw


@pytest.fixture()
def _clean_upstream_caches():
    """These caches are module-global; leave them as this test found them."""
    saved = (dict(gw._upstream_tags_cache), dict(gw._upstream_ctx_cache),
             dict(gw._upstream_meta_cache), dict(gw._running_cache))
    yield
    gw._upstream_tags_cache.clear()
    gw._upstream_tags_cache.update(saved[0])
    gw._upstream_ctx_cache.clear()
    gw._upstream_ctx_cache.update(saved[1])
    gw._upstream_meta_cache.clear()
    gw._upstream_meta_cache.update(saved[2])
    gw._running_cache.clear()
    gw._running_cache.update(saved[3])


class _FakeOllama:
    """Answers the three endpoints upstream_catalogue() touches the way a
    real Ollama does: a catalogue on /api/tags, 404 on /running (that is a
    llama-swap path), residency on /api/ps."""

    def __init__(self):
        self.tags = {"models": [
            {"model": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M", "size": 6_600_000_000,
             "modified_at": "2026-08-24T10:00:00.000000000Z",
             "details": {"parameter_size": "9.2B", "quantization_level": "Q4_K_M",
                         "family": "qwen3"}},
            {"model": "gemma4:12b-it-qat", "size": 8_800_000_000,
             "modified_at": "2026-07-23T09:00:00.000000000Z",
             "details": {"parameter_size": "12.2B", "quantization_level": "Q4_0",
                         "family": "gemma4"}},
        ]}

    async def get(self, path, **kw):
        req = httpx.Request("GET", "http://x" + path)
        if path == "/api/tags":
            return httpx.Response(200, json=self.tags, request=req)
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"}]},
                                  request=req)
        return httpx.Response(404, request=req)

    async def post(self, path, **kw):
        return httpx.Response(404, request=httpx.Request("POST", "http://x" + path))


class TestUpstreamCatalogue:

    @pytest.mark.asyncio
    async def test_disabled_reports_disabled_without_touching_the_engine(
            self, monkeypatch, _clean_upstream_caches):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", False)
        monkeypatch.setattr(gw, "client", None)  # would blow up if touched
        assert await gw.upstream_catalogue() == {"enabled": False, "models": []}

    @pytest.mark.asyncio
    async def test_enabled_serves_the_live_tags_with_ctx_and_residency(
            self, monkeypatch, _clean_upstream_caches):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        monkeypatch.setattr(gw, "client", _FakeOllama())
        gw._upstream_tags_cache.update(t=0.0, models=[])
        gw._running_cache.update(t=0.0, ids=set())
        # Pre-warm the ctx cache so upstream_model_ctx() answers inline instead
        # of spawning its background /api/show sweep mid-test.
        gw._upstream_ctx_cache.update(t=time.time(), ctx={"hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M": 24576})

        out = await gw.upstream_catalogue()
        assert out["enabled"] is True
        by_id = {m["id"]: m for m in out["models"]}
        assert set(by_id) == {"hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M", "gemma4:12b-it-qat"}
        assert by_id["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]["running"] is True
        assert by_id["gemma4:12b-it-qat"]["running"] is False
        assert by_id["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]["ctx"] == 24576
        assert by_id["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]["params"] == "9.2B"
        assert by_id["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]["size"] == 6_600_000_000

    @pytest.mark.asyncio
    async def test_a_non_200_answer_keeps_the_last_answer_too(
            self, monkeypatch, _clean_upstream_caches):
        """A 500 mid-restart raises nothing -- it must not clobber the cache."""
        class FiveHundred:
            async def get(self, path, **kw):
                return httpx.Response(
                    500, request=httpx.Request("GET", "http://x" + path))

        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        monkeypatch.setattr(gw, "client", FiveHundred())
        gw._upstream_tags_cache.update(
            t=0.0, models=[{"id": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M",
                            "size": 1, "modified": "", "params": "", "quant": "",
                            "family": ""}])
        gw._running_cache.update(t=time.time(), ids=set())
        gw._upstream_ctx_cache.update(t=time.time(), ctx={})

        out = await gw.upstream_catalogue()
        assert [m["id"] for m in out["models"]] == [
            "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]

    @pytest.mark.asyncio
    async def test_an_unreachable_engine_keeps_the_last_answer(
            self, monkeypatch, _clean_upstream_caches):
        class Down:
            async def get(self, path, **kw):
                raise httpx.ConnectError("down")

        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        monkeypatch.setattr(gw, "client", Down())
        gw._upstream_tags_cache.update(
            t=0.0, models=[{"id": "hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M", "size": 1, "modified": "",
                            "params": "", "quant": "", "family": ""}])
        gw._running_cache.update(t=time.time(), ids=set())
        gw._upstream_ctx_cache.update(t=time.time(), ctx={})

        out = await gw.upstream_catalogue()
        assert [m["id"] for m in out["models"]] == ["hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M"]

    def test_admin_models_endpoint_carries_the_upstream_section(
            self, client, admin_headers, monkeypatch, _clean_upstream_caches):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        monkeypatch.setattr(gw, "client", _FakeOllama())
        gw._upstream_tags_cache.update(t=0.0, models=[])
        gw._running_cache.update(t=0.0, ids=set())
        gw._upstream_ctx_cache.update(t=time.time(), ctx={})

        r = client.get("/admin/api/models", headers=admin_headers)
        assert r.status_code == 200
        up = r.json()["upstream"]
        assert up["enabled"] is True
        assert {m["id"] for m in up["models"]} == {"hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M", "gemma4:12b-it-qat"}


class TestPublicSeedParity:

    def test_public_seed_json_matches_the_python_fallback(self):
        """PUBLIC_MODELS_SEED (app.py) and public_seed.json are hand-kept
        copies of one catalogue. If this fails, edit BOTH."""
        on_disk = json.loads(
            (Path(gw.__file__).parent / "public_seed.json").read_text("utf-8"))
        assert on_disk == gw.PUBLIC_MODELS_SEED
