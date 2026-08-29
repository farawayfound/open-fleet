"""A refused connection right after the llama-swap restart is "not up yet",
not "the engine refused the load".

The save restarts llama-swap and fires the verify probe in the same breath.
A bare llama-swap listens the instant it starts, which hid this; one with a
start-up preload hook (models.json `preload` / `persistent`, 2026-08-29)
begins loading BEFORE it listens, and the ConnectError that followed rolled
two perfectly good changes back in one evening -- apu-box-1's Flash-Next retune
and mac-laptop-1's 27B pin. The probe now retries a connect failure for
VERIFY_CONNECT_GRACE seconds; only one that persists is a verdict.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import app as gw


class RefusesThenAnswers:
    """llama-swap as seen across its own restart: the first `refuse` health
    probes find nothing listening, then the model comes up."""

    def __init__(self, refuse: int, props: dict | None = None):
        self.refuse = refuse
        self.health_calls = 0
        self.answered = False
        self.props = props or {}

    async def get(self, path, **kwargs):
        outer = self
        await asyncio.sleep(0)          # a real suspension point, like a socket
        if path.endswith("/health"):
            self.health_calls += 1
            if self.health_calls <= self.refuse:
                raise httpx.ConnectError("connection refused")
            self.answered = True

        class R:
            status_code = 200

            def json(self_inner):
                if path == "/running":
                    # Nothing is resident until the probe that loads it has
                    # got through -- exactly what llama-swap reports.
                    st = "ready" if outer.answered else "starting"
                    return {"running": [{"model": "m1", "state": st}]}
                if path.endswith("/props"):
                    return {"default_generation_settings": outer.props}
                return {}

            text = ""
        return R()


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(gw, "VERIFY_CONNECT_GRACE", 4.0)
    monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 20.0)
    real_sleep = asyncio.sleep

    async def fast(seconds):
        await real_sleep(0.01)
    monkeypatch.setattr(gw.asyncio, "sleep", fast)


@pytest.mark.asyncio
class TestConnectGrace:
    async def test_two_refusals_inside_the_grace_are_retried(self, monkeypatch, quick):
        fake = RefusesThenAnswers(refuse=2, props={"n_ctx": 4096})
        monkeypatch.setattr(gw, "client", fake)
        ok, why = await gw._try_load("m1", 4096, {"aborted": False})
        assert ok, why
        assert fake.health_calls == 3

    async def test_a_refusal_that_never_ends_is_still_a_refusal(self, monkeypatch, quick):
        """The grace is a window, not a blindfold: an engine that is really
        gone must still fail the verify, with the honest reason."""
        monkeypatch.setattr(gw, "VERIFY_CONNECT_GRACE", 0.0)
        fake = RefusesThenAnswers(refuse=10 ** 6)
        monkeypatch.setattr(gw, "client", fake)
        ok, why = await gw._try_load("m1", 4096, {"aborted": False})
        assert not ok
        assert "ConnectError" in why
