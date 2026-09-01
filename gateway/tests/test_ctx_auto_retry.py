"""Context windows that self-tune down instead of just failing.

Two engines, two moments a window can turn out too big for the box, and one
shared rule: try once at half the window, and remember the number that
actually worked so the fleet stops repeating a failure it has already seen.

  * llama.cpp / llama-swap -- the window is baked into the launch command,
    so "too big" shows up as a Save & Apply that will not load. Retried in
    `_verify_apply()`, and ONLY for an auto-sized model ("ctx" 0): a pinned
    number is an operator's explicit choice and is refused honestly, never
    silently downsized. See test_model_apply.py for the base verify/revert
    machinery this extends.

  * Ollama -- the window is a per-request `options.num_ctx`, so "too big" is
    a single call's failure, not a load. Retried in `native_proxy()`'s
    non-streaming path, and remembered fleet-wide via the same `model_ctx`
    table routing already reads (host_model_ctx / _score_host_model_pairs).
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import app as gw


# ---------------------------------------------------------------------------
# halve_ctx(): the shared sizing step
# ---------------------------------------------------------------------------


class TestHalveCtx:
    def test_halves_and_rounds_to_the_4096_grid(self):
        # 20000 // 2 == 10000, which is not on the grid resolve_ctx() rounds
        # to -- the retry number must land on it too, or a value nothing
        # would ever launch with gets proposed as "the" number.
        assert gw.halve_ctx(20000) == 8192

    def test_a_clean_power_of_two_halves_exactly(self):
        assert gw.halve_ctx(65536) == 32768
        assert gw.halve_ctx(32768) == 16384
        assert gw.halve_ctx(8192) == 4096

    def test_stops_at_the_floor(self):
        """Once a retry would land AT the floor there is still one worthwhile
        number to try; once `ctx` is already there, there is nothing smaller
        left that is not a worse guess than just failing honestly."""
        assert gw.halve_ctx(4096) == 0
        assert gw.halve_ctx(2048) == 0
        assert gw.halve_ctx(0) == 0

    def test_never_returns_below_the_floor(self):
        assert gw.halve_ctx(5000) == 4096

    def test_a_custom_floor_is_honoured(self):
        assert gw.halve_ctx(16384, floor=8192) == 8192
        assert gw.halve_ctx(8192, floor=8192) == 0


# ---------------------------------------------------------------------------
# _verify_apply(): the llama.cpp / llama-swap Save & Apply retry
# ---------------------------------------------------------------------------


REC = dict(gw.DEFAULT_MODEL_RECORD, id="m1", path="/models/m1.gguf")  # ctx=0, auto


@pytest.fixture(autouse=True)
def _clean_apply_state():
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.APPLY_KEY,))
    yield
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.APPLY_KEY,))


_REAL_SLEEP = asyncio.sleep


@pytest.fixture
def nosleep(monkeypatch):
    monkeypatch.setattr(gw.asyncio, "sleep", lambda _s=0: _REAL_SLEEP(0))


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
    return tmp_path


class RetryOnceClient:
    """m1's /health fails the first time with an OOM-flavoured 5xx, and
    succeeds the second -- the shape a graceful auto-ctx retry needs to
    recover from without an operator ever seeing a failed apply."""

    def __init__(self, success_ctx: int, always_fail: bool = False):
        self.health_calls = 0
        self.success_ctx = success_ctx
        self.always_fail = always_fail
        self.logs = "cudaMalloc failed: out of memory"

    async def get(self, path, **kwargs):
        await _REAL_SLEEP(0)
        outer = self

        class R:
            status_code = 200

            def json(self_inner):
                if path == "/running":
                    # Never "ready": the verdict comes from the /health
                    # trigger's status code alone, same as the existing
                    # llama-swap OOM tests in test_model_apply.py.
                    return {"running": []}
                if path.endswith("/props"):
                    return {"default_generation_settings": {"n_ctx": outer.success_ctx}}
                return {}

            @property
            def text(self_inner):
                return outer.logs

        if path.endswith("/health"):
            outer.health_calls += 1
            first = outer.health_calls == 1
            R.status_code = 503 if (first or outer.always_fail) else 200
        return R()


class TestAutoSizedModelSelfTunesDown:
    @pytest.mark.asyncio
    async def test_a_retry_that_fits_is_verified_not_reverted(
            self, monkeypatch, registry, nosleep):
        gw.save_models([REC])
        monkeypatch.setattr(gw, "resolve_ctx", lambda rec: (131072, {}))
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 5.0)
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: restarts.append(a) or (0, "ok"))
        monkeypatch.setattr(gw, "client", RetryOnceClient(success_ctx=65536))

        await gw._verify_apply([REC], {"m1": REC})

        state = gw.get_apply_state()
        assert state["status"] == "ok", state
        assert state["verified"] == ["m1"]
        assert not state["failures"]
        # the halved number is what actually got saved -- "auto" turned into
        # a proven pin, not left to guess the same way again next time.
        assert gw.load_models()[0]["ctx"] == 65536
        # staged once (the halved config) -- no separate revert restart,
        # since the retry succeeded and there was nothing left to put back.
        assert restarts == ["restart"]

    @pytest.mark.asyncio
    async def test_a_retry_that_still_does_not_fit_is_reverted(
            self, monkeypatch, registry, nosleep):
        old = dict(REC)
        gw.save_models([REC])
        monkeypatch.setattr(gw, "resolve_ctx", lambda rec: (131072, {}))
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 5.0)
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: restarts.append(a) or (0, "ok"))
        monkeypatch.setattr(
            gw, "client", RetryOnceClient(success_ctx=65536, always_fail=True))

        await gw._verify_apply([REC], {"m1": old})

        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert f["id"] == "m1"
        assert f["overfilled"] is True
        assert f["retried"] == {"from": 131072, "to": 65536}
        assert "65536" in f["why"] and "131072" in f["why"]
        # one restart to stage + prove the halved retry, one more to put the
        # old record back -- never one restart per model, whatever the retry
        # count, but a genuinely separate operation still costs its own.
        assert restarts == ["restart", "restart"]
        assert gw.load_models()[0]["ctx"] == old["ctx"]

    @pytest.mark.asyncio
    async def test_a_restart_that_does_not_come_back_is_never_probed(
            self, monkeypatch, registry, nosleep):
        """The same trap api_models_put() already refuses for the outer
        save: a stale llama-swap still holding the OLD config answers every
        health check happily, which would prove nothing and call it proven."""
        old = dict(REC)
        gw.save_models([REC])
        monkeypatch.setattr(gw, "resolve_ctx", lambda rec: (131072, {}))
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 5.0)
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: (1, "llama-swap did not come back"))
        fake = RetryOnceClient(success_ctx=65536)
        monkeypatch.setattr(gw, "client", fake)

        await gw._verify_apply([REC], {"m1": old})

        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert "did not restart" in f["why"]
        assert gw.load_models()[0]["ctx"] == old["ctx"]
        # the retry's own health check was never sent -- proving would mean
        # asking a stale engine, and a stale engine always says yes.
        assert fake.health_calls == 1   # only the FIRST attempt, no retry probe

    @pytest.mark.asyncio
    async def test_a_pinned_ctx_is_never_retried(self, monkeypatch, registry, nosleep):
        """An operator who typed a number gets an honest refusal, not a
        number they never chose silently substituted for it."""
        old = dict(REC, ctx=32768)
        new = dict(REC, ctx=131072)
        gw.save_models([new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        called = {"resolve_ctx": 0}
        real_resolve = gw.resolve_ctx

        def counting_resolve(rec):
            if int(rec.get("ctx", 0) or 0) <= 0:
                called["resolve_ctx"] += 1
            return real_resolve(rec)
        monkeypatch.setattr(gw, "resolve_ctx", counting_resolve)
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: restarts.append(a) or (0, "ok"))
        monkeypatch.setattr(gw, "client", RetryOnceClient(success_ctx=65536))

        await gw._verify_apply([new], {"m1": old})

        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert "retried" not in f
        assert called["resolve_ctx"] == 0          # never treated as auto
        assert restarts == ["restart"]              # exactly the old behaviour
        assert gw.load_models()[0]["ctx"] == 32768

    @pytest.mark.asyncio
    async def test_no_smaller_number_left_reverts_like_any_other_failure(
            self, monkeypatch, registry, nosleep):
        """resolve_ctx already picked the floor -- halving it buys nothing,
        so this is a real failure, reported once, not retried forever."""
        old = dict(REC)
        gw.save_models([REC])
        monkeypatch.setattr(gw, "resolve_ctx", lambda rec: (4096, {}))
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: restarts.append(a) or (0, "ok"))
        monkeypatch.setattr(
            gw, "client", RetryOnceClient(success_ctx=4096, always_fail=True))

        await gw._verify_apply([REC], {"m1": old})

        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert "retried" not in f
        assert restarts == ["restart"]


# ---------------------------------------------------------------------------
# native_proxy(): the Ollama per-request retry
# ---------------------------------------------------------------------------


def _oom_response(request):
    return httpx.Response(
        500, request=request,
        json={"error": "cuda error: out of memory when allocating the kv cache"})


def _ok_response(request):
    return httpx.Response(
        200, request=request,
        json={"message": {"role": "assistant", "content": "ok"}, "done": True,
             "prompt_eval_count": 3, "eval_count": 2})


class TestNativeOllamaCtxRetry:
    def test_an_overfit_call_is_retried_at_half_and_remembered(
            self, client, admin_headers, monkeypatch):
        raw, meta = gw.mint_key("ctx-retry-1")
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            body = json.loads(request.content)
            seen.append(body)
            if len(seen) == 1:
                return _oom_response(request)
            return _ok_response(request)
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}],
                 "options": {"num_ctx": 65536}})

        assert r.status_code == 200, r.text
        assert len(seen) == 2
        assert seen[0]["options"]["num_ctx"] == 65536
        assert seen[1]["options"]["num_ctx"] == 32768   # halve_ctx(65536)
        row = gw.db_query(
            "SELECT host, ctx FROM model_ctx WHERE model=?", ("m1",))
        assert row and row[0]["ctx"] == 32768

    def test_a_retry_that_also_fails_returns_its_own_answer(
            self, client, monkeypatch):
        raw, meta = gw.mint_key("ctx-retry-2")
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content))
            return _oom_response(request)
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}],
                 "options": {"num_ctx": 65536}})

        assert r.status_code == 500
        assert len(seen) == 2   # one retry, not a loop
        rows = gw.db_query("SELECT * FROM model_ctx WHERE model=?", ("m1",))
        assert not rows   # nothing worked, so nothing is remembered

    def test_no_num_ctx_means_no_retry(self, client, monkeypatch):
        raw, meta = gw.mint_key("ctx-retry-3")
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content))
            return _oom_response(request)
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 500
        assert len(seen) == 1

    def test_a_failure_unrelated_to_memory_is_not_retried(self, client, monkeypatch):
        raw, meta = gw.mint_key("ctx-retry-4")
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content))
            return httpx.Response(404, request=request,
                                  json={"error": "model 'm1' not found"})
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": False,
                 "messages": [{"role": "user", "content": "hi"}],
                 "options": {"num_ctx": 65536}})

        assert r.status_code == 404
        assert len(seen) == 1

    def test_a_streaming_call_is_never_retried(self, client, monkeypatch):
        """Ollama streams by default, and bytes already relayed to the caller
        cannot be taken back -- so this path is left to fail honestly."""
        raw, meta = gw.mint_key("ctx-retry-5")
        seen: list[dict] = []

        async def fake_send(self, request, stream=False, **kw):
            seen.append(json.loads(request.content))
            return _oom_response(request)
        monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

        r = client.post(
            "/api/chat", headers={"Authorization": "Bearer " + raw},
            json={"model": "m1", "stream": True,
                 "messages": [{"role": "user", "content": "hi"}],
                 "options": {"num_ctx": 65536}})

        assert r.status_code == 500
        assert len(seen) == 1
