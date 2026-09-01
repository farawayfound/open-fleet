"""Save & apply on the Models tab: does the config actually take effect, and
is a config that cannot load rolled back?

Both halves come from one measurement on mac-laptop-1, the fleet's always-on primary.
Its models.json said `-c 182768`, the llama-swap.yaml the gateway generated
from it said `-c 182768`, and the llama-server that answered every request
said `-c 32768` -- because the save's `service_control("restart", ...)` ran
`sudo systemctl` on a machine that has no systemctl, reported a non-zero rc
into a success toast, and nobody checked afterwards whether the model that
came back was the model that had been asked for.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app as gw


# --------------------------------------------------------------------------
# restarting the engine on a cron-supervised Mac
# --------------------------------------------------------------------------


class Recorder:
    """subprocess.run, remembering argv. `alive` drives pgrep's exit code:
    True means "a process matches", which is 0."""

    def __init__(self, alive=(1,)):
        self.calls: list[tuple[str, ...]] = []
        self.alive = list(alive)

    def __call__(self, cmd, **kwargs):
        self.calls.append(tuple(cmd))
        rc = 0
        if cmd[0] == "pgrep":
            rc = self.alive.pop(0) if self.alive else 1

        class P:
            returncode = rc
            stdout = ""
            stderr = ""

        return P()

    def argv0s(self):
        return [c[0] for c in self.calls]


class FakePopen:
    def __init__(self):
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(tuple(cmd))
        self.kwargs.append(kwargs)

        class P:
            pid = 4242

        return P()


@pytest.fixture
def on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(gw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    runner = tmp_path / "run-llama-swap.sh"
    runner.write_text("#!/bin/sh\n")
    monkeypatch.setattr(gw, "_MAC_RUNNER", str(runner))
    monkeypatch.setattr(gw, "_MAC_SWAP_LOG", tmp_path / "llama-swap.log")
    return runner


class TestDarwinRestart:
    def test_restart_kills_the_engine_and_relaunches_the_runner(
            self, monkeypatch, on_darwin):
        # pgrep: gone after the kill, still gone at the orphan-reap check
        # (which then pkills any leftover llama-server -- see
        # test_darwin_orphan_reap.py), then present after the relaunch.
        rec = Recorder(alive=[1, 1, 0])
        pop = FakePopen()
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", pop)
        rc, out = gw.service_control("restart", "llama-swap")
        assert rc == 0, out
        assert rec.argv0s()[0] == "pkill"
        assert pop.calls == [(str(on_darwin),)]

    def test_it_matches_the_engine_not_the_cron_wrapper(
            self, monkeypatch, on_darwin):
        """run-llama-swap.sh ends in `exec llama-swap`, so the script's name is
        gone from the process it becomes. `pgrep -f run-llama-swap` -- what the
        cron line itself uses -- matches only the /bin/sh wrapper above it, and
        killing by that orphans a live llama-swap still holding :8081."""
        rec = Recorder(alive=[1, 1, 0])
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", FakePopen())
        gw.service_control("restart", "llama-swap")
        pattern = rec.calls[0][-1]
        assert "bin/llama-swap" in pattern
        assert "run-llama-swap" not in pattern

    def test_the_engine_is_detached_so_a_gateway_restart_cannot_take_it_down(
            self, monkeypatch, on_darwin):
        rec = Recorder(alive=[1, 1, 0])
        pop = FakePopen()
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", pop)
        gw.service_control("restart", "llama-swap")
        assert pop.kwargs[0]["start_new_session"] is True

    def test_an_engine_that_does_not_come_back_is_a_failure_not_a_success(
            self, monkeypatch, on_darwin):
        rec = Recorder(alive=[1] + [1] * 60)  # never appears after the spawn
        monkeypatch.setattr(gw.subprocess, "run", rec)
        monkeypatch.setattr(gw.subprocess, "Popen", FakePopen())
        rc, out = gw.service_control("restart", "llama-swap")
        assert rc == 1
        assert "did not come back" in out

    def test_cloudflared_is_still_honestly_unsupported(self, on_darwin):
        rc, out = gw.service_control("restart", "cloudflared")
        assert rc == 1
        assert "llama-swap" in out


# --------------------------------------------------------------------------
# proving the applied config loads
# --------------------------------------------------------------------------


REC = dict(gw.DEFAULT_MODEL_RECORD, id="m1", path="/models/m1.gguf", ctx=32768)


@pytest.fixture(autouse=True)
def _clean_apply_state():
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.APPLY_KEY,))
    yield
    gw.db_exec("DELETE FROM settings WHERE key=?", (gw.APPLY_KEY,))


# Captured before anything patches it: a stub written as
# `lambda _s: asyncio.sleep(0)` would call the patched name and recurse.
_REAL_SLEEP = asyncio.sleep


@pytest.fixture
def nosleep(monkeypatch):
    """The verification loop polls on a 2s timer. Tests should not wait."""
    monkeypatch.setattr(gw.asyncio, "sleep", lambda _s=0: _REAL_SLEEP(0))


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A models.json and llama-swap.yaml this test owns."""
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
    monkeypatch.setattr(gw, "service_control", lambda a, u: (0, "ok"))
    return tmp_path


class TestWhatCountsAsChanged:
    def test_a_ctx_edit_relaunches_the_model(self):
        assert gw.build_cmd(REC) != gw.build_cmd(dict(REC, ctx=131072))

    def test_a_description_or_ttl_edit_does_not(self):
        """Editing a label should not evict a resident model."""
        assert gw.build_cmd(REC) == gw.build_cmd(dict(REC, description="x", ttl=60))

    def test_the_diff_names_the_field_that_changed(self):
        changes = gw._launch_diff(REC, dict(REC, ctx=131072))
        assert changes == ["ctx 32768 -> 131072"]

    def test_a_brand_new_model_reads_as_added(self):
        assert gw._launch_diff(None, REC) == ["added"]


class FakeClient:
    """llama-swap, as much of it as a verification touches.

    `states` is what /running reports on successive polls; `props` is what the
    model's own /props says once it is up."""

    def __init__(self, states, props=None, logs="", health_status=200):
        self.states = list(states)
        self.props = props or {}
        self.logs = logs
        self.health_status = health_status
        self.paths: list[str] = []

    async def get(self, path, **kwargs):
        self.paths.append(path)
        outer = self
        if path.endswith("/health"):
            if self.health_status is None:
                # llama-swap holds the request open for the whole of its
                # healthCheckTimeout while a model grinds. Never answers here.
                await asyncio.Event().wait()
            await _REAL_SLEEP(0)

        class R:
            status_code = outer.health_status if path.endswith("/health") else 200

            def json(self_inner):
                if path == "/running":
                    st = (outer.states.pop(0) if outer.states
                          else outer.states_last())
                    return {"running": ([{"model": "m1", "state": st}]
                                        if st else [])}
                if path.endswith("/props"):
                    return {"default_generation_settings": outer.props}
                return {}

            @property
            def text(self_inner):
                return outer.logs

        if path.endswith("/health"):
            R.status_code = self.health_status
            await asyncio.sleep(0)
        return R()

    def states_last(self):
        return None


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
class TestVerification:
    async def test_a_model_that_loads_and_keeps_its_window_passes(
            self, monkeypatch, registry, nosleep):
        gw.save_models([REC])
        fake = FakeClient(states=["starting", "ready"], props={"n_ctx": 32768})
        monkeypatch.setattr(gw, "client", fake)
        await gw._verify_apply([REC], {"m1": REC})
        state = gw.get_apply_state()
        assert state["status"] == "ok"
        assert state["verified"] == ["m1"]
        assert not state["failures"]

    async def test_a_load_that_never_becomes_ready_is_rolled_back(
            self, monkeypatch, registry, nosleep):
        """The user-facing requirement: revert, and say which changes failed."""
        old = dict(REC, ctx=32768)
        new = dict(REC, ctx=1310720)
        gw.save_models([new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        fake = FakeClient(states=[None, None], health_status=None,
                          logs="ggml_metal: failed to allocate")
        monkeypatch.setattr(gw, "client", fake)
        await gw._verify_apply([new], {"m1": old})
        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert f["id"] == "m1"
        assert f["changes"] == ["ctx 32768 -> 1310720"]
        assert f["reverted"] is True
        # and models.json really holds the old record again
        assert gw.load_models()[0]["ctx"] == 32768

    async def test_an_out_of_memory_failure_is_labelled_as_one(
            self, monkeypatch, registry, nosleep):
        new = dict(REC, ctx=1310720)
        gw.save_models([new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        monkeypatch.setattr(gw, "client", FakeClient(
            states=[None], health_status=503,
            logs="llama_model_load: unable to allocate backend buffer"))
        await gw._verify_apply([new], {"m1": REC})
        (f,) = gw.get_apply_state()["failures"]
        assert f["overfilled"] is True
        assert "does not fit in memory" in f["why"]

    async def test_an_unrelated_failure_is_not_called_a_memory_problem(
            self, monkeypatch, registry, nosleep):
        new = dict(REC, path="/models/typo.gguf")
        gw.save_models([new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        monkeypatch.setattr(gw, "client", FakeClient(
            states=[None], health_status=503,
            logs="llama_model_load: no such file or directory"))
        await gw._verify_apply([new], {"m1": REC})
        (f,) = gw.get_apply_state()["failures"]
        assert f["overfilled"] is False
        assert "does not fit" not in f["why"]

    async def test_a_silently_shrunken_window_counts_as_a_failure(
            self, monkeypatch, registry, nosleep):
        """llama.cpp refuses a -c it cannot honour, but an engine that quietly
        serves a smaller one is the exact fault this feature exists to catch:
        a key issued for 131072 that answers at 32768."""
        new = dict(REC, ctx=131072)
        gw.save_models([new])
        monkeypatch.setattr(gw, "client", FakeClient(
            states=["ready"], props={"n_ctx": 32768}))
        await gw._verify_apply([new], {"m1": REC})
        (f,) = gw.get_apply_state()["failures"]
        assert "only 32768" in f["why"]

    async def test_an_auto_sized_model_is_not_held_to_a_number(
            self, monkeypatch, registry, nosleep):
        """ctx 0 means "size it from the hardware", so whatever it picks is by
        definition what was asked for."""
        auto = dict(REC, ctx=0)
        gw.save_models([auto])
        monkeypatch.setattr(gw, "client", FakeClient(
            states=["ready"], props={"n_ctx": 8192}))
        await gw._verify_apply([auto], {"m1": auto})
        assert gw.get_apply_state()["status"] == "ok"

    async def test_a_new_model_that_will_not_load_is_removed_not_reverted(
            self, monkeypatch, registry, nosleep):
        new = dict(gw.DEFAULT_MODEL_RECORD, id="m2", path="/models/m2.gguf")
        gw.save_models([REC, new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        monkeypatch.setattr(gw, "client",
                            FakeClient(states=[None], health_status=503))
        await gw._verify_apply([new], {"m1": REC})
        (f,) = gw.get_apply_state()["failures"]
        assert f["reverted"] is False
        assert [m["id"] for m in gw.load_models()] == ["m1"]

    async def test_one_bad_model_does_not_discard_the_others(
            self, monkeypatch, registry, nosleep):
        """A second model asking for more memory than the box has is not a
        reason to throw away the operator's other edits."""
        good = dict(REC, id="m1", ctx=16384)
        bad = dict(gw.DEFAULT_MODEL_RECORD, id="m2",
                   path="/models/m2.gguf", ctx=1310720)
        gw.save_models([good, bad])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)

        class TwoModelClient(FakeClient):
            async def get(self, path, **kwargs):
                self.paths.append(path)
                # m1 comes up; m2 is refused by the engine.
                code = 500 if "/m2/" in path else 200

                class R:
                    status_code = code

                    def json(self_inner):
                        if path == "/running":
                            return {"running": [{"model": "m1", "state": "ready"}]}
                        return {"default_generation_settings": {"n_ctx": 16384}}

                    @property
                    def text(self_inner):
                        return "out of memory"

                await asyncio.sleep(0)
                return R()

        monkeypatch.setattr(gw, "client", TwoModelClient(states=[]))
        await gw._verify_apply([good, bad], {"m1": good})
        state = gw.get_apply_state()
        assert state["verified"] == ["m1"]
        assert [f["id"] for f in state["failures"]] == ["m2"]
        assert [m["id"] for m in gw.load_models()] == ["m1"]

    async def test_models_are_verified_one_at_a_time(self, monkeypatch, registry, nosleep):
        """llama-swap runs an exclusive swap group, so a parallel verification
        would have each load evict the one being measured."""
        order: list[str] = []
        a = dict(REC, id="m1")
        b = dict(gw.DEFAULT_MODEL_RECORD, id="m2", path="/models/m2.gguf")
        gw.save_models([a, b])

        class Sequential(FakeClient):
            async def get(self, path, **kwargs):
                await asyncio.sleep(0)
                if path.endswith("/health"):
                    # Readiness comes from the proxied call alone here, so the
                    # order this records is the order the loads were started.
                    order.append("start:" + path.split("/")[2])

                class R:
                    status_code = 200

                    def json(self_inner):
                        if path == "/running":
                            return {"running": []}
                        return {"default_generation_settings": {"n_ctx": 32768}}

                    @property
                    def text(self_inner):
                        return ""

                return R()

        monkeypatch.setattr(gw, "client", Sequential(states=[]))
        await gw._verify_apply([a, b], {"m1": a, "m2": b})
        assert order == ["start:m1", "start:m2"]


class TestTheBannerIsSticky:
    def test_a_failure_survives_a_gateway_restart(self):
        """It is written to the settings table, not held in memory, because a
        failed apply is exactly when someone restarts the gateway."""
        gw.set_apply_state({"at": gw.now(), "status": "failed", "queue": ["m1"],
                            "verified": [], "failures": [{"id": "m1",
                                                          "changes": ["ctx 1 -> 2"],
                                                          "why": "nope"}]})
        assert gw.get_apply_state()["status"] == "failed"
        raw = gw.db_query("SELECT value FROM settings WHERE key=?",
                          (gw.APPLY_KEY,))[0]["value"]
        assert json.loads(raw)["failures"][0]["id"] == "m1"

    def test_a_clean_apply_replaces_it(self):
        gw.set_apply_state({"at": gw.now(), "status": "failed", "queue": [],
                            "verified": [], "failures": [{"id": "m1"}]})
        gw.set_apply_state({"at": gw.now(), "status": "ok", "queue": ["m1"],
                            "verified": ["m1"], "failures": []})
        assert gw.get_apply_state()["status"] == "ok"
        assert gw.get_apply_state()["failures"] == []


class TestTheEndpoint:
    def test_apply_state_is_readable(self, client, admin_headers):
        gw.set_apply_state({"at": gw.now(), "status": "ok", "queue": [],
                            "verified": [], "failures": []})
        r = client.get("/admin/api/models/apply", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_it_needs_an_admin(self, client):
        assert client.get("/admin/api/models/apply").status_code == 401

    def test_the_models_page_carries_it(self, client, admin_headers):
        gw.set_apply_state({"at": gw.now(), "status": "failed", "queue": ["m1"],
                            "verified": [], "failures": [{"id": "m1"}]})
        r = client.get("/admin/api/models", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["apply"]["status"] == "failed"


class TestNeverVerifyAgainstAnEngineThatDidNotRestart:
    """The failure that makes the whole feature worthless if it is missed.

    If the restart is refused, a stale llama-swap may still be up holding the
    OLD config -- and it answers every health check happily. Probing it would
    mark the very save that failed to apply as proven, which is precisely the
    fault this exists to catch, dressed up as a green banner."""

    def test_a_failed_restart_is_reported_and_nothing_is_probed(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        gw.save_models([dict(REC, ctx=32768)])
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: (1, "llama-swap did not come back"))
        started: list = []
        monkeypatch.setattr(gw, "start_verify",
                            lambda q, b: started.append(q) or [])
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [dict(REC, ctx=131072)]})
        assert r.status_code == 200
        assert r.json()["verifying"] == []
        assert started == []          # nothing was probed
        state = gw.get_apply_state()
        assert state["status"] == "failed"
        (f,) = state["failures"]
        assert f["id"] == "m1"
        assert "did not restart" in f["why"]
        assert f["reverted"] is False

    def test_a_clean_restart_does_start_the_queue(
            self, client, admin_headers, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        gw.save_models([dict(REC, ctx=32768)])
        monkeypatch.setattr(gw, "service_control", lambda a, u: (0, "ok"))
        started: list = []
        monkeypatch.setattr(gw, "start_verify",
                            lambda q, b: started.append(q) or ["m1"])
        r = client.put("/admin/api/models", headers=admin_headers,
                       json={"models": [dict(REC, ctx=131072)]})
        assert r.json()["verifying"] == ["m1"]
        assert [m["id"] for m in started[0]] == ["m1"]


class TestOneRestartPerRun:
    @pytest.mark.asyncio
    async def test_two_failures_cost_one_restart_not_two(
            self, monkeypatch, registry, nosleep):
        """Each restart evicts whatever is resident -- including the models
        this same queue has already verified and warmed."""
        a = dict(REC, id="m1", ctx=1310720)
        b = dict(gw.DEFAULT_MODEL_RECORD, id="m2",
                 path="/models/m2.gguf", ctx=1310720)
        gw.save_models([a, b])
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda act, u: restarts.append(act) or (0, "ok"))
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 0.01)
        monkeypatch.setattr(gw, "client",
                            FakeClient(states=[None], health_status=503,
                                       logs="out of memory"))
        await gw._verify_apply([a, b], {"m1": dict(REC), "m2": dict(b, ctx=8192)})
        assert len(gw.get_apply_state()["failures"]) == 2
        assert restarts == ["restart"]
        # ...and both really were put back
        assert sorted((m["id"], m["ctx"]) for m in gw.load_models()) == [
            ("m1", 32768), ("m2", 8192)]


class TestAGatewayRestartMidQueue:
    """A queue that never finished must not leave the banner spinning forever
    while un-verified models stay live in llama-swap.yaml."""

    def test_models_that_never_got_their_turn_are_rolled_back(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        monkeypatch.setattr(gw, "service_control", lambda a, u: (0, "ok"))
        # m1 was verified; m2 was still in flight when the process died.
        gw.save_models([dict(REC, id="m1", ctx=65536),
                        dict(gw.DEFAULT_MODEL_RECORD, id="m2",
                             path="/models/m2.gguf", ctx=1310720)])
        gw.set_apply_state({
            "at": gw.now(), "status": "running", "queue": ["m1", "m2"],
            "verified": ["m1"], "failures": [],
            "before": {"m1": dict(REC, id="m1", ctx=32768),
                       "m2": dict(gw.DEFAULT_MODEL_RECORD, id="m2",
                                  path="/models/m2.gguf", ctx=8192)},
        })
        gw.reconcile_orphaned_apply()
        state = gw.get_apply_state()
        assert state["status"] == "failed"
        assert [f["id"] for f in state["failures"]] == ["m2"]
        assert "restarted mid-verification" in state["failures"][0]["why"]
        by_id = {m["id"]: m for m in gw.load_models()}
        assert by_id["m1"]["ctx"] == 65536      # verified, left alone
        assert by_id["m2"]["ctx"] == 8192       # unproven, rolled back

    def test_a_queue_that_had_finished_everything_just_settles(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
        monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
        restarts: list = []
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: restarts.append(a) or (0, "ok"))
        gw.save_models([dict(REC, ctx=65536)])
        gw.set_apply_state({
            "at": gw.now(), "status": "running", "queue": ["m1"],
            "verified": ["m1"], "failures": [], "before": {"m1": dict(REC)},
        })
        gw.reconcile_orphaned_apply()
        assert gw.get_apply_state()["status"] == "ok"
        assert restarts == []                   # nothing to undo
        assert gw.load_models()[0]["ctx"] == 65536

    def test_a_settled_state_is_left_exactly_as_it_was(self, monkeypatch):
        monkeypatch.setattr(gw, "service_control",
                            lambda a, u: pytest.fail("must not restart"))
        gw.set_apply_state({"at": "then", "status": "failed", "queue": ["m1"],
                            "verified": [], "failures": [{"id": "m1"}]})
        gw.reconcile_orphaned_apply()
        assert gw.get_apply_state()["at"] == "then"


class TestFailFastOnAClientError:
    @pytest.mark.asyncio
    async def test_a_404_is_not_waited_out(self, monkeypatch, registry, nosleep):
        """An id llama-swap does not know is an answer, not a slow load.
        Reporting it as "still not ready after 420s" names the wrong problem."""
        new = dict(REC, ctx=65536)
        gw.save_models([new])
        monkeypatch.setattr(gw, "VERIFY_TIMEOUT", 600)   # would hang if waited
        monkeypatch.setattr(gw, "client",
                            FakeClient(states=[None], health_status=404))
        await gw._verify_apply([new], {"m1": REC})
        (f,) = gw.get_apply_state()["failures"]
        assert "does not recognise this model" in f["why"]
        assert "404" in f["why"]
