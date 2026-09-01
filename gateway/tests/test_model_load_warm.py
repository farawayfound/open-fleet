"""The Overview card's three per-model levers, added 2026-09-01 beside the
125B's ctx_max bump on apu-box-1: unload one model without taking the whole
engine down, load one on demand and watch it arrive, and pick which model a
box keeps warm across a reboot -- without touching the other two.

`_manual_load` is a dict and not a queue on purpose: the swap group only ever
holds one model, so a second load fired while one is already in flight would
just evict the first mid-way. /admin/api/status carries `models_configured`
and `manual_load` so the card can render all three levers from one poll,
which is also pinned here -- a card that has to make a second request to
find out whether persistent/preload flipped is the bug this schema exists to
avoid.

Run with: cd gateway && ./.venv/Scripts/python.exe -m pytest tests/test_model_load_warm.py -q
"""
from __future__ import annotations

from urllib.parse import quote

import pytest
import yaml

import app as gw


def rec(mid: str, **kw) -> dict:
    r = dict(gw.DEFAULT_MODEL_RECORD)
    r.update(id=mid, path="/models/" + mid + ".gguf")
    r.update(kw)
    return r


@pytest.fixture(autouse=True)
def _isolate_load_and_cache_state():
    """This file pokes the one manual-load slot and the two residency caches
    the load/unload endpoints clear on success -- module globals that outlive
    any single test because `client` (the TestClient) is session-scoped."""
    manual_snapshot = dict(gw._manual_load)
    props_snapshot = dict(gw._props_cache)
    running_snapshot = dict(gw._running_cache)
    yield
    gw._manual_load.clear()
    gw._manual_load.update(manual_snapshot)
    gw._props_cache.clear()
    gw._props_cache.update(props_snapshot)
    gw._running_cache.clear()
    gw._running_cache.update(running_snapshot)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """A models.json (and, for the warm tests, llama-swap.yaml) this test
    owns, the way test_model_apply.py's `registry` does."""
    monkeypatch.setattr(gw, "MODELS_JSON", tmp_path / "models.json")
    monkeypatch.setattr(gw, "SWAP_CONFIG", tmp_path / "llama-swap.yaml")
    return tmp_path


class _Resp:
    def __init__(self, status_code: int, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class FakeEngine:
    """gw.client, as much of it as the unload path touches: async get/post,
    recording every call and answering with one canned status/body, the way
    llama-swap would for either route."""

    def __init__(self, status: int = 200, text: str = "ok"):
        self.status = status
        self.text = text
        self.calls: list[tuple] = []

    async def get(self, path, timeout=None):
        self.calls.append(("GET", path))
        return _Resp(self.status, self.text)

    async def post(self, path, timeout=None, json=None):
        self.calls.append(("POST", path, json))
        return _Resp(self.status, self.text)


class _RecordingServiceControl:
    """gw.service_control, recording (action, unit) and answering (rc, out)
    without touching systemctl/schtasks/cron -- the warm endpoint's own
    restart lever."""

    def __init__(self, rc: int = 0, out: str = "ok"):
        self.rc = rc
        self.out = out
        self.calls: list[tuple[str, str]] = []

    def __call__(self, action: str, unit: str) -> tuple[int, str]:
        self.calls.append((action, unit))
        return self.rc, self.out


class TestUnloadOneModel:
    """POST /admin/api/models/unload {"model": id} -- the Overview card's
    per-row button."""

    def test_a_persistent_record_refuses_to_unload(self, registry, client, admin_headers):
        gw.save_models([rec("small", persistent=True)])
        r = client.post("/admin/api/models/unload", headers=admin_headers,
                        json={"model": "small"})
        assert r.status_code == 400
        assert "persistent" in r.json()["detail"]

    def test_a_llama_swap_box_posts_the_quoted_engine_path_and_clears_caches(
            self, registry, client, admin_headers, monkeypatch):
        gw.save_models([rec("big")])
        fake = FakeEngine(status=200)
        monkeypatch.setattr(gw, "client", fake)
        gw._props_cache["big"] = (0.0, {"n_ctx": 8192})
        gw._running_cache["t"] = 12345.0

        r = client.post("/admin/api/models/unload", headers=admin_headers,
                        json={"model": "big"})

        assert r.status_code == 200
        assert r.json() == {"status": 200, "body": "ok", "model": "big"}
        assert fake.calls == [("POST", "/api/models/unload/" + quote("big", safe=""), None)]
        assert gw._props_cache == {}
        assert gw._running_cache["t"] == 0.0

    def test_a_failing_unload_leaves_the_caches_alone(
            self, registry, client, admin_headers, monkeypatch):
        gw.save_models([rec("big")])
        monkeypatch.setattr(gw, "client", FakeEngine(status=404, text="no such model"))
        # _props_cache is a shared module global; an earlier test file may
        # have left entries, and this test asserts exact equality below.
        gw._props_cache.clear()
        gw._props_cache["big"] = (0.0, {"n_ctx": 8192})
        gw._running_cache["t"] = 999.0

        r = client.post("/admin/api/models/unload", headers=admin_headers,
                        json={"model": "big"})

        assert r.status_code == 200
        assert r.json()["status"] == 404
        assert gw._props_cache == {"big": (0.0, {"n_ctx": 8192})}
        assert gw._running_cache["t"] == 999.0

    def test_without_a_body_tries_unload_all_through_the_old_get_route(
            self, client, admin_headers, monkeypatch):
        """No {"model": ...} in the body -- an empty POST fails to decode as
        JSON, which is read the same way it always has been: unload
        everything via llama-swap's own /unload."""
        fake = FakeEngine(status=200)
        monkeypatch.setattr(gw, "client", fake)
        gw._props_cache["stale"] = (0.0, {})
        gw._running_cache["t"] = 55.0

        r = client.post("/admin/api/models/unload", headers=admin_headers)

        assert r.status_code == 200
        assert r.json() == {"status": 200, "body": "ok", "via": "/unload"}
        assert fake.calls == [("GET", "/unload")]
        assert gw._props_cache == {}
        assert gw._running_cache["t"] == 0.0


class TestManualLoadEndpoint:
    """POST /admin/api/models/load {"model": id} -- validation, the 409 that
    keeps two loads from fighting over the one swap slot, and the immediate
    ("started", "loading") answer. The eventual "ok"/"failed" transition is
    proven directly against _manual_load_task below, per the house note that
    a create_task fired mid-request is not reliably observable through
    TestClient."""

    def test_an_empty_body_is_400(self, registry, client, admin_headers):
        r = client.post("/admin/api/models/load", headers=admin_headers, json={})
        assert r.status_code == 400

    def test_a_model_this_box_does_not_serve_is_404(self, registry, client, admin_headers):
        gw.save_models([rec("keeper")])
        r = client.post("/admin/api/models/load", headers=admin_headers,
                        json={"model": "nope"})
        assert r.status_code == 404

    def test_a_disabled_model_is_not_known_either(self, registry, client, admin_headers):
        gw.save_models([rec("off", enabled=False)])
        r = client.post("/admin/api/models/load", headers=admin_headers,
                        json={"model": "off"})
        assert r.status_code == 404

    def test_a_second_load_while_one_is_in_flight_is_409(
            self, registry, client, admin_headers):
        gw.save_models([rec("a"), rec("b")])
        gw._manual_load.clear()
        gw._manual_load.update(model="a", status="loading", why="", at=gw.now())

        r = client.post("/admin/api/models/load", headers=admin_headers,
                        json={"model": "b"})

        assert r.status_code == 409
        assert "a" in r.json()["detail"]

    def test_a_valid_request_starts_the_load_and_answers_immediately(
            self, registry, client, admin_headers, monkeypatch):
        gw.save_models([rec("keeper")])

        async def fake_try_load(mid, want_ctx, job):
            return True, ""

        monkeypatch.setattr(gw, "_try_load", fake_try_load)
        gw._manual_load.clear()

        r = client.post("/admin/api/models/load", headers=admin_headers,
                        json={"model": "keeper"})

        assert r.status_code == 200
        body = r.json()
        assert body["started"] == "keeper"
        # Set synchronously before the background task gets a chance to run
        # (asyncio.create_task never yields the caller's coroutine), so this
        # is deterministic, not a race against the fake load above.
        assert body["load"]["model"] == "keeper"
        assert body["load"]["status"] == "loading"


@pytest.mark.asyncio
class TestManualLoadTaskTransitions:
    """_manual_load_task() run directly, the way the background task the
    endpoint fires would if awaited to completion."""

    async def test_a_successful_try_load_marks_the_slot_ok(self, registry, monkeypatch):
        gw.save_models([rec("keeper")])
        calls = []

        async def fake_try_load(mid, want_ctx, job):
            calls.append((mid, want_ctx))
            return True, ""

        monkeypatch.setattr(gw, "_try_load", fake_try_load)
        gw._manual_load.clear()
        gw._manual_load.update(model="keeper", status="loading", why="", at=gw.now())

        await gw._manual_load_task("keeper")

        assert calls == [("keeper", 0)]
        assert gw._manual_load["status"] == "ok"
        assert gw._manual_load["why"] == ""

    async def test_a_failed_try_load_marks_the_slot_failed_with_the_reason(
            self, registry, monkeypatch):
        gw.save_models([rec("keeper")])

        async def fake_try_load(mid, want_ctx, job):
            return False, "out of memory"

        monkeypatch.setattr(gw, "_try_load", fake_try_load)
        gw._manual_load.clear()
        gw._manual_load.update(model="keeper", status="loading", why="", at=gw.now())

        await gw._manual_load_task("keeper")

        assert gw._manual_load["status"] == "failed"
        assert gw._manual_load["why"] == "out of memory"


class TestWarmStandbyEndpoint:
    """PUT /admin/api/models/warm {"model": id | null} -- the one `preload`
    flag models.json allows, and the restart that makes the choice take
    effect now instead of at the next reboot."""

    def test_an_upstream_box_has_no_start_up_preload(
            self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "UPSTREAM_MODELS", True)
        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": None})
        assert r.status_code == 501

    def test_clearing_to_null_drops_every_non_persistent_preload(
            self, registry, client, admin_headers, monkeypatch):
        svc = _RecordingServiceControl()
        monkeypatch.setattr(gw, "service_control", svc)
        gw.save_models([rec("a", preload=True),
                        rec("b", persistent=True, preload=True),
                        rec("c")])

        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": None})

        assert r.status_code == 200
        assert r.json()["changed"] is True
        assert svc.calls == [("restart", "llama-swap")]
        by_id = {m["id"]: m for m in gw.load_models()}
        assert by_id["a"]["preload"] is False
        assert by_id["c"]["preload"] is False
        # persistent implies preload and is never touched by this endpoint
        assert by_id["b"]["preload"] is True

    def test_setting_one_model_clears_the_others(
            self, registry, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "service_control", _RecordingServiceControl())
        gw.save_models([rec("a", preload=True), rec("b")])

        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "b"})

        assert r.status_code == 200
        assert r.json()["changed"] is True
        by_id = {m["id"]: m for m in gw.load_models()}
        assert by_id["a"]["preload"] is False
        assert by_id["b"]["preload"] is True

    def test_an_unknown_model_is_404(self, registry, client, admin_headers):
        gw.save_models([rec("a")])
        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "nope"})
        assert r.status_code == 404

    def test_a_persistent_target_is_refused(self, registry, client, admin_headers):
        gw.save_models([rec("a", persistent=True)])
        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "a"})
        assert r.status_code == 400

    def test_a_disabled_target_is_refused(self, registry, client, admin_headers):
        gw.save_models([rec("a", enabled=False)])
        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "a"})
        assert r.status_code == 400

    def test_no_change_skips_the_restart(
            self, registry, client, admin_headers, monkeypatch):
        svc = _RecordingServiceControl()
        monkeypatch.setattr(gw, "service_control", svc)
        gw.save_models([rec("a", preload=True)])

        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "a"})

        assert r.status_code == 200
        assert r.json() == {"warm": "a", "changed": False}
        assert svc.calls == []

    def test_a_change_rewrites_both_files_and_restarts_the_engine(
            self, registry, client, admin_headers, monkeypatch):
        svc = _RecordingServiceControl(rc=0, out="restarted")
        monkeypatch.setattr(gw, "service_control", svc)
        gw.save_models([rec("a"), rec("b", preload=True)])

        r = client.put("/admin/api/models/warm", headers=admin_headers,
                       json={"model": "a"})

        assert r.status_code == 200
        assert r.json() == {"warm": "a", "changed": True, "restart_rc": 0,
                            "restart_out": "restarted"}
        assert svc.calls == [("restart", "llama-swap")]
        cfg = yaml.safe_load(gw.SWAP_CONFIG.read_text())
        assert cfg["hooks"]["on_startup"]["preload"] == ["a"]


class TestStatusCarriesLoadAndWarmState:
    """GET /admin/api/status: models_configured and manual_load, the two
    fields the Overview card's picker/standby control read from one poll."""

    def test_models_configured_and_manual_load_are_exposed(
            self, registry, client, admin_headers, monkeypatch):
        async def fake_swap_running():
            return True, []

        monkeypatch.setattr(gw, "swap_running", fake_swap_running)
        monkeypatch.setattr(gw, "host_status", lambda: {})
        monkeypatch.setattr(gw, "service_states", lambda: {})
        gw.save_models([rec("a", preload=True),
                        rec("b", persistent=True, preload=True),
                        rec("off", enabled=False)])
        gw._manual_load.clear()
        gw._manual_load.update(model="a", status="ok", why="", at="2026-09-01T00:00:00Z")

        r = client.get("/admin/api/status", headers=admin_headers)

        assert r.status_code == 200
        body = r.json()
        # "off" is disabled and must not appear at all
        assert body["models_configured"] == [
            {"id": "a", "preload": True, "persistent": False},
            {"id": "b", "preload": True, "persistent": True},
        ]
        assert body["manual_load"] == dict(gw._manual_load)
