"""The availability gate — a box that is somebody's daily driver first.

Every assertion here is about ONE decision: does this host advertise its
models to the hub right now. The gate deliberately does not touch /v1 (see
availability() for why refusing in-flight work is worse than doing it), so
these tests also pin down what it must NOT close.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import app as gw


@pytest.fixture
def gated(monkeypatch, tmp_path):
    """Turn the gate on for one test, pointed at a scratch verdict file.

    The real module reads LLMSTACK_AVAILABILITY_FILE at import time, which is
    the right shape for a process that is configured once at boot and the
    wrong shape for a test, so the constants are patched directly."""
    verdict = tmp_path / "availability.json"
    monkeypatch.setattr(gw, "AVAILABILITY_FILE", verdict)
    monkeypatch.setattr(gw, "AVAILABILITY_GATED", True)
    monkeypatch.setattr(gw, "AVAILABILITY_MAX_AGE", 120.0)

    def write(available: bool, reason: str = "", age_s: float = 0.0) -> Path:
        verdict.write_text(json.dumps({
            "available": available, "reason": reason, "ts": time.time() - age_s,
        }))
        gw._avail_cache.update(t=0.0, state=None)  # defeat the 5 s cache
        return verdict

    write.path = verdict  # type: ignore[attr-defined]
    yield write
    gw._avail_cache.update(t=0.0, state=None)


@pytest.fixture
def one_model(monkeypatch):
    """A host with exactly one model in its registry, and nothing else."""
    rec = {"id": "qwopus3.6-35b-coder", "path": "/models/q.gguf", "enabled": True,
           "parallel": 1, "aliases": ["qwopus"], "ctx": 32768}
    monkeypatch.setattr(gw, "load_models", lambda: [rec])
    monkeypatch.setattr(gw, "local_model_ctx", lambda: {rec["id"]: 32768})

    async def _running():
        return {rec["id"]}

    monkeypatch.setattr(gw, "upstream_running_ids", _running)
    return rec


class TestUngated:
    """The other eleven boxes in the fleet must not notice this feature."""

    def test_default_is_ungated_and_available(self):
        av = gw.availability()
        assert av == {"gated": False, "available": True, "reason": "", "age_s": None}

    def test_health_says_nothing_extra(self, client):
        body = client.get("/health").json()
        assert "available" not in body
        assert "availability_reason" not in body

    def test_served_models_is_unfiltered(self, client, admin_headers, one_model):
        body = client.get("/admin/api/served-models", headers=admin_headers).json()
        assert "qwopus3.6-35b-coder" in body["models"]
        assert "unavailable" not in body


class TestGate:
    def test_available_verdict_advertises_normally(
        self, client, admin_headers, gated, one_model
    ):
        gated(True, "idle 14m, display asleep")
        body = client.get("/admin/api/served-models", headers=admin_headers).json()
        assert "qwopus3.6-35b-coder" in body["models"]
        assert body["running"] == ["qwopus3.6-35b-coder"]
        assert body["capacity"]["qwopus3.6-35b-coder"] == 1
        assert body["ctx"]["qwopus3.6-35b-coder"] == 32768
        assert "unavailable" not in body

    def test_unavailable_verdict_advertises_nothing(
        self, client, admin_headers, gated, one_model
    ):
        gated(False, "user active 3s ago")
        body = client.get("/admin/api/served-models", headers=admin_headers).json()
        assert body["models"] == []
        assert body["running"] == []
        assert body["capacity"] == {}
        assert body["ctx"] == {}
        assert body["unavailable"] == "user active 3s ago"

    def test_the_hub_reads_that_as_no_route(self, gated, one_model):
        """The contract that actually matters: _peer_served() is how the hub
        builds its routing table, and an empty catalogue takes this box out of
        it without the hub needing to know why."""
        gated(False, "user active 3s ago")
        body = {"models": [], "running": [], "capacity": {}, "ctx": {},
                "unavailable": "user active 3s ago"}
        assert not {str(m).strip() for m in body.get("models") or []}


class TestFailClosed:
    """A dead watchdog must not leave the fleet running weights on top of
    whatever the owner is playing. Every one of these reads as unavailable —
    and every one says so out loud rather than looking like an idle box."""

    def test_missing_file(self, gated):
        gated.path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        gw._avail_cache.update(t=0.0, state=None)
        av = gw.availability()
        assert av["available"] is False
        assert "no watchdog verdict" in av["reason"]

    def test_stale_verdict(self, gated):
        gated(True, "idle 14m", age_s=600)
        av = gw.availability()
        assert av["available"] is False
        assert "600s old" in av["reason"] and "limit 120s" in av["reason"]
        assert av["age_s"] >= 600

    def test_fresh_verdict_inside_the_window_is_honoured(self, gated):
        gated(True, "idle 14m", age_s=30)
        assert gw.availability()["available"] is True

    def test_unparseable_verdict(self, gated):
        gated.path.write_text("{not json")  # type: ignore[attr-defined]
        gw._avail_cache.update(t=0.0, state=None)
        av = gw.availability()
        assert av["available"] is False
        assert "unreadable" in av["reason"]

    def test_verdict_without_a_timestamp(self, gated):
        gated.path.write_text(json.dumps({"available": True}))  # type: ignore[attr-defined]
        gw._avail_cache.update(t=0.0, state=None)
        av = gw.availability()
        assert av["available"] is False
        assert "no timestamp" in av["reason"]


class TestVisibility:
    """Failing closed is only acceptable because it is never silent."""

    def test_health_reports_the_gate(self, client, gated):
        gated(False, "user active 3s ago")
        body = client.get("/health").json()
        assert body["status"] == "ok"  # the gateway itself is fine
        assert body["available"] is False
        assert body["availability_reason"] == "user active 3s ago"

    def test_host_status_carries_it_to_the_dashboard(self, gated):
        gated(False, "game running: gpu 87%")
        av = gw.host_status()["availability"]
        assert av["gated"] is True
        assert av["available"] is False
        assert av["reason"] == "game running: gpu 87%"


class TestServingIsNotGated:
    """The gate closes advertisement, not the door. A request the hub already
    dispatched — its routing table is up to ROUTE_TTL stale — has to be
    answered, because the proxy only fails over on connect errors."""

    def test_local_catalogue_still_resolves_while_unavailable(
        self, gated, one_model
    ):
        gated(False, "user active 3s ago")
        assert "qwopus3.6-35b-coder" in gw.local_model_ids()
        assert gw.local_capacity()["qwopus3.6-35b-coder"] == 1
