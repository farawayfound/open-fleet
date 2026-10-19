"""The voice node on the dashboard: visible where it exists, pausable, and a
pause that stays a pause.

gpu-laptop-1 carries llm-voice (STT/TTS on :8600) beside its gateway. What is
pinned here:

  * GET /admin/api/voice reports the node's units, readiness and any live
    call, and 404s on a box without it;
  * Pause writes the marker BEFORE stopping (so nothing restarts the unit in
    between), refuses a live call unless forced, undoes the marker if the
    stop fails, and is idempotent; Resume removes the marker and starts it;
  * neither does anything without an admin;
  * /health says voice_paused while paused and nothing otherwise, so every
    other box's /health is unchanged and the hub can tell a pause from an
    outage;
  * the Services card lists the voice units only where installed, and calls
    a paused llm-voice "paused";
  * the three moving parts agree: the sudo argv the gateway runs is a line
    hosts/linux/grants.sh writes, and the marker the gateway writes is the
    path llm-voice.service's ConditionPathExists refuses to start under.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import app as gw

REPO = Path(__file__).resolve().parents[2]
UNITS = {"llm-voice": "active", "gpu-laptop-1-voice-guard": "active", "gpu-laptop-1-power": "active"}
HEALTHY = {"stt": {"ready": True}, "tts": {"ready": True}, "session": {"live": False}}


@pytest.fixture
def node(monkeypatch, tmp_path):
    """A box that carries the voice node, with systemctl and the node's own
    /health stubbed. Returns the list of systemctl calls."""
    calls: list[tuple] = []
    state = {"units": dict(UNITS), "health": dict(HEALTHY), "rc": 0}
    monkeypatch.setattr(gw, "VOICE_PAUSED", tmp_path / "voice.paused")
    env = tmp_path / "voice.env"
    env.write_text("VOICE_BIND=100.64.0.9\nVOICE_PORT=8600\nVOICE_API_KEY=not-read-here\n")
    monkeypatch.setattr(gw, "VOICE_ENV", env)
    monkeypatch.setattr(gw, "voice_units", lambda: dict(state["units"]))

    async def _health(base):
        assert base == "http://100.64.0.9:8600"
        return state["health"]
    monkeypatch.setattr(gw, "voice_health", _health)

    def _systemctl(*args):
        # Like systemd: a stop is only safe once the marker is down (or the
        # guard's Wants= could start it again in between), and a start under
        # the marker returns 0 and leaves the unit where it was.
        calls.append(args)
        if state["rc"] != 0:
            return state["rc"], "sudo: a password is required"
        if args[0] == "stop":
            assert gw.VOICE_PAUSED.exists(), "stopped before the marker was written"
            state["units"]["llm-voice"] = "inactive"
        if args[0] == "start" and not gw.VOICE_PAUSED.exists():
            state["units"]["llm-voice"] = "active"
        return 0, ""
    monkeypatch.setattr(gw, "systemctl", _systemctl)
    return {"calls": calls, "state": state}


class TestReport:
    def test_a_box_without_the_node_answers_404(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "voice_units", lambda: {})
        assert client.get("/admin/api/voice", headers=admin_headers).status_code == 404

    def test_it_reports_units_readiness_and_a_live_call(self, client, admin_headers, node):
        node["state"]["health"] = {**HEALTHY, "session": {"live": True}}
        r = client.get("/admin/api/voice", headers=admin_headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["units"], body["ready"], body["live_session"], body["paused"]) \
            == (UNITS, True, True, False)

    def test_needs_an_admin(self, client, node):
        assert client.get("/admin/api/voice").status_code == 401


class TestPause:
    def test_pause_writes_the_marker_then_stops_the_unit(self, client, admin_headers, node):
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        assert r.status_code == 200, r.text
        assert node["calls"] == [("stop", "llm-voice.service")]
        rec = json.loads(gw.VOICE_PAUSED.read_text())
        # Through a token -- which is all this box sees through the hub -- no
        # person is named: "admin-token" would read as one.
        assert rec["since"] and rec["by"] == ""
        assert r.json()["paused"] is True
        assert r.json()["units"]["llm-voice"] == "inactive"

    def test_paused_but_running_is_stopped_again(self, client, admin_headers, node):
        gw.VOICE_PAUSED.write_text("{}")            # e.g. rebooted before the unit had its Condition
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        assert r.status_code == 200 and "already" not in r.json()
        assert node["calls"] == [("stop", "llm-voice.service")]

    def test_pausing_twice_is_a_no_op(self, client, admin_headers, node):
        client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        assert r.status_code == 200 and r.json()["already"] is True
        assert node["calls"] == [("stop", "llm-voice.service")]

    def test_a_live_call_is_refused_and_nothing_changes(self, client, admin_headers, node):
        node["state"]["health"] = {**HEALTHY, "session": {"live": True}}
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        assert r.status_code == 409
        assert (node["calls"], gw.VOICE_PAUSED.exists()) == ([], False)

    def test_force_pauses_through_a_live_call(self, client, admin_headers, node):
        node["state"]["health"] = {**HEALTHY, "session": {"live": True}}
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={"force": True})
        assert r.status_code == 200, r.text
        assert node["calls"] == [("stop", "llm-voice.service")]

    def test_a_failed_stop_says_why_and_takes_the_marker_back(self, client, admin_headers, node):
        node["state"]["rc"] = 1
        r = client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        assert r.status_code == 502 and "grants.sh" in r.text
        assert gw.VOICE_PAUSED.exists() is False

    def test_without_an_admin_nothing_happens(self, client, node):
        assert client.post("/admin/api/voice/pause", json={}).status_code == 401
        assert (node["calls"], gw.VOICE_PAUSED.exists()) == ([], False)

    def test_a_box_without_the_node_answers_404(self, client, admin_headers, monkeypatch, node):
        monkeypatch.setattr(gw, "voice_units", lambda: {})
        assert client.post("/admin/api/voice/pause", headers=admin_headers,
                           json={}).status_code == 404
        assert node["calls"] == []


class TestResume:
    def test_resume_removes_the_marker_then_starts_the_unit(self, client, admin_headers, node):
        client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        r = client.post("/admin/api/voice/resume", headers=admin_headers)
        assert r.status_code == 200, r.text
        assert node["calls"][-1] == ("start", "llm-voice.service")
        assert gw.VOICE_PAUSED.exists() is False and r.json()["paused"] is False

    def test_resume_actually_brings_it_up(self, client, admin_headers, node):
        client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        r = client.post("/admin/api/voice/resume", headers=admin_headers)
        assert r.json()["units"]["llm-voice"] == "active"   # started AFTER the marker went

    def test_a_failed_start_says_so_and_stays_paused(self, client, admin_headers, node):
        client.post("/admin/api/voice/pause", headers=admin_headers, json={})
        node["state"]["rc"] = 1
        r = client.post("/admin/api/voice/resume", headers=admin_headers)
        assert r.status_code == 502 and "could not start" in r.text and "grants.sh" in r.text
        assert gw.VOICE_PAUSED.exists()

    def test_without_an_admin_nothing_happens(self, client, node):
        gw.VOICE_PAUSED.write_text("{}")
        assert client.post("/admin/api/voice/resume").status_code == 401
        assert (node["calls"], gw.VOICE_PAUSED.exists()) == ([], True)


class TestHealth:
    def test_health_says_voice_paused_only_while_paused(self, client, node):
        assert "voice_paused" not in client.get("/health").json()
        gw.VOICE_PAUSED.write_text("{}")
        assert client.get("/health").json()["voice_paused"] is True


class TestEnvFile:
    def test_only_bind_and_port_are_taken(self, node):
        assert gw.voice_base() == "http://100.64.0.9:8600"

    def test_no_env_file_means_no_probe(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "VOICE_ENV", tmp_path / "missing.env")
        assert gw.voice_base() == ""


class TestServicesCard:
    @pytest.fixture
    def linux(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gw, "VOICE_PAUSED", tmp_path / "voice.paused")
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(gw, "engine_info", lambda: {"kind": "llama-swap"})
        monkeypatch.setattr(gw, "_process_states", lambda units: {u: "inactive" for u in units})

    def states(self, monkeypatch, voice):
        base = {"llama-swap": "active", "llm-gateway": "active",
                "cloudflared": "not installed", "tailscaled": "active"}
        monkeypatch.setattr(gw, "_systemd_states",
                            lambda units: {u: {**base, **voice}.get(u, "not installed")
                                           for u in units})
        return gw.service_states()

    def test_voice_units_are_listed_where_installed(self, monkeypatch, linux):
        got = self.states(monkeypatch, {"llm-voice": "active", "gpu-laptop-1-voice-guard": "active"})
        assert got["llm-voice"] == "active" and got["gpu-laptop-1-voice-guard"] == "active"
        assert "gpu-laptop-1-power" not in got
        assert got["cloudflared"] == "not installed"     # base units unchanged

    def test_a_box_without_them_lists_none(self, monkeypatch, linux):
        assert not set(gw.VOICE_UNITS) & set(self.states(monkeypatch, {}))

    def test_a_retired_llm_voice_without_the_guard_is_not_the_voice_node(self, monkeypatch, linux):
        # apu-box-1 keeps its old, disabled llm-voice.service; no guard beside it.
        assert not set(gw.VOICE_UNITS) & set(self.states(monkeypatch, {"llm-voice": "inactive"}))
        monkeypatch.setattr(gw, "_systemd_states",
                            lambda units: {u: ("inactive" if u == "llm-voice" else "not installed")
                                           for u in units})
        assert gw.voice_units() == {}

    def test_a_paused_node_reads_paused(self, monkeypatch, linux):
        gw.VOICE_PAUSED.write_text("{}")
        got = self.states(monkeypatch, {"llm-voice": "inactive", "gpu-laptop-1-voice-guard": "active"})
        assert got["llm-voice"] == "paused"


class TestTheMovingPartsAgree:
    def test_the_sudo_argv_the_gateway_runs_is_granted(self):
        grants = (REPO / "hosts" / "linux" / "grants.sh").read_text()
        for verb in ("stop", "start"):
            assert f"/usr/bin/systemctl {verb} {gw.VOICE_UNIT}.service" in grants
        # ...and only where the node AND its guard are installed.
        assert f"/etc/systemd/system/{gw.VOICE_GUARD}.service" in grants

    def test_the_unit_refuses_to_start_under_the_gateways_marker(self):
        path = REPO / "hosts" / "gpu-laptop-1" / "voice" / "llm-voice.service"
        if not path.exists():
            pytest.skip("the voice node's unit is not part of the public export")
        unit = path.read_text()
        assert f"ConditionPathExists=!/var/lib/llmstack/{gw.VOICE_PAUSED.name}" in unit

    def test_logs_take_the_voice_units(self, client, admin_headers):
        r = client.get("/admin/api/logs", params={"unit": "llm-voice"}, headers=admin_headers)
        assert r.status_code != 400
