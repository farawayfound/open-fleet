"""A fleet node whose model runs on a device attached to its box.

The first is an NPU appliance on a Windows PC's USB-C: the gateway runs on
the PC and fronts the device's own engine, so everything the gateway can
measure about "its" hardware -- the PC's 11 GB, an integrated GPU, the PC's
network -- is about the wrong machine. The spec sheet says so (`hw: spec`),
and what is pinned here is that nobody downstream draws the PC as the node:

  * the public card (example.org/fleet reads /public/api/overview) shows
    the sheet and what the node serves, and no live memory, VRAM or CPU
    figure, while a normal online box still shows its live ones;
  * NPU compute is published beside the GPU figures, never as one of them;
  * the engine kind reads "vendor", not "openai" -- a public page must not
    suggest OpenAI's service is involved;
  * the node joins Not in use: a NEW peer may arrive `routed: false` through
    the ordinary peers PUT, while an existing peer's flag stays the
    Configurations tab's alone.
"""
from __future__ import annotations

import json

import pytest

import app as gw

NODE = "npu-box-1"
SHEET = gw.DEFAULT_SPECS[NODE]

PC_STATUS = {
    "host": {"cpu_count": 12, "cpu_percent": 40,
             "mem": {"total": 11_400_000_000, "used": 9_000_000_000},
             "gpu": [{"vram_total": 2_000_000_000, "vram_used": 1_000_000_000,
                      "busy_percent": 80}],
             "engine": {"kind": "openai"}},
    "models_running": [{"id": "glm-4.7-flash", "n_ctx": 131072}],
}


class TestTheSheet:
    def test_the_node_is_reserve_and_described_by_its_sheet(self):
        assert SHEET["reserve"] is True and SHEET["hw"] == "spec"

    def test_its_80_gb_are_counted_once(self):
        assert SHEET["ram_gb"] + SHEET["vram_gb"] == 80

    def test_npu_tops_never_pose_as_gpu_tflops(self):
        assert SHEET["npu_tops"] == 190 and not SHEET.get("gpu_tflops")


class TestPublicCard:
    @pytest.fixture(autouse=True)
    def _specs(self, monkeypatch):
        specs = {NODE: SHEET, "gpu-box": {"cpu": "CPU", "gpu": "GPU", "ram_gb": 64}}
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        self.specs = specs

    def card(self, name, status):
        return gw._sanitize_public_host(
            {"name": name, "online": True, "status": status}, self.specs)

    def test_an_attached_device_node_shows_its_sheet_not_the_pc(self):
        card = self.card(NODE, PC_STATUS)
        assert {k: card[k] for k in ("mem", "vram", "cpu_count", "cpu_percent", "gpu_busy")} \
            == dict.fromkeys(("mem", "vram", "cpu_count", "cpu_percent", "gpu_busy"))
        assert card["specs"]["ram_gb"] == 32 and card["specs"]["vram_gb"] == 48
        assert card["specs"]["npu_tops"] == 190
        assert card["online"] is True
        assert [s["n_ctx"] for s in card["serving"]] == [131072]

    def test_its_engine_is_a_vendor_engine_in_public(self):
        assert self.card(NODE, PC_STATUS)["engine"] == "vendor"

    def test_an_ordinary_online_box_keeps_its_live_figures(self):
        card = self.card("gpu-box", PC_STATUS)
        assert card["mem"] == {"total": 11_400_000_000, "used": 9_000_000_000}
        assert card["cpu_count"] == 12
        assert card["specs"]["npu_tops"] is None


class TestJoiningNotInUse:
    @pytest.fixture(autouse=True)
    def _peers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gw, "PEERS_PATH", tmp_path / "peers.json")
        (tmp_path / "peers.json").write_text(json.dumps([
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": "t1",
             "api_url": "", "inference_key": "", "routed": True}]))

    def put(self, client, admin_headers, entries):
        r = client.put("/admin/api/peers", headers=admin_headers, json={"peers": entries})
        assert r.status_code == 200, r.text
        return {p["name"]: p["routed"] for p in gw.load_peers()}

    def test_a_new_peer_can_join_not_in_use(self, client, admin_headers):
        routed = self.put(client, admin_headers, [
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": ""},
            {"name": NODE, "url": "http://100.64.0.2:8080", "token": "t2", "routed": False}])
        assert routed == {"old-box": True, NODE: False}

    def test_a_new_peer_that_says_nothing_is_in_use_as_before(self, client, admin_headers):
        routed = self.put(client, admin_headers, [
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": ""},
            {"name": NODE, "url": "http://100.64.0.2:8080", "token": "t2"}])
        assert routed[NODE] is True

    def test_an_existing_peer_cannot_be_taken_out_of_use_by_the_put(self, client, admin_headers):
        routed = self.put(client, admin_headers, [
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": "", "routed": False}])
        assert routed == {"old-box": True}

    def test_only_a_real_false_takes_a_new_peer_out(self, client, admin_headers):
        routed = self.put(client, admin_headers, [
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": ""},
            {"name": NODE, "url": "http://100.64.0.2:8080", "token": "t2", "routed": "false"}])
        assert routed[NODE] is True

    def test_a_configurations_tab_opened_before_it_joined_cannot_put_it_to_work(
            self, client, admin_headers):
        # The tab saves the order and not-in-use lists it loaded; a peer that
        # registered since is in neither, and keeps the flag it joined with.
        self.put(client, admin_headers, [
            {"name": "old-box", "url": "http://100.64.0.1:8080", "token": ""},
            {"name": NODE, "url": "http://100.64.0.2:8080", "token": "t2", "routed": False}])
        r = client.put("/admin/api/fleet-config", headers=admin_headers,
                       json={"order": ["old-box"], "not_in_use": [], "reserve": {}})
        assert r.status_code == 200, r.text
        assert {p["name"]: p["routed"] for p in gw.load_peers()} == {"old-box": True, NODE: False}
