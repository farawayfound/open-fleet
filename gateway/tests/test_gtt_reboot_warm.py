"""Served-models report, GTT staging + reboot endpoints, and the warm-up
button (mail, page, plan_warm, and the better-host heuristic behind it).

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import time
from types import SimpleNamespace

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolate_state():
    snapshot = dict(gw._routes_cache)
    gw._inflight.clear()
    gw._host_cooldown.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._inflight.clear()
    gw._host_cooldown.clear()


class _FakeAuthRequest:
    """Just enough of a Request for require_api_key()/bearer() -- .url.path
    included, since the endpoints a request budget does not COUNT are also the
    ones it must not BLOCK."""

    def __init__(self, token: str, path: str = "/v1/chat/completions"):
        self.headers = {"authorization": "Bearer " + token}
        self.client = type("C", (), {"host": "127.0.0.1"})()
        self.query_params = {}
        self.url = SimpleNamespace(path=path)


# ---------------------------------------------------------------------------
# served-models report
# ---------------------------------------------------------------------------

class TestServedModelsReport:
    def test_report_includes_meta_bytes_fit_source_and_engine(self, client, admin_headers):
        model_dir = gw.MODELS_DIR / "acme__demo-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "demo-model.Q4_K_M.gguf"
        model_path.write_bytes(b"not actually a gguf header")
        models = gw.load_models()
        rec = dict(gw.DEFAULT_MODEL_RECORD)
        rec.update(id="served-report-test", path=str(model_path), enabled=True)
        models.append(rec)
        gw.save_models(models)
        try:
            r = client.get("/admin/api/served-models", headers=admin_headers)
            assert r.status_code == 200, r.text
            body = r.json()
            assert "served-report-test" in body["models"]
            meta = body["meta"]["served-report-test"]
            assert meta["bytes"] == model_path.stat().st_size
            # gguf_meta() returns {} for a file with no GGUF header, so
            # expert_count is never read and moe stays None (not False).
            assert meta["moe"] is None
            assert meta["source"] == {"repo": "acme/demo-model",
                                      "file": "demo-model.Q4_K_M.gguf"}
            assert isinstance(meta["fit"], str)
            assert isinstance(body["engine"], str)
        finally:
            gw.save_models([m for m in gw.load_models() if m.get("id") != "served-report-test"])

    def test_gguf_meta_empty_and_model_source_maps_paths(self, tmp_path):
        p = tmp_path / "fake.gguf"
        p.write_bytes(b"definitely not a gguf header")
        assert gw.gguf_meta(str(p)) == {}
        assert gw.gguf_meta("") == {}

        mp = gw.MODELS_DIR / "acme__demo-model" / "demo-model.Q4_K_M.gguf"
        assert gw.model_source(str(mp)) == {"repo": "acme/demo-model",
                                            "file": "demo-model.Q4_K_M.gguf"}
        assert gw.model_source("/some/other/place/file.gguf") is None
        assert gw.model_source("") is None

    async def test_peer_served_parses_meta_and_engine(self, monkeypatch):
        async def fake_get(self, url, headers=None, **kw):
            return httpx.Response(200, request=httpx.Request("GET", url), json={
                "models": ["m1"], "running": ["m1"], "capacity": {"m1": 2},
                "ctx": {"m1": 16384},
                "meta": {"m1": {"bytes": 123, "fit": "vram", "moe": False,
                                "source": {"repo": "a/b", "file": "c.gguf"}}},
                "engine": "llama-swap",
            })

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        out = await gw._peer_served({"name": "peerX", "url": "http://peerx:8080", "token": "t"})
        assert out["online"] is True
        assert out["models"] == {"m1"}
        assert out["running"] == {"m1"}
        assert out["capacity"] == {"m1": 2}
        assert out["ctx"] == {"m1": 16384}
        assert out["meta"]["m1"]["fit"] == "vram"
        assert out["engine"] == "llama-swap"


# ---------------------------------------------------------------------------
# GTT: gtt_targets() and _gtt_info()
# ---------------------------------------------------------------------------

class TestGttTargetsAndInfo:
    def test_gtt_targets_for_various_ram_sizes(self):
        GIB = 1024 ** 3
        assert gw.gtt_targets(int(31 * GIB)) == (28, 26)
        assert gw.gtt_targets(int(18.8 * GIB)) == (15, 15)
        assert gw.gtt_targets(int(16 * GIB)) == (13, 13)
        assert gw.gtt_targets(int(4 * GIB)) == (1, 1)
        assert gw.gtt_targets(0) == (0, 0)

    @pytest.mark.skipif(platform.system() != "Darwin",
                        reason="reads the real box: only a Mac answers 'macos' here")
    def test_gtt_info_on_darwin_reports_macos_supported(self):
        info = gw._gtt_info()
        assert info["platform"] == "macos"
        assert info["supported"] is True
        for key in ("live", "live_gib", "ram_gib", "max_gib", "suggested_gib",
                   "staged_gib", "mechanism", "active", "reboot_required",
                   "warnings", "kernels", "helper_installed"):
            assert key in info

    def test_gtt_info_windows_is_unsupported_with_reason(self, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Windows")
        info = gw._gtt_info()
        assert info["platform"] == "windows"
        assert info["supported"] is False
        assert info["reason"]

    def test_gtt_info_linux_no_sysfs_reports_platform_none(self, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        info = gw._gtt_info()
        assert info["platform"] == "none"
        assert info["supported"] is False

    def test_gtt_info_linux_receipt_active_and_reboot_required(self, monkeypatch, tmp_path):
        receipt_path = tmp_path / "gtt.json"
        receipt = {"gib": 24, "mechanism": "modprobe", "warnings": [],
                  "staged_at": "now", "kernels": ["6.9.1"],
                  "running_kernel": "6.9.1", "cmdline_token": "llmstack_gtt=24"}
        receipt_path.write_text(json.dumps(receipt))
        monkeypatch.setattr(gw, "GTT_RECEIPTS", (receipt_path,))
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")

        orig_read_text = pathlib.Path.read_text

        def read_text_with_token(self, *a, **kw):
            if str(self) == "/proc/cmdline":
                return "BOOT_IMAGE=/vmlinuz root=/dev/sda1 llmstack_gtt=24"
            return orig_read_text(self, *a, **kw)

        monkeypatch.setattr(pathlib.Path, "read_text", read_text_with_token)
        info = gw._gtt_info()
        assert info["staged_gib"] == 24
        assert info["mechanism"] == "modprobe"
        assert info["active"] is True
        assert info["reboot_required"] is False

        def read_text_without_token(self, *a, **kw):
            if str(self) == "/proc/cmdline":
                return "BOOT_IMAGE=/vmlinuz root=/dev/sda1"
            return orig_read_text(self, *a, **kw)

        monkeypatch.setattr(pathlib.Path, "read_text", read_text_without_token)
        info2 = gw._gtt_info()
        assert info2["active"] is False
        assert info2["reboot_required"] is True


# ---------------------------------------------------------------------------
# POST /admin/api/gpu
# ---------------------------------------------------------------------------

_FAKE_GPU_INFO = {
    "platform": "linux-amdgpu", "supported": True, "reason": None,
    "helper_installed": True, "grant_ok": True,
    "live": {"gtt_total": None, "vram_total": None}, "live_gib": None,
    "ram_gib": 64.0, "max_gib": 60, "suggested_gib": 50, "staged_gib": None,
    "mechanism": None, "active": False, "reboot_required": False, "warnings": [],
    "staged_at": None, "kernels": [], "running_kernel": None,
    "staged_conf": "", "staged_gtt_mib": None,
}


class TestApiGpuSet:
    def test_bad_and_out_of_range_input_400(self, client, admin_headers):
        r = client.post("/admin/api/gpu", headers=admin_headers, json={"gtt_gb": "banana"})
        assert r.status_code == 400

        r2 = client.post("/admin/api/gpu", headers=admin_headers, json={"gtt_gb": 9999})
        assert r2.status_code == 400
        assert "GiB on this box" in r2.text

    def test_helper_rc2_is_400(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "_gtt_info", lambda: dict(_FAKE_GPU_INFO))

        def fake_run(args, **kw):
            return subprocess.CompletedProcess(args, 2, stdout="", stderr="rejected: bad value")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        r = client.post("/admin/api/gpu", headers=admin_headers, json={"gtt_gb": 24})
        assert r.status_code == 400

    def test_helper_rc3_is_501(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "_gtt_info", lambda: dict(_FAKE_GPU_INFO))

        def fake_run(args, **kw):
            return subprocess.CompletedProcess(args, 3, stdout="", stderr="needs root")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        r = client.post("/admin/api/gpu", headers=admin_headers, json={"gtt_gb": 24})
        assert r.status_code == 501

    def test_helper_rc0_is_200_with_staged_and_out(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "_gtt_info", lambda: dict(_FAKE_GPU_INFO))

        def fake_run(args, **kw):
            return subprocess.CompletedProcess(
                args, 0, stdout='{"gib": 24, "mechanism": "modprobe"}\n', stderr="")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        r = client.post("/admin/api/gpu", headers=admin_headers, json={"gtt_gb": 24})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["staged"] == 24
        assert "out" in body


# ---------------------------------------------------------------------------
# reboot
# ---------------------------------------------------------------------------

class TestReboot:
    def test_reboot_info_endpoint_and_missing_confirm_400(self, client, admin_headers):
        r = client.get("/admin/api/reboot", headers=admin_headers)
        assert r.status_code == 200
        body = r.json()
        assert "supported" in body and "method" in body and "reason" in body

        r2 = client.post("/admin/api/reboot", headers=admin_headers, json={})
        assert r2.status_code == 400

    def test_reboot_confirmed_systemd_runs_systemctl(self, client, admin_headers, monkeypatch):
        monkeypatch.setattr(gw, "reboot_support", lambda: {
            "supported": True, "method": "systemd", "reason": None, "host": gw.HOST_NAME})
        captured: dict = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd

            class _P:
                pass
            return _P()

        monkeypatch.setattr(gw.subprocess, "Popen", fake_popen)
        r = client.post("/admin/api/reboot", headers=admin_headers, json={"confirm": True})
        assert r.status_code == 200, r.text
        assert "systemctl reboot" in " ".join(captured["cmd"])


# ---------------------------------------------------------------------------
# warm-up: mail, page, plan_warm, _warm_better_host
# ---------------------------------------------------------------------------

class TestWarmUpMail:
    def test_key_mail_has_warm_token_link_and_ordering(
            self, client, intake_headers, captured_mail, fake_fleet):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "warmmail@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200, r.text
        mail = captured_mail[-1]
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?", ("warmmail@nasa.gov",))[0]
        assert row["warm_token"]
        assert ("/public/warm/" + row["warm_token"]) in mail["html"]
        assert "Load my model now" in mail["html"]

        text = mail["text"]
        assert text.index("-- load it now --") < text.index("-- your key --")

        html = mail["html"]
        assert html.index('<a href="') < html.index("<h3>Your key</h3>")

    def test_warm_link_edge_cases(self, monkeypatch):
        row = {"warm_token": "a" * 24}

        settings = dict(gw.get_public_settings())
        settings["warm_button"] = False
        assert gw.warm_link(row, settings) == ""

        settings["warm_button"] = True
        settings["public_base_url"] = ""
        monkeypatch.setattr(gw, "PUBLIC_API_URL", "")
        assert gw.warm_link(row, settings) == ""

        settings["public_base_url"] = "https://example.test/v1"
        link = gw.warm_link(row, settings)
        assert link == "https://example.test/public/warm/" + row["warm_token"]


class TestWarmUpPage:
    def test_warm_page_200_contains_token(
            self, client, fake_fleet, intake_headers, captured_mail):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "warmpage@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?", ("warmpage@nasa.gov",))[0]
        resp = client.get("/public/warm/" + row["warm_token"])
        assert resp.status_code == 200
        assert row["warm_token"] in resp.text

    def test_warm_page_unknown_token_404(self, client):
        resp = client.get("/public/warm/" + "x" * 24)
        assert resp.status_code == 404


class TestWarmUpStart:
    def test_returns_items_with_box_only_and_logs_event(
            self, client, fake_fleet, intake_headers, captured_mail, monkeypatch):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "warmstart1@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("warmstart1@nasa.gov",))[0]
        token = row["warm_token"]
        fake_items = [{"role": "primary", "public_id": "gemma4-31b-qat", "name": "Gemma",
                      "phase": "ready", "detail": "", "progress": 1.0, "box": "Box 1",
                      "action": "load", "cand": "", "fleet_id": "gemma4-31b-qat",
                      "host": gw.HOST_NAME}]

        async def fake_plan_warm(row_):
            return fake_items

        async def fake_warm_run(tok):
            return None

        monkeypatch.setattr(gw, "plan_warm", fake_plan_warm)
        monkeypatch.setattr(gw, "_warm_run", fake_warm_run)

        resp = client.post("/public/warm/" + token + "/start")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        item = body["items"][0]
        assert item["box"] == "Box 1"
        assert set(item) == {"name", "role", "box", "phase", "detail", "progress"}

        events = gw.db_query(
            "SELECT * FROM public_events WHERE kind='warm' AND email=?",
            ("warmstart1@nasa.gov",))
        assert len(events) == 1

    def test_second_call_while_running_says_already_in_progress(
            self, client, fake_fleet, intake_headers, captured_mail):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "warmstart2@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("warmstart2@nasa.gov",))[0]
        token = row["warm_token"]
        key_row = gw.db_query("SELECT * FROM api_keys WHERE id=?", (row["key_id"],))[0]
        gw._warm_jobs[token] = {"items": [], "key": key_row, "started": time.time(),
                                "done": False, "finished": None}
        resp = client.post("/public/warm/" + token + "/start")
        assert resp.status_code == 200
        assert resp.json()["message"] == "already in progress"

    def test_revoked_key_403(self, client, fake_fleet, intake_headers, captured_mail):
        r = client.post(
            "/public/api/request", headers=intake_headers,
            json={"email": "warmrevoked@nasa.gov", "kind": "single",
                 "model": "gemma4-31b-qat", "ctx": 8192, "accept_terms": True},
        )
        assert r.status_code == 200
        row = gw.db_query("SELECT * FROM public_keys WHERE email=?",
                          ("warmrevoked@nasa.gov",))[0]
        gw.db_exec("UPDATE api_keys SET archived_at=? WHERE id=?", (gw.now(), row["key_id"]))
        resp = client.post("/public/warm/" + row["warm_token"] + "/start")
        assert resp.status_code == 403


class TestPlanWarmAndBetterHost:
    async def test_plan_warm_team_primary_and_worker_land_on_different_hosts(self, fake_fleet):
        gw._routes_cache.update(
            cands={"gemma4-31b-qat": ["hostX", "hostY"],
                  "gemma-4-26b": ["hostX", "hostY"], "gemma4:26b": ["hostX", "hostY"]},
            running={}, cap={}, meta={}, reachable={"hostX", "hostY"},
        )
        row = {"kind": "team",
              "models": json.dumps({"primary": "gemma4-31b-qat",
                                    "workers": ["gemma4-26b-a4b"]})}
        items = await gw.plan_warm(row)
        assert items[0]["host"] == "hostX"
        assert items[1]["host"] == "hostY"

    def _set_ubserver_serving_fida(self):
        gw._routes_cache.update(
            cands={"fidA": ["server-1"]},
            meta={("server-1", "fidA"): {"bytes": 5 * gw.GIB, "moe": False,
                                        "source": {"repo": "a/b", "file": "c.gguf"}}},
            engine={"apu-box-1": "llama-swap"},
            reachable={"apu-box-1", "server-1"},
        )
        # 'small' class, role='primary' -> tier 2, which is what apu-box-1
        # (tier 1, 'big') should be able to beat.
        return gw.host_tier("server-1", "fidA", "primary")[0]

    def test_warm_better_host_returns_ai_max(self, fake_fleet):
        current_tier = self._set_ubserver_serving_fida()
        better = gw._warm_better_host(current_tier, ["fidA"], "primary", set())
        assert better is not None
        assert better["host"] == "apu-box-1"

    def test_warm_better_host_none_variants(self, fake_fleet):
        current_tier = self._set_ubserver_serving_fida()

        # apu-box-1 has a request in flight -- left alone rather than restarted.
        gw._inflight["apu-box-1"] = 1
        assert gw._warm_better_host(current_tier, ["fidA"], "primary", set()) is None
        gw._inflight["apu-box-1"] = 0

        # apu-box-1 runs Ollama; the source is a GGUF -> engine mismatch.
        gw._routes_cache["engine"] = {"apu-box-1": "ollama"}
        assert gw._warm_better_host(current_tier, ["fidA"], "primary", set()) is None

        # apu-box-1 already serves the model -- nothing to provision.
        gw._routes_cache["engine"] = {"apu-box-1": "llama-swap"}
        gw._routes_cache["cands"] = {"fidA": ["server-1", "apu-box-1"]}
        assert gw._warm_better_host(current_tier, ["fidA"], "primary", set()) is None


class TestBudgetExcludesWarm:
    def test_budget_req_sql_excludes_warm_endpoint(self):
        raw, meta = gw.mint_key("budget-warm-test", max_rpd=1)
        gw.db_exec(
            "INSERT INTO usage(ts,key_id,model,endpoint,stream,status,prompt_tokens,"
            "completion_tokens,total_tokens) VALUES (?,?,?,?,?,?,?,?,?)",
            (gw.now(), meta["id"], "m", "/v1/warm", 0, 200, 10, 5, 15),
        )
        key = gw.require_api_key(_FakeAuthRequest(raw))
        assert key["id"] == meta["id"]


class TestGpuconfArgv:
    """How the gateway reaches the GPU-memory helper as root.

    The helper must run OUTSIDE this unit's mount namespace -- the gateway's
    own unit sets ProtectSystem=full, so an inherited namespace gives the
    helper a read-only /etc and /boot and it cannot write the bootloader
    config. These pin the escape mechanism, and pin it to the sudoers grant
    that hosts/linux/grants.sh actually writes, so the two cannot drift apart
    silently -- a mismatch there is invisible until a box says "grant missing".
    """

    def test_linux_escapes_the_mount_namespace_via_nsenter(self, monkeypatch):
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw.Path, "exists", lambda self: True)
        argv = gw.gpuconf_argv("19")
        assert argv[:2] == ["sudo", "-n"]
        assert argv[2] == gw.NSENTER
        # -t 1 -m is the whole point: target PID 1, mount namespace only.
        assert argv[3:7] == ["-t", "1", "-m", "--"]
        assert argv[-2:] == [gw.GPUCONF_HELPER, "19"]

    def test_never_uses_pipe_bearing_systemd_run(self, monkeypatch):
        """--pipe passes file descriptors inside the StartTransientUnit D-Bus
        message; dbus-broker (Fedora) resets the connection on it, while the
        same call succeeds on apu-box-1's Ubuntu. Whatever this returns, it must
        not depend on descriptor passing."""
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw.Path, "exists", lambda self: True)
        assert "--pipe" not in gw.gpuconf_argv("auto")
        # --scope would be the other tempting fix and is wrong for a different
        # reason: a scope runs as a child of the caller, inheriting its
        # namespace, so it never escapes ProtectSystem at all.
        assert "--scope" not in gw.gpuconf_argv("auto")

    def test_falls_back_to_plain_sudo_without_nsenter(self, monkeypatch):
        """A Mac has no mount namespace to escape, and any Linux box missing
        util-linux still gets a call that works rather than one that errors."""
        monkeypatch.setattr(gw.platform, "system", lambda: "Darwin")
        assert gw.gpuconf_argv("auto") == ["sudo", "-n", gw.GPUCONF_HELPER, "auto"]

    def test_argv_is_covered_by_the_sudoers_grant_we_ship(self, monkeypatch):
        """The grant pins every argument up to the helper's own. If gpuconf_argv
        changes shape, grants.sh has to change with it or every box reports the
        helper as ungranted."""
        grants = (pathlib.Path(__file__).resolve().parents[2]
                  / "hosts" / "linux" / "grants.sh").read_text()
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        monkeypatch.setattr(gw.Path, "exists", lambda self: True)
        argv = gw.gpuconf_argv("19")
        # Rebuild the line grants.sh writes, with its $GPUCONF_DST expanded.
        pinned = " ".join(argv[2:-1]).replace(gw.GPUCONF_HELPER, "$GPUCONF_DST")
        assert pinned + " *," in grants, (
            "sudoers grant does not cover " + pinned)
