"""Regressions for the defects an adversarial review of this change found.

Each test here is a bug that shipped in the first draft and was fixed. They
are grouped in one file because what they have in common is how they failed:
every one was silent. A model classed as fitting a card it actually spills
out of, a cooldown that never engages, a 413 that blames the prompt for an
upstream's 500, an explicit rank of 0 read as "unranked" -- none of these
raise, they just route (or throttle) wrongly, which is exactly the kind of
thing that survives to production unless a test pins it.

Run with: $SP/venv/bin/python -m pytest gateway/tests -q
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest

import app as gw

pytestmark = pytest.mark.asyncio

HELPER = Path(__file__).resolve().parent.parent / "bin" / "llmstack-gpuconf"


@pytest.fixture(autouse=True)
def _isolate():
    snapshot = dict(gw._routes_cache)
    gw._host_cooldown.clear()
    gw._inflight.clear()
    yield
    gw._routes_cache.clear()
    gw._routes_cache.update(snapshot)
    gw._host_cooldown.clear()
    gw._inflight.clear()
    gw._warm_jobs.clear()


# ---------------------------------------------------------------------------
# 1. an Ollama box sized its models without the VRAM headroom every llama.cpp
#    box applies, so the same file was "fits the card" on one and "spills" on
#    the other -- including on the two OS halves of one dual-boot laptop
# ---------------------------------------------------------------------------

class TestOllamaFitUsesTheSameHeadroom:

    async def _fit_via_ollama(self, monkeypatch, weights: int, vram: int) -> str:
        """Run _refresh_upstream_ctx() against a fake Ollama and return the
        `fit` it recorded for the one model it reports."""
        class FakeClient:
            async def get(self, path, **kw):
                return httpx.Response(
                    200, json={"models": [{"model": "m:latest", "size": weights}]},
                    request=httpx.Request("GET", "http://x" + path))

            async def post(self, path, **kw):
                return httpx.Response(
                    200,
                    json={"model_info": {"llama.block_count": 32,
                                         "llama.attention.head_count_kv": 8,
                                         "llama.attention.head_count": 32,
                                         "llama.embedding_length": 4096,
                                         "llama.context_length": 8192}},
                    request=httpx.Request("POST", "http://x" + path))

        monkeypatch.setattr(gw, "client", FakeClient())
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: vram)
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        gw._upstream_meta_cache["meta"] = {}
        await gw._refresh_upstream_ctx()
        return str(gw._upstream_meta_cache["meta"]["m:latest"]["fit"])

    async def test_a_model_inside_the_headroom_fits(self, monkeypatch):
        # 4 GiB of weights + 1 GiB reserve against 8 GiB * 0.90 = 7.2 GiB.
        fit = await self._fit_via_ollama(monkeypatch, 4 * gw.GIB, 8 * gw.GIB)
        assert fit == "vram"

    async def test_a_model_past_the_headroom_spills(self, monkeypatch):
        # 7 GiB + 1 GiB = 8 GiB: inside raw VRAM, PAST the 7.2 GiB headroom.
        # This is the case that used to come back "vram" and win tier 0.
        fit = await self._fit_via_ollama(monkeypatch, 7 * gw.GIB, 8 * gw.GIB)
        assert fit == "spill"

    async def test_both_engines_agree_on_the_same_file(self, monkeypatch, tmp_path):
        """The dual-boot case: one chassis, two gateways, one GGUF. Whichever
        engine reports it, the tier must come out the same."""
        weights, vram = 7 * gw.GIB, 8 * gw.GIB
        ollama_fit = await self._fit_via_ollama(monkeypatch, weights, vram)

        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"\0")
        monkeypatch.setattr(gw, "load_models", lambda: [
            {"id": "m", "path": str(gguf), "enabled": True, "aliases": [],
             "n_cpu_moe": 0, "mmproj": ""}])
        monkeypatch.setattr(gw, "model_bytes",
                            lambda p: weights if str(p) == str(gguf) else 0)
        monkeypatch.setattr(gw, "vram_total_bytes", lambda: vram)
        monkeypatch.setattr(gw.platform, "system", lambda: "Linux")
        llamacpp_fit = gw.local_model_meta()["m"]["fit"]

        assert ollama_fit == llamacpp_fit == "spill"


# ---------------------------------------------------------------------------
# 2. host_tier() read an explicit rank of 0 as "unranked"
# ---------------------------------------------------------------------------

class TestExplicitRankZero:

    def test_rank_zero_is_a_rank_not_an_absence(self, monkeypatch, tmp_path):
        """`rank: 0` means first in its tier. Read through `or`, it fell
        through to the branch default and put the box LAST instead."""
        specs = dict(gw.DEFAULT_SPECS)
        specs["ranked-first"] = {"klass": "small", "rank": 0, "vram_gb": 0}
        specs["ranked-third"] = {"klass": "small", "always_on": 3, "vram_gb": 0}
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        gw._routes_cache["meta"] = {}

        first = gw.host_tier("ranked-first", "m", "worker")
        third = gw.host_tier("ranked-third", "m", "worker")
        assert first[1] == 0
        assert first < third

    def test_an_unranked_box_still_gets_its_branch_default(self, monkeypatch):
        specs = dict(gw.DEFAULT_SPECS)
        specs["unranked"] = {"klass": "small", "vram_gb": 0}
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        gw._routes_cache["meta"] = {}
        assert gw.host_tier("unranked", "m", "worker")[1] == 6

    def test_a_junk_rank_does_not_raise(self, monkeypatch):
        specs = dict(gw.DEFAULT_SPECS)
        specs["junk"] = {"klass": "small", "rank": "not-a-number", "vram_gb": 0}
        monkeypatch.setattr(gw, "load_specs", lambda: specs)
        gw._routes_cache["meta"] = {}
        assert gw.host_tier("junk", "m", "worker")[1] == 6


# ---------------------------------------------------------------------------
# 3. a stale ctx_reject turned a real upstream failure into a 413 about the
#    caller's prompt
# ---------------------------------------------------------------------------

class TestStaleCtxRejectDoesNotBecomeA413:

    async def test_a_5xx_after_a_context_rejection_is_a_502(self, client, monkeypatch):
        """smallbox cannot hold the prompt (413 material), bigbox then 500s.
        The caller must be told the fleet failed -- not that their prompt was
        too long for a box that never ran it."""
        async def _targets(model, **kw):
            return [("smallbox", "m"), ("bigbox", "m")]
        monkeypatch.setattr(gw, "resolve_targets", _targets)
        monkeypatch.setattr(gw, "load_peers", lambda: [
            {"name": "smallbox", "url": "http://smallbox:8080", "token": "t"},
            {"name": "bigbox", "url": "http://bigbox:8080", "token": "t"}])

        async def _key(peer):
            return "k"
        monkeypatch.setattr(gw, "peer_inference_key", _key)
        # smallbox's ceiling is far too small for the prompt below, so it is
        # skipped with ctx_reject set; bigbox has never reported one.
        gw._routes_cache.update(ctx={("smallbox", "m"): 2048}, running={},
                                cap={}, meta={}, engine={})

        async def _send(self, request, **kw):
            return httpx.Response(500, json={"error": "cuda oom"}, request=request)
        monkeypatch.setattr(httpx.AsyncClient, "send", _send)

        raw, meta = gw.mint_key("ctx-reject-key")
        gw.db_exec(
            "INSERT INTO agents(key_id,enabled,name,system_prompt,rules,"
            "allowed_models,force_model,param_overrides,ctx_limit,updated_at)"
            " VALUES (?,1,?,?,?,?,?,?,?,?)",
            (int(meta["id"]), "t", "", "", json.dumps(["m"]), "m",
             json.dumps({}), 32768, gw.now()))
        gw.db_exec("INSERT INTO public_keys(created_at,email,domain,kind,models,"
                   "ctx,key_id,status) VALUES (?,?,?,?,?,?,?,?)",
                   (gw.now(), "a@b.test", "b.test", "single",
                    json.dumps({"model": "m"}), 32768, int(meta["id"]), "issued"))
        gw._public_key_cache.clear()
        gw._public_key_row_cache.clear()

        r = client.post("/v1/chat/completions",
                        headers={"Authorization": "Bearer " + raw},
                        json={"model": "m", "max_tokens": 16,
                              "messages": [{"role": "user", "content": "x" * 40000}]})
        assert r.status_code != 413, "a 500 from bigbox was reported as a prompt-length error"
        assert r.status_code in (500, 502)


# ---------------------------------------------------------------------------
# 4. the warm-up cooldown never engaged if any one model could not be loaded
# ---------------------------------------------------------------------------

class TestWarmCooldownEngages:

    def _issued_row(self) -> dict:
        raw, meta = gw.mint_key("fleet-pass:warm@corp.test")
        gw.db_exec(
            "INSERT INTO public_keys(created_at,email,domain,kind,models,ctx,"
            "key_id,status,warm_token) VALUES (?,?,?,?,?,?,?,?,?)",
            (gw.now(), "warm@corp.test", "corp.test", "single",
             json.dumps({"model": "m"}), 8192, int(meta["id"]), "issued",
             "tok_" + "a" * 20))
        return gw.db_query("SELECT * FROM public_keys WHERE warm_token=?",
                           ("tok_" + "a" * 20,))[0]

    async def test_a_run_with_an_unloadable_model_still_starts_the_cooldown(
            self, client, monkeypatch):
        """One model the fleet cannot serve is marked failed forever. Gating
        the cooldown on "everything ready" therefore never engaged, leaving
        the button an unthrottled way to re-run real work for the models that
        DO resolve."""
        row = self._issued_row()
        token = str(row["warm_token"])
        planned = []

        async def _plan(r):
            planned.append(1)
            return [{"role": "primary", "public_id": "m", "name": "M",
                     "phase": "failed", "detail": "nothing serves it",
                     "progress": None, "box": "Box 1", "action": "none",
                     "cand": None, "fleet_id": "", "host": ""}]
        monkeypatch.setattr(gw, "plan_warm", _plan)

        async def _run(tok):
            job = gw._warm_jobs[tok]
            job["done"] = True
            job["finished"] = time.time()
        monkeypatch.setattr(gw, "_warm_run", _run)

        first = client.post("/public/warm/" + token + "/start")
        assert first.status_code == 200
        # let the task run
        client.get("/public/warm/" + token + "/status")
        second = client.post("/public/warm/" + token + "/start")
        assert second.status_code == 200
        assert second.json()["message"] == "warmed a moment ago"
        assert len(planned) == 1, "the second click re-planned and re-ran the work"

    async def test_the_cooldown_expires(self, client, monkeypatch):
        row = self._issued_row()
        token = str(row["warm_token"])
        planned = []

        async def _plan(r):
            planned.append(1)
            return []
        monkeypatch.setattr(gw, "plan_warm", _plan)

        async def _run(tok):
            job = gw._warm_jobs[tok]
            job["done"] = True
            # finished long enough ago that the window has passed
            job["finished"] = time.time() - gw.WARM_COOLDOWN - 1
        monkeypatch.setattr(gw, "_warm_run", _run)

        client.post("/public/warm/" + token + "/start")
        client.get("/public/warm/" + token + "/status")
        client.post("/public/warm/" + token + "/start")
        assert len(planned) == 2

    async def test_a_planning_failure_does_not_wedge_the_token(
            self, client, monkeypatch):
        """The placeholder that closes the double-click race must not outlive
        a plan_warm() that raised, or the button is dead until a restart."""
        row = self._issued_row()
        token = str(row["warm_token"])

        async def _boom(r):
            raise RuntimeError("fleet unreachable")
        monkeypatch.setattr(gw, "plan_warm", _boom)

        r = client.post("/public/warm/" + token + "/start")
        assert r.status_code == 503
        assert token not in gw._warm_jobs

        async def _plan(r):
            return []
        monkeypatch.setattr(gw, "plan_warm", _plan)

        async def _run(tok):
            gw._warm_jobs[tok].update(done=True, finished=time.time())
        monkeypatch.setattr(gw, "_warm_run", _run)
        assert client.post("/public/warm/" + token + "/start").status_code == 200


# ---------------------------------------------------------------------------
# 5. the helper: a prefix collision read a freshly staged value as already
#    active, and an empty array crashed it outright on macOS's bash 3.2
# ---------------------------------------------------------------------------

class TestHelperOnASandboxTree:
    """Exercises gateway/bin/llmstack-gpuconf against a fake filesystem via
    its own LLMSTACK_GPUCONF_ROOT/_OS hooks -- no root, no real bootloader."""

    def _tree(self, tmp_path: Path, ram_kb: int = 32481816,
              cmdline: str = "BOOT_IMAGE=/vmlinuz ro quiet") -> Path:
        for d in ("proc", "etc/default", "boot/grub", "lib/modules/7.0.0-29-generic",
                  "usr/sbin"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / "proc/meminfo").write_text("MemTotal:       %d kB\n" % ram_kb)
        (tmp_path / "proc/cmdline").write_text(cmdline + "\n")
        (tmp_path / "etc/default/grub").write_text(
            'GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"\nGRUB_CMDLINE_LINUX=""\n')
        # A stand-in for update-grub that does the one thing the helper checks
        # for: source /etc/default/grub, then the grub.d snippet on top, and
        # write the resulting cmdline into grub.cfg. A stub that merely exits 0
        # would let a broken staging pass.
        stub = tmp_path / "usr/sbin/update-grub"
        stub.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            'GRUB_CMDLINE_LINUX_DEFAULT=""\n'
            f'. "{tmp_path}/etc/default/grub"\n'
            f'for f in "{tmp_path}"/etc/default/grub.d/*.cfg; do\n'
            '  [ -e "$f" ] && . "$f"\n'
            "done\n"
            f'printf "linux /vmlinuz ro %s\\n" "$GRUB_CMDLINE_LINUX_DEFAULT" '
            f'> "{tmp_path}/boot/grub/grub.cfg"\n')
        stub.chmod(0o755)
        initrd = tmp_path / "usr/sbin/update-initramfs"
        initrd.write_text("#!/bin/sh\nexit 0\n")
        initrd.chmod(0o755)
        return tmp_path

    def _run(self, root: Path, *args: str) -> subprocess.CompletedProcess:
        env = {"LLMSTACK_GPUCONF_ROOT": str(root), "LLMSTACK_GPUCONF_OS": "Linux",
               "PATH": str(root / "usr/sbin") + ":/usr/bin:/bin"}
        return subprocess.run(["/bin/bash", str(HELPER), *args],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_status_on_a_clean_box_emits_valid_json(self, tmp_path):
        """The empty-array crash: KERNELS and warnings are both empty here,
        and on bash 3.2 a bare ${arr[@]} under `set -u` aborts the script."""
        root = self._tree(tmp_path)
        (root / "lib/modules/7.0.0-29-generic").rmdir()
        p = self._run(root, "status")
        assert p.returncode == 0, p.stderr
        d = json.loads(p.stdout.strip().splitlines()[-1])
        assert d["gib"] is None and d["kernels"] == [] and d["warnings"] == []
        assert d["max_gib"] > 0 and d["suggested_gib"] > 0

    def test_a_staged_value_is_not_active_until_the_cmdline_carries_it(self, tmp_path):
        root = self._tree(tmp_path)
        staged = json.loads(self._run(root, "20").stdout.strip().splitlines()[-1])
        assert staged["gib"] == 20
        assert staged["active"] is False and staged["reboot_required"] is True

        (root / "proc/cmdline").write_text(
            "BOOT_IMAGE=/vmlinuz ro quiet " + staged["cmdline_token"] + "\n")
        after = json.loads(self._run(root, "status").stdout.strip().splitlines()[-1])
        assert after["active"] is True and after["reboot_required"] is False

    def test_a_token_that_is_a_prefix_of_the_live_one_is_not_active(self, tmp_path):
        """4 GiB is ttm.pages_limit=1048576; 40 GiB is ...=10485760. A
        substring match called the freshly staged 4 GiB "already active"
        while the box was still booted at 40."""
        root = self._tree(
            tmp_path,
            cmdline="BOOT_IMAGE=/vmlinuz ttm.pages_limit=10485760 "
                    "ttm.page_pool_size=10485760")
        d = json.loads(self._run(root, "4").stdout.strip().splitlines()[-1])
        assert d["gib"] == 4
        assert d["cmdline_token"] == "ttm.pages_limit=1048576"
        assert d["active"] is False, "a prefix of the live token read as active"
        assert d["reboot_required"] is True

    def test_inline_tokens_are_stripped_so_the_cmdline_holds_exactly_one(self, tmp_path):
        """server-1 was provisioned with the tokens written straight into
        GRUB_CMDLINE_LINUX_DEFAULT. The grub.d snippet appends to that, so
        the old copy has to go or the kernel gets two different values."""
        root = self._tree(tmp_path)
        (root / "etc/default/grub").write_text(
            'GRUB_CMDLINE_LINUX_DEFAULT="ttm.pages_limit=3932160 '
            'ttm.page_pool_size=3932160 amd_iommu=off fsck.repair=yes"\n'
            'GRUB_CMDLINE_LINUX=""\n')
        p = self._run(root, "12")
        assert p.returncode == 0, p.stderr
        grub = (root / "etc/default/grub").read_text()
        assert "ttm.pages_limit" not in grub
        assert "amd_iommu=off" in grub and "fsck.repair=yes" in grub
        assert (root / "etc/default/grub.llmstack-bak").exists()
        # and the kernel that actually boots gets exactly one of each token,
        # with the box's other boot arguments untouched
        cfg = (root / "boot/grub/grub.cfg").read_text()
        assert cfg.count("ttm.pages_limit=") == 1
        assert cfg.count("ttm.page_pool_size=") == 1
        assert "amd_iommu=off" in cfg and "fsck.repair=yes" in cfg

    def test_staging_twice_does_not_accumulate_tokens(self, tmp_path):
        root = self._tree(tmp_path)
        self._run(root, "12")
        self._run(root, "16")
        cfg = (root / "boot/grub/grub.cfg").read_text()
        assert cfg.count("ttm.pages_limit=") == 1
        assert "ttm.pages_limit=4194304" in cfg      # 16 GiB, the later value

    def test_validation_rejects_more_than_the_box_has(self, tmp_path):
        root = self._tree(tmp_path)
        p = self._run(root, "999")
        assert p.returncode == 2
        assert "out of range" in p.stderr

    def test_auto_leaves_the_os_its_reserve(self, tmp_path):
        ram_kb = 32481816                    # apu-box-1's real MemTotal
        root = self._tree(tmp_path, ram_kb=ram_kb)
        d = json.loads(self._run(root, "auto").stdout.strip().splitlines()[-1])
        assert d["gib"] == d["suggested_gib"]
        assert d["gib"] <= d["max_gib"]
        # The dashboard shows these two numbers before anything is staged, so
        # it recomputes them in Python. The two must not drift: a slider whose
        # maximum the helper then rejects is worse than no slider.
        assert gw.gtt_targets(ram_kb * 1024) == (d["max_gib"], d["suggested_gib"])

    @pytest.mark.parametrize("ram_kb", [32481816, 19716096, 16777216, 8388608])
    def test_app_and_helper_agree_on_every_box_in_the_fleet(self, tmp_path, ram_kb):
        root = self._tree(tmp_path / str(ram_kb), ram_kb=ram_kb)
        d = json.loads(self._run(root, "status").stdout.strip().splitlines()[-1])
        assert gw.gtt_targets(ram_kb * 1024) == (d["max_gib"], d["suggested_gib"])


class TestReQueueingADownloadDoesNotForkASecondWriter:
    """apu-tablet-2 queued its six models twice within two minutes -- once by
    hand, once by `pull-models.ps1 -Watch`, whose own comment calls itself
    safe to re-run. `start_download` only refused a file already finished on
    disk, so the second call started a second worker on the same .part. Each
    resumed from whatever size it happened to stat and then appended at EOF,
    so the two ranges interleaved: 32 GiB of downloads that reported healthy
    progress and were, byte for byte, garbage. Nothing raises here -- the
    failure only surfaces much later, when llama.cpp rejects the GGUF."""

    def _dest(self, repo, filename):
        return (gw.MODELS_DIR / repo.replace("/", "__") / filename).resolve()

    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        """Keep the worker parked so the job stays 'downloading' for the
        duration of the test without touching HuggingFace."""
        started = []

        def _park(job_id, repo, filename, dest):
            started.append(job_id)
            gw.job_update(job_id, status="downloading")
            time.sleep(30)

        monkeypatch.setattr(gw, "_download_worker", _park)
        self.started = started
        yield
        for job_id in started:
            gw._download_cancel.add(job_id)
            gw._download_threads.pop(job_id, None)

    def test_second_queue_returns_the_first_job(self):
        repo, filename = "vendor/some-repo", "model-Q4_K_M.gguf"
        first = gw.start_download(repo, filename)
        second = gw.start_download(repo, filename)

        assert second["job_id"] == first["job_id"]
        assert second["already_running"] is True
        # One worker, and one row -- not two of either.
        assert len(self.started) == 1
        rows = gw.db_query(
            "SELECT id FROM jobs WHERE dest=? AND status IN ('queued','downloading')",
            (str(self._dest(repo, filename)),),
        )
        assert len(rows) == 1

    def test_a_row_that_outlived_its_worker_is_re_driven_not_duplicated(self):
        repo, filename = "vendor/other-repo", "other-Q4_K_M.gguf"
        first = gw.start_download(repo, filename)
        # What a gateway restart leaves behind: the row still says
        # 'downloading', but nothing is behind it any more.
        gw._download_threads.pop(first["job_id"], None)

        second = gw.start_download(repo, filename)

        assert second["job_id"] == first["job_id"]
        assert "already_running" not in second
        rows = gw.db_query(
            "SELECT id FROM jobs WHERE dest=?",
            (str(self._dest(repo, filename)),),
        )
        assert len(rows) == 1
