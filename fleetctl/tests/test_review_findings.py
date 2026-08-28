"""What an adversarial review of fleetctl found, pinned down.

Six reviewers read the installer with different questions (a stranger's
fresh clone, Windows, destructive actions on a live box, the package
families, CI truthfulness, the hw.py extraction) and one skeptic per finding
tried to refute it. These are the ones that survived, each as the test that
would have caught it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fleetctl import planner
from fleetctl import steps as steps_mod
from fleetctl.runner import Ctx


def step(step_id: str):
    obj = next(s for s in steps_mod.catalogue() if s.id == step_id)
    return obj() if isinstance(obj, type) else obj


def ctx_for(facts, repo, tmp_path, **kw):
    plan, prov = planner.build(facts, repo=repo)
    ctx = Ctx(plan, facts, repo=repo, root=str(tmp_path / "sandbox"), **kw)
    ctx.prov = prov
    return ctx


# --------------------------------------------------------------------------
class TestAPlanIsForTheBoxTheFactsDescribe:
    """`plan --host <linux box>` from a Windows workstation wrote
    `paths.prefix: C:\\llmstack` into a Linux host's committed plan: every
    shape-derived value came from the detecting machine's OS while the
    host.yml's platform.os said otherwise, and nothing noticed."""

    def test_a_host_yml_naming_another_os_is_refused(self, windows_facts, empty_repo):
        with pytest.raises(planner.PlanError) as e:
            planner.build(windows_facts, repo=empty_repo,
                          overrides={"platform": {"os": "linux"}})
        assert "platform.os: linux" in str(e.value)
        assert "windows machine" in str(e.value)
        assert "--facts" in str(e.value)

    def test_the_same_os_is_fine(self, windows_facts, empty_repo):
        plan, _ = planner.build(windows_facts, repo=empty_repo,
                                overrides={"platform": {"os": "windows"}})
        assert plan["paths"]["prefix"].startswith("C:")


# --------------------------------------------------------------------------
class TestABoxWithNoTailnetStillGetsAnAddress:
    """public_api_url is required and was derived only from a Tailscale
    address, so a fresh machine with no tailnet could not finish install.sh
    at all: "required and nothing supplied it", and no hint."""

    def test_the_lan_address_is_the_fallback(self, linux_facts, empty_repo):
        linux_facts["tailscale"] = {"present": False, "ipv4": None, "name": None}
        linux_facts["lan_ipv4"] = "192.168.1.20"
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert plan["network"]["public_api_url"] == "http://192.168.1.20:8080/v1"
        assert "LAN" in prov["network.public_api_url"]
        assert not planner.validate(plan)

    def test_the_tailnet_still_wins(self, linux_facts, empty_repo):
        linux_facts["lan_ipv4"] = "192.168.1.20"
        plan, prov = planner.build(linux_facts, repo=empty_repo)
        assert plan["network"]["public_api_url"] == "http://100.64.0.1:8080/v1"
        assert "tailnet" in prov["network.public_api_url"]

    def test_a_stated_url_is_never_overridden(self, linux_facts, empty_repo):
        linux_facts["tailscale"] = {"present": False, "ipv4": None, "name": None}
        linux_facts["lan_ipv4"] = "192.168.1.20"
        plan, _ = planner.build(linux_facts, repo=empty_repo, overrides={
            "network": {"public_api_url": "https://api.example.com/v1"}})
        assert plan["network"]["public_api_url"] == "https://api.example.com/v1"


# --------------------------------------------------------------------------
class TestApplyFetchesOnlyWhatIsMissing:
    """Turning on llama_swap for a box whose llama-server was installed and
    RUNNING re-fetched the whole llama.cpp release over the top of it."""

    def test_a_present_engine_is_not_downloaded_again(self, linux_facts, empty_repo,
                                                       tmp_path, monkeypatch):
        linux_facts["engines"] = {"llama_server": "/usr/local/bin/llama-server",
                                  "llama_swap": None, "ollama": None, "lmstudio": None}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert ctx.plan["engine"]["kind"] == "llama.cpp"
        assert ctx.plan["engine"]["llama_swap"]
        engine = step("engine")
        fetched = []
        monkeypatch.setattr(type(engine), "_release_archive",
                            lambda self, c: fetched.append("llama.cpp"))
        monkeypatch.setattr(type(engine), "_llama_swap",
                            lambda self, c: fetched.append("llama-swap"))
        engine.apply(ctx)
        assert fetched == ["llama-swap"]

    def test_nothing_missing_means_nothing_fetched(self, linux_facts, empty_repo,
                                                   tmp_path, monkeypatch):
        linux_facts["engines"] = {"llama_server": "/usr/local/bin/llama-server",
                                  "llama_swap": "/usr/local/bin/llama-swap",
                                  "ollama": None, "lmstudio": None}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        engine = step("engine")
        monkeypatch.setattr(type(engine), "_release_archive",
                            lambda self, c: pytest.fail("fetched llama.cpp"))
        monkeypatch.setattr(type(engine), "_llama_swap",
                            lambda self, c: pytest.fail("fetched llama-swap"))
        engine.apply(ctx)


# --------------------------------------------------------------------------
class TestAScheduledTaskThatAnswersIsLeftAlone:
    """Creating ONE missing task used to /End, /Delete and /Create every
    task -- so enabling llama-swap on a working Windows peer stopped its
    running gateway on the way."""

    def test_only_the_absent_task_is_recreated(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        services = step("services")
        tasks = services._tasks(ctx)
        assert set(tasks) == {"llm-gateway", "llama-swap"}

        def probe(argv, timeout=60):
            # llm-gateway answers /Query; llama-swap does not exist yet.
            rc = 0 if "llm-gateway" in argv else 1
            return SimpleNamespace(returncode=rc, stdout="", stderr="")

        calls: list[list[str]] = []
        ctx.probe = probe
        ctx.run = lambda argv, **kw: calls.append(list(argv))
        services.apply(ctx)
        touched = {c[3] for c in calls if c[:2] == ["schtasks", "/Delete"]}
        assert touched == {"llama-swap"}
        assert not any(c[:2] == ["schtasks", "/End"] and c[3] == "llm-gateway" for c in calls)
        ran = [c[3] for c in calls if c[:2] == ["schtasks", "/Run"]]
        assert ran == ["llama-swap"]


# --------------------------------------------------------------------------
class TestAWrapperWithANonAsciiPathStillGetsWritten:
    """`encoding="ascii"` on the Windows wrappers raised on a prefix with an
    accent in it -- after the venv and the files were in place, before any
    task was scheduled."""

    def test_ascii_stays_ascii(self, windows_facts, empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        assert step("wrappers")._encoding(ctx, "@echo off\r\n") == "ascii"

    def test_non_ascii_falls_back_to_a_codec_that_can_hold_it(self, windows_facts,
                                                              empty_repo, tmp_path):
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        enc = step("wrappers")._encoding(ctx, "cd /d C:\\Us\u00e9rs\\llmstack\r\n")
        assert enc in ("oem", "utf-8")
        "C:\\Us\u00e9rs".encode(enc)  # and it really can

    def test_posix_wrappers_are_utf8_regardless(self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert step("wrappers")._encoding(ctx, "caf\u00e9") == "utf-8"


# --------------------------------------------------------------------------
class TestAGroupChangeDoesNotChownTheModels:
    """Fixing video/render membership on an existing account used to
    `chown -R` the whole stack, models directory included."""

    def test_an_existing_account_gets_the_group_and_nothing_else(
            self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = lambda argv, timeout=60: SimpleNamespace(returncode=0, stdout="1001",
                                                             stderr="")
        calls: list[list[str]] = []
        ctx.sudo = lambda argv, **kw: calls.append(list(argv))
        step("service-account").apply(ctx)
        assert any(c[:2] == ["usermod", "-aG"] for c in calls)
        assert not any(c[0] == "chown" for c in calls)

    def test_a_new_account_takes_the_stack(self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.probe = lambda argv, timeout=60: SimpleNamespace(returncode=1, stdout="",
                                                             stderr="")
        calls: list[list[str]] = []
        ctx.sudo = lambda argv, **kw: calls.append(list(argv))
        step("service-account").apply(ctx)
        assert calls[0][0] == "useradd"
        assert any(c[0] == "chown" for c in calls)
