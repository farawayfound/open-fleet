"""What an adversarial review of fleetctl found, pinned down.

Six reviewers read the installer with different questions (a stranger's
fresh clone, Windows, destructive actions on a live box, the package
families, CI truthfulness, the hw.py extraction) and one skeptic per finding
tried to refute it. These are the ones that survived, each as the test that
would have caught it.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest
import sys

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


# --------------------------------------------------------------------------
class TestTheFirstCommandAStrangerRuns:
    """`detect --quick` is the first thing both bootstraps and the README run.
    It printed `python … (3.12.10, need >= None)` -- the quick path built the
    python fact by hand and left out the minimum the slow path reports."""

    def test_quick_detect_still_names_the_minimum(self):
        from fleetctl import facts as facts_mod
        f = facts_mod.gather(quick=True)
        assert f["python"]["min"] == ".".join(map(str, facts_mod.MIN_PYTHON))
        assert f["python"]["exe"]
        assert f["python"]["version"]

    def test_quick_and_slow_agree_on_the_shape(self):
        from fleetctl import facts as facts_mod
        quick = facts_mod.gather(quick=True)["python"]
        assert set(quick) == set(facts_mod.find_python())

    def test_a_venv_is_not_the_interpreter_a_service_gets_pointed_at(self, monkeypatch):
        from fleetctl import facts as facts_mod
        monkeypatch.setattr(facts_mod.sys, "prefix", "/somewhere/.venv")
        monkeypatch.setattr(facts_mod.sys, "base_prefix", "/usr")
        monkeypatch.setattr(facts_mod.sys, "executable", "/somewhere/.venv/bin/python")
        monkeypatch.setattr(facts_mod.sys, "_base_executable", "/usr/bin/python3", raising=False)
        assert facts_mod.this_python()["exe"] == "/usr/bin/python3"


# --------------------------------------------------------------------------
class TestGatewayEnvValuesAreEscapedForTheShellThatReadsThem:
    """gateway.env(.cmd) is not a data file on darwin/windows: run-gateway.sh
    `source`s it and run-gateway.cmd `call`s it. Every value in it comes from
    host.yml/fleet.yml/--set, or from a foreign line carried across a
    previous apply (EnvFile.foreign()) -- operator-controlled, not
    attacker-reachable, but a path or URL with a stray space/`$`/`&`/paren
    used to run as shell (or corrupt batch parsing) instead of sitting inert
    as a value."""

    PAYLOAD = "http://x/$(id > /tmp/pwned); true"

    def _render_with_public_api_url(self, facts, repo, tmp_path, value):
        from fleetctl.steps.stack import EnvFile

        value_escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        (repo / "fleet.yml").write_text(
            f'network:\n  public_api_url: "{value_escaped}"\n', encoding="utf-8")
        ctx = ctx_for(facts, repo, tmp_path)
        return EnvFile()._render(ctx, "TOKEN123")

    def test_darwin_wraps_a_dangerous_value_in_single_quotes(
            self, darwin_facts, empty_repo, tmp_path):
        text = self._render_with_public_api_url(darwin_facts, empty_repo, tmp_path,
                                                 self.PAYLOAD)
        assert f"export LLMSTACK_PUBLIC_API_URL='{self.PAYLOAD}'" in text
        # Not the bare payload sitting unquoted after `=`, where bash would
        # treat `$(...)` as a command substitution rather than as text.
        assert f"PUBLIC_API_URL={self.PAYLOAD}\n" not in text

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs a real bash")
    @pytest.mark.skipif(sys.platform == "win32",
                        reason="exercises a POSIX bash: the Windows runner checks out CRLF "
                               "and Git Bash cannot source a C:\\ path; the fleet's bash boxes "
                               "are covered by the ubuntu/macos legs of the matrix")
    def test_sourcing_the_rendered_file_does_not_run_the_payload(
            self, darwin_facts, empty_repo, tmp_path):
        marker = tmp_path / "pwned"
        payload = f"http://x/$(touch {marker}); id"
        text = self._render_with_public_api_url(darwin_facts, empty_repo, tmp_path,
                                                 payload)
        envf = tmp_path / "gateway.env"
        envf.write_text(text, encoding="utf-8")
        out = subprocess.run(
            ["bash", "-c", f'source "{envf}"; printf %s "$LLMSTACK_PUBLIC_API_URL"'],
            capture_output=True, text=True, timeout=10, check=True)
        assert out.stdout == payload, "sourced value must equal the literal payload"
        assert not marker.exists(), "the payload ran as a command instead of sitting inert"

    def test_windows_refuses_a_value_cmd_cannot_carry(
            self, windows_facts, empty_repo, tmp_path):
        with pytest.raises(RuntimeError) as exc:
            self._render_with_public_api_url(
                windows_facts, empty_repo, tmp_path, "http://x & calc.exe")
        assert "LLMSTACK_PUBLIC_API_URL" in str(exc.value)

    def test_a_benign_value_is_unaffected_on_either_platform(
            self, darwin_facts, windows_facts, empty_repo, tmp_path):
        d = self._render_with_public_api_url(darwin_facts, empty_repo, tmp_path,
                                             "https://api.example.com/v1")
        assert "export LLMSTACK_PUBLIC_API_URL=https://api.example.com/v1" in d
        w = self._render_with_public_api_url(windows_facts, empty_repo, tmp_path,
                                             "https://api.example.com/v1")
        assert "set LLMSTACK_PUBLIC_API_URL=https://api.example.com/v1" in w

    def test_a_carried_darwin_value_does_not_gain_a_quote_layer_on_reapply(
            self, darwin_facts, empty_repo, tmp_path):
        """A foreign (not-fleetctl-owned) value with a space -- e.g. a
        hand-set SMTP password -- must round-trip through
        foreign()/_render() without picking up an extra layer of quoting on
        every apply: shlex.quote() on write must be matched by
        shapes.unquote_shell_value() on read, or a value the file already
        carries drifts (or gains embedded literal quote characters) each
        time fleetctl runs."""
        from fleetctl.steps.stack import EnvFile

        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        step_ = EnvFile()
        first = step_._render(ctx, "TOKEN123", {"FOO_SECRET": "a b"})
        assert "export FOO_SECRET='a b'" in first
        carried_again = step_.foreign(ctx, first)
        assert carried_again["FOO_SECRET"] == "a b"          # unquoted back out
        second = step_._render(ctx, "TOKEN123", carried_again)
        assert second == first, "re-rendering a carried value must be idempotent"


# --------------------------------------------------------------------------
class TestPushShTargetsAreStable:
    """push.sh's apu-box-1/gpu-laptop-1 defaults were the bare mDNS/NetBIOS names
    `user@apu-box-1` and `user@box.local` -- exactly the class of address
    deploy-gateway.sh's own commit history documents going stale after a
    rename or a DHCP change and reading as "offline" while the box was
    reachable the whole time (see deploy-gateway.sh's ssh_target(), which
    pins these same two hosts to tailnet addresses). Separately, the apu-box-1
    admin-token staging pointed at one already-finished Claude Code session's
    scratchpad path, so "admin token staged" could never actually run and
    never said so."""

    def _text(self, repo):
        path = repo / "push.sh"
        if not path.is_file():
            pytest.skip("no push.sh in this checkout")
        return path.read_text(encoding="utf-8")

    def test_ai_max_defaults_to_its_tailnet_address(self, repo):
        text = self._text(repo)
        assert '${2:-user@100.64.0.113}' in text
        assert '${2:-user@apu-box-1}' not in text

    def test_zephyrus_defaults_to_its_tailnet_address(self, repo):
        text = self._text(repo)
        assert '${2:-user@100.64.0.36}' in text
        assert '${2:-user@box.local}' not in text

    def test_the_admin_token_path_is_not_a_dead_session_scratchpad(self, repo):
        text = self._text(repo)
        assert "AppData/Local/Temp/claude" not in text
        assert "AIMAX_ADMIN_TOKEN_FILE" in text

    def test_an_unset_token_var_warns_instead_of_staying_silent(self, repo):
        text = self._text(repo)
        assert "NOT staged" in text


# --------------------------------------------------------------------------
class TestUbserverBootstrapHasAFirewallAndNarrowerGrants:
    """server-1's bootstrap installed no network-layer restriction at all
    (the break-glass admin token and Cockpit were reachable from the whole
    LAN), and its models-directory grant went to `o+` (every local account)
    a line after already granting the narrower `g+` access the usermod
    above it exists to provide. This does not execute the script (it only
    runs on a rebuild) -- it reads the checked-in text."""

    def _text(self, repo):
        path = repo / "hosts" / "server-1" / "bootstrap.sh"
        if not path.is_file():
            pytest.skip("no hosts/server-1/bootstrap.sh in this checkout")
        return path.read_text(encoding="utf-8")

    def test_ufw_default_denies_incoming_and_allows_tailnet_and_ssh(self, repo):
        text = self._text(repo)
        assert "ufw default deny incoming" in text
        assert "ufw allow in on tailscale0" in text
        assert "ufw allow 22/tcp" in text
        assert "ufw --force enable" in text

    def test_the_firewall_is_enabled_only_after_sshd_is_confirmed_listening(self, repo):
        text = self._text(repo)
        guard_at = text.index(":22")
        enable_at = text.index("ufw --force enable")
        assert guard_at < enable_at, "the :22 listening check must precede the enable"

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="exercises a POSIX bash: the Windows runner checks out CRLF "
                               "and Git Bash cannot source a C:\\ path; the fleet's bash boxes "
                               "are covered by the ubuntu/macos legs of the matrix")
    def test_the_script_still_parses_as_bash(self, repo):
        path = repo / "hosts" / "server-1" / "bootstrap.sh"
        if not path.is_file():
            pytest.skip("no hosts/server-1/bootstrap.sh in this checkout")
        if shutil.which("bash") is None:
            pytest.skip("no bash to check syntax with")
        subprocess.run(["bash", "-n", str(path)], check=True, timeout=10)

    def test_the_models_grant_is_group_only_not_world(self, repo):
        text = self._text(repo)
        assert "chmod g+x /home/user" in text
        assert 'chmod -R g+rX "$MODELS_DIR"' in text
        assert "chmod o+x /home/user" not in text
        assert 'chmod -R o+rX "$MODELS_DIR"' not in text
