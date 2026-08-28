"""The step catalogue, and the promise that check() never changes anything.

The dry-run purity tests are the important ones here. `apply --dry-run` is
what CI runs on hosted runners and inside distro containers, and what a
person runs on a machine they are nervous about -- so a check() that quietly
created a directory would break the one guarantee the whole command is sold
on.
"""
from __future__ import annotations

import pytest

from fleetctl import planner
from fleetctl import steps as steps_mod
from fleetctl.runner import Ctx
from fleetctl.steps import NEEDS_WORK, SKIPPED


def ctx_for(facts, repo, tmp_path, **kw):
    plan, _ = planner.build(facts, repo=repo)
    return Ctx(plan, facts, repo=repo, root=str(tmp_path / "sandbox"), **kw)


class TestCatalogue:
    def test_every_step_is_in_the_run_order(self):
        # catalogue() raises if a step is missing from ORDER; calling it is
        # the assertion. A step not in ORDER would silently never run.
        assert steps_mod.catalogue()

    def test_the_order_is_the_order(self):
        ids = [s.id for s in steps_mod.catalogue()]
        assert ids == [i for i in steps_mod.ORDER if i in ids]

    def test_ids_are_unique(self):
        ids = [s.id for s in steps_mod.catalogue()]
        assert len(ids) == len(set(ids))

    def test_the_engine_comes_before_the_config_that_names_it(self):
        ids = steps_mod.ORDER
        assert ids.index("engine") < ids.index("swap-config")
        assert ids.index("envfile") < ids.index("services")
        assert ids.index("services") < ids.index("health")
        assert ids.index("venv") < ids.index("services")
        assert ids.index("python-runtime") < ids.index("venv")

    def test_health_is_last(self):
        assert steps_mod.ORDER[-1] == "health"

    def test_system_steps_are_marked_as_such(self):
        marked = {s.id for s in steps_mod.catalogue() if s.system}
        # These are the ones that reach outside the plan's own paths.
        assert marked == {"packages", "service-account", "grants", "power",
                          "firewall", "gpu-cap", "services"}


class TestSelection:
    def test_steps_skip_in_the_plan_is_honoured(self, linux_facts, empty_repo,
                                                tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["steps"]["skip"] = ["firewall", "power"]
        chosen = {s.id for s in steps_mod.selected(ctx)}
        assert "firewall" not in chosen and "power" not in chosen
        assert "directories" in chosen

    def test_only_narrows_to_exactly_those(self, linux_facts, empty_repo,
                                           tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        chosen = {s.id for s in steps_mod.selected(ctx, ["envfile"])}
        assert chosen == {"envfile"}


class TestWantedness:
    def test_a_service_account_is_a_linux_idea(self, linux_facts, windows_facts,
                                               darwin_facts, empty_repo, tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        step = by_id["service-account"]
        assert step.wanted(ctx_for(linux_facts, empty_repo, tmp_path))
        assert not step.wanted(ctx_for(windows_facts, empty_repo, tmp_path))
        assert not step.wanted(ctx_for(darwin_facts, empty_repo, tmp_path))

    def test_wrappers_only_where_the_supervisor_cannot_hold_an_environment(
            self, linux_facts, windows_facts, darwin_facts, empty_repo, tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        step = by_id["wrappers"]
        # systemd reads EnvironmentFile itself; cron and schtasks cannot.
        assert not step.wanted(ctx_for(linux_facts, empty_repo, tmp_path))
        assert step.wanted(ctx_for(windows_facts, empty_repo, tmp_path))
        assert step.wanted(ctx_for(darwin_facts, empty_repo, tmp_path))

    def test_no_engine_means_no_engine_step(self, linux_facts, empty_repo,
                                            tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["kind"] = "none"
        assert not by_id["engine"].wanted(ctx)

    def test_the_gpu_ceiling_step_only_runs_where_a_ceiling_was_asked_for(
            self, darwin_facts, empty_repo, tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        assert not by_id["gpu-cap"].wanted(ctx)
        ctx.plan["sizing"]["metal_gib"] = 42
        assert by_id["gpu-cap"].wanted(ctx)


class TestCheckIsReadOnly:
    """Every check(), on a sandbox that does not exist yet, must leave it
    not existing."""

    @pytest.mark.parametrize("which", ["linux", "windows", "darwin"])
    def test_no_check_creates_anything(self, request, empty_repo, tmp_path, which):
        facts = request.getfixturevalue(f"{which}_facts")
        sandbox = tmp_path / "sandbox"
        ctx = ctx_for(facts, empty_repo, tmp_path)
        for step in steps_mod.selected(ctx):
            if step.wanted(ctx):
                step.check(ctx)
        assert not sandbox.exists(), "a check() created the sandbox root"

    @pytest.mark.parametrize("which", ["linux", "windows", "darwin"])
    def test_a_dry_run_apply_writes_nothing(self, request, empty_repo, tmp_path,
                                            which):
        facts = request.getfixturevalue(f"{which}_facts")
        sandbox = tmp_path / "sandbox"
        ctx = ctx_for(facts, empty_repo, tmp_path, dry_run=True)
        planned = []
        for step in steps_mod.selected(ctx):
            if not step.wanted(ctx):
                continue
            result = step.check(ctx)
            planned += result.plan
            if result.state in NEEDS_WORK:
                # run(), which is the real entry point and the one place the
                # dry-run contract is enforced. Calling apply() directly here
                # DID download a llama.cpp release and build a virtualenv --
                # which is exactly why the guard is not left to each step.
                step.run(ctx)
        assert not sandbox.exists()
        # The dry run's report comes from check().plan, not from anything the
        # applies recorded -- because on a fresh box no apply ran at all.
        assert planned, "a dry run of an empty box that would do nothing"
        assert any("create" in line for line in planned)


class TestSandboxRefusesSystemWork:
    def test_a_system_command_is_recorded_not_run(self, linux_facts, empty_repo,
                                                  tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert ctx.sandboxed
        assert ctx.run(["definitely-not-a-real-command"], system=True) is None
        assert any("sandbox" in a for a in ctx.actions)

    def test_root_ok_is_false_under_a_sandbox_however_privileged(
            self, linux_facts, empty_repo, tmp_path):
        linux_facts["privilege"] = {"root": True, "sudo_nopasswd": True}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert ctx.root_ok is False

    def test_health_is_skipped_in_a_sandbox(self, linux_facts, empty_repo,
                                            tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert by_id["health"].check(ctx).state == SKIPPED


class TestPathSandboxing:
    def test_a_posix_path_is_prefixed(self, linux_facts, empty_repo, tmp_path):
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert str(ctx.path("/etc/llmstack")).endswith("sandbox/etc/llmstack".replace(
            "/", __import__("os").sep))

    def test_a_drive_letter_becomes_a_directory(self, windows_facts, empty_repo,
                                                tmp_path):
        """So a Windows plan can be exercised on Linux CI, which is where
        most of the Windows-plan testing actually happens."""
        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        parts = ctx.path(r"C:\llmstack\state").parts
        assert parts[-3:] == ("C", "llmstack", "state")

    def test_no_root_means_no_rewriting(self, linux_facts, empty_repo):
        plan, _ = planner.build(linux_facts, repo=empty_repo)
        ctx = Ctx(plan, linux_facts, repo=empty_repo)
        assert str(ctx.path("/etc/llmstack")) == "/etc/llmstack".replace(
            "/", __import__("os").sep) or str(ctx.path("/etc/llmstack")) == "/etc/llmstack"


class TestEnvFileRendering:
    def _env(self, facts, repo, tmp_path):
        from fleetctl.steps.stack import EnvFile

        ctx = ctx_for(facts, repo, tmp_path)
        return EnvFile()._render(ctx, "TOKEN123"), ctx

    def test_linux_writes_bare_assignments_for_systemd(self, linux_facts,
                                                       empty_repo, tmp_path):
        text, _ = self._env(linux_facts, empty_repo, tmp_path)
        assert "\nLLMSTACK_PORT=8080\n" in text
        assert "export " not in text and "\nset " not in text

    def test_macos_exports_because_a_shell_wrapper_sources_it(self, darwin_facts,
                                                              empty_repo, tmp_path):
        text, _ = self._env(darwin_facts, empty_repo, tmp_path)
        assert "export LLMSTACK_PORT=8080" in text

    def test_windows_uses_set_because_a_cmd_wrapper_calls_it(self, windows_facts,
                                                             empty_repo, tmp_path):
        text, _ = self._env(windows_facts, empty_repo, tmp_path)
        assert "set LLMSTACK_PORT=8080" in text
        assert text.lstrip().startswith("rem ")

    def test_a_none_is_absent_not_empty(self, linux_facts, empty_repo, tmp_path):
        """The gateway distinguishes an unset LLMSTACK_AVAILABILITY_FILE
        (never gated) from an empty one."""
        text, _ = self._env(linux_facts, empty_repo, tmp_path)
        assert "AVAILABILITY_FILE" not in text

    def test_a_list_becomes_a_comma_separated_value(self, linux_facts,
                                                    empty_repo, tmp_path):
        (empty_repo / "fleet.yml").write_text(
            "access:\n  admin_emails: [a@b.c, d@e.f]\n", encoding="utf-8")
        text, _ = self._env(linux_facts, empty_repo, tmp_path)
        assert "LLMSTACK_ADMIN_EMAILS=a@b.c,d@e.f" in text

    def test_an_ollama_box_says_so(self, linux_facts, empty_repo, tmp_path):
        linux_facts["engines"]["ollama"] = "/usr/local/bin/ollama"
        text, _ = self._env(linux_facts, empty_repo, tmp_path)
        assert "LLMSTACK_MODELS_FROM_UPSTREAM=1" in text
        assert "11434" in text
        # No llama.cpp binaries to point at.
        assert "LLMSTACK_LLAMA_SERVER" not in text


class TestEnvFileKeepsTheToken:
    def test_an_existing_token_survives_a_rewrite(self, linux_facts, empty_repo,
                                                  tmp_path):
        """The hub holds this as the peer's credential. Everything else in
        the file is regenerated from the plan; this one value is carried."""
        from fleetctl.steps.stack import EnvFile

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        step = EnvFile()
        ctx.write(ctx.plan["paths"]["envfile"],
                  "LLMSTACK_ADMIN_TOKEN=keepme\nLLMSTACK_PORT=1\n")
        assert step._token(ctx) == "keepme"
        step.apply(ctx)
        text = ctx.read(ctx.plan["paths"]["envfile"])
        assert "LLMSTACK_ADMIN_TOKEN=keepme" in text
        assert "LLMSTACK_PORT=8080" in text     # everything else IS updated

    def test_a_staged_token_is_adopted(self, linux_facts, empty_repo, tmp_path):
        """push.sh can pre-stage a token so the hub already holds a peer entry
        for a box that has never been provisioned."""
        from fleetctl.steps.stack import EnvFile

        (empty_repo / ".admin-token").write_text("staged-token\n", encoding="utf-8")
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert EnvFile()._token(ctx) == "staged-token"

    def test_a_fresh_box_gets_a_long_random_one(self, linux_facts, empty_repo,
                                                tmp_path):
        from fleetctl.steps.stack import EnvFile

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        a, b = EnvFile()._token(ctx), EnvFile()._token(ctx)
        assert a != b and len(a) >= 30


class TestServiceUnits:
    def test_the_systemd_unit_can_write_where_gpuconf_needs_to(self, linux_facts,
                                                               empty_repo, tmp_path):
        """ProtectSystem=full makes /etc and /boot read-only for the unit, and
        sudo does not escape a unit's mount namespace -- so without these the
        GPU ceiling control fails with what looks like a permissions bug."""
        from fleetctl.steps.services import Services

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        unit = Services()._units(ctx)["llm-gateway.service"]
        assert "ProtectSystem=full" in unit
        rw = [ln for ln in unit.splitlines() if ln.startswith("ReadWritePaths=")][0]
        for needed in ("/etc/default", "/etc/modprobe.d", "/boot"):
            assert needed in rw

    def test_the_gateway_waits_for_its_upstream(self, linux_facts, empty_repo,
                                                tmp_path):
        from fleetctl.steps.services import Services

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["llama_swap"] = True
        unit = Services()._units(ctx)["llm-gateway.service"]
        assert "llama-swap.service" in unit

    def test_a_visible_device_filter_lands_in_the_engine_unit(self, linux_facts,
                                                              empty_repo, tmp_path):
        from fleetctl.steps.services import Services

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["visible_devices"] = "1"
        unit = Services()._units(ctx)["llama-swap.service"]
        assert "GGML_VK_VISIBLE_DEVICES=1" in unit

    def test_cron_lines_carry_an_id_so_a_rerun_replaces_only_its_own(
            self, darwin_facts, empty_repo, tmp_path):
        """Matching on 'llmstack' would eat a hand-written entry; matching on
        the command would eat nothing after a path change."""
        from fleetctl.steps.services import Services

        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        lines = Services()._cron_lines(ctx)
        tags = [t for t, _ in lines]
        assert "llmstack-gateway" in tags
        assert any(t.endswith("keepalive") for t in tags)

    def test_the_keepalive_matches_the_engine_not_itself(self, darwin_facts,
                                                         empty_repo, tmp_path):
        """A keepalive whose pgrep pattern matches its own shell never
        restarts anything."""
        from fleetctl.steps.services import Services

        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        for _, line in Services()._cron_lines(ctx):
            if "uvicorn" in line:
                assert "pgrep -qf 'uvicorn app:app'" in line


class TestWindowsWrappers:
    def test_the_engine_wrapper_pins_the_visible_device(self, windows_facts,
                                                        empty_repo, tmp_path):
        from fleetctl.steps.stack import Wrappers

        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        ctx.plan["engine"]["visible_devices"] = "1"
        files = Wrappers()._files(ctx)
        swap = next(v for k, v in files.items() if "llama-swap" in k)
        assert "set GGML_VK_VISIBLE_DEVICES=1" in swap

    def test_wrappers_are_crlf_and_ascii(self, windows_facts, empty_repo,
                                         tmp_path):
        """A .cmd file with LF endings and a BOM is a batch file cmd.exe
        mis-parses."""
        from fleetctl.steps.stack import Wrappers

        ctx = ctx_for(windows_facts, empty_repo, tmp_path)
        Wrappers().apply(ctx)
        for path in Wrappers()._files(ctx):
            raw = ctx.path(path).read_bytes()
            assert b"\r\n" in raw
            assert not raw.startswith(b"\xef\xbb\xbf")
