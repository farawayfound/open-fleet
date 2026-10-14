"""The step catalogue, and the promise that check() never changes anything.

The dry-run purity tests are the important ones here. `apply --dry-run` is
what CI runs on hosted runners and inside distro containers, and what a
person runs on a machine they are nervous about -- so a check() that quietly
created a directory would break the one guarantee the whole command is sold
on.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fleetctl import planner
from fleetctl import steps as steps_mod
from fleetctl.runner import Ctx
from fleetctl.steps import DRIFT, MISSING, OK, NEEDS_WORK, SKIPPED


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
        # The power cap's boot unit is ordered ahead of llama-swap by its own
        # systemd unit text (Before=llama-swap.service); at PROVISION time it
        # only has to exist before the step that first enables llama-swap.
        assert ids.index("power-cap") < ids.index("services")

    def test_health_is_last(self):
        assert steps_mod.ORDER[-1] == "health"

    def test_system_steps_are_marked_as_such(self):
        marked = {s.id for s in steps_mod.catalogue() if s.system}
        # These are the ones that reach outside the plan's own paths.
        assert marked == {"packages", "service-account", "grants", "power",
                          "firewall", "gpu-cap", "power-cap", "services"}


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

    def test_the_power_cap_step_is_linux_only_and_needs_a_knob_set(
            self, linux_facts, darwin_facts, windows_facts, empty_repo, tmp_path):
        by_id = {s.id: s for s in steps_mod.catalogue()}
        step = by_id["power-cap"]
        # A box with no power: section at all is untouched -- every plan
        # starts with all three keys None.
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        assert ctx.plan["power"] == {"gpu_sclk_max_mhz": None, "cpu_boost": None,
                                     "cpu_max_mhz": None}
        assert not step.wanted(ctx)
        # Any ONE of the three is enough -- they are independent knobs.
        ctx.plan["power"]["cpu_boost"] = False
        assert step.wanted(ctx)
        # Linux-only: the sysfs paths involved do not exist elsewhere, even
        # if a host.yml somehow set the values on the wrong platform.
        dctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        dctx.plan["power"]["gpu_sclk_max_mhz"] = 2500
        assert not step.wanted(dctx)
        wctx = ctx_for(windows_facts, empty_repo, tmp_path)
        wctx.plan["power"]["cpu_max_mhz"] = 2000
        assert not step.wanted(wctx)

    def test_cpu_boost_false_is_a_real_value_not_treated_as_unset(
            self, linux_facts, empty_repo, tmp_path):
        """cpu_boost: false is a request to turn boost OFF -- `is not None`,
        never a truthiness check, or the one setting that matters most
        (disable boost) would be indistinguishable from not asking at all."""
        by_id = {s.id: s for s in steps_mod.catalogue()}
        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["power"]["cpu_boost"] = False
        assert by_id["power-cap"].wanted(ctx)


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


class TestTheLlamaSwapKeepaliveMatchesALiveEngine:
    """Regression pin for the mac-laptop-1 incident: run-llama-swap.sh execs the
    llama-swap binary, so no live process ever carries the wrapper's own
    filename. A keepalive that pgreps for 'run-llama-swap' matches nothing
    (or its own cron shell) and spawns a doomed duplicate llama-swap every
    five minutes -- each duplicate's start-up preload is what orphaned a
    21 GB llama-server on mac-laptop-1 before the duplicate failed to bind the
    port."""

    # What run-llama-swap.sh's final `exec` line hands the kernel -- the argv
    # a LIVE engine process carries, as ps/pgrep would see it.
    WRAPPER_ARGV = ("/opt/llmstack/bin/llama-swap "
                     "-config /opt/llmstack/etc/llama-swap.yaml")

    def test_the_pattern_is_a_substring_of_the_wrapper_s_exec_d_argv(
            self, darwin_facts, empty_repo, tmp_path):
        from fleetctl.steps.services import Services

        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        assert ctx.plan["platform"]["service"] == "cron"
        assert ctx.plan["engine"]["llama_swap"]

        swap_lines = [line for tag, line in Services()._cron_lines(ctx)
                      if "llama-swap" in tag]
        assert swap_lines
        for line in swap_lines:
            assert "pgrep -qf 'bin/llama-swap -config'" in line
            pattern = line.split("pgrep -qf '", 1)[1].split("'", 1)[0]
            # It would actually match a live engine, not just look plausible.
            assert pattern in self.WRAPPER_ARGV
            # And it can never match the wrapper's own filename -- that is
            # the pattern that started this incident.
            assert "run-llama-swap" not in pattern

    def test_the_gateway_keepalive_is_unaffected(self, darwin_facts,
                                                  empty_repo, tmp_path):
        from fleetctl.steps.services import Services

        ctx = ctx_for(darwin_facts, empty_repo, tmp_path)
        gw_lines = [line for tag, line in Services()._cron_lines(ctx)
                    if "gateway" in tag]
        assert gw_lines
        for line in gw_lines:
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


class TestPowerCap:
    """check()/apply() for the CPU/GPU power cap step, against a fake
    helper answering through ctx.probe -- no real box, no real sysfs."""

    def _step_ctx(self, linux_facts, empty_repo, tmp_path, **power):
        from fleetctl.steps.system import PowerCap

        ctx = ctx_for(linux_facts, empty_repo, tmp_path)
        ctx.plan["power"].update(power)
        return PowerCap(), ctx

    def _status(self, **fields) -> str:
        base = {"gpu_sclk_max_mhz": None, "gpu_sclk_active_mhz": None,
                "gpu_perf_level": None, "cpu_boost": None,
                "cpu_boost_active": None, "cpu_max_mhz": None,
                "cpu_max_active_mhz": None, "warnings": []}
        base.update(fields)
        return json.dumps(base)

    def test_unset_means_not_wanted_and_the_helper_is_never_probed(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path)
        assert not step.wanted(ctx)

    def test_unset_with_nothing_ever_installed_is_SKIPPED(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path)
        c = step.check(ctx)
        assert c.state == SKIPPED

    # -- the revert path: host.yml's power: keys removed after a previous
    # apply already installed and enabled the unit ----------------------

    def test_unset_stays_wanted_when_a_previous_apply_left_the_unit_behind(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path)
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        # Naively reading "nothing configured" as "nothing to do" would let
        # the runner skip check()/apply() outright and leave this enabled
        # forever, reapplying a stale cap at every future boot.
        assert step.wanted(ctx)

    def test_unset_with_a_leftover_unit_is_DRIFT_not_silence(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path)
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        c = step.check(ctx)
        assert c.state == DRIFT
        assert step.UNIT_NAME in c.detail
        assert any("disable" in line for line in c.plan)

    def test_unset_apply_disables_and_removes_the_unit_and_conf(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path)
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        ctx.write(step._conf_path(ctx), step._conf_text(ctx))
        issued = []
        ctx.sudo = lambda argv, **kw: issued.append([str(a) for a in argv])
        step.apply(ctx)
        assert any("disable" in a and "--now" in a for a in issued)
        assert any(step.UNIT_PATH in a for a in issued if "rm" in a)
        assert any(step._conf_path(ctx) in a for a in issued if "rm" in a)
        assert any("daemon-reload" in a for a in issued)
        # A reboot is what actually restores the sysfs defaults -- same
        # staged/pending-reboot honesty as GpuCap's GTT window -- so the
        # operator needs to be told, not left assuming apply() undid it live.
        assert ctx.notes

    def test_helper_missing_is_MISSING(self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   cpu_boost=False)
        c = step.check(ctx)
        assert c.state == MISSING
        assert any("install" in line for line in c.plan)

    def _install_helper(self, ctx) -> None:
        # check()/apply() only look at whether the path exists.
        p = ctx.path(ctx.plan["paths"]["bin"] + "/llmstack-powercap")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)

    def test_helper_present_but_config_and_unit_missing_is_DRIFT(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500)
        self._install_helper(ctx)
        ctx.probe = lambda argv, **kw: (
            SimpleNamespace(returncode=1, stdout="", stderr="")
            if argv[0] == "systemctl" else
            SimpleNamespace(returncode=0, stdout=self._status(gpu_sclk_max_mhz=2500,
                                                              gpu_sclk_active_mhz=2500)))
        c = step.check(ctx)
        assert c.state == DRIFT
        assert any("powercap.conf" in line for line in c.plan)
        assert any(step.UNIT_NAME in line or "systemd" in line for line in c.plan)

    def test_matching_config_and_active_state_is_OK(self, linux_facts, empty_repo,
                                                     tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500, cpu_boost=False,
                                   cpu_max_mhz=2000)
        self._install_helper(ctx)
        ctx.write(step._conf_path(ctx), step._conf_text(ctx))
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        ctx.probe = lambda argv, **kw: (
            SimpleNamespace(returncode=0, stdout="", stderr="")
            if argv[0] == "systemctl" else
            SimpleNamespace(returncode=0, stdout=self._status(
                gpu_sclk_max_mhz=2500, gpu_sclk_active_mhz=2500,
                cpu_boost=0, cpu_boost_active=0,
                cpu_max_mhz=2000, cpu_max_active_mhz=2000)))
        c = step.check(ctx)
        assert c.state == OK, c

    def test_an_active_value_that_disagrees_with_the_config_is_DRIFT(
            self, linux_facts, empty_repo, tmp_path):
        """The GTT-style 'wanted vs active' read-back -- except here a
        mismatch is never 'staged, pending reboot': the write already ran
        and either took or it did not."""
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500)
        self._install_helper(ctx)
        ctx.write(step._conf_path(ctx), step._conf_text(ctx))
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        ctx.probe = lambda argv, **kw: (
            SimpleNamespace(returncode=0, stdout="", stderr="")
            if argv[0] == "systemctl" else
            SimpleNamespace(returncode=0, stdout=self._status(
                gpu_sclk_max_mhz=2500, gpu_sclk_active_mhz=2900)))
        c = step.check(ctx)
        assert c.state == DRIFT
        assert "2500" in c.detail and "2900" in c.detail

    def test_unparsable_status_is_DRIFT_not_a_crash(self, linux_facts, empty_repo,
                                                     tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   cpu_boost=True)
        self._install_helper(ctx)
        ctx.write(step._conf_path(ctx), step._conf_text(ctx))
        ctx.write(step.UNIT_PATH, step._unit_text(ctx))
        ctx.probe = lambda argv, **kw: (
            SimpleNamespace(returncode=0, stdout="", stderr="")
            if argv[0] == "systemctl" else
            SimpleNamespace(returncode=0, stdout="not json at all"))
        c = step.check(ctx)
        assert c.state == DRIFT

    def test_apply_writes_conf_and_unit_and_restarts_the_service(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500, cpu_boost=False,
                                   cpu_max_mhz=2000)
        issued = []
        ctx.sudo = lambda argv, **kw: issued.append([str(a) for a in argv])
        step.apply(ctx)
        conf = ctx.read(step._conf_path(ctx))
        assert "GPU_SCLK_MAX_MHZ=2500" in conf
        assert "CPU_BOOST=0" in conf
        assert "CPU_MAX_MHZ=2000" in conf
        unit = ctx.read(step.UNIT_PATH)
        assert "Before=llama-swap.service" in unit
        assert "ExecStart=" in unit and "llmstack-powercap apply" in unit
        # Ordering-only, never Requires=/BindsTo= -- a knob this box's
        # hardware cannot honour must not stop llama-swap from starting.
        assert "Requires=" not in unit and "BindsTo=" not in unit
        assert any("daemon-reload" in " ".join(a) for a in issued)
        assert any("restart" in a and step.UNIT_NAME in a for a in issued)

    def test_apply_is_idempotent_the_second_time_round(self, linux_facts,
                                                        empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   cpu_boost=True)
        ctx.sudo = lambda argv, **kw: None
        step.apply(ctx)
        issued = []
        ctx.sudo = lambda argv, **kw: issued.append([str(a) for a in argv])
        step.apply(ctx)
        # No daemon-reload the second time: neither file's content changed.
        assert not any("daemon-reload" in " ".join(a) for a in issued)
        # It still restarts the unit every apply -- the config could have
        # changed underneath an already-running RemainAfterExit unit.
        assert any("restart" in a for a in issued)

    def test_apply_passes_check_false_to_the_restart(
            self, linux_facts, empty_repo, tmp_path):
        # A knob this box's hardware cannot honour (out-of-range MHz, no
        # boost file, ...) is exactly what makes `llmstack-powercap apply`
        # exit nonzero -- see gateway/tests/test_powercap.py -- and that
        # must surface through check()'s own DRIFT reading of `status`, not
        # as a crashed `systemctl restart` that raises StepFailed and aborts
        # the whole per-host apply run before `services`/`health` ever get a
        # turn. Mirrors the sibling Power step's own
        # `systemctl restart systemd-logind` call.
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500)
        calls = []
        ctx.sudo = lambda argv, **kw: calls.append(([str(a) for a in argv], kw))
        step.apply(ctx)
        restart = next(c for c in calls if "restart" in c[0])
        assert restart[1].get("check") is False

    def test_conf_value_formatting(self):
        from fleetctl.steps.system import PowerCap

        assert PowerCap._conf_value(None) == ""
        assert PowerCap._conf_value(True) == "1"
        assert PowerCap._conf_value(False) == "0"
        assert PowerCap._conf_value(2500) == "2500"

    def test_check_never_writes_and_a_dry_run_apply_writes_nothing(
            self, linux_facts, empty_repo, tmp_path):
        step, ctx = self._step_ctx(linux_facts, empty_repo, tmp_path,
                                   gpu_sclk_max_mhz=2500)
        ctx.probe = lambda argv, **kw: None
        step.check(ctx)
        assert not ctx.path(ctx.plan["paths"]["etc"]).exists()
        ctx.dry_run = True
        step.run(ctx)
        assert not ctx.path(step.UNIT_PATH).exists()
