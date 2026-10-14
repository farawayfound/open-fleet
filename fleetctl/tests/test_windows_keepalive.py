"""A Windows gateway that exits has to come back, and nothing was doing that.

gpu-laptop-5, 2026-09-27: the laptop was in Modern Standby, the network
dropped, the gateway's self-watchdog exited the process on purpose, and the
scheduled task -- ONSTART, RestartCount 99 -- never started it again. Measured
on that machine the same day: RestartCount does not react to an action that
ran and returned non-zero, and a repetition with MultipleInstances IgnoreNew
does exactly what is wanted. It is refused while the task runs and honoured
once it has gone.

So the task gets a second trigger. These tests pin what makes that safe to
roll onto boxes that are serving, and onto a laptop its owner also works at:
a task with its keepalive is left alone, a task without one is given it
WITHOUT being stopped, a task this account cannot read is never decided
about, and a task its owner has switched off stays off.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fleetctl import planner
from fleetctl import steps as steps_mod
from fleetctl.runner import Ctx
from fleetctl.steps import BLOCKED, DRIFT, MISSING, OK
from fleetctl.steps import services as services_mod

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"

BOOT = "<BootTrigger><Enabled>true</Enabled></BootTrigger>"


def repeat(interval="PT5M", enabled=True):
    return ("<TimeTrigger><Repetition><Interval>%s</Interval>"
            "<StopAtDurationEnd>false</StopAtDurationEnd></Repetition>"
            "<StartBoundary>2026-09-27T00:00:00</StartBoundary>"
            "<Enabled>%s</Enabled></TimeTrigger>"
            % (interval, "true" if enabled else "false"))


def task_xml(*triggers, policy="IgnoreNew", enabled=True, namespace=True):
    """What `schtasks /Query /TN <name> /XML` prints, cut down to the parts
    that are read -- including the declaration that claims UTF-16 about
    text that has already been decoded, and the default namespace."""
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2"%s>\n'
        "  <Triggers>%s</Triggers>\n"
        "  <Settings>\n"
        "    <MultipleInstancesPolicy>%s</MultipleInstancesPolicy>\n"
        "    <RestartOnFailure><Interval>PT1M</Interval><Count>99</Count>"
        "</RestartOnFailure>\n"
        "    <Enabled>%s</Enabled>\n"
        "  </Settings>\n"
        "</Task>\n"
        % (' xmlns="%s"' % NS if namespace else "", "".join(triggers), policy,
           "true" if enabled else "false"))


BOOT_ONLY = task_xml(BOOT)
WITH_KEEPALIVE = task_xml(BOOT, repeat())

NOT_FOUND = (1, "", "ERROR: The system cannot find the file specified.")
DENIED = (1, "", "ERROR: Access is denied.")


def services():
    return next(s for s in steps_mod.catalogue() if s.id == "services")


def ctx_for(facts, repo, tmp_path):
    plan, _ = planner.build(facts, repo=repo)
    return Ctx(plan, facts, repo=repo, root=None)


def answers(by_task: dict):
    """A ctx.probe that answers `schtasks /Query /TN <name> /XML` from a
    table of {task name: (returncode, stdout, stderr)}."""
    def probe(argv, timeout=60, env=None):
        assert list(argv[:2]) == ["schtasks", "/Query"], argv
        rc, out, err = by_task[argv[3]]
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)
    return probe


def both(gateway, swap=None):
    return answers({"llm-gateway": gateway,
                    "llama-swap": swap if swap is not None else (0, WITH_KEEPALIVE, "")})


def state_of(ctx, xml):
    ctx.probe = answers({"llm-gateway": (0, xml, "")})
    return services()._task_state(ctx, "llm-gateway")


def recorder(ctx, rc=0, stderr=""):
    calls: list[list[str]] = []
    envs: list[dict | None] = []

    def run(argv, **kw):
        calls.append([str(a) for a in argv])
        envs.append(kw.get("env"))
        failed = rc if argv[0] == "powershell" else 0
        return SimpleNamespace(returncode=failed, stdout="",
                               stderr=stderr if failed else "")
    ctx.run = run
    return calls, envs


@pytest.fixture
def ctx(windows_facts, empty_repo, tmp_path):
    return ctx_for(windows_facts, empty_repo, tmp_path)


# --------------------------------------------------------------------------
class TestCheckCanTellATaskThatWillNotComeBack:
    def test_both_tasks_with_a_keepalive_are_ok(self, ctx):
        ctx.probe = both((0, WITH_KEEPALIVE, ""))
        result = services().check(ctx)
        assert result.state == OK
        assert result.plan == []

    def test_a_boot_only_task_is_drift_and_the_plan_says_nothing_stops(self, ctx):
        """The task every Windows box in the fleet had before this."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        result = services().check(ctx)
        assert result.state == DRIFT
        assert len(result.plan) == 1
        assert "llm-gateway" in result.plan[0]
        assert "keepalive" in result.plan[0]
        assert "not stopped" in result.plan[0]

    def test_restart_count_alone_does_not_count_as_a_keepalive(self, ctx):
        """RestartOnFailure carries an <Interval> of its own. It must not be
        mistaken for a repetition -- it is the setting that failed."""
        assert "<RestartOnFailure><Interval>PT1M</Interval>" in BOOT_ONLY
        assert state_of(ctx, BOOT_ONLY)[0] == services_mod.BARE

    def test_a_repetition_inside_a_disabled_trigger_is_not_a_keepalive(self, ctx):
        """A flat search for <Repetition> is satisfied by this, and reports
        a task that will never come back as one that will."""
        xml = task_xml(BOOT, repeat(enabled=False))
        assert "<Repetition><Interval>PT5M</Interval>" in xml
        assert state_of(ctx, xml) == (services_mod.BARE, "no keepalive trigger")

    def test_one_live_repetition_among_disabled_ones_is_enough(self, ctx):
        xml = task_xml(BOOT, repeat(enabled=False), repeat("PT2M"))
        assert state_of(ctx, xml)[0] == services_mod.KEPT

    def test_a_repetition_that_may_run_in_parallel_is_not_a_keepalive(self, ctx):
        """Without IgnoreNew a repetition starts a second gateway every five
        minutes, each failing to bind the port behind the first."""
        state, why = state_of(ctx, task_xml(BOOT, repeat(), policy="Parallel"))
        assert state == services_mod.BARE
        assert "Parallel" in why

    def test_the_namespace_is_not_what_is_matched(self, ctx):
        """Task XML has a default namespace, and which one is Microsoft's."""
        assert state_of(ctx, task_xml(BOOT, repeat(), namespace=False))[0] \
            == services_mod.KEPT
        assert state_of(ctx, task_xml(BOOT, namespace=False))[0] == services_mod.BARE

    def test_an_absent_task_is_missing(self, ctx):
        ctx.probe = both(NOT_FOUND, NOT_FOUND)
        result = services().check(ctx)
        assert result.state == MISSING
        assert len(result.plan) == 2

    def test_access_denied_stays_blocked_and_is_never_called_missing(self, ctx):
        """An unelevated run cannot read a SYSTEM task. That is `blocked`.
        Read as absent, the apply that follows deletes a running task."""
        ctx.probe = both(DENIED, DENIED)
        result = services().check(ctx)
        assert result.state == BLOCKED
        assert "llm-gateway" in result.detail

    def test_no_schtasks_at_all_is_blocked(self, ctx):
        """The Windows plan checked on a Linux runner: probe() answers None."""
        ctx.probe = lambda argv, timeout=60, env=None: None
        assert services().check(ctx).state == BLOCKED

    @pytest.mark.parametrize("answer", ["", "SUCCESS: nothing to show", "<Task><Triggers>"])
    def test_an_answer_that_is_not_task_xml_decides_nothing(self, ctx, answer):
        """Calling it bare would re-harden the task on every run; calling it
        kept would be a guess."""
        assert state_of(ctx, answer)[0] == services_mod.OPAQUE
        ctx.probe = both((0, answer, ""))
        assert services().check(ctx).state == BLOCKED


# --------------------------------------------------------------------------
class TestApplyGivesAKeepaliveWithoutStoppingAnything:
    def test_a_boot_only_task_is_hardened_in_place(self, ctx):
        ctx.probe = both((0, BOOT_ONLY, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)

        verbs = [c[1] for c in calls if c[0] == "schtasks"]
        assert "/End" not in verbs
        assert "/Delete" not in verbs
        assert "/Create" not in verbs

        ps = [c for c in calls if c[0] == "powershell"]
        assert len(ps) == 1
        script = ps[0][-1]
        assert "$task = Get-ScheduledTask -TaskName 'llm-gateway'" in script
        assert "Set-ScheduledTask -TaskName 'llm-gateway' -Trigger $t -Settings $s" in script
        assert "-AtStartup" in script                     # the boot trigger stays
        assert "-RepetitionInterval (New-TimeSpan -Minutes 5)" in script
        assert "$s.MultipleInstances = 'IgnoreNew'" in script
        assert "RepetitionDuration" not in script          # absent means indefinitely
        assert "$s.StopIfGoingOnBatteries = $false" in script   # a laptop, unplugged

        # llama-swap had its keepalive: not one command names it.
        assert not any("llama-swap" in " ".join(c) for c in calls)

    def test_the_settings_it_already_has_are_changed_not_replaced(self, ctx):
        """A fresh settings set would put back every default the owner had
        changed, `Enabled` among them."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)
        script = next(c for c in calls if c[0] == "powershell")[-1]
        assert "$s = $task.Settings" in script
        assert "New-ScheduledTaskSettingsSet" not in script
        assert "Enabled =" not in script

    def test_the_triggers_already_there_are_kept(self, ctx):
        """-Trigger replaces the whole collection. What the task already had
        is read back and passed in again, and each of ours is added only if
        the task has no LIVE one of its kind -- so a trigger somebody added
        by hand survives, and a second pass adds nothing twice."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)
        script = next(c for c in calls if c[0] == "powershell")[-1]
        assert "$t = @($task.Triggers)" in script
        assert script.count("$t +=") == 2
        assert script.index("MSFT_TaskBootTrigger") < script.index("-AtStartup")
        assert script.index("$_.Repetition.Interval") < script.index("-RepetitionInterval")
        assert script.count("$_.Enabled -and") == 2        # a disabled one does not count
        assert "{" + "name" + "}" not in script

    def test_it_is_started_now_rather_than_at_the_next_repetition(self, ctx):
        """The health step waits thirty seconds, not five minutes. On a task
        already running, IgnoreNew refuses this and nothing changes."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)
        assert [c[3] for c in calls if c[:2] == ["schtasks", "/Run"]] == ["llm-gateway"]

    def test_a_new_task_gets_both_triggers_at_creation(self, ctx):
        ctx.probe = both(NOT_FOUND, NOT_FOUND)
        calls, _ = recorder(ctx)
        services().apply(ctx)
        created = [c[3] for c in calls if c[:2] == ["schtasks", "/Create"]]
        assert created == ["llm-gateway", "llama-swap"]
        scripts = [c[-1] for c in calls if c[0] == "powershell"]
        assert len(scripts) == 2
        assert all("-RepetitionInterval" in s and "-AtStartup" in s for s in scripts)
        assert any("every 5 min" in a for a in ctx.actions)
        # llama-swap first: it is the gateway's upstream.
        assert [c[3] for c in calls if c[:2] == ["schtasks", "/Run"]] == [
            "llama-swap", "llm-gateway"]

    def test_tasks_that_have_everything_are_not_touched_at_all(self, ctx):
        """A healthy box: no command of any kind. This is the run every
        provisioned peer gets from here on."""
        ctx.probe = both((0, WITH_KEEPALIVE, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)
        assert calls == []

    def test_a_task_that_cannot_be_read_is_not_touched(self, ctx):
        ctx.probe = both(DENIED, DENIED)
        calls, _ = recorder(ctx)
        services().apply(ctx)
        assert calls == []

    def test_a_hardening_that_fails_is_said_out_loud(self, ctx):
        """It was `check=False` and silence. A keepalive that never took is
        exactly the state this whole change exists to end."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        recorder(ctx, rc=1, stderr="Set-ScheduledTask : Access is denied.\n"
                                   "At line:1 char:400\n+ ... Set-ScheduledTask ...\n"
                                   "    + CategoryInfo : PermissionDenied")
        services().apply(ctx)
        assert len(ctx.notes) == 1
        assert "llm-gateway" in ctx.notes[0]
        assert ctx.notes[0].endswith("Set-ScheduledTask : Access is denied.")
        assert not any("keepalive added" in a for a in ctx.actions)

    def test_a_new_task_whose_hardening_failed_does_not_claim_a_keepalive(self, ctx):
        """Created, and it will start at boot -- and that is all. The line
        must not say otherwise."""
        ctx.probe = both(NOT_FOUND)
        recorder(ctx, rc=1, stderr="Set-ScheduledTask : Access is denied.")
        services().apply(ctx)
        said = [a for a in ctx.actions if "llm-gateway" in a]
        assert said == ["scheduled task llm-gateway (ONSTART, SYSTEM) -- "
                        "WITHOUT its keepalive"]
        assert not any("every 5 min" in a for a in ctx.actions)
        assert len(ctx.notes) == 1

    def test_the_powershell_child_is_given_windows_powershells_own_modules(self, ctx):
        """Started from pwsh 7 it would inherit a module path it cannot use."""
        ctx.probe = both((0, BOOT_ONLY, ""))
        calls, envs = recorder(ctx)
        services().apply(ctx)
        env = envs[[c[0] for c in calls].index("powershell")]
        assert env and r"WindowsPowerShell\v1.0\Modules" in env["PSModulePath"]


# --------------------------------------------------------------------------
class TestATaskItsOwnerSwitchedOffStaysOff:
    """A keepalive cannot tell a crash from somebody ending the task on
    purpose, and on a laptop the owner also works at, on purpose is the
    ordinary case. Ending a task no longer keeps it off. Disabling it does,
    and nothing here undoes that."""

    OFF = task_xml(BOOT, repeat(), enabled=False)
    OFF_AND_BARE = task_xml(BOOT, enabled=False)

    def test_it_is_its_own_state(self, ctx):
        assert state_of(ctx, self.OFF) == (services_mod.DISABLED,
                                           "switched off on this box")

    def test_a_trigger_switched_off_is_not_the_task_switched_off(self, ctx):
        assert state_of(ctx, task_xml(BOOT, repeat(enabled=False)))[0] \
            == services_mod.BARE

    def test_check_is_content_and_says_why(self, ctx):
        ctx.probe = both((0, self.OFF, ""), (0, self.OFF, ""))
        result = services().check(ctx)
        assert result.state == OK
        assert result.plan == []
        assert "llm-gateway, llama-swap switched off" in result.detail
        assert "Enable-ScheduledTask" in result.detail

    @pytest.mark.parametrize("xml", [OFF, OFF_AND_BARE])
    def test_apply_does_not_enable_harden_or_start_it(self, ctx, xml):
        """Not even to give it the keepalive it lacks: that waits until it
        is switched back on."""
        ctx.probe = both((0, xml, ""), (0, xml, ""))
        calls, _ = recorder(ctx)
        services().apply(ctx)
        assert calls == []

    def test_the_other_task_is_still_looked_after(self, ctx):
        ctx.probe = both((0, self.OFF, ""), (0, BOOT_ONLY, ""))
        result = services().check(ctx)
        assert result.state == DRIFT
        assert [ln.split()[1] for ln in result.plan] == ["llama-swap"]
        calls, _ = recorder(ctx)
        services().apply(ctx)
        assert not any("llm-gateway" in " ".join(c) for c in calls)
        assert [c[3] for c in calls if c[:2] == ["schtasks", "/Run"]] == ["llama-swap"]


# --------------------------------------------------------------------------
class TestTheIntervalIsOneNumber:
    def test_the_plan_the_script_and_the_constant_agree(self, ctx):
        assert services_mod.KEEPALIVE_MINUTES == 5
        assert "-Minutes 5)" in services_mod._TASK_TRIGGERS
        ctx.probe = both(NOT_FOUND, NOT_FOUND)
        assert all("every 5 min" in line for line in services().check(ctx).plan)

    @pytest.mark.parametrize("interval", ["PT2M", "PT1H", "P1D"])
    def test_an_interval_somebody_chose_by_hand_is_still_a_keepalive(
            self, ctx, interval):
        assert state_of(ctx, task_xml(BOOT, repeat(interval)))[0] == services_mod.KEPT
