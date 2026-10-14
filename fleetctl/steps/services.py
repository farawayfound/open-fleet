"""Supervision, three ways: systemd units, crontab lines, scheduled tasks.

One job -- "keep the gateway running, and start it at boot" -- expressed by
whichever supervisor the box actually has. The three are not
interchangeable and the differences are load-bearing:

  systemd     a unit reads EnvironmentFile directly and has ExecStart, so no
              wrapper script is needed. It is also the only one of the three
              that can express "after llama-swap".
  cron        chosen over launchd on every Mac here for one reason: a
              LaunchAgent cannot be bootstrapped over ssh without an Aqua
              session, and these boxes are provisioned over ssh. `@reboot`
              plus a 5-minute keepalive is the whole supervisor.
  schtasks    ONSTART, as SYSTEM, so the gateway is up before anyone logs in.
              Deliberately NOT a per-user task: a SYSTEM task cannot depend
              on anything inside a user profile, and gpu-desktop-1's did until the
              profile was renamed out from under it. Plus a five-minute
              keepalive, the same shape as the Macs' -- see KEEPALIVE_MINUTES.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from .. import shapes
from . import BLOCKED, DRIFT, MISSING, OK, Check, Step

# How often Task Scheduler is asked to start a task that is not running.
#
# It exists because "restart on failure" does not do what its name says.
# gpu-laptop-5's gateway exited at 03:28 on 2026-09-27 -- the laptop was in
# Modern Standby, the network dropped, uvicorn's accept loop died with
# WinError 64 and the self-watchdog called os._exit(1), which is its job --
# and the task carried RestartCount 99. Nothing started it again, asleep or
# awake; the box read offline until someone looked.
#
# Measured the same day, on the same machine, with two throwaway tasks:
#
#   RestartCount 3, one minute apart, action exits 1
#       ran once. Never restarted. RestartCount answers a task that could not
#       be LAUNCHED; an action that ran and returned non-zero has, as far as
#       Task Scheduler is concerned, finished.
#   repeat every minute, MultipleInstances IgnoreNew, action exits 1 after 150 s
#       started 12:44:21. The 12:45:21 and 12:46:21 repetitions were refused
#       (0x800710E0) because it was running. It exited at 12:46:51 and the
#       12:47:21 repetition started it again. Never two copies.
#
# So the supervisor on Windows is the repetition, as `*/5 * * * *` is on the
# Macs. While the gateway is up it costs nothing: Task Scheduler consults its
# own table and declines, and no process is created. RestartCount stays,
# for the failure it does cover.
#
# What it costs is the old way of stopping one. A keepalive cannot tell a
# crash from somebody ending the task on purpose, and on a laptop the owner
# also works at, "on purpose" is the ordinary case. So the off switch is the
# task's own: DISABLED. A disabled task has no trigger that fires, and
# nothing here enables one, starts one, or touches one -- see DISABLED below.
KEEPALIVE_MINUTES = 5

# What every fleet task carries, set on the settings the task ALREADY has
# rather than on a fresh set. A fresh one would put back every default the
# owner had changed, `Enabled` among them. IgnoreNew is what makes the
# keepalive safe: without it a repetition would start a second gateway
# beside the first.
_TASK_SETTINGS = (
    "$s = $task.Settings; "
    "$s.DisallowStartIfOnBatteries = $false; "
    "$s.StopIfGoingOnBatteries = $false; "
    "$s.StartWhenAvailable = $true; "
    "$s.MultipleInstances = 'IgnoreNew'; "
    "$s.RestartCount = 99; $s.RestartInterval = 'PT1M'; "
    "$s.ExecutionTimeLimit = 'PT0S'")

# The two triggers a fleet task must have, each added only if the task has
# nothing of its kind. -Trigger REPLACES the whole collection, so the ones
# already there are read back and passed in again: a trigger somebody added
# by hand is theirs, and a hardening pass that quietly dropped it would be
# this tool deciding what runs on a box it was only asked to keep alive.
#
# No -RepetitionDuration: on Windows 10 1607 and later its absence means
# "indefinitely". -At is midnight today, already in the past, so the
# repetition is live the moment the task is saved.
_TASK_TRIGGERS = (
    "$t = @($task.Triggers); "
    "if (-not ($t | Where-Object { $_.Enabled -and $_.CimClass.CimClassName "
    "-eq 'MSFT_TaskBootTrigger' })) { $t += New-ScheduledTaskTrigger -AtStartup }; "
    "if (-not ($t | Where-Object { $_.Enabled -and $_.Repetition -and "
    "$_.Repetition.Interval })) "
    "{ $t += New-ScheduledTaskTrigger -Once -At (Get-Date).Date "
    "-RepetitionInterval (New-TimeSpan -Minutes " + str(KEEPALIVE_MINUTES) + ") }")

ABSENT, OPAQUE, BARE, KEPT = "absent", "opaque", "bare", "kept"
# There, and switched off by whoever owns the box. Not drift and not work:
# the one state in which the right thing to do is nothing.
DISABLED = "disabled"


def _local(tag: str) -> str:
    """`{namespace}Repetition` -> `Repetition`. Task XML has a default
    namespace, and which one is Microsoft's to change."""
    return tag.rsplit("}", 1)[-1]


def _child(node, name: str):
    return next((c for c in node if _local(c.tag) == name), None)


def _text(node, name: str, default: str = "") -> str:
    hit = _child(node, name) if node is not None else None
    return (hit.text or "").strip() if hit is not None else default


def _read_task(xml: str) -> tuple[bool, bool, str] | None:
    """(enabled, has a live keepalive, instance policy), or None if this is
    not task XML.

    Read as a document, not searched as text. A flat search for a
    repetition is satisfied by one inside a trigger somebody has DISABLED,
    and by RestartOnFailure's own <Interval> if the pattern is ever
    loosened -- and either answer reports a task that will never come back
    as one that will. Any interval counts: how often is an operator's to
    change, and a task that repeats has its keepalive.
    """
    body = re.sub(r"^\s*<\?xml[^>]*\?>", "", xml or "").strip()
    if not body:
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    settings = _child(root, "Settings")
    enabled = _text(settings, "Enabled", "true").lower() != "false"
    policy = _text(settings, "MultipleInstancesPolicy", "IgnoreNew")
    live = False
    triggers = _child(root, "Triggers")
    for trigger in (triggers if triggers is not None else ()):
        if _text(trigger, "Enabled", "true").lower() == "false":
            continue
        if _text(_child(trigger, "Repetition"), "Interval"):
            live = True
    return enabled, live, policy

SYSTEMD_GATEWAY = """\
[Unit]
Description=llmstack gateway -- auth, metering and admin for {host}
After=network-online.target{after}
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={gateway}
EnvironmentFile={envfile}

ExecStart={venv}/bin/uvicorn app:app \\
    --host ${{LLMSTACK_BIND}} --port ${{LLMSTACK_PORT}} \\
    --proxy-headers --forwarded-allow-ips 127.0.0.1 \\
    --timeout-keep-alive 75 \\
    --timeout-graceful-shutdown 25

Restart=on-failure
RestartSec=3
LimitNOFILE=65536

# Long enough to END the answers in flight, and stated here because the
# distro default cannot be trusted to be: Mint ships
# DefaultTimeoutStopSec=10s in /etc/systemd/system.conf.d/50_linuxmint.conf,
# and on apu-box-1 that turned a routine reconcile deploy into SIGKILL ten
# seconds into a drain -- two Copilot streams cut mid-body, which reaches a
# client through cloudflared as ERR_HTTP2_PROTOCOL_ERROR.
#
# The two numbers are a pair, and the ORDER matters. uvicorn gives the drain
# 25 s and then cancels whatever is left; systemd allows 30 s before it
# resorts to SIGKILL. Keeping uvicorn's the smaller one means the process
# always gets to finish on its own terms -- the app's SIGTERM hook frames a
# 503 onto every open stream (see install_drain_signal_handlers in app.py)
# and exits -- with SIGKILL left as the backstop it should be, not the
# routine outcome it was.
TimeoutStopSec=30

# The gateway shells out to `sudo systemctl restart llama-swap` when the model
# set changes, so privilege escalation cannot be blanket-disabled here. The
# sudoers grant is restricted to those exact verbs -- see hosts/linux/grants.sh.
NoNewPrivileges=no
PrivateTmp=yes
ProtectSystem=full
# /etc/default, /etc/modprobe.d and /boot are here for llmstack-gpuconf:
# ProtectSystem=full makes them read-only for this unit, and sudo does NOT
# escape a unit's mount namespace -- so without them the GPU ceiling control
# fails with what looks like a permissions bug that "should" not happen.
ReadWritePaths={state} {etc} /etc/default /etc/modprobe.d /boot

[Install]
WantedBy=multi-user.target
"""

SYSTEMD_SWAP = """\
[Unit]
Description=llama-swap -- on-demand llama.cpp model server for {host}
After=network-online.target

[Service]
Type=simple
User={user}
Group={user}
EnvironmentFile={envfile}
ExecStart={bin}/llama-swap --config {swap_config} --listen {listen} --watch-config
Restart=on-failure
RestartSec=3
LimitNOFILE=65536
LimitMEMLOCK=infinity
# A model load must not be the thing the OOM killer reaches for first, and it
# must not be descheduled behind a desktop it is not running.
Nice=-5
OOMScoreAdjust=-500
PrivateTmp=yes
ProtectSystem=full
ReadWritePaths={state} {etc}
{extra_env}
[Install]
WantedBy=multi-user.target
"""


class Services(Step):
    id = "services"
    title = "supervision"
    system = True

    # ---------------------------------------------------------------- systemd
    def _units(self, ctx) -> dict[str, str]:
        p, e = ctx.plan["paths"], ctx.plan["engine"]
        user = ctx.plan["service"]["user"] or "root"
        out = {
            "llm-gateway.service": SYSTEMD_GATEWAY.format(
                host=ctx.plan["host"]["name"], user=user, gateway=p["gateway"],
                envfile=p["envfile"], venv=p["venv"], state=p["state"], etc=p["etc"],
                after=" llama-swap.service" if e["llama_swap"] else ""),
        }
        if e["llama_swap"]:
            vis = e["visible_devices"]
            extra = f"Environment=GGML_VK_VISIBLE_DEVICES={vis}\n" if vis else ""
            out["llama-swap.service"] = SYSTEMD_SWAP.format(
                host=ctx.plan["host"]["name"], user=user, envfile=p["envfile"],
                bin=p["bin"], swap_config=p["swap_config"],
                listen=e["swap_listen"] or "127.0.0.1:8081",
                state=p["state"], etc=p["etc"], extra_env=extra)
        return out

    # ------------------------------------------------------------------- cron
    def _cron_lines(self, ctx) -> list[tuple[str, str]]:
        """(id-comment, crontab line). The id comment is how a re-run strips
        only its OWN lines: matching on 'llmstack' would eat a hand-written
        entry, and matching on the command would eat nothing after a path
        change."""
        p = ctx.plan["paths"]
        gw = shapes.join(ctx.family, p["bin"], "run-gateway.sh")
        log = shapes.join(ctx.family, p["logs"], "gateway.log")
        # pgrep matches the ENGINE, not this cron line -- a keepalive whose
        # pattern matches its own shell never restarts anything.
        start = f"pgrep -qf 'uvicorn app:app' || nohup {gw} >>{log} 2>&1 &"
        out = [("llmstack-gateway", f"@reboot {start}"),
               ("llmstack-gateway-keepalive", f"*/5 * * * * {start}")]
        if ctx.plan["engine"]["llama_swap"]:
            ls = shapes.join(ctx.family, p["bin"], "run-llama-swap.sh")
            lslog = shapes.join(ctx.family, p["logs"], "llama-swap.log")
            # NOT 'run-llama-swap': the wrapper execs the binary, so no live
            # process ever carries the wrapper's name -- that pattern matches
            # nothing (and on some crons, its own shell), so the keepalive
            # spawned a doomed duplicate every five minutes. Each duplicate
            # began its start-up preload before failing to bind the port,
            # which is what put an orphaned 21 GB llama-server on mac-laptop-1.
            # 'bin/llama-swap -config' is the binary's own argv.
            lstart = (f"pgrep -qf 'bin/llama-swap -config' || "
                      f"nohup {ls} >>{lslog} 2>&1 &")
            out += [("llmstack-llama-swap", f"@reboot {lstart}"),
                    ("llmstack-llama-swap-keepalive", f"*/5 * * * * {lstart}")]
        return out

    def _crontab(self, ctx) -> str:
        r = ctx.probe(["crontab", "-l"])
        return (r.stdout or "") if r and r.returncode == 0 else ""

    # --------------------------------------------------------------- schtasks
    def _tasks(self, ctx) -> dict[str, str]:
        p = ctx.plan["paths"]
        out = {"llm-gateway": shapes.join(ctx.family, p["bin"], "run-gateway.cmd")}
        if ctx.plan["engine"]["llama_swap"]:
            out["llama-swap"] = shapes.join(ctx.family, p["bin"],
                                            "run-llama-swap.cmd")
        return out

    @staticmethod
    def _task_state(ctx, name: str) -> tuple[str, str]:
        """(ABSENT | OPAQUE | BARE | KEPT | DISABLED, why).

        Five answers, because each wants something different done. ABSENT is
        created. BARE -- there, and without its keepalive -- is given one
        and otherwise left exactly as it is, running instance included.
        OPAQUE is a task this account may not read, and nothing is decided
        about a task nobody has looked at. DISABLED is a task its owner has
        switched off, and is left off: not hardened, not started.
        """
        r = ctx.probe(["schtasks", "/Query", "/TN", name, "/XML"])
        if r is None:
            return OPAQUE, "schtasks did not answer"
        text = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            # A hardened task is INVISIBLE, not absent. apu-tablet-2's tasks
            # carry an ACL that refuses a non-elevated query, and reading
            # that as "missing" is how a verify reports a perfectly good box
            # as unprovisioned -- and how the apply that followed would
            # delete and recreate a running task. Same distinction
            # deploy-gateway.sh draws between a box that is offline and one
            # that refused our ssh key.
            if "access is denied" in text.lower():
                return OPAQUE, "access is denied"
            return ABSENT, ""
        task = _read_task(r.stdout or "")
        if task is None:
            # It answered, and not with anything that can be read. Deciding
            # it has no keepalive would re-harden it on every run; deciding
            # it has one would be a guess.
            return OPAQUE, "schtasks answered with something that is not task XML"
        enabled, live, policy = task
        if not enabled:
            return DISABLED, "switched off on this box"
        if not live:
            return BARE, "no keepalive trigger"
        if policy.lower() != "ignorenew":
            # A repetition without IgnoreNew is a second gateway every five
            # minutes, each one failing to bind :8080 behind the first.
            return BARE, f"keepalive with MultipleInstances {policy}"
        return KEPT, ""

    @staticmethod
    def _harden(ctx, name: str) -> bool:
        """Settings and triggers onto a task that exists. Never stops it.

        Set-ScheduledTask replaces the definition and leaves the running
        instance alone -- measured, not assumed: a task given a new trigger
        set 57 s into a 150 s run finished that run on time. That is what
        lets this reach a box whose gateway is serving.

        Reported when it fails. This call used to be `check=False` and
        nothing else, and check() only asked whether the task EXISTED, so a
        hardening that never took was invisible for the life of the box.
        """
        script = (f"$ErrorActionPreference = 'Stop'; "
                  f"$task = Get-ScheduledTask -TaskName '{name}'; "
                  f"{_TASK_SETTINGS}; {_TASK_TRIGGERS}; "
                  f"Set-ScheduledTask -TaskName '{name}' -Trigger $t "
                  f"-Settings $s | Out-Null")
        # Windows PowerShell's own module path, stated. Started from a pwsh 7
        # shell it inherits one it cannot use -- the reason acl_strangers()
        # gives its child a clean environment, and the same child.
        modules = (os.environ.get("ProgramFiles", r"C:\Program Files")
                   + r"\WindowsPowerShell\Modules;"
                   + os.environ.get("SystemRoot", r"C:\Windows")
                   + r"\System32\WindowsPowerShell\v1.0\Modules")
        r = ctx.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                     script], system=True, check=False,
                    env={"PSModulePath": modules})
        if r is not None and r.returncode != 0:
            # What it SAID, which is the top: PowerShell names the fault and
            # then spends six lines pointing at where in the command it was.
            said = []
            for line in ((r.stderr or "") + (r.stdout or "")).splitlines():
                if line.lstrip().startswith(("At ", "+")):
                    break
                if line.strip():
                    said.append(line.strip())
            ctx.note(f"could not set the keepalive on {name} "
                     f"(exit {r.returncode}): " + " ".join(said)[:300])
            return False
        return True

    # ------------------------------------------------------------------ check
    def check(self, ctx) -> Check:
        mech = ctx.plan["platform"]["service"]
        if mech == "systemd":
            want = self._units(ctx)
            todo = []
            for name, text in want.items():
                path = f"/etc/systemd/system/{name}"
                if ctx.read(path) != text:
                    todo.append(f"write {path}")
            for name in want:
                r = ctx.probe(["systemctl", "is-enabled", name])
                if r is None or r.returncode != 0:
                    todo.append(f"systemctl enable --now {name}")
            if not todo:
                return Check(OK, f"{len(want)} unit(s) installed and enabled")
            return Check(DRIFT, f"{len(todo)} change(s)", todo)

        if mech == "cron":
            have = self._crontab(ctx)
            want = self._cron_lines(ctx)
            todo = [f"add crontab line: {tag}" for tag, line in want
                    if f"{line} # {tag}" not in have]
            if not todo:
                return Check(OK, f"{len(want)} crontab line(s) present")
            return Check(DRIFT if have else MISSING, f"{len(todo)} line(s) to add",
                         todo)

        if mech == "schtasks":
            todo, opaque, bare, off = [], [], [], []
            for name in self._tasks(ctx):
                state, why = self._task_state(ctx, name)
                if state == OPAQUE:
                    opaque.append(name)
                elif state == DISABLED:
                    off.append(name)
                elif state == ABSENT:
                    todo.append(f"create SYSTEM task {name} (ONSTART, and "
                                f"every {KEEPALIVE_MINUTES} min if it is not "
                                f"running)")
                elif state == BARE:
                    bare.append(name)
                    todo.append(f"give {name} its keepalive ({why}); a "
                                f"running {name} is not stopped")
            if opaque:
                return Check(BLOCKED,
                             f"cannot read {', '.join(opaque)} without elevation "
                             f"(the task may well be there -- Access is denied is "
                             f"not the same answer as not found)",
                             todo)
            if not todo:
                if off:
                    # Said, because a run that ends `ok` over a gateway that
                    # is not running should say why it is content.
                    return Check(OK, f"{', '.join(off)} switched off on this "
                                     f"box by its owner -- left off "
                                     f"(Enable-ScheduledTask to bring back)")
                return Check(OK, f"{len(self._tasks(ctx))} scheduled task(s) "
                                 f"present, each with its keepalive")
            if len(bare) == len(todo):
                return Check(DRIFT, f"{len(bare)} task(s) with nothing to start "
                                    f"them again after an exit", todo)
            return Check(MISSING, f"{len(todo) - len(bare)} task(s) to create"
                         + (f", {len(bare)} to give a keepalive" if bare else ""),
                         todo)

        return Check(BLOCKED, f"no supervisor here (service manager: {mech!r}) -- "
                              f"nothing will start this gateway at boot")

    # ------------------------------------------------------------------ apply
    def apply(self, ctx) -> None:
        mech = ctx.plan["platform"]["service"]
        if mech == "systemd":
            for name, text in self._units(ctx).items():
                if ctx.write(f"/etc/systemd/system/{name}", text):
                    ctx.did(f"wrote /etc/systemd/system/{name}")
            ctx.sudo(["systemctl", "daemon-reload"])
            # llama-swap first: it is the gateway's upstream, and a gateway
            # that starts against a dead upstream reports the box as having
            # no models for as long as its engine cache lives.
            for name in reversed(list(self._units(ctx))):
                ctx.sudo(["systemctl", "enable", "--now", name])
                ctx.did(f"enabled {name}")
        elif mech == "cron":
            want = self._cron_lines(ctx)
            tags = {tag for tag, _ in want}
            keep = [ln for ln in self._crontab(ctx).splitlines()
                    if not any(ln.rstrip().endswith(f"# {t}") for t in tags)]
            new = keep + [f"{line} # {tag}" for tag, line in want]
            text = "\n".join(new).strip() + "\n"
            if ctx.dry_run:
                ctx.actions.append(f"would install {len(want)} crontab line(s)")
                return
            ctx.run(["crontab", "-"], input_text=text)
            ctx.did(f"installed {len(want)} crontab line(s)")
        elif mech == "schtasks":
            # Only the tasks check() found absent are created. Recreating
            # every task because ONE was missing meant that enabling
            # llama-swap on a working Windows peer stopped and deleted its
            # running llm-gateway task as a side effect. A task that is
            # there is never ended and never deleted: the most that happens
            # to it is a keepalive it did not have.
            touched = []
            for name, cmd in self._tasks(ctx).items():
                state, _ = self._task_state(ctx, name)
                if state in (KEPT, OPAQUE, DISABLED):
                    continue
                touched.append(name)
                if state == ABSENT:
                    ctx.run(["schtasks", "/End", "/TN", name], system=True,
                            check=False)
                    ctx.run(["schtasks", "/Delete", "/TN", name, "/F"],
                            system=True, check=False)
                    ctx.run(["schtasks", "/Create", "/TN", name, "/TR",
                             f'"{cmd}"', "/SC", "ONSTART", "/RU", "SYSTEM",
                             "/RL", "HIGHEST", "/F"], system=True)
                # StartWhenAvailable so a trigger missed while the box slept
                # is honoured when it wakes, no execution time limit (a
                # service is not a job to be killed after three days), both
                # battery flags off (unplugging a laptop must not stop its
                # gateway), and the keepalive -- see KEEPALIVE_MINUTES for
                # why RestartCount alone was never enough.
                kept = self._harden(ctx, name)
                if state == ABSENT and kept:
                    ctx.did(f"scheduled task {name} (ONSTART + every "
                            f"{KEEPALIVE_MINUTES} min, SYSTEM)")
                elif state == ABSENT:
                    # Created, and it will start at boot -- and that is all.
                    # Saying "every 5 min" here would be the silent failure
                    # this step was rewritten to end; the note says why.
                    ctx.did(f"scheduled task {name} (ONSTART, SYSTEM) -- "
                            f"WITHOUT its keepalive")
                elif kept:
                    ctx.did(f"{name}: keepalive added, every "
                            f"{KEEPALIVE_MINUTES} min; nothing was stopped")
            # Started now, not at the next repetition: the health step that
            # follows waits thirty seconds, not five minutes. On a task that
            # is already running this is refused by IgnoreNew and changes
            # nothing.
            for name in reversed(touched):
                ctx.run(["schtasks", "/Run", "/TN", name], system=True, check=False)


class Restart(Step):
    """Not in ORDER -- invoked directly by `fleetctl update`, which ships new
    gateway sources onto a box that is already provisioned and has to bounce
    the service without reprovisioning anything."""

    id = ""
    title = "restart"

    @staticmethod
    def bounce(ctx) -> None:
        mech = ctx.plan["platform"]["service"]
        if mech == "systemd":
            ctx.sudo(["systemctl", "restart", "llm-gateway"])
        elif mech == "cron":
            ctx.run(["pkill", "-f", "uvicorn app:app"], check=False)
            gw = shapes.join(ctx.family, ctx.plan["paths"]["bin"], "run-gateway.sh")
            log = shapes.join(ctx.family, ctx.plan["paths"]["logs"], "gateway.log")
            # The keepalive would pick it up within five minutes anyway; this
            # is so `update` does not return before the box is back.
            ctx.run(["bash", "-c", f"nohup {gw} >>{log} 2>&1 </dev/null &"],
                    check=False)
        elif mech == "schtasks":
            ctx.run(["schtasks", "/End", "/TN", "llm-gateway"],
                    system=True, check=False)
            ctx.run(["schtasks", "/Run", "/TN", "llm-gateway"], system=True)
