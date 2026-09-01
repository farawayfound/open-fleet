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
              profile was renamed out from under it.
"""
from __future__ import annotations

from .. import shapes
from . import BLOCKED, DRIFT, MISSING, OK, Check, Step

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
            todo, opaque = [], []
            for name in self._tasks(ctx):
                r = ctx.probe(["schtasks", "/Query", "/TN", name])
                if r is None:
                    opaque.append(name)
                elif r.returncode == 0:
                    continue
                elif "access is denied" in ((r.stderr or "") + (r.stdout or "")).lower():
                    # A hardened task is INVISIBLE, not absent. apu-tablet-2's
                    # tasks carry an ACL that refuses a non-elevated query, and
                    # reading that as "missing" is how a verify reports a
                    # perfectly good box as unprovisioned -- and how the apply
                    # that followed would delete and recreate a running task.
                    # Same distinction deploy-gateway.sh draws between a box
                    # that is offline and one that refused our ssh key.
                    opaque.append(name)
                else:
                    todo.append(f"create SYSTEM task {name} (ONSTART)")
            if opaque:
                return Check(BLOCKED,
                             f"cannot read {', '.join(opaque)} without elevation "
                             f"(the task may well be there -- Access is denied is "
                             f"not the same answer as not found)",
                             todo)
            if not todo:
                return Check(OK, f"{len(self._tasks(ctx))} scheduled task(s) present")
            return Check(MISSING, f"{len(todo)} task(s) to create", todo)

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
            # Only the tasks check() found absent. Recreating every task
            # because ONE was missing meant that enabling llama-swap on a
            # working Windows peer stopped and deleted its running
            # llm-gateway task as a side effect. A task that answers /Query
            # is left exactly as it is.
            created = []
            for name, cmd in self._tasks(ctx).items():
                r = ctx.probe(["schtasks", "/Query", "/TN", name])
                if r is not None and r.returncode == 0:
                    continue
                created.append(name)
                ctx.run(["schtasks", "/End", "/TN", name], system=True, check=False)
                ctx.run(["schtasks", "/Delete", "/TN", name, "/F"],
                        system=True, check=False)
                ctx.run(["schtasks", "/Create", "/TN", name, "/TR", f'"{cmd}"',
                         "/SC", "ONSTART", "/RU", "SYSTEM", "/RL", "HIGHEST", "/F"],
                        system=True)
                # StartWhenAvailable so a missed trigger is still honoured, no
                # execution time limit (a service is not a job to be killed
                # after three days), and a relaunch if the action fails.
                ctx.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                         f"$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
                         f"-DontStopIfGoingOnBatteries -StartWhenAvailable "
                         f"-MultipleInstances IgnoreNew -RestartCount 3 "
                         f"-RestartInterval (New-TimeSpan -Minutes 1) "
                         f"-ExecutionTimeLimit ([TimeSpan]::Zero); "
                         f"Set-ScheduledTask -TaskName '{name}' -Settings $s | Out-Null"],
                        system=True, check=False)
                ctx.did(f"scheduled task {name} (ONSTART, SYSTEM)")
            for name in reversed(created):
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
