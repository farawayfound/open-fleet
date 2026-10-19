"""Steps that change the machine rather than the install.

Packages, the service account, the sudoers grant, never-sleep, the firewall,
the GPU memory ceiling and the CPU/GPU power cap. All of them are `system`
steps: they touch things outside the plan's own paths, so they are refused
under `--root` and reported as blocked without root. That is deliberate --
see steps/__init__.py on why `blocked` and `skipped` are different words.
"""
from __future__ import annotations

import ipaddress
import json

from .. import shapes
from . import BLOCKED, DRIFT, MISSING, OK, SKIPPED, Check, Step


class Packages(Step):
    id = "packages"
    title = "distro packages"
    system = True

    def _wanted(self, ctx) -> tuple[str, ...]:
        fam = ctx.plan["platform"]["package_manager"]
        pkgs = list(shapes.BASE_PACKAGES.get(fam, ()))
        if ctx.plan["engine"].get("build_from_source"):
            pkgs += list(shapes.BUILD_PACKAGES.get(fam, ()))
        return tuple(dict.fromkeys(pkgs))

    def wanted(self, ctx) -> bool:
        return bool(self._wanted(ctx))

    def needs_root(self, ctx) -> bool:
        # Homebrew is not merely allowed to run unprivileged -- it REFUSES to
        # run as root, and installs into a prefix the console user owns. A
        # Mac's packages step gated on root reported `blocked (needs root)`
        # for a package that was installed, about a command no amount of sudo
        # would have helped. Every other package manager here does need it.
        return ctx.plan["platform"]["package_manager"] != "brew"

    def _installed(self, ctx, pkg: str) -> bool:
        fam = ctx.plan["platform"]["package_manager"]
        query = shapes.pkg_argv(fam, shapes.PKG_QUERY)
        if not query:
            return False
        r = ctx.probe([*query, pkg])
        if r is None:
            return False
        if fam == "apt":
            # dpkg-query exits 0 for a package that is merely KNOWN; the
            # status string is what says it is actually unpacked and
            # configured. `rc` (removed, config files kept) exits 0 too.
            return r.returncode == 0 and "install ok installed" in (r.stdout or "")
        return r.returncode == 0

    def check(self, ctx) -> Check:
        fam = ctx.plan["platform"]["package_manager"]
        if fam not in shapes.PKG_INSTALL:
            return Check(SKIPPED, f"no package manager for {fam!r} "
                                  f"-- nothing to install")
        want = self._wanted(ctx)
        missing = [p for p in want if not self._installed(ctx, p)]
        if not missing:
            return Check(OK, f"{len(want)} package(s) present")
        verb = " ".join(shapes.pkg_argv(fam, shapes.PKG_INSTALL) or ())
        return Check(MISSING, f"{len(missing)} of {len(want)} missing: "
                              + ", ".join(missing),
                     [f"{verb} {' '.join(missing)}"])

    def apply(self, ctx) -> None:
        fam = ctx.plan["platform"]["package_manager"]
        missing = [p for p in self._wanted(ctx) if not self._installed(ctx, p)]
        if not missing:
            return
        refresh = shapes.pkg_argv(fam, shapes.PKG_REFRESH)
        if refresh:
            # A refresh failure is not fatal: a box with a stale index can
            # still install everything it already has metadata for, and the
            # install below will say so plainly if it cannot.
            try:
                (ctx.run if fam == "brew" else ctx.sudo)(refresh, system=True)
            except Exception as exc:  # noqa: BLE001
                ctx.note(f"package index refresh failed ({exc.__class__.__name__}) "
                         f"-- continuing with what is cached")
        cmd = [*(shapes.pkg_argv(fam, shapes.PKG_INSTALL) or ()), *missing]
        (ctx.run if fam == "brew" else ctx.sudo)(cmd, system=True, timeout=2400)
        ctx.did(f"installed {len(missing)} package(s)")


class ServiceAccount(Step):
    id = "service-account"
    title = "service account"
    system = True

    def wanted(self, ctx) -> bool:
        # Linux only. On macOS the stack is the console user's, and on Windows
        # the scheduled tasks run as SYSTEM -- neither has an account to make.
        return ctx.family == "linux" and bool(ctx.plan["service"]["user"])

    def _wants_gpu_groups(self, ctx) -> bool:
        """Does THIS ACCOUNT open the GPU, or only this box have one?

        video and render are not cosmetic where they are needed: without them
        the Vulkan loader does not raise, it silently falls back to llvmpipe
        and every model runs on the CPU at a plausible-looking speed. But the
        account needs them only where it is the one holding the device open
        -- a box running llama.cpp under llmstack. Ollama runs under its own
        account and hub runs no engine at all, so requiring the groups
        everywhere reported permanent drift on those two. Drift that is
        always there is drift nobody reads, which is how the real kind gets
        missed. The gateway's own GPU readings come out of sysfs and need no
        group membership at all.
        """
        engine = ctx.plan["engine"]
        return (engine["kind"] == "llama.cpp"
                and engine.get("backend") not in (None, "cpu"))

    # What ROOT runs or loads from, and the service account therefore must
    # never own. /etc/sudoers.d/llmstack-fleet gives the account NOPASSWD sudo
    # on <bin>/llmstack-gpuconf, a root systemd oneshot runs <bin>/llmstack-
    # powercap (gpu-laptop-1-power.sh on gpu-laptop-1), and /etc/ld.so.conf.d puts
    # <prefix>/lib on the system loader path. The owner of a directory can
    # replace its entries, so an account that owned any of them could swap
    # what root executes -- and that account is the one the internet reaches.
    # So code execution in the gateway WAS root, on gpu-laptop-1 and apu-box-1, until
    # it was fixed by hand on 2026-10-12; this step is what used to chown the
    # whole prefix to the account on a fresh install.
    #
    # prefix is looked at at the top only (gateway/ and venv/ under it ARE the
    # account's, as are voice/ and research/); bin and lib are looked at all
    # the way down, because a FILE the account owns inside a root-owned
    # directory is one it can still rewrite in place.
    def _root_dirs(self, ctx) -> list[tuple[str, bool]]:
        p = ctx.plan["paths"]
        return [(p["prefix"], False), (p["bin"], True),
                (shapes.join(ctx.family, p["prefix"], "lib"), True)]

    def _handed_over(self, ctx, user: str) -> list[str]:
        """What under those directories the service account owns right now."""
        out: list[str] = []
        for path, deep in self._root_dirs(ctx):
            out += ctx.owned_by(user, path, recursive=deep)
        return out

    def _root_owns(self, ctx) -> None:
        """prefix, bin and lib are root's; created root-owned when missing.

        `install -d -o root -g root` creates a missing directory root-owned
        and sets the mode of an existing one, but not every coreutils
        re-owns an existing directory -- the explicit chown is what makes
        this heal a stack an older fleetctl (or bootstrap script) handed over.
        prefix is not chowned recursively: gateway/ and venv/ under it are
        the account's, and are given back by the caller.
        """
        p = ctx.plan["paths"]
        prefix, binp = p["prefix"], p["bin"]
        lib = shapes.join(ctx.family, prefix, "lib")
        ctx.sudo(["install", "-d", "-o", "root", "-g", "root", "-m", "0755",
                  prefix, binp])
        ctx.sudo(["chown", "root:root", prefix])
        ctx.sudo(["chown", "-R", "root:root", binp])
        # lib is not in the plan and fleetctl does not create it: the llama.cpp
        # bootstraps do. Only make it root's if it is there.
        if ctx.path(lib).is_dir():
            ctx.sudo(["chown", "-R", "root:root", lib])

    @staticmethod
    def _brief(paths: list[str], n: int = 4) -> str:
        shown = ", ".join(paths[:n])
        return shown + (f" (+{len(paths) - n} more)" if len(paths) > n else "")

    def check(self, ctx) -> Check:
        user = ctx.plan["service"]["user"]
        r = ctx.probe(["id", "-u", user])
        gpu = self._wants_gpu_groups(ctx)
        if r is not None and r.returncode == 0:
            problems: list[str] = []
            plan: list[str] = []
            if gpu:
                groups = ctx.probe(["id", "-nG", user])
                have = set((groups.stdout or "").split()) if groups else set()
                need = {"video", "render"} - have
                if need:
                    problems.append(f"not in {', '.join(sorted(need))}")
                    plan.append(f"usermod -aG {','.join(sorted(need))} {user}")
            handed = self._handed_over(ctx, user)
            if handed:
                # DRIFT, not a warning: this is the line between "the gateway
                # was compromised" and "the box was".
                problems.append(f"owns {self._brief(handed)}, which root "
                                f"executes from (gateway RCE would be root)")
                plan.append("install -d -o root -g root "
                            f"{ctx.plan['paths']['prefix']} {ctx.plan['paths']['bin']}")
                plan.append(f"chown -R root:root {ctx.plan['paths']['bin']} "
                            f"(and {shapes.join(ctx.family, ctx.plan['paths']['prefix'], 'lib')})")
            if problems:
                return Check(DRIFT, f"{user} exists, " + "; ".join(problems), plan)
            if not gpu:
                return Check(OK, f"{user} exists (no local engine to give it "
                                 f"the GPU for), prefix/bin/lib are root's")
            return Check(OK, f"{user} exists, in video and render, "
                             f"prefix/bin/lib are root's")
        plan = [f"useradd --system {user}"]
        if gpu:
            plan.append(f"usermod -aG video,render {user}")
        return Check(MISSING, f"no {user} account", plan)

    def apply(self, ctx) -> None:
        user = ctx.plan["service"]["user"]
        r = ctx.probe(["id", "-u", user])
        created = r is None or r.returncode != 0
        if created:
            ctx.sudo(["useradd", "--system", "--create-home",
                      "--home-dir", f"/var/lib/{user}",
                      "--shell", "/usr/sbin/nologin", user])
            ctx.did(f"created {user}")
        if self._wants_gpu_groups(ctx):
            ctx.sudo(["usermod", "-aG", "video,render", user], check=False)
            ctx.did(f"{user} in video, render")
        # An existing account is only ever walked for what check() flagged:
        # prefix, bin and lib handed to it by an older bootstrap or fleetctl.
        # A group that was merely missing is not a reason to walk every file
        # under the models directory -- often the biggest tree on the box, and
        # sometimes somebody else's disk.
        if not created:
            if self._handed_over(ctx, user):
                self._root_owns(ctx)
                ctx.did(f"prefix, bin and lib are root's again "
                        f"(they were {user}'s)")
            return
        # A NEW account taking over a stack that was laid down before it
        # existed. It takes what it WRITES -- state, models, the gateway
        # sources and the venv -- and nothing root runs from. prefix and bin
        # (and lib) stay root's: see _root_dirs for why the service account
        # owning them is a privilege escalation, not a tidiness question.
        self._root_owns(ctx)
        for d in (ctx.plan["paths"]["state"], ctx.plan["paths"]["models"],
                  ctx.plan["paths"]["gateway"], ctx.plan["paths"]["venv"]):
            ctx.sudo(["chown", "-R", f"{user}:{user}", d], check=False)
        ctx.did(f"{user} owns state, models, gateway and venv; "
                f"prefix and bin are root's")


class Grants(Step):
    id = "grants"
    title = "sudoers grant"
    system = True

    def wanted(self, ctx) -> bool:
        return ctx.family == "linux" and bool(ctx.plan["service"]["user"])

    def check(self, ctx) -> Check:
        script = ctx.repo / "hosts" / "linux" / "grants.sh"
        if not script.is_file():
            return Check(SKIPPED, "hosts/linux/grants.sh is not in this checkout")
        there, why = ctx.exists_state("/etc/sudoers.d/llmstack-fleet")
        if there:
            return Check(OK, "/etc/sudoers.d/llmstack-fleet present")
        if there is None:
            # /etc/sudoers.d is 0750 root:root by design. An unprivileged
            # check reading that as "no fleet grant" sends an operator to
            # re-run grants.sh against a box that already has it -- harmless
            # here, and the same lie that was not harmless about the swap
            # config two steps up.
            return Check(BLOCKED, f"/etc/sudoers.d is not readable from this "
                                  f"account ({why}); re-run with sudo")
        return Check(MISSING, "no fleet grant",
                     ["run hosts/linux/grants.sh (writes "
                      "/etc/sudoers.d/llmstack-fleet)"])

    def apply(self, ctx) -> None:
        script = ctx.repo / "hosts" / "linux" / "grants.sh"
        # The helper's SOURCE is the checkout's copy. Passing the installed
        # path made grants.sh `install` the file onto itself, which coreutils
        # refuses ("are the same file") -- and that failed every fresh apply
        # right after the step that had just installed the helper.
        helper = ctx.repo / "gateway" / "bin" / "llmstack-gpuconf"
        args = ["bash", str(script), ctx.plan["service"]["user"]]
        if helper.is_file():
            args.append(str(helper))
        ctx.sudo(args)
        ctx.did("fleet sudoers grant installed")


class Power(Step):
    id = "power"
    title = "never sleep"
    system = True

    def wanted(self, ctx) -> bool:
        return ctx.family in ("linux", "windows")

    # Windows: two settings, on EVERY power scheme. gpu-desktop-1 vanished from the
    # fleet for eleven hours because the never-sleep setting had been made on
    # a scheme that was not the one in force; a driver utility or a game
    # switching plans must not be able to reintroduce the cliff. UNATTENDSLEEP
    # is the one that bites after a wake nobody caused -- the box wakes for
    # the network, sees no user, and is asleep again two minutes later.
    SUB_SLEEP = "238c9fa8-0aad-41ed-83f4-97be242c8f20"
    STANDBYIDLE = "29f6c1db-86da-48c5-9fdb-f2b67b1f44da"
    UNATTENDSLEEP = "7bc4a2f9-d8fc-4469-b07b-33eb785aaca0"

    LOGIND = """\
# Written by fleetctl. This machine is a headless inference server that
# happens to be a laptop: closing the lid, or idling, must never suspend it.
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
IdleAction=ignore
"""
    LOGIND_PATH = "/etc/systemd/logind.conf.d/99-headless-server.conf"

    def check(self, ctx) -> Check:
        if ctx.family == "linux":
            if ctx.plan["platform"]["service"] != "systemd":
                return Check(SKIPPED, "no systemd-logind here")
            current, why = ctx.read_state(self.LOGIND_PATH)
            if current == self.LOGIND:
                return Check(OK, self.LOGIND_PATH)
            if current is None and why.startswith("unreadable"):
                return Check(BLOCKED, f"{self.LOGIND_PATH} ({why}); "
                                      f"re-run with sudo to compare it")
            return Check(MISSING if current is None else DRIFT,
                         "lid/idle suspend not disabled",
                         [f"write {self.LOGIND_PATH}",
                          "mask sleep.target suspend.target hibernate.target"])
        import re

        r = ctx.probe(["powercfg", "/l"])
        if r is None or r.returncode != 0:
            return Check(BLOCKED, "powercfg did not answer")
        schemes = re.findall(r"GUID:\s+(\S+)", r.stdout or "")
        # Read both settings back from every scheme. This used to list the
        # schemes and say MISSING regardless, so an apply that had just pinned
        # them re-checked as UNCHANGED and stopped the run (gpu-laptop-2,
        # 2026-10-19). A setting powercfg will not show -- UNATTENDSLEEP is
        # hidden until apply unhides it -- counts as not pinned.
        unpinned = 0
        for scheme in schemes:
            for setting in (self.STANDBYIDLE, self.UNATTENDSLEEP):
                q = ctx.probe(["powercfg", "/q", scheme, self.SUB_SLEEP, setting])
                values = re.findall(r"Current (?:AC|DC) Power Setting Index:\s*0x([0-9a-fA-F]+)",
                                    (q.stdout if q and q.returncode == 0 else "") or "")
                if len(values) != 2 or any(int(v, 16) != 0 for v in values):
                    unpinned += 1
                    break
        if not unpinned:
            return Check(OK, f"standby + unattended-sleep never on all {len(schemes)} scheme(s)")
        return Check(MISSING, f"{unpinned} power scheme(s) to pin",
                     [f"set standby + unattended-sleep to never on all "
                      f"{len(schemes)} schemes"])

    def apply(self, ctx) -> None:
        if ctx.family == "linux":
            ctx.ensure_dir("/etc/systemd/logind.conf.d")
            if ctx.write(self.LOGIND_PATH, self.LOGIND):
                ctx.did(f"wrote {self.LOGIND_PATH}")
            ctx.sudo(["systemctl", "mask", "sleep.target", "suspend.target",
                      "hibernate.target", "hybrid-sleep.target"], check=False)
            ctx.sudo(["systemctl", "restart", "systemd-logind"], check=False)
            return
        import re

        r = ctx.probe(["powercfg", "/l"])
        # Unhide UNATTENDSLEEP first: powercfg /q does not list it by default,
        # so a setting made here could not otherwise be read back and audited.
        ctx.run(["powercfg", "/attributes", self.SUB_SLEEP, self.UNATTENDSLEEP,
                 "-ATTRIB_HIDE"], system=True, check=False)
        n = 0
        for line in (r.stdout or "").splitlines() if r else []:
            m = re.search(r"GUID:\s+(\S+)", line)
            if not m:
                continue
            for setting in (self.STANDBYIDLE, self.UNATTENDSLEEP):
                for verb in ("/setacvalueindex", "/setdcvalueindex"):
                    ctx.run(["powercfg", verb, m.group(1), self.SUB_SLEEP,
                             setting, "0"], system=True, check=False)
            n += 1
        act = ctx.probe(["powercfg", "/getactivescheme"])
        m = re.search(r"GUID:\s+(\S+)", (act.stdout if act else "") or "")
        if m:
            # Re-apply the running scheme so this takes effect now rather than
            # at the next scheme switch.
            ctx.run(["powercfg", "/setactive", m.group(1)], system=True, check=False)
        ctx.did(f"never-sleep set on {n} power scheme(s)")


def _rule_admits_only(show: str, allow: list) -> bool:
    """Does `netsh advfirewall firewall show rule` say every rule of that name
    admits exactly the planned set? "Any", anything wider, a second rule left
    unscoped, or output this cannot read (an English label is assumed) is not
    -- apply then deletes them all and writes one."""
    want = {ipaddress.ip_network(str(a), strict=False) for a in allow}
    lines = [ln for ln in show.splitlines() if ln.strip().lower().startswith("remoteip:")]
    if not lines:
        return False
    for line in lines:
        try:
            have = {ipaddress.ip_network(x.strip(), strict=False)
                    for x in line.split(":", 1)[1].split(",") if x.strip()}
        except ValueError:
            return False                      # "Any", "LocalSubnet" and the like
        if have != want:
            return False
    return True


class Firewall(Step):
    id = "firewall"
    title = "firewall"
    system = True

    def wanted(self, ctx) -> bool:
        return ctx.family in ("linux", "windows")

    def check(self, ctx) -> Check:
        port = ctx.plan["network"]["port"]
        if ctx.family == "windows":
            r = ctx.probe(["netsh", "advfirewall", "firewall", "show", "rule",
                           "name=llm-gateway"])
            have = r is not None and r.returncode == 0
            allow = ctx.plan["network"]["allow_from"]
            if have and allow and not _rule_admits_only(r.stdout or "", allow):
                return Check(MISSING, "llm-gateway admits more than "
                             + ", ".join(map(str, allow)),
                             [f"allow TCP {port} inbound from "
                              f"{', '.join(map(str, allow))} only"])
            return Check(OK if have else MISSING,
                         "llm-gateway rule present" if have else "no rule",
                         [] if have else
                         [f"allow TCP {port} inbound (llm-gateway)",
                          "block the engine port off-box"])
        deaf = []
        for tool, probe in (("firewall-cmd", ["firewall-cmd", "--state"]),
                            ("ufw", ["ufw", "status"])):
            r = ctx.probe(probe)
            if r is None:
                continue                      # the tool is not installed
            if r.returncode == 0:
                # The old code returned MISSING here unconditionally, so on
                # Linux this step could never converge: apply added the rule
                # and the re-check still said "is running" -> UNCHANGED,
                # which aborted every apply on a box with an active firewall
                # (mini-pc-1, 2026-09-09). A rule for the port, or a blanket
                # allow on the tailnet interface (the fleet posture), is OK.
                blob = (r.stdout or "") + (r.stderr or "")
                if tool == "ufw" and "inactive" in blob.lower():
                    return Check(SKIPPED, "ufw installed but inactive")
                if tool == "firewall-cmd":
                    # `firewall-cmd --state` says only "running": the rule
                    # lives in `--list-ports`. Reading the state output for
                    # the port meant this step could never converge on a
                    # firewalld box -- apu-tablet-1's first Fedora apply
                    # (2026-09-16) added the rule and then aborted with
                    # UNCHANGED, one step before the services were written.
                    ports = ctx.probe(["firewall-cmd", "--list-ports"])
                    blob += " " + ((ports.stdout or "") if ports is not None else "")
                if f"{port}/tcp" in blob or "on tailscale0" in blob:
                    return Check(OK, f"{tool}: {port}/tcp reachable")
                return Check(MISSING, f"{tool} is running without a rule for {port}/tcp",
                             [f"allow {port}/tcp on the tailnet interface"])
            # Installed, and would not answer. `ufw status` unprivileged
            # exits with "You need to be root to run this script", which the
            # old code read as absence -- so an unprivileged run reported
            # "no firewalld or ufw here" about hub and server-1, both of
            # which are running ufw. A firewall the tool cannot see is not a
            # firewall that is not there.
            blob = ((r.stdout or "") + (r.stderr or "")).lower()
            if any(w in blob for w in ("root", "permission", "denied",
                                       "operation not permitted")):
                deaf.append(tool)
        if deaf:
            return Check(BLOCKED,
                         f"{', '.join(deaf)} would not answer this account; "
                         f"re-run with sudo to see whether a rule is needed")
        # Not a failure. Plenty of these boxes are behind a router and a
        # tailnet ACL and have no local firewall at all; inventing one is not
        # this tool's call.
        return Check(SKIPPED, "no firewalld or ufw here")

    def apply(self, ctx) -> None:
        port = str(ctx.plan["network"]["port"])
        if ctx.family == "windows":
            ctx.run(["netsh", "advfirewall", "firewall", "delete", "rule",
                     "name=llm-gateway"], system=True, check=False)
            rule = ["netsh", "advfirewall", "firewall", "add", "rule",
                    "name=llm-gateway", "dir=in", "action=allow",
                    "protocol=TCP", f"localport={port}"]
            allow = ctx.plan["network"]["allow_from"]
            if allow:
                rule.append("remoteip=" + ",".join(map(str, allow)))
            ctx.run(rule, system=True)
            up = ctx.plan["engine"]["swap_listen"]
            if up and ":" in up:
                # The engine is loopback-only; this is the belt to that
                # brace. The gateway in front of it is the only authenticated,
                # metered door on the box.
                eport = up.rsplit(":", 1)[1]
                ctx.run(["netsh", "advfirewall", "firewall", "delete", "rule",
                         "name=llama-swap-lan-block"], system=True, check=False)
                ctx.run(["netsh", "advfirewall", "firewall", "add", "rule",
                         "name=llama-swap-lan-block", "dir=in", "action=block",
                         "protocol=TCP", f"localport={eport}"], system=True)
            ctx.did(f"{port} allowed in, engine port blocked off-box")
            return
        if ctx.probe(["firewall-cmd", "--state"]) is not None:
            ctx.sudo(["firewall-cmd", "--permanent", f"--add-port={port}/tcp"],
                     check=False)
            ctx.sudo(["firewall-cmd", "--reload"], check=False)
            ctx.did(f"firewalld: {port}/tcp allowed")
        elif ctx.probe(["ufw", "status"]) is not None:
            ctx.sudo(["ufw", "allow", f"{port}/tcp"], check=False)
            ctx.did(f"ufw: {port}/tcp allowed")


class GpuCap(Step):
    id = "gpu-cap"
    title = "GPU memory ceiling"
    system = True

    def wanted(self, ctx) -> bool:
        return bool(ctx.plan["sizing"]["metal_gib"] or ctx.plan["sizing"]["gtt_gib"])

    def _helper(self, ctx) -> str:
        return shapes.join(ctx.family, ctx.plan["paths"]["bin"], "llmstack-gpuconf")

    def check(self, ctx) -> Check:
        want = ctx.plan["sizing"]["metal_gib"] or ctx.plan["sizing"]["gtt_gib"]
        helper = ctx.path(self._helper(ctx))
        if not helper.exists():
            return Check(MISSING, "llmstack-gpuconf not installed yet",
                         [f"install the helper, then stage {want} GiB"])
        r = ctx.probe([str(helper), "status"])
        if r is None or r.returncode != 0:
            return Check(DRIFT, "helper will not report status",
                         [f"stage {want} GiB"])
        import json

        try:
            receipt = json.loads(r.stdout or "{}")
        except ValueError:
            return Check(DRIFT, "helper emitted unparsable status",
                         [f"stage {want} GiB"])
        if receipt.get("gib") == want and receipt.get("active"):
            return Check(OK, f"{want} GiB, active")
        if receipt.get("gib") == want:
            # Staged is the honest answer on Linux, where the setting is a
            # kernel cmdline token and only a reboot makes it real.
            return Check(OK, f"{want} GiB staged, pending reboot")
        return Check(DRIFT, f"helper reports {receipt.get('gib')} GiB, plan says {want}",
                     [f"stage {want} GiB"])

    def apply(self, ctx) -> None:
        want = ctx.plan["sizing"]["metal_gib"] or ctx.plan["sizing"]["gtt_gib"]
        ctx.sudo([self._helper(ctx), str(want)])
        ctx.did(f"GPU ceiling staged at {want} GiB")


class PowerCap(Step):
    id = "power-cap"
    title = "CPU/GPU power cap"
    system = True

    # Unlike GpuCap's GTT window, none of these three knobs is a kernel
    # cmdline token -- power_dpm_force_performance_level, pp_od_clk_voltage
    # and cpufreq/boost are live sysfs writes that take effect immediately
    # and are LOST on every reboot. A oneshot unit reapplies them at boot,
    # ordered before llama-swap so the cap is in force before the GPU takes
    # traffic -- but only ordered, never required: a knob this box's
    # hardware cannot support must be loud in `systemctl status`, not a
    # reason llama-swap fails to start.
    CONF_NAME = "powercap.conf"
    UNIT_NAME = "llmstack-powercap.service"
    UNIT_PATH = "/etc/systemd/system/" + UNIT_NAME
    KEYS = ("gpu_sclk_max_mhz", "cpu_boost", "cpu_max_mhz")

    UNIT = """\
[Unit]
Description=Re-apply {host}'s CPU/GPU power cap at boot (see host.yml power:)
Before=llama-swap.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={helper} apply

[Install]
WantedBy=multi-user.target
"""

    def wanted(self, ctx) -> bool:
        if ctx.family != "linux":
            return False
        p = ctx.plan["power"]
        if any(p[k] is not None for k in self.KEYS):
            return True
        # All three keys were removed from host.yml -- the documented revert
        # path. Stay wanted long enough for check()/apply() to notice a unit
        # from a PREVIOUS apply is still installed and enabled, and tear it
        # down. Returning False here (the naive reading of "nothing to do")
        # would make the runner skip check() and apply() outright, leaving
        # the unit enabled and re-applying the stale cap at every future
        # boot forever, with no signal anywhere -- not even `fleetctl
        # check` -- that the box never actually reverted.
        return ctx.read(self.UNIT_PATH) is not None

    def _cfg(self, ctx) -> dict:
        p = ctx.plan["power"]
        return {k: p[k] for k in self.KEYS}

    def _helper(self, ctx) -> str:
        return shapes.join(ctx.family, ctx.plan["paths"]["bin"], "llmstack-powercap")

    def _conf_path(self, ctx) -> str:
        return shapes.join(ctx.family, ctx.plan["paths"]["etc"], self.CONF_NAME)

    @staticmethod
    def _conf_value(v) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        return str(v)

    def _conf_text(self, ctx) -> str:
        cfg = self._cfg(ctx)
        return (
            "# Written by fleetctl -- see hosts/{host}/host.yml's power: section.\n"
            "# Read by llmstack-powercap at boot, before llama-swap starts serving.\n"
            "# Blank means: this knob is not managed here.\n"
            "GPU_SCLK_MAX_MHZ={gpu}\n"
            "CPU_BOOST={boost}\n"
            "CPU_MAX_MHZ={cpu}\n"
        ).format(host=ctx.plan["host"]["name"],
                 gpu=self._conf_value(cfg["gpu_sclk_max_mhz"]),
                 boost=self._conf_value(cfg["cpu_boost"]),
                 cpu=self._conf_value(cfg["cpu_max_mhz"]))

    def _unit_text(self, ctx) -> str:
        return self.UNIT.format(host=ctx.plan["host"]["name"], helper=self._helper(ctx))

    def _describe(self, cfg: dict) -> str:
        bits = []
        if cfg["gpu_sclk_max_mhz"] is not None:
            bits.append(f"gpu sclk <= {cfg['gpu_sclk_max_mhz']}MHz")
        if cfg["cpu_boost"] is not None:
            bits.append("cpu boost " + ("on" if cfg["cpu_boost"] else "off"))
        if cfg["cpu_max_mhz"] is not None:
            bits.append(f"cpu <= {cfg['cpu_max_mhz']}MHz")
        return ", ".join(bits)

    def check(self, ctx) -> Check:
        cfg = self._cfg(ctx)
        if not any(cfg[k] is not None for k in self.KEYS):
            # The revert path: host.yml no longer wants a cap. wanted() only
            # kept this step in play because the unit is still there from a
            # previous apply -- if it is gone too there is nothing left to
            # report.
            if ctx.read(self.UNIT_PATH) is None:
                return Check(SKIPPED, "no power cap configured")
            return Check(DRIFT, "power cap removed from host.yml but "
                                f"{self.UNIT_NAME} is still enabled",
                         [f"systemctl disable --now {self.UNIT_NAME}",
                          f"remove {self.UNIT_PATH}",
                          f"remove {self._conf_path(ctx)}"])

        helper = ctx.path(self._helper(ctx))
        todo = []
        if not helper.exists():
            todo.append(f"install {self._helper(ctx)}")
        if ctx.read(self._conf_path(ctx)) != self._conf_text(ctx):
            todo.append(f"write {self._conf_path(ctx)}")
        if ctx.read(self.UNIT_PATH) != self._unit_text(ctx):
            todo.append(f"write {self.UNIT_PATH}")
        r = ctx.probe(["systemctl", "is-enabled", self.UNIT_NAME])
        if r is None or r.returncode != 0:
            todo.append(f"systemctl enable --now {self.UNIT_NAME}")

        if not helper.exists():
            return Check(MISSING, "llmstack-powercap not installed yet", todo)

        # Read back the ACTIVE state directly from the helper -- the same
        # "wanted vs active" shape GpuCap uses, except a sysfs write here is
        # synchronous: there is no "staged, pending reboot", only active now
        # or the write failed.
        r2 = ctx.probe([str(helper), "status"])
        if r2 is None or r2.returncode != 0:
            return Check(DRIFT, "helper will not report status", todo or ["re-apply"])
        try:
            active = json.loads((r2.stdout or "{}").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return Check(DRIFT, "helper emitted unparsable status", todo or ["re-apply"])

        mismatches = []
        if (cfg["gpu_sclk_max_mhz"] is not None
                and active.get("gpu_sclk_active_mhz") != cfg["gpu_sclk_max_mhz"]):
            mismatches.append(
                f"gpu sclk cap: want {cfg['gpu_sclk_max_mhz']}MHz, "
                f"active {active.get('gpu_sclk_active_mhz')}")
        want_boost = None if cfg["cpu_boost"] is None else int(cfg["cpu_boost"])
        if want_boost is not None and active.get("cpu_boost_active") != want_boost:
            mismatches.append(f"cpu boost: want {want_boost}, "
                              f"active {active.get('cpu_boost_active')}")
        if (cfg["cpu_max_mhz"] is not None
                and active.get("cpu_max_active_mhz") != cfg["cpu_max_mhz"]):
            mismatches.append(
                f"cpu max: want {cfg['cpu_max_mhz']}MHz, "
                f"active {active.get('cpu_max_active_mhz')}")

        if not todo and not mismatches:
            return Check(OK, self._describe(cfg) + " -- active")
        return Check(DRIFT, "; ".join(mismatches) or "unit/config out of date",
                     todo + mismatches)

    def apply(self, ctx) -> None:
        cfg = self._cfg(ctx)
        if not any(cfg[k] is not None for k in self.KEYS):
            # Teardown: host.yml no longer wants a cap. Disable the unit so
            # it stops re-applying a stale one at every future boot, then
            # remove its files so a stale powercap.conf cannot be picked up
            # by some other means later. The sysfs values THEMSELVES only
            # return to their hardware defaults on the next reboot -- same
            # staged/pending-reboot honesty GpuCap's GTT window already
            # uses, and there is no live "undo" for a manual
            # pp_od_clk_voltage commit to invent here.
            ctx.sudo(["systemctl", "disable", "--now", self.UNIT_NAME], check=False)
            for p in (self.UNIT_PATH, self._conf_path(ctx)):
                ctx.sudo(["rm", "-f", p], check=False)
            ctx.sudo(["systemctl", "daemon-reload"], check=False)
            ctx.note(f"power cap removed from host.yml -- reboot "
                     f"{ctx.plan['host']['name']} to restore default "
                     f"clocks/boost")
            ctx.did("power cap disabled and removed")
            return
        changed = False
        if ctx.write(self._conf_path(ctx), self._conf_text(ctx)):
            changed = True
            ctx.did(f"wrote {self._conf_path(ctx)}")
        if ctx.write(self.UNIT_PATH, self._unit_text(ctx)):
            changed = True
            ctx.did(f"wrote {self.UNIT_PATH}")
        if changed:
            ctx.sudo(["systemctl", "daemon-reload"])
        ctx.sudo(["systemctl", "enable", self.UNIT_NAME], check=False)
        # restart, not start: the unit is RemainAfterExit=yes, so a box that
        # already has it running would otherwise never re-run ExecStart and
        # pick up a config that just changed. check=False: a knob this box's
        # hardware cannot honour (out-of-range MHz, no boost file, ...) is
        # exactly what makes `llmstack-powercap apply` exit nonzero -- see
        # its own module docstring -- and that must surface as the DRIFT
        # check() already knows how to read from the helper's own `status`,
        # not as a crashed `systemctl restart` that aborts the whole apply
        # run before `services`/`health` ever get a turn (the sibling Power
        # step's analogous restart of systemd-logind, just above, makes the
        # same choice for the same reason).
        ctx.sudo(["systemctl", "restart", self.UNIT_NAME], check=False)
        ctx.did(f"power cap applied ({self._describe(cfg)})")
