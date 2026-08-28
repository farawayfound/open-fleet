"""Steps that change the machine rather than the install.

Packages, the service account, the sudoers grant, never-sleep, the firewall
and the GPU ceiling. All of them are `system` steps: they touch things
outside the plan's own paths, so they are refused under `--root` and reported
as blocked without root. That is deliberate -- see steps/__init__.py on why
`blocked` and `skipped` are different words.
"""
from __future__ import annotations

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

    def check(self, ctx) -> Check:
        user = ctx.plan["service"]["user"]
        r = ctx.probe(["id", "-u", user])
        gpu = self._wants_gpu_groups(ctx)
        if r is not None and r.returncode == 0:
            if not gpu:
                return Check(OK, f"{user} exists (no local engine to give it "
                                 f"the GPU for)")
            groups = ctx.probe(["id", "-nG", user])
            have = set((groups.stdout or "").split()) if groups else set()
            need = {"video", "render"} - have
            if need:
                return Check(DRIFT, f"{user} exists, not in {', '.join(sorted(need))}",
                             [f"usermod -aG {','.join(sorted(need))} {user}"])
            return Check(OK, f"{user} exists, in video and render")
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
        # The recursive chown is for a NEW account taking over a stack that
        # was laid down before it existed. An existing account that only
        # needed a group is not a reason to walk every file under the models
        # directory -- which is often the biggest tree on the box, and
        # sometimes somebody else's disk.
        if not created:
            return
        for d in (ctx.plan["paths"]["prefix"], ctx.plan["paths"]["state"],
                  ctx.plan["paths"]["models"], ctx.plan["paths"]["gateway"],
                  ctx.plan["paths"]["bin"], ctx.plan["paths"]["venv"]):
            ctx.sudo(["chown", "-R", f"{user}:{user}", d], check=False)
        ctx.did(f"{user} owns the stack")


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
        helper = shapes.join(ctx.family, ctx.plan["paths"]["bin"], "llmstack-gpuconf")
        ctx.sudo(["bash", str(script), ctx.plan["service"]["user"], helper])
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
        r = ctx.probe(["powercfg", "/l"])
        if r is None or r.returncode != 0:
            return Check(BLOCKED, "powercfg did not answer")
        schemes = [ln for ln in (r.stdout or "").splitlines() if "GUID:" in ln]
        return Check(MISSING, f"{len(schemes)} power scheme(s) to pin",
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
                return Check(MISSING, f"{tool} is running",
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
            ctx.run(["netsh", "advfirewall", "firewall", "add", "rule",
                     "name=llm-gateway", "dir=in", "action=allow",
                     "protocol=TCP", f"localport={port}"], system=True)
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
