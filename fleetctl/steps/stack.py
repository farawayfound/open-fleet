"""The gateway itself: directories, interpreter, sources, venv, env file.

These are the steps that are the same on all fifteen boxes once the shape has
been factored out, and they are the ones that genuinely run under `--root`.
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import shapes
from ..facts import MIN_PYTHON, _python_version
from . import BLOCKED, DRIFT, MISSING, OK, Check, Step

GATEWAY_FILES = ("app.py", "hw.py", "requirements.txt",
                 "public_seed.json", "public_domains_seed.json")


class Preflight(Step):
    id = "preflight"
    title = "sources and privileges"

    def check(self, ctx) -> Check:
        src = ctx.repo / "gateway"
        problems = []
        for name in ("app.py", "hw.py", "requirements.txt"):
            if not (src / name).is_file():
                problems.append(f"missing {src / name}")
        if problems:
            return Check(BLOCKED, "; ".join(problems))
        needs_root = shapes.SHAPES[ctx.family]["needs_root"]
        if needs_root and not ctx.root_ok and not ctx.dry_run:
            # Not fatal. The file steps still work as this user on a
            # user-owned prefix, and saying which steps will be blocked is
            # more useful than refusing to start.
            return Check(OK, f"sources at {src}; no root -- system steps will "
                             f"report blocked")
        return Check(OK, f"sources at {src}")

    def apply(self, ctx) -> None:
        return None


class Directories(Step):
    id = "directories"
    title = "directory layout"

    def _dirs(self, ctx) -> list[str]:
        p = ctx.plan["paths"]
        want = [p["prefix"], p["gateway"], p["static"], p["bin"], p["state"],
                p["etc"], p["models"]]
        # Deduplicated because ETC == STATE on the user-level shapes, and a
        # report that says "would create <state>" twice reads as a bug.
        out: list[str] = []
        for d in want:
            if d and d not in out:
                out.append(d)
        return out

    def check(self, ctx) -> Check:
        missing = [d for d in self._dirs(ctx) if not ctx.path(d).is_dir()]
        if not missing:
            return Check(OK, f"{len(self._dirs(ctx))} directories present")
        return Check(MISSING, f"{len(missing)} of {len(self._dirs(ctx))} missing",
                     [f"create {d}" for d in missing])

    def apply(self, ctx) -> None:
        for d in self._dirs(ctx):
            if ctx.ensure_dir(d):
                ctx.did(f"created {d}")


class PythonRuntime(Step):
    id = "python-runtime"
    title = "a usable interpreter"

    def check(self, ctx) -> Check:
        want = ctx.plan["paths"]["python"]
        if want:
            ver = _python_version(want)
            if ver and ver >= MIN_PYTHON:
                return Check(OK, f"{want} ({'.'.join(map(str, ver))})")
            # A pinned interpreter that is not there is the plan being wrong
            # about this box, and it is worth saying which one was asked for.
            detail = f"{want} is not a usable interpreter"
        else:
            detail = "no interpreter chosen"

        found = (ctx.facts.get("python") or {})
        if found.get("exe"):
            return Check(DRIFT, detail,
                         [f"use {found['exe']} ({found['version']}) instead"])

        tried = ", ".join(c["exe"] for c in found.get("candidates", [])[:6])
        # Windows is where this actually happens: `python` on PATH is the
        # Microsoft Store stub, which prints an advert and exits 49, and the
        # only real interpreter is a per-user install a SYSTEM task cannot
        # reach. Say so, rather than "python not found".
        hint = ""
        if ctx.family == "windows":
            hint = ("\n  install Python for ALL USERS (a per-user install under "
                    "C:\\Users\\... is unreachable from a SYSTEM scheduled task):"
                    "\n  https://www.python.org/downloads/windows/")
        elif ctx.family == "darwin":
            hint = ("\n  macOS ships 3.9; the gateway needs 3.10+. "
                    "brew install python@3.12")
        return Check(BLOCKED,
                     f"no python >= {'.'.join(map(str, MIN_PYTHON))} here."
                     f" tried: {tried}{hint}")

    def apply(self, ctx) -> None:
        found = (ctx.facts.get("python") or {})
        if found.get("exe"):
            ctx.plan["paths"]["python"] = found["exe"]
            ctx.did(f"interpreter: {found['exe']} ({found['version']})")


class GatewayFiles(Step):
    id = "gateway-files"
    title = "gateway sources"

    def _pairs(self, ctx) -> list[tuple[Path, str]]:
        src = ctx.repo / "gateway"
        gw = ctx.plan["paths"]["gateway"]
        out: list[tuple[Path, str]] = []
        for name in GATEWAY_FILES:
            if (src / name).is_file():
                out.append((src / name, shapes.join(ctx.family, gw, name)))
        out.append((src / "static" / "index.html",
                    shapes.join(ctx.family, ctx.plan["paths"]["static"], "index.html")))
        return out

    def check(self, ctx) -> Check:
        stale = []
        for source, dest in self._pairs(ctx):
            target = ctx.path(dest)
            try:
                same = (target.exists()
                        and target.stat().st_size == source.stat().st_size
                        and target.read_bytes() == source.read_bytes())
            except OSError:
                same = False
            if not same:
                stale.append((source, dest, target.exists()))
        if not stale:
            return Check(OK, f"{len(self._pairs(ctx))} files current")
        return Check(DRIFT if any(existed for _, _, existed in stale) else MISSING,
                     f"{len(stale)} of {len(self._pairs(ctx))} differ",
                     [f"install {s.name} -> {d}" for s, d, _ in stale])

    def apply(self, ctx) -> None:
        n = 0
        for source, dest in self._pairs(ctx):
            if ctx.copy(source, dest):
                n += 1
        # gpuconf is a root helper, not a gateway source, but it travels with
        # the gateway on every shape -- it is what lets the dashboard read a
        # box's GPU ceiling back honestly instead of "helper not installed".
        helper = ctx.repo / "gateway" / "bin" / "llmstack-gpuconf"
        if helper.is_file() and ctx.family != "windows":
            if ctx.copy(helper, shapes.join(ctx.family, ctx.plan["paths"]["bin"],
                                            "llmstack-gpuconf"), mode=0o755):
                n += 1
        ctx.did(f"installed {n} file(s)")


class Venv(Step):
    id = "venv"
    title = "virtualenv + requirements"

    def _python(self, ctx) -> str:
        venv = ctx.plan["paths"]["venv"]
        sub = "Scripts/python.exe" if ctx.family == "windows" else "bin/python"
        return shapes.join(ctx.family, venv, *sub.split("/"))

    def check(self, ctx) -> Check:
        vpy = ctx.path(self._python(ctx))
        if not vpy.exists():
            return Check(MISSING, f"no venv at {ctx.plan['paths']['venv']}",
                         [f"create venv at {ctx.plan['paths']['venv']}",
                          "pip install -r requirements.txt"])
        # Ask the venv what it has rather than trusting that a previous run
        # finished: an interrupted pip leaves a venv that exists and cannot
        # import fastapi, which is a box that starts and immediately dies.
        r = ctx.probe([str(vpy), "-c", "import fastapi, uvicorn, httpx, jwt, psutil, yaml"])
        if r is None or r.returncode != 0:
            missing = (r.stderr.strip().splitlines()[-1] if r and r.stderr
                       else "could not run the venv interpreter")
            return Check(DRIFT, f"venv incomplete: {missing}",
                         ["pip install -r requirements.txt"])
        return Check(OK, f"venv complete at {ctx.plan['paths']['venv']}")

    def apply(self, ctx) -> None:
        base = ctx.plan["paths"]["python"]
        if not base:
            raise RuntimeError("no interpreter -- python-runtime must run first")
        venv = ctx.path(ctx.plan["paths"]["venv"])
        vpy = ctx.path(self._python(ctx))
        if not vpy.exists():
            ctx.run([base, "-m", "venv", str(venv)], timeout=600)
            ctx.did(f"created venv at {ctx.plan['paths']['venv']}")
        req = ctx.path(shapes.join(ctx.family, ctx.plan["paths"]["gateway"],
                                   "requirements.txt"))
        if not req.exists():
            req = ctx.repo / "gateway" / "requirements.txt"
        ctx.run([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel"],
                timeout=900)
        ctx.run([str(vpy), "-m", "pip", "install", "-q", "-r", str(req)], timeout=1800)
        ctx.did("requirements installed")


class EnvFile(Step):
    id = "envfile"
    title = "gateway.env"

    # The values that turn a generic install into THIS box. One table, three
    # renderings (see shapes.env_lines) -- the fifteen installers each carried
    # their own copy of this, which is why nine of them repeated
    # CF_ACCESS_TEAM_DOMAIN and LLMSTACK_ADMIN_EMAILS verbatim.
    def _values(self, ctx) -> dict:
        p, n, e, s, a = (ctx.plan["paths"], ctx.plan["network"], ctx.plan["engine"],
                         ctx.plan["sizing"], ctx.plan["access"])
        vals = {
            "LLMSTACK_HOME": p["prefix"],
            "LLMSTACK_STATE": p["state"],
            "LLMSTACK_ETC": p["etc"],
            "LLMSTACK_UPSTREAM": n["upstream"],
            "LLMSTACK_MODELS_DIR": p["models"],
            "LLMSTACK_STATIC": p["static"],
            "LLMSTACK_BIND": n["bind"],
            "LLMSTACK_PORT": n["port"],
            "LLMSTACK_HOST_NAME": ctx.plan["host"]["name"],
            "LLMSTACK_PUBLIC_API_URL": n["public_api_url"],
            "CF_ACCESS_TEAM_DOMAIN": a["cf_team_domain"],
            "CF_ACCESS_AUD": a["cf_aud"],
            "LLMSTACK_ADMIN_EMAILS": a["admin_emails"],
        }
        if e["kind"] == "llama.cpp":
            # From the plan, not recomputed from `bin`: on a box that built
            # llama.cpp itself these point outside the fleet prefix, and the
            # planner is where that is worked out (see _derive).
            vals["LLMSTACK_LLAMA_SERVER"] = p["llama_server"]
            vals["LLMSTACK_LLAMA_BENCH"] = p["llama_bench"]
            vals["LLMSTACK_SWAP_CONFIG"] = p["swap_config"]
        if e["models_from_upstream"]:
            vals["LLMSTACK_MODELS_FROM_UPSTREAM"] = 1
        if s["context_budget_gib"]:
            vals["LLMSTACK_CONTEXT_BUDGET_GIB"] = s["context_budget_gib"]
        if s["vram_headroom"]:
            vals["LLMSTACK_VRAM_HEADROOM"] = s["vram_headroom"]
        if s["max_auto_ctx"]:
            vals["LLMSTACK_MAX_AUTO_CTX"] = s["max_auto_ctx"]
        if ctx.plan["availability"]["file"]:
            vals["LLMSTACK_AVAILABILITY_FILE"] = ctx.plan["availability"]["file"]
        return vals

    NOTES = {
        "LLMSTACK_BIND": (
            "0.0.0.0, not loopback: the hub calls this box's admin API as a\n"
            "fleet peer. Bound to loopback it drops out of the routing table\n"
            "and its models become unreachable fleet-wide. The engine behind\n"
            "it stays on 127.0.0.1 -- this gateway is the only authenticated,\n"
            "metered door on the box."),
        "LLMSTACK_PUBLIC_API_URL": (
            "What the dashboard hands to API clients of THIS box, so it must\n"
            "be a hostname with no Cloudflare Access policy on it: an\n"
            "Access-gated hostname answers a bearer request with a 302 to an\n"
            "SSO page, which a client sees as a hang rather than a 401."),
        "LLMSTACK_ADMIN_TOKEN": (
            "Break-glass admin credential. Works over the tailnet even if\n"
            "Cloudflare Access is misconfigured or Cloudflare is unreachable.\n"
            "Generated once and never rewritten -- see the note in envfile.py\n"
            "about why this file is created and not updated."),
        "LLMSTACK_AVAILABILITY_FILE": (
            "An outside watchdog owns the answer to 'may the fleet use this\n"
            "machine right now'; the gateway stops advertising its models\n"
            "whenever the answer is no."),
    }

    def foreign(self, ctx, current: str | None) -> dict[str, str]:
        """Keys in the existing file that this plan does not own.

        The admin token had a special case because regenerating it would
        break the hub's credential for the peer. It turned out not to be the
        only credential in here. A privileged dry run on hub proposed:

            remove HF_TOKEN
            remove LLMSTACK_PUBLIC_INTAKE_TOKEN
            remove LLMSTACK_SMTP_FROM / HOST / PASSWORD / PORT / TLS / USER

        -- eight values, six of them the mail credentials the daily brief
        sends with, none of them anything the plan can express. `apply`
        rewrites this file from the plan, so all eight would have gone.

        The rule is ownership, not prefix: LLMSTACK_SMTP_PASSWORD looks
        exactly like a key fleetctl writes. fleetctl owns the keys it
        renders and carries every other one across untouched, which also
        means a value somebody set by hand survives an apply instead of
        needing to be re-set after one.
        """
        if not current:
            return {}
        mine = set(self._values(ctx)) | {"LLMSTACK_ADMIN_TOKEN"}
        out: dict[str, str] = {}
        for line in current.splitlines():
            m = re.match(r"^(?:set |export )?([A-Z][A-Z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) not in mine:
                out[m.group(1)] = m.group(2)
        return out

    def _render(self, ctx, admin_token: str,
                carried: dict[str, str] | None = None) -> str:
        vals = self._values(ctx)
        vals["LLMSTACK_ADMIN_TOKEN"] = admin_token
        head = [
            f"llmstack gateway -- runtime configuration for {ctx.plan['host']['name']}",
            "",
            "GENERATED BY fleetctl. Regenerating is safe: the admin token and",
            "any value fleetctl does not own are read back out of the existing",
            "file and kept.",
        ]
        cstyle = shapes.ENV_COMMENT[ctx.family]
        lines = [cstyle.format(t=h) if h else cstyle.format(t="").rstrip() for h in head]
        lines += [""]
        lines += shapes.env_lines(ctx.family, vals, self.NOTES)
        if carried:
            lines += [""]
            for note in ("Carried across, not written by fleetctl. Whatever put "
                         "these here",
                         "owns them -- delete a line to be rid of it. On hub "
                         "this is the",
                         "mail credentials the daily brief sends with."):
                lines.append(cstyle.format(t=note))
            lines += shapes.env_lines(ctx.family, carried)
        return "\n".join(lines) + "\n"

    def _token(self, ctx) -> str:
        """Keep the token that is already there.

        Every one of the old installers guarded its env file with
        `if [[ ! -f ]]` for this reason, and paid for it: a value that changed
        upstream could never be applied without deleting the file first. The
        token is the only part that must not be regenerated -- the hub holds
        it as this peer's credential -- so it is the only part carried across,
        and everything else in the file is rewritten from the plan.
        """
        import secrets

        current, why = ctx.read_state(ctx.plan["paths"]["envfile"])
        if current is None and why.startswith("unreadable"):
            # Never invent one over a file we cannot see. The token in there
            # is what the hub holds as this peer's credential, and minting a
            # replacement because a permission bit hid the old one would take
            # the box off the fleet in a way nothing reports.
            raise RuntimeError(
                f"{ctx.plan['paths']['envfile']} exists and this account cannot "
                f"read it ({why}). Re-run with sudo -- writing a new file here "
                f"would mint a new admin token and break the hub's entry for "
                f"this peer.")
        current = current or ""
        m = re.search(r"^(?:set |export )?LLMSTACK_ADMIN_TOKEN=(.*)$", current, re.M)
        if m and m.group(1).strip():
            return m.group(1).strip()
        staged = ctx.repo / ".admin-token"
        if staged.is_file():
            # push.sh can pre-stage a token so the hub already holds a peer
            # entry for a box that has never been provisioned.
            text = staged.read_text(encoding="utf-8").strip()
            if text:
                return text
        return secrets.token_urlsafe(30)[:40]

    def check(self, ctx) -> Check:
        path = ctx.plan["paths"]["envfile"]
        current, why = ctx.read_state(path)
        if current is None and why.startswith("unreadable"):
            # On the Linux boxes this file is 0640 root:llmstack. An
            # unprivileged check that called that "missing" would be answered
            # by an apply that overwrote it -- see _token().
            return Check(BLOCKED,
                         f"{path} is there and this account cannot read it "
                         f"({why}); re-run with sudo to compare it")
        try:
            want = self._render(ctx, self._token(ctx),
                                self.foreign(ctx, current))
        except RuntimeError as exc:
            return Check(BLOCKED, str(exc))
        if current == want:
            return Check(OK, path)
        if current is None:
            return Check(MISSING, "no env file",
                         [f"write {path} ({len(self._values(ctx)) + 1} values)"])
        changed = _diff_keys(current, want)
        kept = self.foreign(ctx, current)
        detail = f"{len(changed)} value(s) differ"
        if kept:
            detail += f" ({len(kept)} not fleetctl's, kept as-is)"
        return Check(DRIFT, detail,
                     [f"{verb} {k}" for verb, k in changed]
                     or ["rewrite (formatting only)"])

    def apply(self, ctx) -> None:
        current, _ = ctx.read_state(ctx.plan["paths"]["envfile"])
        text = self._render(ctx, self._token(ctx), self.foreign(ctx, current))
        # 0600: the admin token is in here. On Linux the file is root-owned
        # and group-readable by the service account; that regrade happens in
        # the grants step, which is the one that knows the account exists.
        if ctx.write(ctx.plan["paths"]["envfile"], text, mode=0o600):
            ctx.did(f"wrote {ctx.plan['paths']['envfile']}")


def _diff_keys(a: str, b: str) -> list[tuple[str, str]]:
    """(verb, key) for every key the two files disagree about.

    The verb matters. `apply` rewrites this file from the plan, so a key the
    box has and the plan does not is one the apply will DROP -- and calling
    that "update CF_ACCESS_AUD" in a dry run, as this did, describes the one
    outcome it is not. Anything hand-added to a box's env file shows up here
    as `remove`, which is the operator's cue to put it in host.yml instead.
    """
    def table(text: str) -> dict[str, str]:
        out = {}
        for line in text.splitlines():
            m = re.match(r"^(?:set |export )?([A-Z][A-Z0-9_]*)=(.*)$", line.strip())
            if m:
                out[m.group(1)] = m.group(2)
        return out

    ta, tb = table(a), table(b)
    out: list[tuple[str, str]] = []
    for key in sorted(set(ta) | set(tb)):
        if ta.get(key) == tb.get(key):
            continue
        if key not in ta:
            out.append(("add", key))
        elif key not in tb:
            out.append(("remove", key))
        else:
            out.append(("update", key))
    return out


class Wrappers(Step):
    id = "wrappers"
    title = "run-gateway / run-llama-swap"

    def wanted(self, ctx) -> bool:
        # systemd has ExecStart and EnvironmentFile and needs no wrapper. cron
        # and schtasks have neither: a crontab line and a scheduled task can
        # only name one program, so the environment has to be sourced by a
        # script that then execs the real thing.
        return shapes.SHAPES[ctx.family]["wrappers"]

    def _files(self, ctx) -> dict[str, str]:
        p = ctx.plan["paths"]
        envf, binp, venv = p["envfile"], p["bin"], p["venv"]
        out: dict[str, str] = {}
        if ctx.family == "windows":
            py = shapes.join(ctx.family, venv, "Scripts", "python.exe")
            out[shapes.join(ctx.family, binp, "run-gateway.cmd")] = (
                "@echo off\r\n"
                f'call "{envf}"\r\n'
                f'cd /d "{p["gateway"]}"\r\n'
                f'"{py}" -m uvicorn app:app --host %LLMSTACK_BIND% '
                f'--port %LLMSTACK_PORT% --timeout-keep-alive 75 '
                f'>> "{shapes.join(ctx.family, p["logs"], "gateway.log")}" 2>&1\r\n')
            if ctx.plan["engine"]["llama_swap"]:
                vis = ctx.plan["engine"]["visible_devices"]
                # Not optional where it is set. A box with an APU and a
                # discrete card has two Vulkan devices and llama.cpp picks
                # device 0 by default -- the iGPU, which advertises system RAM
                # and looks like a fine card. Every model would load onto it
                # and simply run several times slower. Filtering here rather
                # than per-model means the setting cannot be lost by editing a
                # model on the dashboard.
                vis_line = f"set GGML_VK_VISIBLE_DEVICES={vis}\r\n" if vis else ""
                out[shapes.join(ctx.family, binp, "run-llama-swap.cmd")] = (
                    "@echo off\r\n"
                    f'call "{envf}"\r\n'
                    f"{vis_line}"
                    f'"{shapes.join(ctx.family, binp, "llama-swap.exe")}" '
                    f'--config "{p["swap_config"]}" --listen '
                    f'{ctx.plan["engine"]["swap_listen"]} '
                    f'>> "{shapes.join(ctx.family, p["logs"], "llama-swap.log")}" 2>&1\r\n')
        else:
            py = shapes.join(ctx.family, venv, "bin", "uvicorn")
            out[shapes.join(ctx.family, binp, "run-gateway.sh")] = (
                "#!/bin/bash\n"
                f'source "{envf}"\n'
                f'cd "{p["gateway"]}"\n'
                f'exec "{py}" app:app --host "$LLMSTACK_BIND" '
                f'--port "$LLMSTACK_PORT" --proxy-headers --timeout-keep-alive 75\n')
            if ctx.plan["engine"]["llama_swap"]:
                out[shapes.join(ctx.family, binp, "run-llama-swap.sh")] = (
                    "#!/bin/bash\n"
                    f'source "{envf}"\n'
                    f'exec "{shapes.join(ctx.family, binp, "llama-swap")}" '
                    f'-config "{p["swap_config"]}" '
                    f'-listen {ctx.plan["engine"]["swap_listen"]}\n')
        return out

    def check(self, ctx) -> Check:
        want = self._files(ctx)
        stale, opaque = [], []
        for path, text in want.items():
            current, why = ctx.read_state(path)
            if current == text:
                continue
            # Same distinction the env file and the swap config draw: a
            # wrapper we cannot read is not a wrapper that is wrong.
            (opaque if current is None and why.startswith("unreadable")
             else stale).append(path)
        if opaque:
            return Check(BLOCKED,
                         f"{len(opaque)} wrapper(s) cannot be read from this "
                         f"account; re-run with sudo to compare them")
        if not stale:
            return Check(OK, f"{len(want)} wrapper(s) current")
        return Check(DRIFT if any(ctx.exists_state(p)[0] for p in stale) else MISSING,
                     f"{len(stale)} of {len(want)} differ",
                     [f"write {p}" for p in stale])

    def apply(self, ctx) -> None:
        for path, text in self._files(ctx).items():
            nl = "" if ctx.family == "windows" else "\n"
            if ctx.write(path, text, mode=0o755, newline=nl,
                         encoding=self._encoding(ctx, text)):
                ctx.did(f"wrote {path}")

    @staticmethod
    def _encoding(ctx, text: str) -> str:
        """cmd.exe reads a batch file in the console's OEM code page, so the
        wrappers are ASCII wherever they can be. A prefix with a non-ASCII
        character in it (a user's name, a localised folder) used to make
        this a hard `ascii` encode error -- after the venv and the files
        were in place and before any task was scheduled, which is the worst
        possible place to stop. Fall back to the OEM code page where Python
        has it (Windows), else UTF-8, and say so."""
        if ctx.family != "windows":
            return "utf-8"
        if text.isascii():
            return "ascii"
        import codecs
        try:
            codecs.lookup("oem")
            enc = "oem"
        except LookupError:
            enc = "utf-8"
        ctx.did(f"wrapper has non-ASCII characters; written as {enc}")
        return enc


class SwapConfig(Step):
    id = "swap-config"
    title = "llama-swap seed config"

    def wanted(self, ctx) -> bool:
        return bool(ctx.plan["engine"]["llama_swap"])

    SEED = """\
# GENERATED BY THE llmstack GATEWAY -- DO NOT EDIT BY HAND.
# Source of truth is {models_json}
#
# This is only a valid starting point for a box with no models registered
# yet. From here on the gateway owns this file: saving a model rewrites it
# and restarts llama-swap.
healthCheckTimeout: {timeout}
logLevel: info
startPort: 5800
metricsMaxInMemory: 5000
models: {{}}
"""

    def _text(self, ctx) -> str:
        # 1800 on a box that loads 100+ GB of weights off disk, 900 elsewhere.
        # A health-check timeout shorter than the load is a model that llama-
        # swap kills halfway in and reports as broken.
        big = (ctx.plan["sizing"]["vram_gb"] or 0) >= 32
        return self.SEED.format(
            models_json=shapes.join(ctx.family, ctx.plan["paths"]["state"],
                                    "models.json"),
            timeout=1800 if big else 900)

    def check(self, ctx) -> Check:
        path = ctx.plan["paths"]["swap_config"]
        there, why = ctx.exists_state(path)
        if there:
            # Never rewritten. After the first boot this file belongs to the
            # gateway, and replacing it would drop every registered model.
            return Check(OK, f"{path} exists (owned by the gateway from here on)")
        if there is None:
            # Not "no seed config". gpu-laptop-1 keeps this file in
            # /etc/llmstack, which is drwxrwx--- root:llmstack, so an
            # unprivileged check cannot stat it -- and it was reporting a
            # config holding seven registered models as absent, to the one
            # step that answers "absent" by writing an empty one.
            return Check(BLOCKED,
                         f"{path} cannot be seen from this account ({why}); "
                         f"re-run with sudo. Writing a seed over a live "
                         f"config would drop every model registered on this "
                         f"box")
        return Check(MISSING, "no seed config", [f"write {path}"])

    def apply(self, ctx) -> None:
        path = ctx.plan["paths"]["swap_config"]
        there, why = ctx.exists_state(path)
        if there is None:
            raise RuntimeError(
                f"{path} cannot be seen from this account ({why}) -- refusing "
                f"to write a seed config that might land on a live one. "
                f"Re-run with sudo.")
        if not there and ctx.write(path, self._text(ctx)):
            ctx.did(f"wrote seed {path}")
