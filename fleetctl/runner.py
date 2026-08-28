"""The context every step is handed: paths, processes, and a way to say no.

Three modes, and the difference between them is the whole testing story.

  live        the default. Steps read the box and change it.
  --dry-run   every step's check() runs for real -- packages are queried,
              files are stat'ed, services are asked their state -- and no
              apply() is called. What the run prints is what it WOULD do.
              This is what the CI matrix exercises on a hosted runner and in
              each distro container: it proves the plan is right for that box
              without installing anything on it.
  --root DIR  a sandbox. Every absolute path this tool touches is prefixed
              with DIR, so a complete apply can be run as an ordinary user
              against a fake filesystem tree. Borrowed from
              gateway/bin/llmstack-gpuconf, which has had exactly this hook
              (LLMSTACK_GPUCONF_ROOT) since it was written, for exactly this
              reason -- and it is what lets a real `apply` be tested on a
              real box without touching the real install.

Under --root the file steps genuinely run: directories are made, the env file
is written, the venv is built. The SYSTEM steps -- packages, services,
firewall, power, sudoers -- cannot be, because there is no fake systemd and
no fake apt, so they report `sandboxed` and record what they would have done.
Saying that plainly is the point; a sandbox that silently reported success
for a step it never ran would be worse than no sandbox.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
_COLOR = (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
          and os.environ.get("TERM") != "dumb")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


CYAN = lambda s: _c("1;36", s)     # noqa: E731
GREEN = lambda s: _c("1;32", s)    # noqa: E731
YELLOW = lambda s: _c("1;33", s)   # noqa: E731
RED = lambda s: _c("1;31", s)      # noqa: E731
DIM = lambda s: _c("2", s)         # noqa: E731


class Ctx:
    def __init__(self, plan: dict, facts: dict, *, repo: Path,
                 dry_run: bool = False, root: str | None = None,
                 verbose: bool = False, assume_yes: bool = False) -> None:
        self.plan = plan
        self.facts = facts
        self.repo = repo
        self.dry_run = dry_run
        self.root = str(Path(root).resolve()) if root else None
        self.verbose = verbose
        self.assume_yes = assume_yes
        self.family = plan["platform"]["os"]
        self.actions: list[str] = []
        self.notes: list[str] = []
        self._t0 = time.time()

    # -- identity -----------------------------------------------------------
    @property
    def sandboxed(self) -> bool:
        return self.root is not None

    @property
    def root_ok(self) -> bool:
        """May this run change the SYSTEM (units, firewall, packages)?

        A sandbox never may -- not because it lacks permission but because
        the things it would change are not the things under the sandbox root.
        """
        if self.sandboxed:
            return False
        return bool(self.facts.get("privilege", {}).get("root")
                    or self.facts.get("privilege", {}).get("sudo_nopasswd"))

    # -- paths --------------------------------------------------------------
    def path(self, p: str | Path) -> Path:
        """A path from the plan, resolved for this run.

        Under --root, `/etc/llmstack` becomes `<root>/etc/llmstack` and
        `C:\\llmstack` becomes `<root>/C/llmstack` -- the drive letter turns
        into a directory so a Windows plan can be exercised on Linux CI,
        which is where most of the Windows-plan testing actually happens.
        """
        s = str(p)
        if not self.root:
            return Path(s)
        if len(s) >= 2 and s[1] == ":":                # C:\... or C:/...
            rel = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(self.root) / s[0] / rel
        return Path(self.root) / s.lstrip("\\/").replace("\\", "/")

    def display(self, p: str | Path) -> str:
        """What to PRINT for a path: what the box will actually have, with
        the sandbox root noted separately rather than smeared through every
        line of output."""
        return str(p)

    # -- talking ------------------------------------------------------------
    def say(self, msg: str) -> None:
        print(f"    {msg}")

    def debug(self, msg: str) -> None:
        if self.verbose:
            print(DIM(f"      {msg}"))

    def note(self, msg: str) -> None:
        """Something the operator should read at the end, not scroll past."""
        self.notes.append(msg)

    def did(self, msg: str) -> None:
        self.actions.append(msg)
        self.say(msg)

    # -- processes ----------------------------------------------------------
    def run(self, argv: Sequence[str], *, system: bool = False, check: bool = True,
            capture: bool = True, timeout: int = 600, cwd: str | None = None,
            env: dict | None = None, input_text: str | None = None):
        """Run a command.

        `system=True` marks a command that changes the machine outside the
        plan's own paths -- systemctl, apt-get, netsh, schtasks, powercfg. It
        is refused in a sandbox and reported instead, because there is
        nothing under --root for it to affect.
        """
        printable = " ".join(str(a) for a in argv)
        if system and self.sandboxed:
            self.actions.append(f"[sandbox] would run: {printable}")
            self.debug(f"sandboxed, not run: {printable}")
            return None
        if self.dry_run:
            self.actions.append(f"would run: {printable}")
            return None
        self.debug(f"$ {printable}")
        full_env = {**os.environ, **(env or {})}
        r = subprocess.run([str(a) for a in argv], capture_output=capture, text=True,
                           timeout=timeout, cwd=cwd, env=full_env, input=input_text)
        if check and r.returncode != 0:
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            raise StepFailed(f"{printable}\n  exit {r.returncode}\n  "
                             + "\n  ".join(out.splitlines()[-12:]))
        return r

    def probe(self, argv: Sequence[str], timeout: int = 60):
        """A read-only command whose failure is an answer, not an error.

        Always runs, even under --dry-run: a dry run whose checks did not
        actually look at the box would report a plan for an imaginary
        machine, which is the one thing a dry run must not do.
        """
        try:
            return subprocess.run([str(a) for a in argv], capture_output=True,
                                  text=True, timeout=timeout)
        except Exception:  # noqa: BLE001
            return None

    def sudo(self, argv: Sequence[str], **kw) -> Any:
        """The same command, escalated if it has to be and can be."""
        if self.facts.get("privilege", {}).get("root") or self.family == "windows":
            return self.run(argv, system=True, **kw)
        return self.run(["sudo", "-n", *[str(a) for a in argv]], system=True, **kw)

    # -- files --------------------------------------------------------------
    def ensure_dir(self, p: str | Path, mode: int = 0o755) -> bool:
        target = self.path(p)
        if target.is_dir():
            return False
        if self.dry_run:
            self.actions.append(f"would create {self.display(p)}")
            return True
        target.mkdir(parents=True, exist_ok=True)
        if self.family != "windows":
            try:
                target.chmod(mode)
            except OSError:
                pass
        return True

    # No newline translation, anywhere a file is read back for COMPARISON.
    #
    # Python's default is universal newlines: a file's CRLF becomes LF on the
    # way in. The Windows wrapper scripts are written CRLF on purpose -- a
    # .cmd with LF endings is a batch file cmd.exe mis-parses -- so a check
    # that read them translated could never match what it had just written.
    # The step wrote the file, re-checked, still saw drift, and the run
    # stopped with UNCHANGED: an apply that ran without error and did not
    # change the answer. Found on gpu-desktop-1, gpu-laptop-2 and mini-pc-1, and invisible
    # on Linux, where the wrappers are LF either way.
    _NL = ""

    @staticmethod
    def _slurp(target: Path, encoding: str = "utf-8") -> str:
        """open(), not Path.read_text(newline=...).

        read_text() only grew a `newline` argument in Python 3.13, and the
        floor here is 3.10. This passed on a 3.13 workstation and raised
        "unexpected keyword argument" on three Windows boxes running
        3.12.10 -- which is the entire argument for the CI matrix spanning
        versions rather than testing on whatever the author happens to run.
        """
        with open(target, encoding=encoding, newline=Ctx._NL) as fh:
            return fh.read()

    def read(self, p: str | Path) -> str | None:
        try:
            return self._slurp(self.path(p))
        except OSError:
            return None

    def read_state(self, p: str | Path) -> tuple[str | None, str]:
        """(text, "") | (None, "absent") | (None, "unreadable: why").

        `read()` cannot tell those last two apart, and on this fleet the
        difference is dangerous rather than cosmetic: /etc/llmstack/gateway.env
        is 0640 root:llmstack, so an unprivileged run reads "absent" for a
        file that is very much there. The env step would then have written a
        fresh one -- with a NEW admin token, breaking the credential the hub
        holds for that peer. Same distinction the schtasks check draws, and
        the same one deploy-gateway.sh draws between a box that is offline
        and one that refused our key.
        """
        target = self.path(p)
        try:
            return self._slurp(target), ""
        except FileNotFoundError:
            return None, "absent"
        except PermissionError as exc:
            return None, f"unreadable: {exc.strerror or exc}"
        except OSError as exc:
            if target.exists():
                return None, f"unreadable: {exc.strerror or exc}"
            return None, "absent"

    def exists_state(self, p: str | Path) -> tuple[bool | None, str]:
        """(True, "") | (False, "absent") | (None, "unreadable: why").

        The same distinction read_state() draws, for a check that only wants
        to know whether a file is THERE. `Path.exists()` cannot draw it: it
        answers False for a path it is not allowed to look at, which is not a
        corner case on this fleet. /etc/llmstack is drwxrwx--- root:llmstack,
        so an unprivileged run cannot stat anything inside it -- and
        gpu-laptop-1's llama-swap.yaml, holding seven registered models, read as
        "no seed config" to the one step whose job is to write a seed when
        there is none. /etc/sudoers.d is 0750 and told the same lie about a
        grant that was installed.
        """
        target = self.path(p)
        try:
            target.stat()
            return True, ""
        except (FileNotFoundError, NotADirectoryError):
            return False, "absent"
        except OSError as exc:
            return None, f"unreadable: {exc.strerror or exc}"

    def write(self, p: str | Path, text: str, *, mode: int = 0o644,
              newline: str = "\n", encoding: str = "utf-8") -> bool:
        """Write only if the content differs. Returns whether it changed.

        Content-compare rather than always-write, because "did anything
        change" is what an idempotent run reports, and a file whose mtime
        moves on every run makes every run look like it did something.
        """
        target = self.path(p)
        current = None
        try:
            current = self._slurp(target, encoding)
        except OSError:
            pass
        if current == text:
            return False
        if self.dry_run:
            verb = "would create" if current is None else "would update"
            self.actions.append(f"{verb} {self.display(p)}")
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=encoding, newline=newline)
        if self.family != "windows":
            try:
                target.chmod(mode)
            except OSError:
                pass
        return True

    def copy(self, src: str | Path, dst: str | Path, mode: int = 0o644) -> bool:
        source = Path(src)
        target = self.path(dst)
        if target.exists() and source.exists():
            try:
                if (source.stat().st_size == target.stat().st_size
                        and source.read_bytes() == target.read_bytes()):
                    return False
            except OSError:
                pass
        if self.dry_run:
            self.actions.append(f"would install {source.name} -> {self.display(dst)}")
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if self.family != "windows":
            try:
                target.chmod(mode)
            except OSError:
                pass
        return True

    def elapsed(self) -> float:
        return time.time() - self._t0


class StepFailed(RuntimeError):
    """A step that could not do its job. Carries the command output, because
    "systemctl failed" without the journal is not a diagnosis."""
