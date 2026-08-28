"""The step catalogue: check, then apply, and never the second without the first.

Every step answers two questions independently.

  check(ctx) -> Check     read-only, ALWAYS safe, always runs. Says what the
                          box looks like now and, if that is not what the
                          plan wants, what applying would do.
  apply(ctx)              makes it so. Only ever called when check() came
                          back MISSING or DRIFT, so an idempotent re-run does
                          nothing at all rather than doing everything again
                          harmlessly.

That split is what makes `--dry-run` worth anything. The old installers had
no equivalent: they were straight-line scripts whose only mode was to run,
and "what would this do to my box" could only be answered by reading them.
Several were not in fact idempotent -- apu-box-1's llama.cpp step wiped its
build directory and recompiled from scratch on every run, and its
llama-swap step re-fetched `latest` with no version pin, so re-running a
provision could silently change the binary version.

STATES
  ok        already as the plan says
  missing   not there
  drift     there, and different from the plan
  blocked   cannot be done from here (needs root, wrong OS, no package
            manager) -- reported, never silently skipped
  skipped   the plan says not to (steps.skip, or the step does not apply)

`blocked` and `skipped` are deliberately different words. A step that needs
root and did not get it has NOT been done, and a run that called that
"skipped" would end green over a box that is half provisioned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

OK = "ok"
MISSING = "missing"
DRIFT = "drift"
BLOCKED = "blocked"
SKIPPED = "skipped"

NEEDS_WORK = (MISSING, DRIFT)


@dataclass
class Check:
    state: str
    detail: str = ""
    # What apply() would do, in words. This is the dry-run report, so it has
    # to be specific enough to act on: "would install 7 packages: git, cmake,
    # ..." rather than "would install packages".
    plan: list[str] = field(default_factory=list)


class Step:
    """One idempotent unit of provisioning."""

    id: str = ""
    title: str = ""
    # Marks a step that changes the machine outside the plan's own paths.
    # Refused in a sandbox, and reported as blocked without root.
    system: bool = False

    def needs_root(self, ctx) -> bool:  # noqa: ANN001
        """Does this step need privilege ON THIS BOX?

        `system` is the static answer and is right nearly everywhere: a step
        that writes a unit file or installs a package needs root. It is not
        the same question, though, and Homebrew is where the two come apart
        -- brew changes the machine outside the plan's paths (so it stays
        `system`, and a sandbox still refuses it) while refusing outright to
        run as root. Gating it on root reported `blocked (needs root)` on
        every Mac, about a command sudo would have made worse.
        """
        return self.system

    def wanted(self, ctx) -> bool:  # noqa: ANN001
        """Is this step part of the plan for this box at all?"""
        return True

    def check(self, ctx) -> Check:  # noqa: ANN001
        raise NotImplementedError

    def apply(self, ctx) -> None:  # noqa: ANN001
        raise NotImplementedError

    # THE entry point. Never call apply() directly.
    #
    # Ctx.run(), Ctx.write() and friends each honour --dry-run, but not
    # everything an apply does goes through them: the engine step reaches
    # for the network itself to read a release feed, and a dry run that
    # downloaded a 200 MB archive would have broken the one promise the
    # command is sold on. So the contract lives here, once, rather than as a
    # guard every step has to remember.
    def run(self, ctx) -> None:  # noqa: ANN001
        if ctx.dry_run:
            return
        self.apply(ctx)


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------
# Order matters more than it looks. The engine has to exist before a swap
# config can name its binary; the env file has to exist before a service unit
# that reads it; the service has to be running before health can mean
# anything. On the Windows boxes the original install.ps1 said so explicitly
# -- "the probe feeds the guard, the guard's verdict gates the gateway's
# catalogue, and llama-swap is the gateway's upstream" -- and that ordering is
# preserved here.
ORDER = (
    "preflight",
    "packages",
    "service-account",
    "directories",
    "python-runtime",
    "gateway-files",
    "venv",
    "envfile",
    "engine",
    "swap-config",
    "wrappers",
    "grants",
    "power",
    "firewall",
    "gpu-cap",
    "services",
    "health",
)


def catalogue() -> list[Step]:
    """Every step, in run order. Imported here rather than at module scope so
    a broken step module names itself in the traceback."""
    from . import engine, health, services, stack, system

    steps: dict[str, Step] = {}
    for mod in (stack, system, engine, services, health):
        for obj in vars(mod).values():
            if (isinstance(obj, type) and issubclass(obj, Step)
                    and obj is not Step and obj.id):
                steps[obj.id] = obj()
    missing = set(steps) - set(ORDER)
    if missing:
        raise RuntimeError(f"steps not in ORDER: {sorted(missing)}")
    return [steps[i] for i in ORDER if i in steps]


def selected(ctx, only: Iterable[str] | None = None) -> list[Step]:  # noqa: ANN001
    """The steps this run will consider, honouring `steps.skip` in host.yml
    and `--only` on the command line."""
    skip = set(ctx.plan.get("steps", {}).get("skip") or [])
    plan_only = set(ctx.plan.get("steps", {}).get("only") or [])
    want = set(only or ()) or plan_only
    out = []
    for step in catalogue():
        if step.id in skip:
            continue
        if want and step.id not in want:
            continue
        out.append(step)
    return out
