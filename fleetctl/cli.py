"""fleetctl -- detect, plan, apply, verify, update, register, peers.

    fleetctl detect                    what is in this box
    fleetctl plan [--write]            turn that into hosts/<name>/host.yml
    fleetctl apply [--dry-run]         make the box match the plan
    fleetctl verify                    is it still matching?
    fleetctl update                    new gateway sources, restart, done
    fleetctl register --hub <name>     tell the hub this peer exists
    fleetctl peers [--push]            is the hub addressing them all well?

`apply --dry-run` is the important one. Every step's check() is real -- the
package manager is queried, files are compared byte for byte, services are
asked their state -- and no apply() runs. So it is safe on a stranger's
machine, safe in CI, and it is the same code path that a real apply takes to
decide what to do. A dry run that reported on an imaginary box would be worse
than no dry run at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import facts as facts_mod
from . import planner, shapes
from . import steps as steps_mod
from .runner import CYAN, DIM, GREEN, RED, YELLOW, Ctx, StepFailed
from .steps import BLOCKED, DRIFT, MISSING, NEEDS_WORK, OK, SKIPPED

STATE_MARK = {
    OK: (GREEN, "ok"),
    MISSING: (YELLOW, "missing"),
    DRIFT: (YELLOW, "drift"),
    BLOCKED: (RED, "blocked"),
    SKIPPED: (DIM, "skipped"),
}


def repo_root() -> Path:
    """The checkout this file lives in.

    Not cwd: `fleetctl` is routinely run from a home directory over ssh, and
    resolving the gateway sources relative to wherever the shell happened to
    be is how a deploy installs nothing and reports success.
    """
    env = os.environ.get("FLEETCTL_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
def _parse_set(pairs: list[str]) -> dict:
    """--set network.port=8081 --set sizing.metal_gib=42"""
    out: dict[str, dict] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or "." not in key:
            raise SystemExit(f"--set wants section.key=value, got {pair!r}")
        sect, _, leaf = key.strip().partition(".")
        from . import hostfile

        out.setdefault(sect, {})[leaf] = hostfile._scalar(raw, 0, pair)
    return out


def _gather(args) -> tuple[dict, dict, dict]:
    # `--facts` reads a `detect --json` dump instead of measuring this box.
    # Two jobs: planning for a machine that is not the one you are sitting at
    # (every host.yml in this repo was generated that way, from a detect run
    # over ssh), and making the planner testable -- CI checks the Windows and
    # macOS plans on a Linux runner by feeding it captured facts.
    blob = getattr(args, "facts", None)
    if blob:
        f = json.loads(Path(blob).read_text(encoding="utf-8"))
    else:
        f = facts_mod.gather(quick=getattr(args, "quick", False))
    plan, prov = planner.build(f, repo=repo_root(),
                               name=getattr(args, "host", None),
                               overrides=_parse_set(getattr(args, "set", [])),
                               fresh=getattr(args, "fresh", False))
    _warn_unknown_host(args, plan)
    return f, plan, prov


def _warn_unknown_host(args, plan: dict) -> None:
    """Say when --host named a box this checkout has no plan for.

    Silence here is dangerous in a way it is not for most flags: the fleet
    name is the key everything is looked up by -- the spec sheet, the routing
    table, the metering, the hub's peer list -- so `--host mac-desktop` (the ssh
    alias) instead of `--host mac-desktop-1` (the fleet name) planned a box from
    scratch, exit 0, and reported two correct values as drifted. An apply
    would have provisioned the machine under an identity the hub does not
    know. A first provision has no host.yml either, which is why this is a
    line of output and not an error.
    """
    name = getattr(args, "host", None) or plan["host"]["name"]
    if (repo_root() / "hosts" / name / "host.yml").is_file():
        return
    # stderr, not stdout. `plan --json` and `detect --json` are parsed by the
    # thing that called them -- the CI step does exactly that -- and a note on
    # stdout is two characters of prose in front of a JSON document. Every
    # hosted runner failed with `Expecting value: line 1 column 3` the first
    # time this workflow ran, which is the whole argument for having it.
    say = lambda text: print(text, file=sys.stderr)  # noqa: E731
    say(YELLOW(f"  note: no hosts/{name}/host.yml -- planning from the shape "
               f"and this box's own answers."))
    known = sorted(p.parent.name for p in (repo_root() / "hosts").glob("*/host.yml"))
    if known:
        near = [k for k in known if k.replace("-", "") == name.replace("-", "")]
        if near:
            say(YELLOW(f"        did you mean --host {near[0]}?  "
                       f"(the FLEET name, not the ssh alias)"))
        else:
            say(DIM(f"        known: {', '.join(known)}"))


# --------------------------------------------------------------------------
def cmd_detect(args) -> int:
    f = facts_mod.gather(quick=args.quick)
    if args.json:
        print(json.dumps(f, indent=2, default=str))
        return 0
    os_ = f["os"]
    print(CYAN(f"==> {f['hostname']}"))
    print(f"  os            {os_['name']}  ({os_['family']}/{os_['arch']})")
    if os_["distro_id"]:
        like = f" like={os_['distro_like']}" if os_["distro_like"] else ""
        print(f"  distro        {os_['distro_id']} {os_['distro_version']}{like}"
              f"  -> {os_['package_manager']}")
    print(f"  supervisor    {f['service_manager']}")
    priv = f["privilege"]
    print(f"  privilege     root={priv['root']}  sudo-nopasswd={priv['sudo_nopasswd']}")
    py = f["python"]
    if py.get("exe"):
        print(f"  python        {py['exe']}  ({py['version']}, need >= {py.get('min')})")
    else:
        print(RED(f"  python        none >= {py.get('min')} found"))
        for cand in py.get("candidates", []):
            print(DIM(f"                  tried {cand['exe']}"
                      f" -> {cand['version'] or 'not usable'}"))
    print(f"  cpu           {f['cpu']['model']}  ({f['cpu']['count']} threads)")
    print(f"  ram           {f['ram_gb']} GB")
    if f["gpus"]:
        for g in f["gpus"]:
            vram = g.get("vram_total")
            size = f"{vram / 1024 ** 3:.1f} GB" if vram else "size unknown"
            print(f"  gpu           {g.get('card')}  ({size})")
    else:
        print("  gpu           none detected")
    print(f"  llama backend {f['llama_backend']}")
    ts = f["tailscale"]
    print(f"  tailnet       {ts['ipv4'] or 'not joined'}"
          + (f"  ({ts['name']})" if ts.get("name") else ""))
    eng = [k for k, v in f["engines"].items() if v]
    print(f"  engines here  {', '.join(eng) if eng else 'none'}")
    inst = f["install"]
    if inst["present"]:
        print(f"  installed at  {inst['prefix']}  "
              f"(gateway {inst['deployed_sha'] or 'unstamped'}"
              + ("" if inst["has_hw"] else RED(", MISSING hw.py")) + ")")
    else:
        print(f"  installed at  nothing at {inst['prefix']}")
    return 0


def cmd_plan(args) -> int:
    f, plan, prov = _gather(args)
    problems = planner.validate(plan, strict=not args.loose)
    if args.json:
        print(json.dumps({"plan": plan, "provenance": prov,
                          "problems": problems}, indent=2, default=str))
        return 1 if problems else 0

    name = plan["host"]["name"]
    print(CYAN(f"==> plan for {name}"))
    for sect, keys in planner.SCHEMA.items():
        rows = [(k, plan[sect][k]) for k in keys if plan[sect][k] is not None]
        if not rows:
            continue
        print(f"  {sect}")
        for key, value in rows:
            shown = ", ".join(str(v) for v in value) if isinstance(value, list) \
                else str(value)
            line = f"    {key:<22} {shown}"
            if args.explain:
                line += DIM(f"    [{prov.get(f'{sect}.{key}', '?')}]")
            print(line)
    if problems:
        print()
        print(RED(f"  {len(problems)} problem(s):"))
        for p in problems:
            print(RED(f"    - {p}"))
    if args.write:
        if problems and not args.loose:
            print(RED("  not written -- fix the problems above, or pass --loose"))
            return 1
        out = Path(args.out) if args.out else (repo_root() / "hosts" / name / "host.yml")
        planner.write(plan, out, prov)
        print()
        print(GREEN(f"  wrote {out}"))
    return 1 if problems else 0


def _flush_notes(ctx: Ctx) -> None:
    # Before every way out of _run_steps, not only the one at the bottom. A
    # step that failed to narrow a file left a note saying why, and the
    # UNCHANGED return that followed threw the run away with the note still
    # in it -- three CI rounds to learn what one line would have said.
    if ctx.notes:
        print()
        for note in ctx.notes:
            print(YELLOW(f"  note: {note}"))
        ctx.notes.clear()


def _run_steps(ctx: Ctx, args, *, apply_it: bool) -> int:
    chosen = steps_mod.selected(ctx, args.only)
    worst = 0
    # What a real apply WOULD do comes out of check(), not out of anything an
    # apply recorded -- on a dry run none of them ran.
    planned: list[str] = []
    print(CYAN(f"==> {'apply' if apply_it else 'verify'} on "
               f"{ctx.plan['host']['name']}"
               + (f"  [sandbox {ctx.root}]" if ctx.sandboxed else "")
               + ("  [dry run]" if ctx.dry_run else "")))
    for step in chosen:
        if not step.wanted(ctx):
            continue
        colour, _ = STATE_MARK[SKIPPED]
        try:
            result = step.check(ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED('ERROR')}   {step.id:<16} check failed: {exc}")
            worst = max(worst, 2)
            continue

        state = result.state
        # A system step in a sandbox, or without root, has not been done --
        # and saying "skipped" for that would let a half-provisioned box end
        # a run green.
        # Two separate gates that used to be one. A sandbox blocks every
        # system step, whatever privilege the caller has, because the things
        # those steps change do not live under --root. Root is the narrower
        # question, and Homebrew answers it differently -- see Step.needs_root.
        if state in NEEDS_WORK and step.system and (
                ctx.sandboxed or (step.needs_root(ctx) and not ctx.root_ok)):
            why = "sandboxed" if ctx.sandboxed else "needs root"
            state = BLOCKED
            result = steps_mod.Check(BLOCKED, f"{result.detail} ({why})", result.plan)

        colour, word = STATE_MARK[state]
        print(f"  {colour(word.ljust(8))}{step.id:<16} {result.detail}")
        for line in result.plan:
            print(DIM(f"            . {line}"))
        planned += result.plan

        if state == BLOCKED:
            worst = max(worst, 2)
        elif state in NEEDS_WORK:
            worst = max(worst, 1)
            if apply_it:
                try:
                    # step.run(), never step.apply(): the dry-run contract is
                    # enforced there, in one place.
                    step.run(ctx)
                    if ctx.dry_run:
                        continue
                except (StepFailed, Exception) as exc:  # noqa: BLE001
                    print(f"  {RED('FAILED')}  {step.id:<16} {exc}")
                    _flush_notes(ctx)
                    return 2
                after = step.check(ctx)
                if after.state in NEEDS_WORK:
                    # An apply that ran without error and did not change the
                    # answer is the failure mode worth catching: it is what an
                    # installer that "succeeded" and provisioned nothing looks
                    # like from the outside.
                    print(f"  {RED('UNCHANGED')} {step.id:<16} "
                          f"still {after.state}: {after.detail}")
                    _flush_notes(ctx)
                    return 2
                worst = 0 if worst == 1 else worst

    _flush_notes(ctx)
    if ctx.dry_run and planned:
        print()
        print(CYAN(f"  {len(planned)} action(s) a real apply would take"))
    return worst


def cmd_apply(args) -> int:
    f, plan, prov = _gather(args)
    problems = planner.validate(plan)
    if problems:
        print(RED("==> the plan is not complete enough to apply:"))
        for p in problems:
            print(RED(f"    - {p}"))
        print(DIM("    fix hosts/<name>/host.yml, or pass --set section.key=value"))
        return 2
    ctx = Ctx(plan, f, repo=repo_root(), dry_run=args.dry_run, root=args.root,
              verbose=args.verbose)
    rc = _run_steps(ctx, args, apply_it=True)
    print()
    if args.dry_run:
        print(CYAN(f"  dry run complete in {ctx.elapsed():.1f}s -- nothing changed"))
    elif rc == 0:
        print(GREEN(f"  {plan['host']['name']} is provisioned "
                    f"({ctx.elapsed():.1f}s)"))
    return rc


def cmd_verify(args) -> int:
    f, plan, prov = _gather(args)
    ctx = Ctx(plan, f, repo=repo_root(), dry_run=True, root=args.root,
              verbose=args.verbose)
    return _run_steps(ctx, args, apply_it=False)


def cmd_update(args) -> int:
    """New gateway sources onto a box that is already provisioned.

    Deliberately a much smaller thing than `apply`: the files land, the
    service bounces, /health has to answer. Nothing touches packages, units,
    the firewall or the env file, because an update that could change those
    is an update nobody will run on a Friday.
    """
    f, plan, prov = _gather(args)
    ctx = Ctx(plan, f, repo=repo_root(), dry_run=args.dry_run, root=args.root,
              verbose=args.verbose)
    from .steps.services import Restart
    from .steps.stack import GatewayFiles

    files = GatewayFiles()
    before = files.check(ctx)
    print(CYAN(f"==> update {plan['host']['name']}"))
    print(f"  {before.state:<9} gateway-files    {before.detail}")
    if before.state == OK and not args.force:
        print(GREEN("  already current -- nothing to do"))
        return 0
    if args.dry_run:
        for line in before.plan:
            print(DIM(f"            . {line}"))
        print(DIM("            . restart the gateway and wait for /health"))
        return 0
    files.apply(ctx)
    Restart.bounce(ctx)
    from .steps.health import Health

    Health().apply(ctx)
    sha = args.sha or _git_sha()
    if sha:
        ctx.write(shapes.join(ctx.family, plan["paths"]["gateway"], "DEPLOYED_SHA"),
                  sha + "\n")
        print(f"  stamped {sha}")
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return ""
        sha = r.stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root(),
                           capture_output=True, text=True, timeout=30)
        # A dirty tree is marked so a hand-deployed box never masquerades as
        # converged to a real commit.
        return sha + ("-dirty" if (d.stdout or "").strip() else "")
    except Exception:  # noqa: BLE001
        return ""


def _admin_token(ctx, plan: dict) -> tuple[str, str]:
    """(this box's admin token, "") or ("", why there isn't one).

    Read on the box and used from the box -- over loopback for the hub's own
    API, over the tailnet for a peer's. Nothing here carries one between
    machines.

    `read_state()` rather than `read()`, because the two ways to fail want
    two different fixes and this file is the exact case that motivated the
    distinction: /etc/llmstack/gateway.env is 0640 root:llmstack, so an
    unprivileged run gets nothing from a file that is very much there.
    Reporting that as "no admin token in <path>" sends you looking for a
    missing value when what you needed was sudo.
    """
    import re

    path = plan["paths"]["envfile"]
    env, why = ctx.read_state(path)
    if env is None:
        return "", f"{path}: {why}"
    m = re.search(r"^(?:set |export )?LLMSTACK_ADMIN_TOKEN=(.*)$", env, re.M)
    token = m.group(1).strip() if m else ""
    return (token, "") if token else ("", f"no LLMSTACK_ADMIN_TOKEN in {path}")


def _no_token(why: str) -> None:
    print(RED(f"  {why}"), file=sys.stderr)
    print(DIM("  re-run with sudo" if "unreadable" in why
              else "  run `fleetctl apply` first"), file=sys.stderr)


def cmd_register(args) -> int:
    """Hand the hub this box's name, URL and admin token.

    Printed rather than sent when there is no way to reach the hub from here:
    the peer list is the hub's own state, and a box that cannot talk to it
    should say exactly what to paste rather than failing.
    """
    f, plan, prov = _gather(args)
    ctx = Ctx(plan, f, repo=repo_root(), root=args.root)
    token, why = _admin_token(ctx, plan)
    if not token:
        _no_token(why)
        return 2
    # The TAILNET address, not public_api_url.
    #
    # These are two different addresses for two different callers, and using
    # one for the other is a live failure mode this repo already documents
    # from the other side. `public_api_url` is what the dashboard hands to
    # API CLIENTS. The hub calls `<url>/admin/api/...` with this peer's admin
    # token, and a `public_api_url` fails that call in two different ways.
    # `llm.example.com` has a Cloudflare Access policy on it and answers
    # a bearer request with a 302 to an SSO page; `mac-desktop-1.example.com`
    # answers 530, because nothing is behind it at all. The hub reads either
    # as the peer being unreachable and drops its models from the routing
    # table. Only hub, apu-box-1 and gpu-laptop-1 serve one of these hostnames --
    # every other box that advertised one was advertising somebody else's.
    #
    # A tailnet address has neither problem, and it is also the only name for
    # a box here that survives a DHCP lease moving and a rename -- the same
    # brittleness the deploy script's address table was fixed for after the
    # 2026-08-26 power cut, and that seven of the hub's own thirteen peers
    # were still carrying until `fleetctl peers` went and looked.
    tailnet = plan["network"].get("tailnet_ip")
    if tailnet:
        url = f"http://{tailnet}:{plan['network']['port']}"
        why = "tailnet address (reachable from the hub, no Access policy)"
    else:
        url = plan["network"]["public_api_url"].removesuffix("/v1")
        why = "no tailnet_ip known -- falling back to public_api_url"
    entry = {"name": plan["host"]["name"], "url": url, "token": token}
    hub = args.hub or plan["fleet"]["hub"] or "(no fleet.hub set)"
    print(CYAN(f"==> register {entry['name']} with {hub}"))
    print(json.dumps(entry, indent=2))
    print(DIM(f"  url: {why}"))
    if not tailnet:
        print(YELLOW("  the hub calls <url>/admin/api/... with this peer's admin "
                     "token; if that\n  hostname has a Cloudflare Access policy "
                     "on it the call gets a 302 and\n  the peer reads as offline. "
                     "Set network.tailnet_ip in this box's host.yml."))
    print()
    print("  on the hub, add this to its peer list:")
    print(DIM("    curl -fsS -X PUT http://127.0.0.1:8080/admin/api/peers \\"))
    print(DIM("      -H 'Authorization: Bearer <the hub admin token>' \\"))
    print(DIM("      -H 'Content-Type: application/json' \\"))
    print(DIM("      -d '<the current peer list, with the entry above added>'"))
    return 0


def _hub_call(base: str, token: str, method: str = "GET", body=None):
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + "/admin/api/peers", data=data, method=method,
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode()), ""
    except urllib.error.HTTPError as exc:
        # Read the body. The handler rejects a bad entry with
        # `400 bad peer entry: <name>`, which names the one peer of thirteen
        # it refused -- and the whole push aborts, so without that name there
        # is nothing to go on but "HTTP 400".
        detail = ""
        try:
            detail = (json.loads(exc.read().decode()) or {}).get("detail") or ""
        except Exception:  # noqa: BLE001 -- a body we cannot parse is not the error
            pass
        return None, f"HTTP {exc.code} from {base}" + (f": {detail}" if detail else "")
    except urllib.error.URLError as exc:
        return None, f"{base}: {exc.reason}"


def _peer_fingerprint(rows: list[dict]) -> list[tuple]:
    """What the hub's list looks like, for telling whether it moved."""
    return sorted((p.get("name", ""), (p.get("url") or "").rstrip("/"),
                   p.get("api_url", ""), bool(p.get("has_token")))
                  for p in rows)


def _push_damage(current: list[dict], after: list[dict]) -> tuple[list, list]:
    """(tokens lost, peers gone) -- measured against the BEFORE state.

    Against before, not against absolute truth. A peer that never had an
    admin token still has none afterwards, and reporting that as damage every
    single run is how you teach somebody to skip the one line that means the
    write really did break something.
    """
    was = {p.get("name"): p for p in current}
    now = {p.get("name"): p for p in after}
    # A peer that vanished entirely is reported as gone and not also as a
    # lost token: it is one event, and the more serious of the two names it.
    lost = sorted(n for n, p in was.items()
                  if n in now and p.get("has_token")
                  and not now[n].get("has_token"))
    return lost, sorted(set(was) - set(now))


def cmd_peers(args) -> int:
    """Is the hub addressing every peer the way it should be?

    The peer list is the hub's own live state, and nothing regenerates it --
    entries are added by hand once and then outlive whatever was true when
    they were written. That is how seven of thirteen came to be addressed by
    bare name or DHCP lease while `deploy-gateway.sh`, fixed for the same
    thing, addressed all fourteen by tailnet IP.

    So: read what the hub has, compare it against what the repo records, and
    say the difference. `--push` writes the corrected list back; without it
    this changes nothing and is safe to run from cron.
    """
    from . import peers as peers_mod

    f, plan, prov = _gather(args)
    repo = repo_root()
    want, problems = peers_mod.desired(repo)
    for p in problems:
        print(YELLOW(f"  {p}"), file=sys.stderr)

    hub = plan["fleet"]["hub"] or ""
    me = plan["host"]["name"]
    base = args.hub_url or plan["fleet"]["hub_url"] or "http://127.0.0.1:8080"
    # A loopback hub_url is the fleet.yml default and only means anything on
    # the hub itself. Saying so beats a connection refused that reads like
    # the hub being down.
    if peers_mod.address_kind(base) == "loopback" and hub and me != hub:
        print(RED(f"  {base} is loopback and this box is {me!r}, not the hub "
                  f"({hub!r})."), file=sys.stderr)
        print(DIM(f"  run this on {hub}, or pass --hub-url "
                  f"http://<hub tailnet ip>:8080"), file=sys.stderr)
        return 2

    ctx = Ctx(plan, f, repo=repo, root=args.root)
    token, why = _admin_token(ctx, plan)
    if not token:
        _no_token(why)
        return 2

    current, err = _hub_call(base, token)
    if err:
        print(RED(f"  cannot read the hub's peer list: {err}"), file=sys.stderr)
        return 2

    payload, rows = peers_mod.reconcile(current, want, hub=hub)
    drift = [r for r in rows if r["action"] in ("retarget", "unregistered")]

    if args.json:
        print(json.dumps({"hub": hub, "rows": rows}, indent=2))
        return 1 if drift else 0

    print(CYAN(f"==> peer addresses on {hub or base}"))
    for r in rows:
        if r["action"] == "unregistered":
            print(f"  {YELLOW('unregistered')} {r['name']:<11} "
                  f"known here, absent from the hub -- `fleetctl register`")
            continue
        mark = {"ok": GREEN("ok        "), "retarget": YELLOW("retarget  "),
                "foreign": DIM("foreign   ")}[r["action"]]
        note = "" if r["kind"] == peers_mod.DURABLE else DIM(f"  <- {r['kind']}")
        print(f"  {mark} {r['name']:<11} {r['url']}{note}")
        if r["want_url"] != r["url"]:
            print(f"  {'':<10} {'':<11} {GREEN('-> ' + r['want_url'])}")
        if r["want_api"] != r["api_url"]:
            print(f"  {'':<10} {'':<11} api_url {r['api_url'] or '(none)'}")
            print(f"  {'':<10} {'':<11}      -> {GREEN(r['want_api'])}")

    stale = peers_mod.brittle(rows)
    if stale:
        print()
        print(YELLOW(f"  {len(stale)} of {len(current)} peers are addressed by "
                     f"something other than a tailnet IP."))
        print(DIM("  Those resolve only where the hub's resolver does, and only "
                  "until the next\n  DHCP lease or rename moves them."))
    if not drift:
        print(GREEN("  every peer is addressed as the repo says"))
        return 0
    if not args.push:
        print()
        print(DIM("  nothing changed -- re-run with --push to write this back"))
        return 1

    # Read it again immediately before writing, and refuse if it moved.
    #
    # `PUT /admin/api/peers` replaces the whole file and the handler has no
    # version check, so anything added between the read this plan was built
    # from and the write -- by the hand-edited curl that `fleetctl register`
    # itself tells you to run, say -- would be deleted without a word. This
    # cannot close the window, but it narrows it to one round trip and makes
    # what is left a refusal instead of a silent loss.
    fresh, err = _hub_call(base, token)
    if err:
        print(RED(f"  re-read before writing failed: {err}"), file=sys.stderr)
        return 2
    if _peer_fingerprint(fresh) != _peer_fingerprint(current):
        print(RED("  the hub's peer list changed while this was deciding "
                  "-- nothing written."), file=sys.stderr)
        print(DIM("  somebody else is editing it; re-run to see the new "
                  "state."), file=sys.stderr)
        return 2

    # Tokens go back empty on purpose: the hub reads that as "keep the stored
    # secret", so a retarget never moves a token anywhere.
    _, err = _hub_call(base, token, method="PUT", body={"peers": payload})
    if err:
        print(RED(f"  push failed: {err}"), file=sys.stderr)
        return 2
    after, err = _hub_call(base, token)
    if err:
        print(YELLOW(f"  pushed, but could not read it back: {err}"))
        return 0
    print(GREEN(f"  pushed -- {len(after)} peers, "
                f"{len([r for r in rows if r['action'] == 'retarget'])} retargeted"))

    lost, gone = _push_damage(current, after)
    if gone:
        print(RED(f"  WARNING: these peers are no longer in the list: "
                  f"{', '.join(gone)}"))
    if lost:
        print(RED(f"  WARNING: the admin token is gone for {', '.join(lost)}"))
    return 2 if (lost or gone) else 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fleetctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, *, host=True, sandbox=True):
        if host:
            sp.add_argument("--host", help="fleet name for this box "
                                           "(default: the tailnet name)")
            sp.add_argument("--set", action="append", default=[], metavar="K=V",
                            help="override one plan value, e.g. "
                                 "--set sizing.metal_gib=42")
        if sandbox:
            sp.add_argument("--root", metavar="DIR",
                            help="sandbox: prefix every absolute path with DIR, "
                                 "so a whole apply can run as an ordinary user "
                                 "against a fake tree")
        sp.add_argument("--facts", metavar="FILE",
                        help="plan from a saved `detect --json` dump instead "
                             "of measuring this box")
        sp.add_argument("-v", "--verbose", action="store_true")

    d = sub.add_parser("detect", help="what is in this box")
    d.add_argument("--json", action="store_true")
    d.add_argument("--quick", action="store_true",
                   help="skip the readings that shell out")
    d.set_defaults(func=cmd_detect)

    pl = sub.add_parser("plan", help="facts + host.yml -> a complete plan")
    common(pl, sandbox=False)
    pl.add_argument("--write", action="store_true", help="write hosts/<name>/host.yml")
    pl.add_argument("--out", help="write somewhere else")
    pl.add_argument("--explain", action="store_true",
                    help="show which layer each value came from")
    pl.add_argument("--json", action="store_true")
    pl.add_argument("--fresh", action="store_true",
                    help="ignore the existing hosts/<name>/host.yml and plan "
                         "from the shape, fleet.yml and detection alone -- what "
                         "you want after improving detection, since a written "
                         "plan is otherwise read back as an input")
    pl.add_argument("--loose", action="store_true",
                    help="do not require the values a box cannot be provisioned "
                         "without")
    pl.add_argument("--quick", action="store_true")
    pl.set_defaults(func=cmd_plan)

    ap = sub.add_parser("apply", help="make the box match the plan")
    common(ap)
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="run every check, apply nothing")
    ap.add_argument("--only", action="append", metavar="STEP",
                    help="just this step (repeatable)")
    ap.add_argument("--quick", action="store_true")
    ap.set_defaults(func=cmd_apply)

    ve = sub.add_parser("verify", help="is the box still as the plan says?")
    common(ve)
    ve.add_argument("--only", action="append", metavar="STEP")
    ve.add_argument("--quick", action="store_true")
    ve.set_defaults(func=cmd_verify)

    up = sub.add_parser("update", help="new gateway sources + restart")
    common(up)
    up.add_argument("-n", "--dry-run", action="store_true")
    up.add_argument("--force", action="store_true",
                    help="restart even when the sources are already current")
    up.add_argument("--sha", help="stamp this commit instead of git HEAD")
    up.add_argument("--only", action="append", metavar="STEP")
    up.add_argument("--quick", action="store_true")
    up.set_defaults(func=cmd_update)

    rg = sub.add_parser("register", help="tell the hub this peer exists")
    common(rg)
    rg.add_argument("--hub", help="default: fleet.hub from fleet.yml")
    rg.add_argument("--quick", action="store_true")
    rg.set_defaults(func=cmd_register)

    pr = sub.add_parser("peers", help="is the hub addressing every peer well?")
    common(pr)
    pr.add_argument("--push", action="store_true",
                    help="write the corrected list back to the hub "
                         "(without this, nothing changes)")
    pr.add_argument("--hub-url", help="default: fleet.hub_url from fleet.yml")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--quick", action="store_true")
    pr.set_defaults(func=cmd_peers)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except planner.PlanError as exc:
        print(RED(f"fleetctl: {exc}"), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(RED(f"fleetctl: {exc}"), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(YELLOW("\ninterrupted -- the box is in whatever state the last "
                     "completed step left it"), file=sys.stderr)
        return 130
