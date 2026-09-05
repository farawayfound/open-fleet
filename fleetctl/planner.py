"""Facts plus intent equals a plan: `fleetctl plan`.

Four layers, each overriding the one before it:

  1  shape defaults   fleetctl/shapes.py -- the three layouts, three engines
  2  site defaults    fleet.yml at the repo root -- everything true of every
                      box in ONE fleet: its Cloudflare Access domain, who
                      administers it, which port the fleet agreed on
  3  the host plan    hosts/<name>/host.yml -- what is true of this box alone
  4  detection        facts.py, for anything none of the above stated

Layer 2 is the one that pays for itself immediately. Counting the old
installers, five values -- LLMSTACK_BIND, LLMSTACK_PORT, LLMSTACK_UPSTREAM,
CF_ACCESS_TEAM_DOMAIN, LLMSTACK_ADMIN_EMAILS -- appeared in nine host
directories each, identical every time. They are one file now.

What is left in layer 3, after the shape and the site have had their turn, is
about a dozen values per box: its name, its address, the URL it publishes,
which engine and backend, where its weights live, how much of the GPU it may
use, and whether somebody sits in front of it. That is the honest per-box
surface -- and it is the surface the original proposal estimated at "~10
hardcoded values per box", which turns out to be right about the IDENTITY and
about a third of the real total. The rest was shape, and shape is layer 1.

Every value carries where it came from, so `--explain` can answer "why does
this box think it has 44 GiB" with a file and a layer rather than a shrug.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import hostfile, shapes

# --------------------------------------------------------------------------
# the plan's shape
# --------------------------------------------------------------------------
# Declared rather than accreted, so `plan --check` can reject a host.yml with
# a typo in it instead of silently ignoring the key. Every leaf is
# (type, required). A required leaf with no value from any layer is an error
# a person has to answer; everything else may be null.
SCHEMA: dict[str, dict[str, tuple[type | tuple, bool]]] = {
    # Site-wide, and the reason `fleetctl` itself names no host: a fork of
    # this repo changes fleet.yml and nothing else. The old installers had
    # the hub's name and this fleet's Cloudflare domain baked into fourteen
    # separate scripts.
    "fleet": {
        "name": (str, False),
        "hub": (str, False),          # the fleet name of the hub box
        "hub_url": (str, False),      # how to reach its admin API
    },
    "host": {
        "name": (str, True),
        "role": (str, True),          # peer | hub
        "klass": (str, False),        # routing class: gpu|big|small|fallback|hub
        "description": (str, False),
    },
    "platform": {
        "os": (str, True),            # linux | darwin | windows
        "arch": (str, False),
        "package_manager": (str, False),
        "service": (str, True),       # systemd | cron | schtasks | none
    },
    "paths": {
        "prefix": (str, True), "gateway": (str, True), "static": (str, True),
        "bin": (str, True), "venv": (str, True), "state": (str, True),
        "etc": (str, True), "models": (str, True), "envfile": (str, True),
        "swap_config": (str, False), "logs": (str, False), "python": (str, False),
        # Where the engine's binaries actually are. Derived from `bin` on a
        # box fleetctl installed; stated here on a box that built its own.
        "llama_server": (str, False), "llama_bench": (str, False),
        "llama_swap": (str, False),
    },
    "network": {
        "bind": (str, True), "port": (int, True), "upstream": (str, True),
        "tailnet_ip": (str, False), "public_api_url": (str, True),
    },
    "engine": {
        "kind": (str, True),          # llama.cpp | ollama | none
        "backend": (str, False),      # vulkan | cuda | rocm | metal | cpu
        "llama_swap": (bool, False),
        "swap_listen": (str, False),
        "visible_devices": (str, False),
        "models_from_upstream": (bool, False),
        "build_from_source": (bool, False),
        # Pin the release-asset pattern when the default matches more than one
        # build and the wrong one wins. masternode is why: the CUDA entry
        # matches both -cuda-12.4- and -cuda-13.3-, and 12.4 carries no sm_120
        # kernels, so a Blackwell card ran everything through PTX JIT at half
        # the tokens/s. Not a global "newest wins" -- 13.3 drops sm_50..75, so
        # a pre-Ampere box still needs 12.4.
        "llama_asset": (str, False),
    },
    "sizing": {
        "vram_gb": ((int, float), False),
        "ram_gb": ((int, float), False),
        "context_budget_gib": ((int, float), False),
        "vram_headroom": ((int, float), False),
        "max_auto_ctx": (int, False),
        "metal_gib": (int, False),
        "gtt_gib": (int, False),
    },
    "access": {
        "cf_team_domain": (str, False),
        "cf_aud": (str, False),
        "admin_emails": (list, False),
    },
    "availability": {
        "file": (str, False),         # set => an outside watchdog gates the box
        "guard": (str, False),        # name of the scheduled guard, if any
    },
    "service": {
        "user": (str, False),         # Linux service account; SYSTEM on Windows
        "units": (list, False),
    },
    "steps": {
        "skip": (list, False),
        "only": (list, False),
    },
}

class PlanError(ValueError):
    pass


def _blank() -> dict:
    return {sect: {k: None for k in keys} for sect, keys in SCHEMA.items()}


def _set(plan: dict, prov: dict, dotted: str, value: Any, layer: str) -> None:
    if value is None:
        return
    sect, _, key = dotted.partition(".")
    plan[sect][key] = value
    prov[dotted] = layer


def _merge(plan: dict, prov: dict, doc: dict, layer: str, source: str) -> None:
    """Overlay one layer's document, rejecting keys the schema has never
    heard of. A silently-ignored key in host.yml is a value someone believes
    is in force and is not."""
    for sect, body in (doc or {}).items():
        if sect not in SCHEMA:
            raise PlanError(f"{source}: unknown section {sect!r} "
                            f"(known: {', '.join(sorted(SCHEMA))})")
        if body is None:
            continue
        if not isinstance(body, dict):
            raise PlanError(f"{source}: section {sect!r} must be a mapping")
        for key, value in body.items():
            if key not in SCHEMA[sect]:
                raise PlanError(
                    f"{source}: unknown key {sect}.{key!r} "
                    f"(known: {', '.join(sorted(SCHEMA[sect]))})")
            _set(plan, prov, f"{sect}.{key}", value, f"{layer} ({source})")


def load_site(repo: Path) -> tuple[dict, str]:
    """fleet.yml, if this checkout has one. Absent is fine -- a single-box
    install has nothing to share with anybody."""
    for name in ("fleet.yml", "fleet.yaml"):
        p = repo / name
        if p.is_file():
            return hostfile.load(p), str(p)
    return {}, "(no fleet.yml)"


def load_host(repo: Path, name: str) -> tuple[dict, str]:
    p = repo / "hosts" / name / "host.yml"
    if p.is_file():
        return hostfile.load(p), str(p)
    return {}, f"(no hosts/{name}/host.yml)"


def build(facts: dict, *, repo: Path, name: str | None = None,
          overrides: dict | None = None, fresh: bool = False) -> tuple[dict, dict]:
    """The plan for this box, and where each value came from.

    `fresh` drops layer 3 -- the existing hosts/<name>/host.yml -- and plans
    from the shape, the site and detection alone. Normally that file IS an
    input, which is what makes `plan --write` keep hand edits; the cost is
    that a value which was WRONG when it was generated survives every
    regeneration, because the plan reads back what it wrote. Improving
    detection and re-running is exactly when you want the old answer ignored.
    """
    plan = _blank()
    prov: dict[str, str] = {}

    family = facts["os"]["family"]
    if family not in shapes.SHAPES:
        raise PlanError(
            f"no shape for os family {family!r} -- fleetctl knows "
            f"{', '.join(sorted(shapes.SHAPES))}")
    shape = shapes.SHAPES[family]

    # --- layer 1: the shape ------------------------------------------------
    _set(plan, prov, "platform.os", family, "detected")
    _set(plan, prov, "platform.arch", facts["os"]["arch"], "detected")
    _set(plan, prov, "platform.package_manager", facts["os"]["package_manager"],
         "detected")
    _set(plan, prov, "platform.service", facts["service_manager"], "detected")
    _set(plan, prov, "service.user", shape["service_user"], "shape")
    _set(plan, prov, "service.units", list(shape["units"]), "shape")
    _set(plan, prov, "host.role", "peer", "shape")

    # facts["home"] and nothing else. `shapes.default_home()` answers for the
    # machine doing the PLANNING, which is only the same machine some of the
    # time -- and when it is not, the fallback is silently wrong. Planning a
    # Linux box from a Linux box as root derived `/root/.ollama/models`,
    # which the distro matrix caught and this workstation could not have:
    # here the family did not match, so the fallback never fired.
    home = facts.get("home") or ""
    for key, value in shapes.paths_for(family, shapes_default_prefix(family, home),
                                       home).items():
        _set(plan, prov, f"paths.{key}", value, "shape")
    _set(plan, prov, "network.bind", "0.0.0.0", "shape")
    _set(plan, prov, "network.port", 8080, "shape")

    # --- layer 2: the site -------------------------------------------------
    site, site_src = load_site(repo)
    _merge(plan, prov, site, "site", site_src)

    # --- layer 3: this host ------------------------------------------------
    # The name has to be settled before the file can be found. Command line
    # first, then the site file's guess, then the tailnet name (which is what
    # the fleet actually calls a box), then the OS hostname lowercased.
    guess = (name
             or (overrides or {}).get("host", {}).get("name")
             or (facts.get("tailscale") or {}).get("name")
             or (facts.get("hostname") or "").split(".")[0].lower())
    _set(plan, prov, "host.name", guess, "detected")
    if fresh:
        host_doc, host_src = {}, "(--fresh: hosts/%s/host.yml ignored)" % guess
    else:
        host_doc, host_src = load_host(repo, guess) if guess else ({}, "(no host name)")
    _merge(plan, prov, host_doc, "host.yml", host_src)

    # --- layer 4: the command line ----------------------------------------
    _merge(plan, prov, overrides or {}, "override", "--set")

    # Every shape-derived value above -- the path separators, the .exe and
    # .cmd suffixes, the env-file dialect, the supervisor -- came from the
    # OS in `facts`, which is the machine that ran detect. A host.yml that
    # names a different OS is a plan for a box these facts do not describe:
    # `plan --host <linux box>` from a Windows workstation wrote
    # `paths.prefix: C:\llmstack` into a Linux host's committed plan. The
    # honest answer is to refuse, and say how to plan that box properly.
    declared = plan["platform"]["os"]
    if declared != family:
        raise PlanError(
            f"hosts/{guess}/host.yml says platform.os: {declared}, but these "
            f"facts describe a {family} machine. Run fleetctl on that box, or "
            f"plan it from its own facts:\n"
            f"  ssh {guess} 'python3 -m fleetctl detect --json' > facts.json\n"
            f"  python3 -m fleetctl plan --facts facts.json --host {guess} --write")

    # --- fill the gaps from what was measured ------------------------------
    _repath(plan, prov, family, home)
    _fill_from_facts(plan, prov, facts, family, shape)
    _derive(plan, prov, family, home, facts)
    return plan, prov


def _repath(plan: dict, prov: dict, family: str, home: str) -> None:
    """Move the shape's paths when somebody moved the prefix.

    Layer 1 laid down nine paths derived from the default prefix. If a later
    layer then set `paths.prefix` to somewhere else, the other eight are
    still pointing at the old one -- a plan that installs the venv beside an
    application it did not install. Only paths still carrying the shape's
    provenance are moved: a host that pinned `paths.models` at an external
    SSD said so on purpose."""
    if prov.get("paths.prefix", "").startswith("shape"):
        return
    for key, value in shapes.paths_for(family, plan["paths"]["prefix"], home).items():
        if key != "prefix" and prov.get(f"paths.{key}", "").startswith("shape"):
            plan["paths"][key] = value
            prov[f"paths.{key}"] = "shape (moved with paths.prefix)"


def shapes_default_prefix(family: str, home: str | None = None) -> str:
    """Where this OS puts the stack. `home` is the TARGET box's, not this
    one's -- see the note on facts["home"]."""
    if family == "windows":
        return r"C:\llmstack"
    if family == "darwin":
        base = home or shapes.default_home(family)
        # No home and no way to learn one: `~` is the honest placeholder --
        # it is wrong to write into a file, which is what makes the plan
        # complain rather than quietly install somewhere arbitrary.
        return shapes.join(family, base, "llmstack") if base else "~/llmstack"
    return "/opt/llmstack"


def _fill_from_facts(plan: dict, prov: dict, facts: dict, family: str,
                     shape: dict) -> None:
    """Anything still unset that the box itself can answer."""
    _set(plan, prov, "sizing.ram_gb", facts.get("ram_gb"), "detected")
    if plan["sizing"]["vram_gb"] is None:
        _set(plan, prov, "sizing.vram_gb", facts.get("vram_gb"), "detected")
    if plan["engine"]["backend"] is None:
        _set(plan, prov, "engine.backend", facts.get("llama_backend"), "detected")
    if plan["paths"]["python"] is None:
        _set(plan, prov, "paths.python", (facts.get("python") or {}).get("exe"),
             "detected")
    if plan["network"]["tailnet_ip"] is None:
        _set(plan, prov, "network.tailnet_ip",
             (facts.get("tailscale") or {}).get("ipv4"), "detected")

    # The engine, when nobody said. An engine already installed wins: a
    # re-provision must not swap a working Ollama box onto llama.cpp behind
    # its operator's back. Otherwise a GPU gets llama.cpp (which is the only
    # one of the two that exposes --n-cpu-moe, the lever a 16 GB card needs
    # to run a 24 GB mixture-of-experts model at all) and everything else
    # gets llama.cpp on CPU, which is still a peer worth having.
    if plan["engine"]["kind"] is None:
        engines = facts.get("engines") or {}
        if engines.get("llama_swap") or engines.get("llama_server"):
            kind = "llama.cpp"
            why = "detected (already installed)"
        elif engines.get("ollama"):
            kind = "ollama"
            why = "detected (already installed)"
        else:
            kind = "llama.cpp"
            why = "shape (default)"
        _set(plan, prov, "engine.kind", kind, why)


def _derive(plan: dict, prov: dict, family: str, home: str | None = None,
            facts: dict | None = None) -> None:
    """Values that follow from the ones now settled."""
    kind = plan["engine"]["kind"] or "llama.cpp"
    if kind not in shapes.ENGINES:
        raise PlanError(f"unknown engine.kind {kind!r} -- "
                        f"known: {', '.join(sorted(shapes.ENGINES))}")
    eng = shapes.ENGINES[kind]
    for key in ("upstream", "swap_listen", "llama_swap", "models_from_upstream"):
        target = f"engine.{key}" if key != "upstream" else "network.upstream"
        sect, _, leaf = target.partition(".")
        if plan[sect][leaf] is None:
            _set(plan, prov, target, eng[key], f"engine ({kind})")

    # Where llama.cpp actually is on this box.
    #
    # Deriving these from `bin` unconditionally is right for a box fleetctl
    # installed and wrong for one that built its own: server-1's llama-server
    # is under /home/user/llama.cpp/build/bin, its llama-swap came from
    # /usr/local/bin, and the gateway.env it has been running on for months
    # says exactly that. The engine step, looking only in /opt/llmstack/bin,
    # called that working engine `missing` -- and an apply would then have
    # downloaded a release over the top of it and repointed the env at the
    # download. Detection wins over the default; a value in host.yml wins
    # over both, which is what layer 3 is for.
    if kind == "llama.cpp":
        ext = ".exe" if family == "windows" else ""
        binroot = (shapes.join(family, plan["paths"]["bin"], "llama")
                   if family == "windows" else plan["paths"]["bin"])
        found = ((facts or {}).get("engines") or {})
        server = found.get("llama_server")

        def _engine_path(leaf: str, default: str, detected: str | None) -> None:
            """Record where a binary is, and say whether that is news.

            A detected path that matches the shape's own answer is not worth
            writing down -- provenance `shape` is what keeps it out of the
            generated host.yml (see DERIVABLE_LAYERS). A path that does NOT
            match is exactly what layer 3 is for, so it is marked `detected`
            and survives into the file.
            """
            if plan["paths"][leaf] is not None:
                return
            if detected and detected != default:
                _set(plan, prov, f"paths.{leaf}", detected, "detected")
            else:
                _set(plan, prov, f"paths.{leaf}", default, "shape (beside bin)")

        _engine_path("llama_server",
                     shapes.join(family, binroot, "llama-server" + ext), server)
        # llama-bench lives beside llama-server, whatever "beside" means on
        # this box. Sliced off the end of the name rather than parsed as a
        # path: planning for a Mac from a Windows workstation must not run a
        # POSIX path through WindowsPath and get a backslash back.
        tail = "llama-server" + ext
        _engine_path("llama_bench",
                     shapes.join(family, binroot, "llama-bench" + ext),
                     server[: -len(tail)] + "llama-bench" + ext
                     if server and server.endswith(tail) else None)
        if plan["engine"]["llama_swap"]:
            _engine_path("llama_swap",
                         shapes.join(family, plan["paths"]["bin"],
                                     "llama-swap" + ext),
                         found.get("llama_swap"))

    # An Ollama box's weights are Ollama's, in Ollama's own layout. Pointing
    # LLMSTACK_MODELS_DIR at the fleet's own directory there would advertise
    # an empty store beside a full one.
    if kind == "ollama" and prov.get("paths.models", "").startswith("shape"):
        # The store Ollama is ACTUALLY using, when detection found one --
        # the Linux installer creates a system account and stores under
        # /usr/share/ollama, and pointing at ~/.ollama there would advertise
        # an empty directory beside a full one. Failing that, the calling
        # user's home, but only if the box's own home is known: planning for
        # a Mac from a Windows workstation must not write
        # `C:/Users/<me>/.ollama/models` into that Mac's plan.
        store = ((facts or {}).get("engines") or {}).get("ollama_models")
        if store:
            _set(plan, prov, "paths.models", store, "detected (ollama store)")
        elif home:
            _set(plan, prov, "paths.models",
                 shapes.join(family, home, ".ollama", "models"),
                 f"engine ({kind})")

    # The URL a client of THIS box should use. A tailnet address is the
    # honest default: it is reachable from every peer, and unlike a
    # Cloudflare hostname it carries no Access policy -- an Access-gated
    # hostname answers a bearer request with a 302 to an SSO page, which a
    # client experiences as a hang rather than a 401.
    if plan["network"]["public_api_url"] is None and plan["network"]["tailnet_ip"]:
        _set(plan, prov, "network.public_api_url",
             f"http://{plan['network']['tailnet_ip']}:{plan['network']['port']}/v1",
             "derived (tailnet address)")
    elif plan["network"]["public_api_url"] is None and (facts or {}).get("lan_ipv4"):
        # No tailnet. A box on nothing but its LAN still has an address its
        # clients can use; it is a DHCP lease, so the provenance says so
        # and host.yml is where a durable one goes. Before this, a machine
        # with no Tailscale could not complete `install.sh` at all.
        _set(plan, prov, "network.public_api_url",
             f"http://{facts['lan_ipv4']}:{plan['network']['port']}/v1",
             "derived (LAN address -- a DHCP lease; put a durable URL in host.yml)")

    if plan["host"]["klass"] is None:
        vram = plan["sizing"]["vram_gb"] or 0
        ram = plan["sizing"]["ram_gb"] or 0
        if plan["host"]["role"] == "hub":
            klass = "hub"
        elif vram >= 32:
            klass = "big"
        elif vram >= 6:
            klass = "gpu"
        elif ram >= 12:
            klass = "small"
        else:
            klass = "fallback"
        _set(plan, prov, "host.klass", klass, "derived (from vram/ram)")

    if plan["steps"]["skip"] is None:
        _set(plan, prov, "steps.skip", [], "shape")


def validate(plan: dict, *, strict: bool = True) -> list[str]:
    """Every complaint at once, not the first one.

    A person fixing a host.yml wants the whole list; being told about one
    missing key, fixing it and being told about the next is how a five-minute
    edit becomes five round trips.
    """
    problems: list[str] = []
    for sect, keys in SCHEMA.items():
        for key, (typ, required) in keys.items():
            value = plan.get(sect, {}).get(key)
            if value is None:
                if required and strict:
                    problems.append(f"{sect}.{key} is required and nothing supplied it")
                continue
            want = typ if isinstance(typ, tuple) else (typ,)
            # bool is an int subclass; an `int` field must not accept `true`.
            if int in want and isinstance(value, bool):
                problems.append(f"{sect}.{key} should be a number, got a boolean")
            elif not isinstance(value, want):
                names = "/".join(t.__name__ for t in want)
                problems.append(
                    f"{sect}.{key} should be {names}, got "
                    f"{type(value).__name__} ({value!r})")

    if plan["platform"]["os"] not in shapes.SHAPES:
        problems.append(f"platform.os {plan['platform']['os']!r} is not one of "
                        + ", ".join(sorted(shapes.SHAPES)))
    if plan["engine"]["kind"] not in shapes.ENGINES:
        problems.append(f"engine.kind {plan['engine']['kind']!r} is not one of "
                        + ", ".join(sorted(shapes.ENGINES)))
    if plan["host"]["role"] not in ("peer", "hub"):
        problems.append("host.role must be 'peer' or 'hub'")
    port = plan["network"]["port"]
    if isinstance(port, int) and not 1 <= port <= 65535:
        problems.append(f"network.port {port} is not a port number")
    for key, value in (plan.get("paths") or {}).items():
        # A literal `~` is what a user-level shape produces when nobody could
        # say where home is. Nothing expands it -- systemd does not, cmd does
        # not, and Path() treats it as a directory called "~" -- so it has to
        # be caught here rather than become a directory nobody meant.
        if isinstance(value, str) and value.startswith("~"):
            problems.append(
                f"paths.{key} is {value!r}: the home directory of the target "
                f"box is not known. Run `fleetctl detect` on the box, or set "
                f"paths.prefix in its host.yml.")
    if plan["engine"]["kind"] == "llama.cpp" and not plan["engine"]["backend"]:
        problems.append("engine.backend is required for engine.kind: llama.cpp "
                        "(vulkan | cuda | rocm | metal | cpu)")
    return problems


# --------------------------------------------------------------------------
# what a generated host.yml says about itself
# --------------------------------------------------------------------------
# host.yml is meant to be opened and edited, so the generated one explains the
# values a person is most likely to want to change and says nothing about the
# ones they are not. Prose here, not in the code, because this is the file
# somebody reads at 2am when a box is behaving oddly.
COMMENTS: dict[str, str] = {
    "host.name": (
        "The fleet's name for this box, and the key everything else is looked\n"
        "up by: the gateway's spec sheet, the routing table, usage metering\n"
        "and the hub's peer list. A rename that misses one of them costs the\n"
        "box its specs and its rank."),
    "host.klass": (
        "Routing class. Derived from VRAM unless stated: big >= 32 GB,\n"
        "gpu >= 6 GB, small otherwise, fallback for a CPU-only box."),
    "paths.llama_server": (
        "Only here when the engine is NOT in the fleet's own bin -- a box\n"
        "that built llama.cpp itself keeps its binaries where it built them.\n"
        "Delete this line and the box takes the fleet's copy under bin/;\n"
        "leave it and fleetctl will not install over what is already there."),
    "paths.llama_bench": (
        "Beside llama_server. The gateway shells out to it to size a model\n"
        "against this box before offering it."),
    "paths.llama_swap": (
        "Same rule as llama_server: stated only when it is somewhere the\n"
        "shape would not have looked."),
    "network.bind": (
        "0.0.0.0 for a peer, whose admin API the hub calls over the tailnet.\n"
        "The hub itself is the exception and says so here: nothing calls the\n"
        "hub as a peer, cloudflared reaches it on loopback, and a hub bound\n"
        "to 0.0.0.0 puts the fleet's control plane on the tailnet for no\n"
        "reason."),
    "network.tailnet_ip": (
        "Addressed by 100.x, never by name. Names are not stable -- renaming\n"
        "a box retires the MagicDNS name every config points at, and the\n"
        "machine reads as offline while being perfectly reachable."),
    "network.public_api_url": (
        "What the dashboard hands to API clients of THIS box. It must be a\n"
        "hostname with no Cloudflare Access policy on it: an Access-gated\n"
        "hostname answers a bearer request with a 302 to an SSO page, which a\n"
        "client sees as a hang rather than a 401."),
    "engine.kind": (
        "llama.cpp (with llama-swap in front) or ollama. llama.cpp is the\n"
        "only one of the two that exposes --n-cpu-moe, which is what lets a\n"
        "card smaller than the model run a mixture-of-experts at a usable\n"
        "speed by splitting it along tensor roles instead of layers."),
    "engine.backend": (
        "vulkan for AMD and Intel, cuda for NVIDIA, metal on the Macs. ROCm\n"
        "is available and is not the default: on RDNA-class cards Vulkan\n"
        "needs nothing but the installed driver, ships a 35 MB archive\n"
        "against ROCm's 197, and is at parity or better for token generation."),
    "engine.visible_devices": (
        "Which GPU llama.cpp may see, when the box has more than one and the\n"
        "wrong one is first. An APU's integrated GPU advertises gigabytes of\n"
        "system RAM and looks like a perfectly good card to anything picking\n"
        "device 0 -- every model would load onto it and simply run several\n"
        "times slower, with the discrete card idle beside it."),
    "paths.models": (
        "Where the weights are. Frequently NOT on this box's system disk --\n"
        "an external SSD, or Ollama's own store. The gateway survives this\n"
        "path being unreachable and reports which disk it lost, rather than\n"
        "failing to start."),
    "sizing.context_budget_gib": (
        "A hard ceiling on weights + KV cache + scratch, for a box whose GPU\n"
        "can reach past its dedicated VRAM into system memory. Leave null and\n"
        "the measured VRAM is the budget."),
    "sizing.metal_gib": (
        "The Metal wired limit, on a Mac. Unified memory has no dedicated\n"
        "pool, so how much of it a model may hold is a policy number rather\n"
        "than a measurement -- and the rest of it is what keeps the machine\n"
        "usable by the person sitting at it."),
    "availability.file": (
        "Set this and an outside watchdog owns the answer to 'may the fleet\n"
        "use this machine right now'. The gateway stops advertising its\n"
        "models to the hub whenever the answer is no. For a box that is\n"
        "somebody's daily driver before it is a peer."),
    "steps.skip": (
        "Step ids `fleetctl apply` should leave alone on this box. Listed\n"
        "here rather than passed on the command line, so a box that must\n"
        "never have its firewall touched says so in its own plan."),
}

HEADER = """\
fleet host plan -- read by `fleetctl apply`, generated by `fleetctl plan`.

This file is layer 3 of four. Underneath it are the OS shape
(fleetctl/shapes.py: where things live, what supervises them) and the site
defaults (fleet.yml: what is true of every box in this fleet). Above it,
nothing but the command line. Anything not stated here is either detected on
the box or comes from one of those layers -- `fleetctl plan --explain` will
say which, for every value.

Regenerating this file with `fleetctl plan` keeps hand-edited values: it
reads the existing host.yml as an input layer. Delete a line to go back to
the detected or shape default for it."""


# A value that one of the lower layers will supply again next time does not
# belong in host.yml. Writing it there would put the shape and the site back
# into the per-box files -- which is the duplication this whole thing exists
# to remove -- and would freeze a fleet-wide change (a new admin address, a
# different port) into fourteen copies that then have to be found again.
DERIVABLE_LAYERS = ("shape", "site", "engine")

# The three paths judged by provenance rather than by the shape's path table.
#
# `paths` is normally kept-unless-it-matches-the-shape, and shapes.paths_for()
# does not produce these: they depend on which engine the box runs, which is
# not something a layout table knows. Judged that way they were kept every
# time, so a regenerated host.yml grew three boilerplate lines on every box.
# Their provenance already says whether they are news -- `detected` when the
# box's engine is somewhere the shape would not have looked, `shape` when it
# is exactly where it was put.
ENGINE_PATHS = ("llama_server", "llama_bench", "llama_swap")


def _home_from_prefix(family: str, prefix: str) -> str | None:
    """The target box's home directory, read back out of its own prefix.

    `to_document` has a plan and no facts, and on the user-level shapes the
    shape's default prefix is `<home>/llmstack` -- so without this it would
    compare the target's prefix against THIS machine's home and conclude that
    every Mac had moved its install. Which is how `/Users/user/llmstack`
    came to be written into a committed host.yml as though it were a
    deliberate override, and `C:/Users/user/llmstack` into another.
    """
    if family == "windows" or not prefix:
        return None
    tail = "/llmstack"
    norm = prefix.replace("\\", "/").rstrip("/")
    return norm[: -len(tail)] if norm.endswith(tail) else None


def to_document(plan: dict, prov: dict | None = None) -> dict:
    """The plan, trimmed to what is worth writing down.

    Kept: what this box IS (name, class, address, engine, sizing) and
    anything a person set by hand. Dropped: everything the shape, the engine
    table or fleet.yml will hand back unchanged on the next run.

    `paths` is judged differently -- nine of them follow from the prefix, so
    a path is kept only when it differs from what the shape would produce.
    That is how an external weights disk survives and eight boilerplate lines
    do not.
    """
    prov = prov or {}
    family = plan["platform"]["os"]
    prefix = plan["paths"]["prefix"] or ""
    home = _home_from_prefix(family, prefix)
    shape_paths = shapes.paths_for(family, prefix or
                                   shapes_default_prefix(family, home), home)
    doc: dict[str, dict] = {}
    for sect, keys in SCHEMA.items():
        body = {}
        for key in keys:
            value = plan[sect][key]
            if value is None:
                continue
            if sect == "paths" and key not in ENGINE_PATHS:
                if key != "prefix" and shape_paths.get(key) == value:
                    continue
                if key == "prefix" and value == shapes_default_prefix(family, home):
                    continue
            elif prov.get(f"{sect}.{key}", "").startswith(DERIVABLE_LAYERS):
                continue
            if sect == "steps" and not value:
                continue
            body[key] = value
        if body:
            doc[sect] = body
    return doc


def write(plan: dict, path: Path, prov: dict | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return hostfile.dump(to_document(plan, prov), path, COMMENTS, HEADER)


def clone(plan: dict) -> dict:
    return copy.deepcopy(plan)
