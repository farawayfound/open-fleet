"""What address the hub should use for each peer, and whether it does.

The hub reaches every peer at `<url>/admin/api/...` with that peer's admin
token. A peer it cannot reach drops out of the routing table and that box's
models become unreachable fleet-wide -- so of all the live state here, the
peer list is the one where a stale address is not cosmetic.

Two questions that are easy to run together, kept apart:

  * `desired()` -- what address SHOULD the hub use for a box? The tailnet
    address recorded in `hosts/<name>/host.yml`, because it is the only name
    for a machine here that survives both DHCP and a rename.
  * `address_kind()` -- how will an address AGE? A bare name resolves only
    where the resolver does. A LAN address lasts only as long as the lease.

Both were learned the same way, twice. Renaming one of the MacBooks on
2026-08-25 retired the MagicDNS name every ssh config pointed at, and it
read as offline in CI for a day while being perfectly reachable by address.
The 2026-08-26 power cut then handed out fresh DHCP leases on the way back
up (hub .127 -> .126) and every host the deploy script had pinned by name
resolved to a machine that was no longer there.

`deploy-gateway.sh` was fixed for that. The hub's peer list was not: it was
still addressing seven of thirteen peers by bare name or by a LAN lease when
this module was written, including `gpu-laptop-1` at `192.168.1.144` -- an
address from the very lease pool that cut moved. They kept working only
because the hub happens to share a LAN with them.

Nothing here does I/O against the hub; `cli.cmd_peers` does that. These are
the decisions, so they can be tested without a fleet.
"""
from __future__ import annotations

from pathlib import Path

from . import hostfile, planner

DEFAULT_PORT = 8080

# How the hub must address a peer. Anything else is reported as drift.
DURABLE = "tailnet"


def _host_of(url: str) -> str:
    """The host out of a URL, without depending on urllib's parser quirks.

    Deliberately tolerant: this runs over whatever is already in the hub's
    peer list, which is hand-edited state and has held a bare name, a LAN
    address and a trailing slash at various times.
    """
    rest = url.split("://", 1)[-1]
    rest = rest.split("/", 1)[0]          # drop path
    rest = rest.rsplit("@", 1)[-1]        # drop userinfo
    if rest.startswith("["):              # [::1]:8080
        return rest[1:].split("]", 1)[0]
    return rest.split(":", 1)[0]          # drop port


def _ipv4(host: str) -> tuple[int, int, int, int] | None:
    parts = host.split(".")
    if len(parts) != 4:
        return None
    out = []
    for p in parts:
        if not p.isdigit() or not 0 <= int(p) <= 255:
            return None
        out.append(int(p))
    return out[0], out[1], out[2], out[3]


def address_kind(url: str) -> str:
    """How the address in `url` will age: tailnet | lan | loopback | name | ...

    `tailnet` is the only answer that is good enough for a peer URL. The
    100.64.0.0/10 range is the CGNAT block tailscale assigns out of; an
    address in it belongs to a device rather than to a network, so it
    survives a DHCP lease moving and a MagicDNS rename alike.
    """
    host = _host_of(url)
    if not host:
        return "unparsable"
    octets = _ipv4(host)
    if octets is None:
        return "ipv6" if ":" in host else "name"
    a, b, _c, _d = octets
    if a == 127:
        return "loopback"
    if a == 100 and 64 <= b <= 127:
        return DURABLE
    if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
        return "lan"
    if a == 169 and b == 254:
        return "link-local"
    return "public"


def desired(repo: Path) -> tuple[dict[str, dict], list[str]]:
    """{fleet name: {"url", "api_url"}} for every box this checkout knows.

    Reads `hosts/*/host.yml` directly rather than planning each box. The two
    values needed here are recorded rather than detected, so they can be read
    from any machine -- which matters, because the check has to be runnable
    somewhere other than the fourteen boxes it is about.

    Returns the problems alongside rather than raising: a fleet with one
    host.yml missing a tailnet_ip should still get an answer about the other
    thirteen.
    """
    site, _ = planner.load_site(repo)
    default_port = ((site.get("network") or {}).get("port")) or DEFAULT_PORT
    out: dict[str, dict] = {}
    problems: list[str] = []
    for path in sorted((repo / "hosts").glob("*/host.yml")):
        doc = hostfile.load(path)
        net = doc.get("network") or {}
        name = (doc.get("host") or {}).get("name") or path.parent.name
        ip = net.get("tailnet_ip")
        if not ip:
            problems.append(f"{name}: no network.tailnet_ip in {path}")
            continue
        port = net.get("port") or default_port
        url = f"http://{ip}:{port}"
        out[name] = {"url": url, "api_url": net.get("public_api_url") or f"{url}/v1"}
    return out, problems


def reconcile(current: list[dict], want: dict[str, dict], *,
              hub: str = "") -> tuple[list[dict], list[dict]]:
    """(the complete list to PUT back, one report row per peer).

    The list is built by EDITING the hub's own, not by rendering a new one
    from the repo, for three reasons that are all failure modes:

      * `PUT /admin/api/peers` replaces the whole list, so a partial write
        deletes every peer left out of it.
      * A peer the repo has never heard of is somebody's deliberate entry,
        not drift. It is passed through untouched.
      * A peer in the repo but missing from the hub cannot simply be added:
        the hub would have no admin token for it, and an entry with no token
        is an entry whose admin calls all 401. Those are reported for
        `fleetctl register` to handle, and not invented here.

    Every entry goes back with an empty token, which the hub reads as "keep
    the stored secret" -- so this never handles, moves or logs a token.
    """
    rows: list[dict] = []
    out: list[dict] = []
    for p in current:
        name = p.get("name", "")
        cur_url = (p.get("url") or "").rstrip("/")
        cur_api = (p.get("api_url") or "").strip()
        target = want.get(name)
        if target is None:
            rows.append({"name": name, "action": "foreign", "kind": address_kind(cur_url),
                         "url": cur_url, "want_url": cur_url,
                         "api_url": cur_api, "want_api": cur_api})
            out.append({**p, "token": ""})
            continue
        new_url, new_api = target["url"].rstrip("/"), target["api_url"]
        changed = (new_url != cur_url, new_api != cur_api)
        rows.append({
            "name": name,
            "action": "retarget" if any(changed) else "ok",
            "kind": address_kind(cur_url),
            "url": cur_url, "want_url": new_url,
            "api_url": cur_api, "want_api": new_api,
        })
        out.append({**p, "url": new_url, "api_url": new_api, "token": ""})

    seen = {p.get("name") for p in current}
    for name in sorted(want):
        if name in seen or name == hub:
            continue
        rows.append({"name": name, "action": "unregistered", "kind": "",
                     "url": "", "want_url": want[name]["url"],
                     "api_url": "", "want_api": want[name]["api_url"]})
    return out, rows


def brittle(rows: list[dict]) -> list[dict]:
    """The rows whose CURRENT address will not survive a lease or a rename."""
    return [r for r in rows
            if r["url"] and address_kind(r["url"]) != DURABLE]
