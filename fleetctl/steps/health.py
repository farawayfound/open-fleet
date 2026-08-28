"""Does it answer? The last step, and the only one whose failure means the
whole run failed.

Everything before this can be verified by inspection -- a file is there, a
unit is enabled. This one asks the box the only question that matters, over
the loopback interface, the way a peer would.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import MISSING, OK, SKIPPED, Check, Step


def probe(url: str, timeout: float = 4.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fleetctl"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310
            return True, fh.read().decode("utf-8", "replace")[:400]
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, exc.__class__.__name__


class Health(Step):
    id = "health"
    title = "health"

    def _url(self, ctx) -> str:
        # 127.0.0.1 rather than the configured bind: a box bound to 0.0.0.0 is
        # still reachable on loopback, and asking it there is the one probe
        # that cannot be answered by something else on the network.
        return f"http://127.0.0.1:{ctx.plan['network']['port']}/health"

    def check(self, ctx) -> Check:
        if ctx.sandboxed:
            return Check(SKIPPED, "sandboxed -- nothing is running under --root")
        ok, body = probe(self._url(ctx))
        if ok:
            try:
                data = json.loads(body)
                bits = [f"{k}={data[k]}" for k in ("host", "upstream", "version")
                        if k in data]
                return Check(OK, ", ".join(bits) or self._url(ctx))
            except ValueError:
                return Check(OK, self._url(ctx))
        return Check(MISSING, f"{self._url(ctx)} does not answer ({body})",
                     ["start the gateway and wait for /health"])

    def apply(self, ctx) -> None:
        """Wait, rather than start: the services step already started it, and
        a gateway takes a few seconds to open its database and seed."""
        url = self._url(ctx)
        for _ in range(20):
            ok, body = probe(url)
            if ok:
                ctx.did(f"health ok -- {body.strip()[:160]}")
                return
            time.sleep(1.5)
        tail = self._tail(ctx)
        raise RuntimeError(f"{url} never answered.\n{tail}")

    @staticmethod
    def _tail(ctx) -> str:
        """The last thing the gateway said before it did not come up. A
        health failure without this is not a diagnosis."""
        mech = ctx.plan["platform"]["service"]
        if mech == "systemd":
            r = ctx.probe(["journalctl", "-u", "llm-gateway", "-n", "25",
                           "--no-pager"])
            return (r.stdout or r.stderr or "")[-2000:] if r else ""
        from .. import shapes

        log = shapes.join(ctx.family, ctx.plan["paths"]["logs"], "gateway.log")
        text = ctx.read(log) or ""
        return "\n".join(text.splitlines()[-25:])
