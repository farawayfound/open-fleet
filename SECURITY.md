# Security

## Reporting

Use GitHub's private vulnerability reporting on this repository
(Security tab, "Report a vulnerability"). Do not open a public issue for
anything that could expose a running gateway or its keys.

You will get an acknowledgement, and a fix or a stated decision, as quickly
as a small project can manage. Credit is given in the release notes unless
you ask otherwise.

## What the gateway promises

- `/v1` accepts only bearer keys issued by the gateway. Keys are stored
  hashed; a revoked key is refused immediately.
- `/admin` accepts the break-glass admin token, or a Cloudflare Access JWT
  whose **signature** is verified against the team's JWKS and whose `aud`
  claim is checked. Headers alone are never trusted.
- The inference engine behind a gateway is bound to loopback. The gateway is
  the only authenticated door on the box.
- `fleetctl` writes secrets (the admin token, anything you add to
  `gateway.env`) to a file readable only by root and the service account, and
  never overwrites values it did not render.
- The root helper `llmstack-gpuconf` is granted through a sudoers entry that
  names exactly one path and is root-owned so the service account cannot
  replace it.

## What it does not promise

- Anything about the engines it fronts. llama.cpp, llama-swap and Ollama are
  separately maintained; keep them updated.
- Rate limiting beyond per-key budgets. Put the public `/v1` surface behind
  something that does if it faces the internet.
- Protection from a compromised hub. The hub calls every peer's admin API; a
  hub is a fleet-wide trust anchor and should be treated as one.

## For maintainers of a fork

This repository is exported from a private one through a gate that refuses
real addresses, accounts and credentials. If you fork it and run your own
fleet, keep your `fleet.yml` and `hosts/<name>/host.yml` out of any public
copy: they are, by design, a map of your network.
