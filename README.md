```
 ██████╗ ██████╗ ███████╗███╗   ██╗    ███████╗██╗     ███████╗███████╗████████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║    █████╗  ██║     █████╗  █████╗     ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██╔══╝  ██║     ██╔══╝  ██╔══╝     ██║
╚██████╔╝██║     ███████╗██║ ╚████║    ██║     ███████╗███████╗███████╗   ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝    ╚═╝     ╚══════╝╚══════╝╚══════╝   ╚═╝
```

> 🚗💨 Round up your spare machines into one private **fleet** of LLMs.

open-fleet is a self-hosted gateway and installer for running a fleet of
heterogeneous local-LLM machines behind one OpenAI-compatible, keyed, metered API
with a dashboard. Point it at whatever hardware you already own, run the
installer on each box, and get a single `/v1` endpoint that routes across all
of them.

## 🧠 Why

- **Heterogeneous hardware, one fleet.** A gaming desktop, a Mac laptop, an
  old workstation with no GPU at all, open-fleet detects what each box has
  (OS, package manager, GPU backend, VRAM) and provisions the right engine for
  it, instead of asking you to standardize the hardware first.
- **One keyed API.** Every box runs the same gateway. A "hub" gateway holds
  the peer list and routes requests to whichever peer can serve them fastest,
  so any OpenAI-compatible client can point at one URL and one key regardless
  of which machine ends up answering. 🎯
- **Honest capacity.** Context windows are not a slider that lies. Each box
  reports the context window it can *actually* launch a given model with,
  computed from the model's real GGUF geometry against that box's measured
  VRAM — so a request is routed to a box that can hold it, or told plainly
  when it can't.

## 🏗️ Physical setup

- **Pick one always-on box to be the hub.** The hub holds the peer list and
  decides which peer answers each request — it doesn't need a GPU
  (`engine.kind: none` is a perfectly good hub). An asleep hub is an
  unreachable fleet.
- **Put every box on one private network** — Tailscale is the easiest (each
  box gets a stable `100.x` address that survives DHCP leases and renames),
  but a VPN or a plain LAN works too. The hub reaches each peer at that
  stable address, never at a public API URL.
- **Peers listen on `0.0.0.0:8080`** so the hub can call their admin API; the
  hub itself usually binds `127.0.0.1` (only a reverse proxy on the same box
  ever calls it). The inference engine stays strictly on loopback — the
  gateway is the only door.
- **Keep the address you advertise** (`network.public_api_url`) away from any
  SSO/login portal — a bearer request answered with a redirect reads as a
  hang, not a 401.

## 🚀 Quick start

**Linux / macOS:**

```bash
git clone https://github.com/farawayfound/open-fleet.git
cd open-fleet
./install.sh --dry-run    # read what it would do; changes nothing
./install.sh              # detect this box, write hosts/<name>/host.yml, apply
```

**Windows (elevated PowerShell):**

```powershell
git clone https://github.com/farawayfound/open-fleet.git
cd open-fleet
.\install.ps1 -DryRun     # read what it would do; changes nothing
.\install.ps1             # detect this box, write hosts\<name>\host.yml, apply
```

Both bootstraps do one job: get to a Python 3.10+ interpreter, then hand off
to `fleetctl`, the stdlib-only installer underneath. `--dry-run` / `-DryRun`
runs every real check — the package manager is queried, files are compared,
services are asked their state — without changing anything, so it's safe to
run on a machine you're not sure about yet. 🧯

**Deploy from a dev machine.** Keep one checkout, plan each box from its own
detected facts (never plan one OS from another's), commit the plan, then let
the box apply it for real:

```bash
ssh thebox 'cd open-fleet && python3 -m fleetctl detect --json' > facts.json
python3 -m fleetctl plan --facts facts.json --host thebox --write
git add hosts/thebox/host.yml && git commit -m "provision thebox"
ssh thebox 'cd open-fleet && python3 -m fleetctl apply'
```

**Deploy from the orchestrator box (the hub).** Provision the hub first like
any other box, then let it act as the control plane: on **each peer**, after
it has run `apply` at least once (so it has an admin token), tell the hub
about it:

```bash
python3 -m fleetctl register --hub <hub-name>
```

`register` prints the peer's entry (name, tailnet address, admin token) and
the exact `curl` to paste into the hub; the peer list is the hub's own state.
Point clients at the **hub's** `/v1` to get fleet-wide routing.

## 🎁 What you get after install

- A gateway listening on `:8080`, with `GET /health` for a liveness check.
- A dashboard at `/admin/?token=<admin token>`. The admin token is written to
  this box's env file: `/etc/llmstack/gateway.env` on Linux,
  `~/llmstack/gateway.env` on macOS, `C:\llmstack\gateway.env.cmd` on Windows.
- A `hosts/<name>/host.yml` file recording what was detected and decided for
  this box — commit it, since it's the box's record and the input to every
  future `apply`.

## 🧩 How it fits together

Every box runs the same gateway (`gateway/app.py`), which fronts a local
inference engine and exposes three surfaces: `/v1` (OpenAI-compatible, bearer
keys), `/admin` (the dashboard, admin token or Cloudflare Access), and
`/health`. Behind it sits one of:

- **llama.cpp**, behind **llama-swap** (which loads/unloads models on
  demand) — for NVIDIA, AMD, Intel, and Apple Silicon GPUs.
- **Ollama**, if it's already installed on the box — open-fleet adopts it
  rather than replacing it. 🤝
- **none** — a routing-only box that serves nothing itself. This is normally
  the hub.

One gateway is designated the **hub**: it holds a peer list and, for any
model more than one peer serves, decides which peer answers a given request —
ranked by a tiering policy (does the model fit in the peer's memory, how
loaded is it, how fast has it answered before) rather than simple round-robin.
The hub's own dashboard shows the whole fleet; every other box's dashboard
shows just itself.

The inference engine on each box is bound to loopback; the gateway is the
only network-facing, authenticated door. 🚪🔒

## 📋 Minimum requirements

- **Python 3.10+** for `fleetctl` (stdlib-only — nothing else to install). On
  Linux/macOS `install.sh` installs one for you. Windows needs an **elevated**
  PowerShell session to install Python for all users and register the service.
- **`bash` and `git`** on Linux/macOS; **`git` and an elevated PowerShell**
  on Windows.
- **A private network** that the hub can reach every peer over (Tailscale, a
  VPN, or a LAN) — see **Physical setup**. A single box with no fleet needs no
  `fleet.yml` at all.
- **A GPU is optional.** A box with no GPU is a legitimate fleet member — CPU
  inference, or a routing-only hub that serves nothing.
- **One always-on box** to act as the hub.
- **Enough RAM/VRAM for what you want to serve** — open-fleet sizes context
  windows to real memory, not a slider. Routing classes: `big` (≥32 GB),
  `gpu` (≥6 GB), `small`, or `fallback` (CPU-only). Model weights need disk
  space too (often an external drive, or Ollama's own store).

## 🖥️ Supported platforms

| OS | Package family | Engines | GPU backend |
|---|---|---|---|
| Ubuntu, Debian, Linux Mint, Pop!_OS | apt | llama.cpp, Ollama, none | Vulkan (AMD/Intel), CUDA build-from-source (NVIDIA), CPU |
| Fedora, RHEL family | dnf | llama.cpp, Ollama, none | Vulkan, CPU |
| Arch, Manjaro | pacman | llama.cpp, Ollama, none | Vulkan, CPU |
| openSUSE Leap | zypper | llama.cpp, Ollama, none | Vulkan, CPU |
| macOS | Homebrew | llama.cpp, Ollama, none | Metal |
| Windows 10/11 | (all-users Python) | llama.cpp, Ollama, none | Vulkan, CUDA, ROCm, CPU |

CI runs the unit test suite on Ubuntu, macOS, and Windows, and a full
detect-plan-apply-verify pass in a container for every apt/dnf/pacman/zypper
family above.

## ⚙️ Configuration

Runtime configuration is environment variables prefixed `LLMSTACK_`, written
into each box's env file by the installer. Where those values come from is
layered:

```
4  command line       --set sizing.metal_gib=42
3  hosts/<name>/host.yml   what is true of THIS box
2  fleet.yml          what is true of every box in this fleet
1  fleetctl/shapes.py  the OS shape and engine defaults
```

`fleetctl plan --explain` prints which layer produced each value. `fleet.yml`
holds the handful of values that would otherwise be repeated on every box:

```yaml
fleet:
  name: my-fleet
  hub: hub               # the box that holds the peer list and routes
  hub_url: http://127.0.0.1:8080

network:
  bind: 0.0.0.0           # peers; the hub itself may want to pin this
  port: 8080

access:
  cf_team_domain: your-team.cloudflareaccess.com
  admin_emails:
    - admin@example.com
```

Per-box overrides live in `hosts/<name>/host.yml` — about a dozen values
covering the box's name, network address, engine, model storage path, and how
much of its GPU it may use. See `docs/examples/` for annotated samples and
[docs/usage.md](docs/usage.md) for a walkthrough of every field.

## 🔄 Updating

```bash
python -m fleetctl update
```

Pulls new gateway sources onto an already-provisioned box, restarts the
service the way that OS supervises it (systemd, a macOS cron keepalive, or a
Windows scheduled task), and waits for `/health`. `--dry-run` and `--only
<step>` both work here too.

## 🔐 Security model

- `/v1` is bearer-key auth, checked at the gateway that answers the request.
  Keys can carry an expiry and a request/token budget.
- `/admin` (the dashboard) accepts either the box's admin token as `?token=`,
  or a Cloudflare Access JWT — verified by signature against your team's
  JWKS, not merely trusted from a header.
- The inference engine behind each gateway (llama-swap or Ollama) is bound to
  `127.0.0.1`. The gateway is the only network-facing door on the box.
- Revoking a key archives it — it stops authenticating immediately, and the
  row is restorable until its retention period ends. A hashed key can never
  be re-shown.
- CI runs only on hosted runners. This project does not use or require
  self-hosted CI runners on any fleet machine.

## 🛠️ Development

```bash
pip install -r gateway/requirements.txt -r gateway/requirements-dev.txt
ruff check .
python -m pytest gateway/tests fleetctl/tests
```

## 📦 Project status, contributing, license

open-fleet is under active development. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to propose a change,
[SECURITY.md](SECURITY.md) for how to report a vulnerability,
[docs/usage.md](docs/usage.md) for a task-oriented usage guide, and
[docs/fleetctl.md](docs/fleetctl.md) for how the installer itself is built and
tested.

Licensed under Apache-2.0. See [LICENSE](LICENSE).
