# Usage guide

A task-oriented walkthrough of open-fleet: installing a single box, wiring
several boxes into a fleet, issuing keys, adding models, and operating the
result. For the command reference and the reasoning behind the installer's
design, see [fleetctl.md](fleetctl.md).

## 1. Prerequisites

| OS | Needed before you start |
|---|---|
| Ubuntu / Debian / Linux Mint / Pop!_OS | `bash`, `git`, and either `python3` already ≥ 3.10 or a working `apt` (the installer will install Python itself) |
| Fedora / RHEL family | `bash`, `git`, a working `dnf` |
| Arch / Manjaro | `bash`, `git`, a working `pacman` |
| openSUSE Leap | `bash`, `git`, a working `zypper` |
| macOS | `git`; [Homebrew](https://brew.sh) if you don't already have Python 3.10+ |
| Windows 10/11 | `git`, and an **elevated** PowerShell session (installing Python for all users and registering a SYSTEM scheduled task both need it) |

You do not need Python installed ahead of time on Linux or macOS — `install.sh`
will use your package manager to get one if nothing suitable is on `PATH`.
Windows is the exception: Python for all users has to be installed from an
elevated shell, so `install.ps1` expects to already be running as
Administrator.

A GPU is optional. A box with no GPU at all is a legitimate fleet member —
either running CPU-only inference, or as a routing-only hub that serves
nothing itself.

## 2. Install a single box, end to end

Clone the repo on the box you're provisioning, then read before you apply:

```bash
git clone https://github.com/farawayfound/open-fleet.git
cd open-fleet
./install.sh --dry-run
```

`--dry-run` runs `fleetctl detect`, `fleetctl plan`, and then every real
`check()` an `apply` would run — the package manager is queried, files are
compared byte for byte, services are asked their state — without writing
anything. Read its output. It tells you, for each step, whether the box is
already `ok`, `missing` something, has `drift` from the plan, or is `blocked`
(needs a permission this run doesn't have).

When you're satisfied, run it for real:

```bash
./install.sh
```

This detects the box, writes `hosts/<name>/host.yml` (`<name>` defaults to
the box's tailnet or hostname), and applies the plan. `fleetctl plan
--explain` will tell you which of the four config layers produced any given
value, if you want to know why the plan looks the way it does:

```bash
python3 -m fleetctl plan --explain
```

### Reading and editing `host.yml`

`hosts/<name>/host.yml` is generated, but it's meant to be read and hand-edited
— it's the box's record, and it's what a future `apply` reads back as an
input layer (so editing it and re-running `apply` never gets clobbered by a
fresh `plan`, unless you pass `--fresh`). The fields worth knowing:

| Field | What it means |
|---|---|
| `host.name` | The fleet's name for this box. Everything else — the spec sheet, the routing table, usage metering, the hub's peer list — is keyed by it. Renaming a box means updating every place that names it, not just this file. |
| `host.klass` | Routing tier. Derived from VRAM unless you state it: `big` (≥32 GB), `gpu` (≥6 GB), `small` otherwise, `fallback` for CPU-only. |
| `host.description` | Free text. Shows up in the dashboard; has no effect on behavior. |
| `network.public_api_url` | The URL the dashboard hands to `/v1` clients of *this* box. Must resolve to a host with no SSO/auth-portal in front of it — a client sending a bearer token to a login redirect sees a hang, not a 401. |
| `network.tailnet_ip` (or any stable address) | The address the **hub** uses to reach this peer's admin API. Keep it a stable address rather than a name that can stop resolving. |
| `network.bind` | Normally left at the fleet default (`0.0.0.0`, so the hub can reach a peer). The hub itself is usually the one box that should override this to `127.0.0.1` if nothing needs to reach its admin API from elsewhere. |
| `engine.kind` | `llama.cpp` or `ollama`. llama.cpp is the one that supports offloading only part of a mixture-of-experts model to the GPU (`--n-cpu-moe`), which matters for running a model larger than VRAM at a usable speed. |
| `engine.backend` | `vulkan` (AMD/Intel GPUs), `cuda` (NVIDIA), `metal` (Apple Silicon), or `rocm`/`cpu`. Vulkan is the llama.cpp default for AMD/Intel: no extra runtime to install, a much smaller download than ROCm, and comparable token-generation speed on recent cards. |
| `engine.visible_devices` | Which GPU device index llama.cpp may use, on a box with more than one — an integrated GPU can otherwise be selected ahead of a discrete card and run everything far slower. |
| `paths.models` | Where this box's model weights live. Often not the system disk — an external drive, or Ollama's own store. |
| `sizing.vram_gb` / `sizing.ram_gb` | Detected, not usually hand-edited; what the routing and context-sizing logic uses for this box. |
| `sizing.context_budget_gib` | A hard cap on weights + KV cache + scratch, for a GPU that can reach into system memory beyond its own VRAM. Leave unset and measured VRAM is the budget. |
| `sizing.metal_gib` | On a Mac, how much of unified memory a model may claim — a policy choice, since Apple Silicon has no dedicated VRAM pool, and the rest is what keeps the machine usable for whoever is sitting at it. |
| `availability.file` / `availability.guard` | Point `file` at a JSON file an external watchdog maintains, and this box stops advertising its models to the hub whenever that file says it's unavailable — for a machine that's someone's daily driver before it's a fleet peer. |

After editing, re-apply:

```bash
./install.sh -- apply
```

(Anything after `--` is passed to `fleetctl` directly — see `./install.sh -- <cmd>` in the flag reference below.)

### Verify it came up

```bash
curl http://127.0.0.1:8080/health
```

## 3. Create a hub and register peers

One box in the fleet is the **hub**: it holds the peer list and decides which
peer answers a given `/v1` request. Pick whichever box is closest to
always-on — the hub being asleep means the whole fleet is unreachable through
it.

In `fleet.yml`, set:

```yaml
fleet:
  hub: hub-box-name
  hub_url: http://127.0.0.1:8080
```

`fleet.hub` is read by `fleetctl register` as the default target when you
don't pass `--hub` explicitly — it's the name every other box will register
against unless told otherwise.

Provision the hub itself with `install.sh` / `install.ps1` like any other
box (it can use `engine.kind: none` in its `host.yml` if it should only
route, never serve).

Then, on **each peer**, after it's been applied at least once (so it has an
admin token to hand over):

```bash
python3 -m fleetctl register --hub hub-box-name
```

`register` doesn't reach across the network and mutate the hub's state for
you — it prints the peer entry (name, address, admin token) and the exact
`curl` command to add it, because the peer list is the hub's own state and a
box that can't reach the hub should still tell you precisely what to paste:

```
==> register workstation with hub-box-name
{
  "name": "workstation",
  "url": "http://10.0.0.5:8080",
  "token": "...redacted..."
}

  on the hub, add this to its peer list:
    curl -fsS -X PUT http://127.0.0.1:8080/admin/api/peers \
      -H 'Authorization: Bearer <the hub admin token>' \
      -H 'Content-Type: application/json' \
      -d '<the current peer list, with the entry above added>'
```

Run that `curl` (with the hub's own admin token, and the *full* peer list,
existing entries included) on or against the hub. The hub polls each peer's
admin API on its own routing refresh from then on.

## 4. Issue an API key and call the fleet

Open the dashboard at `http://<hub-address>:8080/admin/?token=<admin token>`
(the admin token lives in the hub's env file — see the security model in
[README.md](../README.md)), go to **API keys**, and mint one. Give it an
expiry and a request/token budget if you want either enforced.

The key stays on screen once, at mint time — copy it then.

**curl:**

```bash
curl http://<hub-address>:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPEN_FLEET_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-id",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<hub-address>:8080/v1",
    api_key="<your open-fleet key>",
)

resp = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

Point either client at the **hub's** `/v1`, not an individual peer's, to get
fleet-wide routing. Pointing directly at one peer works too — it just only
ever answers with that box's own models.

`GET /v1/models` lists what's currently servable; `POST /v1/batches` fans a
list of chat requests out across every peer that can serve the model, if your
workload is a big backlog rather than an interactive session.

## 5. Add models

- **Dashboard → Library** (per-box, on that machine's own dashboard):
  search HuggingFace for a GGUF, pull it with a resumable download, and
  manage disk usage — this is the normal path on a llama.cpp box.
- **llama.cpp / llama-swap**: the dashboard's **Models** tab edits the
  llama-swap catalogue directly — each row is one llama-server invocation.
  Leave a model's context (`ctx`) at `0` to size the window automatically
  from the box's measured VRAM and the GGUF's own geometry, or pin a number
  yourself. **Save & apply** loads every changed model for real, one at a
  time, and rolls back anything that doesn't fit rather than leaving the box
  in a half-applied state.
- **Ollama**: open-fleet adopts an existing Ollama install rather than
  wrapping its own installer or catalogue. Pull models with `ollama pull` as
  usual; the gateway reads Ollama's own model list rather than a local
  directory.

## 6. Operating the fleet

**Is a box still what its plan says?**

```bash
python3 -m fleetctl verify
```

Compares the box against `hosts/<name>/host.yml` the same way `apply` would,
without changing anything. `--only <step>` restricts it to one step
(repeatable).

**Pull in new gateway code:**

```bash
python3 -m fleetctl update
```

Copies the current gateway sources onto the box, restarts the service the
way that OS supervises it, and waits for `/health`. `--dry-run` shows what it
would do; `--force` restarts even when the sources are already current;
`--sha <commit>` stamps a specific commit rather than the checkout's current
`git HEAD`.

**Logs and service control, per OS:**

| OS | Service | Logs |
|---|---|---|
| Linux (systemd) | `systemctl status llm-gateway`, `systemctl restart llm-gateway` | `journalctl -u llm-gateway` |
| macOS (cron) | a `@reboot` entry plus a keepalive cron job restart it; no service manager to query directly | dashboard's **Logs** tab, or the log path under the box's state directory |
| Windows (Task Scheduler) | `schtasks /Query /TN llm-gateway`, `schtasks /Run /TN llm-gateway` | dashboard's **Logs** tab |

The dashboard's own **Logs** tab (per machine) is the OS-independent way to
read the service's recent output without shelling in.

**Safe experimentation:**

- `--dry-run` (or `-n`) on `apply` and `update` runs every check for real and
  changes nothing.
- `--root <dir>` runs a *complete* apply — files genuinely get written — but
  with every absolute path prefixed by `<dir>`, so it lands in a sandbox
  directory instead of the real filesystem. Steps that touch the actual OS
  (installing packages, registering services, firewall rules) can't be
  faked this way and report themselves as sandboxed rather than pretending
  to succeed.
- `--facts <file>` plans from a saved `fleetctl detect --json` dump instead
  of measuring the box you're running on — useful for planning a remote box
  from its own facts:

  ```bash
  ssh thatbox 'cd open-fleet && python3 -m fleetctl detect --json' > facts.json
  python3 -m fleetctl plan --facts facts.json --host thatbox --write
  ```

**Command reference** (all of `detect`, `plan`, `apply`, `verify`, `update`,
`register` accept `--host`, `--set section.key=value`, `--facts <file>`, and
`-v`/`--verbose`; `apply`, `verify`, `update`, `register` additionally accept
`--root <dir>`):

```
fleetctl detect [--json] [--quick]
fleetctl plan [--write] [--out FILE] [--explain] [--json] [--fresh] [--loose] [--quick]
fleetctl apply [-n|--dry-run] [--only STEP ...] [--quick]
fleetctl verify [--only STEP ...] [--quick]
fleetctl update [-n|--dry-run] [--force] [--sha SHA] [--only STEP ...] [--quick]
fleetctl register [--hub NAME] [--quick]
```

## 7. Optional: Cloudflare Access in front of the dashboard

open-fleet does not set up a tunnel for you — that's deliberately out of
scope — but the gateway will verify a Cloudflare Access JWT if you put one in
front of the dashboard yourself. Set, in the box's env file:

```
CF_ACCESS_TEAM_DOMAIN=your-team.cloudflareaccess.com
CF_ACCESS_AUD=<the Access application's audience tag>
LLMSTACK_ADMIN_EMAILS=admin@example.com,other-admin@example.com
```

The gateway fetches your team's JWKS and verifies the JWT's **signature**
against it — a spoofed identity header does nothing — and checks the
authenticated email against `LLMSTACK_ADMIN_EMAILS`. Leave `CF_ACCESS_AUD`
unset on a box with no tunnel; it'll keep answering `/admin` to its `?token=`
admin token over your tailnet or LAN instead.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `install.ps1` or a scheduled task can't find Python, even though `python --version` works in your interactive shell | The Microsoft Store's Python stub is on `PATH` — it answers to `python`/`python3`, prints an ad, and exits nonzero. It looks like an interpreter to anything that only checks the command exists | Let `install.ps1` install Python for all users (`InstallAllUsers=1`), or install a real interpreter from python.org and make sure it's ahead of the Store alias on `PATH` |
| `sh install.sh` fails with `Syntax error: "(" unexpected`, eighty-odd lines from anything relevant | On Debian/Ubuntu `/bin/sh` is `dash`, which has no arrays and no `pipefail` — this script needs `bash` | Run `bash install.sh` or `./install.sh` (it re-execs itself under bash on its own, but only if it was invoked as a script, not piped into `sh`) |
| A step reports `blocked` on a dry run | It cannot be verified *or* changed from the privilege level this run has — different from `skipped` (the plan says not to touch it) | Re-run with the privilege the step needs (`sudo`, an elevated shell). Don't read `blocked` as "safe to ignore" |
| `verify` reports files or config as missing that you know are there | An unprivileged read on a permission-restricted path (`0640`, `0750`, etc.) returns "not found" indistinguishably from actually-missing, unless the step is written to tell the two apart | Run `verify`/`apply` with the permissions to actually read the path — usually `sudo` |
| Every model runs at CPU speed despite a GPU being installed | The service account doesn't have the `render`/`video` group (or platform equivalent), so the Vulkan loader silently falls back to `llvmpipe` (software rendering) instead of failing loudly | Add the service account to the groups that own the GPU device nodes, then restart the engine |
| Fetching a llama.cpp release fails, or grabs a build with no working binaries | GitHub's "latest release" for llama.cpp is often a pre-release with no built assets attached — only a tag name | fleetctl lists all releases and picks the newest one that actually carries a matching asset; if you're fetching manually, do the same rather than trusting "latest" |
| `brew`-installed packages read as missing when you check over `ssh` | Homebrew's shim usually isn't on `PATH` in a non-interactive shell, and refuses to run as root | Check with an interactive login shell, or reference the full binary path (e.g. `/opt/homebrew/bin/brew`) rather than relying on `PATH` |
