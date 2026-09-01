# `hosts/`

Two different kinds of thing live here, and it is worth knowing which is which
before you go looking for a file that is not there.

## One directory per machine — written by you

`hosts/<name>/host.yml` is the plan for one box: its role, its class, the
engine it runs, where its files go, what it advertises. `fleetctl` writes it
for you:

```bash
python3 -m fleetctl plan --host thebox --write   # writes hosts/thebox/host.yml
```

`install.sh` / `install.ps1` do the same thing as their first step. The file is
yours to commit — that is the whole point of keeping plans in the repo rather
than in each machine's head. A plan is only ever written from **that box's own
detected facts**; `fleetctl plan` refuses to plan a Linux box from a Mac's
answers rather than quietly producing a plan that cannot apply.

A fresh clone has **no** machine directories in it. That is not a missing
piece: a fleet's machine list is the one thing that cannot be shipped, because
it is a map of somebody's house. Yours appear as you run the installer.

## Per-OS-family assets — shipped with the repo

The three directories that are always here hold the pieces that are the same
on every box of that family. They are small on purpose: `fleetctl` generates
units, tasks, crontab lines and env files itself, so what remains is the work
that has to happen outside the installer's reach — a root-owned daemon, a
sudoers file, a scheduled task's power settings.

| path | family | what it is | who runs it |
| --- | --- | --- | --- |
| `linux/grants.sh` | Linux | the additive `/etc/sudoers.d/llmstack-fleet` grant that lets the gateway's service account bounce its own engine and read GPU state — and installs `llmstack-gpuconf`, the only binary that grant points at | `fleetctl apply` (the `grants` step), or by hand on a box whose account predates it |
| `windows/harden-gateway-task.ps1` | Windows | fixes an already-created scheduled task whose Task Scheduler defaults stop the gateway on battery and never restart it after a crash | by hand, once, on a box provisioned before `install.ps1` started setting these at create time |
| `darwin/close-ollama-port.sh` | macOS | binds a system-daemon Ollama to loopback, so the keyed gateway is the only way in from off-box | by hand, as root, on any Mac running Ollama as a LaunchDaemon |

Each script says at the top why it exists, what it changes, and how to undo
it. All three are idempotent: running one twice is the same as running it once.

## A registry save restarts the engine

Saving a box's model registry — `PUT /admin/api/models` on the box, the same route
through the hub's `/admin/api/fleet/<box>/models`, or the Models tab — rewrites the
llama-swap config and **restarts llama-swap**, which relaunches whatever was loaded
even when the saved change (an alias, a ttl) does not alter the launch command.
A box mid-batch loses its in-flight requests; `preload`/`persistent` models come
back on their own, anything else reloads on its next request. Seen 2026-08-29
aliasing gpu-laptop-1's `qwen3.8-9B` as `qwen3.8-9b-distill` while it was serving.
