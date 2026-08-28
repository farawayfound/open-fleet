# fleetctl

One installer for a fleet that is not uniform.

*Written from inside the fleet it describes. The per-machine directories it
talks about belong to that fleet and are not published — a clone starts with
none, and grows one per box as you run the installer. See
[hosts/README.md](../hosts/README.md).*

## The problem it was written for

Fifteen host directories under `hosts/`, roughly 2,500 lines of bash and
PowerShell between them, provisioning machines that are genuinely different:
a 96 GB APU, two MacBooks, a gaming desktop somebody uses, a Raspberry Pi
class of thing, and a hub with no GPU at all.

Counting what was actually in those scripts, though, the variety was mostly
spelling:

| shape | boxes | supervisor | env file |
|---|---|---|---|
| Linux + systemd | apu-box-1, hub, cpu-box-1, server-1, gpu-laptop-1 | `systemd` units | `/etc/llmstack/gateway.env` |
| macOS + cron | mac-desktop-1, mac-laptop-2, mac-laptop-1 | `@reboot` + a 5-minute keepalive | `~/llmstack/gateway.env` |
| Windows + schtasks | apu-tablet-1, apu-tablet-2, mini-pc-1, gpu-desktop-2, gpu-laptop-2, gpu-desktop-1 | SYSTEM scheduled tasks | `C:\llmstack\gateway.env.cmd` |

| engine | boxes | upstream |
|---|---|---|
| llama.cpp behind llama-swap | apu-box-1, apu-tablet-2, mac-laptop-2, mac-laptop-1, gpu-desktop-1, gpu-laptop-1 | `127.0.0.1:8081` |
| Ollama | mac-desktop-1, cpu-box-1, mini-pc-1, gpu-laptop-2 | `127.0.0.1:11434` |
| none (routes, does not serve) | apu-tablet-1, hub | `8081`, where nothing listens |

Three shapes, three engines, and per-box identity sprinkled through the
middle. Five values — `LLMSTACK_BIND`, `LLMSTACK_PORT`, `LLMSTACK_UPSTREAM`,
`CF_ACCESS_TEAM_DOMAIN`, `LLMSTACK_ADMIN_EMAILS` — appeared in nine host
directories each, identical every time. Changing the admin address meant
editing nine heredocs and hoping none was missed.

## Four layers

```
  4  the command line     --set sizing.metal_gib=42
  3  hosts/<name>/host.yml   what is true of THIS box
  2  fleet.yml            what is true of every box in THIS fleet
  1  fleetctl/shapes.py   the three layouts and the three engines
     + detection          facts.py, for anything none of the above stated
```

`fleetctl plan --explain` prints the layer every value came from, so "why does
this box think it has 44 GiB" has an answer with a file name in it.

What survives into layer 3, after the rest have had their turn, is about a
dozen values per box: its name, its address, the URL it publishes, which
engine and backend, where its weights are, how much of the GPU it may use,
and whether somebody sits in front of it. The gaming desktop's `host.yml` is
71 lines, most of them explaining themselves.

## The commands

```
fleetctl detect                what is in this box
fleetctl plan [--write]        turn that into hosts/<name>/host.yml
fleetctl apply [--dry-run]     make the box match the plan
fleetctl verify                is it still matching?
fleetctl update                new gateway sources, restart, health
fleetctl register --hub <n>    hand the hub this peer
fleetctl peers [--push]        is the hub addressing every peer well?
```

`install.sh` and `install.ps1` wrap the first three. They are thin on purpose:
their only job is getting to a Python 3.10+, because you cannot run Python
before you have Python. Everything else is stdlib-only Python that can be
read, tested and reasoned about.

```
./install.sh                          detect, plan --write, apply
./install.sh --dry-run                ... and say what apply WOULD do
./install.sh -- verify                any other subcommand
.\install.ps1 -DryRun                 the same, elevated, on Windows
```

## check, then apply

Every step answers two questions independently:

```
check(ctx) -> Check     read-only, ALWAYS safe. What does this box look
                        like now, and if that is not the plan, what would
                        applying do?
apply(ctx)              make it so -- only ever called when check() came
                        back missing or drift
```

That split is what `--dry-run` is: real checks, no changes. The package
manager is queried, files are compared byte for byte, services are asked
their state — and nothing is written. It is safe on a machine you do not own,
it is what CI runs on every hosted runner and in every distro container, and
it is the same code path a real apply takes to decide what to do.

Five states, and the distinctions matter:

| | |
|---|---|
| `ok` | already as the plan says |
| `missing` | not there |
| `drift` | there, and different |
| `blocked` | cannot be done or even judged from here |
| `skipped` | the plan says not to, or the step does not apply |

`blocked` and `skipped` are deliberately different words. A step that needed
root and did not get it has **not** been done, and a run that called that
"skipped" would end green over a half-provisioned box.

## `--root`: a real apply, in a sandbox

```
fleetctl apply --root /tmp/sandbox --host gpu-desktop-1
```

Every absolute path is prefixed with the directory, so a complete apply runs
as an ordinary user against a fake filesystem tree. `C:\llmstack` becomes
`<root>/C/llmstack`, which is how a Windows plan gets exercised on Linux CI.

The hook is borrowed from `gateway/bin/llmstack-gpuconf`, which has had
exactly this (`LLMSTACK_GPUCONF_ROOT`) since it was written, for exactly this
reason.

Under `--root` the file steps genuinely run — directories are made, the env
file is written, the venv is built. The **system** steps (packages, services,
firewall, power, sudoers, GPU ceiling) cannot be, because there is no fake
systemd and no fake apt, so they report `sandboxed` and record what they would
have done. A sandbox that silently reported success for a step it never ran
would be worse than no sandbox.

## What testing it found

Every one of these was in the repo already, or was written and then caught.

**llama.cpp's `/releases/latest` has no binaries in it.** Upstream tags every
build as a *pre-release*, and GitHub's "latest" deliberately skips those — it
answers `v0.3.0`, whose only asset is a text file naming the nightly tag.
The gaming desktop's old `install.ps1` asked for `latest` and then filtered
for `bin-win-vulkan-x64.zip`, so it threw on every run. A tablet's copy of the
same script had hit this before and was fixed in place; the desktop's was not.
fleetctl lists `/releases` and takes the newest that actually carries the
asset, and the desktop's own installer does now too.

**The CUDA runtime is not the CUDA build.**
`cudart-llama-bin-win-cuda-12.4-x64.zip` contains
`bin-win-cuda-12.4-x64.zip` as a substring, so an unanchored search fetched
391 MB of runtime instead of the 250 MB build. The patterns are anchored, and
the runtime is fetched too — a CUDA build without it dies on a missing
`cudart64_*.dll` — with its toolkit version derived from the build's so the
two cannot disagree.

**"Access is denied" is not "not found."** `schtasks /Query` refuses a
non-elevated query of apu-tablet-2's ACL-hardened SYSTEM tasks. Read as
"missing", an unprivileged `verify` reports a healthy box as unprovisioned —
and the elevated `apply` that follows deletes and recreates a running task.

**Neither is "permission denied" on the env file.**
`/etc/llmstack/gateway.env` is `0640 root:llmstack`, so an unprivileged check
read "absent" for a file that is very much there. The env step would have
written a fresh one — with a **new admin token**, breaking the credential the
hub holds for that peer. It is `blocked` now, and `apply` refuses rather than
minting a replacement over a file it cannot see.

**A plan built on one machine can carry another's paths.** Every `host.yml`
here was generated from `detect --json` run over ssh, and the first pass wrote
`C:/Users/user/llmstack` into a macOS plan and `C:/Users/user/.ollama/models`
into a Linux one, because the planner reached for `Path.home()` on the machine
doing the planning. The dry runs on mac-laptop-1 and cpu-box-1 duly offered to create
them. The home comes from the facts now, and a plan that cannot know one says
so rather than guessing.

**A fallback that only fires half the time is worse than none.** The first fix
for that used the local home when the planning machine's OS *matched* the
target's — green on a Windows workstation, and `/root/.ollama/models` in a
Linux container. Caught by the distro matrix, which is the only place both
conditions were true.

**A 512 MB iGPU is not a GPU.** `llama_backend()` called cpu-box-1's integrated
Radeon a Vulkan device; it is a carve-out of system RAM, and offloading to it
means every token crosses the bus for a fraction of a model. Under 2 GiB is
`cpu` now — the same floor `windows_gpu_stats()` already applied, moved to
where both paths meet.

**Ollama's store is not always in a home directory.** The Linux installer
creates a system account and stores under `/usr/share/ollama`; pointing at
`~/.ollama` there advertises an empty directory beside a full one.

**`apply()` had no dry-run guard of its own.** `Ctx.run()` and `Ctx.write()`
each honour it, but the engine step reaches for the network directly, so a
dry run downloaded a release and built a virtualenv. The contract lives in
`Step.run()` now, once, instead of being left to each step to remember.

**`sh install.sh` is a thing people do.** On Debian and Ubuntu `/bin/sh` is
dash, which has no arrays and no `set -o pipefail`, and the failure is
`Syntax error: "(" unexpected` eighty lines from the cause. The script
re-execs itself under bash.

**openSUSE's `python3` package is Python 3.6.** On Leap 15.6 the bootstrap
installed it, refused it, and said so — correctly, and uselessly. It asks for
`python313`/`python312`/`python311` first now.

## What running it against the fleet found

The section above is what writing it found. This is what *running* it found —
a dry run on every machine that answered, and then the same dry run again
under `sudo`, so the checks could see the files they had been guessing about.
Every one of these is a check reporting something untrue about a box that had
been in production for months, which is the dangerous kind: the answer to a
check is an apply.

**`apply` would have deleted eight secrets from hub.** It rewrites
`gateway.env` from the plan, and hub's file holds `HF_TOKEN`,
`LLMSTACK_PUBLIC_INTAKE_TOKEN` and six `LLMSTACK_SMTP_*` values — the mail
credentials the hub's daily-brief job sends with. None of them is anything
the plan can express. The admin token had a special case for exactly this
shape of problem, and a rule that names one key does not generalise:
`LLMSTACK_SMTP_PASSWORD` looks exactly like a key fleetctl writes. The rule is
ownership now — fleetctl keeps what it renders and carries everything else
across in a block that says whose it is.

**A file you cannot read is not a file that is not there.** `/etc/llmstack` is
`drwxrwx--- root:llmstack`, so an unprivileged check cannot stat anything
inside it, and `Path.exists()` answers `False` rather than raising. gpu-laptop-1's
`llama-swap.yaml` — seven registered models — read as `no seed config` to the
one step whose answer to that is to write an empty one. `/etc/sudoers.d` is
`0750` and told the same lie about a grant installed that morning.
`Ctx.exists_state()` is `exists()` with the third answer, next to the
`read_state()` the env file has had since the admin-token near miss.

**Homebrew is not on `PATH` in a non-interactive shell,** and refuses to run
as root. `ssh mac-desktop brew list` finds nothing and says nothing, so every Mac
reported `python@3.12 missing` about a package installed since the box was
built — and then reported it `blocked (needs root)`, about the one package
manager here for which sudo is the wrong answer. `system` (changes the machine
outside the plan's paths, so a sandbox refuses it) and `needs_root` are
separate questions now, because for exactly one package manager they have
different answers.

**`ufw status` needs root, and a refusal is not an absence.** The check read a
non-zero exit as "the tool is not installed" and printed `skipped   no
firewalld or ufw here` about two boxes that are running ufw. `skipped` is the
state that means there is nothing to do here.

**An engine somebody built is not an engine to install over.** server-1 built
llama.cpp itself and keeps it in `/home/user/llama.cpp/build/bin`, which its
`gateway.env` has said since it was provisioned. The plan derived that path
from `bin`, so the engine step called a working engine `missing` — and the
apply it proposed would have downloaded a release over the top of it and
repointed the env at the download. `paths.llama_server`, `llama_bench` and
`llama_swap` are in the schema, and an engine planned outside the prefix is
`blocked` rather than reinstalled.

**`video` and `render` were required of every service account.** They are not
cosmetic where they are needed — without them the Vulkan loader falls back to
llvmpipe and every model runs on the CPU at a plausible-looking speed — but
hub runs no engine and cpu-box-1 runs Ollama under Ollama's own account. Both
reported the same drift on every run. Drift that is always there is drift
nobody reads.

**The site default binds `0.0.0.0`, which is wrong for exactly one box.** It
is right for a peer, whose admin API the hub calls over the tailnet. Nothing
calls the hub as a peer; cloudflared reaches hub on loopback. Applying the
default would have put the fleet's control plane on the tailnet in exchange
for nothing. hub pins its own bind, with the reason next to it.

**`--host mac-desktop` is not `--host mac-desktop-1`,** and planning silently from the
shape for a name with no `host.yml` is not a thing to be quiet about: the
fleet name is the key the spec sheet, the routing table, the metering and the
hub's peer list are all keyed by. It says so now and offers the near miss.
This one had already cost something — it is why an earlier run reported two of
mac-desktop's correct values as drifted.

**An empty store is not the store.** Ollama's installer creates
`/usr/share/ollama/.ollama/models` whether or not anything is pulled into it,
so "the first candidate that exists" chose an empty directory on cpu-box-1 while
the box's weights sat in `/var/lib/llmstack/models` — the exact failure the
code's own comment said it was avoiding.

**Three `host.yml` files disagreed with their boxes.** server-1 and cpu-box-1
carried the *hub's* public API URL rather than their own, so an apply would
have handed each box's API clients somebody else's endpoint. server-1's
weights are in `/home/user/models`, not under the fleet prefix. Generated
files are a starting point; the boxes are the authority.

**Two files were reported as differing on all fourteen boxes and were
identical.** `public_domains_seed.json` and `requirements.txt` had no
`.gitattributes` rule, so a Windows checkout held them CRLF — 4717 bytes of
difference in the first — and a deploy from that workstation pushed CRLF
copies over the LF ones the pi runner had installed. Thirty-nine more tracked
files predated the existing `*.py` and `*.sh` rules and were still CRLF in the
working tree, invisible to `git status`, which hashes the converted content.
One of them was `hosts/linux/grants.sh`, which `deploy-gateway.sh` ships on
every Linux deploy — and which already carried a `sed` to strip the `\r`,
which is how somebody found this the first time.

## What deploying it found

Three more, and none of them is in fleetctl. They are in the script that has
been deploying this fleet for months, and they surfaced because deploying
*carefully* takes a path nobody had taken before.

**`deploy-gateway.sh` ships the working tree,** so the careful thing is to
deploy from a clean `git worktree` at the commit — three sessions edit this
repo at once and a dirty tree ships whatever is half-written. A fresh
worktree has no `gateway/.venv`, and the interpreter hunt was `PY=python3;
[[ -x <venv> ]] && PY=<venv>; command -v "$PY" || PY=python`. On Windows
`python3` is the Microsoft Store stub: it exists, it is executable, it is on
PATH, and it prints an ad and exits 49. So `command -v` said yes, there was
no interpreter, and the run died on *"app.py or hw.py does not parse"* about
two files that parse perfectly well. The script's own comment describes the
stub; it just trusted the venv to always be there. Each candidate is now
asked to run, which is what `install.sh` does.

**cpu-box-1 was the last box addressed by a name** — `user@cpu-box-1.local`,
an mDNS name that resolves on the LAN the pi runner sits on and nowhere
else. A deploy from this workstation reported it offline while an ssh
session to it was open in the next terminal, and "offline" is the state this
script deliberately does not fail on, so the run went green having deployed
nothing. Every other box was moved to a 100.x address after the 2026-08-26
power cut, with a comment four lines below explaining exactly why. cpu-box-1
was missed because its name kept working from the one machine that runs the
deploys.

**A staged tree is not a checkout.** The Windows peers trust only the pi's
key, so deploying to them by hand means copying the sources there — where
`git rev-parse HEAD` says `unknown`, and a box stamped `unknown` is one every
later reconcile sees as behind HEAD and redeploys forever. `DEPLOY_SHA` says
which commit was staged.

## What CI found the first time it ran

The workflow had been written and never executed, because the branch had not
been pushed. Nine of sixteen jobs failed on the first run and every failure
was real: `install.sh` committed non-executable (`./install.sh: Permission
denied`, exit 126, in all five container legs); a note printed to stdout in
front of `plan --json`, which every hosted runner then failed to parse; a
`pwsh` default shell on the Windows leg reading a trailing backslash as the
end of a statement so the next line began with `--set`; a `container:` job
trying to run `actions/checkout` in images with no node and no tar, falling
back to a clone that cannot authenticate to a private repo; and a unit test
that asserted something different depending on which runner picked it up.

The last one is the instructive one. `default_home()` answers with the
running machine's home when it is the kind of machine being planned for —
correct, because `install.sh` runs on the box it provisions — and `""`
otherwise. A test asserting that an unknown home is refused therefore passed
on Linux and Windows and failed on macOS. That is not a test, it is a coin
toss, and only a matrix across all three OSes ever shows you which.

## What the hub's own peer list turned out to be

`fleetctl register` used to hand the hub `public_api_url`. That is the wrong
address and it is wrong in a way that hides: `public_api_url` is what the
dashboard gives to API *clients*, while the hub calls `<url>/admin/api/...`
with the peer's admin token. Three boxes here publish a `*.example.com`
hostname with a Cloudflare Access policy on it, and an Access-gated hostname
answers a bearer request with a **302 to an SSO page** — which the hub reads
as the peer being unreachable, dropping its models from the routing table.
Registering gpu-laptop-1 that way would have pointed the hub at a login form.

Fixing that was one line. Reading the hub's live list to check it was the
useful part, because six of the thirteen entries predate any of this:

```
retarget   apu-box-1      http://apu-box-1:8080          <- name
retarget   gpu-laptop-1    http://192.168.1.144:8080   <- lan
retarget   server-1    http://server-1:8080        <- name
retarget   cpu-box-1     http://cpu-box-1:8080         <- name
retarget   mac-desktop-1     http://mac-desktop:8080        <- name
retarget   mini-pc-1      http://mini-pc-1:8080       <- name
retarget   gpu-desktop-1       http://gpu-desktop-1:8080           <- name
```

Seven of thirteen addressed by a bare name or a DHCP lease. They all worked,
which is exactly the problem — the hub shares a LAN with those boxes, so the
names resolved and the lease was current. The 2026-08-26 power cut had
already shown what happens when that stops being true: new leases on the way
back up, every name-pinned host in `deploy-gateway.sh` resolving to a machine
that was no longer there, three boxes reported offline while serving traffic.
The deploy script was fixed for it. The peer list, which nothing regenerates,
was not — an entry is written once by hand and then outlives whatever was
true when it was written.

So `fleetctl peers` exists to make it checkable rather than remembered. It
reads the hub's list, compares it against the `tailnet_ip` in every
`hosts/*/host.yml`, and says the difference; `--push` writes the corrected
list back. Without `--push` it changes nothing, so it is safe from cron.

Two things about the push are deliberate. It **edits the hub's list rather
than rendering a new one** — `PUT /admin/api/peers` replaces the whole list,
so anything left out is deleted, and a peer the repo has never heard of is
somebody's deliberate entry rather than drift. And every entry goes back with
an **empty token**, which the gateway reads as "keep the stored secret" — so
retargeting a peer never reads, moves or logs a token.

Checking that also turned up four boxes advertising a hostname they do not
serve. `server-1` and `cpu-box-1` both claimed `llm.example.com` (which is
Access-gated, and neither box serves it — server-1 has no `cloudflared` at
all, and cpu-box-1's ingress serves `example.org`); `mini-pc-1` claimed
`api.example.com`, which is the **hub's** hostname, so a client
following it was sent to hub and served hub's models; and
`mac-desktop-1.example.com` answers 530, no origin behind it. All four now
advertise their own tailnet URL, which is what the planner derives for a box
that sets nothing.

## What was tested, and where

Real machines, over ssh, changing nothing outside `/tmp`:

| | detect | plan | dry-run | sandboxed apply | idempotent |
|---|---|---|---|---|---|
| hub (Ubuntu 26.04, hub) | ok | ok | ok | ok | ok |
| server-1 (Ubuntu 26.04) | ok | ok | ok | ok | ok |
| cpu-box-1 (Linux Mint 22.3) | ok | ok | ok | ok | ok |
| gpu-laptop-1 (Fedora 44) | ok | ok | ok | ok | ok |
| mac-laptop-1 (macOS, M1 Max) | ok | ok | ok | ok | ok |
| mac-desktop-1 (macOS, M1) | ok | ok | ok | ok | ok |
| apu-tablet-2 (Windows 11) | ok | ok | ok | — | — |
| gpu-laptop-2, mini-pc-1 (Windows 11) | ok | ok | ok | ok | ok |
| gpu-desktop-1 (Windows 11) | ok | ok | ok | ok | ok |
| pi-1, pi-2 (Debian 13, aarch64) | ok | — | — | — | — |

The Linux boxes were then run again under `sudo`, which is the only way to
know whether a check that says `blocked` was right to. Four of them were not:
see the swap config, the sudoers grant, the firewall and the eight secrets
above. A check that cannot see a file and says so is useful; one that cannot
see a file and calls it absent is the reason this second pass exists.

And once the gateway itself was deployed from this branch, `fleetctl verify`
against all six reported `gateway-files: 6 files current` — fleetctl's own
byte-for-byte comparison agreeing with what `deploy-gateway.sh` had just
shipped, hw.py and line endings included.

The three Windows peers are reached through the pi runner, which holds the
only ssh key they trust — the same hop `deploy-gateway.sh` makes from CI. The
two Pis are not fleet members; they are there because Debian on aarch64 is a
shape nothing else here covers.

Containers, each starting with no Python at all, so the bootstrap is under
test rather than assumed:

| image | family | result |
|---|---|---|
| debian:13 | apt | detect → plan → dry-run → apply → re-apply → unit suite |
| ubuntu:24.04 | apt | same |
| ubuntu:22.04 | apt | same (Python 3.10, the floor) |
| fedora:44 | dnf | same |
| archlinux:latest | pacman | same |
| opensuse/leap:15.6 | zypper | same |

`opensuse/tumbleweed:latest` is excluded: its published image is a snapshot
whose repo metadata the mirrors have already rotated away, so `zypper` cannot
install anything until the image is rebuilt. That is the image's problem
rather than zypper's, and Leap covers the family. `install.sh` clears the
metadata cache and retries once, which is the standard cure and does not help
here.

`.github/workflows/fleetctl.yml` runs all of the above on every push that
touches it, plus the unit suite on ubuntu/macos/windows across Python 3.10,
3.12 and 3.13.

## Adding a box

```bash
git clone <this repo> && cd <this repo>
./install.sh --dry-run            # read it, change nothing
./install.sh                      # detect, plan --write, apply
```

Then read `hosts/<name>/host.yml`, correct anything detection could not know
(the URL it should publish, an external weights disk, a GPU ceiling), and
`fleetctl apply` again. Commit the file: it is the box's record.

For a box that is not the one you are sitting at:

```bash
ssh thebox 'cd repo && python3 -m fleetctl detect --json' > facts.json
python3 -m fleetctl plan --facts facts.json --host thebox --write
```

## Adding a distribution

Nothing, if it belongs to one of the four families — `hw.distro_family()`
reads `ID`, then `ID_LIKE`, then whichever of `apt-get`/`dnf`/`pacman`/`zypper`
is on `PATH`. Mint and Pop!_OS cost nothing for exactly that reason.

A genuinely new family needs a row in `BASE_PACKAGES`, `PKG_INSTALL`,
`PKG_QUERY` and `PKG_REFRESH` in `shapes.py`, a branch in `install.sh`, and a
line in the CI matrix. A test asserts every family the detector can name is
one the installer can drive.

## What is deliberately not automated

**Building llama.cpp from source.** apu-box-1 and gpu-laptop-1 do; it is a
twenty-minute compile whose flags are per-card, and
`hosts/*/bootstrap/10-llamacpp.sh` still owns it. `engine.build_from_source:
true` in a host.yml says so, and the engine step tells you where to look
rather than pretending.

**Installing Ollama.** It has its own installer, which does more than drop a
binary. fleetctl says so and carries on; everything else on that box is
provisioned either way.

**Cloudflare tunnels and model pulls.** Provisioned out of band, and
unchanged by this.

**The old installers.** They are still in `hosts/`, and they still work — each
now also copies `gateway/hw.py`, without which the gateway does not import.
They come out when every box has been through fleetctl once.
