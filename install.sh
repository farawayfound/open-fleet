#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install.sh -- get to a Python, then hand over to fleetctl.
#
# That is the whole job. Everything a box actually needs is in fleetctl/,
# which is stdlib-only Python and can therefore be read, tested and reasoned
# about; this file exists because you cannot run Python before you have
# Python, and that one bootstrap problem is the only thing shell is still
# the right tool for.
#
# The counting that motivated it: the fifteen scripts this replaces ran to
# roughly 2,500 lines of bash and PowerShell between them, most of it the
# same work spelled three ways. This is ~120 lines and does not grow when a
# sixteenth box joins.
#
#   ./install.sh                     detect, plan, apply
#   ./install.sh --dry-run           detect, plan, and say what apply WOULD do
#   ./install.sh --host gpu-laptop-1     name the box explicitly
#   ./install.sh -- detect           run any other fleetctl command instead
#
# Anything after `--` is passed to fleetctl verbatim.
# ---------------------------------------------------------------------------

# Re-exec under bash if something invoked this with `sh install.sh`.
#
# Not pedantry: on Debian and Ubuntu /bin/sh is dash, which has no arrays and
# no `set -o pipefail`, and the failure it produces is `Syntax error: "("
# unexpected` on a line eighty below the point of interest. A bootstrap
# script is exactly the thing people run the wrong way -- pasted from a
# README, or as `sh -c` from a provisioning tool -- so it re-launches itself
# rather than blaming them.
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then
    exec bash "$0" "$@"
  fi
  echo "install.sh needs bash (this shell is not one, and none is on PATH)." >&2
  echo "Install bash, or run: python3 -m fleetctl detect" >&2
  exit 1
fi

set -euo pipefail

# Resolve a symlinked entry point (~/bin/install.sh -> checkout/install.sh is
# an ordinary way to make a script reachable) so REPO is the real checkout,
# not the directory the link lives in.
SRC="${BASH_SOURCE[0]}"
while [ -h "$SRC" ]; do
  DIR="$(cd -P "$(dirname "$SRC")" && pwd)"
  SRC="$(readlink "$SRC")"
  case "$SRC" in /*) ;; *) SRC="$DIR/$SRC" ;; esac
done
REPO="$(cd -P "$(dirname "$SRC")" && pwd)"
MIN_MAJOR=3
MIN_MINOR=10

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail] %s\033[0m\n' "$*" >&2; exit 1; }

usable() {
  # Not `--version | cut`: a Homebrew shim, a pyenv stub and Windows' Store
  # placeholder all answer that in their own way. Ask the interpreter.
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" \
    >/dev/null 2>&1
}

find_python() {
  local cand
  # Newest first, then whatever `python3` happens to be. The explicit
  # Homebrew paths matter on macOS, where /usr/bin/python3 is 3.9 and the
  # gateway is written in `X | None` syntax throughout -- and where an ssh
  # session's PATH does not include /opt/homebrew/bin.
  for cand in \
      python3.14 python3.13 python3.12 python3.11 python3.10 \
      /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
      /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
      python3 python; do
    if usable "$cand"; then command -v "$cand" 2>/dev/null || echo "$cand"; return 0; fi
  done
  return 1
}

install_python() {
  # One package, from whichever manager this box has. Everything else the
  # stack needs is installed by fleetctl's `packages` step, which knows what
  # the plan asks for; this is only the interpreter that step runs in.
  local sudo_=""
  [ "$(id -u)" -eq 0 ] || sudo_="sudo"
  if   command -v apt-get >/dev/null 2>&1; then
    log "installing python3 with apt"
    $sudo_ apt-get update -qq || warn "apt index refresh failed; trying anyway"
    $sudo_ apt-get install -y --no-install-recommends python3 python3-venv
  elif command -v dnf >/dev/null 2>&1; then
    # Versioned first. On the RHEL-derived distros `python3` is whatever that
    # release froze on, which can be older than the gateway needs.
    log "installing python with dnf"
    for pkg in python3.13 python3.12 python3.11 python3; do
      $sudo_ dnf install -y "$pkg" && break
    done
  elif command -v pacman >/dev/null 2>&1; then
    log "installing python with pacman";  $sudo_ pacman -Sy --needed --noconfirm python
  elif command -v zypper >/dev/null 2>&1; then
    # openSUSE names the interpreter by version, and `python3` is a very old
    # default: on Leap 15.6 it installs Python 3.6, which cannot run any of
    # this. Measured -- the bootstrap installed it, refused it, and said so.
    log "installing python with zypper"
    # A refresh that partly fails is survivable; one that leaves no usable
    # repository is not. The usual cause is a stale metadata cache pointing
    # at files the mirror has since rotated away -- a rolling release does
    # that routinely, and every Tumbleweed container image is a snapshot of
    # one. `clean --all` and one retry is the standard cure, and it is
    # cheap enough to spend before giving up.
    if ! $sudo_ zypper --non-interactive --gpg-auto-import-keys refresh; then
      warn "zypper refresh failed -- clearing the metadata cache and retrying"
      $sudo_ zypper clean --all >/dev/null 2>&1 || true
      $sudo_ zypper --non-interactive --gpg-auto-import-keys refresh || \
        warn "some zypper repositories still will not refresh"
    fi
    for pkg in python313 python312 python311 python3; do
      $sudo_ zypper --non-interactive install -y "$pkg" && break
    done
  elif command -v brew >/dev/null 2>&1; then
    log "installing python@3.12 with brew"; brew install -q python@3.12
  elif [ -x /opt/homebrew/bin/brew ]; then
    log "installing python@3.12 with brew"; /opt/homebrew/bin/brew install -q python@3.12
  else
    die "no package manager found (apt/dnf/pacman/zypper/brew) -- install Python $MIN_MAJOR.$MIN_MINOR+ and re-run"
  fi
}

# ---------------------------------------------------------------------------
[ -d "$REPO/fleetctl" ] || die "no fleetctl/ beside this script -- run it from a checkout"
[ -f "$REPO/gateway/hw.py" ] || die "no gateway/hw.py -- the checkout is incomplete"

PY="$(find_python || true)"
if [ -z "$PY" ]; then
  log "no Python $MIN_MAJOR.$MIN_MINOR+ found"
  install_python
  PY="$(find_python || true)"
  [ -n "$PY" ] || die "still no usable Python after installing one"
fi
log "python: $PY ($("$PY" --version 2>&1))"

# Everything from here is fleetctl's. Split on `--` so an operator can drive
# any subcommand through the same bootstrap on a box that does not have a
# Python yet.
ARGS=()
PASSTHROUGH=0
for arg in "$@"; do
  if [ "$arg" = "--" ] && [ "$PASSTHROUGH" -eq 0 ]; then PASSTHROUGH=1; ARGS=(); continue; fi
  ARGS+=("$arg")
done

cd "$REPO"
if [ "$PASSTHROUGH" -eq 1 ]; then
  # The same empty-array guard as below: macOS's stock /bin/bash is 3.2,
  # where `set -u` and an empty "${ARGS[@]}" is "unbound variable".
  exec "$PY" -m fleetctl "${ARGS[@]+"${ARGS[@]}"}"
fi

# `--dry-run` / `-n` belong to apply. `plan` has no such flag and argparse
# exits 2 on it -- which made `./install.sh --dry-run`, the one command every
# README tells a new user to run first, die before it reached the preview it
# exists to show. install.ps1 has always filtered it; this is the same.
PLAN_ARGS=()
for arg in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$arg" in --dry-run|-n) ;; *) PLAN_ARGS+=("$arg") ;; esac
done

log "detect"
"$PY" -m fleetctl detect

log "plan"
# --write so the plan lands in hosts/<name>/host.yml and can be reviewed,
# edited and committed. A generated plan that only ever lived in memory would
# make the next run's decisions unreviewable.
if ! "$PY" -m fleetctl plan --write "${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"}"; then
  echo >&2
  echo "the plan is incomplete -- see above. The usual cause on a box with no" >&2
  echo "Tailscale is the URL clients should use; supply it and re-run:" >&2
  echo "  ./install.sh --set network.public_api_url=http://<this box's address>:8080/v1" >&2
  exit 2
fi

log "apply"
exec "$PY" -m fleetctl apply "${ARGS[@]+"${ARGS[@]}"}"
