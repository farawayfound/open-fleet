#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# grants.sh -- fleet-wide sudo grants for the gateway's service account, as a
# SECOND, additive sudoers file.
#
# WHY a second file rather than folding this into each host's own
# /etc/sudoers.d/llmstack: every bootstrap script writes that file from its
# own `cat >` heredoc, so re-running bootstrap after this shipped would
# silently drop anything only this script had added. /etc/sudoers.d/
# llmstack-fleet is written ONLY by this script, always rewritten whole, so
# it survives being re-run from a bootstrap AND from deploy-gateway.sh (which
# now calls it on every gateway deploy -- that is how a box that was missing
# this grant entirely, e.g. gpu-laptop-1, gets it fixed without
# waiting for a full re-provision).
#
# Idempotent: safe to run any number of times: the sudoers file is fully
# regenerated each run, and `install` on the helper binary is a plain
# overwrite.
#
# Usage: grants.sh <service-user> [path-to-llmstack-gpuconf]
#   <service-user>              defaults to "llmstack"
#   [path-to-llmstack-gpuconf]  if given, (re)installed to
#                                /opt/llmstack/bin/llmstack-gpuconf, root
#                                owned 0755 -- the sudoers grant below points
#                                at exactly that path, so it must stay
#                                unwritable by the account it grants to.
# ---------------------------------------------------------------------------
set -euo pipefail

SVC_USER="${1:-llmstack}"
GPUCONF_SRC="${2:-}"
GPUCONF_DST=/opt/llmstack/bin/llmstack-gpuconf
SUDOERS_FILE=/etc/sudoers.d/llmstack-fleet

if [[ $EUID -ne 0 ]]; then
  echo "grants.sh must run as root" >&2
  exit 1
fi

if [[ -n "$GPUCONF_SRC" ]]; then
  [[ -f "$GPUCONF_SRC" ]] || { echo "grants.sh: no such file: $GPUCONF_SRC" >&2; exit 1; }
  install -d -m 0755 "$(dirname "$GPUCONF_DST")"
  install -o root -g root -m 0755 "$GPUCONF_SRC" "$GPUCONF_DST"
fi

# The gateway shells out with the .service suffix and matches the full
# argument vector, not a prefix -- every verb below carries it. gpuconf gets
# no argument list at all so it is granted for ANY arguments (auto/N/status);
# it is the one entry here that is a full program, not a single fixed verb.
#
# The nsenter line is how the gateway actually reaches gpuconf: its own unit
# sets ProtectSystem=full, and sudo does not escape a unit's mount namespace,
# so a direct call inherits a read-only /etc and /boot and cannot write the
# bootloader config. `nsenter -t 1 -m` joins PID 1's mount namespace, where
# both are writable. Every argument up to the helper's own is pinned -- this
# grants nsenter for exactly one program, not as a general run-anything-as-root
# verb, which is what an unpinned grant would be.
#
# This replaced `systemd-run --quiet --pipe --wait --collect <helper>`, which
# escaped the namespace correctly on Ubuntu but fails on Fedora: --pipe passes
# the standard file descriptors inside the StartTransientUnit D-Bus message and
# dbus-broker resets the connection. See gpuconf_argv() in gateway/app.py.
cat >"$SUDOERS_FILE" <<EOF
# Written by hosts/linux/grants.sh -- do not hand-edit, re-run that script
# instead. Additive to /etc/sudoers.d/llmstack (each host's bootstrap owns
# that file); this one is rewritten whole by grants.sh alone.
$SVC_USER ALL=(root) NOPASSWD: \\
$GPUCONF_DST, \\
/usr/bin/nsenter -t 1 -m -- $GPUCONF_DST *, \\
/usr/bin/systemctl reboot, \\
/usr/bin/systemctl restart tailscaled.service, \\
/usr/bin/systemctl restart cloudflared.service, \\
/usr/bin/systemctl restart llama-swap.service, \\
/usr/bin/systemctl start llama-swap.service, \\
/usr/bin/systemctl stop llama-swap.service, \\
/usr/bin/systemctl is-active llama-swap.service, \\
/usr/bin/systemctl is-active llm-gateway.service, \\
/usr/bin/systemctl is-active cloudflared.service, \\
/usr/bin/systemctl is-active tailscaled.service, \\
/usr/bin/systemctl status llama-swap.service
EOF
chmod 0440 "$SUDOERS_FILE"

if ! visudo -cf "$SUDOERS_FILE"; then
  echo "grants.sh: $SUDOERS_FILE failed the visudo check -- removing it" >&2
  rm -f "$SUDOERS_FILE"
  exit 1
fi

echo "grants.sh: $SUDOERS_FILE installed for $SVC_USER"
