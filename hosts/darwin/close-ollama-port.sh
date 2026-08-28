#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bind a Mac's Ollama to loopback, so the keyed gateway on :8080 is the only
# way in from off-box. Run ONCE on that Mac, as root:
#
#   sudo bash hosts/darwin/close-ollama-port.sh [this-box-address]
#
# Ollama installed from the .app runs as a root-owned system LaunchDaemon and
# listens on 0.0.0.0:11434 with no authentication -- anything that can route to
# the box can load models, run them, and read every prompt. On a fleet whose
# whole premise is one keyed door, that is a second door with no lock on it.
#
# Why this is not part of the installer: the daemon is root-owned and a Mac in
# a fleet usually has no passwordless sudo, so nothing automated can reach it.
# Everything else fleetctl does on a Mac is user-level for exactly that reason.
#
# Undo (if something still needs bare Ollama over the network):
#   sudo /usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:OLLAMA_HOST 0.0.0.0" \
#     /Library/LaunchDaemons/com.ollama.server.plist
#   sudo launchctl kickstart -k system/com.ollama.server
# ---------------------------------------------------------------------------
set -euo pipefail
PLIST=/Library/LaunchDaemons/com.ollama.server.plist
ADDR="${1:-this-box}"

[[ "$(id -u)" -eq 0 ]] || { echo "run with sudo: sudo bash $0" >&2; exit 1; }
[[ -f "$PLIST" ]] || { echo "$PLIST not found -- is Ollama installed as a system daemon?" >&2; exit 1; }

cp -p "$PLIST" "${PLIST}.bak-$(date +%Y%m%d)"
/usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:OLLAMA_HOST 127.0.0.1" "$PLIST"

# kickstart -k restarts the daemon, which drops whatever model is resident. The
# next request reloads it (a few seconds) and every consumer fails over to
# another endpoint meanwhile, so this is safe to run at any time.
launchctl kickstart -k system/com.ollama.server
sleep 3

echo "== listeners on 11434 =="
netstat -an -p tcp | grep 11434 | grep LISTEN || echo "(none yet -- give it a second)"
echo
echo "Expect '127.0.0.1.11434' above, NOT '*.11434'."
echo "Then from another box: curl -m5 http://$ADDR:11434/api/tags   -> must fail"
echo "                       curl -m5 -H 'Authorization: Bearer sk-...' \\"
echo "                            http://$ADDR:8080/api/tags        -> must work"
