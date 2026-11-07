#!/usr/bin/env bash
# Guard for the published ports. Docker publishes with DNAT in its own chains and bypasses ufw, and a port bound to
# the tailnet address still accepts a packet that arrives on another interface addressed to that address (Linux does
# not tie a local address to the interface it lives on). So: drop connections that were DNAT'd to 9200 or 5601 and
# did not arrive on tailscale0 or loopback.
# Usage: firewall.sh print | apply | check | remove | install-unit   (all but print need root; check exits 1 if a rule is missing)
#
# It is scoped three ways so it cannot touch anything else on a host that forwards other traffic (the hub carries
# another box's uplink): only packets leaving through this stack's own bridge (-o), only connections DNAT'd from
# 9200 or 5601, and only the client-to-server direction (--ctdir ORIGINAL; without it the replies, which leave the
# container on the bridge and arrive on no tailscale0, would be dropped too and nothing would ever connect).
# The conntrack semantics are untested here: validate with the LAN scan in docs/observability.md before trusting it.
set -euo pipefail
BRIDGE=${FW_BRIDGE:-br-fleet-obs}      # fixed by docker-compose.yml (networks.default.driver_opts)
PORTS="9200 5601"
CHAIN=DOCKER-USER
RULE=()

rule_for() {   # sets RULE for one port
  # ONE "-i" only: iptables refuses a second one ("multiple -i flags not allowed"), so "! -i tailscale0 ! -i lo" would
  # make every apply fail. Loopback needs no exemption: traffic a host sends to its own published port is locally
  # generated, goes through OUTPUT and never reaches the FORWARD chain that DOCKER-USER hangs off.
  RULE=(-o "$BRIDGE" -p tcp -m conntrack --ctstate DNAT --ctdir ORIGINAL --ctorigdstport "$1" ! -i tailscale0 -j DROP)
}
need_root() {
  [ "$(id -u)" = 0 ] || { echo "ERROR: $1 needs root" >&2; exit 1; }
}
ipt() { iptables -w "$@"; }
need_chain() {
  ipt -n -L "$CHAIN" >/dev/null 2>&1 || {
    echo "ERROR: no $CHAIN chain: is docker running, and using its iptables backend?" >&2
    exit 1
  }
}

case "${1:-}" in
  print)
    for p in $PORTS; do
      rule_for "$p"
      printf 'iptables -w -I %s 1' "$CHAIN"
      printf ' %q' "${RULE[@]}"
      printf '\n'
    done
    ;;
  apply)
    need_root apply
    need_chain
    for p in $PORTS; do
      rule_for "$p"
      ipt -C "$CHAIN" "${RULE[@]}" 2>/dev/null || ipt -I "$CHAIN" 1 "${RULE[@]}"
    done
    ;;
  check)
    # What smoke.sh runs: the guard is optional to install and invisible when absent, so it is verified, not assumed.
    need_root check
    need_chain
    missing=0
    for p in $PORTS; do
      rule_for "$p"
      if ipt -C "$CHAIN" "${RULE[@]}" 2>/dev/null; then
        echo "present: port $p"
      else
        echo "MISSING: port $p"
        missing=1
      fi
    done
    exit "$missing"
    ;;
  remove)
    need_root remove
    if ipt -n -L "$CHAIN" >/dev/null 2>&1; then
      for p in $PORTS; do
        rule_for "$p"
        while ipt -C "$CHAIN" "${RULE[@]}" 2>/dev/null; do
          ipt -D "$CHAIN" "${RULE[@]}"
        done
      done
    fi
    ;;
  install-unit)
    need_root install-unit
    self=$(readlink -f "$0")
    cat > /etc/systemd/system/obs-docker-user.service <<EOF
[Unit]
Description=Guard the observability ports in $CHAIN
After=docker.service
PartOf=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
Environment=FW_BRIDGE=$BRIDGE
ExecStart="$self" apply
ExecStop="$self" remove
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now obs-docker-user.service
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 2
    ;;
esac
