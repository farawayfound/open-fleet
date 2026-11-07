#!/usr/bin/env bash
# Hub smoke test. Usage: smoke.sh [--require-data] | --stale MINUTES | --upstream
#  default          stack checks, then document counts in metrics-system.* and logs-llmstack.* (the exit code follows
#                   the stack checks; --require-data makes an empty dataset a failure too)
#  --stale N        list hosts with no system.cpu document for N minutes (and hosts missing from $EXPECTED_HOSTS_FILE)
#  --upstream       hosts whose gateway /health reported upstream != ok in the last 10 minutes, and hosts with no health docs
set -uo pipefail
# shellcheck source=scripts/lib.sh
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/scripts/lib.sh"
set +e
fail=0
require_data=0
check() {
  local l=$1
  shift
  if "$@" >/dev/null 2>&1; then
    echo "PASS $l"
  else
    echo "FAIL $l"
    fail=1
  fi
}
count() { es POST "/$1/_count" "{\"query\":{\"range\":{\"@timestamp\":{\"gte\":\"now-${2}m\"}}}}" 2>/dev/null | jget count; }
code_of() { curl -s --noproxy '*' --max-time 15 -o /dev/null -w '%{http_code}' "$@"; }
kibana_up() {
  local c
  c=$(code_of "http://$TAILNET_IP:5601/")
  [ "$c" = 200 ] || [ "$c" = 302 ] || [ "$c" = 401 ]
}
# The fingerprint of the LAST certificate the server sends. The agents pin the CA, which only works when the
# server's handshake carries it after the leaf. Older OpenSSL prints the leaf PEM again after a "Server certificate"
# line, which would otherwise be taken for the last one, so the scan stops there.
chain_fp() {
  echo | timeout 15 openssl s_client -connect "$TAILNET_IP:9200" -showcerts 2>/dev/null \
    | awk '/^Server certificate/{exit} /BEGIN CERT/{b=""} {b=b $0 "\n"} /END CERT/{last=b} END{printf "%s", last}' \
    | openssl x509 -noout -fingerprint -sha256 | cut -d= -f2 | tr -d ':' | tr 'A-F' 'a-f'
}

stale() {
  es POST '/metrics-system.cpu-*/_search' '{"size":0,"query":{"range":{"@timestamp":{"gte":"now-7d"}}},"aggs":{"h":{"terms":{"field":"host.name","size":500},"aggs":{"last":{"max":{"field":"@timestamp"}}}}}}' \
    | python3 -c '
import sys, json, os, datetime
mins = int(sys.argv[1])
exp = sys.argv[2]
d = json.load(sys.stdin)
if "aggregations" not in d:
    print("ERROR: no aggregation in the response (no metrics-system.cpu data yet?)")
    sys.exit(2)
now = datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
seen = {b["key"]: b["last"]["value"] for b in d["aggregations"]["h"]["buckets"]}
bad = sorted(h for h, t in seen.items() if now - t > mins * 60000)
if exp and os.path.exists(exp):
    bad += sorted(h for h in open(exp).read().split() if h and h not in seen)
for h in bad:
    print("STALE", h)
sys.exit(1 if bad else 0)' "$1" "${EXPECTED_HOSTS_FILE:-}"
}
upstream() {
  es POST '/logs-llmstack.health-*/_search' '{"size":0,"query":{"range":{"@timestamp":{"gte":"now-10m"}}},"aggs":{"h":{"terms":{"field":"host.name","size":500},"aggs":{"bad":{"filter":{"bool":{"must_not":[{"match":{"llmstack.health.upstream":"ok"}}]}}}}}}}' \
    | python3 -c '
import sys, json, os
d = json.load(sys.stdin)
if "aggregations" not in d:
    print("ERROR: no aggregation in the response (no logs-llmstack.health data yet?)")
    sys.exit(2)
seen = {x["key"]: x["bad"]["doc_count"] for x in d["aggregations"]["h"]["buckets"]}
exp = sys.argv[1]
bad = [h for h, n in seen.items() if n]
miss = [h for h in open(exp).read().split() if h and h not in seen] if exp and os.path.exists(exp) else []
for h in sorted(bad):
    print("UPSTREAM-NOT-OK", h)
for h in sorted(miss):
    print("NO-HEALTH-DATA", h)
sys.exit(1 if bad or miss else 0)' "${EXPECTED_HOSTS_FILE:-}"
}

case "${1:-}" in
  --stale)
    stale "${2:-10}"
    exit $?
    ;;
  --upstream)
    upstream
    exit $?
    ;;
  --require-data) require_data=1 ;;
  "") ;;
  *)
    sed -n '2,6p' "$0"
    exit 2
    ;;
esac

check "ES answers over TLS with credentials" test "$(es_code GET /)" = 200
check "ES rejects anonymous requests (401)" test "$(code_of --cacert "$SEC/ca.crt" https://127.0.0.1:9200/)" = 401
check "ES reachable on the tailnet IP (SAN verified)" test "$(code_of --cacert "$SEC/ca.crt" "https://$TAILNET_IP:9200/")" = 401
check "ES sends the CA in its chain (fingerprint pin works)" test "$(chain_fp)" = "$(fingerprint)"
check "Kibana status is available" kb_ok
check "Kibana reachable on the tailnet IP" kibana_up
# Without the guard, a neighbour on the LAN can reach both ports using the tailnet address. It is optional to install,
# so a host that skipped it must not pass.
check "LAN guard rules present in DOCKER-USER (scripts/firewall.sh apply)" as_root "$HERE/scripts/firewall.sh" check
check "system package installed" test "$(kb GET "/api/fleet/epm/packages/system/${SYSTEM_PKG_VERSION:-3.5.0}" 2>/dev/null | jget item.status)" = installed
echo "cluster health: $(es GET /_cluster/health 2>/dev/null | jget status)"
# Data checks. An agent that has not run yet leaves these at zero, which is not a stack fault.
for ix in 'metrics-system.cpu-*' 'metrics-system.memory-*' 'metrics-system.diskio-*' 'logs-llmstack.health-*' 'logs-llmstack.journal-*'; do
  n=$(count "$ix" 5)
  echo "docs last 5m in $ix: ${n:-0}"
  if [ "${n:-0}" -lt 1 ]; then
    if [ "$require_data" = 1 ]; then
      fail=1
    else
      echo "WARN no documents in $ix (fine before the first agent runs)"
    fi
  fi
done
# The request log is opt-in on the gateway (LLMSTACK_ACCESS_LOG), and quiet when nobody is calling it: informational.
echo "docs last 60m in logs-llmstack.requests-* (informational): $(count 'logs-llmstack.requests-*' 60)"
echo "distinct hosts last 5m: $(es POST '/metrics-system.cpu-*/_search' '{"size":0,"query":{"range":{"@timestamp":{"gte":"now-5m"}}},"aggs":{"c":{"cardinality":{"field":"host.name"}}}}' 2>/dev/null | jget aggregations.c.value)"
em=$(es POST '/metrics-system.cpu-*/_count' '{"query":{"bool":{"filter":[{"term":{"event.module":"system"}},{"range":{"@timestamp":{"gte":"now-5m"}}}]}}}' 2>/dev/null | jget count)
if [ "${em:-0}" -ge 1 ]; then
  echo "PASS event.module: system present (Hosts view entity)"
elif [ "$require_data" = 1 ]; then
  echo "FAIL event.module: system present (Hosts view entity)"
  fail=1
else
  echo "WARN event.module: system not seen yet (fine before the first agent runs)"
fi
exit $fail
