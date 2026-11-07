#!/usr/bin/env bash
# Shared helpers. Sourced by setup.sh and smoke.sh. No secret ever reaches a command line (argv is world-readable in
# /proc): the credential goes to curl through -K <(...) and request bodies go through stdin. --noproxy '*' keeps a
# proxy variable in the environment from receiving either.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
ENV_FILE=${OBS_ENV:-$HERE/.env}
[ -f "$ENV_FILE" ] || { echo "ERROR: missing $ENV_FILE (copy .env.example or export OBS_ENV=<path>)" >&2; exit 2; }
# Absolute, because the script cd's to $HERE below and compose is handed this path later.
ENV_FILE=$(readlink -f "$ENV_FILE")
SEC=$HERE/.secrets
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
: "${TAILNET_IP:?TAILNET_IP is not set in $ENV_FILE}"
KIBANA_PUBLIC_URL=${KIBANA_PUBLIC_URL:-http://$TAILNET_IP:5601}
cd "$HERE"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "== $*"; }
dc() { docker compose --env-file "$ENV_FILE" -f "$HERE/docker-compose.yml" "$@"; }
as_root() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi; }
cred() { printf 'user = "elastic:%s"\n' "$(cat "$SEC/elastic_password")"; }

# es_raw METHOD PATH [JSON]  -> "<body>\n<http code>". Never fails on an HTTP status, only on a transport error.
es_raw() {
  local m=$1 p=$2 d=${3-}
  if [ -n "$d" ]; then
    printf '%s' "$d" | curl -sS --noproxy '*' --max-time 120 -w '\n%{http_code}' -X "$m" --cacert "$SEC/ca.crt" \
      -K <(cred) -H 'Content-Type: application/json' --data-binary @- "https://127.0.0.1:9200$p"
  else
    curl -sS --noproxy '*' --max-time 120 -w '\n%{http_code}' -X "$m" --cacert "$SEC/ca.crt" \
      -K <(cred) -H 'Content-Type: application/json' "https://127.0.0.1:9200$p" </dev/null
  fi
}
# kb_raw METHOD PATH [JSON]  -> same shape. The two headers are what Kibana's API refuses to run without.
kb_raw() {
  local m=$1 p=$2 d=${3-}
  local hdr=(-H 'kbn-xsrf: true' -H 'elastic-api-version: 2023-10-31' -H 'Content-Type: application/json')
  if [ -n "$d" ]; then
    printf '%s' "$d" | curl -sS --noproxy '*' --max-time 300 -w '\n%{http_code}' -X "$m" -K <(cred) \
      "${hdr[@]}" --data-binary @- "http://127.0.0.1:5601$p"
  else
    curl -sS --noproxy '*' --max-time 300 -w '\n%{http_code}' -X "$m" -K <(cred) \
      "${hdr[@]}" "http://127.0.0.1:5601$p" </dev/null
  fi
}
# Sets RESP_BODY and RESP_CODE from an es_raw/kb_raw result. Not for use inside $( ), which would lose them.
split_resp() { RESP_CODE=${1##*$'\n'}; RESP_BODY=${1%$'\n'*}; }

# es / kb METHOD PATH [JSON]: print the body; return 1 (and say why on stderr) on a non-2xx status.
es() {
  local out
  out=$(es_raw "$@") || return 1
  split_resp "$out"
  printf '%s\n' "$RESP_BODY"
  case $RESP_CODE in
    2??) return 0 ;;
  esac
  echo "ERROR: Elasticsearch $1 $2 returned HTTP $RESP_CODE: $(printf '%s' "$RESP_BODY" | head -c 400)" >&2
  return 1
}
kb() {
  local out
  out=$(kb_raw "$@") || return 1
  split_resp "$out"
  printf '%s\n' "$RESP_BODY"
  case $RESP_CODE in
    2??) return 0 ;;
  esac
  echo "ERROR: Kibana $1 $2 returned HTTP $RESP_CODE: $(printf '%s' "$RESP_BODY" | head -c 400)" >&2
  return 1
}
# es_code METHOD PATH -> the HTTP status alone
es_code() {
  curl -s --noproxy '*' --max-time 30 -o /dev/null -w '%{http_code}' --cacert "$SEC/ca.crt" \
    -K <(cred) -X "$1" "https://127.0.0.1:9200$2" </dev/null
}
# jget a.b.0.c  (JSON on stdin; prints a scalar as text, anything else as JSON; empty when absent or not JSON)
jget() {
  python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for k in sys.argv[1].split("."):
    if isinstance(d, list):
        try:
            d = d[int(k)]
        except (ValueError, IndexError):
            d = None
    elif isinstance(d, dict):
        d = d.get(k)
    else:
        d = None
    if d is None:
        break
print("" if d is None else d if isinstance(d, str) else json.dumps(d))' "$1"
}
wait_for() {  # wait_for WHAT TRIES CMD...   (5 s between tries)
  local what=$1 tries=$2 i
  shift 2
  for ((i = 0; i < tries; i++)); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  die "timed out waiting for $what"
}
es_ok() { [ "$(es_code GET '/_cluster/health?wait_for_status=yellow&timeout=5s')" = 200 ]; }
kb_ok() { [ "$(kb GET /api/status 2>/dev/null | jget status.overall.level)" = available ]; }
# The CA's SHA-256, lowercase hex with no colons: the exact form ssl.ca_trusted_fingerprint wants (Go hex-decodes it).
fingerprint() {
  openssl x509 -in "$SEC/ca.crt" -noout -fingerprint -sha256 | cut -d= -f2 | tr -d ':' | tr 'A-F' 'a-f'
}
