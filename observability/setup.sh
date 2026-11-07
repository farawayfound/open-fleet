#!/usr/bin/env bash
# Idempotent operations for the observability stack. Run as the same user every time (root recommended: it owns
# .secrets and creates the snapshot directory for uid 1000). Usage: setup.sh <command> [args]
#   preflight | up | packages | views | dashboards | retention | snapshots | viewer
#   key <name> [days] | revoke <name> [--all] | kibana-token | fingerprint | status | docker-order | all
set -euo pipefail
# shellcheck source=scripts/lib.sh
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/scripts/lib.sh"

SNAP=${ES_SNAPSHOT_DIR:-./snapshots}
# The dataset names are the system package's. process_summary is the odd one: its dataset is dotted
# (system.process.summary) while its metricset is not.
DATASETS_SYS="system.cpu system.memory system.load system.network system.diskio system.filesystem system.fsstat system.uptime system.process system.process.summary system.socket_summary"
KB_CHANGED=0

cmd_preflight() {
  local c
  for c in docker openssl python3 curl ss ip; do
    command -v "$c" >/dev/null || die "missing command: $c"
  done
  docker compose version >/dev/null || die "docker compose plugin missing"
  [[ $TAILNET_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "TAILNET_IP is not an IPv4 address"
  [ "$TAILNET_IP" != 0.0.0.0 ] || die "TAILNET_IP must not be 0.0.0.0"
  [ -n "$(ip -4 addr show to "$TAILNET_IP")" ] || die "$TAILNET_IP is not assigned to this host (is tailscale up?)"
  case ",${ES_CERT_IPS:-}," in
    *",$TAILNET_IP,"*) ;;
    *) die "ES_CERT_IPS must include TAILNET_IP" ;;
  esac
  case ",${ES_CERT_IPS:-}," in
    *",127.0.0.1,"*) ;;
    *) die "ES_CERT_IPS must include 127.0.0.1" ;;
  esac
  # Kibana connects to https://es01:9200 and verifies the name, so the certificate must carry it.
  case ",${ES_CERT_DNS:-es01,localhost}," in
    *",es01,"*) ;;
    *) die "ES_CERT_DNS must include es01 (Kibana verifies that hostname)" ;;
  esac
  case "${ES_IMAGE:-}" in
    *":${STACK_VERSION:-unset}" | *":${STACK_VERSION:-unset}@sha256:"*) ;;
    *) die "ES_IMAGE tag does not match STACK_VERSION" ;;
  esac
  case "${KIBANA_IMAGE:-}" in
    *":${STACK_VERSION:-unset}" | *":${STACK_VERSION:-unset}@sha256:"*) ;;
    *) die "KIBANA_IMAGE tag does not match STACK_VERSION" ;;
  esac
  [ "$(cat /proc/sys/vm/max_map_count)" -ge 1048576 ] || die "vm.max_map_count < 1048576 (sysctl -w vm.max_map_count=1048576, and persist it in /etc/sysctl.d)"
  local avail_gb root free_gb min_mem=${OBS_MIN_AVAIL_MEM_GB:-8} min_free=${OBS_MIN_FREE_GB:-120}
  avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  # Only before the first start: once Elasticsearch runs, its own heap is what lowers MemAvailable, and a re-run of
  # `up` or `all` on a healthy box must not fail on it.
  if [ -z "$(dc ps -q es01 2>/dev/null)" ]; then
    [ "$avail_gb" -ge "$min_mem" ] || die "MemAvailable ${avail_gb} GB < ${min_mem} GB"
  fi
  root=$(docker info -f '{{.DockerRootDir}}')
  # No -P: GNU df refuses --output together with it.
  free_gb=$(df -BG --output=avail "$root" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge "$min_free" ] || die "free space on $root is ${free_gb} GB < ${min_free} GB"
  local p
  for p in 9200 5601; do
    if [ -z "$(dc ps -q es01 2>/dev/null)" ] && [ -n "$(ss -H -ltn "sport = :$p")" ]; then
      die "port $p already in use"
    fi
  done
  log "preflight ok (mem ${avail_gb} GB available, ${free_gb} GB free on $root)"
}

init_secrets() {
  install -d -m 0700 "$SEC"
  # install -d leaves the mode of a directory that already exists; this one holds the CA, the passwords and the
  # Kibana token, so it is put right every time.
  chmod 0700 "$SEC"
  if [ ! -s "$SEC/elastic_password" ]; then
    (umask 077; openssl rand -hex 24 > "$SEC/elastic_password")
  fi
  # Compose bind-mounts a file secret as it is, and Elasticsearch refuses an ELASTIC_PASSWORD_FILE whose mode is not
  # one of 400, 440, 600 or 640 (es01 crash-looped on a 444 file on the first bring-up, 2026-11-06). The container
  # runs as uid 1000 in gid 0, and the file is root:root, so 0440 is what lets it read the password and nobody else.
  # Set every time, not only on creation, so a file that was put right by hand stays right.
  chmod 0440 "$SEC/elastic_password"
  if [ ! -s "$SEC/kibana-keys.env" ]; then
    (
      umask 077
      {
        echo "KB_ENC_SECURITY=$(openssl rand -hex 32)"
        echo "KB_ENC_SAVEDOBJECTS=$(openssl rand -hex 32)"
        echo "KB_ENC_REPORTING=$(openssl rand -hex 32)"
      } > "$SEC/kibana-keys.env"
    )
  fi
}

fetch_ca() {
  dc exec -T es01 cat /usr/share/elasticsearch/config/certs/ca.crt > "$SEC/ca.crt.new"
  mv "$SEC/ca.crt.new" "$SEC/ca.crt"
  chmod 0644 "$SEC/ca.crt"
}

# Renders .secrets/kibana.yml from kibana.yml.tpl. The file is bind-mounted, and replacing it (mv) gives it a new
# inode that a running container does not see, so a change sets KB_CHANGED and the caller recreates Kibana.
render_kibana() {
  local tok
  [ ! -d "$SEC/kibana.yml" ] || die "$SEC/kibana.yml is a directory (compose created it before the file existed): remove it and re-run"
  if [ -s "$SEC/kibana-token" ]; then
    tok=$(cat "$SEC/kibana-token")
  else
    es DELETE /_security/service/elastic/kibana/credential/token/fleet-obs >/dev/null 2>&1 || true
    tok=$(es POST /_security/service/elastic/kibana/credential/token/fleet-obs | jget token.value)
    [ -n "$tok" ] || die "could not create the Kibana service token"
    (umask 077; printf '%s' "$tok" > "$SEC/kibana-token")
  fi
  set -a
  # shellcheck disable=SC1091
  . "$SEC/kibana-keys.env"
  set +a
  KB_TOKEN=$tok KIBANA_PUBLIC_URL=$KIBANA_PUBLIC_URL python3 - "$HERE/kibana.yml.tpl" > "$SEC/kibana.yml.new" <<'PY'
import os
import sys

t = open(sys.argv[1]).read()
for k in ("KIBANA_PUBLIC_URL", "KB_TOKEN", "KB_ENC_SECURITY", "KB_ENC_SAVEDOBJECTS", "KB_ENC_REPORTING"):
    t = t.replace("{{%s}}" % k, os.environ[k])
sys.stdout.write(t)
PY
  if [ -f "$SEC/kibana.yml" ] && cmp -s "$SEC/kibana.yml.new" "$SEC/kibana.yml"; then
    rm -f "$SEC/kibana.yml.new"
  else
    chmod 0444 "$SEC/kibana.yml.new"
    mv "$SEC/kibana.yml.new" "$SEC/kibana.yml"
    KB_CHANGED=1
  fi
}

cmd_kibana_token() {   # rotate: drop the token file and credential, re-render, recreate Kibana
  [ -s "$SEC/ca.crt" ] || fetch_ca
  rm -f "$SEC/kibana-token"
  render_kibana
  dc --profile kibana up -d --force-recreate kibana
  wait_for Kibana 90 kb_ok
  log "Kibana restarted with a new service token"
}

cmd_up() {
  cmd_preflight
  init_secrets
  as_root install -d -m 0700 -o 1000 -g 0 "$SNAP"
  dc up -d es01                               # starts the one-shot `setup` first (depends_on)
  wait_for "es01 container files" 24 dc exec -T es01 test -s /usr/share/elasticsearch/config/certs/ca.crt
  fetch_ca
  wait_for Elasticsearch 90 es_ok
  es PUT /_cluster/settings "{\"persistent\":{\"cluster.routing.allocation.disk.watermark.low\":\"${WM_LOW_FREE:-100gb}\",\"cluster.routing.allocation.disk.watermark.high\":\"${WM_HIGH_FREE:-60gb}\",\"cluster.routing.allocation.disk.watermark.flood_stage\":\"${WM_FLOOD_FREE:-30gb}\"}}" >/dev/null
  render_kibana
  if [ "$KB_CHANGED" = 1 ]; then
    dc --profile kibana up -d --force-recreate kibana
  else
    dc --profile kibana up -d kibana
  fi
  wait_for Kibana 90 kb_ok
  log "stack is up. Elasticsearch and Kibana are on $TAILNET_IP and 127.0.0.1"
}

# The system package must exist BEFORE any agent writes: it brings the time-series templates and the host
# dashboards, and a data stream that was auto-created from the built-in template first cannot be switched over.
cmd_packages() {
  local out v st want=${SYSTEM_PKG_VERSION:-3.5.0}
  if out=$(kb_raw POST /api/fleet/setup); then
    split_resp "$out"
    case $RESP_CODE in
      2??) ;;
      *) log "WARN: POST /api/fleet/setup returned HTTP $RESP_CODE; continuing" ;;
    esac
  else
    log "WARN: POST /api/fleet/setup was not reachable; continuing"
  fi
  out=$(kb_raw POST "/api/fleet/epm/packages/system/$want" '{"force":false}') || die "Kibana is not reachable"
  split_resp "$out"
  v=$want
  case $RESP_CODE in
    2??) ;;
    404)
      echo "**********************************************************************" >&2
      echo "WARN: system package $want is not in the registry (HTTP 404)." >&2
      echo "WARN: installing the latest compatible version instead." >&2
      echo "**********************************************************************" >&2
      kb POST /api/fleet/epm/packages/system '{"force":false}' >/dev/null || die "could not install the system package (does the Kibana container have outbound HTTPS to epr.elastic.co?)"
      v=$(kb GET /api/fleet/epm/packages/system | jget item.installationInfo.version) || v=
      [ -n "$v" ] || v=$(kb GET /api/fleet/epm/packages/system | jget item.version) || v=
      echo "WARN: ********* installed system package version: ${v:-unknown} *********" >&2
      echo "WARN: set SYSTEM_PKG_VERSION to it here, and in the fleetctl step, before any node is enabled." >&2
      ;;
    *)
      if printf '%s' "$RESP_BODY" | grep -qi 'already installed'; then
        log "system $want was already installed"
      else
        echo "ERROR: POST /api/fleet/epm/packages/system/$want returned HTTP $RESP_CODE: $(printf '%s' "$RESP_BODY" | head -c 400)" >&2
        die "could not install the system package (does the Kibana container have outbound HTTPS to epr.elastic.co?)"
      fi
      ;;
  esac
  st=$(kb GET "/api/fleet/epm/packages/system${v:+/$v}" | jget item.status) || st=
  [ "$st" = installed ] || die "system package is not installed (version ${v:-?}, status ${st:-unknown})"
  log "system ${v:-?} installed"
}

mkview() {  # id name pattern
  kb POST /api/data_views/data_view "$(python3 -c 'import json, sys
print(json.dumps({"override": True, "data_view": {"id": sys.argv[1], "name": sys.argv[2], "title": sys.argv[3], "timeFieldName": "@timestamp"}}))' "$1" "$2" "$3")" >/dev/null
}
cmd_views() {
  mkview fleet-obs-logs "Fleet logs (llmstack)" "logs-llmstack*"
  mkview fleet-obs-requests "Gateway requests" "logs-llmstack.requests*"
  mkview fleet-obs-metrics "Fleet metrics (system)" "metrics-system*"
  log "data views ready"
}

cmd_dashboards() {
  local f n=0 out
  for f in "$HERE"/dashboards/*.ndjson; do
    [ -e "$f" ] || continue
    out=$(curl -sS --noproxy '*' --max-time 300 -w '\n%{http_code}' -X POST -K <(cred) -H 'kbn-xsrf: true' \
      --form "file=@$f" "http://127.0.0.1:5601/api/saved_objects/_import?overwrite=true") || die "Kibana is not reachable"
    split_resp "$out"
    [ "$RESP_CODE" = 200 ] && [ "$(printf '%s' "$RESP_BODY" | jget success)" = true ] \
      || die "import of $(basename "$f") failed (HTTP $RESP_CODE): $(printf '%s' "$RESP_BODY" | head -c 400)"
    n=$((n + 1))
  done
  log "imported $n dashboard file(s)"
}

# Run AFTER `packages` and BEFORE any agent writes: a backing index only takes new settings when it rolls over.
cmd_retention() {
  local days=${RETENTION_DAYS:-31} body ds
  [[ $days =~ ^[0-9]+$ ]] || die "RETENTION_DAYS must be an integer"
  es PUT /_ilm/policy/fleet-obs "$(sed "s/__DELETE_AFTER__/${days}d/" "$HERE/policies/ilm-fleet-obs.json")" >/dev/null
  body='{"template":{"settings":{"index.lifecycle.name":"fleet-obs","index.number_of_replicas":0}}}'
  # The built-in logs and metrics lifecycle policies roll over and never delete, so without this the disk fills.
  # logs@custom and metrics@custom are the hooks the built-in templates compose for anything that is not a package
  # data stream (logs-llmstack.*). Package templates compose their own <dataset>@custom instead, hence the loop.
  es PUT /_component_template/logs@custom "$body" >/dev/null
  es PUT /_component_template/metrics@custom "$body" >/dev/null
  for ds in $DATASETS_SYS; do
    es PUT "/_component_template/metrics-$ds@custom" "$body" >/dev/null
  done
  log "retention ${days}d after rollover (rollover at 7d or 30gb), replicas 0"
}

# Logs and metrics only, and NO global state: the .security index (API key hashes) is not snapshotted.
cmd_snapshots() {
  es PUT /_snapshot/local_fs '{"type":"fs","settings":{"location":"/usr/share/elasticsearch/snapshots/fleet"}}' >/dev/null
  es PUT /_slm/policy/nightly '{"schedule":"0 30 1 * * ?","name":"<nightly-{now/d}>","repository":"local_fs","config":{"indices":["logs-*","metrics-*"],"include_global_state":false},"retention":{"expire_after":"14d","min_count":3,"max_count":14}}' >/dev/null
  log "nightly snapshots registered (copy them off the box ONLY to an encrypted destination)"
}

cmd_viewer() {   # the read-only daily login for Kibana
  if [ ! -s "$SEC/obs_viewer_password" ]; then
    (umask 077; openssl rand -hex 16 > "$SEC/obs_viewer_password")
  fi
  es POST /_security/user/obs-viewer "$(python3 -c 'import json, sys
print(json.dumps({"password": open(sys.argv[1]).read().strip(), "roles": ["viewer"]}))' "$SEC/obs_viewer_password")" >/dev/null
  log "user obs-viewer ready (password in .secrets/obs_viewer_password)"
}

# key <name> [days]  ->  .secrets/agent-output-<name>.env (0600), which the operator carries to the node.
cmd_key() {
  local name=${1:-} days=${2:-365}
  [[ $name =~ ^[a-z0-9][a-z0-9-]{0,40}$ ]] || die "key name must be lowercase alphanumeric or dash"
  [[ $days =~ ^[1-9][0-9]*$ ]] || die "days must be a positive integer"
  [ -s "$SEC/ca.crt" ] || fetch_ca
  local body resp id key exp_ms exp out
  # Least privilege: it can create documents in, and auto-create, exactly the two datasets the agents write, in the
  # default namespace. monitor is the cluster privilege Elastic's own standalone-agent docs require alongside.
  body=$(python3 -c 'import json, sys
n, d = sys.argv[1], sys.argv[2]
print(json.dumps({"name": "elastic-agent-" + n, "expiration": d + "d",
  "role_descriptors": {"fleet_obs_ingest": {"cluster": ["monitor"],
    "indices": [{"names": ["metrics-system.*-default", "logs-llmstack.*-default"], "privileges": ["auto_configure", "create_doc"]}]}},
  "metadata": {"managed_by": "fleet-obs", "node": n}}))' "$name" "$days")
  resp=$(es POST /_security/api_key "$body")
  id=$(printf '%s' "$resp" | jget id)
  key=$(printf '%s' "$resp" | jget api_key)
  exp_ms=$(printf '%s' "$resp" | jget expiration)
  [ -n "$id" ] && [ -n "$key" ] || die "api key creation failed"
  exp=$(python3 -c 'import datetime, sys
print(datetime.datetime.fromtimestamp(int(sys.argv[1]) / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))' "${exp_ms:-0}")
  out=$SEC/agent-output-$name.env
  # id:api_key, NOT the base64 "encoded" value: the agent's output.api_key does not accept that form.
  (
    umask 077
    printf 'ES_HOSTS=https://%s:9200\nES_API_KEY=%s:%s\nES_CA_FINGERPRINT=%s\nES_KEY_EXPIRES=%s\nES_KEY_NAME=%s\n' \
      "$TAILNET_IP" "$id" "$key" "$(fingerprint)" "$exp" "elastic-agent-$name" > "$out.new"
  )
  mv "$out.new" "$out"
  chmod 0600 "$out"
  log "wrote $out (key expires $exp). Older keys named elastic-agent-$name stay valid until: setup.sh revoke $name"
}

# revoke <name> [--all]: invalidate the older active keys of that name (keep the newest), or all of them.
cmd_revoke() {
  local name=${1:-} mode=${2:-} ids
  [ -n "$name" ] || die "usage: revoke <name> [--all]"
  [[ $name =~ ^[a-z0-9][a-z0-9-]{0,40}$ ]] || die "key name must be lowercase alphanumeric or dash"
  ids=$(es GET "/_security/api_key?name=elastic-agent-$name&active_only=true" | python3 -c '
import sys, json
keys = json.load(sys.stdin).get("api_keys", [])
keys.sort(key=lambda k: k["creation"])
sel = keys if sys.argv[1] == "--all" else keys[:-1]
print(json.dumps([k["id"] for k in sel]))' "$mode")
  if [ "$ids" = "[]" ]; then
    log "nothing to revoke"
    return 0
  fi
  es DELETE /_security/api_key "{\"ids\":$ids}" >/dev/null
  log "revoked $ids"
}

cmd_status() {
  [ -s "$SEC/ca.crt" ] || fetch_ca
  log "cluster: $(es GET /_cluster/health | jget status)"
  log "kibana: $(kb GET /api/status 2>/dev/null | jget status.overall.level || true)"
  log "system pkg: $(kb GET "/api/fleet/epm/packages/system/${SYSTEM_PKG_VERSION:-3.5.0}" 2>/dev/null | jget item.status || true)"
  log "CA fingerprint: $(fingerprint)"
  log "CA $(openssl x509 -in "$SEC/ca.crt" -noout -enddate)"
  log "http leaf $(echo | openssl s_client -connect 127.0.0.1:9200 -CAfile "$SEC/ca.crt" 2>/dev/null | openssl x509 -noout -enddate)"
  es GET '/_security/api_key?active_only=true' | python3 -c '
import sys, json, datetime
for k in json.load(sys.stdin).get("api_keys", []):
    e = k.get("expiration")
    when = datetime.datetime.fromtimestamp(e / 1000, datetime.timezone.utc).date() if e else "never"
    print("api key %-28s expires %s" % (k["name"], when))'
}

# Make Docker wait for the tailnet address, so the published-port bind cannot fail after a reboot.
cmd_docker_order() {
  local wait unit=/etc/systemd/system/docker.service.d/after-tailscale.conf
  # The address is written into a shell line inside a unit file: accept nothing but a dotted quad.
  [[ $TAILNET_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "TAILNET_IP is not an IPv4 address"
  wait="for i in \$(seq 1 30); do ip -4 addr show to $TAILNET_IP | grep -q inet && exit 0; sleep 2; done; exit 0"
  as_root install -d /etc/systemd/system/docker.service.d
  # systemd treats $ as the start of a variable reference in an Exec line; $$ is how a literal one is written.
  printf '[Unit]\nAfter=tailscaled.service\nWants=tailscaled.service\n[Service]\nExecStartPre=/bin/sh -c '"'"'%s'"'"'\n' "${wait//\$/\$\$}" \
    | as_root tee "$unit" >/dev/null
  as_root systemctl daemon-reload
  log "docker.service now waits up to 60 s for $TAILNET_IP ($unit). Reboot-test it."
}

# The guard is a separate, optional step (scripts/firewall.sh). Until it is installed the two ports are reachable by a
# neighbour on the LAN that uses the tailnet address, so say so where the operator is looking.
warn_firewall() {
  if ! as_root "$HERE/scripts/firewall.sh" check >/dev/null 2>&1; then
    echo "**********************************************************************" >&2
    echo "WARNING: the LAN guard for ports 9200 and 5601 is NOT in place (or could not be checked)." >&2
    echo "WARNING: run: sudo $HERE/scripts/firewall.sh apply && sudo $HERE/scripts/firewall.sh install-unit" >&2
    echo "**********************************************************************" >&2
  fi
}

cmd_all() {
  cmd_up
  cmd_packages
  cmd_views
  cmd_dashboards
  cmd_retention
  cmd_snapshots
  cmd_viewer
  # `all` is re-runnable: it mints the hub's own key once, not on every run.
  [ -s "$SEC/agent-output-hub.env" ] || cmd_key hub
  cmd_status
  warn_firewall
}

case "${1:-}" in
  preflight) cmd_preflight ;;
  up) cmd_up; warn_firewall ;;
  packages) cmd_packages ;;
  views) cmd_views ;;
  dashboards) cmd_dashboards ;;
  retention) cmd_retention ;;
  snapshots) cmd_snapshots ;;
  viewer) cmd_viewer ;;
  key) shift; cmd_key "$@" ;;
  revoke) shift; cmd_revoke "$@" ;;
  kibana-token) cmd_kibana_token ;;
  fingerprint) [ -s "$SEC/ca.crt" ] || fetch_ca; fingerprint ;;
  status) cmd_status ;;
  docker-order) cmd_docker_order ;;
  all) cmd_all ;;
  *) sed -n '2,5p' "$0"; exit 2 ;;
esac
