# observability/

Elasticsearch and Kibana for the fleet's own metrics and logs, on one box, reachable only over the tailnet. The design,
the reasoning, what is collected, the Tailscale ACL, sizing and the honest limits are in
[`../docs/observability.md`](../docs/observability.md). This page is the command reference.

```
docker-compose.yml        es01 + kibana (profile "kibana") + a one-shot certificate setup
.env.example              versions, addresses, sizes. No secrets. Copy to .env or point OBS_ENV at your own file.
kibana.yml.tpl            rendered into .secrets/kibana.yml (token and encryption keys)
setup.sh                  every operation, idempotent
smoke.sh                  stack checks, --stale, --upstream
scripts/lib.sh            shared helpers
scripts/setup-certs.sh    CA, leaf, chain (runs in a container)
scripts/firewall.sh       DOCKER-USER guard for 9200 and 5601
policies/                 the ILM policy
dashboards/               *.ndjson exports, imported by setup.sh dashboards
.secrets/                 generated, mode 0700, gitignored: passwords, token, keys, per-node ingest keys, CA copy
```

## Bring-up

```
cp .env.example .env                  # edit TAILNET_IP and ES_CERT_IPS; or: export OBS_ENV=/path/to/private.env
sudo ./setup.sh docker-order          # Docker waits for tailscale0 at boot; reboot-test it once
sudo ./setup.sh preflight
sudo ./setup.sh all                   # up, packages, views, dashboards, retention, snapshots, viewer, key hub, status
sudo scripts/firewall.sh apply && sudo scripts/firewall.sh install-unit
./smoke.sh
```

`sudo` drops the environment, so with a private env file write `sudo OBS_ENV=/path/to/private.env ./setup.sh <command>`
(an exported `OBS_ENV` does not survive it and the script would look for `.env` next to itself).

Never run `docker compose up` for Kibana by hand on a fresh checkout: if `.secrets/kibana.yml` does not exist yet,
Docker creates a directory of that name. `setup.sh up` renders it first.

## `setup.sh`

| Command | Does |
|---|---|
| `preflight` | commands, address, SANs, image tags vs `STACK_VERSION`, `vm.max_map_count`, memory, disk, free ports |
| `up` | secrets, certificates, Elasticsearch, disk watermarks, Kibana service token, Kibana |
| `packages` | the Kibana `system` package at `SYSTEM_PKG_VERSION` (3.5.0); falls back to the latest and says so |
| `views` | data views `logs-llmstack*`, `logs-llmstack.requests*`, `metrics-system*` |
| `dashboards` | imports `dashboards/*.ndjson` |
| `retention` | ILM policy (rollover 7 d or 30 GB, delete 31 d after rollover) and the `@custom` hooks |
| `snapshots` | local repository and a nightly job, logs and metrics only |
| `viewer` | read-only Kibana user `obs-viewer` |
| `key <name> [days]` | per-node ingest key into `.secrets/agent-output-<name>.env` (default 365 days) |
| `revoke <name> [--all]` | invalidate older keys of that name (keep the newest), or all |
| `kibana-token` | rotate the Kibana service token and recreate Kibana |
| `fingerprint` | the CA's SHA-256, lowercase hex, no colons |
| `status` | cluster, Kibana, package, CA fingerprint and expiry, leaf expiry, API key expiries |
| `docker-order` | systemd drop-in: Docker starts after Tailscale |
| `all` | up, packages, views, dashboards, retention, snapshots, viewer, key hub, status |

`smoke.sh` takes no secrets: it reads `.secrets/` itself. `smoke.sh --stale 10` and `--upstream` are meant for cron;
set `EXPECTED_HOSTS_FILE` (one `host.name` per line) to also catch hosts that never reported.

## Day two

- **Node key:** `sudo ./setup.sh key <node>`, carry `.secrets/agent-output-<node>.env` to the node (see the doc), then
  `fleetctl apply --only observability-agent` there. Retire a node with `revoke <node> --all`.
- **Down:** `docker compose --env-file <env> --profile kibana down` keeps the data. `down -v` destroys the data and the
  CA (every node would need a new fingerprint).
- **Upgrade:** snapshot, bump `STACK_VERSION` and both image tags, `docker compose --env-file <env> pull`,
  `./setup.sh up`, `./setup.sh packages`, agents last. Details in the doc.
- **Renew the leaf certificate early:** `FORCE_LEAF=1 docker compose --env-file <env> run --rm setup`. The CA
  fingerprint does not change.
- **Back up** `.secrets/` like any other credential store. It and the certificate volumes are the only state that
  cannot be rebuilt. Copy snapshots off the box only to an encrypted destination.
