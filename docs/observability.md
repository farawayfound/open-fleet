# Observability

Metrics and logs from every box in the fleet, in one Elasticsearch that you own, read through Kibana. The pieces:

- **The hub stack** (`observability/`): Elasticsearch and Kibana in Docker on one box, on the tailnet only.
- **The node agent** (`fleetctl` step `observability-agent`): a standalone Elastic Agent on each Linux box, with a
  policy file that fleetctl renders, so `check`, `apply --dry-run` and drift detection work like they do for every
  other step.

This page is the design and the operator runbook. `observability/README.md` is the short version of the commands.

## Why this shape

| Choice | Reason |
|---|---|
| Standalone agents, no Fleet Server | Nothing to enrol and no enrolment credential on every box. The agent policy is a file fleetctl renders, so what runs on a node is exactly what `fleetctl check` compares. |
| One API key per node | Revoking one box does not touch the others. Each key can only create documents in the two datasets the agents write, in the `default` namespace. |
| TLS with a pinned CA fingerprint | The agent trusts one fingerprint instead of a CA bundle. That only works if the server sends the CA in its handshake, so the certificate file the hub serves is the leaf followed by the CA. |
| Kibana on plain HTTP | It is bound to the tailnet address and loopback, never `0.0.0.0` and never a public tunnel. Day-to-day login is a read-only user, so the superuser password does not cross the wire. |
| The Kibana `system` package installed first | It ships the time-series templates and the host dashboards the Infrastructure > Hosts view reads. If an agent writes first, the data streams are created from Elasticsearch's generic template and the package cannot take them over cleanly. |

```
  node (Linux)                                     hub
  +-----------------------------+                  +------------------------------------------+
  | gateway --> access log file |                  |  Elasticsearch  <tailnet>:9200  and :9200 |
  | gateway /health, /admin/api |--+               |  Kibana         <tailnet>:5601  and :5601 |
  | journald (listed units)     |  |  HTTPS, API   |   (loopback copies for the operator)      |
  | host metrics (system)       |  +-------------> |  TLS on HTTP and transport, one CA        |
  |                             |     key, pinned  +------------------------------------------+
  |  elastic-agent (standalone) |     CA print              ^ setup.sh: keys, package, retention
  +-----------------------------+                           | smoke.sh, firewall.sh
```

## What is collected

| Source | Dataset (data stream `<type>-<dataset>-default`) | Notes |
|---|---|---|
| Host metrics | `metrics-system.cpu`, `.memory`, `.load`, `.network`, `.diskio`, `.filesystem`, `.fsstat`, `.uptime`, `.process`, `.process.summary`, `.socket_summary` | Exactly what the Hosts view reads. Process command lines are dropped on the node (engines take secrets in argv). |
| journald | `logs-llmstack.journal` | The units listed in the `observability:` block, Linux with systemd only. Lines that look like they carry credentials are dropped whole. |
| Gateway state | `logs-llmstack.health`, `.status`, `.gpu`, `.usage`, plus on the hub `.fleet_summary`, `.fleet_overview`, `.peers` | Polled over loopback with httpjson. The gateway's `/health` always answers HTTP 200: the state is the `upstream` field (`ok`, `degraded`, `down`), not the status code. |
| Gateway requests | `logs-llmstack.requests` | One JSON line per metered request. Opt-in on the gateway: it writes the file only when `LLMSTACK_ACCESS_LOG=<path>` is set, and fleetctl sets that in the gateway env file when observability is enabled for the host. The agent tails the file with a `filestream` input. Each line carries the key's id and display name, never the key, and no prompt text. |

Not collected in v1: Fleet-managed policies, APM, uptime checks, Elastic Agent self-monitoring, alert rules, container
or Kubernetes data, cloud metadata, and any Windows or macOS box (the step is blocked for them, not skipped quietly).
GPU numbers are in `logs-llmstack.gpu` only: they show in Discover and custom dashboards, not in the Hosts view.

## Sizing

The hub gives Elasticsearch a 2 GB heap in a 4 GB container and Kibana a 1.5 GB container; the preflight wants 8 GB
of available memory. That is a small cluster: expect Kibana to be slow on a wide time range, and watch
`docker stats` after the first week. The numbers are in `.env.example`.

Volume is an estimate, not a measurement. At 30 s system metrics and 60 s process data a node writes on the order of
100 to 200 documents a minute, so twenty nodes are a few million documents a day, probably 1 to 3 GB a day. Measure
with `GET _cat/indices?v&h=index,store.size` after 24 hours and adjust the retention.

## Bring-up on the hub

Everything is `observability/setup.sh`, run as root (it owns `.secrets/` and creates the snapshot directory for the
container's user). Every command can be run again.

1. **Env file.** Copy `observability/.env.example` to `observability/.env`, or keep a private copy anywhere and point
   `OBS_ENV` at it (under `sudo` write `sudo OBS_ENV=<path> observability/setup.sh ...`, since `sudo` drops the
   variable). It holds no secrets: the tailnet address, the certificate SANs, image tags and sizes. The stack
   version is `STACK_VERSION` in that file (9.5.6); the two image tags must carry it, and the preflight checks they do.
2. **Docker after Tailscale.** `sudo observability/setup.sh docker-order` installs a systemd drop-in so
   `docker.service` waits for the tailnet address. Without it a reboot can start Docker before the address exists and
   the published-port bind fails. Reboot-test it once; the wait is bounded (60 s), so a very slow tailnet login can
   still lose the race, and `setup.sh up` fixes that.
3. **Preflight.** `sudo observability/setup.sh preflight` checks the commands, the address, the certificate SANs, the
   image tags, `vm.max_map_count`, free memory and free disk, and that nothing else holds 9200 or 5601.
4. **Everything else.** `sudo observability/setup.sh all` runs, in order: `up` (certificates, Elasticsearch, a Kibana
   service-account token, Kibana), `packages` (the `system` package, pinned to 3.5.0), `views` (data views),
   `dashboards` (imports `dashboards/*.ndjson`), `retention`, `snapshots`, `viewer` (the read-only login) and `key hub`
   (the hub's own ingest key), then `status`. Retention runs after the package and before any agent writes, because a
   backing index only takes new settings when it rolls over.
5. **Network.** Apply the Tailscale ACL below, then the host guard (next section).
6. **Check.** `observability/smoke.sh`. Document counts are zero until an agent runs; those print as warnings and do
   not fail the run unless you pass `--require-data`.
7. **Log in.** Open `http://<tailnet address>:5601` as `obs-viewer`; the password is in
   `observability/.secrets/obs_viewer_password`. The `elastic` password is for `setup.sh` only.

If the `system` package install fails with a 404 for 3.5.0, `packages` installs the latest compatible version and
says so loudly; set `SYSTEM_PKG_VERSION` here and in the fleetctl step to match before enabling any node. Kibana needs
outbound HTTPS to `epr.elastic.co` for this.

### Tailscale ACL

Elasticsearch is reachable by the nodes and the operator, Kibana by the operator only. Tags are placeholders:

```json
{ "acls": [
  { "action": "accept", "src": ["tag:fleet", "tag:operator"], "dst": ["tag:obs-hub:9200"] },
  { "action": "accept", "src": ["tag:operator"],              "dst": ["tag:obs-hub:5601"] }
] }
```

The ACL is documentation in this repository: it lives in your tailnet's policy file, not here, and nothing in this
repository applies it.

### Host guard (`scripts/firewall.sh`)

Docker publishes ports with its own NAT rules, which bypass `ufw`, and a service bound to the tailnet address still
accepts a packet that arrives on another interface addressed to that address. `scripts/firewall.sh` closes that:

- `print` shows the rules, `apply` installs them, `remove` deletes them, `install-unit` makes a systemd unit that
  applies them after Docker starts. All of them can be run repeatedly. `check` exits 1 if a rule is missing; `smoke.sh`
  runs it and fails without the guard, and `setup.sh up` and `all` print a warning until it is installed. After a
  reboot the containers can restart a second or two before the unit applies the rules; that window is accepted.
- It drops connections that were translated to port 9200 or 5601, in the client-to-server direction, leaving through
  this stack's own bridge (`br-fleet-obs`, fixed in `docker-compose.yml`), and that did not arrive on `tailscale0` or
  loopback (loopback needs no rule of its own: a host's traffic to its own published port never reaches the
  forwarding path the rule sits on). A container on another Docker network that calls these ports through the
  host address is dropped too; use the `fleet-obs` network for that. It changes nothing else: other forwarded traffic on the host, including a box using this one as its
  uplink, never matches the rule.
- It needs Docker's iptables backend (the `DOCKER-USER` chain). The conntrack semantics are untested here: scan ports
  9200 and 5601 on the host's LAN address from a LAN neighbour and from a tailnet device the ACL does not allow, and
  confirm both fail, before relying on it. IPv6 is not published and not covered.

## Adding a node

1. On the hub: `sudo observability/setup.sh key <node>` writes `observability/.secrets/agent-output-<node>.env` (mode
   0600): `ES_HOSTS`, `ES_API_KEY` (the `id:secret` form, not the base64 `encoded` value), `ES_CA_FINGERPRINT` (the CA's
   SHA-256, lowercase hex, no colons), `ES_KEY_EXPIRES` and `ES_KEY_NAME`. The key lasts 365 days and can create
   documents only in `metrics-system.*-default` and `logs-llmstack.*-default`.
2. Carry it to the node without a temporary copy:

   ```
   ssh <node> 'sudo install -d -m 0700 -o root -g root /etc/fleet-obs && sudo install -m 0600 -o root -g root /dev/stdin /etc/fleet-obs/agent-output.env' < observability/.secrets/agent-output-<node>.env
   ```
3. Enable `observability:` for that host in the fleet config, then `sudo python -m fleetctl apply --dry-run --only
   observability-agent`, read the plan, and apply it. See `docs/fleetctl.md` for the block and its keys.
4. Verify after about five minutes: `smoke.sh --stale 10` lists the node as stale if no `system.cpu` document arrived,
   and Observability > Infrastructure > Hosts shows it.

Add nodes one at a time and leave at least half an hour between them. After the whole fleet is on, run
`smoke.sh --stale 10` and `smoke.sh --upstream` from cron. Set `EXPECTED_HOSTS_FILE` to a file with one `host.name` per
line and both also report the hosts that never sent anything. Agent self-monitoring is off, so these two checks and the
key-expiry output of `setup.sh status` are the only signals that an agent has stopped.

Keys minted together expire together. Stagger them (`key <node> <days>`), or re-mint before the date in
`ES_KEY_EXPIRES`. A re-minted key does not retire the old one: `setup.sh revoke <node>` invalidates every older active
key of that name and keeps the newest; `revoke <node> --all` removes them all (the rollback for one node).

## Retention

`setup.sh retention` installs the ILM policy `fleet-obs`: roll over at 7 days or 30 GB per primary shard, delete 31
days after rollover (`RETENTION_DAYS`). `min_age` counts from rollover, not from document timestamps, so a document
is deleted between 31 and 38 days after it was written. The built-in logs and metrics policies roll over and never
delete, so this is what keeps the disk from filling.

The policy is attached through `@custom` component templates: `logs@custom` and `metrics@custom` for everything the
built-in templates cover (including `logs-llmstack.*`), and one `metrics-<dataset>@custom` per system dataset, because
the package's templates compose their own hook. Replicas are 0 (one node). `setup.sh retention` overwrites those templates on
every run; keep local edits elsewhere.

The disk watermarks are absolute free-space values (`WM_*` in the env file) and apply to the whole filesystem, not just
Elasticsearch's data. At the flood stage Elasticsearch blocks writes, so size them for the disk the volume is on.

## Backup, upgrade, down

**Backup.**
- A nightly snapshot job covers `logs-*` and `metrics-*` with no global state, so the security index (API key hashes)
  and Kibana's saved objects are not in it. Dashboards live in `observability/dashboards/*.ndjson` in git.
- The snapshot directory is on the same disk as the data. Copy it off the box only to an encrypted destination.
- The only irreplaceable state is `observability/.secrets/` and the certificate volumes. A lost CA key means a new
  fingerprint on every node. Back `.secrets/` up the way you back up other credentials.
- The leaf certificate lasts 3 years and is re-issued on `setup.sh up` when it is within 30 days of expiry, or by hand
  with `FORCE_LEAF=1 docker compose --env-file <env> run --rm setup`. The CA fingerprint does not change.

**Upgrade.**
1. `POST /_snapshot/local_fs/pre-upgrade-<date>?wait_for_completion=true`.
2. Bump `STACK_VERSION`, `ES_IMAGE` and `KIBANA_IMAGE` together, then `docker compose --env-file <env> pull`.
3. `setup.sh up` (Elasticsearch first, then Kibana).
4. `setup.sh packages`. Bump `SYSTEM_PKG_VERSION` only after checking the package supports the Kibana version.
5. Agents last, one at a time, never newer than the stack. A standalone agent does not pick up new integration
   versions on its own, so after a package bump the step's version constant and the rendered policy change through a
   normal pull request and `fleetctl apply`.

**Down.**
- `docker compose --env-file <env> --profile kibana down` stops the stack and keeps every volume.
- `down -v` destroys the data and the certificates, including the CA, so every node needs a new fingerprint.
- A reboot is safe: the containers are `restart: unless-stopped` and the `docker-order` drop-in is installed.
- To remove the guard: `scripts/firewall.sh remove`, `systemctl disable --now obs-docker-user.service`, and delete the
  `docker.service.d/after-tailscale.conf` drop-in.

## Honest limits

- **Not measured.** The memory and disk figures, the agent's footprint on a node, the `@custom` hook behaviour on
  package-managed data streams, the `DOCKER-USER` rule semantics, and the `kbn-xsrf` requirement on the package
  install endpoint are from documentation and source reading. Run the first bring-up on a throwaway machine, and keep
  the verification steps above.
- **A node's key can write any `host.name`** within its datasets, so a compromised node can forge another node's
  metrics. Per-node keys limit the spread; they do not prevent it.
- **The gateway admin token is a full admin credential** and there is no read-only one. The agent reads it from a
  root-only file on the node to poll the status endpoint.
- **Gateway responses flow from a lower-privilege account into a root agent.** httpjson has no body-size cap, so a
  compromised gateway could return a very large body. Request timeouts and the retry cap bound the damage. Keep
  gateway, engine and tunnel logs at a non-debug level: debug lines can carry prompt text, and the journald filter
  catches only credential-shaped lines.
- **If Elasticsearch is down, agents drop events** once their bounded in-memory queue fills, and a restart loses the
  queue. They do not queue and retry for long. Run the outage test once (stop the container for ten minutes, restart,
  confirm agents recover).
- **Clocks matter.** Time-series metrics with a badly skewed timestamp are rejected. Keep NTP healthy on every node.
- **Kibana is plain HTTP.** Its protection is the tailnet ACL and the host guard, not the transport. The generated
  `kibana.yml` is world-readable inside the 0700 `.secrets/` directory, so that directory must stay root-only.
- **The system metrics here are 30 s resolution** (60 s for process data). If the Hosts charts look sparse, lower the
  period in the `observability:` block.
- **9.6 is not out.** The whole stack is pinned to 9.5.6. When Elastic publishes 9.6.0 images and agent archives,
  switch in this order: new agent checksums in the fleet config, snapshot, bump the three image variables, `up`,
  `packages`, then agents one at a time.
