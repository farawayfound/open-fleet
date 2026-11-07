# Dashboards

Saved-object exports (`*.ndjson`) placed in this directory are imported by `setup.sh dashboards` (and by
`setup.sh all`) with `overwrite=true`, so the import can be re-run safely.

The Kibana `system` package (installed by `setup.sh packages`) already ships the host dashboards, and the
Observability > Infrastructure > Hosts view reads the `metrics-system.*` data streams directly. Add a file here only
for something the package does not have, for example a view of the gateway request log (`logs-llmstack.requests`)
or the health documents (`logs-llmstack.health`).

Export one from Kibana (Stack Management > Saved Objects, tick the dashboard, "include related objects"), or with the
API:

    curl -sS -K <(printf 'user = "elastic:%s"\n' "$(cat .secrets/elastic_password)") \
      -H 'kbn-xsrf: true' -H 'Content-Type: application/json' -X POST http://127.0.0.1:5601/api/saved_objects/_export \
      -d '{"type":["dashboard"],"includeReferencesDeep":true}' > dashboards/<name>.ndjson

Keep the exports in git: the nightly snapshots deliberately exclude Kibana's own state, so this directory is the
backup of the dashboards.
