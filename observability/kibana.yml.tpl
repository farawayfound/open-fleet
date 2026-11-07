# Rendered by setup.sh into .secrets/kibana.yml (mounted read-only). {{PLACEHOLDERS}} are replaced; nothing here is
# a secret until it is rendered, which is why the rendered file lives in the 0700 .secrets directory and not in
# container env (env values show in `docker inspect`).
server.host: "0.0.0.0"          # inside the container only; compose publishes to specific host IPs
server.name: "kibana"
server.shutdownTimeout: "5s"
server.publicBaseUrl: "{{KIBANA_PUBLIC_URL}}"
elasticsearch.hosts: ["https://es01:9200"]
elasticsearch.ssl.certificateAuthorities: ["/usr/share/kibana/config/certs/ca.crt"]
elasticsearch.serviceAccountToken: "{{KB_TOKEN}}"
xpack.security.encryptionKey: "{{KB_ENC_SECURITY}}"
xpack.encryptedSavedObjects.encryptionKey: "{{KB_ENC_SAVEDOBJECTS}}"
xpack.reporting.encryptionKey: "{{KB_ENC_REPORTING}}"
telemetry.optIn: false
telemetry.allowChangingOptInStatus: false
# Deliberately NOT set: xpack.fleet.* . Fleet's package manager (EPM) must stay on, it is how the `system`
# package's templates and dashboards get installed (setup.sh packages). It reads epr.elastic.co, so this
# container needs outbound HTTPS. No Fleet Server is involved: agents run standalone.
