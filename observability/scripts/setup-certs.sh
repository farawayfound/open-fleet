#!/usr/bin/env bash
# Runs inside the elasticsearch image as root, once per `up`. Idempotent.
# The CA is created once (10 years). The leaf (3 years) is re-issued when ES_CERT_IPS or ES_CERT_DNS change, when
# FORCE_LEAF=1, and when it is within 30 days of expiry. Re-issuing the leaf under the same CA keeps the CA
# fingerprint, so the nodes pinned to it need no change.
set -euo pipefail
BIN=/usr/share/elasticsearch/bin/elasticsearch-certutil
CAP=/ca-private
ESD=/certs-es
KBD=/certs-kb
umask 077
mkdir -p "$CAP" "$ESD" "$KBD"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if [ ! -s "$CAP/ca.key" ] || [ ! -s "$CAP/ca.crt" ]; then
  "$BIN" ca --silent --pem --days 3650 --out "$work/ca.zip"
  unzip -q -o "$work/ca.zip" -d "$work"                       # -> $work/ca/ca.crt, ca.key
  install -m 600 "$work/ca/ca.key" "$CAP/ca.key"
  install -m 644 "$work/ca/ca.crt" "$CAP/ca.crt"
  rm -f "$CAP/leaf.san"
fi

want="ips=${ES_CERT_IPS};dns=${ES_CERT_DNS}"
have=$(cat "$CAP/leaf.san" 2>/dev/null || true)
renew=0
if [ -s "$ESD/es01-chain.crt" ] && command -v openssl >/dev/null; then
  # openssl x509 reads the first certificate in the chain file, which is the leaf.
  openssl x509 -in "$ESD/es01-chain.crt" -noout -checkend $((30 * 86400)) >/dev/null || renew=1
fi
if [ ! -s "$ESD/es01.key" ] || [ ! -s "$ESD/es01-chain.crt" ] || [ "$want" != "$have" ] \
  || [ "${FORCE_LEAF:-0}" = 1 ] || [ "$renew" = 1 ]; then
  "$BIN" cert --silent --pem --days 1095 --name es01 --dns "$ES_CERT_DNS" --ip "$ES_CERT_IPS" \
    --ca-cert "$CAP/ca.crt" --ca-key "$CAP/ca.key" --out "$work/es01.zip"
  unzip -q -o "$work/es01.zip" -d "$work"                     # -> $work/es01/es01.crt, es01.key
  # The chain (leaf, then CA) is what lets ssl.ca_trusted_fingerprint work on the agents: they pin the CA, and
  # certutil's own output holds the leaf alone, so without this the server never sends the CA to pin.
  cat "$work/es01/es01.crt" "$CAP/ca.crt" > "$work/es01-chain.crt"
  install -m 640 -o 0 -g 0 "$work/es01/es01.key" "$ESD/es01.key"
  install -m 640 -o 0 -g 0 "$work/es01-chain.crt" "$ESD/es01-chain.crt"
  printf '%s' "$want" > "$CAP/leaf.san"
fi

install -m 644 "$CAP/ca.crt" "$ESD/ca.crt"
install -m 644 "$CAP/ca.crt" "$KBD/ca.crt"
# Elasticsearch and Kibana run as uid 1000, gid 0: group read on a root:root 0640 file is how they get the key.
chown 0:0 "$ESD" "$KBD"
chmod 750 "$ESD"
chmod 755 "$KBD"
chmod 700 "$CAP"
echo "certs ready ($want)"
