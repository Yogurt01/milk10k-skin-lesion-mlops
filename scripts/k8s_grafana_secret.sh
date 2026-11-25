#!/usr/bin/env bash
# Grafana admin login for the monitoring release on the local cluster (`make k8s-grafana-secret`). It
#   1. creates k8s/secrets/grafana-admin-password once (random, mode 600, gitignored); an existing file is kept,
#   2. applies k8s/monitoring/namespace.yaml,
#   3. applies the Secret grafana-admin (keys admin-user, admin-password) in that namespace. The Secret is built by
#      kubectl from the file and piped straight into `kubectl apply`: the password is never printed and never on a
#      command line.
# The Helm values only name the Secret (grafana.admin.existingSecret). Grafana reads it when its Pod starts; after a
# change, restart Grafana: kubectl -n monitoring rollout restart deployment kps-grafana.
# Read the password when you need it: cat k8s/secrets/grafana-admin-password (user: admin).
set -euo pipefail
umask 077

PROFILE="${K8S_PROFILE:-milk10k}"          # minikube profile = kube context
NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/k8s/secrets"
FILE="$DIR/grafana-admin-password"

mkdir -p "$DIR"
chmod 700 "$DIR"
if [ ! -s "$FILE" ]; then
    python3 -c 'import secrets; print(secrets.token_urlsafe(24), end="")' > "$FILE"
    echo "created ${FILE#"$ROOT"/} (random; it is only in that file)"
fi
chmod 600 "$FILE"

kubectl --context "$PROFILE" apply -f "$ROOT/k8s/monitoring/namespace.yaml"
kubectl --context "$PROFILE" -n "$NAMESPACE" create secret generic grafana-admin \
    --from-literal=admin-user=admin --from-file=admin-password="$FILE" --dry-run=client -o yaml \
    | kubectl --context "$PROFILE" apply -f -
