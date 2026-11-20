#!/usr/bin/env bash
# Write the kubeconfig Jenkins uses to deploy to the local cluster (`make ci-kubeconfig`). Run on the host while the
# minikube profile is running, with its admin context. It
#   1. applies k8s/namespace.yaml and k8s/ci-rbac.yaml (ServiceAccount ci-deployer, rights in one namespace only),
#   2. requests a time-limited token for that ServiceAccount (TokenRequest API, `kubectl create token`),
#   3. writes ci/secrets/kubeconfig (mode 600, gitignored): API server on the minikube network, the cluster's public
#      CA certificate (from the kube-root-ca.crt ConfigMap), the token.
# The token is never printed and never put on a command line. After running it, restart Jenkins (make ci-down,
# make ci-up): Configuration as Code reads the file into a Jenkins credential at start.
set -euo pipefail
umask 077

PROFILE="${K8S_PROFILE:-milk10k}"      # minikube profile = kube context = Docker network
NAMESPACE="${K8S_NAMESPACE:-milk10k}"
SA="ci-deployer"
DURATION="${CI_TOKEN_DURATION:-720h}"  # requested lifetime; the API server may shorten it
APISERVER_PORT=8443                     # minikube's default API server port
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/ci/secrets/kubeconfig"

kubectl --context "$PROFILE" get --raw /readyz >/dev/null \
    || { echo "ci-kubeconfig: cluster $PROFILE not reachable (minikube start -p $PROFILE)" >&2; exit 1; }

kubectl --context "$PROFILE" apply -f "$ROOT/k8s/namespace.yaml" -f "$ROOT/k8s/ci-rbac.yaml"

server="https://$(minikube -p "$PROFILE" ip):$APISERVER_PORT"
ca_b64="$(kubectl --context "$PROFILE" -n "$NAMESPACE" get configmap kube-root-ca.crt -o jsonpath='{.data.ca\.crt}' \
    | base64 -w0)"
[ -n "$ca_b64" ] || { echo "ci-kubeconfig: no CA certificate in kube-root-ca.crt" >&2; exit 1; }
token="$(kubectl --context "$PROFILE" -n "$NAMESPACE" create token "$SA" --duration "$DURATION")"

mkdir -p "$ROOT/ci/secrets"
chmod 700 "$ROOT/ci/secrets"
# Written in place (not renamed over): the file keeps its inode and mode 600.
{
    printf 'apiVersion: v1\nkind: Config\n'
    printf 'clusters:\n- name: %s\n  cluster:\n    server: %s\n    certificate-authority-data: %s\n' \
        "$PROFILE" "$server" "$ca_b64"
    printf 'users:\n- name: %s\n  user:\n    token: %s\n' "$SA" "$token"
    printf 'contexts:\n- name: %s-ci\n  context:\n    cluster: %s\n    user: %s\n    namespace: %s\n' \
        "$PROFILE" "$PROFILE" "$SA" "$NAMESPACE"
    printf 'current-context: %s-ci\n' "$PROFILE"
} > "$OUT"
chmod 600 "$OUT"

# Only the expiry time is shown (from the token's payload), never the token.
expiry="$(printf '%s' "$token" | python3 -c '
import base64, datetime, json, sys
payload = sys.stdin.read().split(".")[1]
payload += "=" * (-len(payload) % 4)
exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
print(datetime.datetime.fromtimestamp(exp, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))')"
unset token
echo "ci-kubeconfig: wrote ${OUT#"$ROOT"/} (mode 600) for ServiceAccount $SA in namespace $NAMESPACE," \
     "API server $server; token valid until $expiry. Restart Jenkins to load it."
