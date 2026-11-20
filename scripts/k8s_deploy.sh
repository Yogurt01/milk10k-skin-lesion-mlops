#!/usr/bin/env bash
# Deploy the API image to the local minikube cluster with the Helm chart in k8s/. One script for the host
# (`make k8s-deploy`) and for the Jenkins Kubernetes stage, so both do exactly the same.
#
#   scripts/k8s_deploy.sh check            exit 0 if the API server answers, 3 if it is not reachable
#   scripts/k8s_deploy.sh load TAG         copy milk10k-api:TAG into the node's containerd (skipped if present)
#   scripts/k8s_deploy.sh deploy TAG       helm upgrade --install with rollback on failure, then rollout status
#   scripts/k8s_deploy.sh test TAG         smoke test + HTTP parity (6 sample lesions) on a Pod running TAG
#   scripts/k8s_deploy.sh cleanup          keep the newest 3 pipeline tags in the node (and any tag still needed)
#
# Settings (environment): KUBE_CONTEXT (empty = the kubeconfig's current context, as in Jenkins), K8S_NODE (node
# container, default milk10k), K8S_NAMESPACE, K8S_RELEASE, K8S_VALUES (extra values files, space-separated),
# REPLICAS (optional override), K8S_TIMEOUT (Helm wait before rollback, default 5m0s), K8S_LOCAL_PORT (port-forward,
# default 18000), K8S_PARITY_JUNIT.
# Only milk10k resources are touched; images are removed one by one, never pruned. Research model, not a medical device.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NODE="${K8S_NODE:-milk10k}"
NAMESPACE="${K8S_NAMESPACE:-milk10k}"
RELEASE="${K8S_RELEASE:-milk10k-api}"
CHART="k8s/milk10k-api"
VALUES="k8s/values-local.yaml ${K8S_VALUES:-}"
LOCAL_PORT="${K8S_LOCAL_PORT:-18000}"
PARITY_JUNIT="${K8S_PARITY_JUNIT:-reports/junit-parity-k8s.xml}"
REPO="milk10k-api"
NODE_REF="docker.io/library/$REPO"     # how containerd names an imported milk10k-api:TAG
CI_TAG='^[0-9]+-[0-9a-f]{7}$'           # pipeline tags: <build>-<sha7>
KEEP=3
TIMEOUT="${K8S_TIMEOUT:-5m0s}"          # how long Helm waits for the new release before rolling back

kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} -n "$NAMESPACE" "$@"; }
hc() { helm ${KUBE_CONTEXT:+--kube-context "$KUBE_CONTEXT"} -n "$NAMESPACE" "$@"; }
node_tags() { docker exec "$NODE" ctr -n k8s.io images ls -q | sed -n "s|^$NODE_REF:||p"; }
need_tag() { [ -n "${1:-}" ] || { echo "k8s: an image tag is required" >&2; exit 2; }; }

check() {
    if kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} get --raw /readyz --request-timeout=5s >/dev/null 2>&1; then
        echo "k8s: API server reachable"
    else
        echo "k8s: cluster not reachable (is the minikube profile running? minikube start -p $NODE)" >&2
        exit 3
    fi
}

load() {
    local tag="$1"; need_tag "$tag"
    if node_tags | grep -qxF "$tag"; then
        echo "k8s: $REPO:$tag already in the node"
        return
    fi
    docker image inspect "$REPO:$tag" >/dev/null
    local start; start=$(date +%s)
    # One platform only: the host's image store may list others whose layers are not present.
    docker save --platform linux/amd64 "$REPO:$tag" | docker exec -i "$NODE" ctr -n k8s.io images import - >/dev/null
    node_tags | grep -qxF "$tag" || { echo "k8s: $REPO:$tag not found in the node after the import" >&2; exit 1; }
    echo "k8s: loaded $REPO:$tag into the node in $(( $(date +%s) - start )) s"
}

deploy() {
    local tag="$1"; need_tag "$tag"
    local args=() f
    for f in $VALUES; do args+=(-f "$f"); done
    [ -n "${REPLICAS:-}" ] && args+=(--set "replicaCount=$REPLICAS")
    local start; start=$(date +%s)
    # Rolls back to the last good release if the new one does not become ready in time (implies --wait=watcher).
    if ! hc upgrade --install "$RELEASE" "$CHART" "${args[@]}" --set "image.tag=$tag" \
            --rollback-on-failure --timeout "$TIMEOUT"; then
        echo "k8s: deploy of $REPO:$tag FAILED; Helm rolled back to the previous release:" >&2
        hc history "$RELEASE" --max 3 >&2 || true
        kc get pods -o wide >&2 || true
        exit 1
    fi
    kc rollout status "deployment/$RELEASE" --timeout=120s
    echo "k8s: $REPO:$tag deployed in $(( $(date +%s) - start )) s"
    # Old Pods may still be shutting down (TERMINATING shows when); they no longer get new connections.
    kc get pods -l "app.kubernetes.io/instance=$RELEASE" -o custom-columns='POD:.metadata.name,READY:.status.containerStatuses[0].ready,IMAGE:.spec.containers[0].image,TERMINATING:.metadata.deletionTimestamp'
}

test_api() {
    local tag="$1"; need_tag "$tag"
    # A Pod of the new image that is not terminating: a port-forward to the Service could pick an old Pod that is
    # still shutting down, and the tests would then check the previous image.
    local pod
    pod="$(kc get pods -l "app.kubernetes.io/instance=$RELEASE" -o json | python3 -c '
import json, sys
want = sys.argv[1]
for p in json.load(sys.stdin)["items"]:
    ready = all(c.get("ready") for c in p["status"].get("containerStatuses", []))
    if p["spec"]["containers"][0]["image"] == want and not p["metadata"].get("deletionTimestamp") and ready:
        print(p["metadata"]["name"]); break' "$REPO:$tag")"
    [ -n "$pod" ] || { echo "k8s: no ready Pod runs $REPO:$tag" >&2; exit 1; }
    echo "k8s: testing Pod $pod ($REPO:$tag)"
    mkdir -p reports
    kc port-forward "pod/$pod" "$LOCAL_PORT:8000" > reports/k8s-port-forward.log 2>&1 &
    local pf=$!
    trap 'kill $pf 2>/dev/null || true' EXIT
    local base="http://127.0.0.1:$LOCAL_PORT"
    timeout 30 sh -c "until curl -fsS -o /dev/null '$base/ready'; do sleep 1; done"
    make smoke API_BASE_URL="$base" 2>&1 | tee reports/smoke-k8s.log
    REQUIRE_MODEL_PACKAGE=1 PYTEST_ADDOPTS=-rP make test-container API_BASE_URL="$base" PARITY_JUNIT="$PARITY_JUNIT"
}

cleanup() {
    # Tags still needed: those of running Pods and of the previous Helm revision (a rollback must find its image).
    local keep_list prev
    keep_list="$(kc get pods -o jsonpath='{range .items[*]}{.spec.containers[0].image}{"\n"}{end}' | sed -n "s|^$REPO:||p")"
    prev="$(hc history "$RELEASE" --max 2 -o json | python3 -c '
import json, sys
revs = json.load(sys.stdin)
print(revs[0]["revision"] if len(revs) > 1 else "")')"
    if [ -n "$prev" ]; then
        keep_list="$keep_list"$'\n'"$(hc get values "$RELEASE" --revision "$prev" -o json \
            | python3 -c 'import json, sys; print((json.load(sys.stdin).get("image") or {}).get("tag", ""))')"
    fi
    local tag n=0
    for tag in $(node_tags | grep -E "$CI_TAG" | sort -t- -k1,1nr); do
        n=$((n + 1))
        if [ "$n" -le "$KEEP" ] || printf '%s\n' "$keep_list" | grep -qxF "$tag"; then
            echo "k8s: keep $REPO:$tag in the node"
        else
            docker exec "$NODE" ctr -n k8s.io images rm "$NODE_REF:$tag" >/dev/null && echo "k8s: removed $REPO:$tag from the node"
        fi
    done
}

case "${1:-}" in
    check) check ;;
    load) load "${2:-}" ;;
    deploy) deploy "${2:-}" ;;
    test) test_api "${2:-}" ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 check | load TAG | deploy TAG | test TAG | cleanup" >&2; exit 2 ;;
esac
