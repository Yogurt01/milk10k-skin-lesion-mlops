#!/usr/bin/env bash
# The GKE test (deploy/README.md): one script behind the `make gke-*` targets. Every kubectl and Helm call names the
# GKE context explicitly and every gcloud call names the project and runs in its own gcloud configuration, so a
# command can never land on the local minikube cluster or on another Google Cloud project.
#
#   scripts/gke.sh render              offline: helm template with the GKE values and placeholders (+ kubeconform)
#   scripts/gke.sh preflight           local checks before anything billable (tools, gcloud configuration, tfvars)
#   scripts/gke.sh credentials         kubeconfig entry for the cluster, renamed to the context milk10k-gke
#   scripts/gke.sh registry-login      Docker credential helper for the registry host (gcloud, no key file)
#   scripts/gke.sh push TAG            tag milk10k-api:TAG with the registry path and push it
#   scripts/gke.sh upload-model        verify the local package, then copy manifest.json + its files to the bucket
#   scripts/gke.sh monitoring          Grafana Secret, kube-prometheus-stack (local + GKE values), dashboard and rules
#   scripts/gke.sh deploy TAG          helm upgrade --install with values-gke.yaml; prints each Pod's sha256 check
#   scripts/gke.sh test                smoke test (and the sample-lesion parity test if present) through a port-forward
#   scripts/gke.sh expose CIDR         LoadBalancer reachable only from CIDR (e.g. your IP/32), then curl /health
#   scripts/gke.sh unexpose            back to ClusterIP (removes the load balancer)
#   scripts/gke.sh down                uninstall the releases, wait for the load balancer to go, terraform destroy
#   scripts/gke.sh leftovers           list everything billable that is left in the project; works without the state
#
# Settings (environment): GKE_PROJECT (default: Terraform output, needed by `leftovers` when the state is gone),
# GKE_REGION (default us-central1), GKE_GCLOUD_CONFIG (gcloud configuration, default milk10k), KPS_CHART (from the
# Makefile), REPLICAS (optional override). Never prints a credential. Research model, not a medical device.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TF_DIR="deploy/terraform"
CONTEXT="milk10k-gke"
NAMESPACE="milk10k"
RELEASE="milk10k-api"
CHART="k8s/milk10k-api"
MODEL_DIR="${MODEL_DIR:-artifacts/model_package/final-P1a-kg-onnx}"
REGION="${GKE_REGION:-us-central1}"
export CLOUDSDK_ACTIVE_CONFIG_NAME="${GKE_GCLOUD_CONFIG:-milk10k}"
export CHECKPOINT_DISABLE=1              # no Terraform version check calls

die() { echo "gke: ERROR: $*" >&2; exit 1; }
log() { echo "gke: $*"; }
kc() { kubectl --context "$CONTEXT" "$@"; }
hc() { helm --kube-context "$CONTEXT" "$@"; }
tf_out() { terraform -chdir="$TF_DIR" output -raw "$1" 2>/dev/null; }
project() {
    local p="${GKE_PROJECT:-$(tf_out project_id || true)}"
    [ -n "$p" ] || die "project unknown: no Terraform state here; set GKE_PROJECT=<project id>"
    echo "$p"
}
gc() { gcloud --project "$(project)" --quiet "$@"; }
need_tag() { [ -n "${1:-}" ] || die "an image tag is required (a pinned tag, never latest)"; }

render() {
    local out; out="$(mktemp -d)"
    local args=(template "$RELEASE" "$CHART" -n "$NAMESPACE" -f k8s/values-gke.yaml --set image.tag=0-0000000
        --set image.repository=us-central1-docker.pkg.dev/example-project/milk10k/milk10k-api
        --set model.gcs.bucket=example-project-milk10k-model --api-versions monitoring.coreos.com/v1/ServiceMonitor)
    helm "${args[@]}" > "$out/gke.yaml"
    helm "${args[@]}" --set service.type=LoadBalancer --set 'service.loadBalancerSourceRanges[0]=203.0.113.10/32' \
        > "$out/gke-lb.yaml"
    log "rendered: $(grep -c '^kind:' "$out/gke.yaml") objects ($(grep '^kind:' "$out/gke.yaml" | cut -d' ' -f2 | paste -sd' '))"
    if grep -q 'hostPath' "$out/gke.yaml"; then die "the GKE render contains a hostPath volume"; fi
    if command -v kubeconform >/dev/null; then
        # Newest published schemas are older than the cluster's Kubernetes version; CRDs (ServiceMonitor) are skipped.
        kubeconform -strict -summary -kubernetes-version "${KUBECONFORM_K8S:-1.34.12}" -ignore-missing-schemas \
            "$out/gke.yaml" "$out/gke-lb.yaml"
    else
        log "kubeconform not installed: schema check skipped"
    fi
    rm -rf "$out"
}

preflight() {
    local t
    for t in terraform gcloud kubectl helm docker; do command -v "$t" >/dev/null || die "$t not found"; done
    gcloud config configurations describe "$CLOUDSDK_ACTIVE_CONFIG_NAME" >/dev/null 2>&1 \
        || die "gcloud configuration $CLOUDSDK_ACTIVE_CONFIG_NAME missing (deploy/README.md, phase B step 4)"
    [ -n "$(gcloud config get account 2>/dev/null)" ] || die "no account in gcloud configuration $CLOUDSDK_ACTIVE_CONFIG_NAME"
    local configured; configured="$(gcloud config get project 2>/dev/null)"
    [ -f "$TF_DIR/terraform.tfvars" ] || die "$TF_DIR/terraform.tfvars missing (copy terraform.tfvars.example)"
    [ "$(stat -c %a "$TF_DIR/terraform.tfvars")" = 600 ] || die "$TF_DIR/terraform.tfvars must have mode 600"
    git check-ignore -q "$TF_DIR/terraform.tfvars" || die "$TF_DIR/terraform.tfvars is not gitignored"
    local want; want="$(sed -n 's/^project_id *= *"\(.*\)"/\1/p' "$TF_DIR/terraform.tfvars")"
    [ "$configured" = "$want" ] || die "gcloud configuration project and terraform.tfvars project_id differ"
    log "preflight OK: gcloud configuration $CLOUDSDK_ACTIVE_CONFIG_NAME (account set), project matches tfvars"
}

credentials() {
    local p zone name; p="$(project)"; zone="$(tf_out zone)"; name="$(tf_out cluster_name)"
    kubectl config delete-context "$CONTEXT" >/dev/null 2>&1 || true
    gc container clusters get-credentials "$name" --zone "$zone"
    kubectl config rename-context "gke_${p}_${zone}_${name}" "$CONTEXT"
    kc get nodes
}

registry_login() { gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet; }

push() {
    local tag="$1"; need_tag "$tag"
    local repo; repo="$(tf_out image_repository)" || die "no Terraform output image_repository"
    docker image inspect "milk10k-api:$tag" >/dev/null || die "local image milk10k-api:$tag not found"
    docker tag "milk10k-api:$tag" "$repo:$tag"
    docker push "$repo:$tag"
    gc artifacts docker images list "$repo" --include-tags --format='table(package.basename(),tags,createTime)'
}

upload_model() {
    local bucket; bucket="$(tf_out model_bucket)" || die "no Terraform output model_bucket"
    scripts/fetch_model.sh --check "$MODEL_DIR"
    local files=("$MODEL_DIR/manifest.json")
    mapfile -t -O 1 files < <(python3 -c 'import json,sys; [print(sys.argv[1] + "/" + f) for f in json.load(open(sys.argv[1] + "/manifest.json"))["files"]]' "$MODEL_DIR")
    gc storage cp "${files[@]}" "gs://$bucket/$(basename "$MODEL_DIR")/"
    local n; n="$(gc storage ls "gs://$bucket/$(basename "$MODEL_DIR")/" | wc -l)"
    [ "$n" -eq "${#files[@]}" ] || die "bucket holds $n objects, expected ${#files[@]}"
    log "uploaded ${#files[@]} files to the private bucket (manifest + listed files; gcloud checks each upload's hash)"
}

monitoring() {
    [ -n "${KPS_CHART:-}" ] || die "KPS_CHART not set (run through make gke-monitoring)"
    K8S_PROFILE="$CONTEXT" scripts/k8s_grafana_secret.sh
    hc upgrade --install kps "$KPS_CHART" -n monitoring -f k8s/monitoring/values-kube-prometheus-stack.yaml \
        -f k8s/monitoring/values-kps-gke.yaml --wait --timeout 10m
    kc -n monitoring wait --for=create statefulset/prometheus-kps-kube-prometheus-stack-prometheus --timeout=2m
    kc -n monitoring rollout status statefulset/prometheus-kps-kube-prometheus-stack-prometheus --timeout=5m
    kc apply -k k8s/monitoring
}

deploy() {
    local tag="$1"; need_tag "$tag"
    local repo bucket; repo="$(tf_out image_repository)"; bucket="$(tf_out model_bucket)"
    [ -n "$repo" ] && [ -n "$bucket" ] || die "Terraform outputs missing (run make tf-apply first)"
    local args=(-f k8s/values-gke.yaml --set "image.tag=$tag" --set "image.repository=$repo" --set "model.gcs.bucket=$bucket")
    [ -n "${REPLICAS:-}" ] && args+=(--set "replicaCount=$REPLICAS")
    kc apply -f k8s/namespace.yaml
    local start; start=$(date +%s)
    hc -n "$NAMESPACE" upgrade --install "$RELEASE" "$CHART" "${args[@]}" --rollback-on-failure --timeout 15m
    kc -n "$NAMESPACE" rollout status "deployment/$RELEASE" --timeout=300s
    log "deployed $tag in $(( $(date +%s) - start )) s"
    kc -n "$NAMESPACE" get deploy "$RELEASE"
    kc -n "$NAMESPACE" get pods -l "app.kubernetes.io/instance=$RELEASE" -o wide
    local pod
    for pod in $(kc -n "$NAMESPACE" get pods -l "app.kubernetes.io/instance=$RELEASE" -o name); do
        echo "$pod: $(kc -n "$NAMESPACE" logs "$pod" -c verify-model | tail -1)"
    done
}

test_api() {
    local log port="" i base
    log="$(mktemp)"
    kubectl --context "$CONTEXT" -n "$NAMESPACE" port-forward "svc/$RELEASE" ":8000" > "$log" 2>&1 &
    PF_PID=$!
    trap 'kill "$PF_PID" 2>/dev/null || true' EXIT
    for i in $(seq 1 30); do
        port="$(sed -n 's/^Forwarding from 127\.0\.0\.1:\([0-9][0-9]*\) -> 8000$/\1/p' "$log" | head -1)"
        [ -n "$port" ] && break
        sleep 1
    done
    [ -n "$port" ] || die "the port-forward did not start: $(cat "$log")"
    base="http://127.0.0.1:$port"
    make smoke API_BASE_URL="$base"
    if [ -d training/sample/images ]; then
        REQUIRE_MODEL_PACKAGE=1 make test-container API_BASE_URL="$base" PARITY_JUNIT=reports/junit-parity-gke.xml
    else
        log "no sample lesions (make sample): parity test skipped"
    fi
}

expose() {
    local cidr="${1:-}"
    [ -n "$cidr" ] || die "usage: scripts/gke.sh expose <your public IP>/32"
    hc -n "$NAMESPACE" upgrade "$RELEASE" "$CHART" --reuse-values --set service.type=LoadBalancer \
        --set "service.loadBalancerSourceRanges[0]=$cidr" --wait --timeout 10m
    local ip="" i
    for i in $(seq 1 60); do
        ip="$(kc -n "$NAMESPACE" get svc "$RELEASE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
        [ -n "$ip" ] && break
        sleep 5
    done
    [ -n "$ip" ] || die "no external IP after 5 minutes"
    kc -n "$NAMESPACE" get svc "$RELEASE" -o wide
    # The source range (your own IP) is not printed: this output ends up in screenshots.
    log "only the given network may connect; checking /health from here:"
    curl -fsS --max-time 10 "http://$ip:8000/health"; echo
}

unexpose() {
    hc -n "$NAMESPACE" upgrade "$RELEASE" "$CHART" --reuse-values --set service.type=ClusterIP \
        --set service.loadBalancerSourceRanges=null --wait --timeout 10m
    kc -n "$NAMESPACE" get svc "$RELEASE"
}

down() {
    # Kubernetes-created load balancers and disks are not in the Terraform state: remove them first, through the
    # cluster, or `terraform destroy` would leave them behind (still billed).
    if kc get --raw /readyz --request-timeout=10s >/dev/null 2>&1; then
        hc -n "$NAMESPACE" uninstall "$RELEASE" --wait || true
        hc -n monitoring uninstall kps --wait || true
        local i
        for i in $(seq 1 36); do
            [ -z "$(gc compute forwarding-rules list --format='value(name)')" ] && break
            log "waiting for the load balancer to be removed ..."
            sleep 5
        done
    else
        log "cluster not reachable: releases not uninstalled (the leftover check shows what remains)"
    fi
    terraform -chdir="$TF_DIR" destroy
    kubectl config delete-context "$CONTEXT" >/dev/null 2>&1 || true
    leftovers
}

leftovers() {
    # Lists by project, not from the Terraform state, so it also works when the state is lost.
    local p found=0 failed=0 n what out state; p="$(project)"
    state="$(gcloud projects describe "$p" --format='value(lifecycleState)' 2>/dev/null || true)"
    if [ "$state" = "DELETE_REQUESTED" ]; then
        log "project $p is pending deletion (shut down: its resources are stopped and no longer billed)"
        return 0
    fi
    [ "$state" = "ACTIVE" ] || die "project $p not readable (state '${state:-unknown}'): check the gcloud configuration"
    log "leftovers in project $p (region $REGION):"
    # A list that fails is an error, never a zero: "could not look" must not read as "nothing left".
    check() {
        what="$1"; shift
        if ! out="$("$@" 2>&1)"; then
            printf '  %-28s %s\n' "$what" "ERROR (could not list)"
            failed=1
            return 0
        fi
        n="$(printf '%s' "$out" | grep -c . || true)"
        printf '  %-28s %s\n' "$what" "$n"
        [ "$n" -eq 0 ] || { printf '%s\n' "$out" | sed 's/^/      /'; found=1; }
    }
    check "GKE clusters" gc container clusters list --format='value(name,location)'
    check "VM instances" gc compute instances list --format='value(name,zone)'
    check "disks" gc compute disks list --format='value(name,zone,sizeGb)'
    check "snapshots" gc compute snapshots list --format='value(name)'
    check "forwarding rules" gc compute forwarding-rules list --format='value(name,region,IPAddress)'
    check "target pools" gc compute target-pools list --format='value(name,region)'
    check "backend services" gc compute backend-services list --format='value(name)'
    check "external IP addresses" gc compute addresses list --format='value(name,address,status)'
    check "registries ($REGION)" gc artifacts repositories list --location="$REGION" --format='value(name)'
    check "buckets" gc storage buckets list --format='value(name)'
    [ "$failed" -eq 0 ] || die "some lists failed (above); nothing is proven: fix access or delete the project"
    [ "$found" -eq 0 ] || die "resources left (above): remove them, or delete the project"
    log "nothing billable found by these lists; last step: gcloud projects delete $p"
}

cmd="${1:-}"; shift || true
case "$cmd" in
    render) render ;;
    preflight) preflight ;;
    credentials) credentials ;;
    registry-login) registry_login ;;
    push) push "${1:-}" ;;
    upload-model) upload_model ;;
    monitoring) monitoring ;;
    deploy) deploy "${1:-}" ;;
    test) test_api ;;
    expose) expose "${1:-}" ;;
    unexpose) unexpose ;;
    down) down ;;
    leftovers) leftovers ;;
    *) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
