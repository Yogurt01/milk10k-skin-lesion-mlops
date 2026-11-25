# Kubernetes: the API on a local cluster

> **Research model, not a medical device.** Model weights: CC BY-NC 4.0, MILK study team, non-commercial use only.
> They are never part of the image or this chart; the cluster reads them from a folder on your machine.

This folder runs the same API image and the same model package as the Compose stack, as several replicas behind one
Kubernetes Service, on a local minikube cluster, with Prometheus and Grafana installed next to it by Helm.
Everything stays on your machine: no registry, no cloud account, no Ingress; the UIs are reached by
`kubectl port-forward` on 127.0.0.1 only.

| File | What it is |
|---|---|
| [`namespace.yaml`](namespace.yaml) | The `milk10k` namespace |
| [`milk10k-api/`](milk10k-api/) | Helm chart: Deployment (3 replicas by default), a ClusterIP Service and an optional ServiceMonitor |
| [`milk10k-api/values.yaml`](milk10k-api/values.yaml) | Defaults with the reasons for each number (threads, memory, probes, rollout) |
| [`values-local.yaml`](values-local.yaml) | Settings for the local cluster (image loaded into the node, model from a host folder, ServiceMonitor on) |
| [`values-ci.yaml`](values-ci.yaml) | Added by the Jenkins deploy: 1 replica, the new Pod starts before the old one stops |
| [`ci-rbac.yaml`](ci-rbac.yaml) | ServiceAccount `ci-deployer` for Jenkins, with rights in this namespace only ([ci/README.md](../ci/README.md#the-kubernetes-stage)) |
| [`monitoring/values-kube-prometheus-stack.yaml`](monitoring/values-kube-prometheus-stack.yaml) | Helm values for Prometheus + Grafana (kube-prometheus-stack), only the keys changed from the chart defaults |
| [`monitoring/namespace.yaml`](monitoring/namespace.yaml) | The `monitoring` namespace |
| [`monitoring/kustomization.yaml`](monitoring/kustomization.yaml) | The API dashboard (ConfigMap for Grafana) and the API alert rules (PrometheusRule) |
| [`monitoring/dashboards/milk10k-api.json`](monitoring/dashboards/milk10k-api.json) | Grafana dashboard "MILK10k: API on Kubernetes" |
| [`monitoring/prometheusrule-api.yaml`](monitoring/prometheusrule-api.yaml) | Generated from [`monitoring/prometheus/alert-rules.yml`](../monitoring/prometheus/alert-rules.yml) (`make k8s-rules`) |

**Reproduce from zero**, in this order:

| Step | Section | Time | Memory added (measured) |
|---|---|---|---|
| Tools: Docker, kubectl, minikube, Helm; model package; API image | [What you need](#what-you-need) | | |
| Cluster with the model folder mounted | [1](#1-create-the-cluster) | about 1 min | empty node about 0.6 GiB |
| API, 3 replicas | [2](#2-deploy), [3](#3-verify) | under 1 min | about 1.9 GiB per Pod |
| Prometheus + Grafana, API dashboard and alerts | [4](#4-monitoring-prometheus-and-grafana) | about 2 min (first time: image downloads) | about 0.6 GiB |
| Deploy from Jenkins (optional) | [5](#5-deploy-from-jenkins) | | |
| Stop or remove | [6](#6-stop-or-remove) | | |

How far this was checked: sections 1 to 3 were run as written in November 2026 with the profile name typed out
(`milk10k`); sections 4 and 6 were run with 1 API replica on that cluster. The `$PROFILE` variable and a run of the
whole guide on a fresh cluster under another profile name are **not checked yet**.

## What you need

- Docker and `kubectl` ([Ubuntu guide](../docs/install-ubuntu.md)), minikube and Helm (below). This folder was checked
  with minikube v1.39.0, Helm v4.3.2, kubectl v1.36 and Kubernetes v1.37.0.
- The model package in `artifacts/model_package/final-P1a-kg-onnx` (`make fetch-model`, about 1.4 GB).
- The API image built locally, e.g. `docker build -t milk10k-api:<tag> serving` (any tag except `latest`).
- **Memory.** One replica uses about 1.9 GiB once the model is loaded (plus file cache the kernel can reclaim), the
  empty cluster about 0.6 GiB, Prometheus + Grafana about 0.6 GiB; with 3 replicas the node used 6.3 GiB without
  monitoring. On a 16 GB laptop, stop the Compose stack (`make down`) and Jenkins (`make ci-down`) first and close what
  you can: the 3-replica checks below were run with about 10 GiB available before the cluster started, and about
  4 GiB stayed available. Check with `free -h`.

### Install minikube and Helm (Linux x86-64)

Pinned versions, each checked against its published sha256 before it is installed:

```bash
cd "$(mktemp -d)"
curl -fLO https://github.com/kubernetes/minikube/releases/download/v1.39.0/minikube-linux-amd64
curl -fLO https://github.com/kubernetes/minikube/releases/download/v1.39.0/minikube-linux-amd64.sha256
echo "$(cat minikube-linux-amd64.sha256)  minikube-linux-amd64" | sha256sum -c
sudo install -m 0755 minikube-linux-amd64 /usr/local/bin/minikube

curl -fLO https://get.helm.sh/helm-v4.3.2-linux-amd64.tar.gz
curl -fLO https://get.helm.sh/helm-v4.3.2-linux-amd64.tar.gz.sha256sum
sha256sum -c helm-v4.3.2-linux-amd64.tar.gz.sha256sum
tar -xzf helm-v4.3.2-linux-amd64.tar.gz linux-amd64/helm
sudo install -m 0755 linux-amd64/helm /usr/local/bin/helm

minikube version && helm version
```

Sources: [minikube start](https://minikube.sigs.k8s.io/docs/start/), [Installing Helm](https://helm.sh/docs/intro/install/).
No `helm repo add` is needed: the monitoring chart is pulled from its OCI registry by version and digest.

## 1. Create the cluster

Run every command from the repository root. The cluster gets its own minikube profile, so other clusters are not
touched; the profile name is also the kubectl context and the name of the node container. Set it once per shell
(`milk10k` everywhere in this repository; use another name, e.g. `milk10k-fresh`, to try the guide next to an
existing cluster):

```bash
PROFILE=milk10k
minikube start -p $PROFILE --driver=docker --memory=8g --cpus=12 --kubernetes-version=v1.37.0 \
  --mount-string="$PWD/artifacts/model_package:/mnt/milk10k-model"
kubectl --context $PROFILE get nodes          # STATUS Ready
```

The model folder is mounted into the node when the cluster is created (only then). With the docker driver the mount
is a plain bind mount of the host folder, and the node container is capped at 8 GiB. The node still reports the
host's CPUs and memory to Kubernetes (node-exporter in section 4 shows the host's 16 CPUs and about 15 GiB too), so
the replica count and the rollout settings, not the scheduler, keep the cluster inside that budget. The cap can only
be changed by deleting and recreating the cluster.

## 2. Deploy

```bash
TAG=<tag>                                     # the tag you built, e.g. the short git commit
minikube -p $PROFILE image load milk10k-api:$TAG
kubectl --context $PROFILE apply -f k8s/namespace.yaml
helm upgrade --install milk10k-api k8s/milk10k-api --kube-context $PROFILE -n milk10k \
  -f k8s/values-local.yaml --set image.tag=$TAG
kubectl --context $PROFILE -n milk10k rollout status deployment/milk10k-api
```

Add `--set replicaCount=2` (or 1) if the machine does not have the memory for 3.

The same as one command, with [`scripts/k8s_deploy.sh`](../scripts/k8s_deploy.sh), which the Jenkins pipeline uses
too: the image is copied into the node only if it is not there yet (`docker save` into the node's containerd), and
`helm upgrade --rollback-on-failure` returns to the previous release if the new one does not become ready in 5 min.

```bash
make k8s-deploy TAG=$TAG K8S_PROFILE=$PROFILE        # optional: REPLICAS=2
```

What each Pod does when it starts:

1. The `verify-model` init container checks every file of the model package against the sha256 in `manifest.json`
   and logs `sha256 check passed: 18 files match manifest.json`. A mismatch stops the Pod here.
2. The API starts and loads the 15 ONNX sessions. Until that is done `/ready` answers 503, so the startup probe
   keeps the Pod out of the Service.
3. Once `/ready` is 200 the Pod is Ready and receives requests.

Expected warning: the namespace warns that the Pod would violate the `restricted` Pod Security Standard because of the
`hostPath` volume. The model folder comes from the node, which needs a hostPath volume; the Baseline and Restricted
standards forbid those, so the namespace warns instead of enforcing. Everything else in the Pod meets Restricted.

`values-local.yaml` also turns on the chart's ServiceMonitor for Prometheus (section 4). The chart creates it only if
the cluster already knows the ServiceMonitor type, so this step works the same with or without monitoring installed;
if you install monitoring afterwards, run the deploy once more to add the ServiceMonitor.

## 3. Verify

```bash
kubectl --context $PROFILE -n milk10k get deploy,pods -o wide        # READY 3/3, one IP per Pod
kubectl --context $PROFILE -n milk10k logs deploy/milk10k-api -c verify-model
kubectl --context $PROFILE -n milk10k describe pod -l app.kubernetes.io/name=milk10k-api | grep -E "Liveness|Readiness|Startup|Limits|Requests" -A2
```

**Smoke test and parity** through a port-forward (with the serving environment from `make setup`):

```bash
kubectl --context $PROFILE -n milk10k port-forward service/milk10k-api 18000:8000 &
python3 scripts/smoke_test.py --base-url http://127.0.0.1:18000
make test-container API_BASE_URL=http://127.0.0.1:18000           # 6 sample lesions, tolerance 0.002
kill %1
```

**Requests spread over the Pods.** `kubectl port-forward` to a Service picks *one* Pod and sends every connection to
it, so it cannot show load balancing. Send requests to the Service from inside the cluster instead, one new connection
each, and count them in each Pod's log (`/openapi.json` is logged with its request id; probes are not):

```bash
POD=$(kubectl --context $PROFILE -n milk10k get pods -o name | head -1)
kubectl --context $PROFILE -n milk10k exec $POD -c api -- python -c '
import urllib.request as u
for i in range(20):
    r = u.Request("http://milk10k-api:8000/openapi.json", headers={"X-Request-ID": f"spread-{i}", "Connection": "close"})
    u.urlopen(r, timeout=10).read()'
for p in $(kubectl --context $PROFILE -n milk10k get pods -o name); do
  echo "$p $(kubectl --context $PROFILE -n milk10k logs $p -c api | grep -c '\[spread-')"
done
```

**Self-healing.** Delete one Pod; the Deployment starts a replacement and the Service keeps answering from the others:

```bash
kubectl --context $PROFILE -n milk10k delete pod <one of the pods>
kubectl --context $PROFILE -n milk10k get pods -w
```

**Rolling update.** Build and load a new tag, then run the same `helm upgrade` with it (or `make k8s-deploy
TAG=<new tag>`). Pods are replaced one at a time (`maxSurge: 0`, `maxUnavailable: 1`): no extra 2.5 GiB Pod is
created on purpose, so one replica fewer serves while each Pod is swapped. This is not a hard cap: the Deployment
does not count Pods that are still terminating, so an old Pod can briefly run next to its replacement (4 Pods were
seen during a 3-replica rollout). Each old Pod waits 5 s (`preStop` sleep) before it stops, so the Service can stop
sending it new connections first. That pause is a precaution: rollouts without it were not tested.

```bash
helm history milk10k-api --kube-context $PROFILE -n milk10k
helm rollback milk10k-api <revision> --kube-context $PROFILE -n milk10k     # back to an earlier release if needed
```

Use Helm, not `kubectl rollout undo`, to go back: Helm keeps the release history, and a change made behind its back
is overwritten by the next `helm upgrade`.

## 4. Monitoring: Prometheus and Grafana

[kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
**93.10.0**, pinned by version and OCI digest (Helm refuses the chart if the digest does not match), installed as
release `kps` in the namespace `monitoring`. It brings the Prometheus Operator, Prometheus v3.16.0 and Grafana 13.3.1
(the same versions as the Compose stack), kube-state-metrics, node-exporter and the default Kubernetes dashboards.
Alertmanager is off; alerts are shown in Prometheus, as in Compose.

**Install** (namespace, Grafana login, Helm release, then the API dashboard and alert rules):

```bash
make k8s-monitoring K8S_PROFILE=$PROFILE
```

which runs:

```bash
scripts/k8s_grafana_secret.sh       # with K8S_PROFILE=$PROFILE: creates k8s/secrets/grafana-admin-password once
                                    # (random, gitignored) and the Secret grafana-admin in namespace monitoring
helm upgrade --install kps oci://ghcr.io/prometheus-community/charts/kube-prometheus-stack:93.10.0@sha256:3dcc77c6c3745df0ef2454e04315cd26d010f6948a8167392ac4e602ecdacba7 \
  --kube-context $PROFILE -n monitoring -f k8s/monitoring/values-kube-prometheus-stack.yaml --wait --timeout 10m
kubectl --context $PROFILE -n monitoring wait --for=create statefulset/prometheus-kps-kube-prometheus-stack-prometheus --timeout=2m
kubectl --context $PROFILE -n monitoring rollout status statefulset/prometheus-kps-kube-prometheus-stack-prometheus --timeout=5m
kubectl --context $PROFILE apply -k k8s/monitoring
```

Helm's `--wait` does not cover Prometheus itself: the operator creates its StatefulSet after Helm returns, hence the
two `kubectl` waits. If the API was deployed before this step, deploy it once more (section 2) so the chart adds the
ServiceMonitor. Ignore the Grafana password commands in Helm's notes: they refer to a Secret this setup does not
create.

```bash
helm list --kube-context $PROFILE -A                   # kps (kube-prometheus-stack-93.10.0) and milk10k-api, deployed
kubectl --context $PROFILE -n monitoring get pods      # 5 Pods Running: grafana 3/3, operator, kube-state-metrics,
                                                       # node-exporter, prometheus 2/2
```

**Open the UIs** (two terminals; ports 19090 and 13000, so they do not clash with the Compose stack):

```bash
kubectl --context $PROFILE -n monitoring port-forward --address 127.0.0.1 svc/kps-kube-prometheus-stack-prometheus 19090:9090
kubectl --context $PROFILE -n monitoring port-forward --address 127.0.0.1 svc/kps-grafana 13000:80
```

- Prometheus: <http://127.0.0.1:19090/targets>. The job `api` has one target per API Pod (labels `namespace`, `pod`),
  scraped every 15 s directly from the Pod, not through the Service.
- Grafana: <http://127.0.0.1:13000>, user `admin`, password in `k8s/secrets/grafana-admin-password`. Dashboards:
  "MILK10k: API on Kubernetes" (from the ConfigMap), and the chart's defaults, e.g. "Kubernetes / Compute Resources /
  Pod" and "Node Exporter / Nodes" for node and Pod resources.

**Send some predictions** through the Service from inside the cluster (the smoke test, piped into an API Pod's Python;
it posts a synthetic gradient image, never a lesion image), then look at the dashboard:

```bash
POD=$(kubectl --context $PROFILE -n milk10k get pods -o name | head -1)
for i in $(seq 20); do
  kubectl --context $PROFILE -n milk10k exec -i $POD -c api -- python - --base-url http://milk10k-api:8000 < scripts/smoke_test.py | tail -1
done
```

**Alert rules.** The API's four rules (ApiDown, ApiModelNotReady, ApiHighErrorRate, ApiHighLatencyP95) are the same
as in Compose: `prometheusrule-api.yaml` is generated from `monitoring/prometheus/alert-rules.yml`, and the chart's
ServiceMonitor names the job `api`, so the same promtool tests apply. They are listed at
<http://127.0.0.1:19090/alerts>. To see ApiDown fire and resolve:

```bash
kubectl --context $PROFILE -n milk10k scale deployment/milk10k-api --replicas=0
# about 30 s: ApiDown pending (no API target at all); 1 min later: firing
kubectl --context $PROFILE -n milk10k scale deployment/milk10k-api --replicas=3    # the count you deployed
# resolved as soon as a Pod is scraped again
```

A restarted Pod can show ApiModelNotReady as *pending* for one scrape while it loads the model; it clears on the next
scrape and only fires after 2 minutes.

## 5. Deploy from Jenkins

The pipeline deploys every build of `main` to the cluster of the `milk10k` profile with 1 replica
(`values-local.yaml` + `values-ci.yaml`) and rolls back a failed deploy; set-up and token renewal:
[ci/README.md](../ci/README.md#the-kubernetes-stage). Its ServiceAccount may create the chart's ServiceMonitor in the
`milk10k` namespace, so the API stays scraped after a Jenkins deploy. If [`ci-rbac.yaml`](ci-rbac.yaml) changes, apply
it again with your own context (`kubectl --context milk10k apply -f k8s/ci-rbac.yaml`); the token stays valid.

## 6. Stop or remove

```bash
minikube stop -p $PROFILE       # or: make k8s-stop. Frees the memory; keeps the cluster, the loaded images and the mount
minikube start -p $PROFILE      # start it again later (the releases come back with it)
helm uninstall milk10k-api --kube-context $PROFILE -n milk10k        # remove the API only
minikube delete -p $PROFILE     # remove the whole cluster, monitoring included
```

**Remove only the monitoring**, in this order. Helm does not delete the operator's CRDs on uninstall, and once the
API release contains a ServiceMonitor, deleting the CRDs first makes every later `helm upgrade` of the API (and the
Jenkins deploy) fail with `no matches for kind "ServiceMonitor"`. So take the ServiceMonitor out of the API release
first:

```bash
helm upgrade milk10k-api k8s/milk10k-api --kube-context $PROFILE -n milk10k --reuse-values --set serviceMonitor.enabled=false
kubectl --context $PROFILE delete -k k8s/monitoring
helm uninstall kps --kube-context $PROFILE -n monitoring --wait
kubectl --context $PROFILE delete crd alertmanagerconfigs.monitoring.coreos.com alertmanagers.monitoring.coreos.com \
  podmonitors.monitoring.coreos.com probes.monitoring.coreos.com prometheusagents.monitoring.coreos.com \
  prometheuses.monitoring.coreos.com prometheusrules.monitoring.coreos.com scrapeconfigs.monitoring.coreos.com \
  servicemonitors.monitoring.coreos.com thanosrulers.monitoring.coreos.com
kubectl --context $PROFILE delete namespace monitoring   # also the Grafana Secret and the webhook certificate Secret
```

After that, deploys work again without monitoring (the chart leaves the ServiceMonitor out while the CRD is missing).
If the CRDs were deleted first by mistake, install the monitoring again (section 4), then run the steps above.

## Design choices

| Topic | Choice | Reason |
|---|---|---|
| Cluster | minikube, docker driver, own profile | The course's local cluster; image loading and host mounts are built in |
| Packaging | Helm chart, values per environment | Replica count, image tag and model source are values, so the same chart can serve a cloud cluster later |
| Image | Loaded into the node, pinned tag, `pullPolicy: Never` locally | No registry needed; the chart refuses an empty tag or `latest` |
| Model | Host folder mounted read-only (hostPath), sha256 check in an init container | 1.4 GB is not downloaded again for every Pod, restart and rollout; the weights stay out of the image |
| Threads | 4 ONNX Runtime threads, 1 running prediction per Pod (queue of 8) | 3 Pods use 12 of 16 CPU threads; a second running prediction added only about 6% throughput in measurements and more memory |
| Memory | request = limit = 2.5 GiB per Pod | Measured peak 2.1 GiB in a container, plus room for queued uploads |
| Probes | startup and readiness on `/ready`, liveness on `/health` (timeout 5 s, 6 failures) | `/ready` is 200 only once the model is loaded; `/health` does no work and stays fast under load, and the generous liveness settings are meant to leave a busy Pod alone (not tested under CPU throttling) |
| Rollout | `maxSurge: 0`, `maxUnavailable: 1`, 5 s `preStop` sleep | No extra Pod is created on purpose (a terminating Pod can still overlap briefly); the pause is a precaution against new connections to a stopping Pod |
| Security | non-root (uid 10001), all capabilities dropped, no privilege escalation, seccomp `RuntimeDefault`, read-only root filesystem with an `emptyDir` for `/tmp`, no service account token | The API needs none of these privileges; `/tmp` holds large uploads |
| Monitoring chart | kube-prometheus-stack 93.10.0, pinned by digest | One release for Prometheus, Grafana, the operator (ServiceMonitor, PrometheusRule) and the Kubernetes dashboards; same Prometheus and Grafana versions as Compose |
| Monitoring footprint | No Alertmanager; etcd, scheduler, controller-manager and kube-proxy not scraped; 1 day of data in the Pod; a memory limit on every component | Fits next to 3 API Pods in the 8 GiB node; minikube serves those control-plane metrics on the node's loopback address only, so their targets would always be down |
| Scraping | ServiceMonitor in the API chart, one target per Pod every 15 s, job `api`; created only when the CRD exists | Per-replica metrics, not one load-balanced target; the same alert rules and tests as Compose; the chart still installs without monitoring |
| Grafana login | Secret from a gitignored file, created by `make k8s-grafana-secret`; UIs by port-forward on 127.0.0.1 | The password is never in a values file, Helm output or a log; nothing is exposed outside the machine |

## What was checked

On the 16 GB reference laptop (minikube v1.39.0, Kubernetes v1.37.0), with 3 replicas:

| Check | Result |
|---|---|
| 3 Pods started together | 3/3 Ready in 10.6 s; each init container logged `sha256 check passed: 18 files match manifest.json`; model load 3.3–3.4 s |
| Requests spread by the Service | 30 new connections from inside the cluster: 11 / 7 / 12 per Pod |
| Smoke test and parity (6 sample lesions) | smoke test passed; max difference to the reference 2.55e-04 (tolerance 0.002) |
| One Pod deleted under traffic | 183 / 183 requests answered 200; 3/3 Ready again after 8.7 s |
| Rolling update to another image tag under traffic | 373 / 373 requests answered 200; 25.3 s |
| Upload above 1 MB (spooled to `/tmp`) | 200 with the read-only root filesystem |
| Memory | about 1.9 GiB per Pod (no swap, no OOM kill); node 6.3 GiB |

Screenshots of a later deployment of the same release: [Pods 3/3](../docs/images/k8s-pods.png),
[sha256 check of each Pod](../docs/images/k8s-verify-model.png), [20 requests spread over the 3 Pods](../docs/images/k8s-spread.png),
[probes, requests and limits](../docs/images/k8s-describe.png), [Helm release](../docs/images/helm-list.png).

Monitoring, with 1 API replica (the 3-replica run is the next check):

| Check | Result |
|---|---|
| Install of release `kps` | about 2 min the first time (images downloaded into the node), 26 s again later; 5 Pods Running, no restart |
| Prometheus targets | all 12 up: API, apiserver, CoreDNS, kubelet (3 endpoints), node-exporter, kube-state-metrics, operator, Prometheus (2), Grafana |
| API target | job `api`, labels `namespace`, `pod`, `service`, `container`; created by the chart under the Jenkins ServiceAccount's rights |
| Grafana | login with the Secret (without it: 401); 24 default dashboards + "MILK10k: API on Kubernetes"; after 40 predictions every API panel has data except "5xx share" (no 5xx happened) |
| Default dashboards | Pod memory 1.9 GiB of the 2.5 GiB limit; node-exporter reports the host (16 CPUs, 14.8 GiB) |
| Alert rules | 4 loaded, health ok; the promtool tests pass on the Kubernetes copy too |
| ApiDown drill (scale to 0 and back to 1) | no target after 6 s; pending after 36 s, firing 1 min later (no API target at all); resolved as soon as the new Pod was scraped |
| Memory | monitoring Pods together about 0.6 GiB (Grafana 0.29 GiB + 2 × 0.08 GiB sidecars); node container 3.7–4.3 GiB with 1 API Pod |
| Remove and reinstall | removing the CRDs before the ServiceMonitor broke the next API deploy (as described in section 6); the order in section 6 worked, a deploy without monitoring then left the ServiceMonitor out |

Measured serving cost (memory, latency, threads): [docs/serving_measurements.md](../docs/serving_measurements.md).
