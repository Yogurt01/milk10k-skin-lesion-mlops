# Kubernetes: the API on a local cluster

> **Research model, not a medical device.** Model weights: CC BY-NC 4.0, MILK study team, non-commercial use only.
> They are never part of the image or this chart; the cluster reads them from a folder on your machine.

This folder runs the same API image and the same model package as the Compose stack, as several replicas behind one
Kubernetes Service, on a local minikube cluster. Everything stays on your machine: no registry, no cloud account.

| File | What it is |
|---|---|
| [`namespace.yaml`](namespace.yaml) | The `milk10k` namespace |
| [`milk10k-api/`](milk10k-api/) | Helm chart: Deployment (3 replicas by default) and a ClusterIP Service |
| [`milk10k-api/values.yaml`](milk10k-api/values.yaml) | Defaults with the reasons for each number (threads, memory, probes, rollout) |
| [`values-local.yaml`](values-local.yaml) | Settings for the local cluster (image loaded into the node, model from a host folder) |

## What you need

- Docker and `kubectl` ([Ubuntu guide](../docs/install-ubuntu.md)), minikube and Helm (below). This folder was checked
  with minikube v1.39.0, Helm v4.3.2, kubectl v1.36 and Kubernetes v1.37.0.
- The model package in `artifacts/model_package/final-P1a-kg-onnx` (`make fetch-model`, about 1.4 GB).
- The API image built locally, e.g. `docker build -t milk10k-api:<tag> serving` (any tag except `latest`).
- **Memory.** One replica uses about 1.9 GiB once the model is loaded (plus file cache the kernel can reclaim), the
  cluster itself about 0.6 GiB. Three
  replicas need about 6.3 GiB free, so on a 16 GB laptop stop the Compose stack (`make down`) and Jenkins
  (`make ci-down`) first, and close what you can. Check with `free -h`.

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

## 1. Create the cluster

Run from the repository root. The cluster gets its own minikube profile, `milk10k`, so other clusters are not
touched. The model folder is mounted into the node when the cluster is created (only then).

```bash
minikube start -p milk10k --driver=docker --memory=8g --cpus=12 --kubernetes-version=v1.37.0 \
  --mount-string="$PWD/artifacts/model_package:/mnt/milk10k-model"
kubectl --context milk10k get nodes          # STATUS Ready
```

With the docker driver the mount is a plain bind mount of the host folder, and the node container is capped at 8 GiB.
The node still reports the host's CPUs and memory to Kubernetes, so the replica count and the rollout settings, not
the scheduler, keep the cluster inside that budget.

## 2. Deploy

```bash
TAG=<tag>                                     # the tag you built, e.g. the short git commit
minikube -p milk10k image load milk10k-api:$TAG
kubectl --context milk10k apply -f k8s/namespace.yaml
helm upgrade --install milk10k-api k8s/milk10k-api --kube-context milk10k -n milk10k \
  -f k8s/values-local.yaml --set image.tag=$TAG
kubectl --context milk10k -n milk10k rollout status deployment/milk10k-api
```

Add `--set replicaCount=2` (or 1) if the machine does not have the memory for 3.

What each Pod does when it starts:

1. The `verify-model` init container checks every file of the model package against the sha256 in `manifest.json`
   and logs `sha256 check passed: 18 files match manifest.json`. A mismatch stops the Pod here.
2. The API starts and loads the 15 ONNX sessions. Until that is done `/ready` answers 503, so the startup probe
   keeps the Pod out of the Service.
3. Once `/ready` is 200 the Pod is Ready and receives requests.

Expected warning: the namespace warns that the Pod would violate the `restricted` Pod Security Standard because of the
`hostPath` volume. The model folder comes from the node, which needs a hostPath volume; the Baseline and Restricted
standards forbid those, so the namespace warns instead of enforcing. Everything else in the Pod meets Restricted.

## 3. Verify

```bash
kubectl --context milk10k -n milk10k get deploy,pods -o wide        # READY 3/3, one IP per Pod
kubectl --context milk10k -n milk10k logs deploy/milk10k-api -c verify-model
kubectl --context milk10k -n milk10k describe pod -l app.kubernetes.io/name=milk10k-api | grep -E "Liveness|Readiness|Startup|Limits|Requests" -A2
```

**Smoke test and parity** through a port-forward (from the repository root, with the serving environment from
`make setup`):

```bash
kubectl --context milk10k -n milk10k port-forward service/milk10k-api 18000:8000 &
python3 scripts/smoke_test.py --base-url http://127.0.0.1:18000
make test-container API_BASE_URL=http://127.0.0.1:18000           # 6 sample lesions, tolerance 0.002
kill %1
```

**Requests spread over the Pods.** `kubectl port-forward` to a Service picks *one* Pod and sends every connection to
it, so it cannot show load balancing. Send requests to the Service from inside the cluster instead, one new connection
each, and count them in each Pod's log (`/openapi.json` is logged with its request id; probes are not):

```bash
POD=$(kubectl --context milk10k -n milk10k get pods -o name | head -1)
kubectl --context milk10k -n milk10k exec $POD -c api -- python -c '
import urllib.request as u
for i in range(20):
    r = u.Request("http://milk10k-api:8000/openapi.json", headers={"X-Request-ID": f"spread-{i}", "Connection": "close"})
    u.urlopen(r, timeout=10).read()'
for p in $(kubectl --context milk10k -n milk10k get pods -o name); do
  echo "$p $(kubectl --context milk10k -n milk10k logs $p -c api | grep -c '\[spread-')"
done
```

**Self-healing.** Delete one Pod; the Deployment starts a replacement and the Service keeps answering from the others:

```bash
kubectl --context milk10k -n milk10k delete pod <one of the pods>
kubectl --context milk10k -n milk10k get pods -w
```

**Rolling update.** Build and load a new tag, then run the same `helm upgrade` with it. Pods are replaced one at a
time (`maxSurge: 0`, `maxUnavailable: 1`): a fourth 2.5 GiB Pod would not fit the memory budget, so one replica
fewer serves while each Pod is swapped. Each old Pod waits 5 s before it stops so the Service stops sending it new
connections first.

```bash
helm history milk10k-api --kube-context milk10k -n milk10k
helm rollback milk10k-api <revision> --kube-context milk10k -n milk10k     # back to an earlier release if needed
```

Use Helm, not `kubectl rollout undo`, to go back: Helm keeps the release history, and a change made behind its back
is overwritten by the next `helm upgrade`.

## 4. Stop or remove

```bash
minikube stop -p milk10k        # frees the memory; keeps the cluster, the loaded images and the mount
minikube start -p milk10k       # start it again later
helm uninstall milk10k-api --kube-context milk10k -n milk10k        # remove the API only
minikube delete -p milk10k      # remove the whole cluster
```

## Design choices

| Topic | Choice | Reason |
|---|---|---|
| Cluster | minikube, docker driver, own profile | The course's local cluster; image loading and host mounts are built in |
| Packaging | Helm chart, values per environment | Replica count, image tag and model source are values, so the same chart can serve a cloud cluster later |
| Image | Loaded into the node, pinned tag, `pullPolicy: Never` locally | No registry needed; the chart refuses an empty tag or `latest` |
| Model | Host folder mounted read-only (hostPath), sha256 check in an init container | 1.4 GB is not downloaded again for every Pod, restart and rollout; the weights stay out of the image |
| Threads | 4 ONNX Runtime threads, 1 running prediction per Pod (queue of 8) | 3 Pods use 12 of 16 CPU threads; a second running prediction added only about 6% throughput in measurements and more memory |
| Memory | request = limit = 2.5 GiB per Pod | Measured peak 2.1 GiB in a container, plus room for queued uploads |
| Probes | startup and readiness on `/ready`, liveness on `/health` (timeout 5 s, 6 failures) | `/ready` is 200 only once the model is loaded; `/health` does no work and stays fast under load, and the generous liveness settings never restart a busy Pod |
| Security | non-root (uid 10001), all capabilities dropped, no privilege escalation, seccomp `RuntimeDefault`, read-only root filesystem with an `emptyDir` for `/tmp`, no service account token | The API needs none of these privileges; `/tmp` holds large uploads |

Measured serving cost (memory, latency, threads): [docs/serving_measurements.md](../docs/serving_measurements.md).
