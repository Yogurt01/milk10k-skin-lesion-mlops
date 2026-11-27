# Cloud: the API and its monitoring on GKE, provisioned with Terraform

> **Research model, not a medical device.** Model weights: CC BY-NC 4.0, MILK study team, non-commercial use only.
> They are never in the image or in git; during the test they sit in a private bucket that is deleted with everything
> else.

This folder runs the same Helm chart as the local cluster ([`k8s/`](../k8s/README.md)) on a short-lived Google
Kubernetes Engine cluster: the API with 3 replicas, Prometheus and Grafana from kube-prometheus-stack, the image in
Artifact Registry and the model package in Cloud Storage. Terraform creates and destroys the cloud resources; the
whole test is meant to run for a few hours and then be removed, project included.

**Status (November 2026): prepared and checked offline, not yet run on Google Cloud.** Every check that needs no
account passed (see [Offline checks](#offline-checks)); everything marked **[verify]** below is a figure or a
behaviour that only the run can confirm.

| File | What it is |
|---|---|
| [`terraform/versions.tf`](terraform/versions.tf) | Terraform and google provider pins, local state, provider settings |
| [`terraform/variables.tf`](terraform/variables.tf) | Inputs with checks (no world-open network, zonal location) |
| [`terraform/main.tf`](terraform/main.tf) | APIs, registry, private model bucket, node identity, Workload Identity, cluster, node pool |
| [`terraform/budget.tf`](terraform/budget.tf) | Budget alert on the project |
| [`terraform/outputs.tf`](terraform/outputs.tf) | Registry path, bucket name, cluster name (read by `scripts/gke.sh`) |
| [`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example) | Template of `terraform.tfvars` (gitignored: project and billing IDs, your IP) |
| [`terraform/.terraform.lock.hcl`](terraform/.terraform.lock.hcl) | Provider hashes, committed so every init installs the same build |
| [`terraform/.tflint.hcl`](terraform/.tflint.hcl) | tflint with the Google ruleset (static rules only) |
| [`../k8s/values-gke.yaml`](../k8s/values-gke.yaml) | Chart values for GKE: image from the registry, model from the bucket, 1.5 CPU per Pod |
| [`../k8s/monitoring/values-kps-gke.yaml`](../k8s/monitoring/values-kps-gke.yaml) | The one monitoring change for GKE (kube-dns instead of CoreDNS) |
| [`../scripts/gke.sh`](../scripts/gke.sh) | The steps behind `make gke-*` (explicit kube context and project on every call) |

## What gets created

| Resource | Why | Cost while it exists |
|---|---|---|
| GKE Standard cluster, **zonal** (`us-central1-a`) | Control plane | $0.10/h management fee, covered by the GKE free tier (one zonal or Autopilot cluster per billing account) |
| Node pool: **2 × e2-standard-4** (8 vCPU, 32 GB), 50 GB pd-balanced each, fixed size | Runs 3 API Pods + monitoring | 2 × $0.134/h + disks |
| Artifact Registry repository `milk10k` (Docker, immutable tags) | The API image (about 640 MB) | 0.5 GB free, then $0.10/GB-month: cents |
| Cloud Storage bucket `<project>-milk10k-model`, **private** | The model package (1.4 GB) for the Pods | cents per month |
| Service account for the nodes (log/metric writer, registry reader) | Instead of the Compute Engine default account (Editor) | free |
| Workload Identity binding: the API's Kubernetes ServiceAccount may read the bucket | No key file anywhere | free |
| Budget with alerts at 25 / 50 / 90 / 100 % and on the forecast | Early warning (a budget never stops spending) | free |
| Created by Kubernetes, not Terraform: a load balancer only while the API is exposed | The outside-the-cluster check | $0.025/h |

## Design choices

| Topic | Choice | Why |
|---|---|---|
| Mode | **Standard, zonal**, fixed node count, upgrade surge 0 | The free tier covers a zonal cluster's fee; nodes and quota use are predictable; node-exporter (host metrics) is allowed. Autopilot rejects node-exporter (host network, host PID, host paths) and sizes its own nodes. A surge node during an upgrade would be a 9th vCPU |
| Size | 8 vCPU, the likely CPU quota of a Free Trial project, which cannot be raised | 3 Pods at the chart's 3 CPU would need 9 vCPU; on GKE each Pod requests **1.5 CPU** (limit 2) and runs **2 ONNX Runtime threads**, memory unchanged (2.5 GiB). Estimated latency about 1.0 s per lesion **[verify]** |
| Image | Built on your machine, pushed with `docker push` through gcloud's credential helper | No change to Jenkins or Compose; no key file; tags are immutable in the registry |
| Model | Private bucket + Workload Identity + an init container (`gcloud storage cp`, official Google Cloud CLI image pinned by digest) into the Pod's own `emptyDir`, then the existing sha256 check of every file | Same "mounted read-only, never in the image" rule as locally; the init container's CPU is not added to the Pod's steady request (a FUSE sidecar would be); the package exists only in the bucket and in running Pods |
| Bucket | Uniform bucket-level access, public access prevention **enforced**, no versioning, **no soft delete**, `force_destroy` | No object can be made public; `terraform destroy` removes the bucket with the package inside and leaves no recoverable copy |
| Access | ClusterIP + `kubectl port-forward` by default; for the outside check only, a `LoadBalancer` limited to **your IP/32** (`loadBalancerSourceRanges`), removed right after | No open public endpoint. The chart refuses a LoadBalancer without source ranges or with `0.0.0.0/0`. No Ingress controller (the community Ingress NGINX is retired) |
| Control plane | Authorized networks = your IP/32 (`admin_cidrs`) | kubectl and Helm work only from your machine |
| Monitoring | kube-prometheus-stack, the same pinned chart and values file as locally, plus a GKE overlay; Google's managed Prometheus collection off | Same dashboards and alerts as on minikube; the managed collectors would cost CPU in an 8 vCPU budget |
| State | Terraform's **local** backend (`terraform/terraform.tfstate`, gitignored, never printed) | One operator, a few hours, then destroy. Kept until the leftover check is empty; the teardown below also works without it |
| Budget | In Terraform **and** a hand-made one created first in the console | The console budget exists before anything else is created; both alert on the cost **without credits** (with credits the net cost is 0 on a trial, and nothing would ever fire) |

## Cost

Prices for `us-central1`, on demand, November 2026. VM prices from a third-party mirror of Google's price list
(gcloud-compute.com, updated 2026-11-22) **[verify]**; the GKE fee, free tier, load balancer and registry prices
from Google's pricing pages; disk and external IP prices could not be read (assumed small) **[verify]**.

| Item | Per hour |
|---|---:|
| 2 × e2-standard-4 | $0.268 |
| 2 × 50 GB pd-balanced boot disk | about $0.014 **[verify]** |
| 2 node external IPv4 addresses | about $0.01 **[verify]** |
| Cluster management fee ($0.10/h) | $0 (free tier credit) |
| Load balancer forwarding rule, only while exposed | $0.025 |
| Registry and bucket (a few GB) | under $0.001 |
| **Total** | **about $0.29/h, $0.32/h with the load balancer** |

| How long | Cost |
|---|---:|
| One test session (about 3 hours up) | about $1–2 |
| A day left running | about $7–8 |
| **Forgotten for 30 days** | **about $210–230**. On a Free Trial that was never upgraded, the $300 credit would run out after about 39 days; then Google stops the resources and nothing is charged to the card. On an upgraded (paid) account the card is charged after the credit. |

Alternatives priced the same way: the same nodes on Spot about $0.18/h (can be preempted during the demo);
Autopilot with the same Pod requests about $0.26–0.29/h **[verify]**; a regional cluster with one node per zone
about $0.51/h (12 vCPU, above the quota, and the fee is not covered).

## Quotas and other risks

- **CPU quota.** Free Trial projects are reported to get 8 vCPU per region **[verify]**, and a trial cannot ask for
  more. 2 × e2-standard-4 = exactly 8. Step 6 of the runbook reads the real values before anything is created; with
  less than 8, stop (upgrading the account is the owner's decision).
- **SSD quota.** pd-balanced disks count as SSD: 2 × 50 GB = 100 GB (the default pool's temporary disk is
  pd-standard). **In-use IP addresses:** 2 nodes + 1 load balancer = 3.
- **Scheduling.** About 6.4 vCPU are left for Pods after GKE's own Pods **[verify]**; the API needs 3 × 1.5 and
  monitoring about 0.45. A rollout replaces one Pod at a time (no extra Pod).
- **Default node pool warning.** `terraform validate` warns that `remove_default_node_pool` is deprecated for clusters
  from GKE 1.36.4-gke.1469000 on. The replacement (no default pool at all) fails on older versions, and the default
  version of the release channel can only be read with the account, so the setting that works on every version stays;
  step 8 checks the version **[verify]**.
- **Workload Identity without a mounted token.** The Pods mount no ServiceAccount token; the GKE metadata server is
  expected to answer the init container without one **[verify]** (if not: `serviceAccount` automount for the init
  container only).
- **Init image as non-root.** The Google Cloud CLI image runs as root by default; here it runs as uid 10001 with a
  read-only root and its configuration in a memory volume **[verify]**.
- **Cost.** A cluster that is not torn down keeps costing about $7 a day. The budget alerts warn; only the teardown stops it.
- **Credentials on this machine.** gcloud may already hold another login. Every gcloud call here runs in a dedicated
  configuration (`milk10k`) with an explicit `--project`, and every kubectl/Helm call names the context `milk10k-gke`.

## Offline checks

None of these needs a Google Cloud account; they were run before the first apply (November 2026).

| Check | Command | Result |
|---|---|---|
| Format, provider install, validation, lint | `make tf-check` | passed (Terraform 1.17.2, google provider 8.12.0, tflint 0.66.0 + Google ruleset 0.41.0); one deprecation warning (see above) |
| The checks can fail | an unformatted file and an invalid machine type in a copy | `fmt -check` exit 3, tflint exit 2 |
| Chart with the GKE values, schema check | `make gke-render` | 4 objects (ServiceAccount, Service, Deployment, ServiceMonitor), valid against the Kubernetes v1.34.12 schemas (newest published for kubeconform); CRDs skipped |
| Monitoring with the GKE overlay | `helm template` of the pinned chart with both values files | renders; only the CoreDNS objects are gone; kubeconform: no invalid object |
| Local renders unchanged | `helm template` of 7 local combinations before and after the GKE additions | byte-identical |
| Chart guards and Terraform safety settings | `pytest tests/test_gke_config.py` | 16 passed: no hostPath on GKE, fetch before sha256 check, LoadBalancer without ranges or with `0.0.0.0/0` refused, private bucket with `force_destroy`, budget without credits, placeholders only |

## What you need

- The local tools of the [Ubuntu guide](../docs/install-ubuntu.md) and of [`k8s/`](../k8s/README.md#what-you-need)
  (Docker, kubectl, Helm), and:
  - **Terraform 1.17.2**: the `linux_amd64` zip from `releases.hashicorp.com/terraform/1.17.2/`, checked against
    `terraform_1.17.2_SHA256SUMS`, whose signature (`.sig`) is checked with HashiCorp's key (fingerprint
    `C874 011F 0AB4 0511 0D02 1055 3436 5D94 72D7 468F`); the binary into `~/.local/bin`.
  - **gcloud** 593.0.0 with `gke-gcloud-auth-plugin` (Google's apt repository; Ubuntu packages `google-cloud-cli`
    and `google-cloud-cli-gke-gcloud-auth-plugin`).
  - Optional checks: **kubeconform** v0.8.0 and **tflint** v0.66.0 (GitHub releases, checked against the release's
    checksum file).
- The model package, verified: `scripts/fetch_model.sh --check artifacts/model_package/final-P1a-kg-onnx`.
- The API image built locally with a pinned tag: `docker build -t milk10k-api:<tag> serving`.

## Run it (the first run is also the check of this guide)

**Before registering** (the trial's 90 days start at sign-up):

- [ ] Offline checks pass on the commit you will use: `make tf-check gke-render test lint`.
- [ ] Model package verified and the image built (above); write the tag down.
- [ ] Sample lesions present for the parity test, if you have the dataset: `make sample SAMPLE_DATA_DIR=<dataset>`.
- [ ] Your public IPv4 address known (it becomes `admin_cidrs` and the load balancer's source range; never commit it).
- [ ] A free afternoon: the run takes about 2–3 hours, the teardown 20 minutes.

Who does what: **you** = the account owner in the browser or the terminal; **agent** = commands run from this
repository on your go. Times are estimates **[verify]**.

| # | Who | Step | Command or place | Time | Cost |
|---:|---|---|---|---|---|
| 1 | you | Register the Free Trial; note the billing account's **currency** | console.cloud.google.com | 10 min | $0 |
| 2 | you | **Budget first**: Billing → Budgets & alerts → create; all projects; amount e.g. $20 (or the same in your currency); thresholds 25 / 50 / 90 / 100 %; untick credits so the alert counts the gross cost **[verify: wording in the console]** | console | 5 min | $0 |
| 3 | you | A project only for this test, linked to the billing account | console, or `gcloud projects create <id>` + `gcloud billing projects link <id> --billing-account=<account>` | 5 min | $0 |
| 4 | you | Own gcloud configuration and logins | `gcloud config configurations create milk10k` → `gcloud auth login` → `gcloud config set project <id>` → `gcloud auth application-default login` → `gcloud auth application-default set-quota-project <id>` | 5 min | $0 |
| 5 | you | Enable the APIs | `gcloud services enable container.googleapis.com artifactregistry.googleapis.com storage.googleapis.com compute.googleapis.com iam.googleapis.com billingbudgets.googleapis.com cloudresourcemanager.googleapis.com serviceusage.googleapis.com` then `gcloud services list --enabled` | 3 min | $0 |
| 6 | you | **Read the quotas; stop below 8 CPUs, 100 GB SSD or 3 addresses** | `gcloud compute regions describe us-central1 --format="table(quotas.metric,quotas.limit,quotas.usage)" --flatten=quotas \| grep -E "^(CPUS\|SSD_TOTAL_GB\|IN_USE_ADDRESSES) "` and `gcloud compute project-info describe --format="table(quotas.metric,quotas.limit)" --flatten=quotas \| grep CPUS_ALL_REGIONS` | 2 min | $0 |
| 7 | you | `terraform.tfvars` from the example, mode 600: project ID, billing account, `admin_cidrs = ["<your IP>/32"]`, budget in your currency | `cd deploy/terraform && cp terraform.tfvars.example terraform.tfvars && chmod 600 terraform.tfvars` | 3 min | $0 |
| 8 | agent | Preflight, init, plan (review the resource count), apply | `make gke-preflight tf-init tf-plan`, then `make tf-apply`; before: `gcloud container get-server-config --zone us-central1-a --format="value(channels)"` (the version question above) | 10–15 min | from here on about $0.29/h |
| 9 | agent | Cluster access; second plan must show no changes | `make gke-credentials` (`kubectl --context milk10k-gke get nodes`), `make tf-plan` → "No changes" | 3 min | |
| 10 | agent | Image to the registry | `make gke-registry-login gke-push TAG=<tag>` | 5–15 min (upload speed) | |
| 11 | agent | Model package to the private bucket | `make gke-model-upload` | 5–20 min (1.4 GB upload) | |
| 12 | agent | Prometheus + Grafana | `make gke-monitoring` | 5 min | |
| 13 | agent | API, 3 replicas; each Pod logs its sha256 check | `make gke-deploy TAG=<tag>` | 5–10 min | |
| 14 | agent | Smoke test and sample-lesion parity through a port-forward | `make gke-test` | 2 min | |
| 15 | agent | Outside the cluster: load balancer for your IP only, `curl /health`; no Ingress NGINX; then remove the load balancer | `make gke-expose CIDR=<your IP>/32`, `kubectl --context milk10k-gke get pods -A -l app.kubernetes.io/name=ingress-nginx` (empty), `make gke-unexpose` | 5 min | +$0.025/h |
| 16 | agent + you | Grafana and Prometheus by port-forward; you take the screenshots | `kubectl --context milk10k-gke -n monitoring port-forward --address 127.0.0.1 svc/kps-grafana 13000:80` (user admin, password in `k8s/secrets/grafana-admin-password`), Prometheus on 19090 as in `k8s/` | 20 min | |
| 17 | agent | **Teardown** (next section) | `make gke-down` | 15–20 min | stops the cost |
| 18 | you | **Delete the project** and check | `gcloud projects delete "$GKE_PROJECT"` | 2 min | $0 |

### Screenshots (you take them; check each for secrets before committing)

Run every command with `$GKE_PROJECT` instead of the literal project ID, so no screenshot shows it; do not show the
billing account ID, your e-mail, your IP or the Grafana password.

| File in `docs/images/` | What it shows | Step |
|---|---|---:|
| `gke-apis.png` | `gcloud services list --enabled` with container, artifactregistry, storage | 5 |
| `gke-budget.png` | The budget page: amount and thresholds (no account ID) | 2 / 8 |
| `gke-terraform-apply.png` | End of `terraform apply`: "Apply complete! Resources: … added" | 8 |
| `gke-plan-no-changes.png` | The second plan: "No changes" | 9 |
| `gke-nodes.png` | `kubectl --context milk10k-gke get nodes`: 2 Ready | 9 |
| `gke-image.png` | The image list printed by `make gke-push` (name, tag, time) | 10 |
| `gke-verify-model.png` | `make gke-deploy` end: one "sha256 check passed: 18 files" line per Pod | 13 |
| `gke-pods.png` | `kubectl --context milk10k-gke -n milk10k get deploy,pods`: 3/3 | 13 |
| `gke-loadbalancer.png` | `make gke-expose`: external IP and `{"status":"ok"}`; the empty Ingress NGINX query | 15 |
| `gke-grafana-nodes.png` | Grafana "Node Exporter / Nodes" for a GKE node | 16 |
| `gke-grafana-pods.png` | Grafana "Kubernetes / Compute Resources / Namespace (Pods)", namespace milk10k, 3 Pods | 16 |
| `gke-grafana-api.png` | "MILK10k: API on Kubernetes" with 3 Pods (optional) | 16 |
| `gke-prom-targets.png` | Prometheus targets: `api` 3/3 up (optional) | 16 |
| `gke-teardown.png` | `make gke-down` end: every leftover count 0 | 17 |
| `gke-project-deleted.png` | `gcloud projects describe "$GKE_PROJECT" --format="value(lifecycleState)"`: DELETE_REQUESTED | 18 |

### Evidence for the grading row "Use the cloud"

| Claim | Evidence |
|---|---|
| GKE cluster provisioned with Terraform | `terraform/`, `gke-terraform-apply.png`, `gke-plan-no-changes.png`, `gke-nodes.png` |
| API with 3 replicas on GKE (the same chart) | `k8s/values-gke.yaml`, `gke-pods.png`, `gke-verify-model.png` |
| Prometheus + Grafana via Helm on GKE | `k8s/monitoring/values-kps-gke.yaml`, `gke-grafana-nodes.png`, `gke-grafana-pods.png` |
| Image in Artifact Registry, model in Cloud Storage with Workload Identity (no key file) | `gke-image.png`, `gke-verify-model.png`, `terraform/main.tf` |
| Reachable from outside, restricted, no Ingress NGINX | `gke-loadbalancer.png` |
| Cost controlled and torn down | budget code + `gke-budget.png`, `gke-teardown.png`, `gke-project-deleted.png` |

## Teardown

`make gke-down` does, in this order:

1. `helm uninstall` of the API and of kube-prometheus-stack, through the cluster: a load balancer or disk created by
   Kubernetes is not in the Terraform state and would survive `terraform destroy` (and stay billed).
2. Waits until no forwarding rule is left in the project.
3. `terraform destroy` (asks for `yes`): cluster, node pool, registry with its images, **bucket with the model**,
   service account, bindings, budget.
4. Removes the `milk10k-gke` kube context and runs the leftover check.

**Leftover check, also without the state** (state lost, or anything created by hand):

```bash
export CLOUDSDK_ACTIVE_CONFIG_NAME=milk10k GKE_PROJECT=<project id>
make gke-leftovers
```

It lists, by project: GKE clusters, VM instances, disks, snapshots, forwarding rules, target pools, backend services,
external IP addresses, registries in the region and buckets. A list that fails counts as a failure, never as zero.
Anything listed is removed with the matching `gcloud … delete`, or by step 5.

5. **Last guarantee: delete the project.** `gcloud projects delete "$GKE_PROJECT"` shuts it down at once (billing
   and all resources stop) and removes it after 30 days. Check: `gcloud projects describe "$GKE_PROJECT"
   --format="value(lifecycleState)"` → `DELETE_REQUESTED`; `make gke-leftovers` then says so.
6. Local clean-up once the above is confirmed: `rm deploy/terraform/terraform.tfstate*`, `gcloud auth
   application-default revoke`, `gcloud config configurations activate default` and
   `gcloud config configurations delete milk10k`. The budget alerts end with the project.
