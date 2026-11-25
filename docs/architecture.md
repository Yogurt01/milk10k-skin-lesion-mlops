# Architecture

> **Research model, not a medical device.** Its output is not a diagnosis and must not be used for clinical decisions.

Status as of 2026-11-25: the API, its model package and the monitoring stack run with one command (`make up`); a
local Jenkins, configured as code, runs the pipeline test → build → deploy → smoke → parity → Kubernetes on every
push to `main`; Prometheus has 4 alert rules. The API also runs as 3 replicas on a local minikube cluster from a Helm
chart (section 6, `make k8s-deploy`), and the pipeline deploys each build's image there (1 replica next to the Compose
stack). Prometheus and Grafana run on that cluster from a pinned Helm chart (kube-prometheus-stack) and scrape each
API Pod; checked with 1, 2 and 3 API replicas (3 replicas and monitoring together exceed the 16 GB laptop, so they run
together only briefly, with Grafana stopped). The cloud parts are planned. The diagrams are derived from the code and are updated whenever a part is added.

Styling in every diagram:
- **Solid** box or arrow: implemented.
- **Dashed**: planned.
- **Orange, dotted**: optional extension beyond the course requirements, planned after the required items.

## 1. Components

```mermaid
flowchart LR
    client(["Client<br/>curl, serving/client.py"])

    subgraph serving["Serving"]
        nginx["nginx reverse proxy<br/>official image in Compose<br/>optional extension"]
        subgraph container["API container, Compose service api"]
            api["FastAPI app, api.py<br/>/health /ready /predict<br/>request id, concurrency limit"]
            metrics["/metrics endpoint, metrics.py"]
            tracing["OpenTelemetry SDK, tracing.py<br/>server span + stage spans"]
            pred["milk10k_serving Predictor<br/>decode, preprocess, assemble"]
            ort["ONNX Runtime, CPU<br/>15 member sessions"]
        end
        pkg[("Model package on the host<br/>15 ONNX files + manifest.json with sha256<br/>not in the image")]
    end

    release[("GitHub Release model-v1.0.0<br/>model package assets")]

    subgraph obs["Observability, Compose services on the internal network monitoring"]
        prom["Prometheus v3.16, alert rules"]
        graf["Grafana 13.3, provisioned<br/>datasource + dashboard"]
        nodeexp["node-exporter v1.12<br/>host CPU, memory, disk"]
        cadv["cAdvisor v0.60<br/>per-container + host network"]
        jaeger["Jaeger v2.23, in-memory"]
    end
    host[("Host: /proc, /sys, / read-only<br/>Docker + containerd sockets")]
    browser(["Browser on the same machine<br/>127.0.0.1 :9090 :3000 :16686"])

    subgraph cicd["CI/CD and deployment"]
        repo["GitHub repository"]
        jenkins["Jenkins 2.580.2 LTS in Compose<br/>JCasC + Job DSL, host Docker socket"]
        registry[("Image registry<br/>with the cloud deployment")]
        compose["Compose deploy of the local image<br/>milk10k-api:build-sha7, smoke + parity"]
        mk["minikube, Helm chart k8s/milk10k-api<br/>3 replicas, make k8s-deploy"]
        kps["kube-prometheus-stack 93.10 via Helm<br/>Prometheus + Grafana, namespace monitoring<br/>make k8s-monitoring"]
        tf["Terraform"]
        gke["GKE, Helm chart, 3 replicas"]
        gcs[("GCS copy of the model package")]
        kpsg["kube-prometheus-stack on GKE"]
    end

    client -->|"HTTP 127.0.0.1:8000, make up or make serve"| api
    client -.->|optional| nginx
    nginx -.-> api
    api --> pred
    pred --> ort
    pkg -->|"read-only bind mount /models, sha256 verified, loaded once at startup"| ort
    release -->|"scripts/fetch_model.sh: curl, sha256 per file"| pkg
    api --> metrics
    api --> tracing
    prom -->|"scrape every 15 s, DNS discovery of api:8000"| metrics
    prom -->|scrape| nodeexp
    prom -->|scrape| cadv
    host -->|read-only mounts| nodeexp
    host -->|read-only mounts| cadv
    graf -->|"datasource prometheus:9090"| prom
    tracing -->|"OTLP/HTTP jaeger:4318, batched, fail-soft"| jaeger
    browser --> graf
    browser --> prom
    browser --> jaeger

    repo -->|"pollSCM H/5"| jenkins
    jenkins -->|"docker build + compose up through the host socket"| compose
    jenkins -.->|"push, with the cloud deployment"| registry
    pkg -->|"hostPath read-only, sha256 check in an init container"| mk
    jenkins -->|"image into the node + helm upgrade, 1 replica"| mk
    registry -.-> gke
    kps -->|"ServiceMonitor: scrape each API Pod every 15 s"| mk
    tf -.-> gke
    tf -.-> gcs
    release -.->|copy| gcs
    gcs -.->|initContainer| gke
    gke -.- kpsg

    classDef implemented fill:#e6f4ea,stroke:#1e7e34,stroke-width:2px,color:#111111
    classDef planned fill:#ffffff,stroke:#666666,stroke-width:1px,stroke-dasharray:5 5,color:#111111
    classDef ext fill:#fff4e5,stroke:#b35c00,stroke-width:1px,stroke-dasharray:2 4,color:#111111
    class api,pred,ort,pkg,release,repo,metrics,tracing,prom,graf,nodeexp,cadv,jaeger,host,jenkins,compose,mk,kps implemented
    style container fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    style obs fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    class registry,tf,gke,gcs,kpsg planned
    class nginx ext
```

| Component | Status | Evidence |
|---|---|---|
| Client | external; example client implemented | Any HTTP client; [`serving/client.py`](../serving/client.py) (standard library only) |
| FastAPI app (`/health`, `/ready`, `/predict`) | implemented | [`serving/src/milk10k_serving/api.py`](../serving/src/milk10k_serving/api.py); request ids and access log; started with `make serve` ([Makefile](../Makefile)) |
| Predictor (decode, preprocessing, assembly) | implemented | [`predictor.py`](../serving/src/milk10k_serving/predictor.py), [`preprocess.py`](../serving/src/milk10k_serving/preprocess.py); cap on decoded image pixels |
| ONNX Runtime, 15 member sessions | implemented | [`package.py`](../serving/src/milk10k_serving/package.py) (`load_package`, `Ensemble`) |
| Model package with sha256 manifest | implemented | Built by [notebook 08](../notebooks/08_export_for_deployment.ipynb) ; verified at startup by `verify_package`. Not in git and not in the image: mounted read-only into the container |
| GitHub Release `model-v1.0.0` | implemented | 19 assets; [`scripts/fetch_model.sh`](../scripts/fetch_model.sh) (`make fetch-model`) downloads them without a token, pins the manifest's sha256 and checks every file |
| `/metrics` (API and model metrics) | implemented | [`metrics.py`](../serving/src/milk10k_serving/metrics.py): requests by route template / method / status, latency histogram, running and waiting predictions, 503-busy counter, model ready / load time / info, stage durations, P(malignant) histogram; labels bounded (unmatched paths → `unmatched`) |
| Prometheus, node-exporter, cAdvisor | implemented | [`monitoring/prometheus/`](../monitoring/prometheus/) (scrape config, 4 alert rules with promtool unit tests in `alert-rules.test.yml`: ApiDown incl. a missing target, ApiModelNotReady, ApiHighErrorRate, ApiHighLatencyP95); all 4 targets up. node-exporter and cAdvisor only on the internal network, host folders read-only; cAdvisor also reads the Docker and containerd sockets to name containers, which gives it control over Docker (accepted for a local stack) |
| Grafana (machine resources and API dashboard) | implemented; dashboard checked by eye (2026-11-15) | [`monitoring/grafana/`](../monitoring/grafana/): provisioned datasource and one dashboard (host, containers, API rows; every panel query returns data through the Prometheus API). Admin password only from `.env`; analytics and update checks off |
| Jaeger v2 via the OTLP exporter | implemented; UI and span tags checked by eye (2026-11-15) | [`tracing.py`](../serving/src/milk10k_serving/tracing.py): OTLP/HTTP to `jaeger:4318` when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, FastAPI instrumentation (probes and `/metrics` excluded), stage spans; [`monitoring/jaeger/config.yaml`](../monitoring/jaeger/config.yaml) (in-memory, 10,000 traces) |
| nginx (reverse proxy, load balancing in a Compose profile, GET-only cache exercise) | optional extension, planned after the required items | Official `nginx` image in Compose only, never the retired community Ingress NGINX on Kubernetes |
| GitHub repository | implemented | Public repository |
| Jenkins in Compose | implemented | [`ci/`](../ci/README.md): `jenkins/jenkins:2.580.2-lts-jdk21` by digest + Docker CLI, Buildx, Compose, make, uv; 69 pinned plugins; JCasC + Job DSL (no setup wizard, no anonymous access, UI on 127.0.0.1:8081); host Docker socket (root-equivalent, documented); `make ci-up` / `make ci-down`. [`Jenkinsfile`](../Jenkinsfile): section 3 |
| Image registry | planned (cloud) | No registry for the local pipeline: the deploy runs on the same machine, so the pipeline tags `milk10k-api:<build>-<sha7>` locally and keeps the newest 3. A registry comes with GKE, created by Terraform |
| API image | implemented | [`serving/Dockerfile`](../serving/Dockerfile): pinned `python:3.12.16-slim-trixie` (tag + digest), pinned requirements then the package with its `api` extra, no torch, uid 10001, healthcheck on `/ready` with the Python standard library, uvicorn without `--workers` |
| Docker Compose stack | implemented (api + 5 monitoring services) | [`compose.yaml`](../compose.yaml), `make up` / `make down`: images pinned by tag + digest, model mounted read-only, UIs on 127.0.0.1 only, internal network `monitoring` for scrapes and OTLP, every service `cap_drop: ALL` + `no-new-privileges` + a memory limit without swap + json-file log rotation (10 MB × 3), Prometheus starts with the API without waiting for its health, so an API that cannot load its model still shows up in monitoring (`milk10k_model_ready` 0); Grafana waits for a healthy Prometheus; checked with one command from a fresh clone |
| Compose deploy by the pipeline | implemented | Deploy stage: `docker compose up -d --no-build --wait` with `API_IMAGE_TAG`, model from the CI copy (read-only), Grafana password from a Jenkins credential; then smoke test and HTTP parity against `http://api:8000`. No automatic rollback: a failed deploy is rolled back by hand ([ci/README.md](../ci/README.md)) |
| minikube with a Helm chart, 3 replicas | implemented; checked with 3 replicas | [`k8s/`](../k8s/README.md): Helm chart `milk10k-api` (Deployment + ClusterIP Service, `replicaCount: 3`, a ServiceMonitor when the cluster has the CRD), local values, namespace; image loaded into the node with a pinned tag; model from the host folder, read-only, sha256-checked by an init container; section 6. The pipeline's Kubernetes stage deploys each build with [`scripts/k8s_deploy.sh`](../scripts/k8s_deploy.sh) (1 replica, rollback on failure; [ci/README.md](../ci/README.md#the-kubernetes-stage)) |
| kube-prometheus-stack via Helm, local cluster | implemented; checked with 1, 2 and 3 API replicas (3 only briefly, Grafana stopped: memory) | [`k8s/monitoring/`](../k8s/monitoring/): chart 93.10.0 pinned by OCI digest, values file with only the changed keys (no Alertmanager, no control-plane scrapes minikube cannot serve, 1 day retention, memory limits), Grafana login from a gitignored file via a Secret, UIs by port-forward on 127.0.0.1; the API chart's ServiceMonitor (one target per Pod, job `api`), the API dashboard as a ConfigMap, the 4 alert rules as a PrometheusRule generated from `alert-rules.yml`; section 6 |
| kube-prometheus-stack on GKE | planned (bonus) | with the cloud deployment |
| Terraform, GKE, GCS copy of the package | planned (bonus) | torn down right after the demo to limit cost |

## 2. `POST /predict` request path

Derived from [`api.py`](../serving/src/milk10k_serving/api.py). The status codes and the steps that raise them are
listed in that file's docstring and in the OpenAPI `responses`. Form parsing and field validation run inside FastAPI,
before the body of the `predict` handler. `predict` is an `async def` that waits for a slot of the
`PredictionLimiter` in the event loop (a waiting request holds no thread), then runs the blocking work in the
threadpool, so at most `MAX_CONCURRENT_PREDICTIONS` threads do inference. `/health`, `/ready` and
`get_predictor` are `async def` and never wait for the threadpool.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant U as uvicorn / FastAPI
    participant G as get_predictor
    participant Q as PredictionLimiter
    participant R as read_limited
    participant M as metadata_errors
    participant P as Predictor
    participant T as to_response
    participant L as model-loader thread
    participant J as Jaeger (OTLP)

    Note over U,L: Startup: the lifespan hook calls ModelState.start() and returns at once
    U->>L: start background load
    L->>L: verify_package: size and sha256 of every manifest file
    L->>L: create 15 ONNX Runtime CPU sessions
    alt load succeeds
        L-->>U: predictor set, then status ready (OpenAPI schema rebuilt with the vocabulary)
    else load fails
        L-->>U: status failed, error with absolute paths redacted (full detail in the server log)
    end

    C->>U: POST /predict (multipart form)
    U->>U: OpenTelemetry server span starts (only when tracing is on)
    U->>U: request_context middleware: request id from X-Request-ID or generated
    opt Content-Length above 2 x MAX_UPLOAD_MB + 1 MB
        U-->>C: 413 from the limit_request_size middleware
    end
    U->>U: parse the multipart body
    opt body is not parseable multipart
        U-->>C: 400
    end
    U->>G: resolve the dependency
    alt model loading
        G-->>C: 503 with Retry-After
    else model failed to load
        G-->>C: 503 with the redacted load error
    else model ready
        G-->>U: Predictor
    end
    U->>U: validate fields: derm_image, sex, fitzpatrick_type required, age_approx 0-120
    opt field missing or invalid
        U-->>C: 422
    end
    U->>Q: wait for a slot in the event loop (span queue_wait)
    alt MAX_CONCURRENT_PREDICTIONS running and MAX_QUEUED_PREDICTIONS waiting
        Q-->>C: 503 with Retry-After 1 (busy, /ready stays 200, busy counter +1)
    else slot free, or one frees up while queued
        Q-->>U: slot held until the prediction finishes
    end
    Note over U,T: From here the work runs in a threadpool thread (span context carried over)
    U->>R: derm_image, then clinical_image if sent (span validate_input, with the next step)
    opt an image above MAX_UPLOAD_MB
        R-->>C: 413
    end
    U->>M: sex, anatomical_site_general, fitzpatrick_type
    opt value outside the package vocabulary (tabular_encoding.json)
        M-->>C: 422
    end
    U->>P: predict(images, metadata), omitted age or site encoded as missing
    P->>P: spans decode, preprocess, inference (one ensemble_member span per session), stage-duration metrics
    opt dermoscopic image missing or not decodable, or either image above OPENCV_IO_MAX_IMAGE_PIXELS
        P-->>C: 422 (InvalidInput)
    end
    P-->>U: Prediction
    U->>T: Prediction and metadata_missing (span assemble_response)
    T-->>C: 200 JSON, always with the disclaimer
    U-)J: finished spans, batched in a background thread (a down Jaeger costs a log line, never the request)
    Note over C,L: Every response carries X-Request-ID and one access log line (id, path, status, latency, size)
    Note over C,L: Every response is counted in milk10k_http_requests_total and the latency histogram (route template)
    Note over C,L: Any other exception: 500 with a generic message (traceback and request id only in the server log)
```

`GET /health` always returns 200. `GET /ready` returns 503 while loading or after a failed load, and 200 when ready,
also under overload. `GET /metrics` returns the Prometheus text format. All three are `async def`, so they answer while
every prediction slot is busy, and none of them creates a trace. In the container the
same path runs unchanged; the Docker healthcheck polls `/ready` on the container port, so a failed load shows as
`unhealthy` (the process keeps running; Docker restarts it only when it exits).

## 3. CI/CD and deployment flow

```mermaid
flowchart LR
    push["git push to main on GitHub"] --> poll["Jenkins pollSCM H/5<br/>main only, no PR builds"]
    poll --> setup["Setup: make setup<br/>serving venv with uv, CPython 3.12"]
    setup --> tests["Unit tests: make test-report<br/>JUnit, coverage >= 90 %"]
    setup --> lint["Lint: make lint"]
    setup --> hyg["Hygiene: make hygiene<br/>no data, weights or secrets in git"]
    tests --> build["Build: docker build<br/>milk10k-api:build-sha7, no registry"]
    lint --> build
    hyg --> build
    build --> deploy["Deploy: fetch_model.sh --check, compose up --wait<br/>attach to milk10k_default, wait for /ready"]
    deploy --> smoke["Smoke: make smoke"]
    smoke --> parity["Parity: make sample from the read-only dataset<br/>make test-container, 6 lesions, tolerance 0.002"]
    parity --> k8s["Kubernetes: namespaced ServiceAccount, attach to the cluster network<br/>image into the minikube node, helm upgrade --rollback-on-failure, 1 replica<br/>smoke + parity on the new Pod; UNSTABLE if the cluster is stopped"]
    k8s --> post["post always: detach networks, delete sample images<br/>keep 3 newest image tags, archive reports"]
    build -.-> pushimg["Bonus: push to a registry"]
    pushimg -.-> approve{"Bonus: approval step"}
    approve -.-> gkedeploy["Deploy to GKE"]
    tfapply["Terraform apply: GKE, registry, bucket"] -.-> gkedeploy
    gkedeploy -.-> teardown["Teardown: helm uninstall, terraform destroy"]

    classDef implemented fill:#e6f4ea,stroke:#1e7e34,stroke-width:2px,color:#111111
    classDef planned fill:#ffffff,stroke:#666666,stroke-width:1px,stroke-dasharray:5 5,color:#111111
    class push,poll,setup,tests,lint,hyg,build,deploy,smoke,parity,k8s,post implemented
    class pushimg,approve,gkedeploy,tfapply,teardown planned
```

Unit tests, lint and hygiene run as parallel stages; any failure skips every later stage, and the `post`
block still detaches the network and deletes the sample images. Each stage has its own timeout; one executor, no
concurrent builds. The required path ends with parity against the deployed image; the Kubernetes stage (local
cluster, no registry: the image is copied into the node) and the GKE branch are bonus items. Measured serving cost, which sizes the replicas and limits, is in
[serving_measurements.md](serving_measurements.md).

## 4. CI network and Docker access

```mermaid
sequenceDiagram
    autonumber
    participant J as Jenkins container<br/>network milk10k-ci_default
    participant D as Host Docker daemon<br/>/var/run/docker.sock
    participant S as Stack network milk10k_default
    participant A as api:8000<br/>milk10k-api:build-sha7
    J->>D: docker build -t milk10k-api:build-sha7 serving
    J->>D: docker compose up -d --no-build --wait<br/>bind mounts resolved on the host:<br/>workspace and model at the same paths
    D->>S: create network, start the 6 services
    J->>D: docker network connect milk10k_default (skipped if attached)
    J->>A: GET /ready until 200 (120 s)
    J->>A: smoke test: /health, /predict, /metrics
    J->>A: parity: 6 sample lesions over HTTP
    Note over J,S: post always, also after a failure
    J->>D: docker network disconnect milk10k_default
```

Jenkins talks to the host's Docker daemon through the mounted socket, which is root-equivalent on the host (a `:ro`
mount does not change that; tested). JENKINS_HOME is mounted at the same absolute path inside and outside the
container because the daemon resolves Compose bind mounts on the host. Jenkins joins the stack network only while a
build needs `http://api:8000` and leaves it in `post { always }`; the stack's own `make down` therefore removes the
network cleanly.

## 5. CI design choices

| Topic | Choice | Reason |
|---|---|---|
| Docker access | Host socket; non-root Jenkins with the socket's group, no capabilities, UI on 127.0.0.1, `main` only | Jenkins must build images and run Compose on this machine; Docker-in-Docker is privileged too and would deploy a nested copy, not the stack the README shows |
| JENKINS_HOME | `~/milk10k-ci/jenkins_home`, same absolute path inside and outside | The host daemon resolves Compose bind mounts, so workspace paths must exist on the host |
| Image destination | No registry for the local pipeline; local tag `<build>-<sha7>` | Build and deploy run on the same machine: no credentials, nothing leaves the laptop |
| Model in CI | Read-only hard-linked copy, checked with `fetch_model.sh --check`, never downloaded | The download takes minutes; the check takes about a second |
| Dataset | Read-only mount at its host path, only for `make_sample.py --data-dir` | The sample lesions are dataset images and may not be stored in git |
| Parity | HTTP parity against the deployed API; the 479-lesion test deselected | Proves the image that actually serves; the full run needs the whole dataset |
| Failed deploy | Manual rollback (documented limitation) | Keeps the pipeline simple; the newest 3 image tags stay available |
| Resources | 2 GiB without extra swap, `-Xmx768m`, 1 executor, started on demand | Measured peak about 1.3 GiB; a 16 GB laptop also runs the stack |
| Tests | Inside the Jenkins container with uv and the Makefile; uv cache under JENKINS_HOME | The same commands as on a developer machine; dependencies downloaded once |
| Jenkins image | 2.580.2 LTS on JDK 21 by digest; kubectl v1.37.2 and Helm v4.3.2 with pinned sha256 | Current LTS; kubectl within one minor version of the cluster, Helm as on the host |
| Cluster credentials | ServiceAccount with a Role in namespace `milk10k` only (the chart's Deployment, Service and ServiceMonitor); 30-day token in a gitignored kubeconfig, loaded as a Jenkins file credential, bound only in the Kubernetes stage | Least privilege for the pipeline's kubectl and Helm, and the same shape a cloud cluster needs; the Docker socket still makes Jenkins root-equivalent on the host |
| Cluster network | Attach to the cluster's Docker network during the stage, detach in `post` | Works whether or not the cluster exists; Docker isolates the networks otherwise |
| Image into the cluster | `docker save` piped into containerd in the node (`ctr images import`) | No registry and no minikube files inside Jenkins; becomes a `docker push` with a registry |
| Replicas in CI | 1 replica, `maxSurge: 1`, `maxUnavailable: 0` | 3 replicas do not fit next to the Compose stack and Jenkins on 16 GB; the old Pod serves until the new one is ready |
| Failed Kubernetes deploy | `helm upgrade --rollback-on-failure`, 5 min | The last good release keeps serving (checked with a wrong image tag) |
| Cluster not running | Stage and build UNSTABLE with a message | Not red for a stopped laptop cluster, never silently green |
| Configuration | JCasC + Job DSL, setup wizard off, plugins pinned | Jenkins is recreated from the repository alone |
| Secrets | Gitignored `ci/secrets/` files as Compose secrets, loaded by JCasC | No credentials in git or in the image |
| Network | Attach to the stack network at deploy, detach in `post { always }` | Jenkins can start before the stack exists, and `make down` removes the network cleanly |
| Image tag | `image: milk10k-api:${API_IMAGE_TAG:-dev}` in `compose.yaml` | A plain `make up` keeps building `dev`; the pipeline deploys its own tag |
| Failure drill | Through Jenkins Replay | The failure path is tested without a broken commit in the history |
| Python versions | CI tests on CPython 3.12.14, the image on 3.12.16 | Parity runs against the image, so it tests the Python that serves |

## 6. Local Kubernetes deployment

```mermaid
flowchart LR
    dev(["Developer machine or the Jenkins Kubernetes stage<br/>scripts/k8s_deploy.sh: image into the node,<br/>helm upgrade --install"])
    hostdir[("Model package folder on the host<br/>15 ONNX files + manifest.json")]
    pf(["kubectl port-forward<br/>smoke test, HTTP parity"])

    subgraph node["minikube node, profile milk10k, docker driver, capped at 8 GiB"]
        images[("Images in the node<br/>milk10k-api, pinned tag, never pulled")]
        mnt[("Bind mount of the host folder<br/>made at cluster start")]
        kubelet["kubelet probes<br/>startup + readiness: GET /ready<br/>liveness: GET /health"]
        incl(["Client inside the cluster"])
        subgraph ns["Namespace milk10k, Helm release milk10k-api"]
            svc["Service milk10k-api<br/>ClusterIP :8000"]
            p1["Pod 1<br/>init container verify-model: sha256<br/>api: 4 threads, 1 running prediction"]
            p2["Pod 2<br/>same as Pod 1"]
            p3["Pod 3<br/>same as Pod 1"]
        end
    end

    dev -->|"image load"| images
    dev -->|"helm upgrade --install"| svc
    hostdir --> mnt
    mnt -->|"hostPath, read-only"| p1
    mnt -->|"hostPath, read-only"| p2
    mnt -->|"hostPath, read-only"| p3
    images --> p1
    images --> p2
    images --> p3
    incl -->|"new connection each request,<br/>spread by kube-proxy"| svc
    svc --> p1
    svc --> p2
    svc --> p3
    pf -->|"one Pod per session"| p1
    kubelet -->|"probes"| p1

    classDef implemented fill:#e6f4ea,stroke:#1e7e34,stroke-width:2px,color:#111111
    class dev,hostdir,pf,images,mnt,kubelet,incl,svc,p1,p2,p3 implemented
    style node fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    style ns fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
```

How to run and check it: [k8s/README.md](../k8s/README.md). Measured with 3 replicas:

- **Pod start:** the init container checks every package file against `manifest.json`; then the API loads the 15
  ONNX sessions while `/ready` answers 503, and the startup probe keeps the Pod out of the Service until it is 200.
  3 Pods started together: 3/3 Ready in 10.6 s (sha256 check about 3 s, model load 3.3–3.4 s).
- **Traffic:** the Service spreads new connections over the Ready Pods (30 requests: 11 / 7 / 12).
  `kubectl port-forward` is pinned to one Pod, so it is used only for the smoke and parity tests (parity 2.55e-04,
  tolerance 0.002).
- **Rollout:** one Pod at a time (`maxSurge: 0`, `maxUnavailable: 1`), because a 4th Pod would not fit the memory
  budget. Terminating Pods are not counted, so an old Pod can briefly overlap with its replacement (4 Pods were seen).
  Each old Pod waits 5 s (`preStop` sleep) before it stops, as a precaution; rollouts without it were not tested.
  No failed request out of 373 during a rollout, 183 out of 183 while a Pod was deleted and replaced (Ready again
  after 8.7 s).
- **Resources and security:** see the design table in [k8s/README.md](../k8s/README.md#design-choices) and the
  measurements in [serving_measurements.md](serving_measurements.md).

### Monitoring on the local cluster

```mermaid
flowchart LR
    make(["make k8s-monitoring<br/>helm upgrade --install kps, chart 93.10.0 by digest<br/>kubectl apply -k k8s/monitoring"])
    secretfile[("k8s/secrets/grafana-admin-password<br/>gitignored, random")]
    browser(["Browser on the same machine<br/>port-forward 127.0.0.1 :19090 :13000"])
    jenkinsdeploy(["API deploy, by hand or from Jenkins<br/>values-local.yaml: serviceMonitor on"])

    subgraph node["minikube node, profile milk10k"]
        subgraph mon["Namespace monitoring"]
            operator["Prometheus Operator<br/>CRDs: ServiceMonitor, PrometheusRule"]
            prom["Prometheus v3.16<br/>1 day of data, rules every 15 s"]
            graf["Grafana 13.3<br/>dashboard sidecar, login from Secret"]
            ksm["kube-state-metrics"]
            nodeexp["node-exporter<br/>sees the host's CPUs and memory"]
            rule[("PrometheusRule milk10k-api<br/>generated from alert-rules.yml")]
            dash[("ConfigMap milk10k-api-dashboard<br/>label grafana_dashboard=1")]
            sec[("Secret grafana-admin")]
        end
        subgraph ns["Namespace milk10k"]
            sm[("ServiceMonitor milk10k-api<br/>port http, 15 s, job api")]
            pods["API Pods, /metrics on :8000"]
        end
        kubelet["kubelet, cAdvisor"]
    end

    make --> operator
    make --> rule
    make --> dash
    secretfile -->|"make k8s-grafana-secret, piped, never printed"| sec
    jenkinsdeploy -->|"created only if the CRD exists"| sm
    operator -->|"reads ServiceMonitor and PrometheusRule (label release=kps), writes the config"| prom
    sm -.->|"selects the Service's endpoints"| pods
    prom -->|"scrape each Pod directly"| pods
    prom --> kubelet
    prom --> ksm
    prom --> nodeexp
    rule --> prom
    dash --> graf
    sec --> graf
    graf -->|"datasource uid prometheus"| prom
    browser --> prom
    browser --> graf

    classDef implemented fill:#e6f4ea,stroke:#1e7e34,stroke-width:2px,color:#111111
    class make,secretfile,browser,jenkinsdeploy,operator,prom,graf,ksm,nodeexp,rule,dash,sec,sm,pods,kubelet implemented
    style node fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    style mon fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    style ns fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
```

- **One target per replica:** the ServiceMonitor points Prometheus at the Service's endpoints, so each API Pod is
  scraped on its own address (labels `namespace`, `pod`), not through the load-balanced ClusterIP.
- **Same rules as Compose:** the PrometheusRule is generated from `monitoring/prometheus/alert-rules.yml` and the job
  is named `api`, so the promtool tests cover both. Scaling the API to 0 fired ApiDown (no target at all) after
  about 1.5 min; it resolved when a Pod was scraped again.
- **Footprint:** the monitoring Pods use about 0.6 GiB together, 1.0–1.7 GiB in the node; no Alertmanager. With 3
  API replicas the 8 GiB node is at its cap and the laptop runs out of memory within an hour (the kernel killed
  Prometheus and other monitoring processes; the API Pods survived), so monitoring normally runs next to 1 or 2
  replicas.
- **Removal:** Helm keeps the operator's CRDs; take the ServiceMonitor out of the API release before deleting them
  (order in [k8s/README.md](../k8s/README.md#6-stop-or-remove)).
