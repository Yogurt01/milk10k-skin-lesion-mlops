# Architecture

> **Research model, not a medical device.** Its output is not a diagnosis and must not be used for clinical decisions.

Status as of 2026-11-15 (M6 closed: the monitoring stack no longer waits for a healthy API, alert rules unit-tested; M6b: API metrics, Prometheus, node-exporter, cAdvisor, Grafana, Jaeger and OpenTelemetry tracing in Compose; M6a: model fetch script, API image). The diagrams are derived from the code and the project's progress checklist. Task IDs
(`P1-…`) refer to that checklist, which is kept outside this repository. The diagrams are updated at the end of each
milestone.

Styling in every diagram:
- **Solid** box or arrow: implemented.
- **Dashed**: planned.
- **Orange, dotted**: owner's extension beyond the course requirements (`EXT`), planned after the required items.

## 1. Components

```mermaid
flowchart LR
    client(["Client<br/>curl, serving/client.py"])

    subgraph serving["Serving"]
        nginx["nginx reverse proxy<br/>official image in Compose<br/>P1-NGX-01..04"]
        subgraph container["API container, Compose service api"]
            api["FastAPI app, api.py<br/>/health /ready /predict<br/>request id, concurrency limit"]
            metrics["/metrics endpoint, metrics.py<br/>P1-OBS-01, P1-OBS-02"]
            tracing["OpenTelemetry SDK, tracing.py<br/>server span + stage spans<br/>P1-OBS-06, P1-OBS-07"]
            pred["milk10k_serving Predictor<br/>decode, preprocess, assemble"]
            ort["ONNX Runtime, CPU<br/>15 member sessions"]
        end
        pkg[("Model package on the host<br/>15 ONNX files + manifest.json with sha256<br/>not in the image")]
    end

    release[("GitHub Release model-v1.0.0<br/>model package assets")]

    subgraph obs["Observability, Compose services on the internal network monitoring"]
        prom["Prometheus v3.16, alert rules<br/>P1-OBS-03, P1-OBS-08"]
        graf["Grafana 13.3, provisioned<br/>datasource + dashboard<br/>P1-OBS-04, P1-OBS-05"]
        nodeexp["node-exporter v1.12<br/>host CPU, memory, disk<br/>P1-OBS-03"]
        cadv["cAdvisor v0.60<br/>per-container + host network<br/>P1-OBS-03"]
        jaeger["Jaeger v2.23, in-memory<br/>P1-OBS-06, P1-OBS-07"]
    end
    host[("Host: /proc, /sys, / read-only<br/>Docker + containerd sockets")]
    browser(["Owner's browser<br/>127.0.0.1 :9090 :3000 :16686"])

    subgraph cicd["CI/CD and deployment"]
        repo["GitHub repository<br/>P1-SET-06"]
        jenkins["Jenkins in Compose<br/>P1-CI-01..10"]
        registry[("Image registry<br/>P1-CI-05, P1-GKE-03")]
        compose["Compose deploy of the pushed image<br/>P1-CI-06"]
        mk["minikube, Helm chart, 3 replicas<br/>P1-K8S-01..07, P1-CI-09"]
        kps["kube-prometheus-stack via Helm<br/>P1-OBS-09, P1-OBS-10"]
        tf["Terraform<br/>P1-IAC-01..05"]
        gke["GKE, Helm chart, 3 replicas<br/>P1-GKE-01..08"]
        gcs[("GCS copy of the model package<br/>P1-GKE-04")]
        kpsg["kube-prometheus-stack on GKE<br/>P1-GKE-07"]
    end

    client -->|"HTTP 127.0.0.1:8000, make up or make serve"| api
    client -.->|EXT| nginx
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

    repo -.->|"pollSCM H/5"| jenkins
    jenkins -.->|"build, push"| registry
    registry -.-> compose
    registry -.-> mk
    registry -.-> gke
    mk -.- kps
    tf -.-> gke
    tf -.-> gcs
    release -.->|copy| gcs
    gcs -.->|initContainer| gke
    gke -.- kpsg

    classDef implemented fill:#e6f4ea,stroke:#1e7e34,stroke-width:2px,color:#111111
    classDef planned fill:#ffffff,stroke:#666666,stroke-width:1px,stroke-dasharray:5 5,color:#111111
    classDef ext fill:#fff4e5,stroke:#b35c00,stroke-width:1px,stroke-dasharray:2 4,color:#111111
    class api,pred,ort,pkg,release,repo,metrics,tracing,prom,graf,nodeexp,cadv,jaeger,host implemented
    style container fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    style obs fill:#f3faf5,stroke:#1e7e34,stroke-width:2px,color:#111111
    class jenkins,registry,compose,mk,kps,tf,gke,gcs,kpsg planned
    class nginx ext
```

| Component | Status | Evidence |
|---|---|---|
| Client | external; example client implemented | Any HTTP client; [`serving/client.py`](../serving/client.py) (standard library only, P1-API-08) |
| FastAPI app (`/health`, `/ready`, `/predict`) | implemented | [`serving/src/milk10k_serving/api.py`](../serving/src/milk10k_serving/api.py); P1-API-01..11; request ids and access log (P1-PY-07); started with `make serve` ([Makefile](../Makefile)) |
| Predictor (decode, preprocessing, assembly) | implemented | [`predictor.py`](../serving/src/milk10k_serving/predictor.py), [`preprocess.py`](../serving/src/milk10k_serving/preprocess.py); P1-PY-05; decoded-image cap P1-API-09 |
| ONNX Runtime, 15 member sessions | implemented | [`package.py`](../serving/src/milk10k_serving/package.py) (`load_package`, `Ensemble`); P1-PY-04 |
| Model package with sha256 manifest | implemented | Built by [notebook 08](../notebooks/08_export_for_deployment.ipynb) (P1-NB-03); verified at startup by `verify_package` (P1-PY-04). Not in git and not in the image: mounted read-only into the container (P1-DCK-02) |
| GitHub Release `model-v1.0.0` | implemented | 19 assets; [`scripts/fetch_model.sh`](../scripts/fetch_model.sh) (`make fetch-model`) downloads them without a token, pins the manifest's sha256 and checks every file (P1-SET-10, P1-BSH-02) |
| `/metrics` (API and model metrics) | implemented | [`metrics.py`](../serving/src/milk10k_serving/metrics.py): requests by route template / method / status, latency histogram, running and waiting predictions, 503-busy counter, model ready / load time / info, stage durations, P(malignant) histogram; labels bounded (unmatched paths → `unmatched`). P1-OBS-01, P1-OBS-02 |
| Prometheus, node-exporter, cAdvisor | implemented | [`monitoring/prometheus/`](../monitoring/prometheus/) (scrape config, 3 alert rules with promtool unit tests in `alert-rules.test.yml`, P1-OBS-08); all 4 targets up (P1-OBS-03). node-exporter and cAdvisor only on the internal network, host folders read-only; cAdvisor also reads the Docker and containerd sockets (Decision log) |
| Grafana (machine resources and API dashboard) | implemented; dashboard checked by the owner (2026-11-15) | [`monitoring/grafana/`](../monitoring/grafana/): provisioned datasource and one dashboard (host, containers, API rows; every panel query returns data through the Prometheus API). Admin password only from `.env`; analytics and update checks off. P1-OBS-04, P1-OBS-05 |
| Jaeger v2 via the OTLP exporter | implemented; UI and span tags checked by the owner (2026-11-15) | [`tracing.py`](../serving/src/milk10k_serving/tracing.py): OTLP/HTTP to `jaeger:4318` when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, FastAPI instrumentation (probes and `/metrics` excluded), stage spans; [`monitoring/jaeger/config.yaml`](../monitoring/jaeger/config.yaml) (in-memory, 10,000 traces). P1-OBS-06, P1-OBS-07 |
| nginx (reverse proxy, load balancing in a Compose profile, GET-only cache exercise) | EXT, planned after the required items | P1-DEC-13, P1-NGX-01..04. Official `nginx` image in Compose only, never the retired community Ingress NGINX on Kubernetes |
| GitHub repository | implemented | Public repository (P1-SET-06) |
| Jenkins in Compose | planned | P1-CI-01..10 |
| Image registry | planned | P1-CI-05; Artifact Registry for GKE: P1-GKE-03, P1-IAC-03 |
| API image | implemented | [`serving/Dockerfile`](../serving/Dockerfile): pinned `python:3.12.16-slim-trixie` (tag + digest), pinned requirements then the package with its `api` extra, no torch, uid 10001, healthcheck on `/ready` with the Python standard library, uvicorn without `--workers` (P1-DCK-01) |
| Docker Compose stack | implemented (api + 5 monitoring services) | [`compose.yaml`](../compose.yaml), `make up` / `make down`: images pinned by tag + digest, model mounted read-only, UIs on 127.0.0.1 only, internal network `monitoring` for scrapes and OTLP, every service `cap_drop: ALL` + `no-new-privileges` + a memory limit without swap + json-file log rotation (10 MB × 3), Prometheus starts with the API without waiting for its health, so an API that cannot load its model still shows up in monitoring (`milk10k_model_ready` 0); Grafana waits for a healthy Prometheus (P1-DCK-03..06); one-command check from a fresh clone: P1-DCK-07 |
| Compose deploy from the registry | planned | P1-CI-06 |
| minikube with a Helm chart, 3 replicas | planned (bonus) | P1-K8S-01..07, P1-CI-09 |
| kube-prometheus-stack via Helm | planned (bonus) | P1-OBS-09, P1-OBS-10; on GKE: P1-GKE-07 |
| Terraform, GKE, GCS copy of the package | planned (bonus) | P1-IAC-01..05, P1-GKE-01..08 (teardown right after the demo: P1-GKE-08) |

## 2. `POST /predict` request path

Derived from [`api.py`](../serving/src/milk10k_serving/api.py). The status codes and the steps that raise them are
listed in that file's docstring and in the OpenAPI `responses`. Form parsing and field validation run inside FastAPI,
before the body of the `predict` handler. `predict` is an `async def` that waits for a slot of the
`PredictionLimiter` in the event loop (a waiting request holds no thread), then runs the blocking work in the
threadpool, so at most `MAX_CONCURRENT_PREDICTIONS` threads do inference (P1-API-05). `/health`, `/ready` and
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

## 3. CI/CD and deployment flow (all planned)

```mermaid
flowchart LR
    push["git push to GitHub"] -.-> poll["Jenkins pollSCM H/5<br/>P1-CI-07"]
    poll -.-> lint["Lint and repo hygiene<br/>P1-CI-08, P1-CI-10"]
    poll -.-> tests["Unit + API tests, JUnit report<br/>P1-CI-03, P1-TST-07"]
    lint -.-> parity["Parity stage<br/>model package from the Release, P1-BSH-02<br/>sample lesions from the read-only dataset mount, P1-BSH-03<br/>P1-CI-04"]
    tests -.-> parity
    parity -.-> build["Build image<br/>tag BUILD_NUMBER + short sha<br/>P1-CI-05"]
    build -.-> pushimg["Push to the registry<br/>P1-CI-05"]
    pushimg -.-> compose["Deploy Compose + smoke test<br/>P1-CI-06, P1-BSH-04"]
    pushimg -.-> helm["Bonus: helm upgrade on minikube<br/>P1-CI-09"]
    pushimg -.-> approve{"Bonus: approval step<br/>P1-CI-09"}
    approve -.-> gkedeploy["Deploy to GKE<br/>P1-GKE-05"]
    tfapply["Terraform apply: GKE, registry, bucket<br/>P1-IAC-01..04"] -.-> gkedeploy
    gkedeploy -.-> teardown["Teardown: helm uninstall, terraform destroy<br/>P1-GKE-08"]

    classDef planned fill:#ffffff,stroke:#666666,stroke-width:1px,stroke-dasharray:5 5,color:#111111
    class push,poll,lint,tests,parity,build,pushimg,compose,helm,approve,gkedeploy,tfapply,teardown planned
```

Lint/hygiene and the unit + API tests run as parallel stages (P1-CI-10). The required path ends at the Compose deploy.
The minikube and GKE branches are bonus items. Measured serving cost, which sizes the replicas and limits, is in
[serving_measurements.md](serving_measurements.md).
