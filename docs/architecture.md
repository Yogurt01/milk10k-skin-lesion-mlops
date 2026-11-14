# Architecture

> **Research model, not a medical device.** Its output is not a diagnosis and must not be used for clinical decisions.

Status as of 2026-11-14. The diagrams are derived from the code and the project's progress checklist. Task IDs
(`P1-…`) refer to that checklist, which is kept outside this repository. The diagrams are updated at the end of each
milestone.

Styling in every diagram:
- **Solid** box or arrow: implemented.
- **Dashed**: planned.
- **Orange, dotted**: owner's extension beyond the course requirements (`EXT`), planned after the required items.

## 1. Components

```mermaid
flowchart LR
    client(["Client<br/>curl, Python client"])

    subgraph serving["Serving"]
        nginx["nginx reverse proxy<br/>official image in Compose<br/>P1-NGX-01..04"]
        api["FastAPI app, api.py<br/>/health /ready /predict"]
        metrics["/metrics endpoint<br/>P1-OBS-01, P1-OBS-02"]
        pred["milk10k_serving Predictor<br/>decode, preprocess, assemble"]
        ort["ONNX Runtime, CPU<br/>15 member sessions"]
        pkg[("Model package<br/>15 ONNX files + manifest.json with sha256")]
    end

    release[("GitHub Release model-v1.0.0<br/>model package assets")]

    subgraph obs["Observability"]
        prom["Prometheus<br/>P1-OBS-03"]
        graf["Grafana<br/>P1-OBS-04, P1-OBS-05"]
        nodeexp["node-exporter<br/>P1-OBS-03"]
        cadv["cAdvisor<br/>P1-OBS-03"]
        jaeger["Jaeger v2<br/>P1-OBS-06, P1-OBS-07"]
    end

    subgraph cicd["CI/CD and deployment"]
        repo["GitHub repository<br/>P1-SET-06"]
        jenkins["Jenkins in Compose<br/>P1-CI-01..10"]
        registry[("Image registry<br/>P1-CI-05, P1-GKE-03")]
        compose["Docker Compose stack<br/>P1-DCK-01..07"]
        mk["minikube, Helm chart, 3 replicas<br/>P1-K8S-01..07, P1-CI-09"]
        kps["kube-prometheus-stack via Helm<br/>P1-OBS-09, P1-OBS-10"]
        tf["Terraform<br/>P1-IAC-01..05"]
        gke["GKE, Helm chart, 3 replicas<br/>P1-GKE-01..08"]
        gcs[("GCS copy of the model package<br/>P1-GKE-04")]
        kpsg["kube-prometheus-stack on GKE<br/>P1-GKE-07"]
    end

    client -->|"HTTP, make serve"| api
    client -.->|EXT| nginx
    nginx -.-> api
    api --> pred
    pred --> ort
    pkg -->|"sha256 verified, loaded once at startup"| ort
    release -.->|"fetch_model.sh, P1-BSH-02"| pkg
    api -.-> metrics
    prom -.->|scrape| metrics
    prom -.->|scrape| nodeexp
    prom -.->|scrape| cadv
    graf -.-> prom
    api -.->|"OTLP, jaeger:4318"| jaeger

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
    class api,pred,ort,pkg,release,repo implemented
    class metrics,prom,graf,nodeexp,cadv,jaeger,jenkins,registry,compose,mk,kps,tf,gke,gcs,kpsg planned
    class nginx ext
```

| Component | Status | Evidence |
|---|---|---|
| Client | external | Any HTTP client. A Python client example is planned (P1-API-08) |
| FastAPI app (`/health`, `/ready`, `/predict`) | implemented | [`serving/src/milk10k_serving/api.py`](../serving/src/milk10k_serving/api.py); P1-API-01..04, P1-API-06, P1-API-10, P1-API-11; started with `make serve` ([Makefile](../Makefile)) |
| Predictor (decode, preprocessing, assembly) | implemented | [`predictor.py`](../serving/src/milk10k_serving/predictor.py), [`preprocess.py`](../serving/src/milk10k_serving/preprocess.py); P1-PY-05 |
| ONNX Runtime, 15 member sessions | implemented | [`package.py`](../serving/src/milk10k_serving/package.py) (`load_package`, `Ensemble`); P1-PY-04 |
| Model package with sha256 manifest | implemented | Built by [notebook 08](../notebooks/08_export_for_deployment.ipynb) (P1-NB-03); verified at startup by `verify_package` (P1-PY-04). Not in git |
| GitHub Release `model-v1.0.0` | implemented (assets uploaded) | P1-SET-10 (in progress: download and verification by `scripts/fetch_model.sh`, P1-BSH-02, is planned) |
| `/metrics` (API and model metrics) | planned | P1-OBS-01, P1-OBS-02 |
| Prometheus, node-exporter, cAdvisor | planned | P1-OBS-03, P1-DCK-03 |
| Grafana (host, containers, API dashboards) | planned | P1-OBS-04, P1-OBS-05 |
| Jaeger v2 via the OTLP exporter | planned | P1-OBS-06, P1-OBS-07; the `OTEL_*` settings already exist in [`settings.py`](../serving/src/milk10k_serving/settings.py) and [`.env.example`](../.env.example), but no tracing code yet |
| nginx (reverse proxy, load balancing in a Compose profile, GET-only cache exercise) | EXT, planned after the required items | P1-DEC-13, P1-NGX-01..04. Official `nginx` image in Compose only, never the retired community Ingress NGINX on Kubernetes |
| GitHub repository | implemented | Public repository (P1-SET-06) |
| Jenkins in Compose | planned | P1-CI-01..10 |
| Image registry | planned | P1-CI-05; Artifact Registry for GKE: P1-GKE-03, P1-IAC-03 |
| Docker Compose stack | planned | P1-DCK-01..07 |
| minikube with a Helm chart, 3 replicas | planned (bonus) | P1-K8S-01..07, P1-CI-09 |
| kube-prometheus-stack via Helm | planned (bonus) | P1-OBS-09, P1-OBS-10; on GKE: P1-GKE-07 |
| Terraform, GKE, GCS copy of the package | planned (bonus) | P1-IAC-01..05, P1-GKE-01..08 (teardown right after the demo: P1-GKE-08) |

## 2. `POST /predict` request path

Derived from [`api.py`](../serving/src/milk10k_serving/api.py). The status codes and the steps that raise them are
listed in that file's docstring and in the OpenAPI `responses`. Form parsing and field validation run inside FastAPI,
before the body of the `predict` handler. `predict` is a sync `def`, so it runs in the framework's threadpool. No
explicit concurrency limit exists yet (P1-API-05, planned).

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant U as uvicorn / FastAPI
    participant G as get_predictor
    participant R as read_limited
    participant M as metadata_errors
    participant P as Predictor
    participant T as to_response
    participant L as model-loader thread

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
    U->>R: derm_image, then clinical_image if sent
    opt an image above MAX_UPLOAD_MB
        R-->>C: 413
    end
    U->>M: sex, anatomical_site_general, fitzpatrick_type
    opt value outside the package vocabulary (tabular_encoding.json)
        M-->>C: 422
    end
    U->>P: predict(images, metadata), omitted age or site encoded as missing
    P->>P: decode, preprocess, run the 15 sessions in sequence, average
    opt dermoscopic image missing or not decodable
        P-->>C: 422 (InvalidInput)
    end
    P-->>U: Prediction
    U->>T: Prediction and metadata_missing
    T-->>C: 200 JSON, always with the disclaimer
    Note over C,L: Any other exception at any step: 500 with a generic message (traceback only in the server log)
```

`GET /health` always returns 200. `GET /ready` returns 503 while loading or after a failed load, and 200 when ready.

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
