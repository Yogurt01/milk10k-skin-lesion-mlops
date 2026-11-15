# monitoring/

Observability configuration for the Docker Compose stack (`compose.yaml` at the repository root, `make up`). Every file
is mounted read-only into its container. UIs on 127.0.0.1 only: Prometheus :9090, Grafana :3000, Jaeger :16686.

| Path | Used by | What |
|---|---|---|
| `prometheus/prometheus.yml` | Prometheus | Scrapes the API (`/metrics`, DNS discovery of the `api` service, so replicas are found), node-exporter, cAdvisor and itself every 15 s on the internal `monitoring` network |
| `prometheus/alert-rules.yml` | Prometheus | `ApiDown`, `ApiHighErrorRate` (5xx share of `/predict` > 10 % for 1 min), `ApiHighLatencyP95` (p95 of `/predict` > 3 s for 5 min); shown at http://localhost:9090/alerts (no Alertmanager) |
| `grafana/provisioning/datasources/prometheus.yml` | Grafana | Datasource `Prometheus` (uid `prometheus`, `http://prometheus:9090`) |
| `grafana/provisioning/dashboards/dashboards.yml` | Grafana | Loads the dashboards below at startup (UI edits are not saved back) |
| `grafana/dashboards/machine-resources.json` | Grafana | "MILK10k: machine resources and API": host CPU, memory, disk space, disk I/O, network; per-container CPU, memory, disk I/O, network; API request rate, `/predict` p50/p95, running and waiting predictions, 503s and 5xx share, inference time, model ready |
| `jaeger/config.yaml` | Jaeger | OTLP/HTTP receiver (4318, internal network), in-memory storage capped at 10,000 traces, UI and API on 16686 (JSON API under `/api/v3`) |

Notes:
- **Host network:** node-exporter runs on a bridge network, so its network collectors would report its own
  container's interface; they are switched off and the host's interfaces come from cAdvisor's root cgroup (`id="/"`).
- **cAdvisor** reads the Docker and containerd sockets (read-only mounts, but a socket still accepts any API call): it
  names containers with them. It is not privileged and has no Linux capabilities.
- **Grafana** admin password: `GRAFANA_ADMIN_PASSWORD` in `.env` (required; `make env` generates one). Usage reporting,
  update checks, plugin downloads and Gravatar are off.

The kube-prometheus-stack values for Kubernetes (P1-OBS-09) will go here too.
