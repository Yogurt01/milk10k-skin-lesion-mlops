"""The Kubernetes copies of the monitoring config match their sources (no cluster needed)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_prometheusrule_is_generated_from_alert_rules():
    # k8s/monitoring/prometheusrule-api.yaml must be alert-rules.yml under `spec:`; edit the rules there, then
    # run `make k8s-rules`.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/k8s_prometheusrule.py"), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_k8s_dashboard_uses_the_provisioned_datasource():
    dashboard = json.loads((ROOT / "k8s/monitoring/dashboards/milk10k-api.json").read_text(encoding="utf-8"))
    targets = [t for p in dashboard["panels"] for t in p.get("targets", [])]
    assert targets
    # kube-prometheus-stack provisions its Prometheus data source with uid "prometheus" (same uid as Compose).
    assert {t["datasource"]["uid"] for t in targets} == {"prometheus"}
    # Every API query is limited to the selected namespace.
    assert all('namespace="$namespace"' in t["expr"] for t in targets)


def test_model_ready_stat_shows_only_pods_scraped_now():
    # A range query reduced to "last value" also lists Pods deleted earlier in the time range; an instant query returns
    # only series Prometheus has now. A removed target's series are marked stale shortly after, about 20 s when checked
    # (k8s/README.md, "What was checked").
    dashboard = json.loads((ROOT / "k8s/monitoring/dashboards/milk10k-api.json").read_text(encoding="utf-8"))
    panel = next(p for p in dashboard["panels"] if p.get("title") == "Model ready per Pod")
    assert panel["type"] == "stat"
    (target,) = panel["targets"]
    assert target["expr"] == 'milk10k_model_ready{namespace="$namespace"}'
    assert target["instant"] is True
    assert target["range"] is False
