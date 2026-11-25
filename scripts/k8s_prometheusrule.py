#!/usr/bin/env python3
"""Write the API's alert rules as a Prometheus Operator PrometheusRule (`make k8s-rules`).

The rules stay in one place, monitoring/prometheus/alert-rules.yml (used by the Compose Prometheus and tested with
promtool). k8s/monitoring/prometheusrule-api.yaml is that file's text indented under `spec:`, without its full-line
comments (read them in the source), so its rule groups are an exact copy and the promtool tests cover them too.
`--check` exits 1 when the generated file is out of date (run by tests/test_k8s_monitoring.py). No third-party
modules: plain text, no YAML parsing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "monitoring/prometheus/alert-rules.yml"
TARGET = ROOT / "k8s/monitoring/prometheusrule-api.yaml"

HEADER = """\
# GENERATED from monitoring/prometheus/alert-rules.yml (comments there) by scripts/k8s_prometheusrule.py
# (make k8s-rules); do not edit.
# The API's alert rules for the Prometheus of the local kube-prometheus-stack release "kps": Prometheus loads every
# PrometheusRule with the label release=kps. The rules use job="api", which the API chart's ServiceMonitor sets.
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: milk10k-api
  namespace: monitoring
  labels:
    release: kps
    app.kubernetes.io/part-of: milk10k
spec:
"""


def render() -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.lstrip().startswith("#")]
    return HEADER + "".join(("  " + line if line.strip() else "") + "\n" for line in kept)


def main() -> int:
    text = render()
    if sys.argv[1:] == ["--check"]:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET.relative_to(ROOT)} is out of date: run make k8s-rules", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
