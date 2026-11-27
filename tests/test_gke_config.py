"""GKE settings (deploy/README.md) without a cloud account: chart renders, chart guards, Terraform safety settings.

The Helm tests need the helm binary (skipped without it); nothing here talks to a cluster or to Google Cloud.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHART = ROOT / "k8s/milk10k-api"
TF = ROOT / "deploy/terraform"
GKE_SETS = [
    "--set", "image.tag=1-abcdef0",
    "--set", "image.repository=us-central1-docker.pkg.dev/example-project/milk10k/milk10k-api",
    "--set", "model.gcs.bucket=example-project-milk10k-model",
]  # fmt: skip

needs_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def helm_template(*args: str) -> subprocess.CompletedProcess:
    cmd = ["helm", "template", "milk10k-api", str(CHART), "-n", "milk10k", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def gke(*extra: str) -> subprocess.CompletedProcess:
    return helm_template("-f", str(ROOT / "k8s/values-gke.yaml"), *GKE_SETS, *extra)


@needs_helm
@pytest.mark.parametrize("values", [[], ["k8s/values-local.yaml"], ["k8s/values-local.yaml", "k8s/values-ci.yaml"]])
def test_local_renders_keep_the_hostpath_model_and_no_new_objects(values):
    args = [a for v in values for a in ("-f", str(ROOT / v))]
    result = helm_template(*args, "--set", "image.tag=1-abcdef0")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "hostPath:" in out
    assert "fetch-model" not in out
    assert "kind: ServiceAccount" not in out
    assert "serviceAccountName" not in out
    assert "loadBalancerSourceRanges" not in out


@needs_helm
def test_gke_render_copies_the_model_from_the_bucket_before_the_sha256_check():
    result = gke()
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "hostPath" not in out
    assert "kind: ServiceAccount" in out
    assert "serviceAccountName: milk10k-api" in out
    assert "automountServiceAccountToken: false" in out
    # fetch-model runs first and is pinned by digest; verify-model then checks what it copied.
    assert out.index("- name: fetch-model") < out.index("- name: verify-model") < out.index("- name: api")
    assert re.search(r"image: gcr\.io/google\.com/cloudsdktool/google-cloud-cli:[\w.-]+@sha256:[0-9a-f]{64}", out)
    assert '"gs://example-project-milk10k-model/final-P1a-kg-onnx/*"' in out
    assert "image: us-central1-docker.pkg.dev/example-project/milk10k/milk10k-api:1-abcdef0" in out
    assert "cpu: 1500m" in out
    assert "ephemeral-storage: 3Gi" in out


@needs_helm
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--set", "service.type=LoadBalancer"], "needs service.loadBalancerSourceRanges"),
        (
            ["--set", "service.type=LoadBalancer", "--set", "service.loadBalancerSourceRanges[0]=0.0.0.0/0"],
            "must not contain 0.0.0.0/0",
        ),
        (["--set", "model.gcs.bucket="], "model.gcs.bucket is required"),
        (["--set", "serviceAccount.create=false"], "set serviceAccount.create=true"),
        (["--set", "image.repository="], "image.repository is required"),
        (["--set", "model.gcs.image.digest="], "pin the image by digest"),
    ],
)
def test_gke_render_refuses_unsafe_or_incomplete_values(extra, message):
    result = gke(*extra)
    assert result.returncode != 0
    assert message in result.stderr


@needs_helm
def test_gke_load_balancer_is_limited_to_the_given_network():
    result = gke("--set", "service.type=LoadBalancer", "--set", "service.loadBalancerSourceRanges[0]=203.0.113.10/32")
    assert result.returncode == 0, result.stderr
    assert "type: LoadBalancer\n  loadBalancerSourceRanges:\n    - 203.0.113.10/32" in result.stdout


def test_gke_values_hold_no_project_specific_value():
    text = (ROOT / "k8s/values-gke.yaml").read_text(encoding="utf-8")
    assert re.search(r'^  repository: ""', text, re.M)
    assert re.search(r'^    bucket: ""', text, re.M)
    # Only the pattern in a comment ("<project>"), never a concrete registry path.
    assert not re.search(r"pkg\.dev/[a-z]", text)


def block(text: str, header: str) -> str:
    """The body of a top-level Terraform block (up to the next line that is a lone closing brace)."""
    start = text.index(header)
    end = text.index("\n}\n", start)
    return text[start:end]


def test_model_bucket_is_private_and_destroyed_with_its_content():
    bucket = block((TF / "main.tf").read_text(encoding="utf-8"), 'resource "google_storage_bucket" "model"')
    assert re.search(r"uniform_bucket_level_access\s*=\s*true", bucket)
    assert re.search(r'public_access_prevention\s*=\s*"enforced"', bucket)
    assert re.search(r"force_destroy\s*=\s*true", bucket)
    assert re.search(r"retention_duration_seconds\s*=\s*0", bucket)  # no soft-deleted copy left behind
    assert "allUsers" not in (TF / "main.tf").read_text(encoding="utf-8")


def test_cluster_is_zonal_deletable_and_without_open_control_plane():
    main = (TF / "main.tf").read_text(encoding="utf-8")
    cluster = block(main, 'resource "google_container_cluster" "this"')
    assert re.search(r"location\s*=\s*var\.zone", cluster)
    assert re.search(r"deletion_protection\s*=\s*false", cluster)
    assert "master_authorized_networks_config" in cluster
    variables = (TF / "variables.tf").read_text(encoding="utf-8")
    assert 'c != "0.0.0.0/0"' in variables


def test_budget_alerts_on_gross_cost():
    budget = (TF / "budget.tf").read_text(encoding="utf-8")
    # On a Free Trial the net cost is zero (paid by the credit): alerts must ignore credits to ever fire.
    assert re.search(r'credit_types_treatment\s*=\s*"EXCLUDE_ALL_CREDITS"', budget)


def test_terraform_uses_local_state_and_ships_only_placeholders():
    for tf in TF.glob("*.tf"):
        assert not re.search(r"^\s*backend\s+\"", tf.read_text(encoding="utf-8"), re.M), tf.name
    example = (TF / "terraform.tfvars.example").read_text(encoding="utf-8")
    assert 'project_id      = "your-project-id"' in example
    assert 'billing_account = "000000-000000-000000"' in example
    assert '["203.0.113.10/32"]' in example  # documentation range (RFC 5737), not a real address
    lock = (TF / ".terraform.lock.hcl").read_text(encoding="utf-8")
    pinned = re.search(r'version\s*=\s*"= ([\d.]+)"', (TF / "versions.tf").read_text(encoding="utf-8")).group(1)
    assert f'version     = "{pinned}"' in lock
