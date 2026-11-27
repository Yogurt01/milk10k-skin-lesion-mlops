# GKE test environment for the MILK10k API (deploy/README.md): APIs, an image registry, a private bucket for the
# model package, a least-privilege node identity, Workload Identity for the API Pods, a zonal Standard cluster with
# one fixed node pool. Research model, not a medical device. The model package is CC BY-NC 4.0 (MILK study team),
# non-commercial only: it lives in the private bucket only while the test runs and is removed with it.

locals {
  labels = {
    app     = "milk10k"
    purpose = "course-demo"
  }

  apis = toset([
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com",
  ])

  # Workload Identity Federation for GKE: the API's Kubernetes ServiceAccount as an IAM principal (no Google service
  # account, no key file).
  api_principal = format(
    "principal://iam.googleapis.com/projects/%s/locations/global/workloadIdentityPools/%s.svc.id.goog/subject/ns/%s/sa/%s",
    data.google_project.this.number, var.project_id, var.k8s_namespace, var.k8s_service_account,
  )
}

data "google_project" "this" {
  project_id = var.project_id
}

# Kept enabled on destroy: APIs cost nothing, and the project itself is deleted at the end of the test.
resource "google_project_service" "apis" {
  for_each = local.apis

  service            = each.value
  disable_on_destroy = false
}

# --- Images -------------------------------------------------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "milk10k"
  format        = "DOCKER"
  description   = "MILK10k API images (no model weights inside)"

  # A pushed tag can never point to another image later (the chart refuses "latest" anyway).
  docker_config {
    immutable_tags = true
  }

  depends_on = [google_project_service.apis]
}

# --- Model package ------------------------------------------------------------------------------------------------

# Private: uniform bucket-level access (IAM only, no object ACLs) and public access prevention enforced, so no
# object can ever be made public. force_destroy and no soft delete: `terraform destroy` removes the bucket with the
# model inside and nothing stays recoverable or billable afterwards.
resource "google_storage_bucket" "model" {
  name          = "${var.project_id}-milk10k-model"
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  versioning {
    enabled = false
  }

  soft_delete_policy {
    retention_duration_seconds = 0
  }

  depends_on = [google_project_service.apis]
}

# The API Pods read the package; nothing else in the cluster can.
resource "google_storage_bucket_iam_member" "api_reads_model" {
  bucket = google_storage_bucket.model.name
  role   = "roles/storage.objectViewer"
  member = local.api_principal

  # The workload identity pool exists only once the cluster does.
  depends_on = [google_container_cluster.this]
}

# --- Node identity ------------------------------------------------------------------------------------------------

# Instead of the Compute Engine default service account (Editor on the project).
resource "google_service_account" "nodes" {
  account_id   = "milk10k-gke-nodes"
  display_name = "MILK10k GKE nodes"

  depends_on = [google_project_service.apis]
}

# Logging, monitoring and metadata writes that every GKE node needs.
resource "google_project_iam_member" "nodes_default" {
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
  member  = "serviceAccount:${google_service_account.nodes.email}"
}

# Pull the API image.
resource "google_artifact_registry_repository_iam_member" "nodes_pull" {
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.nodes.email}"
}

# --- Cluster ------------------------------------------------------------------------------------------------------

resource "google_container_cluster" "this" {
  name     = var.cluster_name
  location = var.zone # zonal: one control plane in one zone

  deletion_protection = false

  # The default pool is created and removed again at once; the node pool below is the only one. Its short-lived
  # nodes still use the dedicated identity and a small standard disk (not counted against the SSD quota).
  remove_default_node_pool = true
  initial_node_count       = 1
  node_config {
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    disk_size_gb    = 30
    disk_type       = "pd-standard"
  }

  network         = "default"
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # The control plane answers only from these networks (kubectl and Helm from your machine).
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.admin_cidrs
      content {
        cidr_block   = cidr_blocks.value
        display_name = "admin-${cidr_blocks.key}"
      }
    }
  }

  # System logs and metrics only; Prometheus and Grafana run in the cluster (kube-prometheus-stack), so Google's
  # managed collection is off (it would run its own collectors on every node).
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = false
    }
  }

  addons_config {
    # Only for Ingress (layer 7); the API uses a LoadBalancer Service (layer 4) or a port-forward.
    http_load_balancing {
      disabled = true
    }
  }

  lifecycle {
    # node_config describes the removed default pool only.
    ignore_changes = [node_config]
  }

  depends_on = [google_project_service.apis, google_project_iam_member.nodes_default]
}

resource "google_container_node_pool" "api" {
  name       = "api"
  cluster    = google_container_cluster.this.id
  location   = var.zone
  node_count = var.node_count

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # No extra node during an upgrade: one more e2-standard-4 would exceed an 8 vCPU quota.
  upgrade_settings {
    strategy        = "SURGE"
    max_surge       = 0
    max_unavailable = 1
  }

  node_config {
    machine_type    = var.node_machine_type
    disk_size_gb    = var.node_disk_size_gb
    disk_type       = var.node_disk_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    # Pods see the GKE metadata server (Workload Identity), never the node's own credentials.
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    resource_labels = local.labels
  }
}
