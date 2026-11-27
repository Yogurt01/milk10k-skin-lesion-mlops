# Terraform and provider pins. The lock file (.terraform.lock.hcl) is committed, so every `terraform init` installs
# exactly this provider build (checked against the hashes in that file).
#
# State: the default local backend (terraform.tfstate in this folder, gitignored, never printed). This is a
# short-lived, single-operator test: the state is kept until `make gke-down` has destroyed everything and the
# leftover check came back empty; the teardown checklist in deploy/README.md also works without it.
terraform {
  required_version = ">= 1.17.0, < 2.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 8.12.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  # Bill API quota to this project instead of the client's default: the Billing Budgets API refuses calls made with
  # user credentials (gcloud auth application-default login) without a quota project.
  billing_project       = var.project_id
  user_project_override = true

  # On every resource that supports labels (cluster, nodes, bucket, registry): lets the leftover check find them.
  default_labels = local.labels
}
