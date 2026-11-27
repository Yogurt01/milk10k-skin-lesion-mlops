# Read by scripts/gke.sh (make gke-*). None of these is a secret.

output "project_id" {
  value = var.project_id
}

output "zone" {
  value = var.zone
}

output "cluster_name" {
  value = google_container_cluster.this.name
}

output "image_repository" {
  description = "Image name without a tag, for docker push and the Helm value image.repository."
  value       = "${google_artifact_registry_repository.images.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/milk10k-api"
}

output "model_bucket" {
  description = "Private bucket that holds the model package during the test."
  value       = google_storage_bucket.model.name
}
