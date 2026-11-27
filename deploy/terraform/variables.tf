# Inputs. Real values go in terraform.tfvars (gitignored); terraform.tfvars.example lists them with placeholders.

variable "project_id" {
  description = "ID of the Google Cloud project used only for this test (it is deleted at the end)."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid project ID: 6-30 characters, lowercase letters, digits and hyphens."
  }
}

variable "billing_account" {
  description = "Billing account ID (XXXXXX-XXXXXX-XXXXXX) for the budget alert. Not a secret, but kept out of git."
  type        = string

  validation {
    condition     = can(regex("^[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}$", var.billing_account))
    error_message = "billing_account must look like 012345-6789AB-CDEF01."
  }
}

variable "admin_cidrs" {
  description = "Networks allowed to reach the cluster's control plane, usually your own public IP as x.x.x.x/32. Never 0.0.0.0/0."
  type        = list(string)

  validation {
    condition     = length(var.admin_cidrs) > 0 && alltrue([for c in var.admin_cidrs : can(cidrhost(c, 0)) && c != "0.0.0.0/0" && c != "::/0"])
    error_message = "admin_cidrs needs at least one valid CIDR and must not contain 0.0.0.0/0 or ::/0."
  }
}

variable "region" {
  description = "Region for the registry and the bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the cluster. A zonal cluster: the GKE free tier credit covers its management fee."
  type        = string
  default     = "us-central1-a"

  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]+-[a-z]$", var.zone))
    error_message = "zone must be a zone name such as us-central1-a (a region would make the cluster regional)."
  }
}

variable "cluster_name" {
  description = "GKE cluster name."
  type        = string
  default     = "milk10k"
}

variable "release_channel" {
  description = "GKE release channel."
  type        = string
  default     = "REGULAR"
}

variable "node_machine_type" {
  description = "Machine type of the node pool. 2 x e2-standard-4 = 8 vCPU, the likely CPU quota of a Free Trial project."
  type        = string
  default     = "e2-standard-4"
}

variable "node_count" {
  description = "Number of nodes (fixed, no autoscaling)."
  type        = number
  default     = 2

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 4
    error_message = "node_count must be between 1 and 4."
  }
}

variable "node_disk_size_gb" {
  description = "Boot disk size per node in GB (images, three 1.4 GB model copies, logs). Counts against the SSD quota."
  type        = number
  default     = 50
}

variable "node_disk_type" {
  description = "Boot disk type per node."
  type        = string
  default     = "pd-balanced"
}

variable "k8s_namespace" {
  description = "Namespace of the API (k8s/namespace.yaml)."
  type        = string
  default     = "milk10k"
}

variable "k8s_service_account" {
  description = "Kubernetes ServiceAccount of the API Pods (the Helm release name); gets read access to the model bucket."
  type        = string
  default     = "milk10k-api"
}

variable "budget_amount" {
  description = "Budget for the test project, in the billing account's currency (check it in the console: Billing > Account management)."
  type        = number
  default     = 20

  validation {
    condition     = var.budget_amount > 0
    error_message = "budget_amount must be positive."
  }
}

variable "budget_currency" {
  description = "Currency code of the billing account (e.g. USD). null = the account's own currency; a different code is refused by the API."
  type        = string
  default     = null
}
