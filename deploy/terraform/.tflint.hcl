# tflint configuration (make tf-check). Static rules only: deep checking (calls to the Google Cloud API) stays off.
plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "google" {
  enabled = true
  version = "0.41.0"
  source  = "github.com/terraform-linters/tflint-ruleset-google"
}
