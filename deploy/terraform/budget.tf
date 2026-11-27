# Budget alert for the test project. A budget only sends e-mails (to the billing account's administrators and
# users); it never stops or caps spending. The teardown (deploy/README.md) is what stops the cost.
#
# Credits excluded: on a Free Trial every cost is paid from the credit, so the net spend stays at zero and an alert
# on it would never fire. Alerts on the gross cost show how fast the credit is being used.
resource "google_billing_budget" "test" {
  billing_account = var.billing_account
  display_name    = "milk10k GKE test"

  budget_filter {
    projects               = ["projects/${data.google_project.this.number}"]
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.25
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }
  # Also when the month's forecast reaches the budget (e.g. a cluster left running).
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  depends_on = [google_project_service.apis]
}
