variable "project_name" {
  description = "Canonical project identifier used in resource names and tags."
  type        = string
  default     = "waymark"
}

variable "glue_snapshot_script_uri" {
  description = "Content-addressed script URI printed by waymark prepare-cloud-snapshot. Null keeps the job unprepared."
  type        = string
  default     = null
  validation {
    condition     = var.glue_snapshot_script_uri == null ? true : can(regex("^s3://waymark-lake-pilot-149270211548-ap-southeast-2/snapshots/[0-9a-f]{64}/scripts/glue_etl_entry.py$", var.glue_snapshot_script_uri))
    error_message = "Glue must use the prepared immutable Waymark snapshot script."
  }
}

variable "environment" {
  description = "Deployment stage label (pilot for bounded C1-T9 execution)."
  type        = string
  default     = "pilot"
}

variable "aws_region" {
  description = "Target AWS Region frozen in O7 (Sydney ap-southeast-2)."
  type        = string
  default     = "ap-southeast-2"

  validation {
    condition     = var.aws_region == "ap-southeast-2"
    error_message = "Waymark O7 contract restricts execution to ap-southeast-2."
  }
}

variable "aws_profile" {
  description = "AWS CLI profile bound to account 149270211548."
  type        = string
  default     = "waymark"
}

variable "expected_account_id" {
  description = "Expected 12-digit AWS account ID frozen in O7."
  type        = string
  default     = "149270211548"

  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_account_id))
    error_message = "expected_account_id must be a 12-digit AWS account number."
  }
}

variable "glue_python_shell_dpu" {
  description = "Right-sized Glue Python Shell DPU allocation frozen in O7 (0.0625 DPU)."
  type        = number
  default     = 0.0625

  validation {
    condition     = var.glue_python_shell_dpu == 0.0625
    error_message = "Waymark O7 freezes Glue Python Shell capacity at 0.0625 DPU."
  }
}

variable "athena_scan_cutoff_bytes" {
  description = "Per-query scan limit in bytes for Athena workgroup cost protection (default 1 GB)."
  type        = number
  default     = 1073741824
}

variable "sagemaker_instance_type" {
  description = "SageMaker instance type for remote training and Batch Transform (ml.m5.large)."
  type        = string
  default     = "ml.m5.large"
}

variable "enable_batch_schedule" {
  description = "Whether the EventBridge Scheduler for Batch Transform is ENABLED. Defaults to false (DISABLED) to prevent unattended charges."
  type        = bool
  default     = false
}

variable "batch_manifest_s3_key" {
  description = "Content-addressed accepted batch descriptor; leave null with the schedule disabled."
  type        = string
  default     = null
  validation {
    condition     = var.batch_manifest_s3_key == null ? true : can(regex("^batch/manifests/[0-9a-f]{64}\\.json$", var.batch_manifest_s3_key))
    error_message = "Use the exact batch/manifests/SHA256.json key prepared by waymark prepare-batch."
  }
}

variable "budget_limit_usd" {
  description = "Hard total project cloud spend cap in USD (PRD F3)."
  type        = string
  default     = "30.0"
}

variable "budget_alert_email" {
  description = "Optional email address to receive USD 15 (50%) and USD 25 (83.3%) AWS Budgets alerts (U2). When empty, the budget resource is skipped in Terraform or managed via AWS Console/CLI."
  type        = string
  default     = ""
}
