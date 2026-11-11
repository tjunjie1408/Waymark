data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

check "caller_account_matches_o7_contract" {
  assert {
    condition     = data.aws_caller_identity.current.account_id == var.expected_account_id
    error_message = "Active AWS caller account (${data.aws_caller_identity.current.account_id}) does not match expected_account_id (${var.expected_account_id})."
  }
}

locals {
  name_prefix        = "${var.project_name}-${var.environment}"
  lake_bucket_name   = "${var.project_name}-lake-${var.environment}-${var.expected_account_id}-${var.aws_region}"
  glue_database_name = "${var.project_name}_${var.environment}_db"
  athena_wg_name     = "${var.project_name}_${var.environment}_wg"
}

# -----------------------------------------------------------------------------
# 1. AWS Resource Group (U1 Project Grouping in Code)
# -----------------------------------------------------------------------------
resource "aws_resourcegroups_group" "waymark" {
  name        = "${local.name_prefix}-project-group"
  description = "Waymark ecommerce fulfillment DE and DS bounded batch system resources."

  resource_query {
    query = jsonencode({
      ResourceTypeFilters = ["AWS::AllSupported"]
      TagFilters = [
        {
          Key    = "Project"
          Values = [var.project_name]
        }
      ]
    })
  }
}

# -----------------------------------------------------------------------------
# 2. S3 Medallion Lake Bucket (Bronze / Silver / Gold / Staging / Artifacts)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "waymark_lake" {
  bucket        = local.lake_bucket_name
  force_destroy = false

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "waymark_lake" {
  bucket = aws_s3_bucket.waymark_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_policy" "require_tls" {
  bucket = aws_s3_bucket.waymark_lake.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.waymark_lake.arn, "${aws_s3_bucket.waymark_lake.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_public_access_block" "waymark_lake" {
  bucket = aws_s3_bucket.waymark_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "waymark_lake" {
  bucket = aws_s3_bucket.waymark_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "waymark_lake" {
  bucket = aws_s3_bucket.waymark_lake.id

  rule {
    id     = "expire-ephemeral-staging-and-athena-results"
    status = "Enabled"

    filter {
      prefix = "staging/"
    }

    expiration {
      days = 7
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  rule {
    id     = "expire-athena-query-results"
    status = "Enabled"

    filter {
      prefix = "athena-results/"
    }

    expiration {
      days = 7
    }
  }
}

# -----------------------------------------------------------------------------
# 3. Glue Data Catalog & Right-Sized 0.0625 DPU Python Shell Job (O7)
# -----------------------------------------------------------------------------
resource "aws_glue_catalog_database" "waymark" {
  name        = local.glue_database_name
  description = "Waymark Bronze, Silver, and Gold fulfillment analytical tables."
}

resource "aws_iam_role" "glue_python_shell" {
  name = "${local.name_prefix}-glue-pythonshell-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "glue_python_shell" {
  name = "${local.name_prefix}-glue-pythonshell-policy"
  role = aws_iam_role.glue_python_shell.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "LakeBucketReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.waymark_lake.arn,
          "${aws_s3_bucket.waymark_lake.arn}/*"
        ]
      },
      {
        Sid    = "GlueCatalogAccess"
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:CreateTable",
          "glue:UpdateTable",
          "glue:BatchCreatePartition",
          "glue:GetPartition",
          "glue:GetPartitions"
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:${var.expected_account_id}:catalog",
          "arn:aws:glue:${var.aws_region}:${var.expected_account_id}:database/${aws_glue_catalog_database.waymark.name}",
          "arn:aws:glue:${var.aws_region}:${var.expected_account_id}:table/${aws_glue_catalog_database.waymark.name}/*"
        ]
      },
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${var.aws_region}:${var.expected_account_id}:log-group:/aws-glue/python-jobs/*"
        ]
      }
    ]
  })
}

resource "aws_glue_job" "bronze_silver_gold_etl" {
  name         = "${local.name_prefix}-medallion-pythonshell"
  role_arn     = aws_iam_role.glue_python_shell.arn
  max_capacity = var.glue_python_shell_dpu
  timeout      = 15
  max_retries  = 0

  command {
    name            = "pythonshell"
    python_version  = "3.9"
    script_location = var.glue_snapshot_script_uri != null ? var.glue_snapshot_script_uri : "s3://${aws_s3_bucket.waymark_lake.id}/scripts/UNPREPARED_DO_NOT_RUN.py"
  }

  default_arguments = {
    "--job-language"        = "python"
    "--TempDir"             = "s3://${aws_s3_bucket.waymark_lake.id}/staging/glue-temp/"
    "--WAYMARK_LAKE_BUCKET" = aws_s3_bucket.waymark_lake.id
    "--WAYMARK_GLUE_DB"     = aws_glue_catalog_database.waymark.name
  }
}

# -----------------------------------------------------------------------------
# 4. Athena Workgroup with 1 GB Per-Query Scan Cutoff Guard (O7)
# -----------------------------------------------------------------------------
resource "aws_athena_workgroup" "waymark" {
  name          = local.athena_wg_name
  state         = "ENABLED"
  force_destroy = true

  configuration {
    bytes_scanned_cutoff_per_query     = var.athena_scan_cutoff_bytes
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = false

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }

    result_configuration {
      output_location = "s3://${aws_s3_bucket.waymark_lake.id}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

# -----------------------------------------------------------------------------
# 5. SageMaker Execution Role for Remote Training & Batch Transform (O6/O7/O8)
# -----------------------------------------------------------------------------
resource "aws_iam_role" "sagemaker_execution" {
  name = "${local.name_prefix}-sagemaker-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_execution" {
  name = "${local.name_prefix}-sagemaker-exec-policy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3LakeAccess"
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.waymark_lake.arn,
          "${aws_s3_bucket.waymark_lake.arn}/*"
        ]
      },
      {
        Sid    = "ECRContainerPull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogsAndMetrics"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# -----------------------------------------------------------------------------
# 6. EventBridge Scheduler -> Lambda Idempotency Adapter -> Batch Transform (O8)
# -----------------------------------------------------------------------------
data "archive_file" "batch_adapter_zip" {
  type        = "zip"
  output_path = "${path.module}/build/batch_adapter.zip"

  source {
    content  = "from waymark.batch_adapter import lambda_handler\n"
    filename = "batch_adapter.py"
  }
  dynamic "source" {
    for_each = toset(["__init__.py", "batch_adapter.py", "batch_contract.py", "batch_delivery.py", "feature_schema.py", "training_environment.py", "artifacts.py", "remote_requirements.txt"])
    content {
      content  = file("${path.module}/../../src/waymark/${source.value}")
      filename = "waymark/${source.value}"
    }
  }
}

resource "aws_cloudwatch_log_group" "batch_adapter" {
  name              = "/aws/lambda/${local.name_prefix}-batch-adapter"
  retention_in_days = 7
}

resource "aws_iam_role" "batch_adapter_lambda" {
  name = "${local.name_prefix}-batch-adapter-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "batch_adapter_lambda" {
  name = "${local.name_prefix}-batch-adapter-lambda-policy"
  role = aws_iam_role.batch_adapter_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "IdempotencyMarkerAndStagingAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.waymark_lake.arn,
          "${aws_s3_bucket.waymark_lake.arn}/*"
        ]
      },
      {
        Sid    = "SageMakerBatchTransformControl"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTransformJob",
          "sagemaker:DescribeTransformJob",
          "sagemaker:ListTransformJobs",
          "sagemaker:AddTags"
        ]
        Resource = "*"
      },
      {
        Sid      = "PassSageMakerExecutionRole"
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
      },
      {
        Sid    = "LambdaLogging"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.batch_adapter.arn}:*"
      }
    ]
  })
}

resource "aws_lambda_function" "batch_adapter" {
  function_name    = "${local.name_prefix}-batch-adapter"
  description      = "Waymark O8 idempotent trigger adapter for SageMaker Batch Transform."
  role             = aws_iam_role.batch_adapter_lambda.arn
  filename         = data.archive_file.batch_adapter_zip.output_path
  source_code_hash = data.archive_file.batch_adapter_zip.output_base64sha256
  handler          = "batch_adapter.lambda_handler"
  runtime          = "python3.12"
  architectures    = ["arm64"]
  memory_size      = 128
  timeout          = 30

  environment {
    variables = {
      WAYMARK_LAKE_BUCKET   = aws_s3_bucket.waymark_lake.id
      WAYMARK_INSTANCE_TYPE = var.sagemaker_instance_type
    }
  }

  depends_on = [aws_cloudwatch_log_group.batch_adapter]
}

resource "aws_iam_role" "scheduler_invoke_lambda" {
  name = "${local.name_prefix}-scheduler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "scheduler.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "scheduler_invoke_lambda" {
  name = "${local.name_prefix}-scheduler-invoke-policy"
  role = aws_iam_role.scheduler_invoke_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "lambda:InvokeFunction"
        Resource = aws_lambda_function.batch_adapter.arn
      }
    ]
  })
}

resource "aws_scheduler_schedule" "daily_batch_scoring" {
  name        = "${local.name_prefix}-daily-batch-scoring"
  description = "Scheduled trigger for Waymark O8 Batch Transform adapter (DISABLED by default)."
  state       = var.enable_batch_schedule ? "ENABLED" : "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = "cron(0 2 * * ? *)"

  target {
    arn      = aws_lambda_function.batch_adapter.arn
    role_arn = aws_iam_role.scheduler_invoke_lambda.arn

    input = jsonencode({
      bucket             = aws_s3_bucket.waymark_lake.id
      batch_manifest_key = var.batch_manifest_s3_key
      attempt_seq        = 1
    })

    retry_policy {
      maximum_event_age_in_seconds = 3600
      maximum_retry_attempts       = 1
    }
  }
  lifecycle {
    precondition {
      condition     = !var.enable_batch_schedule || var.batch_manifest_s3_key != null
      error_message = "An enabled schedule requires a prepared, content-addressed batch manifest."
    }
  }
}

# -----------------------------------------------------------------------------
# 7. Optional AWS Budgets Alert at USD 15 (50%) and USD 25 (83.3%) (U2)
# -----------------------------------------------------------------------------
resource "aws_budgets_budget" "waymark_cost_cap" {
  count = length(trimspace(var.budget_alert_email)) > 0 ? 1 : 0

  name              = "${local.name_prefix}-usd30-cap-alert"
  budget_type       = "COST"
  limit_amount      = var.budget_limit_usd
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2026-11-01_00:00"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [trimspace(var.budget_alert_email)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 83.33
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [trimspace(var.budget_alert_email)]
  }
}
