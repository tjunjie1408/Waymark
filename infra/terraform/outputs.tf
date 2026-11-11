output "caller_account_id" {
  description = "Verified AWS account ID."
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "Active AWS Region."
  value       = data.aws_region.current.region
}

output "resource_group_arn" {
  description = "ARN of the Waymark AWS Resource Group (U1)."
  value       = aws_resourcegroups_group.waymark.arn
}

output "lake_bucket_name" {
  description = "Name of the S3 medallion lake bucket."
  value       = aws_s3_bucket.waymark_lake.id
}

output "lake_bucket_arn" {
  description = "ARN of the S3 medallion lake bucket."
  value       = aws_s3_bucket.waymark_lake.arn
}

output "glue_database_name" {
  description = "Name of the Glue Catalog database."
  value       = aws_glue_catalog_database.waymark.name
}

output "glue_python_shell_job_name" {
  description = "Name of the 0.0625 DPU Glue Python Shell ETL job."
  value       = aws_glue_job.bronze_silver_gold_etl.name
}

output "athena_workgroup_name" {
  description = "Name of the Athena Engine v3 workgroup."
  value       = aws_athena_workgroup.waymark.name
}

output "sagemaker_execution_role_arn" {
  description = "IAM role ARN for SageMaker remote training and Batch Transform."
  value       = aws_iam_role.sagemaker_execution.arn
}

output "batch_adapter_lambda_arn" {
  description = "ARN of the O8 EventBridge-to-BatchTransform idempotency Lambda adapter."
  value       = aws_lambda_function.batch_adapter.arn
}

output "batch_scheduler_arn" {
  description = "ARN of the EventBridge Scheduler schedule (DISABLED by default)."
  value       = aws_scheduler_schedule.daily_batch_scoring.arn
}
