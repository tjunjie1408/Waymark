# Waymark

An end-to-end Data Engineering and Data Science batch system for e-commerce order fulfillment on AWS. At order approval time, Waymark transforms relational marketplace data into an analytical star schema and point-in-time features, estimates delivery duration quantiles (ETA) and delay risk against platform/historical baselines, and publishes scheduled batch predictions for SQL and BI queries.

## Intended workflow

Olist relational source snapshot -> validated Bronze/Silver Parquet -> analytical star schema -> point-in-time features at order approval -> remote ETA quantile and delay-risk evaluation -> scheduled batch predictions -> Athena/BI queries and cost-reconciled reports.

- **Prediction cutoff**: order approval time, before dispatch decisions are made.
- **Stack direction**: Python, Terraform, S3, Glue, Athena, SageMaker managed training and Batch Transform, EventBridge (AWS `us-east-1`, bounded within a USD 30 total cloud budget).
- **Execution model**: local-first data engineering and leakage verification, paired with remote-only managed model training and ephemeral batch scoring.

## Status

Workspace initialized (2026-11-09; planning v0.2 updated 2026-11-10). There is no implemented package, deployed infrastructure, working installation or demo command yet. Data ingestion, model evaluation, cloud integration and performance/cost measurements have not run.

## Evidence standard

Every stage preserves source/data/configuration hashes, reconciles relational joins across multi-item and multi-seller orders, enforces approval-time information boundaries, compares against credible baselines on a frozen temporal holdout, and reports actual AWS resource usage, costs and teardown readback. No model score, speedup, cost saving, calibrated interval or causal business effect is claimed prior to execution.

Public source modules, executable configurations, tests and reconstruction/operations guides will be added as each stage is authorized and verified.
