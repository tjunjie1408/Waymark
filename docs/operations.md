# Operations and recovery

These commands distinguish local preparation from AWS mutation and model fitting. No cloud deployment or training was performed by the repair task. Preserve the legacy lake and historical `configs/*reconciliation.json` records.

## Local verification

```powershell
uv sync --locked
uv run pytest
uv run ruff check src tests scripts
uv run waymark verify-wm0
uv run waymark prepare-cloud-snapshot --output data/handoffs/cloud_snapshot.json
```

Local tests contain no real estimator fitting. The explicitly fake estimator workflow test exercises evaluation, checkpoint bookkeeping and artifact validation only. The default suite excludes `cloud` tests. A clean clone can run unit tests without the ignored Olist data. Preparing a real snapshot requires the original local raw/Bronze/Silver/Gold artifacts and frozen summaries.

Reconstruction refuses existing output files. To investigate a changed transformation, select new `--bronze-dir`, `--silver-dir`, `--gold-dir`, `--features-dir` and `--export-summary` paths. Do not overwrite the accepted data or its manifests. Copying a new output into the old path is not a supported promotion mechanism.

## Future authorized cloud replay

1. Check the prepared snapshot inventory and its printed immutable Glue script URI. Confirm the dedicated account/Region and the USD 30 project budget, including retained resources. The account and Region checks now block mismatches.
2. Set Terraform `glue_snapshot_script_uri` to the printed URI and review the plan. A null value deliberately leaves the Glue job unprepared. Applying storage changes enables versioning and denies plaintext transport. `prevent_destroy` and `force_destroy = false` constrain Terraform destruction; they do not stop an administrator deleting data through AWS directly. Role permissions are unchanged. No apply occurred during local repair.
3. After separate cloud authorization, run `waymark run-c1 --profile waymark --region ap-southeast-2`. It uploads 30 required objects under `snapshots/<content-hash>/`, verifies each, runs the independent Glue verifier, registers a new snapshot-specific Catalog database, and compares table keys, sums, splits and full model-input rows. It preserves the legacy table/database paths.
4. New receipts go under `data/cloud_runs/<snapshot-id>/<Glue-run-id>.json` and the matching immutable S3 acceptance path. Historical receipts are not overwritten. A failed cloud step is not acceptance. Retain run/query IDs and inspect the remote job after a polling timeout; a local timeout does not establish remote cancellation.

The Glue job verifies transport/inventory and Catalog database access. It does not run a distributed ETL transformation or establish performance gains. Local reconstruction remains the producer. A dedicated cloud ETL experiment would require its own method and execution authorization.

The Terraform update also repackages the changed batch-adapter Python source through `archive_file`, so the Lambda code hash can change even though its resource block is unchanged. The batch schedule remains disabled by default. Actual changes and any drift must be reviewed in the plan; this document does not assert a live resource-change count.

For the already prepared local inventory, the future operator sequence from the repository root is:

```powershell
$preparedSnapshot = Get-Content -LiteralPath data/handoffs/repaired_cloud_snapshot_v1.json -Raw | ConvertFrom-Json
$preparedScriptUri = "s3://waymark-lake-pilot-149270211548-ap-southeast-2/snapshots/$($preparedSnapshot.snapshot_id)/scripts/glue_etl_entry.py"
.\data\aws_setup\bin\terraform.exe -chdir=infra/terraform plan "-var=glue_snapshot_script_uri=$preparedScriptUri" -out=review.tfplan
# Only after reviewing and authorizing the saved plan:
.\data\aws_setup\bin\terraform.exe -chdir=infra/terraform apply review.tfplan
# Separately authorized data transfer and cloud acceptance, not model fitting:
uv run waymark run-c1 --profile waymark --region ap-southeast-2
```

Do not apply the default null script variable as a usable replay configuration: it points to `UNPREPARED_DO_NOT_RUN.py`. Terraform apply does not upload the new snapshot or register its analytical tables; the subsequent C1 command performs those steps. Check U1/U2 and the remaining project budget before these AWS actions. Training remains a separate user-launched operation.

## Training environment and source preparation

Use a currently supported, reviewed training image in `ap-southeast-2`, pinned by its ECR `@sha256` digest. Install the exact `src/waymark/remote_requirements.txt` set inside that image. Project Python is 3.12. The source packager includes only the minimal package, requirements, wrapper and a file-hash inventory; private planning files and local credentials are absent.

Prepare the source locally using `waymark prepare-training-source --output data/handoffs/train_source.tar.gz`. Extract it inside the selected environment, then run the packaged entrypoint in no-fit mode:

```text
python train_entry.py --preflight-only --image-uri ECR_URI_AT_SHA256_DIGEST --preflight-output verified_environment.json
```

This checks exact installed versions and imports and records interpreter/package/image/requirements identity. It never fits. Image support, actual installation/import compatibility, quota and job startup are distinct checks; no such container execution was performed by local unit tests. A successful fabricated/mock receipt is not real environment evidence.

After a real environment check and accepted C1 v2 receipt, the user may launch a separately authorized development job:

```text
waymark train-remote --profile waymark --region ap-southeast-2 --cloud-receipt data/cloud_runs/SNAPSHOT/GLUE_RUN.json --environment-receipt data/handoffs/verified_environment.json
```

The default run evaluates development only and reports `WM3_INCOMPLETE`. `--allow-holdout` is explicit authorization for final holdout scoring after the development freeze is written. Freeze the protocol before using it; do not rerun/adapt methods after seeing holdout scores under the same experiment identity. The repaired method is `waymark.training/v2`, with unchanged frozen label/split/smoothing values, shared baseline rules, explicit native categorical handling, endpoint-safe ECE and deterministic top-k tie breaking. Seller ID stays a lineage/bootstrap identity rather than a direct predictor.

Model packages contain a development freeze, native models/calibrator, per-order predictions, baseline predictions, quantile/calibration/ranking metrics, paired seller-week bootstrap comparisons and deterministic TreeSHAP/background samples. SHAP describes raw model associations, not causal benefits or calibrated probability decomposition. Model artifacts and evaluation results are checked before the launcher returns success. Gate acceptance always belongs to the user.

Each job has its own local request/result/failure records and S3 output/checkpoint prefixes. Completed fit stages can be restored within a compatible job restart; an interrupted fit restarts, with uncheckpointed overhead unknown. A new job is a fresh budget attempt. Logs and failed/uncertain submission records must be retained. No job is silently retried by the launcher after uncertain submission.

## Usage, publication and cleanup

Glue usage reads `MaxCapacity` and applies the 60-second billing minimum. Athena records billable bytes per query, including rounding/minimums and exempt DDL. SageMaker cost uses billable instance-hours with the on-demand rate convention; it does not apply a second Spot discount. Supply a dated regional rate/source for an estimate. Without verified pricing, costs remain `UNPRICED_USAGE`, not zero. Reconcile storage, logs, requests, state, failed runs and retained resources with billing before making project-wide cost claims.

The future batch publisher reads actual Parquet, checks keys/schema/finite probabilities/quantile order, writes a content-addressed object and conditionally creates the commit manifest. Readers must resolve the manifest's exact `gold_s3_uri` and hash; scanning every Parquet under a batch prefix is unsafe because uncommitted candidate objects may remain. The first committed logical batch is immutable. Scheduled end-to-end integration and its Athena reader remain separate WM4 work; the schedule stays disabled.

Before teardown, verify backups and inventory retained snapshots/models/receipts. Lake deletion is deliberately protected. Remove that protection only as part of an explicitly approved cleanup after backup verification, then inspect the deletion result. Closing a browser or local command does not stop remote jobs or billing.
