# Operations and recovery

These commands distinguish local preparation, AWS execution and model fitting. The repaired infrastructure and C1 v2 cloud-data reconciliation are deployed and verified; the actual training image passed its no-fit bootstrap. Model fitting and live batch delivery remain unexecuted. Preserve the legacy lake and historical `configs/*reconciliation.json` records.

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

## Before the next Terraform apply

There are two separate deployment scopes. The storage/Glue/Athena repair may be applied with batch scheduling disabled, without first training a model. Enabling the complete prediction workflow additionally requires a real accepted model and verified batch delivery. Do not treat either the architecture drawing or Terraform validation as proof that the entire workflow is ready.

Local checks executed on 2026-11-11: 79 no-real-fit tests passed; Ruff lint and formatting of the 14 touched Python files passed; the actual Lambda ZIP imported and submitted through a fake client under isolated Python with site packages disabled and without loading pandas/numpy/pyarrow/boto3. The prepared cloud snapshot's 30 objects, diagram and IAM resource blocks remained unchanged. Terraform formatting/validation passed. The original IAM-user session was subsequently renewed without adopting another principal. Live STS, resource refresh and saved-plan review then succeeded. The reviewed plan contains 2 additions, 4 in-place updates, 0 deletions/replacements and 0 IAM changes. It binds the verified Glue snapshot and final Lambda ZIP and keeps Scheduler disabled. The live project group and existing USD 30 monthly alert budget were checked; that alert is not a lifetime hard cap. Evidence: ignored `data/reviews/batch_apply_readiness_verified_20261111.json` and `waymark_code_plan_verified_20261111.*`. The exact saved plan was subsequently applied on user order: 2 added, 4 changed, 0 destroyed. AWS readback passed and a post-apply plan with the same variables returned No changes. Evidence: `data/reviews/waymark_apply_verified_20261111.json` and `waymark_postapply_plan_verified_20261111.txt`.

| Scope | Remaining preparation before declaring it ready |
| --- | --- |
| Storage/Glue/Athena repair | APPLIED and configuration readback verified. C1 v2 is now VERIFIED: the immutable script/snapshot are uploaded, Glue succeeded, and Catalog keys plus full feature rows reconciled with local data. Scheduler remains disabled. Do not replace this with an unreviewed default-variable apply; the default null script is not runnable. |
| Local prediction-code completion | Implemented and locally tested: accepted model/input binding, serving/package handoff, successful-job finalization, JSONLines normalization, exact order-ID coverage, conditional publication and committed-object/table resolution. The model/image/environment in tests are synthetic, not actual acceptance receipts. See [batch delivery](batch-delivery.md). |
| Complete scheduled prediction workflow | Obtain the actual accepted model and matching serving environment; supply its content-addressed batch manifest; verify the explicit CLI completion/publication runtime and committed-object Athena table flow against real output. An automatic completion trigger is not provisioned. Review any resulting Terraform changes and the saved plan, then perform separately authorized cloud verification before enabling the schedule. |

Local code and counterexamples are now prepared. The Lambda ZIP includes its package import closure and Terraform still preserves the existing IAM blocks. The schedule takes `batch_manifest_s3_key` instead of hard-coded model/hash placeholders, and refuses enablement without a prepared descriptor. The training source was repackaged separately as `data/handoffs/repaired_training_source_v3.tar.gz`, preserving previous versions. The original-user session renewal and final saved-plan review are complete. The user-authorized saved-plan apply is complete and the disabled-schedule infrastructure configuration is verified. The new snapshot upload and live C1 v2 acceptance are still pending; the full prediction workflow is not accepted. No fitting is required or allowed to validate local infrastructure/code preparation.

Only after the selected scope has no remaining prerequisites and its final plan is reviewed should the status be reported as **ready, awaiting the user's Terraform apply command**. Applying infrastructure does not upload the snapshot, replay C1, start training or establish WM4 acceptance; those remain separate actions.

## Training environment and source preparation

The actual Python 3.12 training image and no-fit Toolkit bootstrap are now prepared and verified. See the [training image guide](../infra/training/README.md) for the immutable ECR digest, real environment/source receipts, known scan findings and quota limits. This establishes container preparation, not a SageMaker job or model acceptance. Sydney `ml.m5.large` Spot and on-demand training quotas were both zero at the preparation checkpoint; no fitting was launched.

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

The CLI-managed `waymark-training` ECR repository and both immutable image candidates are outside Terraform state. A Terraform teardown alone does not remove them. Include their retained storage and any snapshot-specific Catalog databases in the inventory; image deletion or Terraform adoption is a separate reviewed operation.


## Applied checkpoint and training prerequisites (2026-11-11)

The user ordered the exact reviewed saved-plan apply. It completed with 2 additions, 4 updates and no deletions or IAM changes. S3 versioning/HTTPS-only policy, bucket destruction protection, the immutable Glue URI, the verified Lambda ZIP and the disabled Scheduler payload are deployed. Live AWS readback and the correctly parameterized post-apply plan passed. No further apply is required for this unchanged configuration. Do not reapply the consumed saved plan.

C1 v2 subsequently completed through AWS CLI: snapshot `2ee4a4d9923f7614e0565631f7b121675cd45a23e24b8127d98db8f9936b210f` is uploaded and independently verified; one Glue run succeeded in 15 seconds at 0.0625 DPU; all 15 Catalog table key populations, monetary/split/corridor checks and all 99,441 model-input rows reconciled. The new local/S3 receipt is `data/cloud_runs/2ee4a4d9923f7614e0565631f7b121675cd45a23e24b8127d98db8f9936b210f/jr_2f0e833e2af807372fa596dcf7c6579cd0a5c31617535eb504287aca33595511.json`; readback bytes matched. Use `waymark run-c1 --aws-cli` for a separately authorized new replay. After a completed verifier but unfinished reconciliation, `--resume-glue-run RUN_ID` reuses only an exact snapshot-bound run without resubmitting Glue. After acceptance, retain its immutable receipt rather than replacing it through the same run ID. Legacy receipts remain historical.

Evidence: ignored `data/reviews/waymark_code_apply_20261111.txt`, `waymark_apply_verified_20261111.json`, `waymark_postapply_plan_verified_20261111.txt` and `terraform_apply_training_review_20261111.md`. The latest complete local suite passed 91 no-real-fit tests; the real container import/Toolkit bootstrap passed separately. The actual environment receipt is `data/handoffs/verified_training_environment_v2_20261111.json`; the selected image and 2 remaining HIGH glibc scan findings are documented in the training image guide. Training is blocked by zero selected-instance quotas. Service-side startup, Spot recovery, fits and scientific acceptance remain NOT RUN. Future Terraform plans must retain the immutable `glue_snapshot_script_uri` and `enable_batch_schedule=false` until a different reviewed rollout is authorized.

Latest completion evidence: ignored `data/reviews/c1_training_image_closeout_20261111.md`, `waymark_cloud_state_after_preparation_20261111.json` and `c1_v2_awscli_resumed_20261111.txt`. Actual C1 usage is unpriced until verified regional rates/billing are supplied; the monthly USD 30 alert is not a lifetime hard cap. No fitting, holdout evaluation, batch transform, schedule enablement, IAM change or Git publication occurred in this step.
