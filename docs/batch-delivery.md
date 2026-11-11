# Batch delivery: prepare, submit, finalize and consume

The code provides a prediction-only serving entrypoint and explicit post-job CLI operations. No model fitting, serving-image build, model registration, batch input/model upload, transform job or scheduled end-to-end acceptance was performed by this implementation. C1 data verification and the separate training-image preparation are recorded in the [operations guide](operations.md). Synthetic tests use estimators whose `fit` raises an error.

## Runtime ownership

- **Submission:** the existing Python 3.12 arm64 Lambda retains its 128 MB / 30-second configuration. Its ZIP contains the thin handler and standard-library modules; boto3 is supplied by Lambda. It verifies the batch descriptor, streaming object hashes and existing SageMaker model binding, then submits once. It never imports pandas on that path, waits for completion or attempts publication.
- **Prediction:** `infra/inference/Dockerfile` prepares a separate Linux AMD64 Python 3.12 image for `ml.m5.large`. Choose a reviewed, digest-pinned Debian Python 3.12 base at build time. The image installs the exact existing `remote_requirements.txt` versions and the OpenMP runtime, then starts `waymark.batch_inference`. It implements `/ping` and `/invocations` on port 8080, accepts `serve`, and contains no fitting or tuning operation. No image was built or pushed here.
- **Completion and reading:** the existing project Python 3.12 environment runs the explicit CLI. It has pandas/pyarrow and can normalize the bounded output inventory, commit Parquet and expose an exact-object Athena table. There is no automatic completion trigger provisioned by this change. A successful scheduled submission alone does not complete publication.

The container must pass its real no-fit preflight in the selected digest-pinned image. For example, after a separately authorized build, run the image with `--preflight-only --image-uri YOUR_ECR_URI_AT_SHA256_DIGEST` and retain the JSON receipt. Do not substitute a test receipt. The serving package versions must match the frozen model environment. Import checks, server startup, actual model loading and real Batch Transform compatibility are separate checks.

## Model acceptance and local preparation

The preparation command needs an actual reviewed model archive, its compatible serving environment receipt and a user-owned acceptance JSON. The acceptance structure is:

```json
{
  "schema": "waymark.model_acceptance/v1",
  "decision": "REVIEW_REQUIRED",
  "review_reference": "REFERENCE_TO_USER_WM3_REVIEW",
  "model_sha256": "SHA256_OF_REVIEWED_MODEL_TAR_GZ",
  "feature_schema_sha256": "CURRENT_FEATURE_SCHEMA_SHA256",
  "image_uri": "DEDICATED_REGIONAL_ECR_URI_AT_SHA256_DIGEST"
}
```

This template is intentionally rejected. Only the user's actual acceptance uses `decision: accepted`. A completed training job is not acceptance. The feature-schema hash is available as `waymark.batch_contract.FEATURE_SCHEMA_SHA256`; do not compute it from a reordered feature list.

Example local command, with paths and the role ARN filled from the reviewed project artifacts:

```text
waymark prepare-batch --features SCORING_FEATURES.parquet --model-archive MODEL.tar.gz --acceptance ACCEPTANCE.json --environment-receipt SERVING_ENVIRONMENT.json --model-name UNIQUE_ACCEPTED_MODEL_NAME --scoring-cutoff-date YYYY-MM-DD --execution-role-arn EXISTING_SAGEMAKER_ROLE_ARN --output NEW_BUNDLE_DIRECTORY
```

The command does not contact AWS or fit. It selects only `order_id` plus the frozen model features, drops label/evaluation columns, preserves string order IDs and train-only missing/category handling, and writes headerless CSV parts. Embedded newlines are rejected because Batch Transform splits records on lines. Each CSV, model archive, acceptance and AWS input manifest has a content-addressed S3 key. The batch descriptor binds the inventory, complete order-ID sets, model name/archive/image/schema/environment, cutoff and config. Its own key is `batch/manifests/SHA256.json`. An existing output directory is refused.

The bundle includes `batch.json`, `create_model.json`, `bundle.json` and an `objects/` tree. `load_bundle` checks this inventory before later upload/registration. Keep the model acceptance and failed-attempt records; do not relabel a new model or batch as a retry of an old identity.

## Separately authorized AWS operations

These commands perform real AWS reads/writes or start billable compute. They are documented interfaces, not execution performed by the implementation task:

```text
waymark upload-batch --bundle NEW_BUNDLE_DIRECTORY --profile waymark
waymark register-batch-model --bundle NEW_BUNDLE_DIRECTORY --profile waymark
waymark submit-batch --manifest-key batch/manifests/SHA256.json --attempt 1 --profile waymark
```

Each cloud command verifies the dedicated account and Region. Model registration uses the exact prepared `CreateModel` request and refuses to invent a replacement name if an operation fails or its outcome is uncertain. Keep its request/result/failure ledger under ignored `data/batch_runs/`. Submission verifies the model's data URI, digest-pinned image and serving-image environment value. It streams model/input hashes, verifies the exact AWS input manifest and uses `ManifestFile` to avoid matching undeclared S3 prefix siblings. Existing commits or active jobs suppress duplicate submission; repeated attempts never modify an already committed logical batch.

Terraform receives `batch_manifest_s3_key` for a deliberately selected batch. Its default is null and the schedule remains disabled. Enabling without a content-addressed key fails a precondition. Repeated schedule invocations of the same descriptor target the same logical batch; advancing to another scoring population requires selecting its new descriptor. This is bounded snapshot delivery, not an implemented live order-discovery service.

## Completion and publication

After confirming the selected transform attempt has completed, the operator invokes:

```text
waymark finalize-batch --manifest-key batch/manifests/SHA256.json --attempt 1 --profile waymark
```

The finalizer independently checks `Completed`, requested model/input/output/resource identities and the complete raw-output object inventory. `InProgress`, failed/stopped/unknown states, missing/extra files and wrong IDs reject publication. Each raw file must be JSONLines records with explicit `order_id`, `eta_p10_days`, `eta_p50_days`, `eta_p90_days`, `prob_late` and integer `risk_alert_flag`. Output keys are checked against the corresponding input-key plus `.out` mapping described by AWS. Exact service key naming remains a real AWS integration check, not something local fake storage can prove.

Normalization uses the declared JSONLines contract; it does not guess from filename extensions. It verifies exact order-ID coverage and uniqueness, finite numeric values, quantile order and valid probabilities/flags, then emits typed Parquet. Row sequence is not a business key: consumers join on `order_id`. The first conditional commit manifest binds the exact object URI/hash and the full batch descriptor identity. A failed marker write may leave an isolated candidate, never a visible committed batch. Retrying reads and preserves an existing valid commit.

## Committed-only consumption

```text
waymark read-batch --manifest-key batch/manifests/SHA256.json --output NEW_LOCAL_PREDICTIONS.parquet --profile waymark
waymark register-batch-table --manifest-key batch/manifests/SHA256.json --database EXISTING_GLUE_DATABASE --profile waymark
```

The reader fetches the commit manifest and exactly its referenced object, verifies content and order IDs, and never scans the Gold candidate prefix. Local export uses an immutable output path. The table-registration command copies only verified committed bytes into `athena/committed/BATCH_ID/OBJECT_SHA256/predictions.parquet`, verifies that the isolated location has exactly that object, and registers a distinct external Parquet table. Existing incompatible tables are not overwritten. Athena queries the returned immutable table name, for example `SELECT * FROM database.predictions_BATCHID_HASHPREFIX`. No shared mutable latest-view pointer is silently updated.

Glue Catalog supplies schemas/locations; commit manifests select valid batches. Ordinary Parquet prefix scans do not interpret `_SUCCESS` as a transaction boundary. IAM policies remain user-owned and unchanged; these code paths have not established live permission coverage or WM4 acceptance.

References: [AWS Batch Transform input manifest](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TransformS3DataSource.html), [output mapping and assembly](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TransformOutput.html), [custom inference container contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html), [Glue CreateTable](https://docs.aws.amazon.com/boto3/latest/reference/services/glue/client/create_table.html), and [LightGBM OpenMP installation notes](https://lightgbm.readthedocs.io/en/v4.3.0/Installation-Guide.html).
