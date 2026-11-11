# SageMaker training image

This image provides the AWS Training Toolkit `train` entrypoint. It is separate from `infra/inference/Dockerfile`, which serves predictions. Building or checking the image does not launch a model fit.

## Build and source ownership

The default base is an immutable Linux AMD64 official Python 3.12 Debian Trixie digest. The build applies available Debian security updates. `src/waymark/remote_requirements.txt` remains the sole owner of model dependency versions. The additional Training Toolkit release is pinned by version and source-archive SHA256 in the Dockerfile. The Dockerfile-specific ignore file limits the build context to these infrastructure files and the requirements file. Datasets, credentials, Git state and private project documents are excluded.

```powershell
docker build --platform linux/amd64 --provenance=false -f infra/training/Dockerfile -t waymark-training:REVIEWED_VERSION .
waymark prepare-training-source --output data/handoffs/NEW_SOURCE_VERSION.tar.gz
```

`--provenance=false` exports one platform manifest instead of an attestation manifest index. Record the resulting digest and installed dependency inventory. Later rebuilds can differ because Debian updates and transitive packages can change. Execution must use the reviewed ECR digest, never a mutable local tag.

Training source remains separate from the image. The managed launcher uploads its deterministic source archive under its SHA256 and supplies `sagemaker_program` and `sagemaker_submit_directory`. The Toolkit downloads/unpacks it, exports the managed environment variables and forwards user hyperparameters. The entrypoint verifies source inventory, Python/dependencies and feature identity before fitting. It accepts both `--preflight-only` and the Toolkit's `--preflight-only True` argument format.

## No-fit verification

The prepared source can run `train_entry.py --preflight-only --image-uri ECR_URI_AT_SHA256_DIGEST --preflight-output NEW_RECEIPT.json`. This checks actual installed versions/imports and the source manifest, then returns before reading training data or fitting.

For a full local bootstrap check, mount only the verified source at `/source`, SageMaker-format `hyperparameters.json`, `resourceconfig.json` and `inputdataconfig.json` at `/opt/ml/input/config`, and a new receipt directory at `/checks`. Set `sagemaker_program=train_entry.py`, `sagemaker_submit_directory=/source`, `preflight-only=true`, the reviewed `image-uri`, and `preflight-output=/checks/environment.json`. Run the normal image entrypoint with `--network none --hostname algo-1 --add-host algo-1:127.0.0.1`. Do not mount data or credentials. The loopback host entry lets the Toolkit resolve its single host without networking.

This establishes the actual container's source bootstrap, argument forwarding, managed-environment emulation and import-only preflight. It does not establish SageMaker service-side image pull, execution-role access, quota/capacity, Spot recovery or model quality. Every fit remains remote and explicitly user-launched.

## AWS ownership and acceptance

The dedicated `waymark-training` ECR repository in account `149270211548`, Region `ap-southeast-2`, is currently managed through AWS CLI, outside Terraform state. It has immutable tags, AES256 encryption and basic scan-on-push. No IAM changes were made to prepare it. Use a new tag for every candidate and retain digest-bound build, bootstrap and scan evidence. Do not silently replace an existing candidate or treat an environment import receipt as security acceptance.

The first Bookworm candidate exposed OS-package findings without installable Bookworm upgrades. Official Debian security data established fixes in Trixie for several of those findings, motivating the second candidate while preserving Python/model pins. Retain the actual scan and any remaining findings; a successful build or scan completion does not mean zero vulnerabilities. ECR basic OS-package scanning is not a complete audit of Python wheels or application attack paths.

Before training, require C1 v2 acceptance, the real image environment receipt, reviewed remaining scan findings, a positive instance quota/capacity check and the remaining project budget. The observed `ml.m5.large` Spot and on-demand quotas were both zero on 2026-11-11. No quota increase or model job was submitted by image preparation.

## Verified candidate (2026-11-11)

The second candidate is `149270211548.dkr.ecr.ap-southeast-2.amazonaws.com/waymark-training@sha256:b82821aec5b4922f417cad420d79ed63e23254f7c9cb1de1af64be1c7bb1b46a`, with immutable tag `20261111-py312-v2`. ECR digest/media type matched the locally tested Linux AMD64 image. Python 3.12.16, all required imports, the actual Toolkit bootstrap and estimator/calibrator construction passed without fitting or mounted data/credentials. The full bootstrap ran with networking disabled.

Local handoff evidence is `data/handoffs/training_image_preparation_v2_20261111.json`, the actual import receipt is `data/handoffs/verified_training_environment_v2_20261111.json`, and the separate deterministic source is `data/handoffs/repaired_training_source_v4.tar.gz`. The source SHA256 is `b915aa15eb5ee66b5f72ec7218bf01fbbc4f2f88b6fd92ff68d0268d83620545`. Preserve older handoffs and revalidate identities before execution.

The final ECR basic scan reported 2 HIGH, 9 MEDIUM and 3 LOW findings. The remaining HIGH findings are [CVE-2026-5435](https://security-tracker.debian.org/tracker/CVE-2026-5435) and [CVE-2026-19499](https://security-tracker.debian.org/tracker/CVE-2026-19499) in glibc. Debian identifies both current Trixie packages as vulnerable with `no-dsa` / Minor issue notes; the listed fixed versions are in unstable. These findings remain open. No direct Waymark call to the affected DNS diagnostic or monetary-format APIs was found, but complete transitive-native reachability is unverified. Retain the stable, tested image and these known findings rather than silently replacing libc with an unstable build to clear a scanner count. This is preparation evidence, not a claim of zero vulnerabilities or authorization to fit. Training quota remains the explicit runtime blocker.

User disposition (2026-11-11): defer further remediation of the two named glibc findings for now; keep their vulnerability status and evidence unchanged. The user submitted the Sydney Spot quota increase to 4; read-only CLI confirmed PENDING with applied quota still 0. This supersedes earlier request-1 guidance, because the AWS default/Console minimum is 4. Continue using one instance per future job. No duplicate request, automatic training or image modification follows from this decision.

References: [AWS Training Toolkit](https://github.com/aws/sagemaker-training-toolkit), [SageMaker training image contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html), [Debian security tracker](https://security-tracker.debian.org/tracker/).
