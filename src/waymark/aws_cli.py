"""AWS CLI transport for the existing C1 verification interfaces."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AwsCliSession:
    """Bind all C1 CLI calls to one explicit profile and Region."""

    def __init__(self, profile_name: str, region_name: str) -> None:
        self.profile_name = profile_name
        self.region_name = region_name

    def client(self, service: str) -> AwsCliClient:
        if service not in {"sts", "s3", "glue", "athena"}:
            raise ValueError(f"Unsupported C1 service: {service}")
        return AwsCliClient(self, service)


class AwsCliClient:
    """Translate C1 JSON calls while retaining S3 byte/error semantics."""

    def __init__(self, session: AwsCliSession, service: str) -> None:
        self.session = session
        self.service = "s3api" if service == "s3" else service

    def __getattr__(self, operation: str) -> Any:
        if operation.startswith("_"):
            raise AttributeError(operation)

        def invoke(**parameters: Any) -> dict[str, Any]:
            return self.call(operation, parameters)

        return invoke

    def call(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
        logger.info("AWS CLI %s %s", self.service, operation)
        with tempfile.TemporaryDirectory(prefix="waymark-cli-") as directory:
            root = Path(directory)
            request = dict(parameters)
            command = [
                "aws",
                self.service,
                operation.replace("_", "-"),
                "--profile",
                self.session.profile_name,
                "--region",
                self.session.region_name,
                "--output",
                "json",
                "--no-cli-pager",
            ]
            if self.service == "s3api" and operation == "put_object":
                body = root / "body.bin"
                body.write_bytes(request.pop("Body"))
                command.extend(["--body", str(body)])
            download = root / "download.bin"
            if self.service == "s3api" and operation == "get_object":
                # The CLI's customized streaming-output command has no --cli-input-json.
                options = {"Bucket": "--bucket", "Key": "--key", "VersionId": "--version-id"}
                for name, value in request.items():
                    command.extend([options[name], value])
                command.append(str(download))
            else:
                config = root / "request.json"
                config.write_text(json.dumps(request), encoding="utf-8")
                command.extend(["--cli-input-json", f"file://{config}"])
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=900,
                env={**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off"},
            )
            if result.returncode:
                error = re.search(
                    r"An error occurred \(([^)]+)\) when calling the (\w+) operation", result.stderr
                )
                if error:
                    raise ClientError(
                        {"Error": {"Code": error[1], "Message": result.stderr.strip()}}, error[2]
                    )
                raise RuntimeError(f"AWS CLI {self.service}/{operation}: {result.stderr.strip()}")
            response = json.loads(result.stdout) if result.stdout.strip() else {}
            if self.service == "s3api" and operation == "get_object":
                response["Body"] = io.BytesIO(download.read_bytes())
            return response

    def get_paginator(self, operation: str) -> AwsCliResults:
        if self.service != "athena" or operation != "get_query_results":
            raise ValueError("Only complete Athena query results are supported")
        return AwsCliResults(self)


class AwsCliResults:
    """AWS CLI automatically merges every Athena result page before returning."""

    def __init__(self, client: AwsCliClient) -> None:
        self.client = client

    def paginate(self, **parameters: Any) -> Any:
        yield self.client.call("get_query_results", parameters)
