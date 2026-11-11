"""CLI transport preserves binary bytes, pagination and immutable-write errors."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from botocore.exceptions import ClientError

from waymark.aws_cli import AwsCliSession


def test_cli_binary_roundtrip_and_profile_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"\x00\xff\nparquet-bytes"

    def run(command, **kwargs):
        assert command[command.index("--profile") + 1] == "waymark"
        assert command[command.index("--region") + 1] == "ap-southeast-2"
        if "put-object" in command:
            request = json.loads(Path(command[command.index("--cli-input-json") + 1][7:]).read_text())
            assert request["Key"] == "snapshot/test.parquet"
            assert request["IfNoneMatch"] == "*"
            assert Path(command[command.index("--body") + 1]).read_bytes() == payload
            return CompletedProcess(command, 0, '{"VersionId":"v1"}', "")
        assert command[command.index("--bucket") + 1] == "bucket"
        assert command[command.index("--key") + 1] == "snapshot/test.parquet"
        assert "--cli-input-json" not in command
        Path(command[-1]).write_bytes(payload)
        return CompletedProcess(command, 0, '{"VersionId":"v1"}', "")

    monkeypatch.setattr("waymark.aws_cli.subprocess.run", run)
    session = AwsCliSession("waymark", "ap-southeast-2")
    client = session.client("s3")
    client.put_object(Bucket="bucket", Key="snapshot/test.parquet", Body=payload, IfNoneMatch="*")
    result = client.get_object(Bucket="bucket", Key="snapshot/test.parquet")
    assert result["Body"].read() == payload
    assert result["VersionId"] == "v1"


def test_cli_preserves_precondition_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "waymark.aws_cli.subprocess.run",
        lambda command, **kwargs: CompletedProcess(
            command,
            254,
            "",
            "An error occurred (PreconditionFailed) when calling the PutObject operation: exists",
        ),
    )
    with pytest.raises(ClientError) as error:
        AwsCliSession("waymark", "ap-southeast-2").client("s3").put_object(
            Bucket="bucket", Key="existing", Body=b"data", IfNoneMatch="*"
        )
    assert error.value.response["Error"]["Code"] == "PreconditionFailed"


def test_cli_uses_complete_auto_paginated_results(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"Data": [{"VarCharValue": str(n)}]} for n in range(1005)]

    def run(command, **kwargs):
        assert "--no-paginate" not in command
        return CompletedProcess(command, 0, json.dumps({"ResultSet": {"Rows": rows}}), "")

    monkeypatch.setattr("waymark.aws_cli.subprocess.run", run)
    paginator = AwsCliSession("waymark", "ap-southeast-2").client("athena").get_paginator("get_query_results")
    pages = list(paginator.paginate(QueryExecutionId="query-id"))
    assert len(pages[0]["ResultSet"]["Rows"]) == 1005
