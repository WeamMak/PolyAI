import gzip
import io
import json
from datetime import datetime, timezone

import anyio
import pytest
import requests

import app


UTC = timezone.utc


@pytest.fixture(autouse=True)
def observability_environment(monkeypatch):
    monkeypatch.setenv("DEV_PROMETHEUS_URL", "http://dev.example:9090")
    monkeypatch.setenv("PROD_PROMETHEUS_URL", "http://prod.example:9090")
    monkeypatch.setenv("DEV_S3_LOGS_BUCKET", "logs-dev")
    monkeypatch.setenv("PROD_S3_LOGS_BUCKET", "logs-prod")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **arguments):
        self.calls.append(arguments)
        return self.pages


class FakeS3Client:
    def __init__(self, pages, bodies):
        self.paginator = FakePaginator(pages)
        self.bodies = bodies

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator

    def get_object(self, Bucket, Key):
        return {
            "Body": io.BytesIO(self.bodies[Key]),
            "ContentEncoding": "gzip",
        }


class FakeResponse:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


def _gzip_lines(*records):
    lines = [
        record if isinstance(record, str) else json.dumps(record)
        for record in records
    ]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def test_time_window_around_timestamp_is_utc():
    start, end = app._time_window(5, "2026-07-01 12:00:00")

    assert app._utc_iso(start) == "2026-07-01T11:55:00Z"
    assert app._utc_iso(end) == "2026-07-01T12:05:00Z"


def test_get_logs_returns_raw_records_for_copilot(monkeypatch):
    modified = datetime(2026, 7, 1, 12, 1, tzinfo=UTC)
    key = "logs/2026/07/01/120100_test.gz"
    client = FakeS3Client(
        [
            {
                "Contents": [
                    {"Key": key, "LastModified": modified},
                ]
            }
        ],
        {
            key: _gzip_lines(
                {
                    "time": "2026-07-01T12:01:00Z",
                    "stream": "stderr",
                    "log": "internal server error",
                    "attrs": {"com.docker.compose.service": "yolo"},
                },
                "not-json",
            )
        },
    )
    monkeypatch.setattr(app.boto3, "client", lambda *args, **kwargs: client)

    result = app.get_logs(
        environment="dev",
        minutes=5,
        around_timestamp="2026-07-01 12:00:00",
    )

    assert result["ok"] is True
    assert result["malformed_lines"] == 1
    assert result["records"][0]["attrs"] == {
        "com.docker.compose.service": "yolo"
    }
    assert result["records"][0]["_timestamp"] == "2026-07-01T12:01:00Z"
    assert result["records"][0]["_s3_key"] == key


def test_get_logs_uses_all_paginator_pages(monkeypatch):
    modified = datetime(2026, 7, 1, 12, 1, tzinfo=UTC)
    keys = [
        "logs/2026/07/01/120100_first.gz",
        "logs/2026/07/01/120101_second.gz",
    ]
    client = FakeS3Client(
        [
            {"Contents": [{"Key": keys[0], "LastModified": modified}]},
            {"Contents": [{"Key": keys[1], "LastModified": modified}]},
        ],
        {
            keys[0]: _gzip_lines(
                {"time": "2026-07-01T12:01:00Z", "log": "first"}
            ),
            keys[1]: _gzip_lines(
                {"time": "2026-07-01T12:02:00Z", "log": "second"}
            ),
        },
    )
    monkeypatch.setattr(app.boto3, "client", lambda *args, **kwargs: client)

    result = app.get_logs(
        minutes=5,
        around_timestamp="2026-07-01 12:00:00",
    )

    assert [record["log"] for record in result["records"]] == [
        "first",
        "second",
    ]


def test_query_prometheus_returns_raw_data(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse(
            {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"metric": {}, "values": [[1, "12.5"]]}],
                },
            }
        )

    monkeypatch.setattr(app.requests, "get", fake_get)

    result = app.query_prometheus(
        query='rate(node_cpu_seconds_total{mode="idle"}[2m])',
        environment="prod",
        minutes=10,
        around_timestamp="2026-07-01 12:00:00",
    )

    assert result["ok"] is True
    assert result["data"]["resultType"] == "matrix"
    assert "hint" not in result
    assert result["query_adjusted"] is False
    assert result["requested_query"] == result["executed_query"]
    assert len(calls) == 1
    assert calls[0][0] == "http://prod.example:9090/api/v1/query_range"
    assert calls[0][1]["query"].startswith("rate(")
    assert calls[0][2] == 10


def test_query_prometheus_explains_empty_results(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["query"])
        return FakeResponse(
            {
                "status": "success",
                "data": {"resultType": "matrix", "result": []},
            }
        )

    monkeypatch.setattr(app.requests, "get", fake_get)

    result = app.query_prometheus(
        query=(
            'node_cpu_seconds_total{mode="idle",environment="dev"}'
        ),
        environment="dev",
    )

    assert result["ok"] is True
    assert result["query_adjusted"] is True
    assert len(calls) == 2
    assert 'environment="dev"' in result["requested_query"]
    assert "environment" not in result["executed_query"]
    assert "do not add an environment label" in result["hint"]
    assert 'job="node"' in result["hint"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            'metric{environment="dev",mode="idle"}',
            'metric{mode="idle"}',
        ),
        (
            'metric{mode="idle",environment="prod",cpu="0"}',
            'metric{mode="idle",cpu="0"}',
        ),
        (
            'metric{mode="idle",environment="dev"}',
            'metric{mode="idle"}',
        ),
        ('metric{environment="prod"}', "metric{}"),
    ],
)
def test_without_environment_matcher_keeps_valid_promql(query, expected):
    assert app._without_environment_matcher(query) == expected


def test_query_prometheus_retries_with_corrected_query(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["query"])
        if len(calls) == 1:
            result = []
        else:
            result = [{"metric": {"instance": "node"}, "values": []}]
        return FakeResponse(
            {
                "status": "success",
                "data": {"resultType": "matrix", "result": result},
            }
        )

    monkeypatch.setattr(app.requests, "get", fake_get)

    result = app.query_prometheus(
        query=(
            "100 - (avg(rate(node_cpu_seconds_total{"
            'mode="idle",environment="prod"}[5m])) * 100)'
        ),
        environment="prod",
    )

    assert len(calls) == 2
    assert 'environment="prod"' in calls[0]
    assert "environment" not in calls[1]
    assert result["query_adjusted"] is True
    assert result["query"] == calls[1]
    assert "hint" not in result


def test_query_prometheus_returns_transport_error(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(app.requests, "get", fail)

    result = app.query_prometheus(query="up")

    assert result == {"ok": False, "error": "offline"}


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (app.get_logs, {"environment": "staging"}, "environment"),
        (app.get_logs, {"minutes": 0}, "minutes"),
        (app.get_logs, {"limit": 201}, "limit"),
        (app.query_prometheus, {"query": ""}, "query"),
    ],
)
def test_tools_validate_inputs(function, arguments, message):
    result = function(**arguments)

    assert result["ok"] is False
    assert message in result["error"]


def test_mcp_exposes_only_two_generic_tools():
    async def tool_names():
        tools = await app.mcp.list_tools()
        return [tool.name for tool in tools]

    assert anyio.run(tool_names) == ["get_logs", "query_prometheus"]


def test_query_prometheus_tool_description_guides_copilot():
    async def tool_description():
        tools = await app.mcp.list_tools()
        query_tool = next(
            tool for tool in tools if tool.name == "query_prometheus"
        )
        return query_tool.description

    description = anyio.run(tool_description)

    assert "environment argument selects the Prometheus server" in description
    assert "Do not add an" in description
    assert 'environment label such as environment="dev"' in description
    assert "node_cpu_seconds_total" in description
