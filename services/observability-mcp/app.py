import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from mcp.server.fastmcp import FastMCP


UTC = timezone.utc
MAX_MINUTES = 1440
MAX_LOG_RESULTS = 200
MAX_S3_OBJECTS = 500
S3_UPLOAD_BUFFER = timedelta(minutes=5)
HTTP_TIMEOUT_SECONDS = 10

ENVIRONMENTS = {
    "dev": {
        "prometheus_url": "DEV_PROMETHEUS_URL",
        "logs_bucket": "DEV_S3_LOGS_BUCKET",
    },
    "prod": {
        "prometheus_url": "PROD_PROMETHEUS_URL",
        "logs_bucket": "PROD_S3_LOGS_BUCKET",
    },
}

mcp = FastMCP("observability")


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _required_environment_value(name: str, key: str) -> str:
    if not isinstance(name, str) or name.strip().lower() not in ENVIRONMENTS:
        raise ValueError("environment must be 'dev' or 'prod'")

    environment = name.strip().lower()
    variable_name = ENVIRONMENTS[environment][key]
    value = os.environ.get(variable_name, "").strip()
    if not value:
        raise ValueError(f"{variable_name} is required")
    return value


def _prometheus_url(environment: str) -> str:
    value = _required_environment_value(
        environment,
        "prometheus_url",
    ).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ["http", "https"] or not parsed.netloc:
        raise ValueError("Prometheus URL must use HTTP or HTTPS")
    return value


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "timestamp must use ISO-8601, for example 2026-07-01 12:00:00"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _time_window(
    minutes: int,
    around_timestamp: str = "",
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError("minutes must be an integer")
    if minutes < 1 or minutes > MAX_MINUTES:
        raise ValueError(f"minutes must be between 1 and {MAX_MINUTES}")

    delta = timedelta(minutes=minutes)
    if around_timestamp:
        center = _parse_timestamp(around_timestamp)
        return center - delta, center + delta

    end = now or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    end = end.astimezone(UTC)
    return end - delta, end


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _date_prefixes(start: datetime, end: datetime) -> list[str]:
    day = start.date()
    final_day = end.date()
    prefixes = []

    while day <= final_day:
        prefixes.append(day.strftime("logs/%Y/%m/%d/"))
        day += timedelta(days=1)
    return prefixes


def _object_time(value) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("S3 object is missing LastModified")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _list_log_objects(client, bucket: str, start: datetime, end: datetime):
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    earliest = start - S3_UPLOAD_BUFFER
    latest = end + S3_UPLOAD_BUFFER

    for prefix in _date_prefixes(start, end + S3_UPLOAD_BUFFER):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                modified = _object_time(item.get("LastModified"))
                if earliest <= modified <= latest:
                    objects.append(
                        {"Key": item["Key"], "LastModified": modified}
                    )
                if len(objects) >= MAX_S3_OBJECTS:
                    return sorted(
                        objects,
                        key=lambda current: current["LastModified"],
                    )

    return sorted(objects, key=lambda current: current["LastModified"])


def _decode_s3_object(response: dict) -> str:
    body = response["Body"].read()
    encoding = str(response.get("ContentEncoding", "")).lower()
    if encoding == "gzip" or body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    return body.decode("utf-8", errors="replace")


def _log_time(record: dict, fallback: datetime) -> datetime:
    for key in ["time", "timestamp", "@timestamp", "date"]:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str) and value.strip():
            try:
                return _parse_timestamp(value)
            except ValueError:
                continue
    return fallback


@mcp.tool()
def get_logs(
    environment: str = "dev",
    minutes: int = 5,
    around_timestamp: str = "",
    limit: int = 200,
) -> dict:
    """Return raw dev or prod container log records from S3.

    Copilot should filter the returned records by Compose service, message,
    stream, or error text. With an around_timestamp, minutes are read before
    and after that UTC time.
    """
    try:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if limit < 1 or limit > MAX_LOG_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_LOG_RESULTS}")

        normalized_environment = environment.strip().lower()
        bucket = _required_environment_value(
            normalized_environment,
            "logs_bucket",
        )
        region = os.environ.get("AWS_REGION", "").strip()
        if not region:
            raise ValueError("AWS_REGION is required")
        start, end = _time_window(minutes, around_timestamp)

        client = boto3.client("s3", region_name=region)
        objects = _list_log_objects(client, bucket, start, end)
        records = []
        malformed_lines = 0

        for item in objects:
            response = client.get_object(Bucket=bucket, Key=item["Key"])
            for line in _decode_s3_object(response).splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    malformed_lines += 1
                    continue

                timestamp = _log_time(record, item["LastModified"])
                if timestamp < start or timestamp > end:
                    continue

                raw_record = dict(record)
                raw_record["_timestamp"] = _utc_iso(timestamp)
                raw_record["_s3_key"] = item["Key"]
                records.append(raw_record)

        records.sort(key=lambda record: record["_timestamp"])
        truncated = len(records) > limit
        records = records[-limit:]
        return {
            "ok": True,
            "environment": normalized_environment,
            "start": _utc_iso(start),
            "end": _utc_iso(end),
            "records": records,
            "scanned_objects": len(objects),
            "malformed_lines": malformed_lines,
            "truncated": truncated,
        }
    except (ValueError, BotoCoreError, ClientError, OSError, KeyError) as exc:
        return _error(str(exc))


def _prometheus_step(start: datetime, end: datetime) -> int:
    duration = (end - start).total_seconds()
    if duration <= 3600:
        return 15
    if duration <= 21600:
        return 60
    return 300


@mcp.tool()
def query_prometheus(
    query: str,
    environment: str = "dev",
    minutes: int = 10,
    around_timestamp: str = "",
) -> dict:
    """Run a read-only PromQL range query against dev or prod Prometheus.

    The environment argument selects the Prometheus server. Do not add an
    environment label such as environment="dev" or environment="prod" to
    PromQL because the collected metrics do not have that label.

    Node-exporter metrics use job="node". For instance CPU usage, use:
    100 - (avg by (instance) (rate(
      node_cpu_seconds_total{job="node",mode="idle"}[5m]
    )) * 100)

    Copilot should create other PromQL expressions for request, latency, 5xx,
    memory, or other analysis and interpret the returned Prometheus data.
    """
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty PromQL expression")

        normalized_environment = environment.strip().lower()
        url = _prometheus_url(normalized_environment)
        start, end = _time_window(minutes, around_timestamp)
        response = requests.get(
            f"{url}/api/v1/query_range",
            params={
                "query": query.strip(),
                "start": _utc_iso(start),
                "end": _utc_iso(end),
                "step": _prometheus_step(start, end),
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Prometheus returned an invalid response")
        if payload.get("status") != "success":
            raise ValueError(
                payload.get("error", "Prometheus returned an error")
            )

        result = {
            "ok": True,
            "environment": normalized_environment,
            "start": _utc_iso(start),
            "end": _utc_iso(end),
            "query": query.strip(),
            "data": payload.get("data", {}),
            "warnings": payload.get("warnings", []),
        }
        prometheus_data = result["data"]
        if (
            isinstance(prometheus_data, dict)
            and prometheus_data.get("result") == []
        ):
            result["hint"] = (
                "No matching series. The environment argument already "
                "selects the Prometheus server, so do not add an "
                "environment label to PromQL. Retry using labels that exist; "
                'node-exporter metrics use job="node".'
            )
        return result
    except (ValueError, AttributeError, requests.RequestException) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    mcp.run(transport="stdio")
