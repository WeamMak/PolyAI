# Simple Observability MCP

This local MCP server gives VS Code Copilot two generic data-access tools:

- `get_logs` reads raw gzip-compressed container records from S3.
- `query_prometheus` runs a bounded PromQL range query.

Copilot filters records by `attrs.com.docker.compose.service`, searches error
messages, identifies containers, writes PromQL, and correlates the returned
logs and metrics.

## Setup

```bash
source .venv/bin/activate
pip install -r services/observability-mcp/requirements.txt
```

The tracked `.vscode/mcp.json` starts the server with the project virtual
environment and configures:

| Environment | Prometheus | S3 bucket |
| --- | --- | --- |
| dev | `http://dev.weam.fursa.click:9090` | `weam-polyai-logs-dev` |
| prod | `http://prod.weam.fursa.click:9090` | `weam-polyai-logs-prod` |

Use **MCP: List Servers** in VS Code to start or restart `observability`.

The local AWS identity needs `s3:ListBucket` and `s3:GetObject` for both log
buckets. Credentials come from the normal boto3 credential chain; do not put
AWS keys in `.vscode/mcp.json`.

## How Copilot uses the tools

For recent YOLO logs, Copilot calls `get_logs` and retains records whose
`attrs.com.docker.compose.service` is `yolo`. To list containers, it collects
the distinct service labels and can call the tool once for dev and once for
prod.

For CPU usage, Copilot builds a PromQL expression using
`node_cpu_seconds_total` and passes it to `query_prometheus`. For an incident,
it calls both tools with the same `around_timestamp`.

Both tools limit time ranges and output size so raw observability data does not
overwhelm Copilot's context.

## Tests

```bash
pytest services/observability-mcp/tests
```

Tests use fake AWS and Prometheus responses and make no network calls.
