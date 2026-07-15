# Vision Agent

A LangChain-powered AI vision agent with a manual ReAct loop. Accepts text and
base64-encoded images, calls YOLO for detection, and discovers image tools from
the image-processing service through standard MCP.

## Source Layout

```text
app.py          FastAPI routes and request orchestration
agent_loop.py   Model setup and the explicit manual tool-calling loop
config.py       Environment-derived settings
context.py      Request-local image and detection state
mcp_tools.py    MCP discovery, private argument injection, and proxies
storage.py      S3 and opaque image transport
yolo_client.py  YOLO HTTP client and detect_objects tool
```

## Prerequisites

- Python 3.10+
- A running YOLO service (needed for object detection)
- A running image-processing MCP service (needed for image transformations)


## Setup

Install dependencies (from `services/agent/`):

```bash
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Edit .env to set your S3 bucket, Bedrock model, YOLO URL, or MCP URL
```

The agent uses Amazon Bedrock through the AWS SDK. Configure AWS credentials
with `aws configure`; do not copy AWS keys into `.env` or the source code.

`.env` variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `bedrock/openai.gpt-oss-20b-1:0` | Bedrock model used by the agent |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock and S3 |
| `AWS_S3_BUCKET` | required | S3 bucket used to store uploaded images |
| `YOLO_SERVICE_URL` | `http://localhost:8080` | URL of the YOLO microservice for standalone local runs. Docker Compose overrides this to `http://yolo:8080`. |
| `IMG_PROC_MCP_URL` | `http://localhost:8090` | URL of the image-processing MCP service for standalone local runs. Docker Compose overrides this to `http://img-proc-mcp:8090`. |

## Image Flow

The Agent, YOLO service, and image-processing MCP service exchange JSON only.

1. The frontend sends the user's image to the Agent as base64.
1. The Agent uploads the original image to S3.
1. The Agent sends YOLO only the original `image_s3_key`.
1. YOLO downloads the original image from S3, runs prediction, uploads the predicted image to S3, and returns `predicted_image_s3_key`.
1. The Agent reads the predicted image from S3 and returns it to the frontend as `annotated_image` base64.

The Agent does not send image bytes to YOLO, and it does not fetch predicted image bytes from YOLO.

For image-processing prompts:

1. The Agent discovers the six image tools from the MCP `/mcp` endpoint.
1. The LLM chooses a focused MCP tool and describes the requested object using
   `label`, `ordinal`, and `from_side`.
1. The Agent privately injects the current image base64 and the complete YOLO
   detection JSON into the MCP call.
1. The MCP service selects the detection, performs every pixel operation, and
   returns a processed PNG.
1. The Agent treats the returned bytes as opaque data, uploads them to S3, and
   returns the image to the frontend.

The Agent does not import Pillow or interpret bounding boxes. The LLM still
receives text-only tool results; image bytes are never added to LangChain
messages.

Allowed Bedrock models:

```text
bedrock/anthropic.claude-3-haiku-20240307-v1:0
bedrock/amazon.nova-micro-v1:0
bedrock/amazon.nova-lite-v1:0
bedrock/openai.gpt-oss-20b-1:0
bedrock/meta.llama3-1-8b-instruct-v1:0
bedrock/mistral.mistral-7b-instruct-v0:2
```

## Running

```bash
cd services/agent
python app.py
```

The server starts at `http://localhost:8000`.

## Testing with curl

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### Plain text message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello! What can you do?"}]}'
```

### Send a message with an image

```bash
echo "{\"messages\": [{\"role\": \"user\", \"content\": \"What objects are in this image?\", \"image_base64\": \"$(base64 -w0 beatles.jpeg)\"}]}" \
  | curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d @-
```

## API Reference

### `POST /chat`

Request body:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What objects are in this image?",
      "image_base64": "optional base64-encoded JPEG or PNG"
    }
  ]
}
```

Response:

```json
{
  "response": "string",
  "chat_id": "stable chat identifier",
  "active_image_s3_key": "current original or processed image key",
  "prediction_id": "string or null",
  "annotated_image": "base64-encoded annotated or processed image, or null",
  "annotated_image_media_type": "image/jpeg, image/png, or null",
  "tokens_used": {
    "input": 312,
    "output": 22,
    "total": 334
  },
  "agent_loop_time_s": 1.23,
  "iterations": 2,
  "tools_called": ["detect_objects", "blur"],
  "context_limit_exceeded": false
}
```

### `GET /health`

Returns `{"status": "ok"}` when the service is running.
