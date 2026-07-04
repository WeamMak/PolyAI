# Vision Agent

A LangChain-powered AI vision agent with a manual ReAct loop. Accepts text and base64-encoded images, and can call tools (e.g. YOLO object detection) to answer questions.

## Prerequisites

- Python 3.10+
- A running YOLO service (optional - only needed for `detect_objects`)


## Setup

Install dependencies (from `services/agent/`):

```bash
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Edit .env to set your S3 bucket, Bedrock model, or YOLO URL
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

## Image Flow

The Agent and YOLO service exchange JSON only.

1. The frontend sends the user's image to the Agent as base64.
1. The Agent uploads the original image to S3.
1. The Agent sends YOLO only the original `image_s3_key`.
1. YOLO downloads the original image from S3, runs prediction, uploads the predicted image to S3, and returns `predicted_image_s3_key`.
1. The Agent reads the predicted image from S3 and returns it to the frontend as `annotated_image` base64.

The Agent does not send image bytes to YOLO, and it does not fetch predicted image bytes from YOLO.

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
  "prediction_id": "string or null",
  "annotated_image": "base64-encoded annotated JPEG or null",
  "tokens_used": {
    "input": 312,
    "output": 22,
    "total": 334
  },
  "agent_loop_time_s": 1.23,
  "iterations": 2,
  "tools_called": ["detect_objects"],
  "context_limit_exceeded": false
}
```

### `GET /health`

Returns `{"status": "ok"}` when the service is running.
