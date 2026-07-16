# PolyAI

PolyAI is an educational service stack for image-aware chat.

```text
services/
  frontend/  Next.js chat UI
  agent/     FastAPI + LangChain agent with a manual tool-calling loop
  yolo/      FastAPI YOLO object-detection service
  img-proc-mcp/  MCP image-processing tools
  observability-mcp/  Local MCP tools for S3 logs and Prometheus metrics
```

The full Docker Compose stack runs:

```text
frontend -> agent -> yolo
agent -> img-proc-mcp
prometheus -> yolo metrics
grafana -> prometheus
fluent-bit -> S3 container logs (EC2 only)
```

## Environment Files

There are three `.env` locations. They have different jobs.

Root Compose env:

```text
.env
```

Used by Docker Compose for image tags and the browser-facing agent URL:

```env
DOCKERHUB_NAMESPACE=weammakhoul
YOLO_IMAGE_TAG=0.0.1
AGENT_IMAGE_TAG=0.0.1
FRONTEND_IMAGE_TAG=0.0.1
IMG_PROC_MCP_IMAGE_TAG=0.0.1
NEXT_PUBLIC_AGENT_URL=http://localhost:8000
```

Agent env:

```text
services/agent/.env
```

```env
MODEL=bedrock/openai.gpt-oss-20b-1:0
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-polyai-images-bucket
```

YOLO env:

```text
services/yolo/.env
```

```env
CONFIDENCE_THRESHOLD=0.5
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-polyai-images-bucket
```

Do not put AWS access keys in these files on EC2. EC2 should use an IAM role.

## Local Docker Run

Create the service env files:

```bash
cp services/agent/.env.example services/agent/.env
cp services/yolo/.env.example services/yolo/.env
```

Edit both files and set `AWS_S3_BUCKET`.

Local Docker uses `docker-compose.override.yml` automatically. That ignored file builds local images and mounts `~/.aws` read-only so boto3 can use your local AWS profile.

Start the stack:

```bash
docker compose up -d --build
```

Open:

```text
Frontend:   http://localhost:3000
Agent:      http://localhost:8000
YOLO:       http://localhost:8080
Img Proc:   http://localhost:8090
Prometheus: http://localhost:9090
Grafana:    http://localhost:3001
```

In Grafana, the Prometheus data source URL is:

```text
http://prometheus:9090
```

Stop the stack:

```bash
docker compose down
```

Do not use `docker compose down -v` unless you want to delete Grafana and Prometheus data volumes.

## EC2 Deployment

EC2 runs the stack from Docker images. It should not build images.

The deployment flow is:

```text
push to dev/main
GitHub Actions builds changed service images
GitHub Actions pushes unique image tags to Docker Hub
GitHub Actions SSHs into EC2
EC2 runs docker compose pull
EC2 runs docker compose up -d --no-build
```

The EC2 stack directory is:

```text
/home/ubuntu/PolyAI
```

EC2 owns these runtime files:

```text
/home/ubuntu/PolyAI/.env
/home/ubuntu/PolyAI/services/agent/.env
/home/ubuntu/PolyAI/services/yolo/.env
```

For dev EC2, root `.env` should include:

```env
DOCKERHUB_NAMESPACE=weammakhoul
YOLO_IMAGE_TAG=0.0.1
AGENT_IMAGE_TAG=0.0.1
FRONTEND_IMAGE_TAG=0.0.1
IMG_PROC_MCP_IMAGE_TAG=0.0.1
NEXT_PUBLIC_AGENT_URL=http://dev.weam.fursa.click:8000
```

For prod EC2, use:

```env
NEXT_PUBLIC_AGENT_URL=http://prod.weam.fursa.click:8000
```

GitHub Actions updates the image tag values during deploy.

GitHub Actions needs these repository secrets:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
DEV_INSTANCE_SSH_KEY
PROD_INSTANCE_SSH_KEY
```

These repository variables are optional because the workflow has fallbacks:

```text
DOCKERHUB_NAMESPACE
DEV_NEXT_PUBLIC_AGENT_URL
PROD_NEXT_PUBLIC_AGENT_URL
STACK_DIR
```

Make sure Docker starts after reboot:

```bash
sudo systemctl enable docker
```

The Compose services use `restart: unless-stopped`, so containers should come back after the EC2 instance restarts.

### EC2 Container Logs

The EC2 Compose stack runs Fluent Bit to collect Docker JSON logs and upload
gzip-compressed batches to `s3://weam-polyai-logs-dev/logs/` for dev and
`s3://weam-polyai-logs-prod/logs/` for prod in `us-east-1`.
This collector is for the old EC2 deployment only; it is not part of the
Kubernetes deployment.

The S3 bucket should remain private and have an enabled lifecycle rule named
`delete-logs-after-90-days` that expires all objects 90 days after creation.
Each EC2 instance role needs `s3:PutObject` only for its own log prefix:

```text
Dev:  arn:aws:s3:::weam-polyai-logs-dev/logs/*
Prod: arn:aws:s3:::weam-polyai-logs-prod/logs/*
```

After deployment, check the collector and its uploads with:

```bash
docker compose ps fluent-bit
docker compose logs fluent-bit
```

## Networking Notes

Inside Docker, services use Docker DNS names:

```text
agent -> http://yolo:8080
prometheus -> yolo:8080
grafana -> http://prometheus:9090
```

Browsers use public or localhost URLs:

```text
local: http://localhost:8000
dev:   http://dev.weam.fursa.click:8000
prod:  http://prod.weam.fursa.click:8000
```

`NEXT_PUBLIC_AGENT_URL` is baked into the frontend image at build time, so GitHub Actions passes the correct dev/prod value when building the frontend.
