# PolyAI Fursa - Agent Guidelines

This is an **educational project**. Students learn by reading, modifying, and extending real code.

## Project Overview

```
services/
  agent/    <- LangChain agent with manual tool-calling loop
  frontend/ <- Simple chat UI (talks to the agent)
  yolo/     <- YOLO object-detection microservice (FastAPI + Ultralytics)
  img-proc-mcp/ <- MCP image-processing tools (FastMCP + Pillow)
```

---

## Containerization and Deployment

The project is deployed with Docker Compose, not Linux services.

- `docker-compose.yml` is the committed stack definition for frontend, agent, yolo, img-proc-mcp, Prometheus, and Grafana.
- `docker-compose.override.yml` is local-only and ignored by git. It may build local images and mount `~/.aws` read-only for local AWS credentials.
- EC2 should not build images. GitHub Actions builds changed service images, pushes them to Docker Hub with unique tags, then EC2 runs `docker compose pull` and `docker compose up -d --no-build`.
- Do not add systemd/Linux-service deployment steps back into the GitHub Actions workflow.
- Do not use `latest` for deployed application images. Use unique tags such as branch + commit SHA.

Runtime environment ownership:

- Root `.env` is for Docker Compose substitution: `DOCKERHUB_NAMESPACE`, service image tags, and `NEXT_PUBLIC_AGENT_URL`.
- `services/agent/.env` is for agent runtime config: `MODEL`, `AWS_REGION`, `AWS_S3_BUCKET`.
- `services/yolo/.env` is for YOLO runtime config: `CONFIDENCE_THRESHOLD`, `AWS_REGION`, `AWS_S3_BUCKET`.
- EC2 owns the real `.env` files. CI must not overwrite service env files.
- Do not commit real `.env` files or AWS credentials.

Networking rules:

- The agent must call YOLO with `YOLO_SERVICE_URL=http://yolo:8080` in Docker Compose.
- The agent must call the image-processing MCP service with `IMG_PROC_MCP_URL=http://img-proc-mcp:8090` in Docker Compose.
- Prometheus scrapes YOLO with target `yolo:8080`.
- Grafana connects to Prometheus with `http://prometheus:9090`.
- The browser-facing frontend uses `NEXT_PUBLIC_AGENT_URL`, such as `http://localhost:8000`, `http://dev.weam.fursa.click:8000`, or `http://prod.weam.fursa.click:8000`.

AWS credential rules:

- Local Docker may use the ignored override file to mount `~/.aws:/root/.aws:ro`.
- EC2 must use an attached IAM role for Bedrock and S3 access.
- Never put AWS access keys in source code, Dockerfiles, Docker images, or committed env examples.

---

## Course Content Reference

The course curriculum lives at: `github.com/alonitac/Fursa26`

When you are **unsure what concepts have been taught** - e.g., has the course covered `async`/`await`? TypeScript generics? React hooks? - use the GitHub search tools to look up relevant course materials in that repo before writing code.

Search for things like:
- Lesson or tutorial files mentioning the concept in question
- README files describing module or session goals
- Example code in the curriculum that sets the expected style and complexity level

If you find that a concept has not been taught yet, either avoid it or flag it clearly to the student.

---

## Skill Eval Reports

When using a skill from `.agents/skills/`, check whether that skill has an `evals/evals.json` file.

If it does:
- Select the eval cases relevant to the user's task.
- Use their assertions as a checklist while doing the work.
- Before finishing, write a dated report in that skill's `reports/` folder.
- The report must include the report date/time, the user prompt, which evals were used, which evals passed, which failed or were not run, and short evidence for each status.

If no evals exist for the skill, say that no skill evals were available.


## Architecture Constraints

### The LLM never sees image data

The LLM receives **text only**. Images are handled exclusively by the YOLO microservice.

- The `chat()` endpoint in `services/agent/app.py` must strip `image_base64` before building LangChain messages.
- The agent uploads the original image to S3 and stores only the S3 key in `_current_image_s3_key`.
- The `detect_objects` tool sends YOLO only `image_s3_key`; it must not send image bytes to YOLO.
- YOLO downloads the original image from S3, uploads the predicted image to S3, and returns `predicted_image_s3_key`.
- The agent may read the predicted image from S3 and return it to the frontend as base64, but it must not add image bytes to LangChain messages.
- Do **not** add multimodal content (e.g. `image_url`) to `HumanMessage`. The model's role is conversation management, not vision.

---

## Coding Principles

### Keep it explicit, not magic
Prefer readable, step-by-step code over clever abstractions.
Students must be able to follow the execution flow line by line.

**Good:**
```python
response = llm_with_tools.invoke(messages)
for tool_call in response.tool_calls:
    result = TOOLS[tool_call["name"]].invoke(tool_call)
    messages.append(result)
```

**Avoid:**
```python
result = create_react_agent(llm, tools).invoke(state)
```

### Do not use high-level agent frameworks as a black box
`create_react_agent`, `AgentExecutor`, and similar wrappers hide the loop that students need to learn.
Implement the ReAct loop manually in `run_agent()` inside `services/agent/app.py`.


## Kubernetes

When working with Kubernetes manifests, Helm charts, or Kustomize overlays, follow the workflow in `.kubernetes-skill/SKILL.md`.
Load references from `.kubernetes-skill/references/` as needed.