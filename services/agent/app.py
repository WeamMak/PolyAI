import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

import agent_loop
import config
import mcp_tools
from context import current_detection_objects as _current_detection_objects
from context import current_image_s3_key as _current_image_s3_key
from storage import upload_image_base64_to_s3, validate_active_image_key


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

AgentRunResult = agent_loop.AgentRunResult
TokenUsage = agent_loop.TokenUsage
run_agent = agent_loop.run_agent

_chat_rate_limit_lock = Lock()
_next_chat_request_at = 0.0


def check_chat_rate_limit() -> None:
    global _next_chat_request_at

    now = time.monotonic()
    with _chat_rate_limit_lock:
        wait_seconds = _next_chat_request_at - now
        if wait_seconds > 0:
            retry_after = math.ceil(wait_seconds)
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit reached. Please try again in "
                    f"{retry_after} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )

        _next_chat_request_at = now + config.LLM_RATE_LIMIT_SECONDS


async def initialize_agent_tools() -> None:
    mcp_tool_proxies = await mcp_tools.discover_mcp_tool_proxies()
    agent_loop.bind_mcp_tools(mcp_tool_proxies)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_agent_tools()
    yield


app = FastAPI(title="Vision Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://dev.weam.fursa.click:3000",
        "http://prod.weam.fursa.click:3000",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str
    content: str
    image_base64: Optional[str] = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    chat_id: Optional[str] = None
    active_image_s3_key: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    active_image_s3_key: Optional[str] = None
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    annotated_image_media_type: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    check_chat_rate_limit()

    chat_id = request.chat_id or str(uuid.uuid4())
    latest_image_b64 = None

    for message in request.messages:
        if message.role == "user" and message.image_base64:
            latest_image_b64 = message.image_base64

    if latest_image_b64:
        active_image_s3_key = upload_image_base64_to_s3(
            latest_image_b64,
            chat_id,
        )
    elif request.active_image_s3_key:
        active_image_s3_key = validate_active_image_key(
            chat_id,
            request.active_image_s3_key,
        )
    else:
        active_image_s3_key = None

    lc_messages = []
    last_user_index = None

    for message in request.messages:
        if message.role == "user":
            last_user_index = len(lc_messages)
            lc_messages.append(HumanMessage(content=message.content))
        else:
            lc_messages.append(AIMessage(content=message.content))

    if active_image_s3_key and last_user_index is not None:
        latest_user_message = lc_messages[last_user_index]
        latest_user_message.content += (
            "\n[An image is active. Use the available tools according to the "
            "user's instructions.]"
        )

    image_token = _current_image_s3_key.set(active_image_s3_key)
    detections_token = _current_detection_objects.set(None)

    try:
        start_time = time.time()
        agent_result = await run_agent(lc_messages)
        agent_loop_time_s = round(time.time() - start_time, 2)
        final_active_image_s3_key = _current_image_s3_key.get()

        return ChatResponse(
            response=agent_result.response,
            chat_id=chat_id,
            active_image_s3_key=final_active_image_s3_key,
            prediction_id=agent_result.prediction_id,
            annotated_image=agent_result.annotated_image,
            annotated_image_media_type=(
                agent_result.annotated_image_media_type
            ),
            tokens_used=agent_result.tokens_used,
            agent_loop_time_s=agent_loop_time_s,
            iterations=agent_result.iterations,
            tools_called=agent_result.tools_called,
            context_limit_exceeded=agent_result.context_limit_exceeded,
        )
    finally:
        _current_detection_objects.reset(detections_token)
        _current_image_s3_key.reset(image_token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
