import base64
import io
import json
import logging
import math
import os
import time
from contextvars import ContextVar
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
AWS_REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
BEDROCK_MODEL_PREFIX = "bedrock/"
DEFAULT_MODEL = f"{BEDROCK_MODEL_PREFIX}amazon.nova-lite-v1:0"
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
BEDROCK_MODEL_ID = MODEL.removeprefix(BEDROCK_MODEL_PREFIX)

# Bedrock text-only models allowed for the course.
ALLOWED_BEDROCK_MODEL_IDS = {
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "openai.gpt-oss-20b-1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
}

LLM_REQUESTS_PER_SECOND = 0.25  # 15 request per minute
LLM_RATE_LIMIT_CHECK_SECONDS = 0.1
LLM_RATE_LIMIT_BUCKET_SIZE = 1
LLM_RATE_LIMIT_SECONDS = 1 / LLM_REQUESTS_PER_SECOND

# Text-only models
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
    "google_genai:gemini-2.5-flash",
}

if BEDROCK_MODEL_ID not in ALLOWED_BEDROCK_MODEL_IDS:
    allowed_list = "\n  ".join(
        f"{BEDROCK_MODEL_PREFIX}{model_id}"
        for model_id in sorted(ALLOWED_BEDROCK_MODEL_IDS)
    )
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported Bedrock models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

REQUIRED_MODEL_FEATURES = ["structured_output", "tool_calling"]
CONTEXT_LIMIT_WARNING_RATIO = 0.9

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_chat_rate_limit_lock = Lock()
_next_chat_request_at = 0.0


def check_chat_rate_limit():
    """
    Return 429 immediately when the next LLM request slot is not ready yet.

    The LangChain rate limiter still protects the provider call itself. This
    check keeps users from waiting silently at the /chat API boundary.
    """
    global _next_chat_request_at

    now = time.monotonic()
    with _chat_rate_limit_lock:
        wait_seconds = _next_chat_request_at - now
        if wait_seconds > 0:
            retry_after = math.ceil(wait_seconds)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit reached. Please try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        _next_chat_request_at = now + LLM_RATE_LIMIT_SECONDS

@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = base64.b64decode(image_b64)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            files={"file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        response.raise_for_status()
    return json.dumps(response.json())


def fetch_annotated_image_b64(prediction_uid: str) -> Optional[str]:
    """
    Fetch the annotated image from YOLO and encode it for the API response.

    This value is not added to the LangChain messages, so the LLM still only
    receives the text JSON returned by the detection tool.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{YOLO_SERVICE_URL}/prediction/{prediction_uid}/image"
            )
            response.raise_for_status()
    except httpx.HTTPError:
        logging.exception("Could not fetch annotated image for %s", prediction_uid)
        return None

    return base64.b64encode(response.content).decode("utf-8")


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects
}

def validate_model_profile(model_name: Optional[str], profile: dict):
    """
    Stop the app early if the selected model cannot support this agent.
    """
    missing_features = []

    for feature in REQUIRED_MODEL_FEATURES:
        if profile.get(feature) is not True:
            missing_features.append(feature)

    max_input_tokens = profile.get("max_input_tokens")
    has_token_limit = isinstance(max_input_tokens, int) and max_input_tokens > 0

    if not missing_features and has_token_limit:
        return

    error_lines = [
        f"\n[ERROR] MODEL='{model_name}' does not have the required profile support."
    ]

    if missing_features:
        error_lines.append(
            "Missing or unsupported features: " + ", ".join(missing_features)
        )

    if not has_token_limit:
        error_lines.append("Missing or invalid profile value: max_input_tokens")

    error_lines.append("Model profile:")
    error_lines.append(json.dumps(profile, indent=2, sort_keys=True))

    raise SystemExit("\n".join(error_lines))


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


def read_token_usage(response: AIMessage) -> TokenUsage:
    """
    Convert LangChain usage metadata into the API response shape.
    """
    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens")

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        total=total_tokens,
    )


def add_token_usage(current: TokenUsage, latest: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input=current.input + latest.input,
        output=current.output + latest.output,
        total=current.total + latest.total,
    )


def is_near_context_limit(input_tokens: int, profile: dict) -> bool:
    max_input_tokens = profile["max_input_tokens"]
    warning_threshold = max_input_tokens * CONTEXT_LIMIT_WARNING_RATIO
    return input_tokens >= warning_threshold


llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=LLM_REQUESTS_PER_SECOND,
    check_every_n_seconds=LLM_RATE_LIMIT_CHECK_SECONDS,
    max_bucket_size=LLM_RATE_LIMIT_BUCKET_SIZE,
)
  
  
llm = init_chat_model(
    BEDROCK_MODEL_ID,
    model_provider="bedrock_converse",
    temperature=0,
    region_name=AWS_REGION,
)


MODEL_PROFILE = getattr(llm, "profile", None) or {}
validate_model_profile(MODEL, MODEL_PROFILE)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


class AgentRunResult(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


def run_agent(history: list, max_iterations: int = 10) -> AgentRunResult:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    tools_called = []
    prediction_id = None
    annotated_image = None
    tokens_used = TokenUsage()
    context_limit_exceeded = False

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        latest_tokens = read_token_usage(response)
        tokens_used = add_token_usage(tokens_used, latest_tokens)

        if latest_tokens.input and is_near_context_limit(
            latest_tokens.input,
            MODEL_PROFILE,
        ):
            context_limit_exceeded = True

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            return AgentRunResult(
                response=response.content,
                prediction_id=prediction_id,
                annotated_image=annotated_image,
                tokens_used=tokens_used,
                iterations=iteration,
                tools_called=tools_called,
                context_limit_exceeded=context_limit_exceeded,
            )

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tools_called.append(tool_call["name"])
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = tool_fn.invoke(tool_call)          # returns a ToolMessage
            messages.append(tool_result)

            try:
                tool_data = json.loads(tool_result.content)
                if tool_data.get("prediction_uid"):
                    prediction_id = tool_data["prediction_uid"]
                    annotated_image = fetch_annotated_image_b64(prediction_id)
            except json.JSONDecodeError:
                pass

    return AgentRunResult(
        response="I reached the maximum number of tool calls and could not finish safely.",
        prediction_id=prediction_id,
        annotated_image=annotated_image,
        tokens_used=tokens_used,
        iterations=max_iterations,
        tools_called=tools_called,
        context_limit_exceeded=True,
    )



app = FastAPI(title="Vision Agent")

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
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    check_chat_rate_limit()

    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64          # saved for detect_objects tool
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    token = _current_image_b64.set(latest_image)
    try:
        start_time = time.time()
        agent_result = run_agent(lc_messages)
        agent_loop_time_s = round(time.time() - start_time, 2)
        return ChatResponse(
            response=agent_result.response,
            prediction_id=agent_result.prediction_id,
            annotated_image=agent_result.annotated_image,
            tokens_used=agent_result.tokens_used,
            agent_loop_time_s=agent_loop_time_s,
            iterations=agent_result.iterations,
            tools_called=agent_result.tools_called,
            context_limit_exceeded=agent_result.context_limit_exceeded,
        )
    finally:
        _current_image_b64.reset(token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
