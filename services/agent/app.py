import base64
import io
import json
import logging
import os
import time
from contextvars import ContextVar
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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)

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

llm = init_chat_model(
    BEDROCK_MODEL_ID,
    model_provider="bedrock_converse",
    temperature=0,
    region_name=AWS_REGION,
)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


class AgentRunResult(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
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

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            return AgentRunResult(
                response=response.content,
                prediction_id=prediction_id,
                annotated_image=annotated_image,
                iterations=iteration,
                tools_called=tools_called,
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
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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
