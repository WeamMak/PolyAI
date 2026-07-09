import json
import logging
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field

import config
import storage
from context import current_detection_objects, current_image_s3_key
from yolo_client import detect_objects


SYSTEM_PROMPT = (
    "You are an AI vision assistant. The model receives text only and must use "
    "tools for image work. Use detect_objects before every object-specific image "
    "operation. Read its detection JSON, then call the appropriate discovered "
    "image tool with target='object', label, ordinal, and from_side. For whole "
    "image requests, call the image tool with target='entire_image' and do not "
    "call detect_objects. Image tools are rotate, flip, blur, resize, crop, and "
    "add_noise. For two or more edits, call exactly one image tool per response "
    "and continue sequentially in the user's requested order. After any image "
    "edit, call detect_objects again before a later object-specific edit because "
    "the active image has changed. Never supply image_b64, detection_objects, or "
    "S3 keys; the agent injects them privately. Do not include markdown image "
    "placeholders because the frontend displays returned images separately."
)


def validate_model_profile(model_name: Optional[str], profile: dict) -> None:
    missing_features = []

    for feature in config.REQUIRED_MODEL_FEATURES:
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


class AgentRunResult(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    annotated_image_media_type: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


def read_token_usage(response: AIMessage) -> TokenUsage:
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


def read_response_text(response: AIMessage) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    text_parts.append(block_text)

        return "\n".join(text_parts)

    return str(content)


def is_near_context_limit(input_tokens: int, profile: dict) -> bool:
    max_input_tokens = profile["max_input_tokens"]
    warning_threshold = (
        max_input_tokens * config.CONTEXT_LIMIT_WARNING_RATIO
    )
    return input_tokens >= warning_threshold


llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=config.LLM_REQUESTS_PER_SECOND,
    check_every_n_seconds=config.LLM_RATE_LIMIT_CHECK_SECONDS,
    max_bucket_size=config.LLM_RATE_LIMIT_BUCKET_SIZE,
)

llm = init_chat_model(
    config.BEDROCK_MODEL_ID,
    model_provider="bedrock_converse",
    temperature=0,
    region_name=config.AWS_REGION,
    rate_limiter=llm_rate_limiter,
)

MODEL_PROFILE = getattr(llm, "profile", None) or {}
validate_model_profile(config.BEDROCK_MODEL_ID, MODEL_PROFILE)

TOOLS = {detect_objects.name: detect_objects}
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


def normalize_tool_name(raw_tool_name: str) -> str:
    channel_marker = "<|channel|>"

    if channel_marker not in raw_tool_name:
        return raw_tool_name

    tool_name = raw_tool_name.split(channel_marker, 1)[0]
    if not tool_name:
        return raw_tool_name

    return tool_name


def normalize_content_tool_names(content):
    if not isinstance(content, list):
        return content, False

    changed = False
    normalized_blocks = []

    for block in content:
        if not isinstance(block, dict):
            normalized_blocks.append(block)
            continue

        if block.get("type") == "tool_use":
            raw_tool_name = block.get("name")
            if isinstance(raw_tool_name, str):
                tool_name = normalize_tool_name(raw_tool_name)
                if tool_name != raw_tool_name:
                    normalized_block = dict(block)
                    normalized_block["name"] = tool_name
                    normalized_blocks.append(normalized_block)
                    changed = True
                    continue

        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            raw_tool_name = tool_use.get("name")
            if isinstance(raw_tool_name, str):
                tool_name = normalize_tool_name(raw_tool_name)
                if tool_name != raw_tool_name:
                    normalized_tool_use = dict(tool_use)
                    normalized_tool_use["name"] = tool_name
                    normalized_block = dict(block)
                    normalized_block["toolUse"] = normalized_tool_use
                    normalized_blocks.append(normalized_block)
                    changed = True
                    continue

        normalized_blocks.append(block)

    return normalized_blocks, changed


def normalize_ai_message_tool_names(response: AIMessage) -> AIMessage:
    changed = False
    normalized_tool_calls = []

    for tool_call in response.tool_calls:
        normalized_tool_call = dict(tool_call)
        raw_tool_name = normalized_tool_call.get("name")

        if isinstance(raw_tool_name, str):
            tool_name = normalize_tool_name(raw_tool_name)
            if tool_name != raw_tool_name:
                normalized_tool_call["name"] = tool_name
                changed = True

        normalized_tool_calls.append(normalized_tool_call)

    normalized_content, content_changed = normalize_content_tool_names(
        response.content
    )

    if not changed and not content_changed:
        return response

    return response.model_copy(
        update={
            "content": normalized_content,
            "tool_calls": normalized_tool_calls,
        }
    )


def bind_mcp_tools(mcp_tool_proxies: list) -> None:
    global TOOLS, llm_with_tools

    TOOLS = {detect_objects.name: detect_objects}
    TOOLS.update({proxy.name: proxy for proxy in mcp_tool_proxies})
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))


async def run_agent(
    history: list,
    max_iterations: int = 10,
) -> AgentRunResult:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    tools_called = []
    prediction_id = None
    annotated_image = None
    annotated_image_media_type = None
    tokens_used = TokenUsage()
    context_limit_exceeded = False

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(normalize_ai_message_tool_names(response))

        latest_tokens = read_token_usage(response)
        tokens_used = add_token_usage(tokens_used, latest_tokens)

        if latest_tokens.input and is_near_context_limit(latest_tokens.input, MODEL_PROFILE):
            context_limit_exceeded = True

        if not response.tool_calls:
            return AgentRunResult(
                response=read_response_text(response),
                prediction_id=prediction_id,
                annotated_image=annotated_image,
                annotated_image_media_type=annotated_image_media_type,
                tokens_used=tokens_used,
                iterations=iteration,
                tools_called=tools_called,
                context_limit_exceeded=context_limit_exceeded,
            )

        for tool_call in response.tool_calls:
            raw_tool_name = tool_call["name"]
            tool_name = normalize_tool_name(raw_tool_name)

            if tool_name != raw_tool_name:
                logging.warning(
                    "Normalized malformed tool name: %r",
                    raw_tool_name,
                )

            tools_called.append(tool_name)
            tool_fn = TOOLS.get(tool_name)

            if tool_fn is None:
                raise ValueError(
                    f"Unknown tool requested: {raw_tool_name}"
                )

            normalized_tool_call = dict(tool_call)
            normalized_tool_call["name"] = tool_name
            tool_result = await tool_fn.ainvoke(normalized_tool_call)

            try:
                tool_data = json.loads(tool_result.content)
            except (json.JSONDecodeError, TypeError):
                messages.append(tool_result)
                continue

            if tool_data.get("prediction_uid"):
                prediction_id = tool_data["prediction_uid"]

            if isinstance(tool_data.get("detection_objects"), list):
                current_detection_objects.set(tool_data["detection_objects"])

            processed_image_s3_key = tool_data.get("processed_image_s3_key")
            if processed_image_s3_key:
                current_image_s3_key.set(processed_image_s3_key)
                current_detection_objects.set(None)

            image_s3_key = processed_image_s3_key or tool_data.get("predicted_image_s3_key")
            if image_s3_key:
                annotated_image, annotated_image_media_type = storage.fetch_s3_image(image_s3_key)

            safe_tool_data = {
                key: value
                for key, value in tool_data.items()
                if not key.endswith("_s3_key")
            }
            messages.append(tool_result.model_copy(update={"content": json.dumps(safe_tool_data)}))

    return AgentRunResult(
        response=(
            "I reached the maximum number of tool calls and could not finish "
            "safely."
        ),
        prediction_id=prediction_id,
        annotated_image=annotated_image,
        annotated_image_media_type=annotated_image_media_type,
        tokens_used=tokens_used,
        iterations=max_iterations,
        tools_called=tools_called,
        context_limit_exceeded=True,
    )
