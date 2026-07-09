import base64
import json

import anyio
import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


PNG_BYTES = b"\x89PNG\r\n\x1a\nprocessed-image"


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        return ToolMessage(
            content=self.content,
            tool_call_id=tool_call["id"],
        )


class FakeMCPTool:
    name = "blur"
    description = "Blur an image or selected object."
    args_schema = {
        "type": "object",
        "properties": {
            "image_b64": {"type": "string"},
            "detection_objects": {"type": "array", "items": {"type": "object"}},
            "target": {"type": "string", "default": "entire_image"},
            "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "ordinal": {"type": "integer", "default": 1},
            "from_side": {"type": "string", "default": "left"},
            "radius": {"type": "number", "default": 2.0},
        },
        "required": ["image_b64"],
    }

    def __init__(self):
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return [
            {
                "type": "text",
                "text": base64.b64encode(PNG_BYTES).decode("utf-8"),
            }
        ]


class RawStringMCPTool(FakeMCPTool):
    async def ainvoke(self, arguments):
        return base64.b64encode(PNG_BYTES).decode("utf-8")


def run_async(async_function, *args):
    async def runner():
        return await async_function(*args)

    return anyio.run(runner)


def test_run_agent_returns_final_response_without_tools(
    agent_components,
    monkeypatch,
):
    agent_loop = agent_components.agent_loop
    fake_llm = FakeLLM([AIMessage(content="Hello from the agent.")])
    monkeypatch.setattr(agent_loop, "llm_with_tools", fake_llm)

    result = run_async(
        agent_loop.run_agent,
        [HumanMessage(content="Hello")],
    )

    assert result.response == "Hello from the agent."
    assert result.iterations == 1
    assert result.tools_called == []
    assert fake_llm.calls[0][0].type == "system"
    assert fake_llm.calls[0][1].content == "Hello"


def test_run_agent_removes_harmony_channel_marker_from_tool_name(
    agent_components,
    monkeypatch,
    caplog,
):
    agent_loop = agent_components.agent_loop
    malformed_call = {
        "name": "blur<|channel|>",
        "args": {"target": "entire_image", "radius": 2},
        "id": "blur-1",
    }
    fake_llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[malformed_call]),
            AIMessage(content="Blurred the image."),
        ]
    )
    blur_tool = FakeTool(
        json.dumps({"message": "Image processing completed."})
    )
    monkeypatch.setattr(agent_loop, "llm_with_tools", fake_llm)
    monkeypatch.setattr(agent_loop, "TOOLS", {"blur": blur_tool})

    result = run_async(
        agent_loop.run_agent,
        [HumanMessage(content="Blur this image")],
    )

    assert result.response == "Blurred the image."
    assert result.tools_called == ["blur"]
    assert blur_tool.calls[0]["name"] == "blur"
    assert "Normalized malformed tool name" in caplog.text


def test_run_agent_removes_harmony_channel_name_from_tool_name(
    agent_components,
    monkeypatch,
    caplog,
):
    agent_loop = agent_components.agent_loop
    malformed_call = {
        "name": "detect_objects<|channel|>commentary",
        "args": {},
        "id": "detect-1",
    }
    fake_llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[malformed_call]),
            AIMessage(content="Found the objects."),
        ]
    )
    detect_tool = FakeTool(
        json.dumps(
            {
                "prediction_uid": "prediction-1",
                "detection_objects": [],
            }
        )
    )
    monkeypatch.setattr(agent_loop, "llm_with_tools", fake_llm)
    monkeypatch.setattr(agent_loop, "TOOLS", {"detect_objects": detect_tool})

    result = run_async(
        agent_loop.run_agent,
        [HumanMessage(content="What is in this image?")],
    )

    assert result.response == "Found the objects."
    assert result.tools_called == ["detect_objects"]
    assert detect_tool.calls[0]["name"] == "detect_objects"
    assert "Normalized malformed tool name" in caplog.text


def test_run_agent_still_rejects_unknown_tool_name(
    agent_components,
    monkeypatch,
):
    agent_loop = agent_components.agent_loop
    malformed_call = {
        "name": "sharpen<|channel|>commentary",
        "args": {"target": "entire_image", "radius": 2},
        "id": "sharpen-1",
    }
    fake_llm = FakeLLM(
        [AIMessage(content="", tool_calls=[malformed_call])]
    )
    monkeypatch.setattr(agent_loop, "llm_with_tools", fake_llm)
    monkeypatch.setattr(
        agent_loop,
        "TOOLS",
        {"blur": FakeTool("{}")},
    )

    with pytest.raises(
        ValueError,
        match=r"Unknown tool requested: sharpen<\|channel\|>commentary",
    ):
        run_async(
            agent_loop.run_agent,
            [HumanMessage(content="Blur this image")],
        )


def test_run_agent_executes_detect_then_discovered_image_tool(
    agent_components,
    monkeypatch,
):
    agent_loop = agent_components.agent_loop
    request_context = agent_components.context
    detect_call = {"name": "detect_objects", "args": {}, "id": "detect-1"}
    blur_call = {
        "name": "blur",
        "args": {
            "target": "object",
            "label": "dog",
            "ordinal": 2,
            "from_side": "right",
            "radius": 5,
        },
        "id": "blur-1",
    }
    fake_llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[detect_call]),
            AIMessage(content="", tool_calls=[blur_call]),
            AIMessage(content="Blurred the second dog."),
        ]
    )
    detections = [
        {"label": "dog", "box": "[0, 0, 10, 10]"},
        {"label": "dog", "box": "[20, 0, 30, 10]"},
    ]
    detect_tool = FakeTool(
        json.dumps(
            {
                "prediction_uid": "prediction-123",
                "detection_objects": detections,
            }
        )
    )
    blur_tool = FakeTool(
        json.dumps(
            {
                "processed_image_s3_key": (
                    "chats/chat-1/image-1/processed/id/blur.png"
                )
            }
        )
    )

    monkeypatch.setattr(agent_loop, "llm_with_tools", fake_llm)
    monkeypatch.setattr(
        agent_loop,
        "TOOLS",
        {"detect_objects": detect_tool, "blur": blur_tool},
    )
    monkeypatch.setattr(
        agent_loop.storage,
        "fetch_s3_image",
        lambda key: ("processed-b64", "image/png"),
    )

    image_token = request_context.current_image_s3_key.set(
        "chats/chat-1/image-1/original/image.png"
    )
    detection_token = request_context.current_detection_objects.set(None)
    try:
        async def run_and_read_active_key():
            result = await agent_loop.run_agent(
                [HumanMessage(content="Blur the second dog")]
            )
            return result, request_context.current_image_s3_key.get()

        result, active_key = anyio.run(run_and_read_active_key)
    finally:
        request_context.current_detection_objects.reset(detection_token)
        request_context.current_image_s3_key.reset(image_token)

    assert result.response == "Blurred the second dog."
    assert result.prediction_id == "prediction-123"
    assert result.annotated_image == "processed-b64"
    assert result.annotated_image_media_type == "image/png"
    assert result.tools_called == ["detect_objects", "blur"]
    assert active_key.endswith("/blur.png")
    assert blur_tool.calls[0]["args"]["label"] == "dog"
    assert all(
        "processed/id/blur.png" not in str(message.content)
        for message in fake_llm.calls[-1]
    )


def test_public_mcp_schema_hides_only_private_transport_fields(agent_components):
    schema = agent_components.mcp_tools.public_mcp_tool_schema(FakeMCPTool())

    assert "image_b64" not in schema["properties"]
    assert "detection_objects" not in schema["properties"]
    assert schema["required"] == []
    assert set(schema["properties"]) == {
        "target",
        "label",
        "ordinal",
        "from_side",
        "radius",
    }


def test_generic_mcp_proxy_injects_image_and_yolo_results(
    agent_components,
    monkeypatch,
):
    mcp_tools = agent_components.mcp_tools
    request_context = agent_components.context
    mcp_tool = FakeMCPTool()
    uploads = []
    detections = [{"label": "dog", "box": "[10, 0, 20, 10]"}]

    monkeypatch.setattr(
        mcp_tools.storage,
        "read_s3_bytes",
        lambda key: b"\x89PNG\r\n\x1a\noriginal-image",
    )
    monkeypatch.setattr(
        mcp_tools.storage,
        "build_processed_image_s3_key",
        lambda key, name: "chats/chat-1/image-1/processed/id/blur.png",
    )
    monkeypatch.setattr(
        mcp_tools.storage,
        "upload_bytes_to_s3",
        lambda data, key, content_type: uploads.append(
            (data, key, content_type)
        ),
    )

    image_token = request_context.current_image_s3_key.set(
        "chats/chat-1/image-1/original/image.png"
    )
    detection_token = request_context.current_detection_objects.set(detections)
    try:
        proxy = mcp_tools.create_mcp_tool_proxy(mcp_tool)
        tool_message = run_async(
            proxy.ainvoke,
            {
                "name": "blur",
                "args": {
                    "target": "object",
                    "label": "dog",
                    "ordinal": 1,
                    "from_side": "left",
                    "radius": 5,
                },
                "id": "blur-1",
                "type": "tool_call",
            },
        )
    finally:
        request_context.current_detection_objects.reset(detection_token)
        request_context.current_image_s3_key.reset(image_token)

    assert mcp_tool.calls[0]["detection_objects"] == detections
    assert mcp_tool.calls[0]["radius"] == 5
    assert base64.b64decode(mcp_tool.calls[0]["image_b64"]).startswith(
        b"\x89PNG"
    )
    assert uploads == [
        (
            PNG_BYTES,
            "chats/chat-1/image-1/processed/id/blur.png",
            "image/png",
        )
    ]
    assert "image_b64" not in tool_message.content
    assert json.loads(tool_message.content)["processed_image_s3_key"].endswith(
        "/blur.png"
    )


def test_mcp_proxy_rejects_a_raw_server_result_without_adapter_shape(
    agent_components,
    monkeypatch,
):
    mcp_tools = agent_components.mcp_tools
    request_context = agent_components.context

    monkeypatch.setattr(
        mcp_tools.storage,
        "read_s3_bytes",
        lambda key: b"\x89PNG\r\n\x1a\noriginal-image",
    )
    monkeypatch.setattr(
        mcp_tools.storage,
        "upload_bytes_to_s3",
        lambda *args: pytest.fail("invalid MCP output must not be uploaded"),
    )

    image_token = request_context.current_image_s3_key.set(
        "chats/chat-1/image-1/original/image.png"
    )
    try:
        result = run_async(
            mcp_tools.execute_mcp_image_tool,
            RawStringMCPTool(),
            {"target": "entire_image", "radius": 2},
        )
    finally:
        request_context.current_image_s3_key.reset(image_token)

    assert json.loads(result) == {
        "error": agent_components.config.IMAGE_EDIT_ERROR_MESSAGE
    }


def test_detect_objects_stores_and_returns_complete_yolo_json(
    agent_components,
    monkeypatch,
):
    yolo_client = agent_components.yolo_client
    request_context = agent_components.context
    detections = [{"label": "person", "box": "[0, 0, 10, 10]"}]

    async def fake_prediction(key):
        return {
            "prediction_uid": "prediction-1",
            "predicted_image_s3_key": "chats/chat-1/image-1/predicted/image.png",
        }

    async def fake_details(uid):
        return {"detection_objects": detections}

    monkeypatch.setattr(yolo_client, "request_yolo_prediction", fake_prediction)
    monkeypatch.setattr(yolo_client, "get_yolo_prediction_details", fake_details)

    image_token = request_context.current_image_s3_key.set(
        "chats/chat-1/image-1/original/image.png"
    )
    detection_token = request_context.current_detection_objects.set(None)
    try:
        result = run_async(yolo_client.detect_objects.ainvoke, {})
    finally:
        request_context.current_detection_objects.reset(detection_token)
        request_context.current_image_s3_key.reset(image_token)

    data = json.loads(result)
    assert data["prediction_uid"] == "prediction-1"
    assert data["detection_objects"] == detections


def test_validate_active_image_key_rejects_a_different_chat(agent_components):
    with pytest.raises(HTTPException) as exc:
        agent_components.storage.validate_active_image_key(
            "chat-1",
            "chats/chat-2/image-1/processed/id/blur.png",
        )

    assert exc.value.status_code == 400


def test_upload_image_with_invalid_data_returns_safe_error(
    agent_components,
    monkeypatch,
):
    monkeypatch.setattr(
        agent_components.config,
        "AWS_S3_BUCKET",
        "polyai-images",
    )

    with pytest.raises(HTTPException) as exc:
        agent_components.storage.upload_image_base64_to_s3(
            "not-base64",
            "chat-1",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid image data"
