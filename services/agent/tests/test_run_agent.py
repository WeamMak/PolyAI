import base64
import io
import json

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class FakeTool:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, tool_call):
        self.calls.append(tool_call)
        return ToolMessage(
            content=self.content,
            tool_call_id=tool_call["id"],
        )


class FakeYoloResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "prediction_uid": "prediction-123",
            "detection_count": 1,
            "labels": ["person"],
            "time_took": 0.2,
            "predicted_image_s3_key": (
                "chats/chat-123/image-123/predicted/image.jpg"
            ),
        }


class FakeHttpClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        pass

    def post(self, url, json):
        self.url = url
        self.json_body = json
        return FakeYoloResponse()


class FakeS3Body:
    def read(self):
        return b"annotated image bytes"


class FakeS3Client:
    def __init__(self):
        self.bucket = None
        self.key = None

    def get_object(self, Bucket, Key):
        self.bucket = Bucket
        self.key = Key
        return {"Body": FakeS3Body()}


def make_png_bytes(size=(40, 30), color="red"):
    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def make_png_b64(size=(40, 30), color="red"):
    return base64.b64encode(make_png_bytes(size, color)).decode("utf-8")


def test_run_agent_returns_final_response_without_tools(agent_module, monkeypatch):
    fake_llm = FakeLLM([AIMessage(content="Hello from the agent.")])
    monkeypatch.setattr(agent_module, "llm_with_tools", fake_llm)

    result = agent_module.run_agent([HumanMessage(content="Hello")])

    assert result.response == "Hello from the agent."
    assert result.iterations == 1
    assert result.tools_called == []
    assert result.context_limit_exceeded is False
    assert fake_llm.calls[0][0].type == "system"
    assert fake_llm.calls[0][1].content == "Hello"


def test_run_agent_executes_tool_call(agent_module, monkeypatch):
    tool_call = {
        "name": "detect_objects",
        "args": {},
        "id": "call-1",
    }
    fake_llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="The image contains a person."),
        ]
    )
    fake_tool = FakeTool(
        json.dumps(
            {
                "prediction_uid": "prediction-123",
                "predicted_image_s3_key": (
                    "chats/chat-123/image-123/predicted/image.jpg"
                ),
            }
        )
    )
    captured = {}

    monkeypatch.setattr(agent_module, "llm_with_tools", fake_llm)
    monkeypatch.setattr(agent_module, "TOOLS", {"detect_objects": fake_tool})

    def fake_fetch_s3_image_b64(image_s3_key):
        captured["predicted_image_s3_key"] = image_s3_key
        return "annotated-image-b64"

    monkeypatch.setattr(
        agent_module,
        "fetch_s3_image_b64",
        fake_fetch_s3_image_b64,
    )

    result = agent_module.run_agent([HumanMessage(content="Detect objects")])

    assert result.response == "The image contains a person."
    assert result.prediction_id == "prediction-123"
    assert result.annotated_image == "annotated-image-b64"
    assert result.iterations == 2
    assert result.tools_called == ["detect_objects"]
    assert (
        captured["predicted_image_s3_key"]
        == "chats/chat-123/image-123/predicted/image.jpg"
    )
    assert fake_tool.calls[0]["name"] == "detect_objects"
    assert fake_tool.calls[0]["id"] == "call-1"
    assert any(isinstance(message, ToolMessage) for message in fake_llm.calls[1])


def test_run_agent_fetches_processed_image_for_frontend(agent_module, monkeypatch):
    tool_call = {
        "name": "process_image",
        "args": {"operation": "rotate", "angle": 90},
        "id": "call-1",
    }
    fake_llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="Rotated the image."),
        ]
    )
    fake_tool = FakeTool(
        json.dumps(
            {
                "processed_image_s3_key": (
                    "chats/chat-123/image-123/processed/rotate.png"
                ),
            }
        )
    )
    captured = {}

    monkeypatch.setattr(agent_module, "llm_with_tools", fake_llm)
    monkeypatch.setattr(agent_module, "TOOLS", {"process_image": fake_tool})

    def fake_fetch_s3_image_b64(image_s3_key):
        captured["processed_image_s3_key"] = image_s3_key
        return "processed-image-b64"

    monkeypatch.setattr(
        agent_module,
        "fetch_s3_image_b64",
        fake_fetch_s3_image_b64,
    )

    result = agent_module.run_agent([HumanMessage(content="Rotate this image")])

    assert result.response == "Rotated the image."
    assert result.annotated_image == "processed-image-b64"
    assert result.tools_called == ["process_image"]
    assert (
        captured["processed_image_s3_key"]
        == "chats/chat-123/image-123/processed/rotate.png"
    )


def test_detect_objects_sends_s3_key_to_yolo(agent_module, monkeypatch):
    fake_client = FakeHttpClient(timeout=30.0)
    monkeypatch.setattr(agent_module.httpx, "Client", lambda timeout: fake_client)

    token = agent_module._current_image_s3_key.set(
        "chats/chat-123/image-123/original/image.jpg"
    )
    try:
        result = agent_module.detect_objects.invoke({})
    finally:
        agent_module._current_image_s3_key.reset(token)

    assert fake_client.timeout == 30.0
    assert fake_client.url == f"{agent_module.YOLO_SERVICE_URL}/predict"
    assert fake_client.json_body == {
        "image_s3_key": "chats/chat-123/image-123/original/image.jpg"
    }
    assert json.loads(result) == {
        "prediction_uid": "prediction-123",
        "detection_count": 1,
        "labels": ["person"],
        "time_took": 0.2,
        "predicted_image_s3_key": "chats/chat-123/image-123/predicted/image.jpg",
    }


def test_process_image_rotates_entire_image_with_mcp(agent_module, monkeypatch):
    image_s3_key = "chats/chat-123/image-123/original/image.png"
    uploaded = {}
    called = {}

    monkeypatch.setattr(
        agent_module,
        "read_s3_image_bytes",
        lambda key: make_png_bytes(size=(40, 30), color="red"),
    )
    monkeypatch.setattr(
        agent_module,
        "build_processed_image_s3_key",
        lambda key, operation: "chats/chat-123/image-123/processed/rotate.png",
    )

    def fake_call_img_proc_mcp_tool(tool_name, arguments):
        called["tool_name"] = tool_name
        called["arguments"] = arguments
        return make_png_b64(size=(30, 40), color="red")

    def fake_upload_image_bytes_to_s3(image_bytes, key, content_type):
        uploaded["image_bytes"] = image_bytes
        uploaded["key"] = key
        uploaded["content_type"] = content_type

    monkeypatch.setattr(agent_module, "call_img_proc_mcp_tool", fake_call_img_proc_mcp_tool)
    monkeypatch.setattr(agent_module, "upload_image_bytes_to_s3", fake_upload_image_bytes_to_s3)

    token = agent_module._current_image_s3_key.set(image_s3_key)
    try:
        result = agent_module.process_image.invoke(
            {"operation": "rotate", "target": "entire_image", "angle": 90}
        )
    finally:
        agent_module._current_image_s3_key.reset(token)

    data = json.loads(result)

    assert data["operation"] == "rotate"
    assert data["processed_image_s3_key"] == "chats/chat-123/image-123/processed/rotate.png"
    assert called["tool_name"] == "rotate"
    assert called["arguments"]["angle"] == 90
    assert "image_b64" in called["arguments"]
    assert uploaded["key"] == "chats/chat-123/image-123/processed/rotate.png"
    assert uploaded["content_type"] == "image/png"
    assert Image.open(io.BytesIO(uploaded["image_bytes"])).size == (30, 40)


def test_process_image_blurs_second_dog_from_right(agent_module, monkeypatch):
    image_s3_key = "chats/chat-123/image-123/original/image.png"
    called = {}

    monkeypatch.setattr(
        agent_module,
        "read_s3_image_bytes",
        lambda key: make_png_bytes(size=(100, 50), color="white"),
    )
    monkeypatch.setattr(
        agent_module,
        "request_yolo_prediction",
        lambda key: {"prediction_uid": "prediction-123"},
    )
    monkeypatch.setattr(
        agent_module,
        "get_yolo_prediction_details",
        lambda uid: {
            "detection_objects": [
                {"id": 1, "label": "dog", "score": 0.9, "box": "[0, 0, 10, 20]"},
                {"id": 2, "label": "dog", "score": 0.8, "box": "[40, 0, 50, 20]"},
                {"id": 3, "label": "dog", "score": 0.7, "box": "[80, 0, 90, 20]"},
            ]
        },
    )
    monkeypatch.setattr(
        agent_module,
        "build_processed_image_s3_key",
        lambda key, operation: "chats/chat-123/image-123/processed/blur.png",
    )
    monkeypatch.setattr(
        agent_module,
        "upload_image_bytes_to_s3",
        lambda image_bytes, key, content_type: None,
    )

    def fake_call_img_proc_mcp_tool(tool_name, arguments):
        crop = Image.open(io.BytesIO(base64.b64decode(arguments["image_b64"])))
        called["tool_name"] = tool_name
        called["crop_size"] = crop.size
        called["radius"] = arguments["radius"]
        return make_png_b64(size=crop.size, color="black")

    monkeypatch.setattr(agent_module, "call_img_proc_mcp_tool", fake_call_img_proc_mcp_tool)

    token = agent_module._current_image_s3_key.set(image_s3_key)
    try:
        result = agent_module.process_image.invoke(
            {
                "operation": "blur",
                "target": "object",
                "label": "dog",
                "ordinal": 2,
                "from_side": "right",
                "radius": 5,
            }
        )
    finally:
        agent_module._current_image_s3_key.reset(token)

    data = json.loads(result)

    assert data["operation"] == "blur"
    assert data["prediction_uid"] == "prediction-123"
    assert data["selected_object"]["id"] == 2
    assert data["selected_object"]["box"] == [40, 0, 50, 20]
    assert called == {"tool_name": "blur", "crop_size": (10, 20), "radius": 5}


def test_upload_image_without_bucket_returns_client_safe_error(agent_module, monkeypatch):
    monkeypatch.setattr(agent_module, "AWS_S3_BUCKET", None)

    with pytest.raises(HTTPException) as exc:
        agent_module.upload_image_base64_to_s3("unused-image-data", "chat-123")

    assert exc.value.status_code == 500
    assert exc.value.detail == "Image upload is currently unavailable"


def test_upload_image_with_invalid_data_returns_client_safe_error(
    agent_module,
    monkeypatch,
):
    monkeypatch.setattr(agent_module, "AWS_S3_BUCKET", "polyai-images")

    with pytest.raises(HTTPException) as exc:
        agent_module.upload_image_base64_to_s3("not-base64", "chat-123")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid image data"


def test_fetch_s3_image_returns_base64_for_frontend(agent_module, monkeypatch):
    fake_s3_client = FakeS3Client()
    monkeypatch.setattr(agent_module, "AWS_S3_BUCKET", "polyai-images")
    monkeypatch.setattr(agent_module, "get_s3_client", lambda: fake_s3_client)

    result = agent_module.fetch_s3_image_b64(
        "chats/chat-123/image-123/predicted/image.jpg"
    )

    expected = base64.b64encode(b"annotated image bytes").decode("utf-8")
    assert result == expected
    assert fake_s3_client.bucket == "polyai-images"
    assert fake_s3_client.key == "chats/chat-123/image-123/predicted/image.jpg"
