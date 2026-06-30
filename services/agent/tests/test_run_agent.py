import base64
import json

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


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
