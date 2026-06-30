import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage


pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="FastAPI TestClient hangs in the local Python 3.14 environment.",
)


def test_health_endpoint(agent_module):
    with TestClient(agent_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_uses_mocked_agent_loop(agent_module, monkeypatch):
    captured = {}
    uploaded = {}
    image_s3_key = "chats/chat-123/image-123/original/image.jpg"

    def fake_upload_image_base64_to_s3(image_base64, chat_id):
        uploaded["image_base64"] = image_base64
        uploaded["chat_id"] = chat_id
        return image_s3_key

    def fake_run_agent(history):
        captured["history"] = history
        captured["image_s3_key"] = agent_module._current_image_s3_key.get()
        return agent_module.AgentRunResult(
            response="I found one person.",
            prediction_id="prediction-123",
            annotated_image="annotated-image-b64",
            iterations=2,
            tools_called=["detect_objects"],
            context_limit_exceeded=False,
        )

    monkeypatch.setattr(
        agent_module,
        "upload_image_base64_to_s3",
        fake_upload_image_base64_to_s3,
    )
    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)

    if hasattr(agent_module, "check_chat_rate_limit"):
        monkeypatch.setattr(agent_module, "check_chat_rate_limit", lambda: None)

    with TestClient(agent_module.app) as client:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What is in this image?",
                        "image_base64": "raw-image-b64",
                    },
                    {
                        "role": "assistant",
                        "content": "Previous answer.",
                    },
                ]
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "I found one person."
    assert data["prediction_id"] == "prediction-123"
    assert data["annotated_image"] == "annotated-image-b64"
    assert data["iterations"] == 2
    assert data["tools_called"] == ["detect_objects"]
    assert data["context_limit_exceeded"] is False
    assert isinstance(data["agent_loop_time_s"], float)

    assert uploaded["image_base64"] == "raw-image-b64"
    assert uploaded["chat_id"]
    assert captured["image_s3_key"] == image_s3_key
    assert isinstance(captured["history"][0], HumanMessage)
    assert isinstance(captured["history"][1], AIMessage)
    assert "raw-image-b64" not in captured["history"][0].content
    assert "[An image was uploaded." in captured["history"][0].content
    assert agent_module._current_image_s3_key.get() is None
