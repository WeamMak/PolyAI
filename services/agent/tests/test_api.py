import sys

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="FastAPI TestClient hangs in the local Python 3.14 environment.",
)


def test_health_endpoint(agent_module, monkeypatch):
    async def skip_mcp_startup():
        return None

    monkeypatch.setattr(agent_module, "initialize_agent_tools", skip_mcp_startup)

    with TestClient(agent_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_keeps_latest_processed_image_for_follow_up(
    agent_module,
    monkeypatch,
):
    original_key = "chats/chat-1/image-1/original/image.png"
    processed_key = "chats/chat-1/image-1/processed/id/blur.png"

    async def skip_mcp_startup():
        return None

    async def fake_run_agent(history):
        assert agent_module._current_image_s3_key.get() == original_key
        assert "[An image is active." in history[-1].content
        agent_module._current_image_s3_key.set(processed_key)
        return agent_module.AgentRunResult(
            response="Blurred the image.",
            annotated_image="processed-b64",
            annotated_image_media_type="image/png",
            iterations=1,
            tools_called=["blur"],
        )

    monkeypatch.setattr(agent_module, "initialize_agent_tools", skip_mcp_startup)
    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent_module, "check_chat_rate_limit", lambda: None)
    monkeypatch.setattr(
        agent_module,
        "upload_image_base64_to_s3",
        lambda image_b64, chat_id: original_key,
    )

    with TestClient(agent_module.app) as client:
        response = client.post(
            "/chat",
            json={
                "chat_id": "chat-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "Blur this image",
                        "image_base64": "raw-image-b64",
                    }
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["chat_id"] == "chat-1"
    assert data["active_image_s3_key"] == processed_key
    assert data["annotated_image_media_type"] == "image/png"
