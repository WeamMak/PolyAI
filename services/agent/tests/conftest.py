import importlib
import sys
from pathlib import Path

import pytest


class StartupModelStub:
    profile = {
        "max_input_tokens": 1000,
        "structured_output": True,
        "tool_calling": True,
    }

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise AssertionError("Tests should replace llm_with_tools before invoking.")


@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("MODEL", "bedrock/openai.gpt-oss-20b-1:0")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    agent_dir = Path(__file__).resolve().parents[1]
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    import langchain.chat_models

    monkeypatch.setattr(
        langchain.chat_models,
        "init_chat_model",
        lambda *args, **kwargs: StartupModelStub(),
    )

    sys.modules.pop("app", None)
    module = importlib.import_module("app")

    try:
        yield module
    finally:
        sys.modules.pop("app", None)