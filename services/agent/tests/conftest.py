import importlib
import inspect
import sys
from types import SimpleNamespace
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


AGENT_MODULE_NAMES = [
    "app",
    "agent_loop",
    "config",
    "context",
    "mcp_tools",
    "storage",
    "yolo_client",
]


@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("MODEL", "bedrock/openai.gpt-oss-20b-1:0")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    agent_dir = Path(__file__).resolve().parents[1]
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    import langchain.chat_models
    import starlette.routing

    router_parameters = inspect.signature(
        starlette.routing.Router.__init__
    ).parameters

    if sys.version_info >= (3, 14) and "on_startup" not in router_parameters:
        original_router_init = starlette.routing.Router.__init__

        def compatible_router_init(
            self,
            routes=None,
            redirect_slashes=True,
            default=None,
            on_startup=None,
            on_shutdown=None,
            lifespan=None,
            **kwargs,
        ):
            original_router_init(
                self,
                routes=routes,
                redirect_slashes=redirect_slashes,
                default=default,
                lifespan=lifespan,
                **kwargs,
            )
            self.on_startup = list(on_startup or [])
            self.on_shutdown = list(on_shutdown or [])

        monkeypatch.setattr(
            starlette.routing.Router,
            "__init__",
            compatible_router_init,
        )

    monkeypatch.setattr(
        langchain.chat_models,
        "init_chat_model",
        lambda *args, **kwargs: StartupModelStub(),
    )

    for module_name in AGENT_MODULE_NAMES:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("app")

    try:
        yield module
    finally:
        for module_name in AGENT_MODULE_NAMES:
            sys.modules.pop(module_name, None)


@pytest.fixture
def agent_components(agent_module):
    return SimpleNamespace(
        app=agent_module,
        agent_loop=importlib.import_module("agent_loop"),
        config=importlib.import_module("config"),
        context=importlib.import_module("context"),
        mcp_tools=importlib.import_module("mcp_tools"),
        storage=importlib.import_module("storage"),
        yolo_client=importlib.import_module("yolo_client"),
    )
