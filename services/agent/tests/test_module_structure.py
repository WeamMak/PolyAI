from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]


def read_module(filename: str) -> str:
    return (AGENT_DIR / filename).read_text()


def test_app_is_only_the_http_composition_layer():
    app_source = read_module("app.py")

    assert "import boto3" not in app_source
    assert "import httpx" not in app_source
    assert "MultiServerMCPClient" not in app_source
    assert "init_chat_model" not in app_source
    assert "for iteration in range" not in app_source


def test_manual_agent_loop_remains_explicit():
    agent_loop_source = read_module("agent_loop.py")

    assert "async def run_agent" in agent_loop_source
    assert "for iteration in range" in agent_loop_source
    assert "await tool_fn.ainvoke(tool_call)" in agent_loop_source


def test_mcp_adapter_is_imported_only_by_mcp_tools_module():
    modules_with_adapter = []

    for module_path in AGENT_DIR.glob("*.py"):
        if "langchain_mcp_adapters" in module_path.read_text():
            modules_with_adapter.append(module_path.name)

    assert modules_with_adapter == ["mcp_tools.py"]


def test_agent_service_contains_no_pillow_import():
    for module_path in AGENT_DIR.glob("*.py"):
        source = module_path.read_text()
        assert "from PIL" not in source
        assert "import PIL" not in source


def test_agent_dockerfile_copies_every_runtime_module():
    dockerfile = read_module("Dockerfile")

    for module_name in [
        "app.py",
        "agent_loop.py",
        "config.py",
        "context.py",
        "mcp_tools.py",
        "storage.py",
        "yolo_client.py",
    ]:
        assert module_name in dockerfile
