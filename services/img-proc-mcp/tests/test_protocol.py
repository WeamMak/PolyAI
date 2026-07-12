import base64
import importlib.util
import io
from pathlib import Path

import anyio
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_mcp_server():
    app_file = PROJECT_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("img_proc_protocol_test", app_file)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {app_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mcp


def make_image_b64() -> str:
    img = Image.new("RGB", (20, 10), "red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_standard_mcp_session_discovers_and_calls_tool():
    async def run_protocol_test():
        mcp = load_mcp_server()

        async with create_connected_server_and_client_session(mcp) as session:
            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}

            assert tool_names == {
                "rotate",
                "flip",
                "blur",
                "resize",
                "crop",
                "add_noise",
            }

            call_result = await session.call_tool(
                "resize",
                {
                    "image_b64": make_image_b64(),
                    "width": 5,
                    "height": 4,
                },
            )

            assert call_result.isError is not True
            result_b64 = call_result.content[0].text
            result_img = Image.open(
                io.BytesIO(base64.b64decode(result_b64))
            )
            assert result_img.size == (5, 4)

    anyio.run(run_protocol_test)


def test_streamable_http_app_exposes_mcp_without_private_rest_bridge():
    mcp = load_mcp_server()
    app = mcp.streamable_http_app()
    route_paths = {route.path for route in app.routes}

    assert "/mcp" in route_paths
    assert "/tools/call" not in route_paths


def test_kubernetes_service_hostname_is_allowed():
    mcp = load_mcp_server()
    transport_security = mcp.settings.transport_security

    assert "img-proc-mcp-svc:*" in transport_security.allowed_hosts
    assert (
        "http://img-proc-mcp-svc:*"
        in transport_security.allowed_origins
    )
