import base64
import importlib.util
import io
import json
from pathlib import Path

import anyio
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_app_module():
    app_file = PROJECT_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("img_proc_mcp_app_test", app_file)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {app_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_image_b64(size=(40, 30), color="red"):
    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def decode_image(image_b64):
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def call_tool(module, name, arguments):
    result = anyio.run(module.mcp.call_tool, name, arguments)
    return result[1]["result"]


def test_mcp_registers_image_processing_tools():
    module = load_app_module()

    tools = anyio.run(module.mcp.list_tools)
    tool_names = {tool.name for tool in tools}

    assert {
        "rotate",
        "flip",
        "blur",
        "resize",
        "crop",
        "add_noise",
    }.issubset(tool_names)


def test_image_processing_tools_return_valid_png_images():
    module = load_app_module()
    image_b64 = make_image_b64()

    tests = [
        ("rotate", {"image_b64": image_b64, "angle": 90}),
        ("flip", {"image_b64": image_b64, "direction": "horizontal"}),
        ("blur", {"image_b64": image_b64, "radius": 2}),
        ("resize", {"image_b64": image_b64, "width": 20, "height": 10}),
        ("crop", {"image_b64": image_b64, "left": 0, "top": 0, "right": 20, "bottom": 10}),
        ("add_noise", {"image_b64": image_b64, "amount": 0.1, "salt_vs_pepper": 0.5}),
    ]

    for tool_name, arguments in tests:
        result_b64 = call_tool(module, tool_name, arguments)
        img = decode_image(result_b64)

        assert img.format == "PNG"
        assert img.width > 0
        assert img.height > 0


def test_resize_and_crop_return_expected_dimensions():
    module = load_app_module()
    image_b64 = make_image_b64(size=(80, 60))

    resized_b64 = call_tool(
        module,
        "resize",
        {"image_b64": image_b64, "width": 30, "height": 20},
    )
    cropped_b64 = call_tool(
        module,
        "crop",
        {"image_b64": image_b64, "left": 10, "top": 10, "right": 50, "bottom": 40},
    )

    assert decode_image(resized_b64).size == (30, 20)
    assert decode_image(cropped_b64).size == (40, 30)


def test_http_bridge_calls_registered_mcp_tool():
    module = load_app_module()
    image_b64 = make_image_b64()

    class FakeRequest:
        async def json(self):
            return {
                "name": "blur",
                "arguments": {"image_b64": image_b64, "radius": 2},
            }

    response = anyio.run(module.call_tool, FakeRequest())

    assert response.status_code == 200

    body = json.loads(response.body)
    img = decode_image(body["result"])
    assert img.format == "PNG"
