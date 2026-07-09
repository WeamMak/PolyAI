import base64
import importlib.util
import io
from pathlib import Path

import anyio
import pytest
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_app_module():
    app_file = PROJECT_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("img_proc_mcp_app_test", app_file)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {app_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def decode_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    return img


def call_tool(module, name, arguments):
    result = anyio.run(module.mcp.call_tool, name, arguments)
    return result[1]["result"]


def make_image(size=(40, 30), color="red") -> Image.Image:
    return Image.new("RGB", size, color)


def make_detection_image() -> tuple[Image.Image, list[dict]]:
    img = Image.new("RGB", (100, 20), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, 9, 19), fill="red")
    draw.rectangle((10, 0, 19, 19), fill="blue")
    draw.rectangle((40, 0, 49, 19), fill="green")
    draw.rectangle((50, 0, 59, 19), fill="yellow")
    draw.rectangle((80, 0, 89, 19), fill="black")
    draw.rectangle((90, 0, 99, 19), fill="magenta")

    detections = [
        {"id": 1, "label": "dog", "score": 0.9, "box": "[0, 0, 20, 20]"},
        {"id": 2, "label": "dog", "score": 0.8, "box": "[40, 0, 60, 20]"},
        {"id": 3, "label": "dog", "score": 0.7, "box": "[80, 0, 100, 20]"},
    ]
    return img, detections


def test_mcp_registers_exact_homework_tools():
    module = load_app_module()

    tools = anyio.run(module.mcp.list_tools)
    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "rotate",
        "flip",
        "blur",
        "resize",
        "crop",
        "add_noise",
    }


def test_all_tools_return_valid_png_images():
    module = load_app_module()
    image_b64 = image_to_base64(make_image())

    tests = [
        ("rotate", {"image_b64": image_b64, "angle": 90}),
        ("flip", {"image_b64": image_b64, "direction": "horizontal"}),
        ("blur", {"image_b64": image_b64, "radius": 2}),
        ("resize", {"image_b64": image_b64, "width": 20, "height": 10}),
        (
            "crop",
            {
                "image_b64": image_b64,
                "left": 0,
                "top": 0,
                "right": 20,
                "bottom": 10,
            },
        ),
        (
            "add_noise",
            {
                "image_b64": image_b64,
                "amount": 0.1,
                "salt_vs_pepper": 0.5,
            },
        ),
    ]

    for tool_name, arguments in tests:
        img = decode_image(call_tool(module, tool_name, arguments))
        assert img.format == "PNG"


def test_rotate_resize_and_crop_have_expected_dimensions():
    module = load_app_module()
    image_b64 = image_to_base64(make_image(size=(80, 60)))

    rotated = decode_image(
        call_tool(module, "rotate", {"image_b64": image_b64, "angle": 90})
    )
    resized = decode_image(
        call_tool(
            module,
            "resize",
            {"image_b64": image_b64, "width": 30, "height": 20},
        )
    )
    cropped = decode_image(
        call_tool(
            module,
            "crop",
            {
                "image_b64": image_b64,
                "left": 10,
                "top": 10,
                "right": 50,
                "bottom": 40,
            },
        )
    )

    assert rotated.size == (60, 80)
    assert resized.size == (30, 20)
    assert cropped.size == (40, 30)


def test_flip_and_noise_change_pixels_as_requested():
    module = load_app_module()
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 0), (0, 0, 255))
    image_b64 = image_to_base64(img)

    flipped = decode_image(
        call_tool(
            module,
            "flip",
            {"image_b64": image_b64, "direction": "horizontal"},
        )
    ).convert("RGB")
    peppered = decode_image(
        call_tool(
            module,
            "add_noise",
            {
                "image_b64": image_b64,
                "amount": 1.0,
                "salt_vs_pepper": 0.0,
            },
        )
    ).convert("RGB")

    assert flipped.getpixel((0, 0)) == (0, 0, 255)
    assert flipped.getpixel((1, 0)) == (255, 0, 0)
    assert [
        peppered.getpixel((0, 0)),
        peppered.getpixel((1, 0)),
    ] == [(0, 0, 0), (0, 0, 0)]


def test_object_tool_selects_second_dog_from_right_inside_mcp():
    module = load_app_module()
    img, detections = make_detection_image()

    result = decode_image(
        call_tool(
            module,
            "flip",
            {
                "image_b64": image_to_base64(img),
                "direction": "horizontal",
                "target": "object",
                "detection_objects": detections,
                "label": "dog",
                "ordinal": 2,
                "from_side": "right",
            },
        )
    ).convert("RGB")

    assert result.getpixel((5, 10)) == (255, 0, 0)
    assert result.getpixel((45, 10)) == (255, 255, 0)
    assert result.getpixel((55, 10)) == (0, 128, 0)
    assert result.getpixel((85, 10)) == (0, 0, 0)


def test_object_crop_and_resize_return_selected_region_dimensions():
    module = load_app_module()
    img, detections = make_detection_image()
    common_arguments = {
        "image_b64": image_to_base64(img),
        "target": "object",
        "detection_objects": detections,
        "label": "dog",
        "ordinal": 1,
        "from_side": "left",
    }

    cropped = decode_image(call_tool(module, "crop", common_arguments))
    resized = decode_image(
        call_tool(
            module,
            "resize",
            {**common_arguments, "width": 50, "height": 10},
        )
    )

    assert cropped.size == (20, 20)
    assert resized.size == (50, 10)


def test_object_target_requires_detection_results():
    module = load_app_module()
    image_b64 = image_to_base64(make_image())

    with pytest.raises(Exception, match="detection results"):
        call_tool(
            module,
            "blur",
            {
                "image_b64": image_b64,
                "target": "object",
                "label": "dog",
            },
        )
