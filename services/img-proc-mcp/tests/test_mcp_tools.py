import base64
import importlib.util
import io
from pathlib import Path

import anyio
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TESTS_DIR / "outputs"


def load_mcp_server():
    app_file = PROJECT_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("img_proc_mcp_app", app_file)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MCP server from {app_file}")

    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    return app_module.mcp


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_file(image_b64: str, filename: Path) -> None:
    image_bytes = base64.b64decode(image_b64)
    filename.write_bytes(image_bytes)


def create_sample_image() -> Image.Image:
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((20, 20, 90, 90), fill="red")
    draw.rectangle((110, 20, 180, 90), fill="green")
    draw.rectangle((20, 110, 90, 180), fill="blue")
    draw.rectangle((110, 110, 180, 180), fill="yellow")

    return img


def find_test_image() -> Path | None:
    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    for file_path in sorted(TESTS_DIR.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            return file_path

    return None


def load_test_image() -> Image.Image:
    test_image_path = find_test_image()

    if test_image_path is None:
        print("No image found in tests/. Using generated sample image.")
        return create_sample_image()

    print(f"Using test image: {test_image_path}")
    return Image.open(test_image_path)


def get_crop_arguments(image_b64: str, img: Image.Image) -> dict:
    crop_right = max(1, img.width // 2)
    crop_bottom = max(1, img.height // 2)

    return {
        "image_b64": image_b64,
        "left": 0,
        "top": 0,
        "right": crop_right,
        "bottom": crop_bottom,
    }


async def main() -> None:
    mcp = load_mcp_server()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = load_test_image()
    image_b64 = image_to_base64(img)
    base64_to_file(image_b64, OUTPUT_DIR / "original.png")

    tools = await mcp.list_tools()
    print("Available tools:", [tool.name for tool in tools])

    tests = [
        ("blur", {"image_b64": image_b64, "radius": 5}),
        ("rotate", {"image_b64": image_b64, "angle": 45, "expand": True}),
        ("flip", {"image_b64": image_b64, "direction": "horizontal"}),
        ("resize", {"image_b64": image_b64, "width": 100, "height": 100}),
        ("crop", get_crop_arguments(image_b64, img)),
        ("add_noise", {"image_b64": image_b64, "amount": 0.1, "salt_vs_pepper": 0.5}),
    ]

    for tool_name, arguments in tests:
        result = await mcp.call_tool(tool_name, arguments)
        image_b64_result = result[1]["result"]
        output_file = OUTPUT_DIR / f"{tool_name}.png"

        base64_to_file(image_b64_result, output_file)
        print(f"Saved {output_file}")


anyio.run(main)
