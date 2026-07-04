import base64
import io
import os
import random

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from PIL import Image, ImageFilter
from starlette.requests import Request
from starlette.responses import JSONResponse


MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8090"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")

mcp = FastMCP(
    "img-proc",
    host=MCP_HOST,
    port=MCP_PORT,
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "img-proc-mcp:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://img-proc-mcp:*",
        ],
    ),
)


def _decode(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def _encode(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _validate_positive_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive numbers")


def _validate_crop_box(
    img: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    if left < 0 or top < 0:
        raise ValueError("left and top must be 0 or greater")

    if right > img.width or bottom > img.height:
        raise ValueError("crop box must stay inside the image")

    if right <= left or bottom <= top:
        raise ValueError("right must be greater than left, and bottom must be greater than top")


def _noise_pixel(pixel, use_salt: bool):
    if isinstance(pixel, int):
        return 255 if use_salt else 0

    if len(pixel) == 4:
        alpha = pixel[3]
        return (255, 255, 255, alpha) if use_salt else (0, 0, 0, alpha)

    if len(pixel) == 2:
        alpha = pixel[1]
        return (255, alpha) if use_salt else (0, alpha)

    return tuple(255 if use_salt else 0 for _ in pixel)


@mcp.tool()
def rotate(image_b64: str, angle: float, expand: bool = True) -> str:
    """Rotate an image by an angle in degrees. Returns base64-encoded PNG."""
    img = _decode(image_b64)
    rotated_img = img.rotate(angle, expand=expand)
    return _encode(rotated_img)


@mcp.tool()
def flip(image_b64: str, direction: str = "horizontal") -> str:
    """Flip an image horizontally or vertically. Returns base64-encoded PNG."""
    img = _decode(image_b64)

    if direction == "horizontal":
        flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        flipped_img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    return _encode(flipped_img)


@mcp.tool()
def blur(image_b64: str, radius: float = 2.0) -> str:
    """Apply Gaussian blur to an image. Returns base64-encoded PNG."""
    img = _decode(image_b64)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return _encode(blurred_img)


@mcp.tool()
def resize(image_b64: str, width: int, height: int) -> str:
    """Resize an image to the given width and height. Returns base64-encoded PNG."""
    _validate_positive_size(width, height)

    img = _decode(image_b64)
    resized_img = img.resize((width, height))
    return _encode(resized_img)


@mcp.tool()
def crop(image_b64: str, left: int, top: int, right: int, bottom: int) -> str:
    """Crop an image using left, top, right, and bottom coordinates. Returns base64-encoded PNG."""
    img = _decode(image_b64)
    _validate_crop_box(img, left, top, right, bottom)

    cropped_img = img.crop((left, top, right, bottom))
    return _encode(cropped_img)


@mcp.tool()
def add_noise(image_b64: str, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> str:
    """Add salt-and-pepper noise to an image. Returns base64-encoded PNG."""
    if amount < 0 or amount > 1:
        raise ValueError("amount must be between 0 and 1")

    if salt_vs_pepper < 0 or salt_vs_pepper > 1:
        raise ValueError("salt_vs_pepper must be between 0 and 1")

    img = _decode(image_b64)
    noisy_img = img.copy()
    pixels = noisy_img.load()

    for y in range(noisy_img.height):
        for x in range(noisy_img.width):
            should_change_pixel = random.random() < amount

            if should_change_pixel:
                use_salt = random.random() < salt_vs_pepper
                pixels[x, y] = _noise_pixel(pixels[x, y], use_salt)

    return _encode(noisy_img)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/tools/call", methods=["POST"])
async def call_tool(request: Request) -> JSONResponse:
    """
    Small HTTP bridge for the agent service.

    The real MCP tools are still registered above. This route lets the FastAPI
    agent call those tools without adding the MCP Python SDK to the agent
    runtime, where it currently conflicts with the existing FastAPI version.
    """
    body = await request.json()
    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    if not isinstance(tool_name, str):
        return JSONResponse({"error": "name must be a string"}, status_code=400)

    if not isinstance(arguments, dict):
        return JSONResponse({"error": "arguments must be an object"}, status_code=400)

    try:
        result = await mcp.call_tool(tool_name, arguments)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"result": result[1]["result"]})


if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
