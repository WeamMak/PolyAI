import base64
import io

from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageFilter


mcp = FastMCP("img-proc")


def _decode(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def _encode(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@mcp.tool()
def blur(image_b64: str, radius: float = 2.0) -> str:
    """Apply Gaussian blur to an image. Returns base64-encoded PNG."""
    img = _decode(image_b64)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    return _encode(blurred_img)


if __name__ == "__main__":
    mcp.run()
