import base64
import binascii
import io
import json
import logging
import os
import random
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from PIL import Image, ImageFilter, UnidentifiedImageError
from starlette.requests import Request
from starlette.responses import JSONResponse


MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8090"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
DEBUG_OBJECT_SELECTION = os.environ.get("DEBUG_OBJECT_SELECTION") == "1"

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
            "img-proc-mcp-svc:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://img-proc-mcp:*",
            "http://img-proc-mcp-svc:*",
        ],
    ),
)


def _decode(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_b64 must contain a valid base64-encoded image") from exc

    return img.convert("RGBA")


def _encode(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _normalize_target(target: str) -> str:
    normalized = target.strip().lower().replace("-", "_")

    if normalized in ["entire_image", "image", "all"]:
        return "entire_image"

    if normalized in ["object", "region"]:
        return "object"

    raise ValueError("target must be 'entire_image' or 'object'")


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
        raise ValueError(
            "right must be greater than left, and bottom must be greater than top"
        )


def _parse_detection_box(raw_box) -> tuple[int, int, int, int]:
    if isinstance(raw_box, str):
        try:
            values = json.loads(raw_box)
        except json.JSONDecodeError as exc:
            raise ValueError("YOLO returned an invalid bounding box") from exc
    else:
        values = raw_box

    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("YOLO returned an invalid bounding box")

    try:
        left, top, right, bottom = values
        return (
            int(round(float(left))),
            int(round(float(top))),
            int(round(float(right))),
            int(round(float(bottom))),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("YOLO returned an invalid bounding box") from exc


def _clamp_box_to_image(
    box: tuple[int, int, int, int],
    img: Image.Image,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(left, img.width - 1))
    top = max(0, min(top, img.height - 1))
    right = max(left + 1, min(right, img.width))
    bottom = max(top + 1, min(bottom, img.height))
    return left, top, right, bottom


def _detection_center(detection: dict) -> tuple[float, float]:
    left, top, right, bottom = detection["box"]
    return ((left + right) / 2, (top + bottom) / 2)


def _detection_debug_summary(index: int, detection: dict) -> dict:
    left, top, right, bottom = detection["box"]
    center_x, center_y = _detection_center(detection)

    return {
        "sorted_index": index,
        "label": detection.get("label"),
        "score": detection.get("score"),
        "box": [left, top, right, bottom],
        "center_x": center_x,
        "center_y": center_y,
        "area": (right - left) * (bottom - top),
    }


def _log_object_selection(
    label: str,
    ordinal: int,
    from_side: str,
    matches: list[dict],
    selected: dict,
) -> None:
    if not DEBUG_OBJECT_SELECTION:
        return

    candidates = [
        _detection_debug_summary(index, detection)
        for index, detection in enumerate(matches, start=1)
    ]
    selected_summary = _detection_debug_summary(ordinal, selected)

    logging.warning(
        "MCP object selection: label=%s ordinal=%s from_side=%s "
        "candidates=%s selected=%s",
        label,
        ordinal,
        from_side,
        json.dumps(candidates),
        json.dumps(selected_summary),
    )


def _select_detection_object(
    detection_objects: Optional[list[dict]],
    label: Optional[str],
    ordinal: int,
    from_side: str,
    img: Image.Image,
) -> dict:
    if not detection_objects:
        raise ValueError("object targets require YOLO detection results")

    if not label:
        raise ValueError("object targets require a label")

    if ordinal < 1:
        raise ValueError("ordinal must be 1 or greater")

    label_lower = label.strip().lower()
    matches = []

    for detection in detection_objects:
        if not isinstance(detection, dict):
            raise ValueError("YOLO detection results must contain objects")

        detection_label = detection.get("label")
        if not isinstance(detection_label, str):
            raise ValueError("YOLO detection result is missing a label")

        if detection_label.strip().lower() != label_lower:
            continue

        if "box" not in detection:
            raise ValueError("YOLO detection result is missing a bounding box")

        detection_copy = dict(detection)
        detection_copy["box"] = _clamp_box_to_image(
            _parse_detection_box(detection["box"]),
            img,
        )
        matches.append(detection_copy)

    if not matches:
        raise ValueError(f"No detected object matched label '{label}'")

    normalized_side = from_side.strip().lower().replace("-", "_")
    if normalized_side == "right":
        matches.sort(key=lambda item: _detection_center(item)[0], reverse=True)
    elif normalized_side == "left":
        matches.sort(key=lambda item: _detection_center(item)[0])
    elif normalized_side == "bottom":
        matches.sort(key=lambda item: _detection_center(item)[1], reverse=True)
    elif normalized_side == "top":
        matches.sort(key=lambda item: _detection_center(item)[1])
    else:
        raise ValueError("from_side must be left, right, top, or bottom")

    if ordinal > len(matches):
        raise ValueError(
            f"Only {len(matches)} detected object(s) matched label '{label}'"
        )

    selected = matches[ordinal - 1]
    _log_object_selection(
        label_lower,
        ordinal,
        normalized_side,
        matches,
        selected,
    )
    return selected


def _read_target(
    img: Image.Image,
    target: str,
    detection_objects: Optional[list[dict]],
    label: Optional[str],
    ordinal: int,
    from_side: str,
) -> tuple[str, Image.Image, Optional[tuple[int, int, int, int]]]:
    normalized_target = _normalize_target(target)

    if normalized_target == "entire_image":
        return normalized_target, img, None

    selected_object = _select_detection_object(
        detection_objects,
        label,
        ordinal,
        from_side,
        img,
    )
    box = selected_object["box"]
    return normalized_target, img.crop(box), box


def _composite_region(
    img: Image.Image,
    region: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = box
    expected_size = (right - left, bottom - top)

    if region.size != expected_size:
        raise ValueError("processed object region changed size unexpectedly")

    result = img.copy()
    result.alpha_composite(region.convert("RGBA"), dest=(left, top))
    return result


def _noise_pixel(pixel, use_salt: bool):
    alpha = pixel[3]
    color = 255 if use_salt else 0
    return (color, color, color, alpha)


def _add_noise_to_image(
    img: Image.Image,
    amount: float,
    salt_vs_pepper: float,
) -> Image.Image:
    if amount < 0 or amount > 1:
        raise ValueError("amount must be between 0 and 1")

    if salt_vs_pepper < 0 or salt_vs_pepper > 1:
        raise ValueError("salt_vs_pepper must be between 0 and 1")

    noisy_img = img.copy()
    pixels = noisy_img.load()

    for y in range(noisy_img.height):
        for x in range(noisy_img.width):
            if random.random() < amount:
                use_salt = random.random() < salt_vs_pepper
                pixels[x, y] = _noise_pixel(pixels[x, y], use_salt)

    return noisy_img


@mcp.tool()
def rotate(
    image_b64: str,
    angle: float = 90.0,
    expand: bool = True,
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Rotate the whole image or a YOLO-selected object. Returns a base64 PNG."""
    img = _decode(image_b64)
    normalized_target, target_img, box = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )

    if normalized_target == "entire_image":
        return _encode(target_img.rotate(angle, expand=expand))

    rotated_region = target_img.rotate(angle, expand=False)
    return _encode(_composite_region(img, rotated_region, box))


@mcp.tool()
def flip(
    image_b64: str,
    direction: str = "horizontal",
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Flip the whole image or a YOLO-selected object. Returns a base64 PNG."""
    img = _decode(image_b64)
    normalized_target, target_img, box = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )

    if direction == "horizontal":
        flipped_img = target_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        flipped_img = target_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    if normalized_target == "entire_image":
        return _encode(flipped_img)

    return _encode(_composite_region(img, flipped_img, box))


@mcp.tool()
def blur(
    image_b64: str,
    radius: float = 2.0,
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Blur the whole image or a YOLO-selected object. Returns a base64 PNG."""
    if radius < 0:
        raise ValueError("radius must be 0 or greater")

    img = _decode(image_b64)
    normalized_target, target_img, box = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )
    blurred_img = target_img.filter(ImageFilter.GaussianBlur(radius))

    if normalized_target == "entire_image":
        return _encode(blurred_img)

    return _encode(_composite_region(img, blurred_img, box))


@mcp.tool()
def resize(
    image_b64: str,
    width: int,
    height: int,
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Resize the whole image or a YOLO-selected object. Returns a base64 PNG."""
    _validate_positive_size(width, height)
    img = _decode(image_b64)
    _, target_img, _ = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )
    return _encode(target_img.resize((width, height)))


@mcp.tool()
def crop(
    image_b64: str,
    left: Optional[int] = None,
    top: Optional[int] = None,
    right: Optional[int] = None,
    bottom: Optional[int] = None,
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Crop the whole image or a YOLO-selected object. Returns a base64 PNG."""
    img = _decode(image_b64)
    normalized_target, target_img, _ = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )
    coordinates = [left, top, right, bottom]

    if normalized_target == "object" and all(value is None for value in coordinates):
        return _encode(target_img)

    if any(value is None for value in coordinates):
        raise ValueError("crop requires left, top, right, and bottom")

    _validate_crop_box(target_img, left, top, right, bottom)
    return _encode(target_img.crop((left, top, right, bottom)))


@mcp.tool()
def add_noise(
    image_b64: str,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
    target: str = "entire_image",
    detection_objects: Optional[list[dict]] = None,
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
) -> str:
    """Add salt-and-pepper noise to an image or object. Returns a base64 PNG."""
    img = _decode(image_b64)
    normalized_target, target_img, box = _read_target(
        img,
        target,
        detection_objects,
        label,
        ordinal,
        from_side,
    )
    noisy_img = _add_noise_to_image(target_img, amount, salt_vs_pepper)

    if normalized_target == "entire_image":
        return _encode(noisy_img)

    return _encode(_composite_region(img, noisy_img, box))


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
