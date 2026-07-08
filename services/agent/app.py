import base64
import binascii
import io
import json
import logging
import math
import os
import time
import uuid
from contextvars import ContextVar
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
IMG_PROC_MCP_URL = os.environ.get("IMG_PROC_MCP_URL", "http://localhost:8090")
AWS_REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
BEDROCK_MODEL_PREFIX = "bedrock/"
DEFAULT_MODEL = f"{BEDROCK_MODEL_PREFIX}openai.gpt-oss-20b-1:0"
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
BEDROCK_MODEL_ID = MODEL.removeprefix(BEDROCK_MODEL_PREFIX)


LLM_REQUESTS_PER_SECOND = 0.25  # 15 request per minute
LLM_RATE_LIMIT_CHECK_SECONDS = 0.1
LLM_RATE_LIMIT_BUCKET_SIZE = 1
LLM_RATE_LIMIT_SECONDS = 1 / LLM_REQUESTS_PER_SECOND

# Bedrock text-only models allowed for the course.
ALLOWED_BEDROCK_MODEL_IDS = {
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "openai.gpt-oss-20b-1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
}

if BEDROCK_MODEL_ID not in ALLOWED_BEDROCK_MODEL_IDS:
    allowed_list = "\n  ".join(
        f"{BEDROCK_MODEL_PREFIX}{model_id}"
        for model_id in sorted(ALLOWED_BEDROCK_MODEL_IDS)
    )
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported Bedrock models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
    "When the user asks for one image transform, call process_image. "
    "When the user asks for two or more image transforms in one request, call "
    "process_image_edits once with every edit in order. "
    "Supported image operations are rotate, flip, blur, resize, crop, and add_noise. "
    "Use add_noise for salt-and-pepper noise requests. "
    "Do not include markdown image placeholders; the frontend displays returned images separately. "
    "For object-specific requests like 'the second dog from the right', pass "
    "target='object', the object label, ordinal number, and from_side. "
    "Never ask the model to inspect image bytes directly; use tools instead. "
)

REQUIRED_MODEL_FEATURES = ["structured_output", "tool_calling"]
CONTEXT_LIMIT_WARNING_RATIO = 0.9

_current_image_s3_key: ContextVar[Optional[str]] = ContextVar("current_image_s3_key", default=None)
_chat_rate_limit_lock = Lock()
_next_chat_request_at = 0.0


def check_chat_rate_limit():
    """
    Return 429 immediately when the next LLM request slot is not ready yet.

    The LangChain rate limiter still protects the provider call itself. This
    check keeps users from waiting silently at the /chat API boundary.
    """
    global _next_chat_request_at

    now = time.monotonic()
    with _chat_rate_limit_lock:
        wait_seconds = _next_chat_request_at - now
        if wait_seconds > 0:
            retry_after = math.ceil(wait_seconds)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit reached. Please try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        _next_chat_request_at = now + LLM_RATE_LIMIT_SECONDS


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def read_image_type(image_bytes: bytes):
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"

    raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")


def upload_image_base64_to_s3(image_b64: str, chat_id: str) -> str:
    if not AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(
            status_code=500,
            detail="Image upload is currently unavailable",
        )

    try:
        image_bytes = base64.b64decode(image_b64.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image data")

    ext, content_type = read_image_type(image_bytes)
    image_id = str(uuid.uuid4())
    image_s3_key = f"chats/{chat_id}/{image_id}/original/image{ext}"

    try:
        get_s3_client().put_object(
            Bucket=AWS_S3_BUCKET,
            Key=image_s3_key,
            Body=image_bytes,
            ContentType=content_type,
        )
    except (BotoCoreError, ClientError):
        logging.exception("Could not upload image to S3")
        raise HTTPException(
            status_code=502,
            detail="Could not upload image. Please try again later.",
        )

    return image_s3_key


def read_s3_image_bytes(image_s3_key: str) -> bytes:
    if not AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(status_code=500, detail="Image read is currently unavailable")

    try:
        response = get_s3_client().get_object(
            Bucket=AWS_S3_BUCKET,
            Key=image_s3_key,
        )
        return response["Body"].read()
    except (BotoCoreError, ClientError):
        logging.exception("Could not read image from S3: %s", image_s3_key)
        raise HTTPException(status_code=502, detail="Could not read image from S3")


def upload_image_bytes_to_s3(
    image_bytes: bytes,
    image_s3_key: str,
    content_type: str,
) -> None:
    if not AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(status_code=500, detail="Image upload is currently unavailable")

    try:
        get_s3_client().put_object(
            Bucket=AWS_S3_BUCKET,
            Key=image_s3_key,
            Body=image_bytes,
            ContentType=content_type,
        )
    except (BotoCoreError, ClientError):
        logging.exception("Could not upload processed image to S3")
        raise HTTPException(
            status_code=502,
            detail="Could not upload processed image to S3",
        )


def build_processed_image_s3_key(original_image_s3_key: str, operation: str) -> str:
    safe_operation = operation.replace(" ", "-").replace("_", "-")

    if "/original/" in original_image_s3_key:
        image_prefix = original_image_s3_key.split("/original/", 1)[0]
        return f"{image_prefix}/processed/{uuid.uuid4()}/{safe_operation}.png"

    return f"processed/{uuid.uuid4()}/{safe_operation}.png"


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def image_to_png_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return image_bytes_to_base64(buffer.getvalue())


def base64_to_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def image_to_png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def request_yolo_prediction(image_s3_key: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": image_s3_key},
        )
        response.raise_for_status()

    return response.json()


def get_yolo_prediction_details(prediction_uid: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{YOLO_SERVICE_URL}/prediction/{prediction_uid}")
        response.raise_for_status()

    return response.json()


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_s3_key = _current_image_s3_key.get()
    if not image_s3_key:
        return json.dumps({"error": "No image was provided by the user."})

    return json.dumps(request_yolo_prediction(image_s3_key))


def normalize_operation(operation: str) -> str:
    normalized = operation.strip().lower().replace("-", "_")
    operation_aliases = {
        "noise": "add_noise",
        "salt_pepper_noise": "add_noise",
        "salt_and_pepper_noise": "add_noise",
    }

    normalized = operation_aliases.get(normalized, normalized)
    allowed_operations = {"rotate", "flip", "blur", "resize", "crop", "add_noise"}

    if normalized not in allowed_operations:
        raise ValueError(
            "operation must be one of: rotate, flip, blur, resize, crop, add_noise"
        )

    return normalized


def call_img_proc_mcp_tool(tool_name: str, arguments: dict) -> str:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{IMG_PROC_MCP_URL.rstrip('/')}/tools/call",
            json={"name": tool_name, "arguments": arguments},
        )
        response.raise_for_status()

    data = response.json()
    if data.get("error"):
        raise ValueError(data["error"])

    result = data.get("result")
    if not isinstance(result, str):
        raise ValueError("Image processing MCP server returned an invalid result")

    return result


def build_img_proc_arguments(
    operation: str,
    image_b64: str,
    target: str,
    angle: float,
    expand: bool,
    direction: str,
    radius: float,
    width: Optional[int],
    height: Optional[int],
    left: Optional[int],
    top: Optional[int],
    right: Optional[int],
    bottom: Optional[int],
    amount: float,
    salt_vs_pepper: float,
) -> dict:
    arguments = {"image_b64": image_b64}

    if operation == "rotate":
        arguments["angle"] = angle
        arguments["expand"] = expand if target != "object" else False
    elif operation == "flip":
        arguments["direction"] = direction
    elif operation == "blur":
        arguments["radius"] = radius
    elif operation == "resize":
        if width is None or height is None:
            raise ValueError("resize requires width and height")
        arguments["width"] = width
        arguments["height"] = height
    elif operation == "crop":
        if left is None or top is None or right is None or bottom is None:
            raise ValueError("crop requires left, top, right, and bottom")
        arguments["left"] = left
        arguments["top"] = top
        arguments["right"] = right
        arguments["bottom"] = bottom
    elif operation == "add_noise":
        arguments["amount"] = amount
        arguments["salt_vs_pepper"] = salt_vs_pepper

    return arguments


def call_image_operation(
    operation: str,
    image_b64: str,
    target: str,
    angle: float,
    expand: bool,
    direction: str,
    radius: float,
    width: Optional[int],
    height: Optional[int],
    left: Optional[int],
    top: Optional[int],
    right: Optional[int],
    bottom: Optional[int],
    amount: float,
    salt_vs_pepper: float,
) -> str:
    arguments = build_img_proc_arguments(
        operation,
        image_b64,
        target,
        angle,
        expand,
        direction,
        radius,
        width,
        height,
        left,
        top,
        right,
        bottom,
        amount,
        salt_vs_pepper,
    )
    return call_img_proc_mcp_tool(operation, arguments)


def read_edit_value(edit: dict, key: str, default):
    value = edit.get(key, default)
    if value is None:
        return default
    return value


def read_optional_int_edit_value(edit: dict, key: str) -> Optional[int]:
    value = read_edit_value(edit, key, None)
    if value is None:
        return None
    return int(value)


def read_bool_edit_value(edit: dict, key: str, default: bool) -> bool:
    value = read_edit_value(edit, key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ["true", "1", "yes"]:
            return True
        if value_lower in ["false", "0", "no"]:
            return False

    return bool(value)


def normalize_image_target(target: str) -> str:
    normalized = target.strip().lower().replace("-", "_")

    if normalized in ["entire_image", "image", "all"]:
        return "entire_image"

    if normalized in ["object", "region"]:
        return "object"

    raise ValueError("target must be 'entire_image' or 'object'")


def normalize_image_edit(edit: dict, edit_number: int) -> dict:
    if not isinstance(edit, dict):
        raise ValueError(f"edit {edit_number} must be an object")

    raw_operation = read_edit_value(edit, "operation", "")
    if not isinstance(raw_operation, str) or not raw_operation.strip():
        raise ValueError(f"edit {edit_number} requires an operation")

    raw_label = read_edit_value(edit, "label", None)
    label = str(raw_label) if raw_label is not None else None
    target = normalize_image_target(
        str(read_edit_value(edit, "target", "entire_image"))
    )

    if target == "object" and not label:
        raise ValueError(f"edit {edit_number} object-specific edits require a label")

    return {
        "operation": normalize_operation(raw_operation),
        "target": target,
        "label": label,
        "ordinal": int(read_edit_value(edit, "ordinal", 1)),
        "from_side": str(read_edit_value(edit, "from_side", "left")),
        "angle": float(read_edit_value(edit, "angle", 90.0)),
        "expand": read_bool_edit_value(edit, "expand", True),
        "direction": str(read_edit_value(edit, "direction", "horizontal")),
        "radius": float(read_edit_value(edit, "radius", 2.0)),
        "width": read_optional_int_edit_value(edit, "width"),
        "height": read_optional_int_edit_value(edit, "height"),
        "left": read_optional_int_edit_value(edit, "left"),
        "top": read_optional_int_edit_value(edit, "top"),
        "right": read_optional_int_edit_value(edit, "right"),
        "bottom": read_optional_int_edit_value(edit, "bottom"),
        "amount": float(read_edit_value(edit, "amount", 0.02)),
        "salt_vs_pepper": float(read_edit_value(edit, "salt_vs_pepper", 0.5)),
    }


def edit_changes_image_geometry(edit: dict) -> bool:
    if edit["target"] == "object":
        return edit["operation"] == "crop"

    return edit["operation"] in ["rotate", "flip", "resize", "crop"]


def validate_multi_edit_order(edits: list[dict]) -> None:
    original_boxes_still_match = True

    for edit in edits:
        if edit["target"] == "object" and not original_boxes_still_match:
            raise ValueError(
                "object edits must come before whole-image geometry edits"
            )

        if edit_changes_image_geometry(edit):
            original_boxes_still_match = False


def apply_entire_image_edit_to_image(img: Image.Image, edit: dict) -> Image.Image:
    image_b64 = image_to_png_base64(img)
    processed_image_b64 = call_image_operation(
        edit["operation"],
        image_b64,
        "entire_image",
        edit["angle"],
        edit["expand"],
        edit["direction"],
        edit["radius"],
        edit["width"],
        edit["height"],
        edit["left"],
        edit["top"],
        edit["right"],
        edit["bottom"],
        edit["amount"],
        edit["salt_vs_pepper"],
    )
    return base64_to_image(processed_image_b64).convert("RGBA")


def apply_object_edit_to_image(
    img: Image.Image,
    selected_object: dict,
    edit: dict,
) -> Image.Image:
    full_img = img.convert("RGBA")
    box = selected_object["box"]

    if edit["operation"] == "crop":
        return full_img.crop(box)

    crop_img = full_img.crop(box)
    crop_b64 = image_to_png_base64(crop_img)

    processed_crop_b64 = call_image_operation(
        edit["operation"],
        crop_b64,
        "object",
        edit["angle"],
        False,
        edit["direction"],
        edit["radius"],
        edit["width"],
        edit["height"],
        None,
        None,
        None,
        None,
        edit["amount"],
        edit["salt_vs_pepper"],
    )

    processed_crop = base64_to_image(processed_crop_b64).convert("RGBA")

    if processed_crop.size != crop_img.size:
        processed_crop = processed_crop.resize(crop_img.size)

    full_img.paste(processed_crop, box)
    return full_img


def parse_detection_box(raw_box) -> tuple[int, int, int, int]:
    if isinstance(raw_box, str):
        values = json.loads(raw_box)
    else:
        values = raw_box

    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("YOLO returned an invalid bounding box")

    left, top, right, bottom = values
    return (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )


def clamp_box_to_image(
    box: tuple[int, int, int, int],
    img: Image.Image,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(left, img.width - 1))
    top = max(0, min(top, img.height - 1))
    right = max(left + 1, min(right, img.width))
    bottom = max(top + 1, min(bottom, img.height))
    return left, top, right, bottom


def detection_center(detection: dict) -> tuple[float, float]:
    left, top, right, bottom = detection["box"]
    return ((left + right) / 2, (top + bottom) / 2)


def select_detection_object(
    detection_objects: list[dict],
    label: str,
    ordinal: int,
    from_side: str,
    img: Image.Image,
) -> dict:
    if ordinal < 1:
        raise ValueError("ordinal must be 1 or greater")

    label_lower = label.lower()
    matches = []

    for detection in detection_objects:
        if detection.get("label", "").lower() != label_lower:
            continue

        detection_copy = dict(detection)
        detection_copy["box"] = clamp_box_to_image(
            parse_detection_box(detection["box"]),
            img,
        )
        matches.append(detection_copy)

    if not matches:
        raise ValueError(f"No detected object matched label '{label}'")

    from_side = from_side.lower().replace("-", "_")
    if from_side == "right":
        matches.sort(key=lambda item: detection_center(item)[0], reverse=True)
    elif from_side == "left":
        matches.sort(key=lambda item: detection_center(item)[0])
    elif from_side == "bottom":
        matches.sort(key=lambda item: detection_center(item)[1], reverse=True)
    elif from_side == "top":
        matches.sort(key=lambda item: detection_center(item)[1])
    else:
        raise ValueError("from_side must be left, right, top, or bottom")

    if ordinal > len(matches):
        raise ValueError(
            f"Only {len(matches)} detected object(s) matched label '{label}'"
        )

    return matches[ordinal - 1]


def process_entire_image(
    original_image_bytes: bytes,
    operation: str,
    angle: float,
    expand: bool,
    direction: str,
    radius: float,
    width: Optional[int],
    height: Optional[int],
    left: Optional[int],
    top: Optional[int],
    right: Optional[int],
    bottom: Optional[int],
    amount: float,
    salt_vs_pepper: float,
) -> bytes:
    img = Image.open(io.BytesIO(original_image_bytes))
    edit = {
        "operation": operation,
        "angle": angle,
        "expand": expand,
        "direction": direction,
        "radius": radius,
        "width": width,
        "height": height,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "amount": amount,
        "salt_vs_pepper": salt_vs_pepper,
    }
    return image_to_png_bytes(apply_entire_image_edit_to_image(img, edit))


def process_object_region(
    original_img: Image.Image,
    selected_object: dict,
    operation: str,
    angle: float,
    direction: str,
    radius: float,
    width: Optional[int],
    height: Optional[int],
    amount: float,
    salt_vs_pepper: float,
) -> bytes:
    edit = {
        "operation": operation,
        "angle": angle,
        "direction": direction,
        "radius": radius,
        "width": width,
        "height": height,
        "amount": amount,
        "salt_vs_pepper": salt_vs_pepper,
    }
    return image_to_png_bytes(
        apply_object_edit_to_image(original_img, selected_object, edit)
    )


@tool
def process_image(
    operation: str,
    target: str = "entire_image",
    label: Optional[str] = None,
    ordinal: int = 1,
    from_side: str = "left",
    angle: float = 90.0,
    expand: bool = True,
    direction: str = "horizontal",
    radius: float = 2.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    left: Optional[int] = None,
    top: Optional[int] = None,
    right: Optional[int] = None,
    bottom: Optional[int] = None,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
) -> str:
    """
    Transform the uploaded image.

    Use target='entire_image' for full-image edits. Use target='object' for
    object-specific edits after extracting a YOLO bounding box. For object
    targets, provide label, ordinal, and from_side. Example: the second dog
    from the right means label='dog', ordinal=2, from_side='right'.
    """
    image_s3_key = _current_image_s3_key.get()
    if not image_s3_key:
        return json.dumps({"error": "No image was provided by the user."})

    try:
        operation = normalize_operation(operation)
        target = target.strip().lower()
        original_image_bytes = read_s3_image_bytes(image_s3_key)
        original_img = Image.open(io.BytesIO(original_image_bytes))
        prediction_uid = None
        selected_object = None

        if target in ["entire_image", "image", "all"]:
            processed_image_bytes = process_entire_image(
                original_image_bytes,
                operation,
                angle,
                expand,
                direction,
                radius,
                width,
                height,
                left,
                top,
                right,
                bottom,
                amount,
                salt_vs_pepper,
            )
        elif target in ["object", "region"]:
            if not label:
                raise ValueError("object-specific edits require a label")

            prediction = request_yolo_prediction(image_s3_key)
            prediction_uid = prediction["prediction_uid"]
            prediction_details = get_yolo_prediction_details(prediction_uid)
            selected_object = select_detection_object(
                prediction_details["detection_objects"],
                label,
                ordinal,
                from_side,
                original_img,
            )

            if operation == "crop":
                processed_image_bytes = image_to_png_bytes(
                    original_img.convert("RGBA").crop(selected_object["box"])
                )
            else:
                processed_image_bytes = process_object_region(
                    original_img,
                    selected_object,
                    operation,
                    angle,
                    direction,
                    radius,
                    width,
                    height,
                    amount,
                    salt_vs_pepper,
                )
        else:
            raise ValueError("target must be 'entire_image' or 'object'")

        processed_s3_key = build_processed_image_s3_key(image_s3_key, operation)
        upload_image_bytes_to_s3(
            processed_image_bytes,
            processed_s3_key,
            "image/png",
        )

        return json.dumps(
            {
                "operation": operation,
                "target": target,
                "prediction_uid": prediction_uid,
                "processed_image_s3_key": processed_s3_key,
                "selected_object": selected_object,
                "message": "Image processing completed.",
            }
        )
    except Exception as exc:
        logging.exception("Could not process image")
        return json.dumps({"error": str(exc)})


@tool
def process_image_edits(edits: list[dict]) -> str:
    """
    Transform the uploaded image with multiple edits and return one final image.

    Use this when the user asks for two or more image operations in one prompt.
    Each edit uses the same fields as process_image: operation, target, label,
    ordinal, from_side, angle, expand, direction, radius, width, height, crop
    coordinates, amount, and salt_vs_pepper.

    Example:
    edits=[
        {"operation": "flip", "target": "object", "label": "person", "ordinal": 1},
        {"operation": "blur", "target": "object", "label": "person", "from_side": "right"}
    ]
    """
    image_s3_key = _current_image_s3_key.get()
    if not image_s3_key:
        return json.dumps({"error": "No image was provided by the user."})

    try:
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits must be a non-empty list")

        normalized_edits = []
        for index, edit in enumerate(edits, start=1):
            normalized_edits.append(normalize_image_edit(edit, index))

        validate_multi_edit_order(normalized_edits)

        original_image_bytes = read_s3_image_bytes(image_s3_key)
        original_img = Image.open(io.BytesIO(original_image_bytes)).convert("RGBA")
        working_img = original_img.copy()
        prediction_uid = None
        detection_objects = []

        has_object_edit = False
        for edit in normalized_edits:
            if edit["target"] == "object":
                has_object_edit = True

        if has_object_edit:
            prediction = request_yolo_prediction(image_s3_key)
            prediction_uid = prediction["prediction_uid"]
            prediction_details = get_yolo_prediction_details(prediction_uid)
            detection_objects = prediction_details["detection_objects"]

        edit_results = []

        for edit in normalized_edits:
            edit_result = {
                "operation": edit["operation"],
                "target": edit["target"],
            }

            if edit["target"] == "entire_image":
                working_img = apply_entire_image_edit_to_image(working_img, edit)
            else:
                selected_object = select_detection_object(
                    detection_objects,
                    edit["label"],
                    edit["ordinal"],
                    edit["from_side"],
                    original_img,
                )
                working_img = apply_object_edit_to_image(
                    working_img,
                    selected_object,
                    edit,
                )
                edit_result["selected_object"] = selected_object

            edit_results.append(edit_result)

        processed_s3_key = build_processed_image_s3_key(image_s3_key, "multi_edit")
        upload_image_bytes_to_s3(
            image_to_png_bytes(working_img),
            processed_s3_key,
            "image/png",
        )

        return json.dumps(
            {
                "operation": "multi_edit",
                "edit_count": len(normalized_edits),
                "prediction_uid": prediction_uid,
                "processed_image_s3_key": processed_s3_key,
                "edits": edit_results,
                "message": "Image processing completed.",
            }
        )
    except Exception as exc:
        logging.exception("Could not process image edits")
        return json.dumps({"error": str(exc)})


def fetch_s3_image_b64(image_s3_key: str) -> Optional[str]:
    """
    Fetch an image from S3 and encode it for the API response.

    This value is not added to the LangChain messages, so the LLM still only
    receives the text JSON returned by the detection tool.
    """
    if not AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        return None

    try:
        response = get_s3_client().get_object(
            Bucket=AWS_S3_BUCKET,
            Key=image_s3_key,
        )
        image_bytes = response["Body"].read()
    except (BotoCoreError, ClientError):
        logging.exception("Could not fetch image from S3: %s", image_s3_key)
        return None

    return base64.b64encode(image_bytes).decode("utf-8")


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects,
    process_image.name: process_image,
    process_image_edits.name: process_image_edits,
}


def validate_model_profile(model_name: Optional[str], profile: dict):
    """
    Stop the app early if the selected model cannot support this agent.
    """
    missing_features = []

    for feature in REQUIRED_MODEL_FEATURES:
        if profile.get(feature) is not True:
            missing_features.append(feature)

    max_input_tokens = profile.get("max_input_tokens")
    has_token_limit = isinstance(max_input_tokens, int) and max_input_tokens > 0

    if not missing_features and has_token_limit:
        return

    error_lines = [
        f"\n[ERROR] MODEL='{model_name}' does not have the required profile support."
    ]

    if missing_features:
        error_lines.append(
            "Missing or unsupported features: " + ", ".join(missing_features)
        )

    if not has_token_limit:
        error_lines.append("Missing or invalid profile value: max_input_tokens")

    error_lines.append("Model profile:")
    error_lines.append(json.dumps(profile, indent=2, sort_keys=True))

    raise SystemExit("\n".join(error_lines))


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


def read_token_usage(response: AIMessage) -> TokenUsage:
    """
    Convert LangChain usage metadata into the API response shape.
    """
    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens")

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        total=total_tokens,
    )


def add_token_usage(current: TokenUsage, latest: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input=current.input + latest.input,
        output=current.output + latest.output,
        total=current.total + latest.total,
    )


def read_response_text(response: AIMessage) -> str:
    """
    Convert LangChain message content into the plain text API response.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    text_parts.append(block_text)

        return "\n".join(text_parts)

    return str(content)


def is_near_context_limit(input_tokens: int, profile: dict) -> bool:
    max_input_tokens = profile["max_input_tokens"]
    warning_threshold = max_input_tokens * CONTEXT_LIMIT_WARNING_RATIO
    return input_tokens >= warning_threshold


llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=LLM_REQUESTS_PER_SECOND,
    check_every_n_seconds=LLM_RATE_LIMIT_CHECK_SECONDS,
    max_bucket_size=LLM_RATE_LIMIT_BUCKET_SIZE,
)

llm = init_chat_model(
    BEDROCK_MODEL_ID,
    model_provider="bedrock_converse",
    temperature=0,
    region_name=AWS_REGION,
    rate_limiter=llm_rate_limiter,
)

MODEL_PROFILE = getattr(llm, "profile", None) or {}
validate_model_profile(BEDROCK_MODEL_ID, MODEL_PROFILE)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


class AgentRunResult(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


def run_agent(history: list, max_iterations: int = 10) -> AgentRunResult:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    tools_called = []
    prediction_id = None
    annotated_image = None
    tokens_used = TokenUsage()
    context_limit_exceeded = False

    for iteration in range(1, max_iterations + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        latest_tokens = read_token_usage(response)
        tokens_used = add_token_usage(tokens_used, latest_tokens)

        if latest_tokens.input and is_near_context_limit(
            latest_tokens.input,
            MODEL_PROFILE,
        ):
            context_limit_exceeded = True

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            return AgentRunResult(
                response=read_response_text(response),
                prediction_id=prediction_id,
                annotated_image=annotated_image,
                tokens_used=tokens_used,
                iterations=iteration,
                tools_called=tools_called,
                context_limit_exceeded=context_limit_exceeded,
            )

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tools_called.append(tool_call["name"])
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = tool_fn.invoke(tool_call)          # returns a ToolMessage
            messages.append(tool_result)

            try:
                tool_data = json.loads(tool_result.content)
                if tool_data.get("prediction_uid"):
                    prediction_id = tool_data["prediction_uid"]
                if tool_data.get("predicted_image_s3_key"):
                    annotated_image = fetch_s3_image_b64(
                        tool_data["predicted_image_s3_key"]
                    )
                if tool_data.get("processed_image_s3_key"):
                    annotated_image = fetch_s3_image_b64(
                        tool_data["processed_image_s3_key"]
                    )
            except json.JSONDecodeError:
                pass

    return AgentRunResult(
        response="I reached the maximum number of tool calls and could not finish safely.",
        prediction_id=prediction_id,
        annotated_image=annotated_image,
        tokens_used=tokens_used,
        iterations=max_iterations,
        tools_called=tools_called,
        context_limit_exceeded=True,
    )



app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://dev.weam.fursa.click:3000",
        "http://prod.weam.fursa.click:3000",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    check_chat_rate_limit()

    lc_messages = []
    latest_image_b64 = None
    chat_id = str(uuid.uuid4())

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image_b64 = msg.image_base64
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    latest_image_s3_key = None
    if latest_image_b64:
        latest_image_s3_key = upload_image_base64_to_s3(latest_image_b64, chat_id)

    token = _current_image_s3_key.set(latest_image_s3_key)
    try:
        start_time = time.time()
        agent_result = run_agent(lc_messages)
        agent_loop_time_s = round(time.time() - start_time, 2)
        return ChatResponse(
            response=agent_result.response,
            prediction_id=agent_result.prediction_id,
            annotated_image=agent_result.annotated_image,
            tokens_used=agent_result.tokens_used,
            agent_loop_time_s=agent_loop_time_s,
            iterations=agent_result.iterations,
            tools_called=agent_result.tools_called,
            context_limit_exceeded=agent_result.context_limit_exceeded,
        )
    finally:
        _current_image_s3_key.reset(token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
