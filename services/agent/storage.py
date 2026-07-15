import base64
import binascii
import logging
import uuid
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException

import config


def get_s3_client():
    return boto3.client("s3", region_name=config.AWS_REGION)


def read_image_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"

    raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")


def upload_image_base64_to_s3(image_b64: str, chat_id: str) -> str:
    if not config.AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(
            status_code=500,
            detail="Image upload is currently unavailable",
        )

    try:
        image_bytes = base64.b64decode(image_b64.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image data")

    extension, content_type = read_image_type(image_bytes)
    image_id = str(uuid.uuid4())
    image_s3_key = f"chats/{chat_id}/{image_id}/original/image{extension}"
    upload_bytes_to_s3(image_bytes, image_s3_key, content_type)
    return image_s3_key


def read_s3_bytes(s3_key: str) -> bytes:
    if not config.AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(status_code=500, detail="Image read is currently unavailable")

    try:
        response = get_s3_client().get_object(
            Bucket=config.AWS_S3_BUCKET,
            Key=s3_key,
        )
        return response["Body"].read()
    except (BotoCoreError, ClientError):
        logging.exception("Could not read S3 object: %s", s3_key)
        raise HTTPException(status_code=502, detail="Could not read image from S3")


def upload_bytes_to_s3(
    data: bytes,
    s3_key: str,
    content_type: str,
) -> None:
    if not config.AWS_S3_BUCKET:
        logging.error("AWS_S3_BUCKET is not configured")
        raise HTTPException(status_code=500, detail="Image upload is currently unavailable")

    try:
        get_s3_client().put_object(
            Bucket=config.AWS_S3_BUCKET,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
    except (BotoCoreError, ClientError):
        logging.exception("Could not upload S3 object: %s", s3_key)
        raise HTTPException(
            status_code=502,
            detail="Could not upload processed image to S3",
        )


def build_processed_image_s3_key(source_s3_key: str, tool_name: str) -> str:
    key_parts = source_s3_key.split("/")
    safe_tool_name = tool_name.replace("_", "-")

    if len(key_parts) >= 3 and key_parts[0] == "chats":
        image_prefix = "/".join(key_parts[:3])
        return f"{image_prefix}/processed/{uuid.uuid4()}/{safe_tool_name}.png"

    return f"processed/{uuid.uuid4()}/{safe_tool_name}.png"


def validate_active_image_key(chat_id: str, image_s3_key: str) -> str:
    expected_prefix = f"chats/{chat_id}/"
    if not image_s3_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=400,
            detail="The active image does not belong to this chat",
        )

    return image_s3_key


def fetch_s3_image(image_s3_key: str) -> tuple[Optional[str], Optional[str]]:
    try:
        image_bytes = read_s3_bytes(image_s3_key)
        _, content_type = read_image_type(image_bytes)
    except HTTPException:
        logging.exception("Could not fetch image from S3: %s", image_s3_key)
        return None, None

    return base64.b64encode(image_bytes).decode("utf-8"), content_type
