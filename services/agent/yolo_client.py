import json

import httpx
from langchain_core.tools import tool

import config
from context import current_detection_objects, current_image_s3_key


async def request_yolo_prediction(image_s3_key: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{config.YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": image_s3_key},
        )
        response.raise_for_status()
    return response.json()


async def get_yolo_prediction_details(prediction_uid: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{config.YOLO_SERVICE_URL}/prediction/{prediction_uid}"
        )
        response.raise_for_status()
    return response.json()


@tool
async def detect_objects() -> str:
    """Run YOLO and return labels, scores, and boxes for the active image."""
    image_s3_key = current_image_s3_key.get()
    if not image_s3_key:
        return json.dumps({"error": "No image was provided by the user."})

    prediction = await request_yolo_prediction(image_s3_key)
    prediction_details = await get_yolo_prediction_details(
        prediction["prediction_uid"]
    )
    detection_objects = prediction_details.get("detection_objects", [])
    current_detection_objects.set(detection_objects)

    result = dict(prediction)
    result["detection_objects"] = detection_objects
    return json.dumps(result)
