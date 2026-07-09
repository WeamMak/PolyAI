from contextvars import ContextVar
from typing import Optional


current_image_s3_key: ContextVar[Optional[str]] = ContextVar(
    "current_image_s3_key",
    default=None,
)
current_detection_objects: ContextVar[Optional[list[dict]]] = ContextVar(
    "current_detection_objects",
    default=None,
)
