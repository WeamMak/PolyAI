from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import boto3
import logging
import os
import uuid
import tempfile
import time
import signal
import sys
from botocore.exceptions import BotoCoreError, ClientError

from db import engine, get_db
from models import Base, DetectionObject, PredictionSession


class PredictRequest(BaseModel):
    image_s3_key: str


class PredictResponse(BaseModel):
    prediction_uid: str
    detection_count: int
    labels: list[str]
    time_took: float
    predicted_image_s3_key: str


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
def get_confidence_threshold():
    raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")

    if raw_threshold is not None:
        threshold = float(raw_threshold)
        logging.info(f"CONFIDENCE_THRESHOLD set to {threshold} (from environment)")
        return threshold

    threshold = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {threshold}")
    return threshold


CONFIDENCE_THRESHOLD = get_confidence_threshold()
AWS_REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

Base.metadata.create_all(bind=engine)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")

is_shutting_down = False


def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    # Perform cleanup: close DB connections, finish pending work, etc.
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}


def format_timestamp(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d %H:%M:%S")


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def require_s3_bucket():
    if not AWS_S3_BUCKET:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET is not configured")

    return AWS_S3_BUCKET


def get_image_type_from_key(image_s3_key):
    ext = os.path.splitext(image_s3_key)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        return ext, "image/jpeg"

    if ext == ".png":
        return ext, "image/png"

    raise HTTPException(status_code=400, detail="Only image files are supported")


def build_predicted_image_s3_key(original_image_s3_key):
    if "/original/" in original_image_s3_key:
        return original_image_s3_key.replace("/original/", "/predicted/", 1)

    base_name = os.path.basename(original_image_s3_key)
    return f"predicted/{uuid.uuid4()}/{base_name}"


def download_s3_file(image_s3_key, local_path):
    try:
        get_s3_client().download_file(
            require_s3_bucket(),
            image_s3_key,
            local_path,
        )
    except (BotoCoreError, ClientError):
        logging.exception("Could not download image from S3")
        raise HTTPException(status_code=502, detail="Could not download image from S3")


def upload_s3_file(local_path, image_s3_key, content_type):
    try:
        get_s3_client().upload_file(
            local_path,
            require_s3_bucket(),
            image_s3_key,
            ExtraArgs={"ContentType": content_type},
        )
    except (BotoCoreError, ClientError):
        logging.exception("Could not upload predicted image to S3")
        raise HTTPException(
            status_code=502,
            detail="Could not upload predicted image to S3",
        )


def read_s3_file(image_s3_key):
    try:
        response = get_s3_client().get_object(
            Bucket=require_s3_bucket(),
            Key=image_s3_key,
        )
        return response["Body"].read()
    except (BotoCoreError, ClientError):
        logging.exception("Could not read predicted image from S3")
        raise HTTPException(status_code=404, detail="Image not found")


def save_prediction_session(db, uid, original_image, predicted_image):
    """
    Save prediction session to database.
    """
    row = PredictionSession(
        uid=uid,
        original_image=original_image,
        predicted_image=predicted_image,
    )
    db.add(row)


def save_detection_object(db, prediction_uid, label, score, box):
    """
    Save detection object to database.
    """
    row = DetectionObject(
        prediction_uid=prediction_uid,
        label=label,
        score=score,
        box=str(box),
    )
    db.add(row)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict objects in an image.
    """
    start_time = time.time()
    ext, content_type = get_image_type_from_key(request.image_s3_key)

    original_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    original_path = original_file.name
    original_file.close()

    predicted_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    predicted_path = predicted_file.name
    predicted_file.close()

    uid = str(uuid.uuid4())
    predicted_s3_key = build_predicted_image_s3_key(request.image_s3_key)

    try:
        download_s3_file(request.image_s3_key, original_path)

        results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

        annotated_frame = results[0].plot()  # NumPy image with boxes
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        upload_s3_file(predicted_path, predicted_s3_key, content_type)

        detected_labels = []

        try:
            save_prediction_session(
                db,
                uid,
                request.image_s3_key,
                predicted_s3_key,
            )

            for box in results[0].boxes:
                label_idx = int(box.cls[0].item())
                label = model.names[label_idx]
                score = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                save_detection_object(db, uid, label, score, bbox)
                detected_labels.append(label)

            db.commit()
        except SQLAlchemyError:
            db.rollback()
            logging.exception("Could not save prediction result")
            raise

    finally:
        for path in [original_path, predicted_path]:
            if os.path.exists(path):
                os.remove(path)

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time,
        "predicted_image_s3_key": predicted_s3_key,
    }


@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects.
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = (
        db.query(DetectionObject)
        .filter_by(prediction_uid=uid)
        .order_by(DetectionObject.id)
        .all()
    )

    return {
        "uid": session.uid,
        "timestamp": format_timestamp(session.timestamp),
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box,
            }
            for obj in objects
        ],
    }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction.
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session or not session.predicted_image:
        raise HTTPException(status_code=404, detail="Image not found")

    _, content_type = get_image_type_from_key(session.predicted_image)
    image_bytes = read_s3_file(session.predicted_image)
    return Response(content=image_bytes, media_type=content_type)


@app.get("/predictions/label/")
def get_predictions_by_empty_label():
    """
    Return 400 when the label is empty.
    """
    raise HTTPException(status_code=400, detail="Label cannot be empty")


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):
    """
    Return all prediction sessions that contain at least one detected object
    with the given label.
    """
    predictions = []

    session_rows = (
        db.query(PredictionSession)
        .join(DetectionObject)
        .filter(DetectionObject.label == label)
        .distinct()
        .order_by(PredictionSession.timestamp.desc())
        .all()
    )

    for session_row in session_rows:
        object_rows = (
            db.query(DetectionObject)
            .filter_by(prediction_uid=session_row.uid, label=label)
            .order_by(DetectionObject.id)
            .all()
        )

        detection_objects = []

        for object_row in object_rows:
            detection_objects.append(
                {
                    "id": object_row.id,
                    "label": object_row.label,
                    "score": object_row.score,
                    "box": object_row.box,
                }
            )

        predictions.append(
            {
                "uid": session_row.uid,
                "timestamp": format_timestamp(session_row.timestamp),
                "detection_objects": detection_objects,
            }
        )

    return predictions


@app.get("/predictions/score/{min_score}")
def get_detection_objects_by_score(
    min_score: float,
    db: Session = Depends(get_db),
):
    """
    Return all detection objects whose confidence score is greater than or
    equal to min_score.
    """
    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0",
        )

    object_rows = (
        db.query(DetectionObject)
        .filter(DetectionObject.score >= min_score)
        .order_by(DetectionObject.score.desc())
        .all()
    )

    detection_objects = []

    for object_row in object_rows:
        detection_objects.append(
            {
                "id": object_row.id,
                "prediction_uid": object_row.prediction_uid,
                "label": object_row.label,
                "score": object_row.score,
                "box": object_row.box,
            }
        )

    return detection_objects


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.get("/health2")
def health2():
    """
    Health check endpoint.
    """
    return {"status": "ok2"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # pragma: no cover

    uvicorn.run(app, host="0.0.0.0", port=8080)  # pragma: no cover
