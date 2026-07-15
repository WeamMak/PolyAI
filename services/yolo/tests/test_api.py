import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as yolo_app
from app import (
    PredictRequest,
    PredictResponse,
    app as fastapi_app,
    get_confidence_threshold,
    get_detection_objects_by_score,
    get_prediction_by_uid,
    get_prediction_image,
    get_predictions_by_empty_label,
    get_predictions_by_label,
    health,
    predict,
)
from models import Base, DetectionObject, PredictionSession


@pytest.fixture
def db_session(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'test_predictions.db'}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def insert_prediction_for_label_test(db_session):
    session = PredictionSession(
        uid="abc-123",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        original_image="uploads/original/test.jpg",
        predicted_image="uploads/predicted/test.jpg",
    )

    db_session.add(session)
    db_session.add(
        DetectionObject(
            prediction_uid="abc-123",
            label="person",
            score=0.91,
            box="[10, 20, 100, 200]",
        )
    )
    db_session.add(
        DetectionObject(
            prediction_uid="abc-123",
            label="car",
            score=0.82,
            box="[30, 40, 120, 220]",
        )
    )
    db_session.commit()


def assert_http_error(error, status_code, detail):
    assert error.value.status_code == status_code
    assert error.value.detail == detail


def test_predict_route_uses_structured_response_model():
    route = next(
        route
        for route in fastapi_app.routes
        if route.path == "/predict" and "POST" in route.methods
    )

    assert route.response_model is PredictResponse


def test_predict_response_model_accepts_expected_shape():
    response = PredictResponse(
        prediction_uid="a1b2c3",
        detection_count=3,
        labels=["person", "dog", "cat"],
        time_took=1.23,
        predicted_image_s3_key="chats/chat-123/image-123/predicted/image.jpg",
    )

    assert response.prediction_uid == "a1b2c3"
    assert response.detection_count == 3
    assert response.labels == ["person", "dog", "cat"]
    assert response.time_took == 1.23
    assert (
        response.predicted_image_s3_key
        == "chats/chat-123/image-123/predicted/image.jpg"
    )


def test_build_predicted_image_s3_key_keeps_original_image_prefix():
    assert yolo_app.build_predicted_image_s3_key(
        "chats/chat-123/image-123/original/image.png"
    ) == "chats/chat-123/image-123/predicted/image.png"


def test_build_predicted_image_s3_key_keeps_edit_checkpoint_prefix():
    assert yolo_app.build_predicted_image_s3_key(
        "chats/chat-123/image-123/edits/job-123/step-002-flip.png"
    ) == "chats/chat-123/image-123/edits/job-123/predicted/step-002-flip.png"


def test_upload_s3_file_returns_clean_error_for_transfer_failure(monkeypatch):
    class FakeS3Client:
        def upload_file(self, local_path, bucket, image_s3_key, ExtraArgs):
            raise yolo_app.S3UploadFailedError("access denied")

    monkeypatch.setattr(yolo_app, "AWS_S3_BUCKET", "polyai-images")
    monkeypatch.setattr(yolo_app, "get_s3_client", lambda: FakeS3Client())

    with pytest.raises(HTTPException) as error:
        yolo_app.upload_s3_file(
            "/tmp/predicted.png",
            "chats/chat-123/image-123/edits/job-123/predicted/step-002-flip.png",
            "image/png",
        )

    assert_http_error(error, 502, "Could not upload predicted image to S3")


def test_get_confidence_threshold_uses_default(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    assert get_confidence_threshold() == 0.5


def test_get_confidence_threshold_uses_environment_value(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    assert get_confidence_threshold() == 0.7


def test_get_predictions_by_label_returns_matching_sessions(db_session):
    insert_prediction_for_label_test(db_session)

    assert get_predictions_by_label("person", db_session) == [
        {
            "uid": "abc-123",
            "timestamp": "2024-01-01 12:00:00",
            "detection_objects": [
                {
                    "id": 1,
                    "label": "person",
                    "score": 0.91,
                    "box": "[10, 20, 100, 200]",
                }
            ],
        }
    ]


def test_get_predictions_by_label_returns_empty_list_when_no_matches(db_session):
    insert_prediction_for_label_test(db_session)

    assert get_predictions_by_label("dog", db_session) == []


def test_get_predictions_by_empty_label_returns_400():
    with pytest.raises(HTTPException) as error:
        get_predictions_by_empty_label()

    assert_http_error(error, 400, "Label cannot be empty")


def test_get_detection_objects_by_score_returns_matching_objects(db_session):
    insert_prediction_for_label_test(db_session)

    assert get_detection_objects_by_score(0.5, db_session) == [
        {
            "id": 1,
            "prediction_uid": "abc-123",
            "label": "person",
            "score": 0.91,
            "box": "[10, 20, 100, 200]",
        },
        {
            "id": 2,
            "prediction_uid": "abc-123",
            "label": "car",
            "score": 0.82,
            "box": "[30, 40, 120, 220]",
        },
    ]


def test_get_detection_objects_by_score_returns_empty_list_when_no_matches(
    db_session,
):
    insert_prediction_for_label_test(db_session)

    assert get_detection_objects_by_score(1.0, db_session) == []


def test_get_detection_objects_by_score_returns_400_when_score_is_too_low():
    with pytest.raises(HTTPException) as error:
        get_detection_objects_by_score(-0.1)

    assert_http_error(error, 400, "min_score must be between 0.0 and 1.0")


def test_get_detection_objects_by_score_returns_400_when_score_is_too_high():
    with pytest.raises(HTTPException) as error:
        get_detection_objects_by_score(1.1)

    assert_http_error(error, 400, "min_score must be between 0.0 and 1.0")


def test_predict_rejects_non_image_file(db_session):
    request = PredictRequest(image_s3_key="chats/chat-123/original/notes.txt")

    with pytest.raises(HTTPException) as error:
        predict(request, db_session)

    assert_http_error(error, 400, "Only image files are supported")


def test_get_prediction_by_uid_returns_prediction(db_session):
    insert_prediction_for_label_test(db_session)

    data = get_prediction_by_uid("abc-123", db_session)

    assert data["uid"] == "abc-123"
    assert data["timestamp"] == "2024-01-01 12:00:00"
    assert data["original_image"] == "uploads/original/test.jpg"
    assert data["predicted_image"] == "uploads/predicted/test.jpg"
    assert len(data["detection_objects"]) == 2


def test_get_prediction_by_uid_returns_404_when_missing(db_session):
    with pytest.raises(HTTPException) as error:
        get_prediction_by_uid("missing-uid", db_session)

    assert_http_error(error, 404, "Prediction not found")


def test_get_prediction_image_returns_s3_image(db_session, monkeypatch):
    monkeypatch.setattr(
        yolo_app,
        "read_s3_file",
        lambda image_s3_key: b"fake image bytes",
    )

    db_session.add(
        PredictionSession(
            uid="image-123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            original_image="chats/chat-123/image-123/original/test.jpg",
            predicted_image="chats/chat-123/image-123/predicted/test.jpg",
        )
    )
    db_session.commit()

    response = get_prediction_image("image-123", db_session)

    assert response.status_code == 200
    assert response.media_type == "image/jpeg"
    assert response.body == b"fake image bytes"


def test_get_prediction_image_returns_404_when_missing(db_session):
    with pytest.raises(HTTPException) as error:
        get_prediction_image("missing-uid", db_session)

    assert_http_error(error, 404, "Image not found")


def test_app_no_longer_uses_old_database_helpers():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()

    old_import = "import " + "sqlite" + "3"
    old_initializer = "init" + "_db"
    old_path_name = "DB" + "_PATH"

    assert old_import not in source
    assert old_initializer not in source
    assert old_path_name not in source


def test_health():
    assert health() == {"status": "ok"}


def test_ready_when_service_is_accepting_requests(monkeypatch):
    monkeypatch.setattr(yolo_app, "is_shutting_down", False)

    assert yolo_app.ready() == {"status": "ready"}


def test_ready_returns_503_during_shutdown(monkeypatch):
    monkeypatch.setattr(yolo_app, "is_shutting_down", True)

    with pytest.raises(HTTPException) as error:
        yolo_app.ready()

    assert_http_error(error, 503, "Service is shutting down")
