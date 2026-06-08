import os
import pytest
from fastapi.testclient import TestClient
import sqlite3

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db, get_confidence_threshold
TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    init_db()
    return db_file


def insert_prediction_for_label_test(db_file):

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO prediction_sessions (
                uid,
                timestamp,
                original_image,
                predicted_image
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                "abc-123",
                "2024-01-01 12:00:00",
                "uploads/original/test.jpg",
                "uploads/predicted/test.jpg",
            ),
        )

        conn.execute(
            """
            INSERT INTO detection_objects (
                prediction_uid,
                label,
                score,
                box
            )
            VALUES (?, ?, ?, ?)
            """,
            ("abc-123", "person", 0.91, "[10, 20, 100, 200]"),
        )

        conn.execute(
            """
            INSERT INTO detection_objects (
                prediction_uid,
                label,
                score,
                box
            )
            VALUES (?, ?, ?, ?)
            """,
            ("abc-123", "car", 0.82, "[30, 40, 120, 220]"),
        )


def test_get_confidence_threshold_uses_default(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    assert get_confidence_threshold() == 0.5


def test_get_confidence_threshold_uses_environment_value(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    assert get_confidence_threshold() == 0.7


def test_get_predictions_by_label_returns_matching_sessions(client, setup_db):
    insert_prediction_for_label_test(setup_db)
    response = client.get("/predictions/label/person")

    assert response.status_code == 200

    data = response.json()

    assert data == [
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


def test_get_predictions_by_label_returns_empty_list_when_no_matches(client, setup_db):
    insert_prediction_for_label_test(setup_db)
    response = client.get("/predictions/label/dog")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_empty_label_returns_400(client):
    response = client.get("/predictions/label/")

    assert response.status_code == 400
    assert response.json() == {"detail": "Label cannot be empty"}


def test_predict_rejects_non_image_file(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only image files are supported"}
    

def test_get_prediction_by_uid_returns_prediction(client, setup_db):
    insert_prediction_for_label_test(setup_db)

    response = client.get("/prediction/abc-123")

    assert response.status_code == 200

    data = response.json()

    assert data["uid"] == "abc-123"
    assert data["timestamp"] == "2024-01-01 12:00:00"
    assert data["original_image"] == "uploads/original/test.jpg"
    assert data["predicted_image"] == "uploads/predicted/test.jpg"
    assert len(data["detection_objects"]) == 2


def test_get_prediction_by_uid_returns_404_when_missing(client):
    response = client.get("/prediction/missing-uid")

    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found"}


def test_get_prediction_image_returns_file(client, setup_db, tmp_path):
    image_path = tmp_path / "predicted.jpg"
    image_path.write_bytes(b"fake image bytes")

    with sqlite3.connect(setup_db) as conn:
        conn.execute(
            """
            INSERT INTO prediction_sessions (
                uid,
                timestamp,
                original_image,
                predicted_image
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                "image-123",
                "2024-01-01 12:00:00",
                "uploads/original/test.jpg",
                str(image_path),
            ),
        )

    response = client.get("/prediction/image-123/image")

    assert response.status_code == 200
    assert response.content == b"fake image bytes"


def test_get_prediction_image_returns_404_when_missing(client):
    response = client.get("/prediction/missing-uid/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
