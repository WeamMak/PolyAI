import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import PredictRequest, predict
from models import Base, DetectionObject, PredictionSession


TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


class TestPredictionTime(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = os.path.join(self.temp_dir.name, "test_predictions.db")
        database_url = f"sqlite:///{database_path}"

        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
        TestingSessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )
        Base.metadata.create_all(bind=self.engine)
        self.db = TestingSessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_predict_includes_processing_time(self):
        def fake_download_s3_file(image_s3_key, local_path):
            shutil.copyfile(TEST_IMAGE, local_path)

        request = PredictRequest(
            image_s3_key="chats/chat-123/image-123/original/beatles.jpeg"
        )

        with patch("app.download_s3_file", side_effect=fake_download_s3_file), patch(
            "app.upload_s3_file"
        ) as upload_s3_file:
            data = predict(request, self.db)

        self.assertIn("time_took", data)
        self.assertIsInstance(data["time_took"], (int, float))
        self.assertGreaterEqual(data["time_took"], 0)
        self.assertEqual(
            data["predicted_image_s3_key"],
            "chats/chat-123/image-123/predicted/beatles.jpeg",
        )

        upload_s3_file.assert_called_once()
        uploaded_args = upload_s3_file.call_args.args
        self.assertEqual(
            uploaded_args[1],
            "chats/chat-123/image-123/predicted/beatles.jpeg",
        )
        self.assertEqual(uploaded_args[2], "image/jpeg")

        session = (
            self.db.query(PredictionSession)
            .filter_by(uid=data["prediction_uid"])
            .first()
        )
        self.assertIsNotNone(session)
        self.assertEqual(
            session.original_image,
            "chats/chat-123/image-123/original/beatles.jpeg",
        )
        self.assertEqual(
            session.predicted_image,
            "chats/chat-123/image-123/predicted/beatles.jpeg",
        )

        detection_objects = (
            self.db.query(DetectionObject)
            .filter_by(prediction_uid=data["prediction_uid"])
            .all()
        )
        self.assertEqual(len(detection_objects), data["detection_count"])
