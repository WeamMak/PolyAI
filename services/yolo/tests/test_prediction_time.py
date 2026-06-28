import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import PredictRequest, predict
from models import Base


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
        ):
            data = predict(request, self.db)

        self.assertIn("time_took", data)
        self.assertIsInstance(data["time_took"], (int, float))
        self.assertGreaterEqual(data["time_took"], 0)
