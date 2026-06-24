import os
import tempfile
import unittest

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import predict
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
        with open(TEST_IMAGE, "rb") as f:
            file = UploadFile(f, filename="beatles.jpeg")
            data = predict(file, self.db)

        self.assertIn("time_took", data)
        self.assertIsInstance(data["time_took"], (int, float))
        self.assertGreaterEqual(data["time_took"], 0)
