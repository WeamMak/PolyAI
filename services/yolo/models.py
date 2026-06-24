from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=utc_now)
    original_image = Column(String)
    predicted_image = Column(String)

    detection_objects = relationship(
        "DetectionObject",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class DetectionObject(Base):
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(
        String,
        ForeignKey("prediction_sessions.uid"),
        index=True,
    )
    label = Column(String, index=True)
    score = Column(Float, index=True)
    box = Column(String)

    prediction = relationship("PredictionSession", back_populates="detection_objects")
