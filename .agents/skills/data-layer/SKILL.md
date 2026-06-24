---
name: yolo-api-data-layer
description: Refactor or extend the YOLO FastAPI service data layer with SQLAlchemy. Use when asked to replace raw SQLite with SQLAlchemy, make the database backend configurable for SQLite/Postgres, add or change YOLO API database models/tables/columns, update prediction-session or detection-object persistence, add data-layer endpoints such as recent predictions or delete-by-uid, add feedback/rating tables, write tests for /predict or other YOLO API persistence behavior, or fix database architecture issues while preserving existing API behavior.
---

# YOLO API Data Layer

Use this skill for database-layer work in `services/yolo`. The project is educational, so keep code explicit, readable, and close to the existing FastAPI style.

## First Read

Before editing, inspect:

- `services/yolo/app.py`
- `services/yolo/tests/`
- `services/yolo/requirements.txt`
- `services/yolo/README.md` if environment variables or setup docs need updating
- `services/agent/app.py` only to confirm the integration contract with `/predict`

If unsure whether a concept has been taught in the course, search `github.com/alonitac/Fursa26` before choosing a more advanced pattern.

## Non-Negotiables

- Preserve all existing endpoints unless the user explicitly asks to add or remove one.
- Preserve existing status codes and JSON response shapes exactly.
- Preserve `/predict` response keys: `prediction_uid`, `detection_count`, `labels`, `time_took`.
- Preserve existing error details such as `"Only image files are supported"`, `"Prediction not found"`, `"Image not found"`, `"Label cannot be empty"`, and `"min_score must be between 0.0 and 1.0"`.
- Do not send image data to the LLM or change the text-only image handling in `services/agent/app.py`.
- Remove raw application use of `sqlite3`, `DB_PATH`, `init_db()`, manual `CREATE TABLE`, and route-level SQL strings.
- Do not introduce Alembic or a migration framework unless the user explicitly asks for migrations.
- Run the YOLO tests before completion: from `services/yolo`, run `pytest tests/`.

## Target File Shape

For a SQLAlchemy refactor, create these files:

- `services/yolo/models.py`
- `services/yolo/db.py`

Update these files:

- `services/yolo/app.py`
- `services/yolo/tests/*.py`
- `services/yolo/requirements.txt`
- `services/yolo/README.md` if database environment variables change

## `models.py` Pattern

Define declarative models with clear column names matching the existing database tables.

Use this structure as the baseline:

```python
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
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
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)

    prediction = relationship("PredictionSession", back_populates="detection_objects")
```

When adding future tables, keep the same style: explicit class, explicit columns, foreign keys where relationships exist, and simple relationship definitions only when they make the code clearer.

## `db.py` Pattern

Keep database configuration explicit and environment-driven.

Use SQLite by default for development and tests. Use Postgres when `DB_BACKEND=postgres`.

```python
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "db")

if DB_BACKEND == "postgres":
    DATABASE_URL = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Create tables with `Base.metadata.create_all(bind=engine)` during application startup or app initialization. Do not keep the old `init_db()` function.

## `app.py` Refactor Pattern

Use FastAPI dependency injection for database sessions:

```python
from fastapi import Depends
from sqlalchemy.orm import Session

from db import engine, get_db
from models import Base, DetectionObject, PredictionSession
```

Route functions that read or write the database should accept:

```python
db: Session = Depends(get_db)
```

Persistence helpers, if kept, should accept `db` explicitly:

```python
def save_prediction_session(db, uid, original_image, predicted_image):
    row = PredictionSession(
        uid=uid,
        original_image=original_image,
        predicted_image=predicted_image,
    )
    db.add(row)
```

Commit once per request after adding the prediction session and detection objects. Roll back if a database write fails.

Use ORM queries instead of SQL strings:

```python
session = db.query(PredictionSession).filter_by(uid=uid).first()
objects = (
    db.query(DetectionObject)
    .filter_by(prediction_uid=uid)
    .order_by(DetectionObject.id)
    .all()
)
```

For joins, keep the query readable and explicit. Avoid clever helper abstractions unless repeated code becomes genuinely hard to follow.

## Timestamp Compatibility

Raw SQLite currently returns timestamps like:

```text
2024-01-01 12:00:00
```

FastAPI encodes Python `datetime` values as ISO strings with `T`, which would change API responses. Preserve the old shape by formatting timestamps before returning them:

```python
def format_timestamp(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d %H:%M:%S")
```

Use this for every response that includes `timestamp`.

## Tests

Update tests to use SQLAlchemy instead of `sqlite3`, `DB_PATH`, or `init_db()`.

Use a temporary SQLite database and FastAPI dependency override:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import get_db
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
```

Override the app dependency in client fixtures:

```python
@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
```

Insert test data with model instances:

```python
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
db_session.commit()
```

Tests should assert that no raw SQLite setup remains:

- no `import sqlite3`
- no `init_db()`
- no patching `app.DB_PATH`

## Requirements

Add SQLAlchemy to `services/yolo/requirements.txt`.

For Postgres support, add a driver such as `psycopg2-binary`. Keep dependencies minimal and explain any new package in `README.md` when it affects setup.

## Common Feature Recipes

For `GET /predictions/recent`, query `PredictionSession`, order by `timestamp.desc()`, limit to 10, and return a simple response that follows the user's requested shape.

For deleting by uid, load the `PredictionSession`, return 404 if missing, delete through the ORM, and rely on relationship cascade to remove detection objects.

For a new column such as `processing_time_ms`, add it to `PredictionSession`, set it inside `/predict`, and do not change existing response fields unless the user explicitly asks.

For a `UserFeedback` table, add a new model with a foreign key to `prediction_sessions.uid`, clear rating/comment fields, and tests that create feedback for an existing prediction and reject or handle missing predictions according to the requested endpoint behavior.

## Verification Checklist

Before final response:

- `rg "sqlite3|DB_PATH|init_db|CREATE TABLE|SELECT \\*|INSERT INTO" services/yolo` shows no raw application database layer left.
- `pytest tests/` passes from `services/yolo`.
- Existing endpoint response shapes are unchanged.
- New database files are small enough for students to read line by line.
- Final answer summarizes changed files and mentions any verification that could not be run.
