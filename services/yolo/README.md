# YOLO Object Detection Service

This is a FastAPI-based web service that performs object detection on uploaded images using the YOLOv8 model. The application analyzes images, detects objects, and stores prediction results with SQLAlchemy for later retrieval. SQLite is used by default.

## Setup Instructions

1. Make sure the shared project virtualenv is activated (see the root README).

1. Install requirements (from `services/yolo/`):

```bash
pip install -r torch-requirements.txt
pip install -r requirements.txt
```

1. Configure environment:

```bash
cp .env.example .env
# Edit .env to set your S3 bucket and AWS region
```

1. Run the application:

```bash
python app.py
```

The service will be available at http://<your_server_ip>:8080

You can test the api endpoints using `curl` or Postman. See the API Endpoints section below for details on available endpoints and how to use them.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence score (0.0–1.0) for a detection to be reported. Raise it to get only high-confidence results; lower it to catch more objects. |
| `AWS_REGION` | `us-east-1` | AWS region for S3 |
| `AWS_S3_BUCKET` | required | S3 bucket used to read original images and store predicted images |
| `DB_BACKEND` | `sqlite` | Database backend. Use `sqlite` for local development or `postgres` for PostgreSQL. |
| `DATABASE_URL` | `sqlite:///./predictions.db` | SQLite database URL used when `DB_BACKEND` is not `postgres`. |
| `DB_USER` | `user` | PostgreSQL username when `DB_BACKEND=postgres`. |
| `DB_PASSWORD` | `pass` | PostgreSQL password when `DB_BACKEND=postgres`. |
| `DB_HOST` | `localhost` | PostgreSQL host when `DB_BACKEND=postgres`. |
| `DB_PORT` | `5432` | PostgreSQL port when `DB_BACKEND=postgres`. |
| `DB_NAME` | `db` | PostgreSQL database name when `DB_BACKEND=postgres`. |

Example:
```bash
export CONFIDENCE_THRESHOLD=0.7
python app.py
```

PostgreSQL example:

```bash
export DB_BACKEND=postgres
export DB_USER=polyai
export DB_PASSWORD=secret
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=polyai_predictions
python app.py
```

## Running Tests

The test suite uses `pytest` with a temporary SQLite database — no running server needed.

```bash
pytest tests/
```


## API Endpoints

* `POST /predict` - Upload an image for object detection
* `GET /prediction/{uid}` - Get details of a specific prediction by ID
* `GET /predictions/label/{label}` - Get all predictions containing a specific object label (e.g., "person", "car")
* `GET /predictions/score/{min_score}` - Get predictions with confidence score above threshold (e.g., 0.5)
* `GET /prediction/{uid}/image` - Get the processed image with detection boxes
* `GET /image/{type}/{filename}` - Get original or predicted image by filename

## Testing the API

You can use tools like curl, Postman, or a web browser to test the endpoints. For example:

1. Upload an image:
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8080/predict
```

2. View detection results (replace {uid} with the ID returned from the upload):
```bash
curl http://localhost:8080/prediction/{uid}
