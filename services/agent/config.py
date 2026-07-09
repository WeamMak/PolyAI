import os

from dotenv import load_dotenv


load_dotenv()

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
IMG_PROC_MCP_URL = os.environ.get("IMG_PROC_MCP_URL", "http://localhost:8090")
AWS_REGION = os.environ.get(
    "AWS_REGION",
    os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

BEDROCK_MODEL_PREFIX = "bedrock/"
DEFAULT_MODEL = f"{BEDROCK_MODEL_PREFIX}openai.gpt-oss-20b-1:0"
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
BEDROCK_MODEL_ID = MODEL.removeprefix(BEDROCK_MODEL_PREFIX)

LLM_REQUESTS_PER_SECOND = 0.25
LLM_RATE_LIMIT_CHECK_SECONDS = 0.1
LLM_RATE_LIMIT_BUCKET_SIZE = 1
LLM_RATE_LIMIT_SECONDS = 1 / LLM_REQUESTS_PER_SECOND

ALLOWED_BEDROCK_MODEL_IDS = {
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "openai.gpt-oss-20b-1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
}

IMAGE_EDIT_ERROR_MESSAGE = "The image edit could not be completed. Please try again."
MCP_HIDDEN_ARGUMENTS = {"image_b64", "detection_objects"}
REQUIRED_MODEL_FEATURES = ["structured_output", "tool_calling"]
CONTEXT_LIMIT_WARNING_RATIO = 0.9


def validate_configured_model() -> None:
    if BEDROCK_MODEL_ID in ALLOWED_BEDROCK_MODEL_IDS:
        return

    allowed_list = "\n  ".join(
        f"{BEDROCK_MODEL_PREFIX}{model_id}"
        for model_id in sorted(ALLOWED_BEDROCK_MODEL_IDS)
    )
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported Bedrock models:\n"
        f"  {allowed_list}\n"
    )


validate_configured_model()
