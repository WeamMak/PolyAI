FROM python:3.12-slim-trixie AS builder

ENV VIRTUAL_ENV=/opt/venv

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf "$VIRTUAL_ENV"/bin/pip* \
    && rm -rf "$VIRTUAL_ENV"/lib/python*/site-packages/pip*

FROM python:3.12-slim-trixie AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app.py agent_loop.py config.py context.py mcp_tools.py storage.py yolo_client.py ./

ENV YOLO_SERVICE_URL=http://yolo:8080
ENV AWS_REGION=us-east-1

EXPOSE 8000

CMD ["python", "app.py"]
