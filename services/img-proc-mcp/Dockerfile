FROM python:3.12-slim-trixie AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
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
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8090

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app.py .

EXPOSE 8090

CMD ["python", "app.py"]
