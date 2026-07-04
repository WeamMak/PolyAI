# Image Processing MCP Server

This service exposes simple image manipulation tools through MCP.

## Setup

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

```bash
mcp dev app.py
```

The `blur` tool accepts a base64-encoded image and returns a base64-encoded PNG.
