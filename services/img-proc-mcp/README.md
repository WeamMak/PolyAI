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

To run the HTTP service that the agent uses:

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8090 python app.py
```

## Test The Tools

From this directory:

```bash
python tests/test_mcp_tools.py
```

To test with your own image, put a `.png`, `.jpg`, `.jpeg`, `.webp`, or `.bmp` file directly inside `tests/`.

The script uses that image if it finds one. If not, it creates a sample image. It calls every MCP tool and saves the results in:

```text
tests/outputs/
```

## Tools

Each tool accepts a base64-encoded image and returns a base64-encoded PNG.
For object-specific edits, the tool also accepts YOLO detection objects plus
`label`, `ordinal`, and `from_side`. Object matching, cropping, and composition
all happen inside this service.

| Tool | Description |
|---|---|
| `rotate` | Rotate the entire image or a selected detected object. |
| `flip` | Flip the entire image or a selected detected object. |
| `blur` | Blur the entire image or a selected detected object. |
| `resize` | Resize the entire image or return a selected object at the requested size. |
| `crop` | Crop the image or a region relative to a selected object. |
| `add_noise` | Add salt-and-pepper noise to the image or a selected object. |

The standard Streamable HTTP MCP endpoint is:

```text
http://localhost:8090/mcp
```
