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

| Tool | Description |
|---|---|
| `rotate(image_b64, angle, expand=True)` | Rotate the image by an angle in degrees. |
| `flip(image_b64, direction="horizontal")` | Flip the image horizontally or vertically. |
| `blur(image_b64, radius=2.0)` | Apply Gaussian blur. |
| `resize(image_b64, width, height)` | Resize to the given width and height. |
| `crop(image_b64, left, top, right, bottom)` | Crop a rectangular region. |
| `add_noise(image_b64, amount=0.02, salt_vs_pepper=0.5)` | Add salt-and-pepper noise. |
