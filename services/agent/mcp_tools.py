import base64
import copy
import json
import logging

from langchain_core.tools import StructuredTool

import config
import storage
from context import current_detection_objects, current_image_s3_key


MCP_CLIENT = None
MCP_TOOLS = []


def public_mcp_tool_schema(mcp_tool) -> dict:
    args_schema = mcp_tool.args_schema

    if isinstance(args_schema, dict):
        schema = copy.deepcopy(args_schema)
    else:
        schema = args_schema.model_json_schema()

    properties = schema.get("properties", {})
    for hidden_argument in config.MCP_HIDDEN_ARGUMENTS:
        properties.pop(hidden_argument, None)

    required = schema.get("required", [])
    schema["required"] = [
        name
        for name in required
        if name not in config.MCP_HIDDEN_ARGUMENTS
    ]
    return schema


async def execute_mcp_image_tool(mcp_tool, arguments: dict) -> str:
    image_s3_key = current_image_s3_key.get()
    if not image_s3_key:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = storage.read_s3_bytes(image_s3_key)
    private_arguments = dict(arguments)
    private_arguments["image_b64"] = base64.b64encode(image_bytes).decode(
        "utf-8"
    )
    private_arguments["detection_objects"] = (
        current_detection_objects.get() or []
    )

    try:
        result = await mcp_tool.ainvoke(private_arguments)
        if (
            not isinstance(result, list)
            or len(result) != 1
            or not isinstance(result[0], dict)
            or result[0].get("type") != "text"
            or not isinstance(result[0].get("text"), str)
        ):
            raise ValueError(
                "MCP image tool must return one LangChain text block"
            )

        result_b64 = result[0]["text"]
        processed_bytes = base64.b64decode(result_b64, validate=True)

        if not processed_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("MCP image tool did not return a PNG")

        processed_s3_key = storage.build_processed_image_s3_key(
            image_s3_key,
            mcp_tool.name,
        )
        storage.upload_bytes_to_s3(
            processed_bytes,
            processed_s3_key,
            "image/png",
        )
    except Exception:
        logging.exception(
            "Could not execute MCP image tool: %s",
            mcp_tool.name,
        )
        return json.dumps({"error": config.IMAGE_EDIT_ERROR_MESSAGE})

    current_image_s3_key.set(processed_s3_key)
    current_detection_objects.set(None)
    return json.dumps(
        {
            "operation": mcp_tool.name,
            "processed_image_s3_key": processed_s3_key,
            "message": "Image processing completed.",
        }
    )


def create_mcp_tool_proxy(mcp_tool) -> StructuredTool:
    async def invoke_proxy(**arguments):
        return await execute_mcp_image_tool(mcp_tool, arguments)

    return StructuredTool.from_function(
        coroutine=invoke_proxy,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=public_mcp_tool_schema(mcp_tool),
        infer_schema=False,
    )


async def discover_mcp_tools() -> list:
    global MCP_CLIENT, MCP_TOOLS

    from langchain_mcp_adapters.client import MultiServerMCPClient

    MCP_CLIENT = MultiServerMCPClient(
        {
            "img-proc": {
                "url": f"{config.IMG_PROC_MCP_URL.rstrip('/')}/mcp",
                "transport": "http",
            }
        }
    )
    MCP_TOOLS = await MCP_CLIENT.get_tools()
    return MCP_TOOLS


async def discover_mcp_tool_proxies() -> list[StructuredTool]:
    mcp_tools = await discover_mcp_tools()
    return [create_mcp_tool_proxy(mcp_tool) for mcp_tool in mcp_tools]
