import asyncio

import pytest
from prometheus_client import REGISTRY
from starlette.requests import Request
from starlette.responses import Response


def make_chat_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "raw_path": b"/chat",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }
    )


def metric_value(name: str, labels=None) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0 if value is None else value


def test_metrics_endpoint_exposes_expected_metrics_and_buckets(agent_module):
    response = agent_module.metrics()
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"] == agent_module.CONTENT_TYPE_LATEST
    assert "agent_chat_requests_total" in body
    assert "agent_chat_request_duration_seconds" in body
    assert "agent_input_tokens_total" in body
    assert "agent_output_tokens_total" in body
    assert metric_value(
        "agent_chat_requests_total",
        {"status": "success"},
    ) == 0
    assert metric_value(
        "agent_chat_requests_total",
        {"status": "error"},
    ) == 0

    histogram_samples = agent_module.CHAT_REQUEST_DURATION_SECONDS.collect()[
        0
    ].samples
    finite_buckets = [
        float(sample.labels["le"])
        for sample in histogram_samples
        if sample.name == "agent_chat_request_duration_seconds_bucket"
        and sample.labels["le"] != "+Inf"
    ]
    assert finite_buckets == [
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        20,
        30,
        60,
        120,
        300,
    ]


def test_chat_metrics_classify_responses_and_exceptions(agent_module):
    async def successful_response(request):
        return Response(status_code=200)

    async def error_response(request):
        return Response(status_code=429)

    async def server_exception(request):
        raise RuntimeError("agent failed")

    async def exercise_middleware():
        response = await agent_module.record_chat_metrics(
            make_chat_request(),
            successful_response,
        )
        assert response.status_code == 200

        response = await agent_module.record_chat_metrics(
            make_chat_request(),
            error_response,
        )
        assert response.status_code == 429

        with pytest.raises(RuntimeError, match="agent failed"):
            await agent_module.record_chat_metrics(
                make_chat_request(),
                server_exception,
            )

    asyncio.run(exercise_middleware())

    assert metric_value(
        "agent_chat_requests_total",
        {"status": "success"},
    ) == 1
    assert metric_value(
        "agent_chat_requests_total",
        {"status": "error"},
    ) == 2
    assert metric_value("agent_chat_request_duration_seconds_count") == 3


def test_chat_counts_tokens_after_successful_agent_run(
    agent_module,
    monkeypatch,
):
    async def fake_run_agent(history):
        return agent_module.AgentRunResult(
            response="Done.",
            tokens_used=agent_module.TokenUsage(
                input=12,
                output=5,
                total=17,
            ),
        )

    monkeypatch.setattr(agent_module, "check_chat_rate_limit", lambda: None)
    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)

    request = agent_module.ChatRequest(
        messages=[
            agent_module.ChatMessage(role="user", content="Hello"),
        ]
    )
    response = asyncio.run(agent_module.chat(request))

    assert response.tokens_used.input == 12
    assert response.tokens_used.output == 5
    assert metric_value("agent_input_tokens_total") == 12
    assert metric_value("agent_output_tokens_total") == 5


def test_chat_does_not_count_tokens_when_agent_run_fails(
    agent_module,
    monkeypatch,
):
    async def failing_run_agent(history):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(agent_module, "check_chat_rate_limit", lambda: None)
    monkeypatch.setattr(agent_module, "run_agent", failing_run_agent)

    request = agent_module.ChatRequest(
        messages=[
            agent_module.ChatMessage(role="user", content="Hello"),
        ]
    )

    with pytest.raises(RuntimeError, match="agent failed"):
        asyncio.run(agent_module.chat(request))

    assert metric_value("agent_input_tokens_total") == 0
    assert metric_value("agent_output_tokens_total") == 0
