"""Tool quality graded by a Devin session (the devin provider)."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import AsyncClient
from mcp_tef_models.schemas import ToolDefinition

from mcp_tef.api.tool_quality import get_mcp_loader_service
from mcp_tef.config.settings import Settings, get_settings
from mcp_tef.services.devin_service import structured_output_schema
from mcp_tef.services.tool_quality_service import DevinToolEvaluations

TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    ),
    ToolDefinition(name="get_time", description="Get the time"),
]


def evaluation(tool_name: str) -> dict:
    return {
        "tool_name": tool_name,
        "clarity": {"score": 8, "explanation": "clear"},
        "completeness": {"score": 6, "explanation": "no units"},
        "conciseness": {"score": 9, "explanation": "short"},
        "suggested_description": f"Better {tool_name}",
    }


class FakeDevin:
    """Answers the Devin v3 session endpoints and records the requests."""

    def __init__(
        self,
        outputs: list[dict | None],
        final_status: str = "running",
        final_detail: str | None = None,
    ):
        self.outputs = outputs
        self.final_status = final_status
        self.final_detail = final_detail
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"session_id": "devin-1", "url": "https://x/1"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        output = self.outputs.pop(0) if self.outputs else None
        if self.outputs:
            status, detail = "running", "working"
        else:
            status, detail = self.final_status, self.final_detail
        return httpx.Response(
            200, json={"status": status, "status_detail": detail, "structured_output": output}
        )

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


@pytest.fixture
def quality(test_app, client: AsyncClient):
    settings = Settings(
        database_url="sqlite:///:memory:",
        devin_api_key="cog_test",
        devin_org_id="org-test",
        devin_poll_interval=0.01,
    )
    loader = AsyncMock()
    loader.load_tools_from_server = AsyncMock(return_value=TOOLS)
    test_app.dependency_overrides[get_mcp_loader_service] = lambda: loader
    test_app.dependency_overrides[get_settings] = lambda: settings
    real_client = httpx.AsyncClient

    async def call(fake: FakeDevin, model_name: str = "default", **headers: str) -> dict:
        def make_client(**kwargs):
            kwargs["transport"] = httpx.MockTransport(fake)
            return real_client(**kwargs)

        with patch("mcp_tef.services.devin_service.httpx.AsyncClient", make_client):
            response = await client.get(
                "/mcp-servers/tools/quality",
                params={
                    "server_urls": "http://localhost:3000/mcp",
                    "model_provider": "devin",
                    "model_name": model_name,
                },
                headers=headers,
            )
        assert response.status_code == 200, response.text
        return response.json()

    yield call
    test_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_grades_every_tool_in_one_session(quality):
    fake = FakeDevin([None, {"evaluations": [evaluation("get_time"), evaluation("get_weather")]}])

    data = await quality(fake)

    assert data["errors"] is None
    assert [r["tool_name"] for r in data["results"]] == ["get_weather", "get_time"]
    result = data["results"][0]["evaluation_result"]
    assert result["completeness"] == {"score": 6, "explanation": "no units"}
    assert result["suggested_description"] == "Better get_weather"
    assert "tool_name" not in result

    create = fake.requests[0]
    assert create.url.path == "/v3/organizations/org-test/sessions"
    assert create.headers["Authorization"] == "Bearer cog_test"
    body = json.loads(create.content)
    assert body["structured_output_schema"] == structured_output_schema(DevinToolEvaluations)
    assert body["repos"] == []
    assert body["secret_ids"] == []
    assert body["knowledge_ids"] == []
    assert body["max_acu_limit"] == 5
    assert body["memory_enabled"] is False
    assert "devin_mode" not in body
    assert "Tool Name: get_weather" in body["prompt"]
    assert "Tool Name: get_time" in body["prompt"]
    session = "/v3/organizations/org-test/sessions/devin-1"
    assert fake.calls()[1:] == [("GET", session), ("GET", session), ("DELETE", session)]


@pytest.mark.asyncio
async def test_header_key_and_mode(quality):
    fake = FakeDevin([{"evaluations": [evaluation("get_weather"), evaluation("get_time")]}])

    await quality(fake, model_name="lite", **{"X-Model-API-Key": "cog_header"})

    assert fake.requests[0].headers["Authorization"] == "Bearer cog_header"
    assert json.loads(fake.requests[0].content)["devin_mode"] == "lite"


@pytest.mark.asyncio
async def test_session_ending_without_output_is_an_error(quality):
    data = await quality(FakeDevin([{"evaluations": "not a list"}], final_status="exit"))

    assert data["results"] == []
    assert len(data["errors"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["finished", "waiting_for_user"])
async def test_session_done_without_output_is_an_error(quality, detail):
    fake = FakeDevin([None], final_status="running", final_detail=detail)

    data = await quality(fake)

    assert data["results"] == []
    assert len(data["errors"]) == 1
    assert [method for method, _ in fake.calls()] == ["POST", "GET", "DELETE"]


@pytest.mark.asyncio
async def test_exited_session_is_not_terminated(quality):
    fake = FakeDevin([None], final_status="exit")

    await quality(fake)

    assert [method for method, _ in fake.calls()] == ["POST", "GET"]


@pytest.mark.asyncio
async def test_missing_tool_is_an_error(quality):
    data = await quality(FakeDevin([{"evaluations": [evaluation("get_weather")]}]))

    assert data["results"] == []
    assert len(data["errors"]) == 1


@pytest.mark.asyncio
async def test_refused_pat_is_an_error(quality):
    requests: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403, json={"detail": "org not accessible"})

    data = await quality(refuse)

    assert data["results"] == []
    assert len(data["errors"]) == 1
    assert "refused the PAT or org (403)" in data["errors"][0]
    assert "org not accessible" in data["errors"][0]
    assert [r.method for r in requests] == ["POST"]


def test_schema_is_draft_7():
    schema = structured_output_schema(DevinToolEvaluations)

    assert schema["type"] == "object"
    assert "$defs" not in schema
    refs = json.dumps(schema)
    assert "#/$defs/" not in refs
    assert "#/definitions/" in refs
