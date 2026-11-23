"""Unit tests for ToolQualityService's model-graded path."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp_tef_models.schemas import ToolDefinition

from mcp_tef.api.errors import LLMProviderError
from mcp_tef.services.tool_quality_service import ToolQualityService


async def test_evaluate_server_fails_when_a_tool_cannot_be_graded():
    """A grading failure fails the server instead of dropping the tool."""
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("AccessDeniedException"))
    llm_service = MagicMock(provider="bedrock")
    llm_service.make_agent.return_value = agent
    loader = MagicMock()
    loader.load_tools_from_server = AsyncMock(
        return_value=[ToolDefinition(name="get_time", description="Get the time")]
    )
    service = ToolQualityService(mcp_loader_service=loader, llm_service=llm_service)

    with pytest.raises(LLMProviderError, match=r"Failed to grade 1 of 1 tools \(get_time\)"):
        await service.evaluate_server("http://mcp", "streamable-http")
