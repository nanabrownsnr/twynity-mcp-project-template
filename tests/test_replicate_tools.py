"""Test Replicate tool implementations."""

import pytest
from fastmcp import FastMCP

from app.tools.replicate import register_tools


@pytest.fixture
def mcp_instance():
    """Create an isolated MCP server with Replicate tools."""
    mcp = FastMCP("test-replicate-server")
    register_tools(mcp)
    return mcp


@pytest.mark.asyncio
async def test_replicate_tools_registered(mcp_instance):
    """Verify that Replicate tools are properly registered."""
    tools = await mcp_instance.list_tools()
    
    # Check that our Replicate tools are registered
    tool_names = [tool.name for tool in tools]
    assert "run_model" in tool_names
    assert "list_models" in tool_names
    assert "get_model_info" in tool_names
    assert "check_replicate_connection" in tool_names


# Note: Actual functional testing with real API calls would require mocking
# or valid credentials. Since we don't have those here, we can only test structure.