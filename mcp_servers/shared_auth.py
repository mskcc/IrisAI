"""Shared authentication module for all MCP servers.

Provides the StaticBearerProvider token verifier used by all FastMCP servers.
Import this instead of duplicating the class in each server file.
"""
import os
from fastmcp.server.auth import TokenVerifier, AccessToken


class StaticBearerProvider(TokenVerifier):
    """Verify bearer tokens against the shared MCP_SHARED_BEARER_TOKEN env var."""

    async def verify_token(self, token: str) -> AccessToken | None:
        expected_token = os.environ.get("MCP_SHARED_BEARER_TOKEN")
        if token == expected_token:
            return AccessToken(
                token=token,
                client_id="chainlit",
                subject="chainlit",
                scopes=["mcp:connect", "mcp:tools", "mcp:call"]
            )
        return None
