"""Auth middleware for the public Railway deployment.

Two ways to authenticate:
  1. Bearer header — Claude Code, Claude Desktop, the Claude API.
  2. Capability URL — the token as the leading path segment
     (https://host/<token>/mcp), for clients that can't send custom
     headers (claude.ai web custom connectors support only OAuth or
     no-auth). The URL is the credential: share it like a password.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import AUTH_TOKEN

# Unauthenticated paths: Railway healthchecks can't send headers.
_EXEMPT_PATHS = {"/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that carry neither the bearer token nor the
    secret leading path segment."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if hmac.compare_digest(auth, f"Bearer {AUTH_TOKEN}"):
            return await call_next(request)

        # Capability URL: strip the token segment and route the rest
        # ("/<token>/mcp" -> "/mcp") so MCP sees its normal path.
        parts = request.url.path.split("/", 2)  # ["", "<segment>", "rest"]
        if len(parts) == 3 and hmac.compare_digest(parts[1], AUTH_TOKEN):
            request.scope["path"] = "/" + parts[2]
            request.scope["raw_path"] = request.scope["path"].encode()
            return await call_next(request)

        return JSONResponse({"error": "unauthorized"}, status_code=401)
