"""Bearer-token auth middleware for the public Railway deployment."""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import AUTH_TOKEN

# Unauthenticated paths: Railway healthchecks can't send headers.
_EXEMPT_PATHS = {"/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without the expected bearer token."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth, f"Bearer {AUTH_TOKEN}"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)
