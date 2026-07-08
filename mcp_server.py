"""MCP server exposing the knowledge-graph search tool over streamable HTTP.

Wraps the retrieval pipeline from query.py (PPR / fast-dual) as MCP tools so
any MCP client (Claude Code, Claude Desktop, a remote machine via ngrok) can
search the graph.

Tools:
  - search_notes:    run a search against the knowledge graph
  - list_group_ids:  discover available graph partitions (group_ids)
  - set_group_id:    toggle the active group_id used by default in searches

Usage:
    uv run dotenv run -- python mcp_server.py            # listens on :8848
    ngrok http 8848                                       # expose the tunnel

    # connect from Claude Code (any machine):
    claude mcp add --transport http kg-search https://<id>.ngrok.app/mcp \
        --header "Authorization: Bearer $MCP_AUTH_TOKEN"

Set MCP_AUTH_TOKEN in .env to require a bearer token (recommended with ngrok).
Leave it unset for open local access.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

load_dotenv()

from query import (  # noqa: E402  (needs load_dotenv first)
    collect_sources,
    embed_query,
    neo4j_query,
    run_fast_dual,
    run_ppr,
    _VECTOR_SEED_QUERY,
)

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8848"))
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")
MAX_CONTENT_CHARS = 6000

# Active group_id — the "toggle" state. Changed via set_group_id, overridable
# per call in search_notes.
_state = {"group_id": os.getenv("DEFAULT_GROUP_ID", "testing-1")}

# DNS-rebinding protection rejects non-localhost Host headers (421), which
# breaks access via ngrok/cloud domains. Bearer auth is the access control.
mcp = FastMCP(
    "knowledge-graph-search",
    host=HOST,
    port=PORT,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)


# ── tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
async def list_group_ids() -> str:
    """List available knowledge-graph partitions (group_ids) with entity
    counts, and show which one is currently active."""
    rows = await neo4j_query(
        "MATCH (n:Entity) RETURN n.group_id AS group_id, count(n) AS entities "
        "ORDER BY group_id"
    )
    if not rows:
        return "No groups found — is the graph uploaded?"
    lines = ["Available group_ids:"]
    for r in rows:
        marker = " (active)" if r["group_id"] == _state["group_id"] else ""
        lines.append(f"  - {r['group_id']}: {r['entities']} entities{marker}")
    return "\n".join(lines)


@mcp.tool()
async def set_group_id(group_id: str) -> str:
    """Switch the active group_id used by search_notes when no explicit
    group_id is passed.

    Args:
        group_id: The graph partition to make active (see list_group_ids).
    """
    rows = await neo4j_query(
        "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS c", gid=group_id
    )
    count = rows[0]["c"] if rows else 0
    if count == 0:
        return (
            f"Warning: group_id {group_id!r} has no entities. "
            f"Active group left at {_state['group_id']!r}. "
            "Use list_group_ids to see valid options."
        )
    _state["group_id"] = group_id
    return f"Active group_id set to {group_id!r} ({count} entities)."


@mcp.tool()
async def search_notes(
    query: str,
    group_id: str | None = None,
    method: str = "ppr",
    max_files: int = 8,
    include_content: bool = True,
) -> str:
    """Search the personal knowledge graph and return matching source notes.

    Args:
        query: Natural-language question or topic to search for.
        group_id: Graph partition to search. Omit to use the active group
            (see set_group_id / list_group_ids).
        method: "ppr" (fast, single-hop) or "fast-dual" (multi-hop, slower,
            uses an LLM bridge).
        max_files: Maximum number of source files to return (default 8).
        include_content: Include the file contents in the result (default
            True). Set False to get just filenames and Obsidian URIs.
    """
    gid = group_id or _state["group_id"]

    if method == "fast-dual":
        sources = await run_fast_dual(
            query, gid, max_hops=4, max_files=max_files,
            llm_model="gpt-4.1-mini", min_relevance=0.20,
        )
    elif method == "ppr":
        embedding = embed_query(query)
        seed_rows = await neo4j_query(
            _VECTOR_SEED_QUERY, group_id=gid, embedding=embedding, limit=20
        )
        if not seed_rows:
            return (
                f"No seed entities found in group {gid!r}. "
                "Check the group_id with list_group_ids."
            )
        seed_weights = {r["uuid"]: float(r["score"]) for r in seed_rows}
        top_uuids = await run_ppr(gid, seed_weights, top_n=15)
        sources = await collect_sources(top_uuids, gid, max_files)
    else:
        return f"Unknown method {method!r} — use 'ppr' or 'fast-dual'."

    if not sources:
        return f"No source files found for {query!r} in group {gid!r}."

    parts = [f"Found {len(sources)} source files (group: {gid}, method: {method})\n"]
    for s in sources:
        name = Path(s["source_file"]).name
        parts.append(f"## {name}")
        if s.get("obsidian_uri"):
            parts.append(f"Obsidian: {s['obsidian_uri']}")
        if s.get("episodes"):
            parts.append(f"Episodes: {', '.join(s['episodes'])}")
        if include_content:
            try:
                content = Path(s["source_file"]).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except OSError:
                content = "[file not readable]"
            if len(content) > MAX_CONTENT_CHARS:
                content = content[:MAX_CONTENT_CHARS] + "\n[... truncated]"
            parts.append(f"\n{content}")
        parts.append("")
    return "\n".join(parts)


# ── auth middleware ───────────────────────────────────────────────────────────


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without the expected bearer token.

    Only active when MCP_AUTH_TOKEN is set — needed because ngrok exposes
    the server publicly.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {AUTH_TOKEN}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def main() -> None:
    app = mcp.streamable_http_app()
    if AUTH_TOKEN:
        app.add_middleware(BearerAuthMiddleware)
        print(f"Auth: bearer token required ({len(AUTH_TOKEN)} chars)")
    else:
        print("Auth: DISABLED (set MCP_AUTH_TOKEN in .env before exposing via ngrok)")
    print(f"MCP endpoint: http://{HOST}:{PORT}/mcp")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
