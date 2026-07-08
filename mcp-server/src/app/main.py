"""MCP server exposing knowledge-graph search over streamable HTTP.

Standalone deployment of the PPR search path from the core repo's query.py.
Note content is served from Episode.content stored in Neo4j — the Obsidian
vault does not exist on the Railway host.

Tools:
  - search_notes:    PPR search against the knowledge graph
  - list_group_ids:  discover available graph partitions (group_ids)
  - set_group_id:    toggle the active group_id used by default in searches

Run locally:
    uv run python -m app.main
"""

from __future__ import annotations

import sys

import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from scipy.sparse import coo_matrix, diags
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.auth import BearerAuthMiddleware
from app.config import (
    AUTH_TOKEN,
    DEFAULT_GROUP_ID,
    HOST,
    MAX_CONTENT_CHARS,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    PORT,
    QUERY_EMBED_MODEL,
)

# Active group_id — changed via set_group_id, overridable per search_notes call.
_state = {"group_id": DEFAULT_GROUP_ID}

# Long-lived clients, shared across requests.
_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
_openai = AsyncOpenAI()

# DNS-rebinding protection rejects non-localhost Host headers (421), which
# breaks access via the Railway domain. Bearer auth is the access control.
mcp = FastMCP(
    "knowledge-graph-search",
    host=HOST,
    port=PORT,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ── Neo4j / embedding helpers ─────────────────────────────────────────────────

_VECTOR_SEED_QUERY = """
MATCH (n:Entity {group_id: $group_id})
WITH n, vector.similarity.cosine(n.name_embedding, $embedding) AS score
WHERE score > 0.5
ORDER BY score DESC
LIMIT $limit
RETURN n.uuid AS uuid, n.name AS name, score
"""

_LOAD_NODES = """
MATCH (n:Entity {group_id: $group_id})
RETURN n.uuid AS uuid
"""

_LOAD_EDGES = """
MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]->(m:Entity {group_id: $group_id})
RETURN e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid
"""

# Walk MENTIONS backwards from top entities to Episodes; content comes from
# the graph (ep.content), not the local filesystem.
_SOURCES_QUERY = """
MATCH (ep:Episode)-[:MENTIONS]->(n:Entity)
WHERE n.uuid IN $entity_uuids AND n.group_id = $group_id
RETURN DISTINCT
    ep.source_file  AS source_file,
    ep.source       AS obsidian_uri,
    ep.name         AS episode_name,
    ep.episode_idx  AS episode_idx,
    ep.content      AS content
ORDER BY ep.source_file, episode_idx
"""

_GROUPS_QUERY = """
MATCH (n:Entity)
RETURN n.group_id AS group_id, count(n) AS entities
ORDER BY group_id
"""

_GROUP_COUNT_QUERY = """
MATCH (n:Entity {group_id: $gid})
RETURN count(n) AS c
"""


async def neo4j_query(cypher: str, **params) -> list[dict]:
    """Run a Cypher query on the shared driver and return rows as dicts."""
    async with _driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run(cypher, **params)
        return [dict(r) async for r in result]


async def embed_query(text: str) -> list[float]:
    """Embed the query in the same space as Entity.name_embedding."""
    resp = await _openai.embeddings.create(model=QUERY_EMBED_MODEL, input=text)
    return resp.data[0].embedding


async def run_ppr(
    group_id: str, seed_weights: dict[str, float], top_n: int
) -> list[str]:
    """Personalized PageRank over the RELATES_TO graph (ported from query.py)."""
    rows_n = await neo4j_query(_LOAD_NODES, group_id=group_id)
    rows_e = await neo4j_query(_LOAD_EDGES, group_id=group_id)
    if not rows_n:
        return []

    sorted_uuids = sorted(r["uuid"] for r in rows_n)
    idx = {uid: i for i, uid in enumerate(sorted_uuids)}
    n_nodes = len(sorted_uuids)

    # Build column-normalised transition matrix
    rs: list[int] = []
    cs: list[int] = []
    ds: list[float] = []
    for e in rows_e:
        s, t = e["source_uuid"], e["target_uuid"]
        if s in idx and t in idx:
            i, j = idx[s], idx[t]
            rs += [i, j]
            cs += [j, i]
            ds += [1.0, 1.0]

    M = coo_matrix((ds, (rs, cs)), shape=(n_nodes, n_nodes)).tocsr()
    col_sums = np.asarray(M.sum(axis=0)).flatten()
    inv = np.where(col_sums > 0, 1.0 / col_sums, 0.0)
    M = M @ diags(inv, format="csr")

    s = np.zeros(n_nodes)
    for uid, w in seed_weights.items():
        if uid in idx:
            s[idx[uid]] = w
    total = s.sum()
    s = s / total if total > 0 else np.ones(n_nodes) / n_nodes

    # Power iteration (alpha=0.15, up to 100 iters)
    p = s.copy()
    for _ in range(100):
        p_new = 0.85 * (M @ p) + 0.15 * s
        if float(np.abs(p_new - p).sum()) < 1e-6:
            break
        p = p_new

    top_idx = np.argsort(p)[::-1][:top_n]
    return [sorted_uuids[i] for i in top_idx]


async def collect_sources(
    entity_uuids: list[str], group_id: str, max_files: int
) -> list[dict]:
    """Return up to max_files source dicts with episode names and content,
    ranked by how many top entities each file mentions."""
    rows = await neo4j_query(
        _SOURCES_QUERY, entity_uuids=entity_uuids, group_id=group_id
    )
    file_hits: dict[str, int] = {}
    file_meta: dict[str, dict] = {}
    for r in rows:
        sf = r["source_file"]
        file_hits[sf] = file_hits.get(sf, 0) + 1
        meta = file_meta.setdefault(
            sf,
            {
                "source_file": sf,
                "obsidian_uri": r["obsidian_uri"],
                "episodes": [],
                "content_parts": [],
            },
        )
        ep_name = r.get("episode_name")
        if ep_name and ep_name not in meta["episodes"]:
            meta["episodes"].append(ep_name)
            if r.get("content"):
                meta["content_parts"].append(r["content"])

    ranked = sorted(file_hits, key=lambda f: -file_hits[f])[:max_files]
    return [file_meta[f] for f in ranked]


# ── tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
async def list_group_ids() -> str:
    """List available knowledge-graph partitions (group_ids) with entity
    counts, and show which one is currently active."""
    rows = await neo4j_query(_GROUPS_QUERY)
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
    rows = await neo4j_query(_GROUP_COUNT_QUERY, gid=group_id)
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
    max_files: int = 8,
    include_content: bool = True,
) -> str:
    """Search the personal knowledge graph and return matching source notes.

    Args:
        query: Natural-language question or topic to search for.
        group_id: Graph partition to search. Omit to use the active group
            (see set_group_id / list_group_ids).
        max_files: Maximum number of source files to return (default 8).
        include_content: Include the note contents in the result (default
            True). Set False to get just filenames and Obsidian URIs.
    """
    gid = group_id or _state["group_id"]

    embedding = await embed_query(query)
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

    if not sources:
        return f"No source files found for {query!r} in group {gid!r}."

    parts = [f"Found {len(sources)} source files (group: {gid})\n"]
    for s in sources:
        name = s["source_file"].rsplit("/", 1)[-1] if s["source_file"] else "?"
        parts.append(f"## {name}")
        if s.get("obsidian_uri"):
            parts.append(f"Obsidian: {s['obsidian_uri']}")
        if s.get("episodes"):
            parts.append(f"Episodes: {', '.join(s['episodes'])}")
        if include_content:
            content = "\n\n".join(s["content_parts"]) or "[no content stored]"
            if len(content) > MAX_CONTENT_CHARS:
                content = content[:MAX_CONTENT_CHARS] + "\n[... truncated]"
            parts.append(f"\n{content}")
        parts.append("")
    return "\n".join(parts)


# ── app ───────────────────────────────────────────────────────────────────────


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    """Unauthenticated liveness probe for Railway healthchecks."""
    return JSONResponse({"status": "ok"})


def main() -> None:
    """Start the server. Fails closed: no auth token, no server."""
    if not AUTH_TOKEN:
        sys.exit(
            "MCP_AUTH_TOKEN is not set. Refusing to start an unauthenticated "
            "server on a public host — set it in the environment."
        )
    if not NEO4J_PASSWORD:
        sys.exit("NEO4J_PASSWORD is not set.")

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)
    print(f"Auth: bearer token required ({len(AUTH_TOKEN)} chars)")
    print(f"MCP endpoint: http://{HOST}:{PORT}/mcp")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
