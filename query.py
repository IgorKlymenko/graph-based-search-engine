"""RAG query pipeline: PPR on the knowledge graph → source files → LLM answer.

Flow:
  1. Embed the query
  2. Seed Entity nodes via hybrid search (BM25 + vector cosine)
  3. Run PPR — returns top-ranked Entity nodes
  4. Traverse Episode -[:MENTIONS]-> Entity to collect source files
  5. Read unique source documents from disk
  6. Pass query + documents to the LLM and stream the answer
  7. Print Obsidian URIs so you can open the sources directly

Usage:
    uv run dotenv run -- python query.py "what tools does the team use?"
    uv run dotenv run -- python query.py "what tools does the team use?" --group-id testing-1
    uv run dotenv run -- python query.py "..." --top-nodes 20 --max-files 5 --no-stream
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_SEARCH_METHODS = ["ppr", "fast-dual"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help="Natural-language question to answer")
    p.add_argument("--group-id", default="testing-1",
                   help="Graph partition to search (default: testing-1)")
    p.add_argument("--search-method", default="ppr", choices=_SEARCH_METHODS,
                   help="Retrieval method: 'ppr' (single-hop, fast) or 'fast-dual' (multi-hop, uses LLM bridge). Default: ppr")
    p.add_argument("--top-nodes", default=15, type=int,
                   help="[ppr] How many top PPR entity nodes to collect sources from (default: 15)")
    p.add_argument("--max-hops", default=4, type=int,
                   help="[fast-dual] Max reasoning hops (default: 4)")
    p.add_argument("--max-files", default=8, type=int,
                   help="Max source files to pass to the LLM (default: 8)")
    p.add_argument("--model", default="gpt-4.1-mini",
                   help="LLM model for answer generation (default: gpt-4.1-mini)")
    p.add_argument("--no-stream", action="store_true",
                   help="Print full answer at once instead of streaming")
    p.add_argument("--sources-only", action="store_true",
                   help="Print source files and Obsidian URIs, skip LLM answer")
    p.add_argument("--show-prompt", action="store_true",
                   help="Print the exact system+user prompt sent to the LLM before generating")
    return p.parse_args()


# ── Neo4j helper ──────────────────────────────────────────────────────────────

async def neo4j_query(cypher: str, **params) -> list[dict]:
    from services.neo4j import Neo4jGraphDriver
    driver = Neo4jGraphDriver.from_env()
    try:
        return await driver.execute_query(cypher, **params)
    finally:
        await driver.close()


# ── Source collection ─────────────────────────────────────────────────────────

# Given a set of top-ranked entity UUIDs from PPR, walk backwards through
# MENTIONS edges to find which Episodes mention those entities, then return
# the unique source files (with Obsidian URIs and file content).
_SOURCES_QUERY = """
MATCH (ep:Episode)-[:MENTIONS]->(n:Entity)
WHERE n.uuid IN $entity_uuids AND n.group_id = $group_id
RETURN DISTINCT
    ep.source_file AS source_file,
    ep.source      AS obsidian_uri,
    ep.name        AS episode_name
ORDER BY ep.source_file
"""

# Used by PPR seed step: vector similarity search on entity name embeddings
_VECTOR_SEED_QUERY = """
MATCH (n:Entity {group_id: $group_id})
WITH n, vector.similarity.cosine(n.name_embedding, $embedding) AS score
WHERE score > 0.5
ORDER BY score DESC
LIMIT $limit
RETURN n.uuid AS uuid, n.name AS name, score
"""

# PPR graph load queries (same as retrieval/ uses internally)
_LOAD_NODES = """
MATCH (n:Entity {group_id: $group_id})
RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
"""
_LOAD_EDGES = """
MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]->(m:Entity {group_id: $group_id})
RETURN e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid
"""


def embed_query(text: str) -> list[float]:
    """Embed the query with e5-base-v2's 'query: ' prefix (see services/e5.py).

    Must match whatever embedded n.name_embedding in Neo4j — graph_upload.py
    embeds entity names with the same model's 'passage: ' prefix.
    """
    from services.e5 import embed_e5_sync
    return embed_e5_sync([text], is_query=True)[0]


async def run_ppr(group_id: str, seed_weights: dict[str, float], top_n: int) -> list[str]:
    """Run PPR in-process using the retrieval/ library."""
    import numpy as np
    from scipy.sparse import coo_matrix, diags

    rows_n = await neo4j_query(_LOAD_NODES, group_id=group_id)
    rows_e = await neo4j_query(_LOAD_EDGES, group_id=group_id)

    if not rows_n:
        return []

    sorted_uuids = sorted(r["uuid"] for r in rows_n)
    idx = {uid: i for i, uid in enumerate(sorted_uuids)}
    N = len(sorted_uuids)

    # Build column-normalised transition matrix
    rs, cs, ds = [], [], []
    for e in rows_e:
        s, t = e["source_uuid"], e["target_uuid"]
        if s in idx and t in idx:
            i, j = idx[s], idx[t]
            rs += [i, j]; cs += [j, i]; ds += [1.0, 1.0]

    M = coo_matrix((ds, (rs, cs)), shape=(N, N)).tocsr()
    col_sums = np.asarray(M.sum(axis=0)).flatten()
    inv = np.where(col_sums > 0, 1.0 / col_sums, 0.0)
    M = M @ diags(inv, format="csr")

    s = np.zeros(N)
    for uid, w in seed_weights.items():
        if uid in idx:
            s[idx[uid]] = w
    total = s.sum()
    s = s / total if total > 0 else np.ones(N) / N

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
    entity_uuids: list[str],
    group_id: str,
    max_files: int,
) -> list[dict]:
    """Return up to max_files source dicts (with contributing episode names),
    ranked by how many top entities they mention."""
    rows = await neo4j_query(
        _SOURCES_QUERY,
        entity_uuids=entity_uuids,
        group_id=group_id,
    )
    # Count mentions per file — files covering more top entities rank higher
    file_hits: dict[str, int] = {}
    file_meta: dict[str, dict] = {}
    for r in rows:
        sf = r["source_file"]
        file_hits[sf] = file_hits.get(sf, 0) + 1
        meta = file_meta.setdefault(
            sf, {"source_file": sf, "obsidian_uri": r["obsidian_uri"], "episodes": []}
        )
        ep_name = r.get("episode_name")
        if ep_name and ep_name not in meta["episodes"]:
            meta["episodes"].append(ep_name)

    ranked = sorted(file_hits, key=lambda f: -file_hits[f])[:max_files]
    return [file_meta[f] for f in ranked]


# ── fast-dual path ────────────────────────────────────────────────────────────

_EPISODES_BY_UUID_QUERY = """
MATCH (ep:Episode)
WHERE ep.uuid IN $episode_uuids OR ep.doc_id IN $episode_uuids
RETURN ep.uuid AS uuid, ep.source_file AS source_file, ep.source AS obsidian_uri, ep.name AS episode_name
"""


async def run_fast_dual(
    query: str,
    group_id: str,
    max_hops: int,
    max_files: int,
    llm_model: str,
) -> list[dict]:
    """Run fast-dual via the arena runner and return ranked source file dicts."""
    from arena.runner import ArenaArgs, run_fast_sequential_dual
    from arena.tracker import TokenTracker
    from services.e5 import E5Embedder
    from services.neo4j import Neo4jGraphDriver
    from services.openai import OpenAILLMClient

    driver   = Neo4jGraphDriver.from_env()
    embedder = E5Embedder()
    llm      = OpenAILLMClient()
    tracker  = TokenTracker()

    args = ArenaArgs(
        max_hops=max_hops,
        llm_model=llm_model,
        generate=False,  # we do our own generation
    )

    print(f"Running fast-dual (max_hops={max_hops})...")
    result = await run_fast_sequential_dual(driver, embedder, llm, query, group_id, args, tracker)

    if result.error:
        print(f"fast-dual error: {result.error}")
        return []

    # Print top facts so the user can see what was retrieved
    print(f"  {len(result.facts)} facts retrieved:")
    for fact in result.facts[:8]:
        print(f"    [{fact.score:.3f}] {fact.fact[:100]}")
    if len(result.facts) > 8:
        print(f"    … and {len(result.facts) - 8} more")
    print()

    # Collect all episode UUIDs from the returned facts
    episode_uuids: list[str] = []
    for fact in result.facts:
        episode_uuids.extend(fact.episodes)

    unique_ep_uuids = list(dict.fromkeys(episode_uuids))  # preserve order, dedupe
    print(f"  spanning {len(unique_ep_uuids)} episodes")

    if not unique_ep_uuids:
        return []

    # Resolve episode UUIDs → source files
    ep_rows = await neo4j_query(_EPISODES_BY_UUID_QUERY, episode_uuids=unique_ep_uuids)
    # uuid → {source_file, obsidian_uri, episode_name}
    ep_lookup: dict[str, dict] = {
        r["uuid"]: {
            "source_file": r["source_file"],
            "obsidian_uri": r.get("obsidian_uri", ""),
            "episode_name": r.get("episode_name") or r["uuid"][:8],
        }
        for r in ep_rows if r.get("source_file")
    }

    # Rank files: each fact contributes 1 hit per unique source file it references
    file_hits: dict[str, int] = {}
    file_meta: dict[str, dict] = {}
    for fact in result.facts:
        seen: set[str] = set()
        for ep_uuid in fact.episodes:
            ep = ep_lookup.get(ep_uuid)
            if ep and ep["source_file"] not in seen:
                seen.add(ep["source_file"])
                sf = ep["source_file"]
                file_hits[sf] = file_hits.get(sf, 0) + 1
                meta = file_meta.setdefault(
                    sf, {"source_file": sf, "obsidian_uri": ep["obsidian_uri"], "episodes": []}
                )
                if ep["episode_name"] not in meta["episodes"]:
                    meta["episodes"].append(ep["episode_name"])

    ranked = sorted(file_hits, key=lambda f: -file_hits[f])[:max_files]

    cost = result.cost.total_usd
    print(f"  fast-dual cost: ${cost:.4f}  stop={result.stop_reason}")

    return [file_meta[f] for f in ranked]


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"[file not readable: {path}]"


# ── LLM answer ────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a personal knowledge assistant answering questions from the user's notes.
Answer using ONLY the information in the provided documents.
Be specific and cite which document each claim comes from (use the filename).
If the documents don't contain enough information, say so clearly.\
"""


async def generate_answer(
    client: AsyncOpenAI,
    query: str,
    sources: list[dict],
    model: str,
    stream: bool,
    show_prompt: bool = False,
) -> None:
    docs = []
    for s in sources:
        content  = read_file(s["source_file"])
        name     = Path(s["source_file"]).name
        episodes = s.get("episodes", [])
        header   = f"=== {name}" + (f" (episodes: {', '.join(episodes)})" if episodes else "") + " ==="
        docs.append(f"{header}\n{content}")

    user_msg = f"Question: {query}\n\nDocuments:\n\n" + "\n\n".join(docs)

    if show_prompt:
        print("=" * 60)
        print("PROMPT SENT TO LLM")
        print("=" * 60)
        print(f"[system]\n{_SYSTEM}\n")
        print(f"[user]\n{user_msg}")
        print("=" * 60)
        print()

    if stream:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        print(resp.choices[0].message.content)


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI()

    print(f"Query:    {args.query!r}")
    print(f"Group ID: {args.group_id}")
    print(f"Method:   {args.search_method}")
    print()

    if args.search_method == "fast-dual":
        sources = await run_fast_dual(
            args.query,
            args.group_id,
            max_hops=args.max_hops,
            max_files=args.max_files,
            llm_model=args.model,
        )
    else:
        # ppr path
        print("Embedding query (e5-base-v2, 'query: ' prefix)...")
        embedding = embed_query(args.query)

        print("Seeding entity nodes...")
        seed_rows = await neo4j_query(
            _VECTOR_SEED_QUERY,
            group_id=args.group_id,
            embedding=embedding,
            limit=20,
        )
        if not seed_rows:
            print("No seed entities found — is the graph uploaded? Is group_id correct?")
            sys.exit(1)

        seed_weights = {r["uuid"]: float(r["score"]) for r in seed_rows}
        print(f"  {len(seed_weights)} seed entities:")
        for r in seed_rows[:5]:
            print(f"    [{r['score']:.3f}] {r['name']}")
        if len(seed_rows) > 5:
            print(f"    … and {len(seed_rows) - 5} more")
        print()

        print(f"Running PPR (top {args.top_nodes} nodes)...")
        top_entity_uuids = await run_ppr(args.group_id, seed_weights, args.top_nodes)
        # Fetch names for display
        name_rows = await neo4j_query(
            "MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid, n.name AS name",
            uuids=top_entity_uuids,
        )
        uuid_to_name = {r["uuid"]: r["name"] for r in name_rows}
        print(f"  {len(top_entity_uuids)} top entities:")
        for uid in top_entity_uuids[:10]:
            print(f"    {uuid_to_name.get(uid, uid[:8] + '…')}")
        if len(top_entity_uuids) > 10:
            print(f"    … and {len(top_entity_uuids) - 10} more")
        print()

        print(f"Collecting source files (max {args.max_files})...")
        sources = await collect_sources(top_entity_uuids, args.group_id, args.max_files)

    if not sources:
        print("No source files found.")
        sys.exit(1)

    print(f"\n  {len(sources)} source files:\n")
    for s in sources:
        name     = Path(s["source_file"]).name
        uri      = s.get("obsidian_uri", "")
        episodes = s.get("episodes", [])
        print(f"  {name}")
        if uri:
            print(f"    {uri}")
        if episodes:
            print(f"    episodes: {', '.join(episodes)}")
    print()

    if args.sources_only:
        return

    print("=" * 60)
    print(f"Answer ({args.model}):")
    print("=" * 60)
    await generate_answer(
        client, args.query, sources, args.model,
        stream=not args.no_stream, show_prompt=args.show_prompt,
    )


if __name__ == "__main__":
    asyncio.run(main())
