"""Re-embed Entity.name_embedding for every group_id already in Neo4j.

Use after switching ENTITY_EMBED_MODEL (graph_upload.py): reads entity names
back from the graph — no pipeline artifacts needed — embeds them with the
current model, writes the vectors back, and drops/recreates the
entity_name_embedding vector index with the new dimensions.

Usage:
    uv run python reembed_entities.py            # all groups
    uv run python reembed_entities.py --dry-run  # counts only, no writes
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

load_dotenv()

from graph_upload import (  # noqa: E402  (needs load_dotenv first)
    EMBED_BATCH,
    ENTITY_EMBED_MODEL,
    create_indexes,
    embed_all,
)

_ENTITIES_QUERY = """
MATCH (n:Entity)
RETURN n.uuid AS uuid, n.name AS name, n.group_id AS group_id
ORDER BY group_id, uuid
"""

_WRITE_QUERY = """
UNWIND $rows AS row
MATCH (n:Entity {uuid: row.uuid})
SET n.name_embedding = row.embedding
"""


async def main() -> None:
    """Re-embed all entity names and rebuild the vector index."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Report entity counts per group; no embedding, no writes")
    args = p.parse_args()

    uri  = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pw   = os.environ.get("NEO4J_PASSWORD")
    db   = os.environ.get("NEO4J_DATABASE", "neo4j")
    if not pw:
        sys.exit("NEO4J_PASSWORD is not set (set it in .env or the environment).")

    print(f"Neo4j: {uri}  db={db!r}  model={ENTITY_EMBED_MODEL}")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pw))
    try:
        async with driver.session(database=db) as session:
            result = await session.run(_ENTITIES_QUERY)
            rows = [dict(r) async for r in result]
            if not rows:
                sys.exit("No Entity nodes found — nothing to re-embed.")

            groups: dict[str, int] = {}
            for r in rows:
                groups[r["group_id"]] = groups.get(r["group_id"], 0) + 1
            print(f"{len(rows)} entities across {len(groups)} groups:")
            for gid, count in sorted(groups.items()):
                print(f"  - {gid}: {count}")
            if args.dry_run:
                return

            oai = AsyncOpenAI()
            embeddings = await embed_all(
                oai, [r["name"] or "" for r in rows], "entity names",
                model=ENTITY_EMBED_MODEL,
            )

            print("Writing embeddings back to Neo4j...")
            for i in range(0, len(rows), EMBED_BATCH):
                batch = [
                    {"uuid": r["uuid"], "embedding": e}
                    for r, e in zip(rows[i : i + EMBED_BATCH],
                                    embeddings[i : i + EMBED_BATCH])
                ]
                await session.run(_WRITE_QUERY, rows=batch)
                print(f"  written: {min(i + EMBED_BATCH, len(rows))}/{len(rows)}")

            print("Rebuilding vector index...")
            await create_indexes(session, dry_run=False)
            print("Done — all groups re-embedded.")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
