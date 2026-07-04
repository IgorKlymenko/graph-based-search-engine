"""Upload extracted knowledge to Neo4j with a Saga → Episode → Entity hierarchy.

Graph structure:
    (:Saga {uuid, name, group_id})
      -[:HAS_EPISODE]->
    (:Episode {uuid, name, content, doc_id, group_id})
      -[:MENTIONS]->
    (:Entity {uuid, name, summary, group_id, name_embedding})
      -[:RELATES_TO {uuid, fact, fact_embedding, …}]->
    (:Entity)

The retrieval/ layer queries (:Entity {group_id}) and (:Entity)-[:RELATES_TO]->(:Entity).
Every node and edge gets the same group_id so the partition filter works.

Usage:
    uv run python graph_upload.py
    uv run python graph_upload.py --saga my-corpus --group-id testing-1
    uv run python graph_upload.py --dry-run
    uv run python graph_upload.py --skip-embeddings   # schema test only

Reads:
    entity_cache.json       episode texts + entity mentions per chunk
    resolved_entities.json  canonical entities (name, summary, members)
    relations_cache.json    RELATES_TO edges (subject, object, description)

Env vars (or .env):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

load_dotenv()

ENTITY_CACHE    = Path("entity_cache.json")
RESOLVED_CACHE  = Path("resolved_entities.json")
RELATIONS_CACHE = Path("relations_cache.json")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS  = 1536
EMBED_BATCH = 200

_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
NOW_ISO  = datetime.now(timezone.utc).isoformat()


def stable_uuid(key: str) -> str:
    """Deterministic UUID from a string key — idempotent across re-runs."""
    return str(uuid.uuid5(_UUID_NS, key))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--saga", default="testing-1",
                   help="Saga name — top-level container node (default: testing-1)")
    p.add_argument("--group-id", default="testing-1",
                   help="group_id stamped on every node and edge (default: testing-1)")
    p.add_argument("--vault-path", default=None,
                   help="Absolute path to the Obsidian vault root (auto-detected if omitted)")
    p.add_argument("--vault-name", default=None,
                   help="Obsidian vault name as it appears in the app (auto-detected if omitted)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without touching Neo4j")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip OpenAI calls — stores empty lists (useful for schema tests)")
    return p.parse_args()


# ── embeddings ────────────────────────────────────────────────────────────────

async def _embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in sorted(resp.data, key=lambda x: x.index)]


async def embed_all(client: AsyncOpenAI, texts: list[str], label: str) -> list[list[float]]:
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(await _embed_batch(client, texts[i : i + EMBED_BATCH]))
        print(f"  {label}: {min(i + EMBED_BATCH, len(texts))}/{len(texts)} embedded")
    return out


# ── index creation ────────────────────────────────────────────────────────────

async def create_indexes(session: object, dry_run: bool) -> None:
    """Create vector + fulltext indexes that retrieval/ requires."""
    stmts = [
        (
            "entity_name_embedding",
            f"CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS "
            f"FOR (n:Entity) ON (n.name_embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIMS}, "
            f"`vector.similarity_function`: 'cosine'}}}}",
        ),
        (
            "edge_name_and_fact",
            "CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS "
            "FOR ()-[e:RELATES_TO]-() ON EACH [e.fact] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
        ),
    ]
    for name, stmt in stmts:
        if dry_run:
            print(f"  [dry-run] CREATE INDEX {name}")
        else:
            await session.run(stmt)
            print(f"  index {name} ✓")


# ── 1. Saga ───────────────────────────────────────────────────────────────────

_MERGE_SAGA = (
    "MERGE (s:Saga {uuid: $uuid}) "
    "SET s.name = $name, s.group_id = $group_id, s.created_at = $created_at"
)


async def upload_saga(
    session: object, saga_name: str, group_id: str, dry_run: bool
) -> str:
    saga_uuid = stable_uuid(f"saga:{saga_name}")
    print(f"\n[1/5] Saga: {saga_name!r}  group_id={group_id!r}  uuid={saga_uuid[:8]}…")
    if not dry_run:
        await session.run(
            _MERGE_SAGA,
            uuid=saga_uuid, name=saga_name,
            group_id=group_id, created_at=NOW_ISO,
        )
    return saga_uuid


# ── Obsidian URI ──────────────────────────────────────────────────────────────

def _detect_vault(source_files: list[str]) -> tuple[Path, str]:
    """Infer vault root and name from the common prefix of source file paths."""
    paths = [Path(f) for f in source_files]
    common = Path(os.path.commonpath([str(p) for p in paths]))
    # Walk up until we find a directory that looks like a vault root
    # (contains .obsidian/) or just use the common ancestor
    candidate = common if common.is_dir() else common.parent
    while candidate != candidate.parent:
        if (candidate / ".obsidian").exists():
            return candidate, candidate.name
        candidate = candidate.parent
    # Fallback: use the common ancestor regardless
    root = Path(os.path.commonpath([str(p) for p in paths]))
    root = root if root.is_dir() else root.parent
    return root, root.name


def obsidian_uri(source_file: str, vault_path: Path, vault_name: str) -> str:
    """Build an obsidian://open URI for a source file."""
    try:
        rel = Path(source_file).relative_to(vault_path)
    except ValueError:
        rel = Path(source_file).name  # type: ignore[assignment]
    # Obsidian expects forward slashes and percent-encoded spaces
    encoded = quote(str(rel).replace("\\", "/"), safe="/")
    return f"obsidian://open?vault={quote(vault_name)}&file={encoded}"


# ── 2. Episodes ───────────────────────────────────────────────────────────────

_MERGE_EPISODE = """
UNWIND $rows AS row
MERGE (ep:Episode {uuid: row.uuid})
SET ep.name        = row.name,
    ep.content     = row.content,
    ep.doc_id      = row.doc_id,
    ep.source      = row.source,
    ep.source_file = row.source_file,
    ep.episode_idx = row.episode_idx,
    ep.group_id    = row.group_id,
    ep.created_at  = row.created_at
WITH ep, row
MATCH (s:Saga {uuid: row.saga_uuid})
MERGE (s)-[:HAS_EPISODE]->(ep)
"""


async def upload_episodes(
    session: object,
    mentions: list[dict],
    saga_uuid: str,
    group_id: str,
    vault_path: Path,
    vault_name: str,
    dry_run: bool,
) -> dict[str, str]:
    """Create Episode nodes from entity_cache. Returns {chunk_key: episode_uuid}."""
    seen: dict[str, dict] = {}
    for m in mentions:
        key = f"{m['source_file']}|{m['episode_idx']}"
        if key in seen:
            continue
        seen[key] = {
            "uuid":        stable_uuid(f"episode:{key}"),
            "saga_uuid":   saga_uuid,
            "name":        f"{Path(m['source_file']).name} / ep {m['episode_idx']}",
            "content":     m["episode"],
            "doc_id":      m["source_file"],
            "source":      obsidian_uri(m["source_file"], vault_path, vault_name),
            "source_file": m["source_file"],
            "episode_idx": m["episode_idx"],
            "group_id":    group_id,
            "created_at":  NOW_ISO,
        }

    rows = list(seen.values())
    print(f"\n[2/5] Episodes: {len(rows)} nodes")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['uuid'][:8]}…  {r['name']!r}")
        return {k: v["uuid"] for k, v in seen.items()}

    for i in range(0, len(rows), 500):
        result = await session.run(_MERGE_EPISODE, rows=rows[i : i + 500])
        await result.consume()
    print(f"  {len(rows)} done")
    return {k: v["uuid"] for k, v in seen.items()}


# ── 3. Entities ───────────────────────────────────────────────────────────────

_MERGE_ENTITY = """
UNWIND $rows AS row
MERGE (n:Entity {uuid: row.uuid})
SET n.name           = row.name,
    n.summary        = row.summary,
    n.group_id       = row.group_id,
    n.name_embedding = row.name_embedding,
    n.created_at     = row.created_at
"""


async def upload_entities(
    session: object,
    entities: list[dict],
    group_id: str,
    name_embeddings: list[list[float]],
    dry_run: bool,
) -> dict[str, str]:
    """Create Entity nodes. Returns {canonical: entity_uuid}."""
    canonical_to_uuid: dict[str, str] = {}
    rows = []
    for entity, emb in zip(entities, name_embeddings):
        canonical = entity["canonical"]
        reason    = entity.get("reason") or ""
        members   = entity.get("members", [canonical])
        if reason:
            summary = reason
        elif len(members) > 1:
            summary = f"Also known as: {', '.join(members)}"
        else:
            summary = canonical

        eid = stable_uuid(f"entity:{canonical}")
        canonical_to_uuid[canonical] = eid
        rows.append({
            "uuid":           eid,
            "name":           canonical,
            "summary":        summary,
            "group_id":       group_id,
            "name_embedding": emb,
            "created_at":     NOW_ISO,
        })

    print(f"\n[3/5] Entities: {len(rows)} nodes")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['uuid'][:8]}…  {r['name']!r}")
        return canonical_to_uuid

    created = props = 0
    for i in range(0, len(rows), 500):
        result = await session.run(_MERGE_ENTITY, rows=rows[i : i + 500])
        s = await result.consume()
        created += s.counters.nodes_created
        props   += s.counters.properties_set
    print(f"  created={created}  properties_set={props}")
    return canonical_to_uuid


# ── 4. MENTIONS (Episode → Entity) ───────────────────────────────────────────

_MERGE_MENTIONS = """
UNWIND $rows AS row
MATCH (ep:Episode {uuid: row.ep_uuid})
MATCH (n:Entity   {uuid: row.en_uuid})
MERGE (ep)-[:MENTIONS]->(n)
"""


async def upload_mentions(
    session: object,
    mentions: list[dict],
    mention_to_canonical: dict[str, str],
    canonical_to_uuid: dict[str, str],
    episode_key_to_uuid: dict[str, str],
    dry_run: bool,
) -> None:
    seen: set[tuple[str, str]] = set()
    rows = []
    for m in mentions:
        canon   = mention_to_canonical.get(m["text"])
        en_uuid = canonical_to_uuid.get(canon or "")
        ep_uuid = episode_key_to_uuid.get(f"{m['source_file']}|{m['episode_idx']}")
        if not canon or not en_uuid or not ep_uuid:
            continue
        pair = (ep_uuid, en_uuid)
        if pair not in seen:
            seen.add(pair)
            rows.append({"ep_uuid": ep_uuid, "en_uuid": en_uuid})

    print(f"\n[4/5] MENTIONS edges: {len(rows)}")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] ep={r['ep_uuid'][:8]}… -MENTIONS-> en={r['en_uuid'][:8]}…")
        return

    created = 0
    for i in range(0, len(rows), 500):
        result = await session.run(_MERGE_MENTIONS, rows=rows[i : i + 500])
        s = await result.consume()
        created += s.counters.relationships_created
    print(f"  created={created}")


# ── 5. RELATES_TO (Entity → Entity) ──────────────────────────────────────────

_MERGE_RELATES_TO = """
UNWIND $rows AS row
MATCH (src:Entity {uuid: row.src_uuid, group_id: row.group_id})
MATCH (tgt:Entity {uuid: row.tgt_uuid, group_id: row.group_id})
MERGE (src)-[r:RELATES_TO {uuid: row.uuid}]->(tgt)
SET r.fact             = row.fact,
    r.source_node_uuid = row.src_uuid,
    r.target_node_uuid = row.tgt_uuid,
    r.group_id         = row.group_id,
    r.created_at       = row.created_at,
    r.valid_at         = row.valid_at,
    r.invalid_at       = row.invalid_at,
    r.expired_at       = null,
    r.episodes         = row.episodes,
    r.fact_embedding   = row.fact_embedding
"""


async def upload_relations(
    session: object,
    edges: list[dict],
    group_id: str,
    canonical_to_uuid: dict[str, str],
    fact_embeddings: list[list[float]],
    ep_uuids: dict[str, str],
    dry_run: bool,
) -> None:
    # Build doc_id → [episode_uuid, ...] from the chunk-key map
    doc_id_to_ep_uuids: dict[str, list[str]] = {}
    for chunk_key, ep_uuid in ep_uuids.items():
        doc_id = chunk_key.split("|")[0]
        doc_id_to_ep_uuids.setdefault(doc_id, []).append(ep_uuid)

    rows = []
    skipped = 0
    for edge, emb in zip(edges, fact_embeddings):
        if not edge.get("object_resolved") or not edge.get("object_entity_id"):
            skipped += 1
            continue
        src_uuid = canonical_to_uuid.get(edge["subject_entity_id"])
        tgt_uuid = canonical_to_uuid.get(edge["object_entity_id"])
        if not src_uuid or not tgt_uuid:
            skipped += 1
            continue
        rows.append({
            "uuid":           stable_uuid(f"rel:{edge['relation_id']}"),
            "src_uuid":       src_uuid,
            "tgt_uuid":       tgt_uuid,
            "fact":           edge["relation_description"],
            "group_id":       group_id,
            "created_at":     NOW_ISO,
            "valid_at":       edge.get("valid_from"),
            "invalid_at":     edge.get("valid_to"),
            "episodes":       doc_id_to_ep_uuids.get(edge["doc_id"], []),
            "fact_embedding": emb,
        })

    print(f"\n[5/5] RELATES_TO edges: {len(rows)}  ({skipped} skipped — unresolved object)")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['src_uuid'][:8]}… -RELATES_TO-> {r['tgt_uuid'][:8]}…")
            print(f"    {r['fact'][:80]!r}")
        return

    created = props = 0
    for i in range(0, len(rows), 500):
        result = await session.run(_MERGE_RELATES_TO, rows=rows[i : i + 500])
        s = await result.consume()
        created += s.counters.relationships_created
        props   += s.counters.properties_set
    print(f"  created={created}  properties_set={props}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    for path in (ENTITY_CACHE, RESOLVED_CACHE, RELATIONS_CACHE):
        if not path.exists():
            sys.exit(f"missing cache: {path}")

    mentions         = json.loads(ENTITY_CACHE.read_text())["mentions"]
    resolved_data    = json.loads(RESOLVED_CACHE.read_text())
    entities         = resolved_data["entities"]
    mention_to_canon = resolved_data["mention_to_canonical"]
    edges            = json.loads(RELATIONS_CACHE.read_text())

    saga     = args.saga
    group_id = args.group_id

    # Resolve vault path/name — CLI args override auto-detection
    source_files = list({m["source_file"] for m in mentions})
    if args.vault_path and args.vault_name:
        vault_path = Path(args.vault_path)
        vault_name = args.vault_name
    elif args.vault_path:
        vault_path = Path(args.vault_path)
        vault_name = vault_path.name
    else:
        vault_path, vault_name = _detect_vault(source_files)
        if args.vault_name:
            vault_name = args.vault_name

    print(f"Saga:      {saga!r}")
    print(f"group_id:  {group_id!r}")
    print(f"Vault:     {vault_path}  (name={vault_name!r})")
    # Show one example URI so the user can verify it opens correctly
    if source_files:
        example = obsidian_uri(source_files[0], vault_path, vault_name)
        print(f"URI example: {example}")
    print(f"Entities:  {len(entities)}")
    print(f"Mentions:  {len(mentions)}")
    print(f"Relations: {len(edges)}  ({sum(1 for e in edges if e.get('object_resolved'))} resolved)")
    if args.dry_run:
        print("DRY RUN — no Neo4j writes\n")

    # embeddings
    if args.skip_embeddings:
        empty: list[float] = []
        name_embeddings = [empty] * len(entities)
        fact_embeddings = [empty] * len(edges)
    else:
        oai = AsyncOpenAI()
        print("\nEmbedding entity names...")
        name_embeddings = await embed_all(oai, [e["canonical"] for e in entities], "entities")
        print("Embedding relation descriptions...")
        fact_embeddings = await embed_all(oai, [e["relation_description"] for e in edges], "relations")

    # neo4j
    neo4j_uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USERNAME",  "neo4j")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD",  "password")
    neo4j_db   = os.environ.get("NEO4J_DATABASE",  "neo4j")
    print(f"\nNeo4j: {neo4j_uri}  db={neo4j_db!r}")

    if args.dry_run:
        s_uuid = stable_uuid(f"saga:{saga}")
        print(f"\n[1/5] Saga: {saga!r}  group_id={group_id!r}  uuid={s_uuid[:8]}…")
        ep_uuids = await upload_episodes(None, mentions, s_uuid, group_id, vault_path, vault_name, dry_run=True)
        c2u      = await upload_entities(None, entities, group_id, name_embeddings, dry_run=True)
        await upload_mentions(None, mentions, mention_to_canon, c2u, ep_uuids, dry_run=True)
        await upload_relations(None, edges, group_id, c2u, fact_embeddings, ep_uuids, dry_run=True)
        await create_indexes(None, dry_run=True)
        _print_gaps()
        return

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    try:
        async with driver.session(database=neo4j_db) as session:
            s_uuid   = await upload_saga(session, saga, group_id, dry_run=False)
            ep_uuids = await upload_episodes(session, mentions, s_uuid, group_id, vault_path, vault_name, dry_run=False)
            c2u      = await upload_entities(session, entities, group_id, name_embeddings, dry_run=False)
            await upload_mentions(session, mentions, mention_to_canon, c2u, ep_uuids, dry_run=False)
            await upload_relations(session, edges, group_id, c2u, fact_embeddings, ep_uuids, dry_run=False)
            await create_indexes(session, dry_run=False)
    finally:
        await driver.close()

    print("\nUpload complete ✓")
    _print_gaps()


def _print_gaps() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  GAPS — what retrieval/ needs that isn't indexed yet             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. UNRESOLVED RELATION OBJECTS (~36% of edges skipped)          ║
║     172 of 481 edges skipped — object not in mention_to_canonical║
║     These are real connections PPR can't walk.                   ║
║     → Re-run entity extraction with broader entity types,        ║
║       then re-run relation_extraction.ipynb.                     ║
║                                                                  ║
║  2. SCALAR FACTS (facts_cache.json) — not uploaded               ║
║     Headcount, dates, revenue etc. have no second entity.        ║
║     They can't go on RELATES_TO edges — invisible to PPR.        ║
║     → Store as properties on Entity nodes if you need them.      ║
║                                                                  ║
║  3. ENTITY SUMMARIES                                             ║
║     summary = LLM resolution reason (brief, variable quality).   ║
║     → A dedicated per-entity summarisation pass would help.      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    asyncio.run(main())
