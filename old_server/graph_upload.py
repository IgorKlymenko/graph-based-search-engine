"""Upload extracted knowledge to Neo4j with a Saga → Episode → Entity hierarchy.

Graph structure:
    (:Saga {uuid, name, group_id})
      -[:HAS_EPISODE]->
    (:Episode {uuid, name, content, doc_id, group_id})
      -[:MENTIONS]->
    (:Entity {uuid, name, summary, group_id, name_embedding})
      -[:RELATES_TO {uuid, fact, fact_embedding, fact_type, confidence, …}]->
    (:Entity)

    (:Entity)-[:HAS_FACT]->(:Fact {uuid, fact, fact_type, provenance,
                                   confidence, fact_embedding, group_id})
      Node-facts: single-entity facts (definitions with class objects,
      objectless statements, opinions/instructions/predictions, unresolved
      objects). Invisible to PPR — no entity-to-entity edge — but embedded
      and ranked against queries alongside RELATES_TO facts.

The retrieval/ layer queries (:Entity {group_id}) and (:Entity)-[:RELATES_TO]->(:Entity),
plus (:Entity)-[:HAS_FACT]->(:Fact) for fact ranking.
Every node and edge gets the same group_id so the partition filter works.

Usage:
    uv run python graph_upload.py
    uv run python graph_upload.py --saga my-corpus --group-id testing-1
    uv run python graph_upload.py --dry-run
    uv run python graph_upload.py --skip-embeddings   # schema test only

Reads:
    entity_cache.json       episode texts + entity candidates per episode
    resolved_entities.json  canonical entities (name, summary, members)
    relations_cache.json    dict: edges, node_facts, non_facts,
                            rejected_subjects, uncovered_by_episode

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
from typing import NamedTuple
from urllib.parse import quote

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from pipeline.models import unit_key

load_dotenv()

ENTITY_CACHE    = Path("entity_cache.json")
RESOLVED_CACHE  = Path("resolved_entities.json")
RELATIONS_CACHE = Path("relations_cache.json")

# Entity names: OpenAI text-embedding-3-large — no prefixes, no local model to
# load (avoids the sentence-transformers cold start when deploying the server)
ENTITY_EMBED_MODEL = "text-embedding-3-large"
ENTITY_EMBED_DIMS  = 3072
# Previous model: local e5-base-v2, asymmetric ("passage: " here, "query: " at
# search time). Re-upload the graph after switching — stored name_embedding
# vectors and the query embedding must come from the same model.
# ENTITY_EMBED_MODEL = "intfloat/e5-base-v2"
# ENTITY_EMBED_DIMS  = 768

# Fact texts (RELATES_TO descriptions AND Fact nodes): OpenAI, one model so
# both rank in the same space against the query embedding
FACT_EMBED_MODEL = "text-embedding-3-small"
FACT_EMBED_DIMS  = 1536
EMBED_BATCH      = 200
WRITE_BATCH      = 500
# The upload runs last, after hours of extraction, so give the clients a long
# window to ride out a transient failure rather than losing that work.
NEO4J_MAX_RETRY_TIME = 120.0
OPENAI_MAX_RETRIES   = 6

_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
NOW_ISO  = datetime.now(timezone.utc).isoformat()


def stable_uuid(key: str) -> str:
    """Deterministic UUID from a string key — idempotent across re-runs."""
    return str(uuid.uuid5(_UUID_NS, key))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--saga", default=None,
                   help="Saga name — top-level container node (default: the group id)")
    p.add_argument("--group-id", default="testing-1",
                   help="group_id stamped on every node and edge (default: testing-1)")
    p.add_argument("--artifacts-dir", type=Path, default=Path("."),
                   help="Directory holding the cache files (default: repo root)")
    p.add_argument("--vault-path", default=None,
                   help="Absolute path to the Obsidian vault root (auto-detected if omitted)")
    p.add_argument("--vault-name", default=None,
                   help="Obsidian vault name as it appears in the app (auto-detected if omitted)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Skip edges and facts below this confidence (default: 0.0 = keep all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without touching Neo4j")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip embedding calls — stores empty lists (useful for schema tests)")
    p.add_argument("--skip-facts", action="store_true",
                   help="Skip Fact nodes entirely: no node-fact embedding, no "
                        "HAS_FACT upload, no fact_embedding_idx/fact_text indexes")
    return p.parse_args()


# ── cache loading ─────────────────────────────────────────────────────────────

def load_relations_cache(path: Path) -> tuple[list[dict], list[dict]]:
    """Load edges + node_facts. Handles new dict format and legacy list format."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("edges", []), data.get("node_facts", [])
    # legacy: bare list of edges, possibly with unresolved objects mixed in
    print("  ⚠ legacy relations cache (bare list) — splitting on object_resolved")
    edges = [e for e in data if e.get("object_resolved")]
    node_facts = [{
        "fact_id":          e["relation_id"],
        "entity_id":        e["subject_entity_id"],
        "fact_description": e["relation_description"],
        "fact_type":        e.get("fact_type", "assertion"),
        "doc_id":           e["doc_id"],
        "char_start":       e["char_start"],
        "char_end":         e["char_end"],
        "source_quote":     e.get("source_quote", ""),
        "quote_verified":   e.get("quote_verified", False),
        "confidence":       e.get("confidence", 0.5),
        "provenance":       "salvaged_unresolved_edge",
    } for e in data if not e.get("object_resolved")]
    return edges, node_facts


# ── entity identity ───────────────────────────────────────────────────────────

class EntityLookup(NamedTuple):
    """Resolve a canonical name to an entity_id, disambiguated by document.

    Relation extraction records endpoints by canonical NAME, but unit-level
    resolution allows two entities to share a name. Scoping by the edge's doc_id
    separates them. Known limit: two same-named entities inside ONE document
    cannot be told apart here and collapse to the first.
    """

    by_name_doc: dict[tuple[str, str], str]
    by_name: dict[str, list[str]]

    def get(self, name: str, doc_id: str) -> str | None:
        eid = self.by_name_doc.get((name, doc_id))
        if eid is not None:
            return eid
        # Not in this document: only fall back when the name is unambiguous.
        # Guessing among several would silently attach an edge to the wrong
        # entity, where returning None keeps the loss visible in skipped_map.
        ids = self.by_name.get(name) or []
        return ids[0] if len(ids) == 1 else None


def build_entity_lookup(entities: list[dict]) -> EntityLookup:
    by_name_doc: dict[tuple[str, str], str] = {}
    by_name: dict[str, list[str]] = {}
    for e in entities:
        eid = e["entity_id"]
        by_name.setdefault(e["canonical"], []).append(eid)
        for source_file in e.get("source_files", []):
            by_name_doc.setdefault((e["canonical"], source_file), eid)
    return EntityLookup(by_name_doc, by_name)


# ── embeddings ────────────────────────────────────────────────────────────────

async def _embed_batch(
    client: AsyncOpenAI, texts: list[str], model: str = FACT_EMBED_MODEL
) -> list[list[float]]:
    resp = await client.embeddings.create(model=model, input=texts)
    return [r.embedding for r in sorted(resp.data, key=lambda x: x.index)]


async def embed_all(
    client: AsyncOpenAI,
    texts: list[str],
    label: str,
    model: str = FACT_EMBED_MODEL,
) -> list[list[float]]:
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(await _embed_batch(client, texts[i : i + EMBED_BATCH], model))
        print(f"  {label}: {min(i + EMBED_BATCH, len(texts))}/{len(texts)} embedded")
    return out


# ── index creation ────────────────────────────────────────────────────────────

async def create_indexes(
    session: object, dry_run: bool, skip_facts: bool = False
) -> None:
    """Create vector + fulltext indexes that retrieval/ requires.

    entity_name_embedding is dropped and recreated on every run: its dimension
    is tied to whichever embedding model produced n.name_embedding, and
    `CREATE ... IF NOT EXISTS` would silently keep a stale-dimension index
    around after switching models (e.g. OpenAI 1536-dim -> e5 768-dim).

    When skip_facts is set, the Fact-node indexes are not created (no Fact
    nodes exist to index).
    """
    if dry_run:
        print("  [dry-run] DROP INDEX entity_name_embedding")
    else:
        await session.run("DROP INDEX entity_name_embedding IF EXISTS")

    stmts = [
        (
            "entity_name_embedding",
            f"CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS "
            f"FOR (n:Entity) ON (n.name_embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {ENTITY_EMBED_DIMS}, "
            f"`vector.similarity_function`: 'cosine'}}}}",
        ),
        (
            "fact_embedding_idx",
            f"CREATE VECTOR INDEX fact_embedding_idx IF NOT EXISTS "
            f"FOR (f:Fact) ON (f.fact_embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {FACT_EMBED_DIMS}, "
            f"`vector.similarity_function`: 'cosine'}}}}",
        ),
        (
            "edge_name_and_fact",
            "CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS "
            "FOR ()-[e:RELATES_TO]-() ON EACH [e.fact] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
        ),
        (
            "fact_text",
            "CREATE FULLTEXT INDEX fact_text IF NOT EXISTS "
            "FOR (f:Fact) ON EACH [f.fact] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
        ),
    ]
    if skip_facts:
        stmts = [s for s in stmts if s[0] not in {"fact_embedding_idx", "fact_text"}]

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
    print(f"\n[1/6] Saga: {saga_name!r}  group_id={group_id!r}  uuid={saga_uuid[:8]}…")
    if not dry_run:
        async def _tx(tx):
            result = await tx.run(
                _MERGE_SAGA,
                uuid=saga_uuid, name=saga_name,
                group_id=group_id, created_at=NOW_ISO,
            )
            await result.consume()

        await session.execute_write(_tx)  # retried, like the batched writes
    return saga_uuid


# ── Obsidian URI ──────────────────────────────────────────────────────────────

def _detect_vault(source_files: list[str]) -> tuple[Path, str]:
    """Infer vault root and name from the common prefix of source file paths."""
    paths = [Path(f) for f in source_files]
    common = Path(os.path.commonpath([str(p) for p in paths]))
    candidate = common if common.is_dir() else common.parent
    while candidate != candidate.parent:
        if (candidate / ".obsidian").exists():
            return candidate, candidate.name
        candidate = candidate.parent
    root = Path(os.path.commonpath([str(p) for p in paths]))
    root = root if root.is_dir() else root.parent
    return root, root.name


def obsidian_uri(source_file: str, vault_path: Path, vault_name: str) -> str:
    """Build an obsidian://open URI for a source file."""
    try:
        rel = Path(source_file).relative_to(vault_path)
    except ValueError:
        rel = Path(source_file).name  # type: ignore[assignment]
    encoded = quote(str(rel).replace("\\", "/"), safe="/")
    return f"obsidian://open?vault={quote(vault_name)}&file={encoded}"


# ── batched writes ────────────────────────────────────────────────────────────

async def _consume_batch(tx, query: str, rows: list[dict]):
    """Run one batch inside a managed transaction and return its summary."""
    result = await tx.run(query, rows=rows)
    return await result.consume()


async def _run_batched(session, query: str, rows: list[dict]) -> list:
    """Write ``rows`` in batches, retrying transient failures. Returns summaries.

    ``session.run`` does not retry, so a brief Aura hiccup would abort an upload
    that only begins after hours of extraction. ``execute_write`` replays the
    batch on TransientError/ServiceUnavailable/SessionExpired and re-acquires a
    connection; every query routed through here is a MERGE, so replaying a batch
    is idempotent.
    """
    summaries = []
    for i in range(0, len(rows), WRITE_BATCH):
        batch = rows[i : i + WRITE_BATCH]
        summaries.append(await session.execute_write(_consume_batch, query, batch))
    return summaries


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
    candidates: list[dict],
    saga_uuid: str,
    group_id: str,
    vault_path: Path,
    vault_name: str,
    dry_run: bool,
) -> dict[str, str]:
    """Create Episode nodes from entity_cache. Returns {episode_key: episode_uuid}."""
    seen: dict[str, dict] = {}
    for m in candidates:
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
    print(f"\n[2/6] Episodes: {len(rows)} nodes")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['uuid'][:8]}…  {r['name']!r}")
        return {k: v["uuid"] for k, v in seen.items()}

    await _run_batched(session, _MERGE_EPISODE, rows)
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
    """Create Entity nodes. Returns {entity_id: entity_uuid}.

    Keyed on the resolver's entity_id rather than the canonical name: resolution
    works per unit, so two entities may legitimately share a name, and a
    name-keyed uuid would MERGE them back into one node.
    """
    entity_id_to_uuid: dict[str, str] = {}
    rows = []
    for entity, emb in zip(entities, name_embeddings):
        canonical = entity["canonical"]
        reason    = entity.get("reason") or ""
        aliases   = entity.get("member_candidates") or [canonical]
        if reason:
            summary = reason
        elif len(aliases) > 1:
            summary = f"Also known as: {', '.join(aliases)}"
        else:
            summary = canonical

        entity_id = entity["entity_id"]
        eid = stable_uuid(f"entity:{entity_id}")
        entity_id_to_uuid[entity_id] = eid
        rows.append({
            "uuid":           eid,
            "name":           canonical,
            "summary":        summary,
            "group_id":       group_id,
            "name_embedding": emb,
            "created_at":     NOW_ISO,
        })

    print(f"\n[3/6] Entities: {len(rows)} nodes")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['uuid'][:8]}…  {r['name']!r}")
        return entity_id_to_uuid

    created = props = 0
    for s in await _run_batched(session, _MERGE_ENTITY, rows):
        created += s.counters.nodes_created
        props   += s.counters.properties_set
    print(f"  created={created}  properties_set={props}")
    return entity_id_to_uuid


# ── 4. MENTIONS (Episode → Entity) ───────────────────────────────────────────

_MERGE_MENTIONS = """
UNWIND $rows AS row
MATCH (ep:Episode {uuid: row.ep_uuid})
MATCH (n:Entity   {uuid: row.en_uuid})
MERGE (ep)-[:MENTIONS]->(n)
"""


async def upload_mention_edges(
    session: object,
    candidates: list[dict],
    unit_to_entity_id: dict[str, str],
    entity_id_to_uuid: dict[str, str],
    episode_key_to_uuid: dict[str, str],
    dry_run: bool,
) -> None:
    seen: set[tuple[str, str]] = set()
    rows = []
    for m in candidates:
        # per-unit lookup: this occurrence's entity, not the concept's
        entity_id = unit_to_entity_id.get(unit_key(m))
        en_uuid = entity_id_to_uuid.get(entity_id or "")
        ep_uuid = episode_key_to_uuid.get(f"{m['source_file']}|{m['episode_idx']}")
        if not entity_id or not en_uuid or not ep_uuid:
            continue
        pair = (ep_uuid, en_uuid)
        if pair not in seen:
            seen.add(pair)
            rows.append({"ep_uuid": ep_uuid, "en_uuid": en_uuid})

    print(f"\n[4/6] MENTIONS edges: {len(rows)}")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] ep={r['ep_uuid'][:8]}… -MENTIONS-> en={r['en_uuid'][:8]}…")
        return

    created = 0
    for s in await _run_batched(session, _MERGE_MENTIONS, rows):
        created += s.counters.relationships_created
    print(f"  created={created}")


# ── 5. RELATES_TO (Entity → Entity) ──────────────────────────────────────────

_MERGE_RELATES_TO = """
UNWIND $rows AS row
MATCH (src:Entity {uuid: row.src_uuid, group_id: row.group_id})
MATCH (tgt:Entity {uuid: row.tgt_uuid, group_id: row.group_id})
MERGE (src)-[r:RELATES_TO {uuid: row.uuid}]->(tgt)
SET r.fact             = row.fact,
    r.fact_type        = row.fact_type,
    r.confidence       = row.confidence,
    r.quote_verified   = row.quote_verified,
    r.source_quote     = row.source_quote,
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
    lookup: EntityLookup,
    entity_id_to_uuid: dict[str, str],
    fact_embeddings: list[list[float]],
    ep_uuids: dict[str, str],
    min_confidence: float,
    dry_run: bool,
) -> None:
    doc_id_to_ep_uuids: dict[str, list[str]] = {}
    for episode_key, ep_uuid in ep_uuids.items():
        doc_id = episode_key.split("|")[0]
        doc_id_to_ep_uuids.setdefault(doc_id, []).append(ep_uuid)

    rows = []
    skipped_conf = skipped_map = 0
    for edge, emb in zip(edges, fact_embeddings):
        if edge.get("confidence", 1.0) < min_confidence:
            skipped_conf += 1
            continue
        doc_id   = edge["doc_id"]
        src_uuid = entity_id_to_uuid.get(
            lookup.get(edge["subject_entity_id"], doc_id) or "")
        tgt_uuid = entity_id_to_uuid.get(
            lookup.get(edge["object_entity_id"] or "", doc_id) or "")
        if not src_uuid or not tgt_uuid:
            skipped_map += 1
            continue
        rows.append({
            "uuid":           stable_uuid(f"rel:{edge['relation_id']}"),
            "src_uuid":       src_uuid,
            "tgt_uuid":       tgt_uuid,
            "fact":           edge["relation_description"],
            "fact_type":      edge.get("fact_type", "assertion"),
            "confidence":     edge.get("confidence", 1.0),
            "quote_verified": edge.get("quote_verified", False),
            "source_quote":   edge.get("source_quote", ""),
            "group_id":       group_id,
            "created_at":     NOW_ISO,
            "valid_at":       edge.get("valid_from"),
            "invalid_at":     edge.get("valid_to"),
            "episodes":       doc_id_to_ep_uuids.get(edge["doc_id"], []),
            "fact_embedding": emb,
        })

    print(f"\n[5/6] RELATES_TO edges: {len(rows)}  "
          f"(skipped: {skipped_conf} below min-confidence, {skipped_map} no uuid mapping)")
    if skipped_map:
        print("  ⚠ 'no uuid mapping' means an edge endpoint canonical isn't in "
              "resolved_entities.json — caches out of sync, consider re-running resolution")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] {r['src_uuid'][:8]}… -RELATES_TO-> {r['tgt_uuid'][:8]}…")
            print(f"    {r['fact'][:80]!r}")
        return

    created = props = 0
    for s in await _run_batched(session, _MERGE_RELATES_TO, rows):
        created += s.counters.relationships_created
        props   += s.counters.properties_set
    print(f"  created={created}  properties_set={props}")


# ── 6. Fact nodes (Entity → HAS_FACT → Fact) ─────────────────────────────────

_MERGE_FACT = """
UNWIND $rows AS row
MATCH (n:Entity {uuid: row.en_uuid, group_id: row.group_id})
MERGE (f:Fact {uuid: row.uuid})
SET f.fact           = row.fact,
    f.fact_type      = row.fact_type,
    f.provenance     = row.provenance,
    f.confidence     = row.confidence,
    f.quote_verified = row.quote_verified,
    f.source_quote   = row.source_quote,
    f.doc_id         = row.doc_id,
    f.group_id       = row.group_id,
    f.created_at     = row.created_at,
    f.episodes       = row.episodes,
    f.fact_embedding = row.fact_embedding
MERGE (n)-[:HAS_FACT]->(f)
"""


async def upload_node_facts(
    session: object,
    node_facts: list[dict],
    group_id: str,
    lookup: EntityLookup,
    entity_id_to_uuid: dict[str, str],
    fact_embeddings: list[list[float]],
    ep_uuids: dict[str, str],
    min_confidence: float,
    dry_run: bool,
) -> None:
    doc_id_to_ep_uuids: dict[str, list[str]] = {}
    for episode_key, ep_uuid in ep_uuids.items():
        doc_id = episode_key.split("|")[0]
        doc_id_to_ep_uuids.setdefault(doc_id, []).append(ep_uuid)

    rows = []
    skipped_conf = skipped_map = 0
    for nf, emb in zip(node_facts, fact_embeddings):
        if nf.get("confidence", 1.0) < min_confidence:
            skipped_conf += 1
            continue
        en_uuid = entity_id_to_uuid.get(
            lookup.get(nf["entity_id"], nf["doc_id"]) or "")
        if not en_uuid:
            skipped_map += 1
            continue
        rows.append({
            "uuid":           stable_uuid(f"fact:{nf['fact_id']}"),
            "en_uuid":        en_uuid,
            "fact":           nf["fact_description"],
            "fact_type":      nf.get("fact_type", "assertion"),
            "provenance":     nf.get("provenance", "extraction"),
            "confidence":     nf.get("confidence", 1.0),
            "quote_verified": nf.get("quote_verified", False),
            "source_quote":   nf.get("source_quote", ""),
            "doc_id":         nf["doc_id"],
            "group_id":       group_id,
            "created_at":     NOW_ISO,
            "episodes":       doc_id_to_ep_uuids.get(nf["doc_id"], []),
            "fact_embedding": emb,
        })

    print(f"\n[6/6] Fact nodes (HAS_FACT): {len(rows)}  "
          f"(skipped: {skipped_conf} below min-confidence, {skipped_map} no uuid mapping)")
    if dry_run:
        for r in rows[:2]:
            print(f"  [dry-run] en={r['en_uuid'][:8]}… -HAS_FACT-> {r['fact'][:70]!r}")
        return

    created = 0
    for s in await _run_batched(session, _MERGE_FACT, rows):
        created += s.counters.nodes_created
    print(f"  fact nodes created={created}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    entity_cache    = args.artifacts_dir / ENTITY_CACHE
    resolved_cache  = args.artifacts_dir / RESOLVED_CACHE
    relations_cache = args.artifacts_dir / RELATIONS_CACHE
    for path in (entity_cache, resolved_cache, relations_cache):
        if not path.exists():
            sys.exit(f"missing cache: {path}")

    candidates        = json.loads(entity_cache.read_text())["candidates"]
    resolved_data     = json.loads(resolved_cache.read_text())
    entities          = resolved_data["entities"]
    edges, node_facts = load_relations_cache(relations_cache)
    if "unit_to_entity_id" not in resolved_data:
        sys.exit(f"{resolved_cache} predates unit-level resolution — rerun the "
                 f"'resolve' stage with --force")
    unit_to_entity_id = resolved_data["unit_to_entity_id"]
    lookup            = build_entity_lookup(entities)

    group_id = args.group_id
    saga     = args.saga or group_id

    source_files = list({m["source_file"] for m in candidates})
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

    print(f"Saga:       {saga!r}")
    print(f"group_id:   {group_id!r}")
    print(f"Vault:      {vault_path}  (name={vault_name!r})")
    if source_files:
        print(f"URI example: {obsidian_uri(source_files[0], vault_path, vault_name)}")
    print(f"Entities:   {len(entities)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Edges:      {len(edges)}")
    print(f"Node-facts: {len(node_facts)}")
    if args.min_confidence > 0:
        print(f"min-confidence filter: {args.min_confidence}")
    if args.dry_run:
        print("DRY RUN — no Neo4j writes\n")

    # embeddings
    if args.skip_embeddings:
        empty: list[float] = []
        name_embeddings   = [empty] * len(entities)
        edge_embeddings   = [empty] * len(edges)
        nfact_embeddings  = [empty] * len(node_facts)
    else:
        # Embeddings run at the end of a multi-hour pipeline; ride out a blip
        # rather than lose that work to the SDK's default of 2 attempts.
        oai = AsyncOpenAI(max_retries=OPENAI_MAX_RETRIES)
        print(f"\nEmbedding entity names ({ENTITY_EMBED_MODEL})...")
        name_embeddings = await embed_all(
            oai, [e["canonical"] for e in entities], "entity names",
            model=ENTITY_EMBED_MODEL,
        )
        # Previous model: local e5-base-v2 with its "passage: " prefix
        # from services.e5 import embed_e5_sync
        # name_embeddings = embed_e5_sync(
        #     [e["canonical"] for e in entities], is_query=False
        # )
        print("Embedding edge facts...")
        edge_embeddings = await embed_all(
            oai, [e["relation_description"] for e in edges], "edge facts")
        if args.skip_facts:
            nfact_embeddings = []  # node facts are not uploaded
        else:
            print("Embedding node facts...")
            nfact_embeddings = await embed_all(
                oai, [f["fact_description"] for f in node_facts], "node facts")

    # neo4j
    neo4j_uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USERNAME",  "neo4j")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD")
    neo4j_db   = os.environ.get("NEO4J_DATABASE",  "neo4j")
    if not neo4j_pass and not args.dry_run:
        sys.exit(
            "NEO4J_PASSWORD is not set. Refusing to connect with a default "
            "password (set it in .env or the environment, or use --dry-run)."
        )
    print(f"\nNeo4j: {neo4j_uri}  db={neo4j_db!r}")

    if args.dry_run:
        s_uuid = stable_uuid(f"saga:{saga}")
        print(f"\n[1/6] Saga: {saga!r}  group_id={group_id!r}  uuid={s_uuid[:8]}…")
        ep_uuids = await upload_episodes(None, candidates, s_uuid, group_id, vault_path, vault_name, dry_run=True)
        e2u      = await upload_entities(None, entities, group_id, name_embeddings, dry_run=True)
        await upload_mention_edges(None, candidates, unit_to_entity_id, e2u, ep_uuids, dry_run=True)
        await upload_relations(None, edges, group_id, lookup, e2u, edge_embeddings, ep_uuids, args.min_confidence, dry_run=True)
        if args.skip_facts:
            print("\n[6/6] Fact nodes (HAS_FACT): skipped (--skip-facts)")
        else:
            await upload_node_facts(None, node_facts, group_id, lookup, e2u, nfact_embeddings, ep_uuids, args.min_confidence, dry_run=True)
        await create_indexes(None, dry_run=True, skip_facts=args.skip_facts)
        _print_gaps()
        return

    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_pass),
        max_transaction_retry_time=NEO4J_MAX_RETRY_TIME,
    )
    try:
        async with driver.session(database=neo4j_db) as session:
            s_uuid   = await upload_saga(session, saga, group_id, dry_run=False)
            ep_uuids = await upload_episodes(session, candidates, s_uuid, group_id, vault_path, vault_name, dry_run=False)
            e2u      = await upload_entities(session, entities, group_id, name_embeddings, dry_run=False)
            await upload_mention_edges(session, candidates, unit_to_entity_id, e2u, ep_uuids, dry_run=False)
            await upload_relations(session, edges, group_id, lookup, e2u, edge_embeddings, ep_uuids, args.min_confidence, dry_run=False)
            if not args.skip_facts:
                await upload_node_facts(session, node_facts, group_id, lookup, e2u, nfact_embeddings, ep_uuids, args.min_confidence, dry_run=False)
            await create_indexes(session, dry_run=False, skip_facts=args.skip_facts)
    finally:
        await driver.close()

    print("\nUpload complete ✓")
    _print_gaps()


def _print_gaps() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  GAPS — what retrieval/ needs that isn't covered yet             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. RETRIEVAL DOESN'T RANK FACT NODES YET                        ║
║     (:Entity)-[:HAS_FACT]->(:Fact) is now uploaded and indexed   ║
║     (fact_embedding_idx, fact_text), but the retrieval layer     ║
║     only queries RELATES_TO. Extend fact ranking to pull         ║
║     HAS_FACT facts for PPR-selected entities.                    ║
║                                                                  ║
║  2. SCALAR FACTS (facts_cache.json) — not uploaded               ║
║     Headcount, dates, revenue etc. Same Fact-node pattern        ║
║     applies — fold into upload_node_facts or a sibling step.     ║
║                                                                  ║
║  3. ENTITY SUMMARIES                                             ║
║     summary = LLM resolution reason (brief, variable quality).   ║
║     Now that HAS_FACT exists, a summary can be composed from     ║
║     an entity's fact set — cheap dedicated pass.                 ║
║                                                                  ║
║  4. UNCOVERED ENTITIES (uncovered_by_episode in relations cache) ║
║     Not yet turned into candidate-context fallback facts.        ║
║     These entities have no facts at all — the last singleton     ║
║     residue.                                                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    asyncio.run(main())