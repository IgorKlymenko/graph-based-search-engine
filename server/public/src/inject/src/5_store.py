"""The store's contents into Postgres: files, episodes, surfaces, chains, entities.

The whole upload is **one transaction**. Not for speed — for correctness. A
partial write leaves chains with no surfaces, and `alias_hit` then answers "no"
instead of failing, so every subsequent linking decision is quietly made on
incomplete evidence. There is no error to notice.

Runs as ``app_ingest``: a corpus of files has no workspace to derive a tenant
from, so this is the path that claims one outright. ``SET LOCAL`` for both the
role and the tenant, because a session-level setting outlives the transaction
and the next caller on a pooled connection would inherit it.

Every statement is an upsert keyed the way the schema is keyed, so re-running
over an unchanged corpus is a no-op rather than a duplicate.
"""

from __future__ import annotations

from pathlib import Path

from src.inject import episodes as EP

BATCH = 500


def _rows_for_chain_surfaces(store: dict) -> list[tuple]:
    """One row per (chain, distinct surface), with mention count and head flag.

    Candidates are per-occurrence; `chain_surfaces` is per distinct surface. The
    collapse happens here rather than in the database because `n_mentions` is
    the one number that cannot be re-derived later — without offsets or
    per-mention rows, recovering it means re-extracting the episode.
    """
    grouped: dict[tuple[str, str], dict] = {}
    for candidate in store["candidates"].values():
        key = (candidate["chain_id"], candidate["text"])
        row = grouped.setdefault(key, {"n": 0, "is_head": False})
        row["n"] += 1
        row["is_head"] = row["is_head"] or candidate["is_head"]
    return [(chain_id, surface, row["is_head"], row["n"])
            for (chain_id, surface), row in grouped.items()]


def upload(store: dict, *, dsn: str, tenant_id: str) -> dict:
    """Write everything the pipeline produced. Returns per-table counts."""
    import psycopg

    files = list(store["files"].values())
    episodes = list(store["episodes"].values())
    chains = list(store.get("chains", {}).values())
    chain_surfaces = _rows_for_chain_surfaces(store)
    surfaces = sorted({surface for _, surface, _, _ in chain_surfaces})
    # A chain that never merged still gets an entity; one deliberately withheld
    # (contested name, nothing to anchor it) keeps entity_id NULL.
    entities = sorted({c["entity_id"] for c in chains if c.get("entity_id")})

    with psycopg.connect(dsn) as conn, conn.transaction():
        cur = conn.cursor()
        cur.execute("SET LOCAL ROLE app_ingest")
        cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))

        # FK order: files -> episodes -> surfaces -> entities -> chains ->
        # chain_surfaces. Entities before chains because chains.entity_id
        # references them.
        cur.executemany(
            "INSERT INTO files (id, tenant_id, source, external_id,"
            " full_text_hash, render_version, synced_at, removed_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (id) DO UPDATE SET"
            "   full_text_hash = EXCLUDED.full_text_hash,"
            "   synced_at = EXCLUDED.synced_at,"
            "   removed_at = EXCLUDED.removed_at",
            [(f["file_id"], tenant_id, f["source"], f["external_id"],
              f["full_text_hash"], f["render_version"], f["synced_at"],
              f["removed_at"]) for f in files])

        # Content-addressed: the same id always denotes the same text, so an
        # existing row needs no update. Only liveness moves.
        #
        # created_at is supplied rather than defaulted because an episode can be
        # retired *before* it is ever written here -- a document edited twice
        # between uploads creates and removes an episode with no run in between,
        # and the store carries it through already-retired. Defaulting to now()
        # then puts creation after expiry and trips
        # `CHECK (expired_at IS NULL OR expired_at >= created_at)`.
        # LEAST keeps the honest reading: we learned of it no later than the
        # moment it went away. Computed in SQL so the app's clock never has to
        # agree with the database's.
        cur.executemany(
            "INSERT INTO episodes (id, file_id, tenant_id, text,"
            " created_at, expired_at)"
            " VALUES (%s,%s,%s,%s,"
            "         LEAST(now(), COALESCE(%s::timestamptz, now())), %s)"
            " ON CONFLICT (id) DO UPDATE SET expired_at = EXCLUDED.expired_at",
            [(e["episode_id"], e["file_id"], tenant_id, e["text"],
              e.get("removed_at"), e.get("removed_at")) for e in episodes])

        # Embedding left NULL: it is a different model from the linker's and a
        # different budget, so it is backfilled in batches. The trigram index
        # still serves the lexical half of seeding meanwhile, so retrieval
        # degrades rather than breaking.
        cur.executemany(
            "INSERT INTO surfaces (tenant_id, surface) VALUES (%s,%s)"
            " ON CONFLICT (tenant_id, surface) DO NOTHING",
            [(tenant_id, s) for s in surfaces])

        # display_name stays NULL: the name is derived from members by
        # entity_display_name(). Writing one here would be a cache that
        # re-clustering silently overwrites.
        cur.executemany(
            "INSERT INTO entities (id, tenant_id) VALUES (%s,%s)"
            " ON CONFLICT (tenant_id, id) DO NOTHING",
            [(e, tenant_id) for e in entities])

        # Conflict on the natural key, not the id: an unchanged chain keeps its
        # row -- and with it the entity_id that cost review calls to determine.
        cur.executemany(
            "INSERT INTO chains (id, tenant_id, episode_id, entity_id, content_key)"
            " VALUES (%s,%s,%s,%s,%s)"
            " ON CONFLICT (tenant_id, episode_id, content_key) DO UPDATE SET"
            "   entity_id = EXCLUDED.entity_id",
            [(c["chain_id"], tenant_id, c["episode_id"], c.get("entity_id"),
              c["content_key"]) for c in chains])

        cur.executemany(
            "INSERT INTO chain_surfaces"
            " (tenant_id, chain_id, surface, is_head, n_mentions)"
            " VALUES (%s,%s,%s,%s,%s)"
            " ON CONFLICT (tenant_id, chain_id, surface) DO UPDATE SET"
            "   is_head = EXCLUDED.is_head, n_mentions = EXCLUDED.n_mentions",
            [(tenant_id, chain_id, surface, is_head, n)
             for chain_id, surface, is_head, n in chain_surfaces])

    return {"files": len(files), "episodes": len(episodes),
            "surfaces": len(surfaces), "entities": len(entities),
            "chains": len(chains), "chain_surfaces": len(chain_surfaces)}


def assign_entities(store: dict, assignment: dict[str, str],
                    withheld: set[str]) -> None:
    """Stamp stage 4's clustering onto the chain records, in place.

    A withheld chain keeps ``entity_id`` NULL rather than being dropped: the
    mention happened and its episode still cites it, we simply refuse to invent
    an entity for a name the corpus uses for more than one thing.
    """
    for chain_id, chain in store.get("chains", {}).items():
        chain["entity_id"] = (None if chain_id in withheld
                              else assignment.get(chain_id))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--tenant-id", required=True)
    args = parser.parse_args()

    counts = upload(EP.load_store(args.store), dsn=args.dsn,
                    tenant_id=args.tenant_id)
    print("  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
