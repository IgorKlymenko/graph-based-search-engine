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
from src.inject.utils import spans as SP

_MISSING = object()


def _rows_for_chain_surfaces(store: dict,
                             chain_ids: set[str] | None = None) -> list[tuple]:
    """One row per (chain, distinct surface), with mention count and head flag.

    Candidates are per-occurrence; `chain_surfaces` is per distinct surface. The
    collapse happens here rather than in the database because `n_mentions` is
    the one number that cannot be re-derived later -- without offsets or
    per-mention rows, recovering it means re-extracting the episode.

    Only identifying surfaces are rows: a pronoun is how the model linked the
    mentions, not a name, and letting "she" into `surfaces` makes it a seed
    that matches every question. Stage 3 applies the same filter to identity;
    this is the same rule at the write. If the chain's head was a pronoun the
    longest remaining surface takes the flag, as `spans.pick_head` would.

    ``chain_ids`` restricts the rows to those chains (the upload delta).
    """
    grouped: dict[tuple[str, str], dict] = {}
    for candidate in store["candidates"].values():
        if chain_ids is not None and candidate["chain_id"] not in chain_ids:
            continue
        if not SP.is_identifying(candidate["text"]):
            continue
        key = (candidate["chain_id"], candidate["text"])
        row = grouped.setdefault(key, {"n": 0, "is_head": False})
        row["n"] += 1
        row["is_head"] = row["is_head"] or candidate["is_head"]

    # The head names the entity (entity_display_name reads is_head rows), so a
    # chain whose model-chosen head is a pronoun or a judged non-name hands the
    # flag to its longest surface that is a name, as spans.pick_head would.
    names = store.get("names", {})
    by_chain: dict[str, list[str]] = {}
    for chain, surface in grouped:
        by_chain.setdefault(chain, []).append(surface)
    for chain, surfaces in by_chain.items():
        named = [s for s in surfaces if names.get(s, True)] or surfaces
        head = next((s for s in named if grouped[(chain, s)]["is_head"]), None)
        if head is None:
            head = max(named, key=len)
        for surface in surfaces:
            grouped[(chain, surface)]["is_head"] = surface == head
    return [(chain_id, surface, row["is_head"], row["n"])
            for (chain_id, surface), row in grouped.items()]


def plan(store: dict, uploaded: dict | None) -> dict:
    """Rows that differ from the last successful upload; everything when none.

    ``uploaded`` is the snapshot ``upload`` leaves in ``store["uploaded"]``: file
    hashes and liveness, episode liveness, chain entity ids and the surfaces
    sent so far. Comparing against it is what makes a sync cost O(change).
    Measured on a 154-note vault: editing one paragraph changes two episodes,
    and without the plan every file, episode, chain and surface row was re-sent.
    """
    seen = uploaded or {"files": {}, "episodes": {}, "chains": {}, "surfaces": []}
    files = [f for f in store["files"].values()
             if seen["files"].get(f["file_id"])
             != [f["full_text_hash"], f.get("removed_at")]]
    episodes = [e for e in store["episodes"].values()
                if seen["episodes"].get(e["episode_id"], _MISSING)
                != e.get("removed_at")]
    chains = [c for c in store.get("chains", {}).values()
              if seen["chains"].get(c["chain_id"], _MISSING) != c.get("entity_id")]
    chain_surfaces = _rows_for_chain_surfaces(store, {c["chain_id"] for c in chains})
    # A surface row carries the judge's flag, so it is (re)sent when it is new
    # or when its verdict changed -- not merely because one of its chains did.
    known = seen["surfaces"]
    if isinstance(known, list):  # pre-flag snapshots listed surfaces without a verdict
        known = {s: _MISSING for s in known}
    names = store.get("names", {})
    surfaces = sorted(s for s in {s for _, s, _, _ in _rows_for_chain_surfaces(store)}
                      if known.get(s, _MISSING) != names.get(s))
    return {"files": files, "episodes": episodes, "chains": chains,
            "chain_surfaces": chain_surfaces, "surfaces": surfaces}


def snapshot(store: dict) -> dict:
    """What the database holds after ``upload`` -- the next run's baseline."""
    return {
        "files": {f["file_id"]: [f["full_text_hash"], f.get("removed_at")]
                  for f in store["files"].values()},
        "episodes": {e["episode_id"]: e.get("removed_at")
                     for e in store["episodes"].values()},
        "chains": {c["chain_id"]: c.get("entity_id")
                   for c in store.get("chains", {}).values()},
        "surfaces": {s: store.get("names", {}).get(s)
                     for _, s, _, _ in _rows_for_chain_surfaces(store)},
    }


def match_entities(chains: list[dict],
                   existing: dict[str, str]) -> dict[str, str]:
    """Map this run's provisional entity ids onto the ones already in the table.

    Stage 4 names each cluster after its smallest member chain, which is stable
    only while membership is. One chain joining an entity renames it roughly
    1 in n+1 times -- measured: a single added document orphaned an entity on its
    first try. Ids that churn are ids nothing outside can hold.

    So the provisional id is treated as nothing more than a *grouping*: chains
    sharing one are a cluster, and the cluster inherits whichever entity its
    chains already belonged to.

    Four cases, all falling out of one rule -- an existing id goes to the cluster
    holding most of its chains:

      unchanged  every chain still together -> same id, untouched
      merge      one cluster holds chains from two entities -> it keeps the id it
                 took most chains from; the other empties and is swept
      split      two clusters hold one entity's chains -> the larger keeps the
                 id, the smaller is a new entity. Ties break on the smallest
                 chain id, so the outcome does not depend on dict ordering.
      new        a cluster with no prior chains keeps its provisional id

    Splitting is the case with no right answer -- neither half is more "the same
    entity" than the other -- so the rule picks continuity for the bulk of the
    references and accepts that the smaller half is renamed.
    """
    from collections import Counter, defaultdict

    members: dict[str, list[str]] = defaultdict(list)
    for chain in chains:
        if chain.get("entity_id"):
            members[chain["entity_id"]].append(chain["chain_id"])

    claims: dict[str, Counter] = defaultdict(Counter)
    for provisional, chain_ids in members.items():
        for chain_id in chain_ids:
            prior = existing.get(chain_id)
            if prior:
                claims[provisional][prior] += 1

    # An existing id may be carried by at most one cluster, or two clusters
    # would share an identity. Most chains wins; then the smallest chain id.
    winner: dict[str, tuple[int, str, str]] = {}
    for provisional, counter in claims.items():
        tiebreak = min(members[provisional])
        for prior, count in counter.items():
            best = winner.get(prior)
            if best is None or (count, ) > (best[0], ) or (
                    count == best[0] and tiebreak < best[1]):
                winner[prior] = (count, tiebreak, provisional)

    won: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for prior, (count, _tiebreak, provisional) in winner.items():
        won[provisional].append((count, prior))

    # A cluster that won nothing keeps its provisional id: either it is new, or
    # it is the smaller half of a split, and both need an identity of their own.
    return {provisional: (max(candidates)[1] if (candidates := won.get(provisional))
                          else provisional)
            for provisional in members}


def upload(store: dict, *, dsn: str, tenant_id: str, full: bool = False) -> dict:
    """Write what changed since the last upload; everything with ``full``.

    Returns per-table counts of rows sent, plus ``carried`` / ``renamed`` /
    ``emptied`` for the entity ids and ``full`` for which mode ran.
    """
    import psycopg

    baseline = None if full else store.get("uploaded")

    with psycopg.connect(dsn) as conn, conn.transaction():
        cur = conn.cursor()
        cur.execute("SET LOCAL ROLE app_ingest")
        cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))

        # A baseline describes a database that holds the rows it lists. A tenant
        # with no files at all -- a reset, a fresh environment -- does not, and
        # trusting the baseline would leave the corpus missing. Send everything.
        if baseline is not None:
            cur.execute("SELECT count(*) FROM files")
            if cur.fetchone()[0] == 0:
                baseline = None

        # Carry existing entity ids forward before anything is written. Read and
        # write in one transaction so the mapping cannot be computed against a
        # table another run has since changed. Applied to every chain, because
        # the delta is decided on the ids the database will actually hold.
        chains_all = list(store.get("chains", {}).values())
        cur.execute("SELECT id::text, entity_id::text FROM chains"
                    " WHERE entity_id IS NOT NULL")
        mapping = match_entities(chains_all, dict(cur.fetchall()))
        for chain in chains_all:
            if chain.get("entity_id"):
                chain["entity_id"] = mapping[chain["entity_id"]]
        renamed = sum(1 for k, v in mapping.items() if k != v)

        todo = plan(store, baseline)
        files, episodes, chains = todo["files"], todo["episodes"], todo["chains"]
        chain_surfaces, surfaces = todo["chain_surfaces"], todo["surfaces"]
        # A chain that never merged still gets an entity; one deliberately
        # withheld (contested name, nothing to anchor it) keeps entity_id NULL.
        entities = sorted({c["entity_id"] for c in chains if c.get("entity_id")})

        # FK order: files -> episodes -> surfaces -> entities -> chains ->
        # chain_surfaces. Entities before chains because chains.entity_id
        # references them.
        cur.executemany(
            "INSERT INTO files (id, tenant_id, source, external_id, title, url,"
            " full_text_hash, render_version, synced_at, removed_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (id) DO UPDATE SET"
            "   title = EXCLUDED.title,"
            "   url = EXCLUDED.url,"
            "   full_text_hash = EXCLUDED.full_text_hash,"
            "   render_version = EXCLUDED.render_version,"
            "   synced_at = EXCLUDED.synced_at,"
            "   removed_at = EXCLUDED.removed_at",
            [(f["file_id"], tenant_id, f["source"], f["external_id"],
              f.get("title"), f.get("url"),
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
        # is_name is the judge's verdict (NULL = not judged, read as a name);
        # the upsert lets a re-judged surface change its flag in place.
        names = store.get("names", {})
        cur.executemany(
            "INSERT INTO surfaces (tenant_id, surface, is_name) VALUES (%s,%s,%s)"
            " ON CONFLICT (tenant_id, surface) DO UPDATE SET"
            "   is_name = EXCLUDED.is_name",
            [(tenant_id, s, names.get(s)) for s in surfaces])

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

        # The head can move to another surface of an existing chain (a judge
        # verdict changed, a connector named a seed). The upsert below writes
        # the new head before it reaches the old one, and `UNIQUE (tenant_id,
        # chain_id) WHERE is_head` is checked per row, not per statement --
        # measured: the first incremental upload after re-judging failed on
        # exactly that. Clear the flag for every chain in the delta first.
        cur.execute(
            "UPDATE chain_surfaces SET is_head = false"
            " WHERE tenant_id = %s AND is_head AND chain_id = ANY(%s::uuid[])",
            (tenant_id, sorted({chain_id for chain_id, _, _, _ in chain_surfaces})))
        cur.executemany(
            "INSERT INTO chain_surfaces"
            " (tenant_id, chain_id, surface, is_head, n_mentions)"
            " VALUES (%s,%s,%s,%s,%s)"
            " ON CONFLICT (tenant_id, chain_id, surface) DO UPDATE SET"
            "   is_head = EXCLUDED.is_head, n_mentions = EXCLUDED.n_mentions",
            [(tenant_id, chain_id, surface, is_head, n)
             for chain_id, surface, is_head, n in chain_surfaces])

        # Re-clustering moves chains between entities, which leaves rows behind:
        # an entity whose every chain went elsewhere. Nothing else removes them,
        # so without this they accumulate one per rename, forever -- measured,
        # a single added document orphaned one on its first try.
        #
        # Runs last, after every chain has been repointed, or it would delete
        # entities that this run is about to fill.
        #
        # Safe by construction rather than by care: chains.entity_id is
        # ON DELETE RESTRICT, so deleting an entity that still has chains raises
        # instead of silently NULLing them. The NOT EXISTS is what makes the
        # statement legal, not what makes it correct.
        cur.execute(
            "DELETE FROM entities e"
            " WHERE e.tenant_id = %s"
            "   AND NOT EXISTS (SELECT 1 FROM chains c"
            "                    WHERE (c.tenant_id, c.entity_id)"
            "                        = (e.tenant_id, e.id))",
            (tenant_id,))
        emptied = cur.rowcount

    # Only after the commit: a failed transaction must leave the old baseline,
    # so the next run re-sends what this one did not land.
    store["uploaded"] = snapshot(store)
    return {"files": len(files), "episodes": len(episodes),
            "surfaces": len(surfaces), "entities": len(entities),
            "chains": len(chains), "chain_surfaces": len(chain_surfaces),
            "carried": len(mapping) - renamed, "renamed": renamed,
            "emptied": emptied, "full": baseline is None}


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
    parser.add_argument("--full", action="store_true",
                        help="Re-send every row, not the delta since the last upload")
    args = parser.parse_args()

    store = EP.load_store(args.store)
    counts = upload(store, dsn=args.dsn, tenant_id=args.tenant_id,
                    full=args.full)
    EP.save_store(args.store, store)
    print("  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
