"""Seed from surfaces, walk with PPR, collect passages -- inside the tenant wall.

    uv run python -m src.query.retrieve embed --tenant <uuid>
    uv run python -m src.query.retrieve ask --tenant <uuid> "your question"
    uv run python -m src.query.retrieve ask --workspace T0123 "your question"   # as the Slack worker
    uv run python -m src.query.retrieve search --tenant <uuid> --text 10 "q"   # passages only, for agents

``embed`` backfills ``surfaces.embedding`` (names only) and needs re-running
after new documents land. ``ask`` costs one embedding call plus one completion.

The retrieval path is the one the linker was built for:

    question -> nearest surfaces -> their entities -> PPR over co-occurrence
             -> the episodes those entities appear in -> stance + answer

Seeding from *surfaces* rather than entity names is what makes a vague question
work: two chains that say "repeated slack questions" and "the same questions"
are correctly NOT merged, but both are doors into the same neighbourhood.

Two ways in, matching the two roles in CLAUDE.md § Database access: a tenant
claimed outright (``app_ingest`` -- uploads and the CLI) or a workspace
declared and resolved by the database (``app_connector`` -- the Slack worker).
Either way it is one transaction with SET LOCAL, so nothing outlives the
request. Supersedes ``old_server/entity_linking/ask.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from src.inject._stages import load
from src.query import answer as AN

if TYPE_CHECKING:  # annotations only; the walk imports without a driver
    import psycopg

CH = load("2_chains", "src.inject.chains")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1536
# Restart probability. 0.15 is the PageRank convention and it is wrong here:
# measured on a 144-note vault, alpha 0.15 leaves the top 20 entities holding
# 18% of the mass -- the walk diffuses across 3346 of 3354 entities and every
# episode ties. At 0.5 the top 20 hold 53%. Retrieval wants to stay near the
# seeds; ranking pages does not. On the synthetic rig ~0.2 is best; the two
# rigs disagree, so re-measure per corpus (wiki/experiments.md).
ALPHA = 0.50
# Drop pairs that co-occur only once. On that same vault 94.8% of pairs co-occur
# exactly once, and for a once-seen pair PMI collapses to log(n_episodes) -- the
# maximum -- so 75k of 79k edges carried an identical weight.
MIN_CO_OCCURRENCE = 2
ITERATIONS = 30
BATCH = 256
# Relative score cutoff instead of a fixed k. Measured on the synthetic rig:
# keeping walked entities scoring at least this fraction of the best non-seed
# returns three or four of them at precision 0.96 against 0.34 for a fixed
# top-10, and still reaches every bridged pair. Coupled to ALPHA: at 0.5 a
# ratio of 0.3 loses half the bridged pairs and 0.2 keeps them all.
CUTOFF_RATIO = 0.2


def enter(
    cur: psycopg.Cursor,
    *,
    tenant: str | None = None,
    connector: str | None = None,
    workspace: str | None = None,
) -> None:
    """Step inside the tenant wall. SET LOCAL, so it dies with the transaction.

    Exactly one of a claimed tenant (``app_ingest``) or a declared workspace
    (``app_connector``); the database resolves the latter and denies an
    unknown or uninstalled one.
    """
    if tenant and not (connector or workspace):
        cur.execute("SET LOCAL ROLE app_ingest")
        cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    elif connector and workspace and not tenant:
        cur.execute("SET LOCAL ROLE app_connector")
        cur.execute("SELECT set_config('app.connector', %s, true)", (connector,))
        cur.execute("SELECT set_config('app.workspace_id', %s, true)", (workspace,))
    else:
        raise ValueError("give either tenant= or connector= and workspace=")
    # The login roles' search_path is "$user", public -- pgvector's type and the
    # <=> operator live in `extensions`, so an unqualified ::vector fails with
    # `type "vector" does not exist` even with the USAGE grant in place.
    # Transaction-scoped, like everything else here; the resolver functions pin
    # their own empty search_path and are unaffected.
    cur.execute("SET LOCAL search_path = public, extensions")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """One OpenAI call per batch. Dimensions pinned to match the column."""
    client = CH._get_client()
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        response = client.embeddings.create(
            model=EMBED_MODEL, input=chunk, dimensions=EMBED_DIMS
        )
        out.extend(item.embedding for item in response.data)
    return out


def backfill_embeddings(dsn: str, *, tenant: str, chunk: int = BATCH) -> int:
    """Embed every name surface that has no vector yet; returns how many.

    Three separate transactions, never one: the login roles carry a 10-second
    idle-in-transaction ceiling, and the OpenAI calls for a few thousand
    surfaces take a minute. Read the pending list, embed with no connection
    held, then write in chunks -- each write is its own walled transaction.
    Measured: the single-transaction version was terminated by the server
    twice in a row.
    """
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.transaction():
            cur = conn.cursor()
            enter(cur, tenant=tenant)
            # Non-names never seed, so embedding them is spend for nothing.
            # NULL is "not judged" and is read as a name.
            cur.execute(
                "SELECT surface FROM public.surfaces"
                " WHERE embedding IS NULL AND coalesce(is_name, true)"
                " ORDER BY surface"
            )
            pending = [row[0] for row in cur.fetchall()]
        if not pending:
            return 0
        vectors = embed_texts(pending)
        rows = [
            (str(v), f"{EMBED_MODEL}@{EMBED_DIMS}", s) for s, v in zip(pending, vectors)
        ]
        for i in range(0, len(rows), chunk):
            with conn.transaction():
                cur = conn.cursor()
                enter(cur, tenant=tenant)
                cur.executemany(
                    "UPDATE public.surfaces SET embedding = %s::vector, embed_model = %s,"
                    "       updated_at = now()"
                    " WHERE surface = %s",
                    rows[i : i + chunk],
                )
    return len(pending)


RRF_K = 60  # the usual reciprocal-rank-fusion constant
LEXICAL_MIN_CHARS = 3
PER_FILE = 2  # passages per source document, so one note cannot fill the prompt
# Episodes the BM25 text channel adds before passage fusion; 0 = graph only.
# Measured on 28 agent questions (wiki/experiments.md § Agents and tokens):
# graph-only search_8 sufficient 14/28, fused hybrid_8 24/28, no question lost,
# +0.05 s per query. ADR tables 31.
TEXT_CANDIDATES = 10
BM25_K1, BM25_B = 1.2, 0.75


def lexical_hits(question: str, surfaces: list[str]) -> list[str]:
    """Name surfaces that occur verbatim in the question, longest first.

    The lexical half of seeding (ADR tables 29): for short strings exact match
    beats cosine outright -- "ACID" should hit "ACID" at rank 1 without
    depending on an embedding. Word-bounded, case-insensitive; measured missing
    on the live path, where the ACID question retrieved eight tied passages
    and not the note that answers it.
    """
    text = question.lower()
    found = []
    for surface in surfaces:
        needle = surface.strip().lower()
        if len(needle) < LEXICAL_MIN_CHARS:
            continue
        if re.search(r"(?<![\w])" + re.escape(needle) + r"(?![\w])", text):
            found.append(surface)
    return sorted(set(found), key=lambda s: (-len(s), s))


def fuse(
    vector_hits: list[tuple[str, float]],
    lexical: list[str],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of the two surface rankings; best first."""
    score: dict[str, float] = {}
    for rank, (surface, _) in enumerate(vector_hits):
        score[surface] = score.get(surface, 0.0) + 1.0 / (k + rank)
    for rank, surface in enumerate(lexical):
        score[surface] = score.get(surface, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def seed_entities(
    cur: psycopg.Cursor, vector: list[float], question: str, *, k: int, seeds: int
) -> tuple[dict[str, float], list[tuple[str, float]], dict[str, list[str]]]:
    """Nearest surfaces by vector and by exact mention, fused, collapsed onto entities.

    Rank, not raw distance, sets the seed weight: cosine values from one
    embedding model are not calibrated across queries. Also returns the fused
    surface hits (for --explain) and the matched surfaces that belong to more
    than one entity, with those entities' display names -- the signal the
    answer step uses to ask "which one?" (ADR slack 15).
    """
    cur.execute(
        "SELECT surface, 1 - (embedding <=> %s::vector) AS score"
        "  FROM public.surfaces"
        " WHERE embedding IS NOT NULL AND coalesce(is_name, true)"
        " ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(vector), str(vector), k),
    )
    vector_hits = [(surface, float(score)) for surface, score in cur.fetchall()]
    # The lexical half. Bounded scan of name surfaces; at a few hundred thousand
    # rows move this onto the pg_trgm index instead of into Python.
    cur.execute(
        "SELECT surface FROM public.surfaces WHERE coalesce(is_name, true) LIMIT 50000"
    )
    lexical = lexical_hits(question, [row[0] for row in cur.fetchall()])
    fused = fuse(vector_hits, lexical)[:k]
    if not fused:
        return {}, [], {}

    cur.execute(
        "SELECT cs.surface, c.entity_id"
        "  FROM public.chain_surfaces cs"
        "  JOIN public.chains c ON (c.tenant_id, c.id) = (cs.tenant_id, cs.chain_id)"
        " WHERE cs.surface = ANY(%s) AND c.entity_id IS NOT NULL",
        ([surface for surface, _ in fused],),
    )
    holders: dict[str, list[str]] = {}
    for surface, entity in cur.fetchall():
        # str(): psycopg returns uuid.UUID while the graph code stringifies ids
        if str(entity) not in holders.setdefault(surface, []):
            holders[surface].append(str(entity))

    ordered: list[str] = []
    for surface, _ in fused:
        for entity in holders.get(surface, []):
            if entity not in ordered:
                ordered.append(entity)
    contested_ids = {s: e for s, e in holders.items() if len(e) > 1}
    contested: dict[str, list[str]] = {}
    if contested_ids:
        wanted = sorted({e for ids in contested_ids.values() for e in ids})
        cur.execute(
            "SELECT id::text, public.entity_display_name(tenant_id, id)"
            "  FROM public.entities WHERE id = ANY(%s::uuid[])",
            (wanted,),
        )
        names = dict(cur.fetchall())
        contested = {
            s: [names.get(e) or e for e in ids] for s, ids in contested_ids.items()
        }
    hits = [(surface, score) for surface, score in fused if surface in holders][:seeds]
    return (
        {e: 1.0 / (rank + 1) for rank, e in enumerate(ordered[:seeds])},
        hits,
        contested,
    )


def build_graph(
    cur: psycopg.Cursor,
    *,
    min_co: int = MIN_CO_OCCURRENCE,
) -> dict[tuple[str, str], float]:
    """Entity co-occurrence within an episode, weighted by PMI.

    Raw counts and binary weights were both measured and both fail: an entity
    that appears beside everything gets the same weight as a real association,
    and the walk drowns. PMI asks whether two entities co-occur more than
    chance. Pairs at or below chance are dropped rather than down-weighted.
    """
    cur.execute(
        "SELECT episode_id, entity_id FROM public.chains WHERE entity_id IS NOT NULL"
    )
    by_episode: dict[str, set[str]] = defaultdict(set)
    for episode, entity in cur.fetchall():
        by_episode[str(episode)].add(str(entity))

    pairs: Counter = Counter()
    appears: Counter = Counter()
    for entities in by_episode.values():
        ordered = sorted(entities)
        for entity in ordered:
            appears[entity] += 1
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pairs[(a, b)] += 1

    total = max(len(by_episode), 1)
    edges: dict[tuple[str, str], float] = {}
    for (a, b), count in pairs.items():
        if count < min_co:
            continue
        expected = appears[a] * appears[b] / total
        weight = math.log(max(count, 1e-9) / max(expected, 1e-9))
        if weight > 0:
            edges[(a, b)] = weight
    return edges


def ppr(
    edges: dict[tuple[str, str], float],
    seeds: dict[str, float],
    *,
    alpha: float = ALPHA,
    iterations: int = ITERATIONS,
) -> dict[str, float]:
    """Power iteration with restart over the undirected co-occurrence graph.

    Sparse matrices rather than Python loops: measured identical scores at 14x
    the speed (1.6 s -> 0.12 s on 3,400 entities and 79k edges).
    """
    import scipy.sparse as sp

    nodes = sorted({n for pair in edges for n in pair} | set(seeds))
    index = {n: i for i, n in enumerate(nodes)}
    restart = np.zeros(len(nodes))
    for entity, weight in seeds.items():
        if entity in index:
            restart[index[entity]] = weight
    if restart.sum() == 0:
        return {}
    restart /= restart.sum()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for (a, b), w in edges.items():
        i, j = index[a], index[b]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    weights = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    out = np.asarray(weights.sum(axis=1)).ravel()
    share = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
    spread = (sp.diags(share) @ weights).T.tocsr()

    score = restart.copy()
    for _ in range(iterations):
        score = alpha * restart + (1 - alpha) * (spread @ score)
    return {nodes[i]: float(s) for i, s in enumerate(score) if s > 0}


def prune(
    ranked: dict[str, float], seeds: dict[str, float], *, ratio: float = CUTOFF_RATIO
) -> dict[str, float]:
    """Drop the tail of the walk: everything under ``ratio`` of the best non-seed.

    Seeds are always kept -- they matched the question directly. ``ratio`` 0
    keeps everything, which is the fixed-k behaviour this replaces.
    """
    if ratio <= 0:
        return dict(ranked)
    others = [score for entity, score in ranked.items() if entity not in seeds]
    if not others:
        return dict(ranked)
    floor = ratio * max(others)
    return {
        entity: score
        for entity, score in ranked.items()
        if entity in seeds or score >= floor
    }


# ── the text channel ──────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[A-Za-z0-9_.]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def text_tokens(text: str) -> list[str]:
    """Words and identifiers, plus the parts of snake_case and CamelCase names.

    ``keys_settle`` yields ``keys_settle``, ``keys`` and ``settle``, so a
    question that names the symbol matches its definition and one that
    paraphrases it still reaches the docstring.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        token = raw.strip(".")
        if len(token) < 2:
            continue
        out.append(token.lower())
        parts = [
            p for p in _CAMEL.sub("_", token).replace(".", "_").split("_") if len(p) > 1
        ]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


def bm25(
    query: list[str], docs: list[list[str]], *, k1: float = BM25_K1, b: float = BM25_B
) -> list[float]:
    """Okapi BM25 of one query against tokenised documents, in document order."""
    n = len(docs)
    if not n:
        return []
    avg = sum(len(d) for d in docs) / n
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query)}
    scores = []
    for d in docs:
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avg)
        scores.append(
            sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in idf if tf[t])
        )
    return scores


def text_hits(cur: psycopg.Cursor, question: str, *, limit: int) -> list[str]:
    """Episode ids by BM25 over the tenant's live episodes, best first.

    The other half of retrieval for what the graph cannot reach: a docstring
    continuation that names no symbol, a note whose entities the question does
    not mention. Scanned in Python because a tenant is a few thousand episodes;
    past that this is a ``tsvector`` GIN index and ``ts_rank``, same contract.
    """
    cur.execute("SELECT id::text, text FROM public.episodes WHERE expired_at IS NULL")
    rows = cur.fetchall()
    query = text_tokens(question)
    scores = bm25(query, [text_tokens(text) for _, text in rows])
    ranked = sorted(zip(scores, (eid for eid, _ in rows)), reverse=True)
    return [eid for score, eid in ranked[:limit] if score > 0]


def fuse_ranks(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fusion of ranked id lists; best first."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: (-score[i], i))


def collect_episodes(
    cur: psycopg.Cursor,
    ranked: dict[str, float],
    *,
    limit: int,
    per_file: int = PER_FILE,
    text_ids: list[str] | None = None,
) -> list[tuple[str, str, float, str | None]]:
    """Score each episode by the mean score of its ranked entities; take the top few.

    ``text_ids`` is the text channel's ranking of episodes; when given, the two
    orders are fused by reciprocal rank before the top few are taken.

    Mean, not sum and not max: `max` ties every episode holding the top entity,
    `sum` rewards long notes. Measured on the vault with "who do we sell to?":
    the answering notes ranked 25th under sum, 10th under max, 7th under mean.
    Ties on the mean go to the episode holding more of the ranked entities,
    then the larger sum; at most ``per_file`` episodes per source document,
    because one chatty note otherwise fills the prompt (measured: four of eight
    passages from one file on the live path).
    """
    if not ranked and not text_ids:
        return []
    scored, order = rank_episodes(cur, ranked)
    if text_ids:
        order = fuse_ranks(order, text_ids)
        scored = {key: score for key, score in zip(order, _rrf_scores(len(order)))}
    return take_passages(cur, order, scored, limit=limit, per_file=per_file)


def rank_episodes(
    cur: psycopg.Cursor, ranked: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Episode ids scored by the mean of their ranked entities; (scores, order)."""
    if not ranked:
        return {}, []
    cur.execute(
        "SELECT c.episode_id, c.entity_id"
        "  FROM public.chains c"
        "  JOIN public.episodes e ON (e.tenant_id, e.id) = (c.tenant_id, c.episode_id)"
        " WHERE c.entity_id = ANY(%s::uuid[]) AND e.expired_at IS NULL",
        (list(ranked),),
    )
    totals: dict[str, float] = defaultdict(float)
    counts: Counter = Counter()
    for episode, entity in cur.fetchall():
        totals[str(episode)] += ranked.get(str(entity), 0.0)
        counts[str(episode)] += 1
    scored = {k: totals[k] / counts[k] for k in totals}
    order = sorted(scored, key=lambda k: (-scored[k], -counts[k], -totals[k], k))
    return scored, order


def _rrf_scores(n: int, k: int = RRF_K) -> list[float]:
    return [1.0 / (k + rank) for rank in range(n)]


def take_passages(
    cur: psycopg.Cursor,
    order: list[str],
    scored: dict[str, float],
    *,
    limit: int,
    per_file: int,
) -> list[tuple[str, str, float, str | None]]:
    """The top episodes in ``order`` as passages, at most ``per_file`` per document."""
    cur.execute(
        # external_id when there is no title: for the upload connector that is
        # the note's path, which is what a citation wants. Slack threads carry
        # a title.
        "SELECT e.id, e.text, coalesce(f.title, f.external_id), f.id, f.url"
        "  FROM public.episodes e"
        "  JOIN public.files f ON (f.tenant_id, f.id) = (e.tenant_id, e.file_id)"
        " WHERE e.id = ANY(%s::uuid[])",
        (order,),
    )
    seen = {
        str(eid): (text, title or "untitled", str(file_id), url)
        for eid, text, title, file_id, url in cur.fetchall()
    }
    out: list[tuple[str, str, float, str | None]] = []
    per_source: Counter = Counter()
    for key in order:
        if key not in seen:
            continue
        text, title, file_id, url = seen[key]
        if per_source[file_id] >= per_file:
            continue
        per_source[file_id] += 1
        out.append((title, text, scored[key], url))
        if len(out) == limit:
            break
    return out


class Retrieval(NamedTuple):
    """What one question pulled out of the graph, for the answer step and for --explain."""

    passages: list[tuple[str, str, float, str | None]]  # (title, text, score, url)
    contested: dict[str, list[str]]
    hits: list[tuple[str, float]]
    edges: int
    reached: int
    kept: int


def retrieve(
    cur: psycopg.Cursor,
    vector: list[float],
    question: str,
    *,
    k: int = 40,
    seeds: int = 8,
    passages: int = 8,
    alpha: float = ALPHA,
    cutoff: float = CUTOFF_RATIO,
    min_co: int = MIN_CO_OCCURRENCE,
    use_ppr: bool = True,
    text: int = TEXT_CANDIDATES,
) -> Retrieval:
    """Seeds -> walk -> cutoff -> passages, inside an already-entered transaction.

    Takes the question's vector, not the question: the embedding call happens
    before the transaction opens, so no network wait sits inside the 10-second
    idle-in-transaction ceiling the login roles carry.
    """
    seed_weights, hits, contested = seed_entities(
        cur, vector, question, k=k, seeds=seeds
    )
    text_ids = text_hits(cur, question, limit=text) if text else []
    if not seed_weights and not text_ids:
        return Retrieval([], contested, hits, 0, 0, 0)
    edges = build_graph(cur, min_co=min_co) if seed_weights else {}
    ranked = ppr(edges, seed_weights, alpha=alpha) if use_ppr else seed_weights
    reached = len(ranked)
    ranked = prune(ranked, seed_weights, ratio=cutoff) if ranked else {}
    return Retrieval(
        collect_episodes(cur, ranked, limit=passages, text_ids=text_ids),
        contested,
        hits,
        len(edges),
        reached,
        len(ranked),
    )


def search_passages(
    question: str,
    *,
    dsn: str,
    tenant: str | None = None,
    connector: str | None = None,
    workspace: str | None = None,
    **knobs,
) -> Retrieval:
    """Retrieval alone, for a tool an agent calls: the embedding, one walled read."""
    import psycopg

    vector = embed_texts([question])[0]  # outside the transaction: see retrieve()
    with psycopg.connect(dsn) as conn, conn.transaction():
        cur = conn.cursor()
        enter(cur, tenant=tenant, connector=connector, workspace=workspace)
        return retrieve(cur, vector, question, **knobs)


def format_passages(found: Retrieval) -> str:
    """Ranked passages with readable sources, the way a search tool returns them.

    Measured (``scripts/eval_agent_tokens.py``): eight fused passages are about
    1,900 tokens and answer more questions than grep-and-read at 16,000.
    """
    names = AN.display_titles([t for t, _, _, _ in found.passages])
    parts = [
        f"[{i}] {names[title]} (score {score:.2f})\n{text}"
        for i, (title, text, score, _) in enumerate(found.passages, 1)
    ]
    if found.contested:
        line = "; ".join(f"{k}: {', '.join(v)}" for k, v in found.contested.items())
        parts.append(f"Names that mean more than one thing here: {line}")
    return "\n\n".join(parts) if parts else "Nothing indexed matches this question."


def answer_question(
    question: str,
    *,
    dsn: str,
    tenant: str | None = None,
    connector: str | None = None,
    workspace: str | None = None,
    model: str = AN.ANSWER_MODEL,
    allow_reframe: bool = True,
    conversation: list | None = None,
    latest: str | None = None,
    needs_lookup: bool = True,
    **knobs,
) -> dict | None:
    """The whole query path for one question; ``None`` when nothing matched.

    Retrieval runs in one walled transaction; the completion runs after it, so
    the connection is not held open across the model call. In a thread,
    ``question`` is the condensed standalone form, ``conversation`` the earlier
    turns and ``latest`` the message as written; with ``needs_lookup`` false no
    retrieval runs and the answer comes from the conversation alone. When a
    conversation exists, an empty retrieval still yields an answer from it
    rather than ``None``.
    """
    import psycopg

    found = None
    if needs_lookup:
        vector = embed_texts([question])[0]  # outside the transaction: see retrieve()
        with psycopg.connect(dsn) as conn, conn.transaction():
            cur = conn.cursor()
            enter(cur, tenant=tenant, connector=connector, workspace=workspace)
            found = retrieve(cur, vector, question, **knobs)
        if not found.passages and not conversation:
            return None
    passages = found.passages if found else []
    names = AN.display_titles([title for title, _, _, _ in passages])
    verdict = AN.answer(
        question,
        [(names[title], text) for title, text, _, _ in passages],
        contested=found.contested if found else None,
        model=model,
        allow_reframe=allow_reframe,
        conversation=conversation,
        latest=latest,
    )
    verdict["retrieval"] = {
        "lookup": needs_lookup,
        "hits": found.hits if found else [],
        "edges": found.edges if found else 0,
        "reached": found.reached if found else 0,
        "kept": found.kept if found else 0,
        "passages": [(t, s) for t, _, s, _ in passages],
        "links": {names[t]: u for t, _, _, u in passages if u},
    }
    return verdict


# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_DSN = os.environ.get("DATABASE_URL") or os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
)


def handle_embed(args: argparse.Namespace) -> None:
    """Fill in `surfaces.embedding` for every name that does not have one yet."""
    n = backfill_embeddings(args.dsn, tenant=args.tenant)
    print(
        f"embedded {n} surfaces with {EMBED_MODEL}"
        if n
        else "every name surface already has an embedding"
    )


def handle_ask(args: argparse.Namespace) -> None:
    verdict = answer_question(
        args.question,
        dsn=args.dsn,
        tenant=args.tenant,
        connector=args.connector if args.workspace else None,
        workspace=args.workspace,
        model=args.model,
        allow_reframe=not args.no_reframe,
        k=args.k,
        seeds=args.seeds,
        passages=args.passages,
        alpha=args.alpha,
        cutoff=args.cutoff,
        min_co=args.min_co,
        use_ppr=not args.no_ppr,
        text=args.text,
    )
    if verdict is None:
        raise SystemExit(
            "no surfaces matched -- run `embed` first, or ask about something indexed"
        )
    if args.explain:
        r = verdict["retrieval"]
        print(f"seed surfaces ({len(r['hits'])}):")
        for surface, score in r["hits"]:
            print(f"   {score:.3f}  {surface}")
        print(
            f"\ngraph: {r['edges']} PMI edges, {r['reached']} entities reached by PPR, "
            f"{r['kept']} kept above the cutoff"
        )
        for title, score in r["passages"]:
            print(f"   {score:.5f}  {title}")
        print("-" * 60 + "\n")
    print(AN.render(verdict))


def handle_search(args: argparse.Namespace) -> None:
    found = search_passages(
        args.question,
        dsn=args.dsn,
        tenant=args.tenant,
        connector=args.connector if args.workspace else None,
        workspace=args.workspace,
        k=args.k,
        seeds=args.seeds,
        passages=args.passages,
        alpha=args.alpha,
        cutoff=args.cutoff,
        min_co=args.min_co,
        use_ppr=not args.no_ppr,
        text=args.text,
    )
    print(format_passages(found))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("embed", help="backfill surfaces.embedding for names")
    e.add_argument("--tenant", required=True)
    e.set_defaults(handler=handle_embed)

    knobs = argparse.ArgumentParser(add_help=False)
    knobs.add_argument("question")
    who = knobs.add_mutually_exclusive_group(required=True)
    who.add_argument("--tenant", help="claim a tenant (app_ingest)")
    who.add_argument(
        "--workspace",
        help="declare a workspace and let the database resolve it (app_connector)",
    )
    knobs.add_argument("--connector", default="slack")
    knobs.add_argument(
        "--k",
        type=int,
        default=40,
        help="surfaces retrieved before collapsing onto entities",
    )
    knobs.add_argument("--seeds", type=int, default=8, help="entities seeding PPR")
    knobs.add_argument(
        "--passages", type=int, default=8, help="episodes sent to the model"
    )
    knobs.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="restart probability; lower walks further",
    )
    knobs.add_argument(
        "--cutoff",
        type=float,
        default=CUTOFF_RATIO,
        help="keep walked entities scoring at least this fraction of the best non-seed; 0 keeps everything",
    )
    knobs.add_argument(
        "--min-co",
        type=int,
        default=MIN_CO_OCCURRENCE,
        help="drop entity pairs seen together fewer than N times",
    )
    knobs.add_argument(
        "--no-ppr",
        action="store_true",
        help="seeds only, no walk -- the comparison baseline",
    )
    knobs.add_argument(
        "--text",
        type=int,
        default=TEXT_CANDIDATES,
        help="episodes the BM25 text channel adds before fusion; 0 = graph only",
    )

    a = sub.add_parser("ask", help="answer a question from the graph", parents=[knobs])
    a.add_argument(
        "--no-reframe",
        action="store_true",
        help="answer the question as asked; never propose a better one",
    )
    a.add_argument(
        "--explain",
        action="store_true",
        help="show seeds, graph size and chosen passages",
    )
    a.add_argument("--model", default=AN.ANSWER_MODEL)
    a.set_defaults(handler=handle_ask)

    r = sub.add_parser(
        "search",
        help="passages only, no answer: what a tool hands an agent",
        parents=[knobs],
    )
    r.set_defaults(handler=handle_search)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
