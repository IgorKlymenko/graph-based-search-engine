from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from retrieval.config import PPRSearchConfig, SearchFilters
from retrieval.exceptions import GraphLoadError, GraphTooLargeError, SearchTimeoutError, validate_inputs
from retrieval.graph_loader import GraphEdge, GraphNode, load_group
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import Fact, PPRResults, RankedNode
from retrieval.rerank import _ScoredFact, rerank_facts
from retrieval.seed import collect_seeds

log = logging.getLogger(__name__)

_FACT_QUERY = (
    "MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]->(m:Entity {group_id: $group_id}) "
    "WHERE n.uuid IN $uuids OR m.uuid IN $uuids "
    "RETURN e.uuid AS uuid, e.fact AS fact, "
    "e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid, "
    "e.created_at AS created_at, e.valid_at AS valid_at, "
    "e.invalid_at AS invalid_at, e.expired_at AS expired_at, "
    "e.episodes AS episodes, e.fact_embedding AS embedding "
    "LIMIT $limit"
)


async def ppr_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient | None,
    query: str,
    group_id: str,
    config: PPRSearchConfig,
    filters: SearchFilters | None = None,
    _query_embedding_override: list[float] | None = None,
) -> PPRResults:
    """Full PPR pipeline: load graph, collect seeds, power-iterate, rerank facts."""
    validate_inputs(query, group_id)
    nodes, edges = await load_group(driver, group_id, config)

    if not nodes:
        return PPRResults(
            nodes=[], facts=[], seed_uuids=[], num_nodes=0, num_edges=0
        )

    seed_weights, expansion_terms, hyde_answers = await collect_seeds(
        driver,
        query,
        group_id,
        config,
        embedder,
        llm_client,
        query_embedding=_query_embedding_override,
    )

    sorted_uuids = sorted(n.uuid for n in nodes)
    idx: dict[str, int] = {uuid: i for i, uuid in enumerate(sorted_uuids)}
    node_by_uuid = {n.uuid: n for n in nodes}
    N = len(sorted_uuids)

    _check_memory(len(edges), config)
    M = _build_transition_matrix(edges, idx, N)
    s = _build_personalization_vector(seed_weights, idx, N)

    loop = asyncio.get_event_loop()
    try:
        p = await asyncio.wait_for(
            loop.run_in_executor(None, _run_power_iteration, M, s, config)
            if N > 10_000
            else loop.run_in_executor(None, _run_power_iteration, M, s, config),
            timeout=config.ppr_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise SearchTimeoutError("PPR power iteration timed out")

    top_node_indices = np.argsort(p)[::-1][: config.top_nodes]
    candidate_uuids = [sorted_uuids[i] for i in top_node_indices]
    scored_facts = await _load_candidate_facts(
        driver, group_id, candidate_uuids, config.candidate_facts, filters
    )

    node_episodes: dict[str, set[str]] = defaultdict(set)
    for sf in scored_facts:
        node_episodes[sf.fact.source_uuid].update(sf.fact.episodes)
        node_episodes[sf.fact.target_uuid].update(sf.fact.episodes)

    ranked_nodes = [
        RankedNode(
            uuid=sorted_uuids[i],
            name=node_by_uuid[sorted_uuids[i]].name,
            summary=node_by_uuid[sorted_uuids[i]].summary,
            score=float(p[i]),
            episodes=sorted(node_episodes.get(sorted_uuids[i], set())),
        )
        for i in top_node_indices
    ]

    scored_with_ppr: list[_ScoredFact] = []
    for sf in scored_facts:
        src_score = float(p[idx[sf.fact.source_uuid]]) if sf.fact.source_uuid in idx else 0.0
        tgt_score = float(p[idx[sf.fact.target_uuid]]) if sf.fact.target_uuid in idx else 0.0
        ppr_score = max(src_score, tgt_score)
        scored_with_ppr.append(
            _ScoredFact(fact=sf.fact.model_copy(update={"score": ppr_score}), embedding=sf.embedding)
        )

    scored_with_ppr.sort(key=lambda sf: sf.fact.score, reverse=True)
    top_candidates = scored_with_ppr[: config.candidate_facts]

    facts = await rerank_facts(
        query=query,
        candidates=top_candidates,
        mode=config.rerank,
        embedder=embedder,
        llm_client=llm_client,
        min_relevance=config.min_relevance,
    )

    return PPRResults(
        nodes=ranked_nodes,
        facts=facts[: config.top_facts],
        seed_uuids=list(seed_weights.keys()),
        expansion_terms=expansion_terms,
        hyde_answers=hyde_answers,
        num_nodes=N,
        num_edges=len(edges),
    )


async def _load_candidate_facts(
    driver: GraphDriver,
    group_id: str,
    candidate_uuids: list[str],
    limit: int,
    filters: SearchFilters | None,
) -> list[_ScoredFact]:
    """Fetch full fact data (including embeddings) for candidate nodes."""
    uuids = candidate_uuids
    if filters and filters.edge_uuids:
        uuids = [u for u in uuids if u in set(filters.edge_uuids)]

    try:
        rows = await driver.execute_query(
            _FACT_QUERY, group_id=group_id, uuids=uuids, limit=limit
        )
    except Exception as exc:
        raise GraphLoadError("Failed to load candidate facts") from exc

    result: list[_ScoredFact] = []
    for r in rows:
        try:
            fact = Fact(
                uuid=r["uuid"],
                fact=r["fact"],
                source_uuid=r["source_uuid"],
                target_uuid=r["target_uuid"],
                score=0.0,
                created_at=_parse_dt(r["created_at"]),
                valid_at=_parse_dt(r.get("valid_at")),
                invalid_at=_parse_dt(r.get("invalid_at")),
                expired_at=_parse_dt(r.get("expired_at")),
                episodes=r.get("episodes") or [],
            )
        except Exception:
            log.warning("Skipping malformed fact row: uuid=%r", r.get("uuid"))
            continue
        embedding: list[float] | None = r.get("embedding")
        result.append(_ScoredFact(fact=fact, embedding=embedding))

    return result


def _build_transition_matrix(
    edges: list[GraphEdge],
    idx: dict[str, int],
    N: int,
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for edge in edges:
        src, tgt = edge.source_uuid, edge.target_uuid
        if src not in idx or tgt not in idx:
            continue
        i, j = idx[src], idx[tgt]
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]

    M = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    col_sums = np.asarray(M.sum(axis=0)).flatten()
    inv_col = np.where(col_sums > 0, 1.0 / col_sums, 0.0)
    M = M @ diags(inv_col, format="csr")
    return M


def _build_personalization_vector(
    seed_weights: dict[str, float],
    idx: dict[str, int],
    N: int,
) -> np.ndarray:
    s = np.zeros(N, dtype=np.float64)
    for uuid, weight in seed_weights.items():
        if uuid in idx:
            s[idx[uuid]] = weight
    total = s.sum()
    if total == 0:
        s[:] = 1.0 / N
    else:
        s /= total
    return s


def _run_power_iteration(
    M: csr_matrix,
    s: np.ndarray,
    config: PPRSearchConfig,
) -> np.ndarray:
    N = len(s)
    p = s.copy()
    for _ in range(config.max_iter):
        p_new = (1.0 - config.alpha) * (M @ p) + config.alpha * s
        leaked = 1.0 - p_new.sum()
        if abs(leaked) > 1e-10:
            p_new += leaked / N
        delta = float(np.abs(p_new - p).sum())
        p = p_new
        if delta < config.tol:
            break
    return p


def _check_memory(num_edges: int, config: PPRSearchConfig) -> None:
    estimated_bytes = num_edges * 3 * 8
    try:
        import psutil
        available = psutil.virtual_memory().available
        if estimated_bytes > 0.8 * available:
            raise GraphTooLargeError(
                num_nodes=0,
                num_edges=num_edges,
                limits={"max_edges": config.max_edges},
            )
    except ImportError:
        pass


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        log.debug("Could not parse datetime: %r", value)
        return None
