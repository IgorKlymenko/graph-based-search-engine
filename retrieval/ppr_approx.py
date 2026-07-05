from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field

from retrieval.config import RerankMode, SearchFilters
from retrieval.exceptions import GraphLoadError, SearchTimeoutError, validate_inputs
from retrieval.graph_loader import GraphEdge, GraphNode, load_group
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import Fact, PPRResults, RankedNode
from retrieval.ppr import _load_candidate_facts, _parse_dt
from retrieval.rerank import _ScoredFact, rerank_facts
from retrieval.seed import collect_seeds

log = logging.getLogger(__name__)


class FastPPRConfig(BaseModel):
    num_seeds: int = Field(default=10, ge=1, le=100)
    top_k: int = Field(default=50, ge=1, le=500)
    push_threshold: float = Field(default=1e-5, gt=0.0)
    num_walks: int = Field(default=200, ge=10, le=5000)
    alpha: float = Field(default=0.15, gt=0.0, lt=1.0)
    rerank: RerankMode = RerankMode.embedding
    min_relevance: float | None = None
    expand_seeds: bool = False
    expansion_terms: int = Field(default=6, ge=0, le=20)
    expansion_model: str = "gpt-4o-mini"
    max_nodes: int = Field(default=100_000, ge=1)
    max_edges: int = Field(default=500_000, ge=1)
    ppr_timeout_seconds: float = Field(default=10.0, gt=0.0)
    top_nodes: int = Field(default=10, ge=1, le=200)
    top_facts: int = Field(default=15, ge=1, le=500)
    candidate_facts: int = Field(default=50, ge=1, le=1000)


async def fast_ppr_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient | None,
    query: str,
    group_id: str,
    config: FastPPRConfig,
    filters: SearchFilters | None = None,
    _query_embedding_override: list[float] | None = None,
    rerank_embedder: Embedder | None = None,
) -> PPRResults:
    """Approximate PPR via bidirectional random walk. Drop-in replacement for ppr_search().

    ``embedder`` seeds the search against entity ``name_embedding`` (e5, 768-dim).
    Facts are embedded in a different space at upload time (OpenAI
    text-embedding-3-small, 1536-dim), so ``rerank_embedder`` must embed the query
    in that fact space — otherwise the rerank dot products are cross-space garbage
    and every fact is dropped. Defaults to ``embedder`` for callers that use a
    single embedding space.
    """
    validate_inputs(query, group_id)
    from retrieval.config import PPRSearchConfig

    loader_config = PPRSearchConfig(
        num_seeds=config.num_seeds,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
    )
    nodes, edges = await load_group(driver, group_id, loader_config)

    if not nodes:
        return PPRResults(nodes=[], facts=[], seed_uuids=[], num_nodes=0, num_edges=0)

    seed_weights, expansion_terms, hyde_answers = await collect_seeds(
        driver,
        query,
        group_id,
        _seed_config(config),
        embedder,
        llm_client,
        query_embedding=_query_embedding_override,
    )

    adj, rev_adj = _build_adjacency(nodes, edges)

    # L1-normalize seed weights
    total = sum(seed_weights.values())
    seeds_norm = (
        {k: v / total for k, v in seed_weights.items()} if total > 0 else {}
    )
    if not seeds_norm:
        n = len(nodes)
        seeds_norm = {node.uuid: 1.0 / n for node in nodes[:1]}

    def _compute() -> dict[str, float]:
        estimates, residuals = _forward_push(adj, seeds_norm, config.alpha, config.push_threshold)
        # Merge residuals into estimates (unprocessed mass attributed to source)
        for v, r in residuals.items():
            estimates[v] = estimates.get(v, 0.0) + config.alpha * r

        # Top-K candidates by forward estimate
        top_candidates = sorted(estimates.keys(), key=lambda u: estimates[u], reverse=True)[
            : config.top_k
        ]

        rng = np.random.default_rng(seed=42)
        final_scores: dict[str, float] = {}
        for v in top_candidates:
            bwd = _backward_walks(rev_adj, v, config.num_walks, config.alpha, rng)
            # Combine: forward estimate + seed-weighted backward contribution
            seed_mass = seeds_norm.get(v, 0.0)
            final_scores[v] = estimates.get(v, 0.0) + (1.0 - config.alpha) * bwd * seed_mass
        # Nodes only in estimates but not top_k
        for v, score in estimates.items():
            if v not in final_scores:
                final_scores[v] = score
        return final_scores

    loop = asyncio.get_event_loop()
    try:
        scores = await asyncio.wait_for(
            loop.run_in_executor(None, _compute),
            timeout=config.ppr_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise SearchTimeoutError("FAST-PPR timed out")

    node_by_uuid = {n.uuid: n for n in nodes}
    ranked = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)

    candidate_uuids = [u for u in ranked[: config.top_nodes] if u in node_by_uuid]
    scored_facts = await _load_candidate_facts(
        driver, group_id, candidate_uuids, config.candidate_facts, filters
    )

    node_episodes: dict[str, set[str]] = defaultdict(set)
    for sf in scored_facts:
        node_episodes[sf.fact.source_uuid].update(sf.fact.episodes)
        node_episodes[sf.fact.target_uuid].update(sf.fact.episodes)

    ranked_nodes = [
        RankedNode(
            uuid=u,
            name=node_by_uuid[u].name,
            summary=node_by_uuid[u].summary,
            score=float(scores[u]),
            episodes=sorted(node_episodes.get(u, set())),
        )
        for u in ranked[: config.top_nodes]
        if u in node_by_uuid
    ]

    scored_with_ppr: list[_ScoredFact] = []
    for sf in scored_facts:
        src_score = scores.get(sf.fact.source_uuid, 0.0)
        tgt_score = scores.get(sf.fact.target_uuid, 0.0)
        ppr_score = max(src_score, tgt_score)
        scored_with_ppr.append(
            _ScoredFact(
                fact=sf.fact.model_copy(update={"score": min(ppr_score, 1.0)}),
                embedding=sf.embedding,
            )
        )

    scored_with_ppr.sort(key=lambda sf: sf.fact.score, reverse=True)
    top_candidates_facts = scored_with_ppr[: config.candidate_facts]

    facts = await rerank_facts(
        query=query,
        candidates=top_candidates_facts,
        mode=config.rerank,
        embedder=rerank_embedder or embedder,
        llm_client=llm_client,
        min_relevance=config.min_relevance,
    )

    return PPRResults(
        nodes=ranked_nodes,
        facts=facts[: config.top_facts],
        seed_uuids=list(seed_weights.keys()),
        expansion_terms=expansion_terms,
        hyde_answers=hyde_answers,
        num_nodes=len(nodes),
        num_edges=len(edges),
    )


def _build_adjacency(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build undirected adjacency and reverse-adjacency lists."""
    adj: dict[str, list[str]] = defaultdict(list)
    rev_adj: dict[str, list[str]] = defaultdict(list)
    node_uuids = {n.uuid for n in nodes}
    for edge in edges:
        src, tgt = edge.source_uuid, edge.target_uuid
        if src not in node_uuids or tgt not in node_uuids:
            continue
        # Undirected: both directions in both maps
        adj[src].append(tgt)
        adj[tgt].append(src)
        rev_adj[tgt].append(src)
        rev_adj[src].append(tgt)
    return dict(adj), dict(rev_adj)


def _forward_push(
    adj: dict[str, list[str]],
    seeds: dict[str, float],
    alpha: float,
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Iterative forward push from seed distribution."""
    estimates: dict[str, float] = defaultdict(float)
    residuals: dict[str, float] = dict(seeds)
    queue = list(seeds.keys())

    while queue:
        v = queue.pop()
        r = residuals.pop(v, 0.0)
        if r < threshold:
            if r > 0:
                residuals[v] = r
            continue
        estimates[v] += alpha * r
        push_mass = (1.0 - alpha) * r
        neighbours = adj.get(v, [])
        if not neighbours:
            continue
        per_neighbour = push_mass / len(neighbours)
        for u in neighbours:
            prev = residuals.get(u, 0.0)
            residuals[u] = prev + per_neighbour
            if prev < threshold <= residuals[u]:
                queue.append(u)

    return dict(estimates), residuals


def _backward_walks(
    rev_adj: dict[str, list[str]],
    target: str,
    num_walks: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Estimate backward walk probability reaching `target`. Returns fraction of walks that hit."""
    hits = 0
    for _ in range(num_walks):
        node = target
        while True:
            if rng.random() < alpha:
                break
            parents = rev_adj.get(node, [])
            if not parents:
                break
            node = parents[int(rng.integers(0, len(parents)))]
        hits += 1
    return hits / num_walks


def _seed_config(config: FastPPRConfig):  # type: ignore[return]
    """Return a minimal PPRSearchConfig-compatible object for seed collection."""
    from retrieval.config import PPRSearchConfig, RerankMode

    return PPRSearchConfig(
        num_seeds=config.num_seeds,
        expand_seeds=config.expand_seeds,
        expansion_terms=config.expansion_terms,
        expansion_model=config.expansion_model,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        rerank=config.rerank,
    )
