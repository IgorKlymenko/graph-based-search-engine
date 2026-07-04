from __future__ import annotations

import asyncio
import logging

import numpy as np
from pydantic import BaseModel, Field

from retrieval.config import SearchFilters, TemporalPPRSearchConfig
from retrieval.exceptions import validate_inputs
from retrieval.interfaces import Embedder, GraphDriver
from retrieval.models import MultihopHop, MultihopSearchResults
from retrieval.shared import EmbeddingFuser, FactMerger, StopCriterion
from retrieval.temporal import temporal_ppr_search

log = logging.getLogger(__name__)


class TreeHopConfig(BaseModel):
    max_hops: int = Field(default=4, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    fusion_alpha: float = Field(default=0.6, gt=0.5, lt=0.95)
    top_k_facts_for_fusion: int = Field(default=5, ge=1, le=50)
    total_timeout_seconds: float = Field(default=60.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)


async def treehop_search(
    driver: GraphDriver,
    embedder: Embedder,
    query: str,
    group_id: str,
    config: TreeHopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    """Embedding-space multihop — zero LLM calls between hops."""
    validate_inputs(query, group_id)
    merger = FactMerger()
    hops: list[MultihopHop] = []
    stop_criterion = StopCriterion(config.min_new_facts, config.max_hops)
    fuser = EmbeddingFuser(config.fusion_alpha)
    stop_reason = "budget"

    current_emb: list[float] = await embedder.embed(query)
    original_emb = current_emb

    async def _run() -> None:
        nonlocal stop_reason, current_emb

        for hop_idx in range(config.max_hops):
            temporal_result = await temporal_ppr_search(
                driver,
                embedder,
                None,
                query,
                group_id,
                config.temporal,
                filters,
                _query_embedding_override=current_emb,
            )

            new_uuids = merger.add(temporal_result.facts)
            hops.append(
                MultihopHop(
                    query=f"[embedding-hop-{hop_idx}]",
                    temporal=temporal_result,
                    new_uuids=new_uuids,
                )
            )

            _log_cosine_similarity(original_emb, current_emb, hop_idx)

            stop, reason = stop_criterion.should_stop(hop_idx, len(new_uuids), dynamic=True)
            if stop:
                stop_reason = reason
                break

            top_facts = merger.sorted_facts()[: config.top_k_facts_for_fusion]
            if not top_facts:
                stop_reason = "convergence"
                break

            fact_embs = await _embed_facts(embedder, top_facts)
            if not fact_embs:
                stop_reason = "convergence"
                break

            current_emb = fuser.fuse(current_emb, fact_embs)

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)
    return MultihopSearchResults(
        hops=hops, merged=merger.sorted_facts(), stop_reason=stop_reason
    )


async def _embed_facts(embedder: Embedder, facts) -> list[list[float]]:
    """Embed fact text concurrently; skip failures."""
    results = await asyncio.gather(
        *[embedder.embed(f.fact) for f in facts], return_exceptions=True
    )
    return [r for r in results if not isinstance(r, BaseException)]


def _log_cosine_similarity(
    original: list[float], current: list[float], hop_idx: int
) -> None:
    a = np.array(original, dtype=np.float64)
    b = np.array(current, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    log.debug("TreeHop hop=%d cosine_similarity=%.4f", hop_idx, sim)
    if sim < 0.3:
        log.warning("TreeHop hop=%d: low cosine similarity %.4f vs original query", hop_idx, sim)
