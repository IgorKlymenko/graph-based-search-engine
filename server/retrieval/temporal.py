from __future__ import annotations

import logging
from datetime import datetime, timezone

from retrieval.config import SearchFilters, TemporalPPRSearchConfig
from retrieval.exceptions import validate_inputs
from retrieval.filters import make_aware, temporal_match
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import Fact, Passage, TemporalPPRResults
from retrieval.ppr import ppr_search

log = logging.getLogger(__name__)

_EPISODE_QUERY = (
    "MATCH (e:Episodic) "
    "WHERE e.uuid IN $uuids "
    "RETURN e.uuid AS uuid, e.name AS name, e.content AS content, "
    "e.valid_at AS valid_at, e.source_description AS source_description"
)


async def temporal_ppr_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient | None,
    query: str,
    group_id: str,
    config: TemporalPPRSearchConfig,
    filters: SearchFilters | None = None,
    _query_embedding_override: list[float] | None = None,
) -> TemporalPPRResults:
    """PPR + temporal filter + optional passage hydration."""
    validate_inputs(query, group_id)
    as_of = make_aware(config.as_of) or datetime.now(timezone.utc)

    ppr_result = await ppr_search(
        driver,
        embedder,
        llm_client,
        query,
        group_id,
        config.ppr,
        filters,
        _query_embedding_override=_query_embedding_override,
    )

    relevant_count = len(ppr_result.facts)
    filtered = [f for f in ppr_result.facts if temporal_match(f, config.mode, as_of)]
    dropped = relevant_count - len(filtered)

    passages: list[Passage] = []
    if config.hydrate and filtered:
        passages = await _hydrate_passages(driver, filtered, config.max_passages)

    return TemporalPPRResults(
        facts=filtered[: config.max_facts],
        mode=config.mode.value,
        as_of=as_of,
        relevant_count=relevant_count,
        dropped_by_temporal=dropped,
        num_seeds=len(ppr_result.seed_uuids),
        num_nodes=ppr_result.num_nodes,
        num_edges=ppr_result.num_edges,
        passages=passages,
        seed_uuids=ppr_result.seed_uuids,
        expansion_terms=ppr_result.expansion_terms,
        hyde_answers=ppr_result.hyde_answers,
    )


async def _hydrate_passages(
    driver: GraphDriver,
    facts: list[Fact],
    max_passages: int,
) -> list[Passage]:
    """Fetch source passages for the given facts, ranked by best relevance score."""
    # Collect episode UUIDs in order of first appearance, with best score per episode
    seen: dict[str, float] = {}
    for f in facts:
        fact_score = f.rerank_score if f.rerank_score is not None else f.score
        for ep_uuid in f.episodes:
            if ep_uuid not in seen or fact_score > seen[ep_uuid]:
                seen[ep_uuid] = fact_score

    if not seen:
        return []

    ranked_episodes = sorted(seen.keys(), key=lambda u: seen[u], reverse=True)[:max_passages]
    episode_to_best_score = {u: seen[u] for u in ranked_episodes}

    # Map episode → supporting fact UUIDs
    episode_to_fact_uuids: dict[str, list[str]] = {u: [] for u in ranked_episodes}
    for f in facts:
        for ep_uuid in f.episodes:
            if ep_uuid in episode_to_fact_uuids:
                episode_to_fact_uuids[ep_uuid].append(f.uuid)

    try:
        rows = await driver.execute_query(_EPISODE_QUERY, uuids=ranked_episodes)
    except Exception as exc:
        log.warning("Passage hydration failed: %s", exc)
        return []

    row_by_uuid = {r["uuid"]: r for r in rows}
    passages: list[Passage] = []
    for ep_uuid in ranked_episodes:
        row = row_by_uuid.get(ep_uuid)
        if row is None:
            continue
        try:
            passages.append(
                Passage(
                    uuid=ep_uuid,
                    name=row["name"],
                    content=row["content"],
                    valid_at=row.get("valid_at"),
                    source_description=row.get("source_description") or "",
                    supports_fact_uuids=episode_to_fact_uuids.get(ep_uuid, []),
                    best_relevance=episode_to_best_score.get(ep_uuid),
                )
            )
        except Exception as exc:
            log.warning("Failed to build Passage for %r: %s", ep_uuid, exc)

    return passages
