from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from retrieval.config import SearchFilters, TemporalPPRSearchConfig
from retrieval.exceptions import InvalidInputError, LLMCallError, validate_inputs
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import MultihopHop, MultihopSearchResults, TemporalPPRResults
from retrieval.shared import BridgeGenerator, FactMerger
from retrieval.temporal import temporal_ppr_search

log = logging.getLogger(__name__)


class DAGMultihopConfig(BaseModel):
    decompose_model: str = "gpt-4o-mini"
    decompose_timeout_seconds: float = Field(default=20.0, gt=0.0)
    max_subqueries: int = Field(default=6, ge=1, le=20)
    bridge_model: str = "gpt-4o-mini"
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=180.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)


async def dag_multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: DAGMultihopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    """DAG-decomposition multihop: parallel resolution of independent sub-queries."""
    validate_inputs(query, group_id)
    bridge_gen = BridgeGenerator(llm_client, config.bridge_model, config.bridge_timeout_seconds)
    global_merger = FactMerger()
    hops: list[MultihopHop] = []
    stop_reason = "fixed"

    async def _run() -> None:
        subqueries = await _decompose_query(query, config, llm_client, bridge_gen)

        results_by_id: dict[int, TemporalPPRResults] = {}
        for layer in _topological_layers(subqueries):
            layer_tasks = [
                temporal_ppr_search(
                    driver,
                    embedder,
                    llm_client,
                    _enrich_subquery(sq, results_by_id),
                    group_id,
                    config.temporal,
                    filters,
                )
                for sq in layer
            ]
            layer_results = await asyncio.gather(*layer_tasks)

            for sq, result in zip(layer, layer_results):
                results_by_id[sq["id"]] = result
                new_uuids = global_merger.add(result.facts)
                hops.append(
                    MultihopHop(
                        query=sq["question"],
                        temporal=result,
                        new_uuids=new_uuids,
                    )
                )

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)
    return MultihopSearchResults(
        hops=hops, merged=global_merger.sorted_facts(), stop_reason=stop_reason
    )


async def _decompose_query(
    query: str,
    config: DAGMultihopConfig,
    llm_client: LLMClient,
    bridge_gen: BridgeGenerator,
) -> list[dict]:
    """Decompose query into DAG of sub-questions. Falls back to single-node on any failure."""
    _fallback = [{"id": 0, "question": query, "depends_on": []}]
    try:
        subqueries = await bridge_gen.decompose(query, config.decompose_model, llm_client)
    except LLMCallError as exc:
        log.warning("DAG decomposition failed, using single-node fallback: %s", exc)
        return _fallback

    if not subqueries or not isinstance(subqueries, list):
        log.warning("DAG decomposition returned empty or invalid result, using fallback")
        return _fallback

    validated: list[dict] = []
    seen_ids: set[int] = set()
    for item in subqueries[: config.max_subqueries]:
        try:
            sq_id = int(item["id"])
            question = str(item["question"]).strip()[:500]
            depends_on = [int(d) for d in item.get("depends_on") or []]
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Invalid sub-query entry: %s", exc)
            continue
        if not question:
            continue
        if any(d >= sq_id or d not in seen_ids for d in depends_on):
            log.warning("Sub-query %d has invalid dependencies, using fallback", sq_id)
            return _fallback
        seen_ids.add(sq_id)
        validated.append({"id": sq_id, "question": question, "depends_on": depends_on})

    if not validated:
        return _fallback
    return validated


def _topological_layers(subqueries: list[dict]) -> list[list[dict]]:
    """Group sub-queries into parallelizable layers by dependency order."""
    resolved: set[int] = set()
    remaining = list(subqueries)
    layers: list[list[dict]] = []

    while remaining:
        ready = [sq for sq in remaining if all(d in resolved for d in sq["depends_on"])]
        if not ready:
            raise InvalidInputError("Circular dependency in sub-query DAG")
        layers.append(ready)
        resolved.update(sq["id"] for sq in ready)
        remaining = [sq for sq in remaining if sq["id"] not in resolved]

    return layers


def _enrich_subquery(sq: dict, results_by_id: dict[int, TemporalPPRResults]) -> str:
    """Prepend dependency facts as context (capped at 10 facts × 200 chars each)."""
    dep_facts = []
    for dep_id in sq["depends_on"]:
        dep_result = results_by_id.get(dep_id)
        if dep_result:
            dep_facts.extend(dep_result.facts[:5])
    if not dep_facts:
        return sq["question"]
    context = " | ".join(f.fact[:200] for f in dep_facts[:10])
    return f"Context: {context}\n\nQuestion: {sq['question']}"
