from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from retrieval.config import SearchFilters, TemporalPPRSearchConfig
from retrieval.exceptions import validate_inputs
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import MultihopHop, MultihopSearchResults, TemporalPPRResults
from retrieval.shared import BridgeGenerator, FactMerger, StopCriterion
from retrieval.temporal import temporal_ppr_search

log = logging.getLogger(__name__)


class BridgeStrategy(str, Enum):
    single = "single"
    dual = "dual"


class BeamMultihopConfig(BaseModel):
    beam_width: int = Field(default=3, ge=1, le=10)
    max_hops: int = Field(default=4, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    bridge_model: str = "gpt-4o-mini"
    bridge_strategy: BridgeStrategy = BridgeStrategy.single
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=180.0, gt=0.0)
    beam_score_fn: str = "mean_rerank"
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)


@dataclass
class BeamMember:
    query: str
    merger: FactMerger = field(default_factory=FactMerger)
    hops: list[MultihopHop] = field(default_factory=list)
    score: float = 0.0


async def beam_multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: BeamMultihopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    """Beam-search multihop: parallel reasoning hypotheses, pruned each hop."""
    validate_inputs(query, group_id)
    if config.min_new_facts == 0:
        log.warning("min_new_facts=0: convergence stopping is disabled; loop runs to max_hops")
    bridge_gen = BridgeGenerator(llm_client, config.bridge_model, config.bridge_timeout_seconds)
    stop_criterion = StopCriterion(config.min_new_facts, config.max_hops)
    beam: list[BeamMember] = [BeamMember(query=query)]
    global_merger = FactMerger()
    stop_reason = "budget"

    async def _run() -> None:
        nonlocal stop_reason, beam

        for hop_idx in range(config.max_hops):
            # Fan out: all beam members searched in parallel
            hop_tasks = [
                temporal_ppr_search(
                    driver, embedder, llm_client, m.query, group_id, config.temporal, filters
                )
                for m in beam
            ]
            hop_results: list[TemporalPPRResults] = await asyncio.gather(*hop_tasks)

            for member, result in zip(beam, hop_results):
                new_uuids = member.merger.add(result.facts)
                global_merger.add(result.facts)
                member.hops.append(
                    MultihopHop(
                        query=member.query,
                        temporal=result,
                        new_uuids=new_uuids,
                    )
                )
                member.score = _beam_score(member, config.beam_score_fn)

            # Prune to top beam_width before bridge generation
            beam.sort(key=lambda m: m.score, reverse=True)
            beam[:] = beam[: config.beam_width]

            total_new = len(global_merger.all_uuids)
            stop, reason = stop_criterion.should_stop(hop_idx, total_new, dynamic=True)
            if stop:
                stop_reason = reason
                break

            if hop_idx == config.max_hops - 1:
                break

            # Generate next queries
            if config.bridge_strategy == BridgeStrategy.dual:
                dual_tasks = [
                    bridge_gen.dual(query, m.merger.sorted_facts(), dynamic=True) for m in beam
                ]
                dual_results = await asyncio.gather(*dual_tasks)
                new_beam: list[BeamMember] = []
                for member, (fwd, bwd) in zip(beam, dual_results):
                    for q in [fwd, bwd]:
                        if q:
                            new_member = BeamMember(query=q)
                            new_member.merger.add(global_merger.sorted_facts())
                            new_beam.append(new_member)
                if not new_beam:
                    stop_reason = "llm_stop"
                    break
                beam = new_beam[: config.beam_width]
            else:
                single_tasks = [
                    bridge_gen.single(query, m.merger.sorted_facts(), dynamic=True) for m in beam
                ]
                bridge_results: list[str | None] = await asyncio.gather(*single_tasks)
                new_beam = []
                for bridge in bridge_results:
                    if bridge:
                        new_member = BeamMember(query=bridge)
                        new_member.merger.add(global_merger.sorted_facts())
                        new_beam.append(new_member)
                if not new_beam:
                    stop_reason = "llm_stop"
                    break
                beam = new_beam[: config.beam_width]

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)

    all_hops = [h for m in beam for h in m.hops]
    return MultihopSearchResults(
        hops=all_hops,
        merged=global_merger.sorted_facts(),
        stop_reason=stop_reason,
    )


def _beam_score(member: BeamMember, fn: str) -> float:
    facts = member.merger.sorted_facts()
    if not facts:
        return 0.0
    if fn == "mean_rerank":
        scores = [f.rerank_score if f.rerank_score is not None else f.score for f in facts]
        return sum(scores) / len(scores)
    if fn == "max_rerank":
        return max(f.rerank_score if f.rerank_score is not None else f.score for f in facts)
    if fn == "fact_count":
        return float(len(facts))
    return 0.0
