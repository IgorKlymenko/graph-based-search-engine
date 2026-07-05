"""Method runner: executes retrieval methods with full token and timing instrumentation."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from retrieval.config import (
    MultihopSearchConfig,
    PPRSearchConfig,
    TemporalMode,
    TemporalPPRSearchConfig,
)
from retrieval.filters import make_aware, temporal_match
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import (
    Fact,
    MultihopHop,
    MultihopSearchResults,
    TemporalPPRResults,
)
from retrieval.multihop import multihop_search
from retrieval.ppr_approx import FastPPRConfig, fast_ppr_search
from retrieval.shared import BridgeGenerator, FactMerger, StopCriterion
from retrieval.temporal import temporal_ppr_search
from retrieval.treehop import TreeHopConfig, treehop_search
from retrieval.multihop_beam import BeamMultihopConfig, BridgeStrategy, beam_multihop_search
from retrieval.dag_multihop import DAGMultihopConfig, dag_multihop_search

from arena.pricing import compute_cost, load_pricing
from arena.tracker import (
    CostBreakdown,
    HopTrace,
    MethodResult,
    StageTimings,
    TokenTracker,
    TokenUsage,
    TrackingEmbedder,
    TrackingLLMClient,
    count_tokens,
    timed,
)

log = logging.getLogger(__name__)


@dataclass
class ArenaArgs:
    temporal_mode: str = "currently_true"
    as_of: datetime | None = None
    max_hops: int = 4
    beam_width: int = 3
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    hydrate: bool = True
    redact_facts: bool = False
    verbose: bool = False
    generate: bool = True
    fast_ppr_push_threshold: float = 1e-5
    # Min cosine for the fact-embedding rerank. None falls back to rerank.py's
    # per-mode default (0.30), which is too strict for text-embedding-3-small.
    fact_min_relevance: float | None = None


def _temporal_config(args: ArenaArgs) -> TemporalPPRSearchConfig:
    return TemporalPPRSearchConfig(
        mode=TemporalMode(args.temporal_mode),
        as_of=args.as_of,
        hydrate=args.hydrate,
        ppr=PPRSearchConfig(
            expansion_model=args.llm_model,
            min_relevance=args.fact_min_relevance,
        ),
    )


def _multihop_config(args: ArenaArgs) -> MultihopSearchConfig:
    return MultihopSearchConfig(
        max_hops=args.max_hops,
        bridge_model=args.llm_model,
        temporal=_temporal_config(args),
    )


def _make_hop_trace(idx: int, hop: MultihopHop) -> HopTrace:
    return HopTrace(
        hop_index=idx,
        query=hop.query,
        ppr_seeds=hop.temporal.seed_uuids,
        nodes_loaded=hop.temporal.num_nodes,
        edges_loaded=hop.temporal.num_edges,
        facts_before_temporal=hop.temporal.relevant_count,
        facts_after_temporal=len(hop.temporal.facts),
        dropped_by_temporal=hop.temporal.dropped_by_temporal,
        new_fact_uuids=hop.new_uuids,
        all_fact_uuids=frozenset(f.uuid for f in hop.temporal.facts),
        bridge_queries=[],
        token_usage=TokenUsage(0, 0),
        timing_ms=0.0,
        expansion_terms=hop.temporal.expansion_terms,
        hyde_answers=hop.temporal.hyde_answers,
    )


_GENERATE_PROMPT = """\
Answer the following question using only the facts provided below from a knowledge graph.
Be direct and concise. Cite the relevant facts in your answer.
If the facts are insufficient to answer the question, say so briefly.

Question: {query}

Facts:
{facts_text}"""


async def generate_answer(
    query: str,
    facts: list[Fact],
    llm_client: LLMClient,
    tracker: TokenTracker,
    model: str,
    timings: StageTimings,
) -> str | None:
    """Generate a grounded answer from retrieved facts using the LLM."""
    if not facts:
        return None
    facts_text = "\n".join(f"{i + 1}. {f.fact}" for i, f in enumerate(facts))
    prompt = _GENERATE_PROMPT.format(query=query, facts_text=facts_text)
    try:
        start = time.perf_counter()
        answer = await llm_client.complete(prompt, model=model, max_tokens=512)
        timings.generate_ms = (time.perf_counter() - start) * 1000
        tracker.record(
            stage="generate",
            tokens_in=count_tokens(prompt, model),
            tokens_out=count_tokens(answer, model),
            model=model,
        )
        return answer
    except Exception as exc:
        log.warning("Answer generation failed: %s", exc)
        return None


def _make_cost(tracker: TokenTracker, args: ArenaArgs) -> CostBreakdown:
    pricing = load_pricing()
    by_stage = tracker.by_stage()
    llm_usage = TokenUsage(0, 0)
    embed_usage = TokenUsage(0, 0)
    for stage, usage in by_stage.items():
        if stage in ("llm", "llm_json", "generate"):
            llm_usage = llm_usage + usage
        elif stage in ("embed", "embed_batch"):
            embed_usage = embed_usage + usage
    return CostBreakdown(
        llm_cost_usd=compute_cost(llm_usage, args.llm_model, pricing),
        embed_cost_usd=compute_cost(embed_usage, args.embed_model, pricing),
    )


def _error_result(method_id: str, query: str, group_id: str, exc: Exception) -> MethodResult:
    log.error("Method %r failed: %s", method_id, exc, exc_info=True)
    return MethodResult(
        method_id=method_id,
        query=query,
        group_id=group_id,
        facts=[],
        hops=[],
        stop_reason="error",
        token_usage=TokenUsage(0, 0),
        cost=CostBreakdown(0.0, 0.0),
        timing=StageTimings(),
        error=str(exc),
    )


async def _temporal_fast_ppr(
    driver: GraphDriver,
    embedder: TrackingEmbedder,
    llm_client: TrackingLLMClient,
    query: str,
    group_id: str,
    config: TemporalPPRSearchConfig,
    timings: StageTimings,
    push_threshold: float = 1e-5,
    rerank_embedder: Embedder | None = None,
) -> TemporalPPRResults:
    """Temporal wrapper around fast_ppr_search for use in fast-ppr multihop."""
    fast_cfg = FastPPRConfig(
        num_seeds=config.ppr.num_seeds,
        expansion_model=config.ppr.expansion_model,
        expand_seeds=config.ppr.expand_seeds,
        max_nodes=config.ppr.max_nodes,
        max_edges=config.ppr.max_edges,
        top_facts=config.ppr.top_facts,
        candidate_facts=config.ppr.candidate_facts,
        top_nodes=config.ppr.top_nodes,
        rerank=config.ppr.rerank,
        min_relevance=config.ppr.min_relevance,
        push_threshold=push_threshold,
    )
    ppr_result = await fast_ppr_search(
        driver, embedder, llm_client, query, group_id, fast_cfg,
        rerank_embedder=rerank_embedder,
    )
    as_of = make_aware(config.as_of) or datetime.now(timezone.utc)
    relevant_count = len(ppr_result.facts)
    filtered = [f for f in ppr_result.facts if temporal_match(f, config.mode, as_of)]
    dropped = relevant_count - len(filtered)
    return TemporalPPRResults(
        facts=filtered[: config.max_facts],
        mode=config.mode.value,
        as_of=as_of,
        relevant_count=relevant_count,
        dropped_by_temporal=dropped,
        num_seeds=len(ppr_result.seed_uuids),
        num_nodes=ppr_result.num_nodes,
        num_edges=ppr_result.num_edges,
        seed_uuids=ppr_result.seed_uuids,
        expansion_terms=ppr_result.expansion_terms,
        hyde_answers=ppr_result.hyde_answers,
    )


async def run_ppr_sequential(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    try:
        config = _multihop_config(cli_args)
        with timed("total", timings):
            result: MultihopSearchResults = await multihop_search(
                driver, t_emb, t_llm, query, group_id, config
            )
        hops = [_make_hop_trace(i, h) for i, h in enumerate(result.hops)]
        return MethodResult(
            method_id="ppr",
            query=query,
            group_id=group_id,
            facts=result.merged,
            hops=hops,
            stop_reason=result.stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("ppr", query, group_id, exc)


async def run_fast_ppr_sequential(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    temporal_cfg = _temporal_config(cli_args)
    bridge_gen = BridgeGenerator(t_llm, cli_args.llm_model, 15.0)
    merger = FactMerger()
    stop_criterion = StopCriterion(1, cli_args.max_hops)
    hops_raw: list[MultihopHop] = []
    stop_reason = "budget"

    try:
        with timed("total", timings):
            current_query = query
            for hop_idx in range(cli_args.max_hops):
                temporal_result = await _temporal_fast_ppr(
                    driver, t_emb, t_llm, current_query, group_id, temporal_cfg, timings,
                    push_threshold=cli_args.fast_ppr_push_threshold,
                )
                new_uuids = merger.add(temporal_result.facts)
                hops_raw.append(
                    MultihopHop(query=current_query, temporal=temporal_result, new_uuids=new_uuids)
                )
                stop, reason = stop_criterion.should_stop(hop_idx, len(new_uuids), dynamic=True)
                if stop:
                    stop_reason = reason
                    break
                if hop_idx == cli_args.max_hops - 1:
                    break
                bridge = await bridge_gen.single(query, merger.sorted_facts(), dynamic=True)
                if not bridge:
                    stop_reason = "llm_stop"
                    break
                current_query = bridge

        hops = [_make_hop_trace(i, h) for i, h in enumerate(hops_raw)]
        return MethodResult(
            method_id="fast-ppr",
            query=query,
            group_id=group_id,
            facts=merger.sorted_facts(),
            hops=hops,
            stop_reason=stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("fast-ppr", query, group_id, exc)


async def run_treehop(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_emb = TrackingEmbedder(embedder, tracker)
    try:
        config = TreeHopConfig(
            max_hops=cli_args.max_hops,
            temporal=_temporal_config(cli_args),
        )
        with timed("total", timings):
            result = await treehop_search(driver, t_emb, query, group_id, config)
        hops = [_make_hop_trace(i, h) for i, h in enumerate(result.hops)]
        return MethodResult(
            method_id="treehop",
            query=query,
            group_id=group_id,
            facts=result.merged,
            hops=hops,
            stop_reason=result.stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("treehop", query, group_id, exc)


async def run_beam(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    try:
        config = BeamMultihopConfig(
            beam_width=cli_args.beam_width,
            max_hops=cli_args.max_hops,
            bridge_model=cli_args.llm_model,
            bridge_strategy=BridgeStrategy.single,
            temporal=_temporal_config(cli_args),
        )
        with timed("total", timings):
            result = await beam_multihop_search(driver, t_emb, t_llm, query, group_id, config)
        hops = [_make_hop_trace(i, h) for i, h in enumerate(result.hops)]
        return MethodResult(
            method_id="beam",
            query=query,
            group_id=group_id,
            facts=result.merged,
            hops=hops,
            stop_reason=result.stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("beam", query, group_id, exc)


async def run_beam_dual(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    try:
        config = BeamMultihopConfig(
            beam_width=cli_args.beam_width,
            max_hops=cli_args.max_hops,
            bridge_model=cli_args.llm_model,
            bridge_strategy=BridgeStrategy.dual,
            temporal=_temporal_config(cli_args),
        )
        with timed("total", timings):
            result = await beam_multihop_search(driver, t_emb, t_llm, query, group_id, config)
        hops = [_make_hop_trace(i, h) for i, h in enumerate(result.hops)]
        return MethodResult(
            method_id="beam-dual",
            query=query,
            group_id=group_id,
            facts=result.merged,
            hops=hops,
            stop_reason=result.stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("beam-dual", query, group_id, exc)


async def run_dag(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    try:
        config = DAGMultihopConfig(
            decompose_model=cli_args.llm_model,
            bridge_model=cli_args.llm_model,
            temporal=_temporal_config(cli_args),
        )
        with timed("total", timings):
            result = await dag_multihop_search(driver, t_emb, t_llm, query, group_id, config)
        hops = [_make_hop_trace(i, h) for i, h in enumerate(result.hops)]
        return MethodResult(
            method_id="dag",
            query=query,
            group_id=group_id,
            facts=result.merged,
            hops=hops,
            stop_reason=result.stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("dag", query, group_id, exc)


async def run_sequential_dual(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
    """Sequential multihop with dual (forward + backward) BDTR bridge per hop."""
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    temporal_cfg = _temporal_config(cli_args)
    bridge_gen = BridgeGenerator(t_llm, cli_args.llm_model, 15.0)
    merger = FactMerger()
    stop_criterion = StopCriterion(1, cli_args.max_hops)
    hops_raw: list[MultihopHop] = []
    stop_reason = "budget"

    try:
        with timed("total", timings):
            fwd_query = query
            bwd_query = query

            for hop_idx in range(cli_args.max_hops):
                fwd_result = await temporal_ppr_search(
                    driver, t_emb, t_llm, fwd_query, group_id, temporal_cfg
                )
                bwd_result = await temporal_ppr_search(
                    driver, t_emb, t_llm, bwd_query, group_id, temporal_cfg
                )
                new_fwd = merger.add(fwd_result.facts)
                new_bwd = merger.add(bwd_result.facts)
                new_uuids = new_fwd | new_bwd

                hops_raw.append(
                    MultihopHop(
                        query=f"{fwd_query} | {bwd_query}",
                        temporal=fwd_result,
                        new_uuids=new_uuids,
                    )
                )

                stop, reason = stop_criterion.should_stop(
                    hop_idx, len(new_uuids), dynamic=True
                )
                if stop:
                    stop_reason = reason
                    break
                if hop_idx == cli_args.max_hops - 1:
                    break

                new_fwd_q, new_bwd_q = await bridge_gen.dual(
                    query, merger.sorted_facts(), dynamic=True
                )
                if not new_fwd_q and not new_bwd_q:
                    stop_reason = "llm_stop"
                    break
                fwd_query = new_fwd_q or query
                bwd_query = new_bwd_q or query

        hops = [_make_hop_trace(i, h) for i, h in enumerate(hops_raw)]
        return MethodResult(
            method_id="sequential-dual",
            query=query,
            group_id=group_id,
            facts=merger.sorted_facts(),
            hops=hops,
            stop_reason=stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("sequential-dual", query, group_id, exc)


async def run_fast_sequential_dual(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
    rerank_embedder: Embedder | None = None,
) -> MethodResult:
    """Sequential dual multihop using fast-ppr (forward push) instead of full PPR.

    ``embedder`` seeds against entity names (e5); ``rerank_embedder`` embeds the
    query in the fact-embedding space (OpenAI) for fact reranking. When omitted,
    both fall back to ``embedder`` — correct only if facts and names share a space.
    """
    timings = StageTimings()
    t_llm = TrackingLLMClient(llm_client, tracker)
    t_emb = TrackingEmbedder(embedder, tracker)
    t_rerank_emb = (
        TrackingEmbedder(rerank_embedder, tracker) if rerank_embedder else t_emb
    )
    temporal_cfg = _temporal_config(cli_args)
    bridge_gen = BridgeGenerator(t_llm, cli_args.llm_model, 15.0)
    merger = FactMerger()
    stop_criterion = StopCriterion(1, cli_args.max_hops)
    hops_raw: list[MultihopHop] = []
    stop_reason = "budget"

    try:
        with timed("total", timings):
            fwd_query = query
            bwd_query = query

            for hop_idx in range(cli_args.max_hops):
                fwd_result = await _temporal_fast_ppr(
                    driver, t_emb, t_llm, fwd_query, group_id, temporal_cfg, timings,
                    push_threshold=cli_args.fast_ppr_push_threshold,
                    rerank_embedder=t_rerank_emb,
                )
                bwd_result = await _temporal_fast_ppr(
                    driver, t_emb, t_llm, bwd_query, group_id, temporal_cfg, timings,
                    push_threshold=cli_args.fast_ppr_push_threshold,
                    rerank_embedder=t_rerank_emb,
                )
                new_fwd = merger.add(fwd_result.facts)
                new_bwd = merger.add(bwd_result.facts)
                new_uuids = new_fwd | new_bwd

                hops_raw.append(
                    MultihopHop(
                        query=f"{fwd_query} | {bwd_query}",
                        temporal=fwd_result,
                        new_uuids=new_uuids,
                    )
                )

                stop, reason = stop_criterion.should_stop(
                    hop_idx, len(new_uuids), dynamic=True
                )
                if stop:
                    stop_reason = reason
                    break
                if hop_idx == cli_args.max_hops - 1:
                    break

                new_fwd_q, new_bwd_q = await bridge_gen.dual(
                    query, merger.sorted_facts(), dynamic=True
                )
                if not new_fwd_q and not new_bwd_q:
                    stop_reason = "llm_stop"
                    break
                fwd_query = new_fwd_q or query
                bwd_query = new_bwd_q or query

        hops = [_make_hop_trace(i, h) for i, h in enumerate(hops_raw)]
        return MethodResult(
            method_id="fast-dual",
            query=query,
            group_id=group_id,
            facts=merger.sorted_facts(),
            hops=hops,
            stop_reason=stop_reason,
            token_usage=tracker.total(),
            cost=_make_cost(tracker, cli_args),
            timing=timings,
        )
    except Exception as exc:
        return _error_result("fast-dual", query, group_id, exc)


METHOD_REGISTRY: dict[str, Callable] = {
    "ppr": run_ppr_sequential,
    "fast-ppr": run_fast_ppr_sequential,
    "treehop": run_treehop,
    "beam": run_beam,
    "beam-dual": run_beam_dual,
    "dag": run_dag,
    "sequential-dual": run_sequential_dual,
    "fast-dual": run_fast_sequential_dual,
}


def _add_generation_cost(
    result: MethodResult, tracker: TokenTracker, args: ArenaArgs
) -> None:
    """Add the generation stage tokens/cost to an already-finalised MethodResult."""
    gen = tracker.by_stage().get("generate", TokenUsage(0, 0))
    if gen.total == 0:
        return
    pricing = load_pricing()
    result.token_usage = TokenUsage(
        tokens_in=result.token_usage.tokens_in + gen.tokens_in,
        tokens_out=result.token_usage.tokens_out + gen.tokens_out,
    )
    result.cost = CostBreakdown(
        llm_cost_usd=result.cost.llm_cost_usd + compute_cost(gen, args.llm_model, pricing),
        embed_cost_usd=result.cost.embed_cost_usd,
    )


async def execute_methods(
    method_ids: list[str],
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    args: ArenaArgs,
) -> list[MethodResult]:
    """Run each method sequentially. Errors are captured in MethodResult.error, not raised."""
    results: list[MethodResult] = []
    for method_id in method_ids:
        tracker = TokenTracker()
        run_fn = METHOD_REGISTRY[method_id]
        result = await run_fn(driver, embedder, llm_client, query, group_id, args, tracker)
        if args.generate and not result.error:
            result.generated_answer = await generate_answer(
                query, result.facts, llm_client, tracker, args.llm_model, result.timing
            )
            _add_generation_cost(result, tracker, args)
        results.append(result)
    return results


class StubGraphDriver:
    """Minimal in-memory stub satisfying the GraphDriver protocol."""

    async def execute_query(self, query: str, **params: object) -> list[dict]:
        return []


class StubEmbedder:
    """Minimal in-memory stub satisfying the Embedder protocol."""

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 64

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


class StubLLMClient:
    """Minimal in-memory stub satisfying the LLMClient protocol."""

    async def complete(self, prompt: str, *, model: str, **kwargs: object) -> str:
        return "STOP"

    async def complete_json(
        self, prompt: str, schema: dict, *, model: str, **kwargs: object
    ) -> dict:
        return {}
