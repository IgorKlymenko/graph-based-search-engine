"""JSON, JSONL, and Markdown report serialization."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from arena.tracker import HopTrace, MethodResult, StageTimings


def _dt_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timings_dict(t: StageTimings) -> dict[str, float]:
    return {
        "graph_load_ms": round(t.graph_load_ms, 3),
        "seed_collection_ms": round(t.seed_collection_ms, 3),
        "ppr_iteration_ms": round(t.ppr_iteration_ms, 3),
        "rerank_ms": round(t.rerank_ms, 3),
        "temporal_filter_ms": round(t.temporal_filter_ms, 3),
        "hydration_ms": round(t.hydration_ms, 3),
        "bridge_total_ms": round(t.bridge_total_ms, 3),
        "decompose_ms": round(t.decompose_ms, 3),
    }


def _hop_dict(hop: HopTrace) -> dict:
    return {
        "hop_index": hop.hop_index,
        "query": hop.query,
        "ppr_seeds": hop.ppr_seeds,
        "nodes_loaded": hop.nodes_loaded,
        "edges_loaded": hop.edges_loaded,
        "facts_before_temporal": hop.facts_before_temporal,
        "facts_after_temporal": hop.facts_after_temporal,
        "dropped_by_temporal": hop.dropped_by_temporal,
        "new_fact_count": len(hop.new_fact_uuids),
        "bridge_queries": hop.bridge_queries,
        "tokens_in": hop.token_usage.tokens_in,
        "tokens_out": hop.token_usage.tokens_out,
        "timing_ms": round(hop.timing_ms, 3),
    }


def _fact_dict(fact, redact: bool = False, first_hop: int = 0) -> dict:
    text = f"[REDACTED-{fact.uuid}]" if redact else fact.fact
    return {
        "uuid": fact.uuid,
        "fact": text,
        "score": round(fact.score, 6),
        "rerank_score": round(fact.rerank_score, 6) if fact.rerank_score is not None else None,
        "valid_at": _dt_str(fact.valid_at),
        "invalid_at": _dt_str(fact.invalid_at),
        "first_hop": first_hop,
    }


def _first_hop_map(result: MethodResult) -> dict[str, int]:
    """Build UUID → first hop index mapping."""
    mapping: dict[str, int] = {}
    for hop in result.hops:
        for uuid in hop.new_fact_uuids:
            if uuid not in mapping:
                mapping[uuid] = hop.hop_index
    return mapping


def _result_dict(result: MethodResult, redact: bool = False) -> dict:
    first_hop_map = _first_hop_map(result)
    return {
        "method_id": result.method_id,
        "facts": [
            _fact_dict(f, redact, first_hop_map.get(f.uuid, 0)) for f in result.facts
        ],
        "hops": [_hop_dict(h) for h in result.hops],
        "stop_reason": result.stop_reason,
        "tokens_in": result.token_usage.tokens_in,
        "tokens_out": result.token_usage.tokens_out,
        "llm_cost_usd": round(result.cost.llm_cost_usd, 6),
        "embed_cost_usd": round(result.cost.embed_cost_usd, 6),
        "total_cost_usd": round(result.cost.total_usd, 6),
        "total_ms": round(result.timing.total_ms, 3),
        "stage_timings": _timings_dict(result.timing),
        "error": result.error,
    }


def to_json(
    results: Sequence[MethodResult],
    query: str,
    group_id: str,
    temporal_mode: str,
    as_of: datetime | None = None,
    redact: bool = False,
) -> str:
    """Serialize results to JSON string following §9.1 schema."""
    payload = {
        "arena_version": "1.0",
        "query": query,
        "group_id": group_id,
        "temporal_mode": temporal_mode,
        "as_of": _dt_str(as_of),
        "results": [_result_dict(r, redact) for r in results],
    }
    return json.dumps(payload, indent=2)


def to_jsonl_header(methods: list[str], query_count: int) -> str:
    """Return the header line for a JSONL batch report."""
    header = {
        "type": "header",
        "arena_version": "1.0",
        "methods": methods,
        "query_count": query_count,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return json.dumps(header)


def to_jsonl_line(
    results: Sequence[MethodResult],
    query: str,
    group_id: str,
    temporal_mode: str,
    as_of: datetime | None = None,
    redact: bool = False,
) -> str:
    """Return a single JSONL line for one query's results."""
    payload = {
        "arena_version": "1.0",
        "query": query,
        "group_id": group_id,
        "temporal_mode": temporal_mode,
        "as_of": _dt_str(as_of),
        "results": [_result_dict(r, redact) for r in results],
    }
    return json.dumps(payload)


def to_markdown(
    results: Sequence[MethodResult],
    query: str = "",
    group_id: str = "",
    temporal_mode: str = "",
    redact: bool = False,
) -> str:
    """Serialize results to a plain Markdown report."""
    from arena.display import render_markdown

    return render_markdown(results, query, group_id, temporal_mode, redact)
