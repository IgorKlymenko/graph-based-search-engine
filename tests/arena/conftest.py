"""Shared fixtures and stubs for arena tests."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from arena.runner import StubEmbedder, StubGraphDriver, StubLLMClient
from arena.tracker import (
    CostBreakdown,
    HopTrace,
    MethodResult,
    StageTimings,
    TokenUsage,
)
from retrieval.models import Fact


def _make_uuid() -> str:
    return str(uuid.uuid4())


def make_fact(text: str = "Test fact.", score: float = 0.8) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        uuid=_make_uuid(),
        fact=text,
        source_uuid=_make_uuid(),
        target_uuid=_make_uuid(),
        score=score,
        rerank_score=score,
        created_at=now,
    )


def make_hop_trace(
    hop_index: int = 0,
    query: str = "test query",
    new_uuids: frozenset[str] | None = None,
    nodes_loaded: int = 100,
    edges_loaded: int = 400,
    facts_before: int = 10,
    facts_after: int = 8,
    dropped: int = 2,
) -> HopTrace:
    uuids = new_uuids if new_uuids is not None else frozenset([_make_uuid()])
    return HopTrace(
        hop_index=hop_index,
        query=query,
        ppr_seeds=["Alice", "Bob"],
        nodes_loaded=nodes_loaded,
        edges_loaded=edges_loaded,
        facts_before_temporal=facts_before,
        facts_after_temporal=facts_after,
        dropped_by_temporal=dropped,
        new_fact_uuids=uuids,
        all_fact_uuids=uuids,
        bridge_queries=[],
        token_usage=TokenUsage(tokens_in=100, tokens_out=20),
        timing_ms=50.0,
    )


def make_method_result(
    method_id: str = "ppr",
    query: str = "test query",
    group_id: str = "testgraph",
    n_facts: int = 3,
    n_hops: int = 1,
    stop_reason: str = "convergence",
    error: str | None = None,
    tokens_in: int = 500,
    tokens_out: int = 100,
    llm_cost: float = 0.0001,
    embed_cost: float = 0.00001,
    total_ms: float = 200.0,
) -> MethodResult:
    facts = [make_fact(f"Fact {i}.", score=0.9 - i * 0.05) for i in range(n_facts)]
    hop_uuid_sets = [frozenset(f.uuid for f in facts)]
    hops = [
        make_hop_trace(
            hop_index=i,
            new_uuids=hop_uuid_sets[0] if i == 0 else frozenset(),
        )
        for i in range(n_hops)
    ]
    timings = StageTimings(total_ms=total_ms)
    return MethodResult(
        method_id=method_id,
        query=query,
        group_id=group_id,
        facts=facts,
        hops=hops,
        stop_reason=stop_reason,
        token_usage=TokenUsage(tokens_in=tokens_in, tokens_out=tokens_out),
        cost=CostBreakdown(llm_cost_usd=llm_cost, embed_cost_usd=embed_cost),
        timing=timings,
        error=error,
    )


@pytest.fixture
def stub_driver() -> StubGraphDriver:
    return StubGraphDriver()


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient()


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch):
    """Set ARENA_USE_STUBS=1 for CLI tests."""
    monkeypatch.setenv("ARENA_USE_STUBS", "1")
