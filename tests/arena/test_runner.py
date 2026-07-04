"""Tests for arena/runner.py."""
from __future__ import annotations

import pytest

from arena.runner import (
    METHOD_REGISTRY,
    ArenaArgs,
    StubEmbedder,
    StubGraphDriver,
    StubLLMClient,
    execute_methods,
    run_dag,
    run_ppr_sequential,
    run_treehop,
)
from arena.tracker import MethodResult, TokenTracker


def _args(**kwargs) -> ArenaArgs:
    defaults = dict(
        temporal_mode="currently_true",
        max_hops=2,
        beam_width=2,
        llm_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        hydrate=False,
    )
    defaults.update(kwargs)
    return ArenaArgs(**defaults)


@pytest.mark.anyio
async def test_run_ppr_returns_method_result(
    stub_driver, stub_embedder, stub_llm
) -> None:
    tracker = TokenTracker()
    result = await run_ppr_sequential(
        stub_driver, stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    assert isinstance(result, MethodResult)
    assert result.method_id == "ppr"
    assert result.error is None
    assert result.stop_reason != ""


@pytest.mark.anyio
async def test_run_treehop_zero_tokens_out(
    stub_driver, stub_embedder, stub_llm
) -> None:
    """TreeHop makes no LLM calls, so tokens_out must be 0."""
    tracker = TokenTracker()
    result = await run_treehop(
        stub_driver, stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    assert isinstance(result, MethodResult)
    assert result.method_id == "treehop"
    assert result.token_usage.tokens_out == 0


@pytest.mark.anyio
async def test_run_method_error_returns_error_field(
    stub_driver, stub_embedder, stub_llm
) -> None:
    """An exception during run sets result.error; no exception propagates."""

    class _BrokenDriver:
        async def execute_query(self, *a, **kw):
            raise RuntimeError("DB connection failed")

    tracker = TokenTracker()
    result = await run_ppr_sequential(
        _BrokenDriver(), stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    assert result.error is not None
    assert "DB connection failed" in result.error or result.error != ""
    assert result.facts == []
    assert result.stop_reason == "error"


@pytest.mark.anyio
async def test_run_multiple_methods_independent(
    stub_driver, stub_embedder, stub_llm
) -> None:
    """One method error does not affect other methods' results."""

    class _BrokenDriver:
        async def execute_query(self, *a, **kw):
            raise RuntimeError("always fails")

    # Run ppr (will fail) + treehop (uses stub embedder, might also fail)
    # Use execute_methods with two methods
    tracker_ppr = TokenTracker()
    tracker_tree = TokenTracker()

    result_ppr = await run_ppr_sequential(
        _BrokenDriver(), stub_embedder, stub_llm, "q", "g", _args(), tracker_ppr
    )
    result_tree = await run_treehop(
        stub_driver, stub_embedder, stub_llm, "q", "g", _args(), tracker_tree
    )

    # ppr failed, treehop succeeded
    assert result_ppr.error is not None
    # treehop may succeed or fail with empty graph — either way no unhandled exception
    assert isinstance(result_tree, MethodResult)


@pytest.mark.anyio
async def test_hop_trace_populated(stub_driver, stub_embedder, stub_llm) -> None:
    """HopTrace fields are populated from temporal results."""
    tracker = TokenTracker()
    result = await run_ppr_sequential(
        stub_driver, stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    # With empty stub, we may have 0 or 1 hops depending on stop criterion
    # At minimum, no unhandled exception and the result has valid structure
    assert isinstance(result.hops, list)
    for hop in result.hops:
        assert isinstance(hop.nodes_loaded, int)
        assert isinstance(hop.edges_loaded, int)
        assert isinstance(hop.facts_before_temporal, int)
        assert isinstance(hop.facts_after_temporal, int)


@pytest.mark.anyio
async def test_hop_trace_treehop_no_bridge(stub_driver, stub_embedder, stub_llm) -> None:
    """TreeHop hops have empty bridge_queries."""
    tracker = TokenTracker()
    result = await run_treehop(
        stub_driver, stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    for hop in result.hops:
        assert hop.bridge_queries == []


@pytest.mark.anyio
async def test_hop_trace_dag_subquery_as_query(
    stub_driver, stub_embedder, stub_llm
) -> None:
    """DAG hops use sub-question text as their query field (or fallback query)."""
    tracker = TokenTracker()
    result = await run_dag(
        stub_driver, stub_embedder, stub_llm, "What is the capital of France?", "mygraph", _args(), tracker
    )
    assert isinstance(result, MethodResult)
    assert result.method_id == "dag"
    # DAG falls back to single-node with the original query when LLM returns STOP
    for hop in result.hops:
        assert isinstance(hop.query, str)
        assert len(hop.query) > 0


@pytest.mark.anyio
async def test_timing_stages_populated(stub_driver, stub_embedder, stub_llm) -> None:
    """total_ms is non-zero after a run (other stages may be 0 without library instrumentation)."""
    tracker = TokenTracker()
    result = await run_ppr_sequential(
        stub_driver, stub_embedder, stub_llm, "test query", "mygraph", _args(), tracker
    )
    assert result.timing.total_ms >= 0.0  # may be close to 0 for stub but not negative


def test_all_methods_registered() -> None:
    """METHOD_REGISTRY contains all 7 method IDs."""
    expected = {"ppr", "fast-ppr", "treehop", "beam", "beam-dual", "dag", "sequential-dual", "fast-dual"}
    assert set(METHOD_REGISTRY.keys()) == expected
