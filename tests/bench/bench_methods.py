"""Multi-method benchmarks — in-process stubs, no database."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from retrieval.config import (
    MultihopSearchConfig,
    TemporalPPRSearchConfig,
    TemporalMode,
)
from retrieval.models import MultihopSearchResults, TemporalPPRResults
from retrieval.multihop_beam import BeamMultihopConfig, BridgeStrategy
from retrieval.treehop import TreeHopConfig
from tests.conftest import StubEmbedder, StubLLM, make_fact, make_uuid, make_stub_driver

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _make_temporal_result(facts: list[Any]) -> TemporalPPRResults:
    return TemporalPPRResults(
        facts=facts,
        mode="currently_true",
        as_of=_NOW,
        relevant_count=len(facts),
        dropped_by_temporal=0,
        num_seeds=1,
        num_nodes=2,
        num_edges=1,
    )


# ---------------------------------------------------------------------------
# Multihop sequential overhead — 4 hops
# ---------------------------------------------------------------------------


def bench_multihop_sequential_4hops(benchmark: pytest.FixtureRequest) -> None:
    """Multihop sequential overhead, 4 hops (stub I/O). Target: < 50ms."""
    from unittest.mock import patch

    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0] + 2000), score=0.9)])

    llm = StubLLM(complete_response="bridge query")
    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=4,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    def _run() -> MultihopSearchResults:
        call_count[0] = 0
        with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
            return asyncio.run(
                __import__("retrieval.multihop", fromlist=["multihop_search"]).multihop_search(
                    make_stub_driver(), StubEmbedder(), llm, "q", "g", config
                )
            )

    result = benchmark(_run)
    assert len(result.hops) == 4


# ---------------------------------------------------------------------------
# Beam width=3, 4 hops parallel
# ---------------------------------------------------------------------------


def bench_beam_width3_4hops(benchmark: pytest.FixtureRequest) -> None:
    """Beam width=3, 4 hops parallel (stub I/O). Target: < 80ms overhead vs sequential."""
    from unittest.mock import patch

    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0] + 2000), score=0.9)])

    llm = StubLLM(complete_response="bridge query")
    config = BeamMultihopConfig(
        beam_width=3,
        max_hops=4,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    def _run() -> MultihopSearchResults:
        call_count[0] = 0
        with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
            return asyncio.run(
                __import__("retrieval.multihop_beam", fromlist=["beam_multihop_search"]).beam_multihop_search(
                    make_stub_driver(), StubEmbedder(), llm, "q", "g", config
                )
            )

    result = benchmark(_run)
    assert isinstance(result, MultihopSearchResults)


# ---------------------------------------------------------------------------
# TreeHop 4 hops, embedding fusion only
# ---------------------------------------------------------------------------


def bench_treehop_4hops(benchmark: pytest.FixtureRequest) -> None:
    """TreeHop 4 hops, embedding fusion only (stub embedder). Target: < 20ms overhead vs hop-0."""
    from unittest.mock import patch

    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0] + 2000), score=0.9)])

    config = TreeHopConfig(
        max_hops=4,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    def _run() -> MultihopSearchResults:
        call_count[0] = 0
        with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
            return asyncio.run(
                __import__("retrieval.treehop", fromlist=["treehop_search"]).treehop_search(
                    make_stub_driver(), StubEmbedder(), "query", "g", config
                )
            )

    result = benchmark(_run)
    assert isinstance(result, MultihopSearchResults)


# ---------------------------------------------------------------------------
# DAG: decompose + 3-layer parallel
# ---------------------------------------------------------------------------


def bench_dag_decompose_3layer(benchmark: pytest.FixtureRequest) -> None:
    """DAG: decompose + 3-layer parallel (stub I/O). Target: < 60ms orchestration overhead."""
    from unittest.mock import patch
    from retrieval.dag_multihop import DAGMultihopConfig, dag_multihop_search

    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0] + 2000), score=0.9)])

    # 3-layer DAG: layer 0 (sq 0,1 parallel), layer 1 (sq 2), layer 2 (sq 3)
    llm = StubLLM(complete_json_response={"subqueries": [
        {"id": 0, "question": "q0?", "depends_on": []},
        {"id": 1, "question": "q1?", "depends_on": []},
        {"id": 2, "question": "q2?", "depends_on": [0, 1]},
        {"id": 3, "question": "q3?", "depends_on": [2]},
    ]})
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    def _run() -> MultihopSearchResults:
        call_count[0] = 0
        with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
            return asyncio.run(
                dag_multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)
            )

    result = benchmark(_run)
    assert isinstance(result, MultihopSearchResults)


# ---------------------------------------------------------------------------
# FAST-PPR top-50, 10k nodes, 50k edges
# ---------------------------------------------------------------------------


def bench_fast_ppr_10k_50k(benchmark: pytest.FixtureRequest) -> None:
    """FAST-PPR top-50, 10k nodes, 50k edges. Target: < 500ms."""
    from retrieval.ppr_approx import _forward_push, _backward_walks, _build_adjacency
    from retrieval.graph_loader import GraphNode, GraphEdge
    import numpy as np

    n, m = 10_000, 50_000
    rng = np.random.default_rng(42)
    nodes = [GraphNode(uuid=f"n{i}", name=f"Node{i}", summary="") for i in range(n)]
    edges = []
    for _ in range(m):
        src = f"n{int(rng.integers(0, n))}"
        tgt = f"n{int(rng.integers(0, n))}"
        if src != tgt:
            edges.append(GraphEdge(source_uuid=src, target_uuid=tgt))

    adj, rev_adj = _build_adjacency(nodes, edges)
    seeds = {"n0": 1.0}
    top_k = 50

    def _run() -> dict[str, float]:
        estimates, residuals = _forward_push(adj, seeds, alpha=0.15, threshold=1e-5)
        for v, r in residuals.items():
            estimates[v] = estimates.get(v, 0.0) + 0.15 * r
        candidates = sorted(estimates.keys(), key=lambda u: estimates[u], reverse=True)[:top_k]
        rng2 = np.random.default_rng(42)
        final: dict[str, float] = {}
        for v in candidates:
            bwd = _backward_walks(rev_adj, v, 200, 0.15, rng2)
            final[v] = estimates.get(v, 0.0) + 0.85 * bwd * seeds.get(v, 0.0)
        return final

    result = benchmark(_run)
    assert len(result) <= top_k
