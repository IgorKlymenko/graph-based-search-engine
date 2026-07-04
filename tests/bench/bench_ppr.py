"""PPR algorithm benchmarks — in-process, no database."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from retrieval.config import PPRSearchConfig
from retrieval.graph_loader import GraphEdge, GraphNode
from retrieval.ppr import (
    _build_personalization_vector,
    _build_transition_matrix,
    _run_power_iteration,
)


def _make_nodes(n: int) -> list[GraphNode]:
    return [GraphNode(uuid=f"n{i}", name=f"Node{i}", summary="") for i in range(n)]


def _make_ring_edges(n: int) -> list[GraphEdge]:
    return [GraphEdge(source_uuid=f"n{i}", target_uuid=f"n{(i + 1) % n}") for i in range(n)]


def _make_random_edges(n: int, m: int, seed: int = 42) -> list[GraphEdge]:
    rng = np.random.default_rng(seed)
    nodes = [f"n{i}" for i in range(n)]
    edges = []
    for _ in range(m):
        src = nodes[int(rng.integers(0, n))]
        tgt = nodes[int(rng.integers(0, n))]
        if src != tgt:
            edges.append(GraphEdge(source_uuid=src, target_uuid=tgt))
    return edges


def _build_idx(nodes: list[GraphNode]) -> dict[str, int]:
    sorted_uuids = sorted(n.uuid for n in nodes)
    return {uuid: i for i, uuid in enumerate(sorted_uuids)}


# ---------------------------------------------------------------------------
# 1k nodes, 5k edges — transition matrix build
# ---------------------------------------------------------------------------


def bench_ppr_transition_matrix_1k_5k(benchmark: pytest.FixtureRequest) -> None:
    """PPR transition matrix, 1k nodes, 5k edges. Target: < 50ms."""
    n, m = 1_000, 5_000
    nodes = _make_nodes(n)
    edges = _make_random_edges(n, m)
    idx = _build_idx(nodes)

    def _run() -> csr_matrix:
        return _build_transition_matrix(edges, idx, n)

    result = benchmark(_run)
    assert result.shape == (n, n)


# ---------------------------------------------------------------------------
# 1k nodes, 5k edges — power iteration
# ---------------------------------------------------------------------------


def bench_ppr_power_iteration_1k_5k(benchmark: pytest.FixtureRequest) -> None:
    """PPR power iteration, 1k nodes, 5k edges. Target: < 200ms."""
    n, m = 1_000, 5_000
    nodes = _make_nodes(n)
    edges = _make_random_edges(n, m)
    idx = _build_idx(nodes)
    M = _build_transition_matrix(edges, idx, n)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    config = PPRSearchConfig()

    def _run() -> np.ndarray:
        return _run_power_iteration(M, s, config)

    result = benchmark(_run)
    assert abs(result.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 10k nodes, 50k edges — power iteration
# ---------------------------------------------------------------------------


def bench_ppr_power_iteration_10k_50k(benchmark: pytest.FixtureRequest) -> None:
    """PPR power iteration, 10k nodes, 50k edges. Target: < 2s."""
    n, m = 10_000, 50_000
    nodes = _make_nodes(n)
    edges = _make_random_edges(n, m)
    idx = _build_idx(nodes)
    M = _build_transition_matrix(edges, idx, n)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    config = PPRSearchConfig()

    def _run() -> np.ndarray:
        return _run_power_iteration(M, s, config)

    result = benchmark(_run)
    assert abs(result.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Temporal filter — 10k facts
# ---------------------------------------------------------------------------


def bench_temporal_filter_10k(benchmark: pytest.FixtureRequest) -> None:
    """Temporal filter, 10k facts. Target: < 10ms."""
    from datetime import datetime, timezone
    from retrieval.config import TemporalMode
    from retrieval.filters import temporal_match
    from tests.conftest import make_fact, make_uuid

    as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
    facts = [make_fact(uuid=make_uuid(i + 1), invalid_at=None) for i in range(10_000)]

    def _run() -> list:
        return [f for f in facts if temporal_match(f, TemporalMode.currently_true, as_of)]

    result = benchmark(_run)
    assert len(result) == 10_000
