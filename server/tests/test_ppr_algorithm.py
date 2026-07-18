"""Tests for PPR matrix math — no database required."""
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


def _build_idx(nodes: list[GraphNode]) -> dict[str, int]:
    sorted_uuids = sorted(n.uuid for n in nodes)
    return {uuid: i for i, uuid in enumerate(sorted_uuids)}


def test_empty_graph_returns_empty_results() -> None:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    idx: dict[str, int] = {}
    M = _build_transition_matrix(edges, idx, 0)
    assert M.shape == (0, 0)


def test_single_node_score_is_one() -> None:
    nodes = _make_nodes(1)
    idx = _build_idx(nodes)
    edges: list[GraphEdge] = []
    M = _build_transition_matrix(edges, idx, 1)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, 1)
    config = PPRSearchConfig()
    p = _run_power_iteration(M, s, config)
    assert abs(p[0] - 1.0) < 1e-6


def test_two_nodes_mass_flows_to_neighbor() -> None:
    nodes = _make_nodes(2)
    idx = _build_idx(nodes)
    edges = [GraphEdge(source_uuid="n0", target_uuid="n1")]
    M = _build_transition_matrix(edges, idx, 2)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, 2)
    config = PPRSearchConfig(alpha=0.15)
    p = _run_power_iteration(M, s, config)
    # n1 should have non-zero score
    assert p[idx["n1"]] > 0.0


def test_dangling_node_no_mass_sink() -> None:
    """A node with no out-edges (dangling) should not accumulate all mass."""
    nodes = _make_nodes(3)
    idx = _build_idx(nodes)
    # n0 → n1, but n2 is dangling (no edges)
    edges = [GraphEdge(source_uuid="n0", target_uuid="n1")]
    M = _build_transition_matrix(edges, idx, 3)
    s = np.ones(3) / 3.0
    config = PPRSearchConfig()
    p = _run_power_iteration(M, s, config)
    # No single node should have score = 1.0
    assert max(p) < 1.0


def test_scores_sum_to_one() -> None:
    n = 10
    nodes = _make_nodes(n)
    idx = _build_idx(nodes)
    edges = _make_ring_edges(n)
    M = _build_transition_matrix(edges, idx, n)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    config = PPRSearchConfig()
    p = _run_power_iteration(M, s, config)
    assert abs(p.sum() - 1.0) < 1e-6


def test_deterministic_index_mapping() -> None:
    """Same node list gives same idx dict on repeated calls."""
    nodes = _make_nodes(5)
    idx1 = _build_idx(nodes)
    idx2 = _build_idx(nodes)
    assert idx1 == idx2


def test_column_stochastic_matrix() -> None:
    """Each column of M sums to <= 1.0."""
    nodes = _make_nodes(4)
    idx = _build_idx(nodes)
    edges = _make_ring_edges(4)
    M = _build_transition_matrix(edges, idx, 4)
    col_sums = np.asarray(M.sum(axis=0)).flatten()
    assert np.all(col_sums <= 1.0 + 1e-9)


def test_convergence_within_max_iter() -> None:
    n = 100
    nodes = _make_nodes(n)
    idx = _build_idx(nodes)
    edges = _make_ring_edges(n)
    M = _build_transition_matrix(edges, idx, n)
    seed_weights = {"n0": 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    config = PPRSearchConfig(max_iter=100, tol=1e-6)
    p = _run_power_iteration(M, s, config)
    assert abs(p.sum() - 1.0) < 1e-4


def test_alpha_near_one_stays_near_seed() -> None:
    """High alpha means teleportation dominates → top node is the seed."""
    n = 5
    nodes = _make_nodes(n)
    idx = _build_idx(nodes)
    edges = _make_ring_edges(n)
    M = _build_transition_matrix(edges, idx, n)
    seed_uuid = "n0"
    seed_weights = {seed_uuid: 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    config = PPRSearchConfig(alpha=0.99)
    p = _run_power_iteration(M, s, config)
    top_idx = int(np.argmax(p))
    assert sorted(idx.keys())[top_idx] == seed_uuid or p[idx[seed_uuid]] == max(p)


def test_no_seed_matches_graph() -> None:
    """All seeds unknown → uniform personalization."""
    n = 4
    nodes = _make_nodes(n)
    idx = _build_idx(nodes)
    edges = _make_ring_edges(n)
    M = _build_transition_matrix(edges, idx, n)
    seed_weights: dict[str, float] = {"unknown-uuid": 1.0}
    s = _build_personalization_vector(seed_weights, idx, n)
    # Should be uniform
    assert abs(s[0] - 1.0 / n) < 1e-9
    assert abs(s.sum() - 1.0) < 1e-9
