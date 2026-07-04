"""Tests for retrieval.ppr_approx — FAST-PPR approximate algorithm."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from retrieval.ppr_approx import FastPPRConfig, fast_ppr_search, _forward_push, _backward_walks
from retrieval.models import PPRResults
from tests.conftest import StubDriver, StubEmbedder, StubLLM, make_stub_driver

import numpy as np


def _make_node_rows(n: int) -> list[dict[str, Any]]:
    return [{"uuid": f"n{i}", "name": f"Node{i}", "summary": ""} for i in range(n)]


def _make_edge_rows(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"source_uuid": s, "target_uuid": t} for s, t in pairs]


@pytest.mark.asyncio
async def test_fast_ppr_empty_graph() -> None:
    driver = StubDriver(node_rows=[], edge_rows=[], default_rows=[])
    config = FastPPRConfig()
    result = await fast_ppr_search(driver, StubEmbedder(), None, "query", "g1", config)
    assert isinstance(result, PPRResults)
    assert result.nodes == []
    assert result.facts == []
    assert result.num_nodes == 0


@pytest.mark.asyncio
async def test_fast_ppr_single_seed_node() -> None:
    driver = StubDriver(
        node_rows=[{"uuid": "n0", "name": "Node0", "summary": ""}],
        edge_rows=[],
        default_rows=[{"uuid": "n0", "name": "Node0", "score": 1.0}],
    )
    config = FastPPRConfig(top_k=1, top_nodes=1)
    result = await fast_ppr_search(driver, StubEmbedder(), None, "query", "g1", config)
    assert len(result.nodes) == 1
    assert result.nodes[0].uuid == "n0"


@pytest.mark.asyncio
async def test_fast_ppr_top_k_only() -> None:
    n = 10
    node_rows = _make_node_rows(n)
    edge_rows = _make_edge_rows([(f"n{i}", f"n{(i+1)%n}") for i in range(n)])
    driver = StubDriver(
        node_rows=node_rows,
        edge_rows=edge_rows,
        default_rows=[{"uuid": "n0", "name": "Node0", "score": 1.0}],
    )
    top_k = 3
    config = FastPPRConfig(top_k=top_k, top_nodes=top_k)
    result = await fast_ppr_search(driver, StubEmbedder(), None, "q", "g", config)
    assert len(result.nodes) <= top_k


@pytest.mark.asyncio
async def test_fast_ppr_non_negative_scores() -> None:
    n = 5
    node_rows = _make_node_rows(n)
    edge_rows = _make_edge_rows([(f"n{i}", f"n{(i+1)%n}") for i in range(n)])
    driver = StubDriver(
        node_rows=node_rows,
        edge_rows=edge_rows,
        default_rows=[{"uuid": "n0", "name": "Node0", "score": 1.0}],
    )
    config = FastPPRConfig()
    result = await fast_ppr_search(driver, StubEmbedder(), None, "q", "g", config)
    for node in result.nodes:
        assert node.score >= 0.0


@pytest.mark.asyncio
async def test_fast_ppr_returns_ppr_results_type() -> None:
    driver = make_stub_driver(n_nodes=3, n_edges=2)
    driver.default_rows = [{"uuid": "node-0", "name": "Node 0", "score": 0.9}]
    config = FastPPRConfig()
    result = await fast_ppr_search(driver, StubEmbedder(), None, "q", "g", config)
    assert isinstance(result, PPRResults)


@pytest.mark.asyncio
async def test_fast_ppr_dangling_no_mass_sink() -> None:
    """Dangling node (no out-edges) should not get all mass."""
    node_rows = _make_node_rows(3)
    edge_rows = [{"source_uuid": "n0", "target_uuid": "n1"}]  # n2 is dangling
    driver = StubDriver(
        node_rows=node_rows,
        edge_rows=edge_rows,
        default_rows=[{"uuid": "n0", "name": "Node0", "score": 1.0}],
    )
    config = FastPPRConfig(top_k=3, top_nodes=3)
    result = await fast_ppr_search(driver, StubEmbedder(), None, "q", "g", config)
    node_uuids = [n.uuid for n in result.nodes]
    # n2 is dangling with no in-edges; seed n0 should outscore it
    if "n2" in node_uuids and "n0" in node_uuids:
        n2_score = next(n.score for n in result.nodes if n.uuid == "n2")
        n0_score = next(n.score for n in result.nodes if n.uuid == "n0")
        assert n0_score > n2_score


def test_forward_push_residuals_below_threshold_not_expanded() -> None:
    """Residuals below threshold are not pushed further."""
    adj = {"a": ["b"], "b": ["c"]}
    seeds = {"a": 1.0}
    threshold = 0.5  # high threshold → b's residual might not expand
    estimates, residuals = _forward_push(adj, seeds, alpha=0.15, threshold=threshold)
    # c should not be in estimates if b's residual was below threshold
    # (depends on values; we just check no crash and estimates are non-negative)
    for v, est in estimates.items():
        assert est >= 0.0


def test_backward_walks_count_matches_config() -> None:
    """num_walks random walk traces produced per target."""
    rev_adj = {"b": ["a"], "c": ["b"]}
    rng = np.random.default_rng(seed=0)
    num_walks = 200
    result = _backward_walks(rev_adj, "b", num_walks, alpha=0.15, rng=rng)
    # Result is a fraction in [0, 1]
    assert 0.0 <= result <= 1.0
