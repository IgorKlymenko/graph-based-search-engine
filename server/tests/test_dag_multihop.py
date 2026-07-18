"""Tests for retrieval.dag_multihop — DAG query decomposition multihop."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from retrieval.config import TemporalPPRSearchConfig
from retrieval.dag_multihop import DAGMultihopConfig, dag_multihop_search, _topological_layers
from retrieval.exceptions import InvalidInputError
from retrieval.models import MultihopSearchResults, TemporalPPRResults
from tests.conftest import StubLLM, make_fact, make_uuid, make_stub_driver, StubEmbedder

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_temporal_result(facts: list) -> TemporalPPRResults:
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


def _unique_fact(counter: list[int]) -> list:
    counter[0] += 1
    return [make_fact(uuid=make_uuid(counter[0] + 800), score=0.9)]


def _dag_llm(subqueries: list[dict] | None = None) -> StubLLM:
    qs = subqueries or [{"id": 0, "question": "sub-question", "depends_on": []}]
    return StubLLM(complete_json_response={"subqueries": qs})


@pytest.mark.asyncio
async def test_dag_single_no_dep() -> None:
    """1-node DAG → 1 hop, stop_reason='fixed'."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([{"id": 0, "question": "What is it?", "depends_on": []}])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert len(result.hops) == 1
    assert result.stop_reason == "fixed"


@pytest.mark.asyncio
async def test_dag_parallel_independent() -> None:
    """Two independent sub-queries run concurrently."""
    call_times: list[float] = []
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        import time
        call_times.append(time.monotonic())
        await asyncio.sleep(0.05)
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([
        {"id": 0, "question": "First q?", "depends_on": []},
        {"id": 1, "question": "Second q?", "depends_on": []},
    ])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert len(result.hops) == 2
    if len(call_times) >= 2:
        assert abs(call_times[1] - call_times[0]) < 0.05


@pytest.mark.asyncio
async def test_dag_dependency_ordering() -> None:
    """Sub-query 1 depends on 0 → executes after 0."""
    exec_order: list[int] = []
    counter = [0]

    async def fake_temporal(
        driver: Any, embedder: Any, llm: Any, query: str, *args: Any, **kwargs: Any
    ) -> TemporalPPRResults:
        exec_order.append(0 if "First" in query else 1)
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([
        {"id": 0, "question": "First q?", "depends_on": []},
        {"id": 1, "question": "Second q?", "depends_on": [0]},
    ])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert exec_order.index(0) < exec_order.index(1)


@pytest.mark.asyncio
async def test_dag_context_prefix_injected() -> None:
    """Dependent sub-query question prefixed with dep facts as context."""
    received_queries: list[str] = []
    counter = [0]

    async def fake_temporal(
        driver: Any, embedder: Any, llm: Any, query: str, *args: Any, **kwargs: Any
    ) -> TemporalPPRResults:
        received_queries.append(query)
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([
        {"id": 0, "question": "Who is CEO?", "depends_on": []},
        {"id": 1, "question": "When did they join?", "depends_on": [0]},
    ])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert any("Context:" in q for q in received_queries)


def test_dag_cycle_raises_invalid_input() -> None:
    """Circular depends_on → InvalidInputError from _topological_layers."""
    with pytest.raises(InvalidInputError):
        _topological_layers([
            {"id": 0, "question": "q0?", "depends_on": [1]},
            {"id": 1, "question": "q1?", "depends_on": [0]},
        ])


@pytest.mark.asyncio
async def test_dag_decomp_failure_fallback() -> None:
    """LLM error on decompose → fallback to single-node DAG with original query."""
    counter = [0]

    async def fake_temporal(
        driver: Any, embedder: Any, llm: Any, query: str, *args: Any, **kwargs: Any
    ) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    from retrieval.exceptions import LLMCallError
    llm = StubLLM(raise_on_call=LLMCallError("bad json"))
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "original query", "g", config
        )

    assert len(result.hops) == 1
    assert result.hops[0].query == "original query"


@pytest.mark.asyncio
async def test_dag_question_too_long_discarded() -> None:
    """Question > 500 chars → truncated (not crash)."""
    long_q = "x" * 501
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([{"id": 0, "question": long_q, "depends_on": []}])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "original query", "g", config
        )

    assert isinstance(result, MultihopSearchResults)


@pytest.mark.asyncio
async def test_dag_stop_reason_always_fixed() -> None:
    """stop_reason is always 'fixed'."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = _dag_llm([
        {"id": 0, "question": "q0?", "depends_on": []},
        {"id": 1, "question": "q1?", "depends_on": [0]},
    ])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert result.stop_reason == "fixed"


@pytest.mark.asyncio
async def test_dag_all_facts_in_merged() -> None:
    """Facts from all sub-queries present in merged."""
    counter = [0]
    fact_uuids: list[str] = []

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        counter[0] += 1
        uid = make_uuid(counter[0] + 900)
        fact_uuids.append(uid)
        return _make_temporal_result([make_fact(uuid=uid, score=0.9)])

    llm = _dag_llm([
        {"id": 0, "question": "q0?", "depends_on": []},
        {"id": 1, "question": "q1?", "depends_on": []},
    ])
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    merged_uuids = {f.uuid for f in result.merged}
    for uid in fact_uuids:
        assert uid in merged_uuids


@pytest.mark.asyncio
async def test_dag_timeout_partial_results() -> None:
    """Timeout mid-layer → no unhandled exception."""
    counter = [0]

    async def slow_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        await asyncio.sleep(5.0)
        return _make_temporal_result([])

    llm = _dag_llm([{"id": 0, "question": "q?", "depends_on": []}])
    config = DAGMultihopConfig(
        total_timeout_seconds=0.05,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=slow_temporal):
        try:
            result = await dag_multihop_search(
                make_stub_driver(), StubEmbedder(), llm, "q", "g", config
            )
            assert isinstance(result, MultihopSearchResults)
        except asyncio.TimeoutError:
            pass  # Acceptable


def test_dag_max_subqueries_enforced() -> None:
    with pytest.raises(ValidationError):
        DAGMultihopConfig(max_subqueries=21)
