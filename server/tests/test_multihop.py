"""Tests for retrieval.multihop — sequential multihop orchestration."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from retrieval.config import MultihopSearchConfig, TemporalPPRSearchConfig, TemporalMode
from retrieval.models import Fact, MultihopSearchResults, TemporalPPRResults
from retrieval.multihop import multihop_search
from tests.conftest import StubDriver, StubEmbedder, StubLLM, make_fact, make_uuid, make_stub_driver

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_temporal_result(facts: list[Fact], num_nodes: int = 2) -> TemporalPPRResults:
    return TemporalPPRResults(
        facts=facts,
        mode="currently_true",
        as_of=_NOW,
        relevant_count=len(facts),
        dropped_by_temporal=0,
        num_seeds=1,
        num_nodes=num_nodes,
        num_edges=1,
    )


@pytest.mark.asyncio
async def test_fixed_hops_exact_count() -> None:
    """dynamic=False, fixed_hops=3 → exactly 3 hops."""
    hop_count = 0
    facts_per_hop = [
        [make_fact(uuid=make_uuid(i * 3 + j), score=0.9 - j * 0.1) for j in range(2)]
        for i in range(3)
    ]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        nonlocal hop_count
        f = facts_per_hop[min(hop_count, 2)]
        hop_count += 1
        return _make_temporal_result(f)

    driver = make_stub_driver()
    llm = StubLLM(complete_response="bridge query")
    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=3,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(driver, StubEmbedder(), llm, "query", "g1", config)

    assert len(result.hops) == 3


@pytest.mark.asyncio
async def test_fixed_stop_reason() -> None:
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        # Different fact each hop so convergence never triggers
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0] + 50))])

    llm = StubLLM(complete_response="next query")
    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=2,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "fixed"


@pytest.mark.asyncio
async def test_dynamic_llm_stop() -> None:
    """Bridge returns '' on hop 2 → stop_reason='llm_stop'."""
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    llm = StubLLM(complete_response="")
    config = MultihopSearchConfig(
        dynamic=True,
        max_hops=5,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "llm_stop"
    assert len(result.hops) >= 1


@pytest.mark.asyncio
async def test_dynamic_convergence() -> None:
    """Zero new facts on hop 2 → stop_reason='convergence'."""
    fixed_fact = make_fact(uuid=make_uuid(1), score=0.9)
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([fixed_fact])

    llm = StubLLM(complete_response="next query")
    config = MultihopSearchConfig(
        dynamic=True,
        max_hops=5,
        min_new_facts=1,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "convergence"


@pytest.mark.asyncio
async def test_dynamic_budget_exhausted() -> None:
    """New facts every hop, bridge never stops → stop_reason='budget'."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        counter[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(counter[0]), score=0.9)])

    llm = StubLLM(complete_response="keep going")
    config = MultihopSearchConfig(
        dynamic=True,
        max_hops=3,
        min_new_facts=1,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "budget"


@pytest.mark.asyncio
async def test_llm_error_stop_reason() -> None:
    """Bridge returns None → stop_reason='llm_error'."""
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    from retrieval.exceptions import LLMCallError

    llm = StubLLM(raise_on_call=LLMCallError("fail"))
    config = MultihopSearchConfig(
        dynamic=True,
        max_hops=5,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "llm_error"


@pytest.mark.asyncio
async def test_merge_best_rerank_score_wins() -> None:
    """Same UUID from two hops → higher rerank_score kept."""
    shared_uuid = make_uuid(99)
    hop1 = [make_fact(uuid=shared_uuid, score=0.5, rerank_score=0.5)]
    hop2 = [make_fact(uuid=shared_uuid, score=0.5, rerank_score=0.9)]
    hops_data = [hop1, hop2]
    call_idx = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        idx = min(call_idx[0], len(hops_data) - 1)
        call_idx[0] += 1
        return _make_temporal_result(hops_data[idx])

    llm = StubLLM(complete_response="bridge")
    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=2,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    merged = {f.uuid: f for f in result.merged}
    assert merged[shared_uuid].rerank_score == 0.9


@pytest.mark.asyncio
async def test_merge_sorted_descending() -> None:
    facts = [
        make_fact(uuid=make_uuid(1), score=0.3, rerank_score=0.3),
        make_fact(uuid=make_uuid(2), score=0.9, rerank_score=0.9),
        make_fact(uuid=make_uuid(3), score=0.6, rerank_score=0.6),
    ]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(facts)

    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=1,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(
            make_stub_driver(), StubEmbedder(), StubLLM(), "q", "g", config
        )

    scores = [f.rerank_score or f.score for f in result.merged]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_new_uuids_disjoint_across_hops() -> None:
    """Each hop's new_uuids should be disjoint from all previous hops."""
    call_idx = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_idx[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_idx[0] + 100), score=0.9)])

    llm = StubLLM(complete_response="bridge")
    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=3,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    seen: set[str] = set()
    for hop in result.hops:
        assert hop.new_uuids.isdisjoint(seen)
        seen |= hop.new_uuids


@pytest.mark.asyncio
async def test_bridge_called_between_hops() -> None:
    """Bridge stub called hops - 1 times for fixed multihop."""
    call_count = [0]
    bridge_calls = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    class CountingLLM(StubLLM):
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            bridge_calls[0] += 1
            return "next query"

    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=3,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(
            make_stub_driver(), StubEmbedder(), CountingLLM(), "q", "g", config
        )

    assert bridge_calls[0] == 2  # 3 hops → 2 bridges


@pytest.mark.asyncio
async def test_bridge_output_over_500_chars_is_stop() -> None:
    """501-char bridge response → stop_reason='llm_stop'."""
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    long_response = "x" * 501
    llm = StubLLM(complete_response=long_response)
    config = MultihopSearchConfig(
        dynamic=True,
        max_hops=5,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(make_stub_driver(), StubEmbedder(), llm, "q", "g", config)

    assert result.stop_reason == "llm_stop"


@pytest.mark.asyncio
async def test_original_query_on_first_hop() -> None:
    original = "original query text"

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result([make_fact(uuid=make_uuid(1), score=0.9)])

    config = MultihopSearchConfig(
        dynamic=False,
        fixed_hops=1,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await multihop_search(
            make_stub_driver(), StubEmbedder(), StubLLM(), original, "g", config
        )

    assert result.hops[0].query == original
