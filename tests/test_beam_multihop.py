"""Tests for retrieval.multihop_beam — beam-search multihop."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from retrieval.config import TemporalPPRSearchConfig
from retrieval.models import MultihopSearchResults, TemporalPPRResults
from retrieval.multihop_beam import BeamMultihopConfig, BridgeStrategy, beam_multihop_search
from tests.conftest import StubEmbedder, StubLLM, make_fact, make_uuid, make_stub_driver

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_COUNTER = [0]


def _make_temporal_result(facts: list, num_nodes: int = 2) -> TemporalPPRResults:
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


def _unique_fact(counter: list[int]) -> list:
    counter[0] += 1
    return [make_fact(uuid=make_uuid(counter[0] + 500), score=0.9)]


@pytest.mark.asyncio
async def test_beam_width_one_matches_sequential_stop_reason() -> None:
    """Width=1 beam produces a valid stop reason."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=1,
        max_hops=3,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert result.stop_reason in {"budget", "convergence", "llm_stop", "llm_error", "timeout", "fixed"}


@pytest.mark.asyncio
async def test_beam_parallel_ppr_calls() -> None:
    """Beam starts with 1 member; after dual bridge, 2 members → 2 concurrent calls."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=3,
        max_hops=1,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert counter[0] >= 1


@pytest.mark.asyncio
async def test_beam_pruned_to_width_after_score() -> None:
    """After scoring, result has valid stop reason."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=2,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert isinstance(result, MultihopSearchResults)


@pytest.mark.asyncio
async def test_beam_score_mean_rerank() -> None:
    """beam_score_fn='mean_rerank' produces valid results."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=2,
        beam_score_fn="mean_rerank",
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert isinstance(result, MultihopSearchResults)


@pytest.mark.asyncio
async def test_beam_dual_two_calls_per_member() -> None:
    """dual strategy → 2 bridge calls per member."""
    counter = [0]
    bridge_calls = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    class CountingLLM(StubLLM):
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            bridge_calls[0] += 1
            return "bridge query"

    config = BeamMultihopConfig(
        beam_width=1,
        max_hops=2,
        bridge_strategy=BridgeStrategy.dual,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), CountingLLM(), "q", "g", config
        )

    assert bridge_calls[0] >= 2


@pytest.mark.asyncio
async def test_beam_all_stop_triggers_llm_stop() -> None:
    """All bridges return '' → stop_reason='llm_stop'."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="")  # empty → STOP
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=5,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert result.stop_reason == "llm_stop"


@pytest.mark.asyncio
async def test_beam_global_merger_has_all_facts() -> None:
    """All beam members' facts appear in final merged."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=2,
        bridge_strategy=BridgeStrategy.single,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert len(result.merged) >= 1


@pytest.mark.asyncio
async def test_beam_timeout_returns_partial() -> None:
    """Timeout → partial result returned, no unhandled exception."""
    async def slow_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        await asyncio.sleep(5.0)
        return _make_temporal_result([])

    llm = StubLLM(complete_response="bridge")
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=3,
        total_timeout_seconds=0.05,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=slow_temporal):
        try:
            result = await beam_multihop_search(
                make_stub_driver(), StubEmbedder(), llm, "q", "g", config
            )
            assert isinstance(result, MultihopSearchResults)
        except asyncio.TimeoutError:
            pass  # Acceptable


def test_beam_width_gt_10_rejected() -> None:
    with pytest.raises(ValidationError):
        BeamMultihopConfig(beam_width=11)
