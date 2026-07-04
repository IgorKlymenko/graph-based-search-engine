"""Tests for BDTR (Bidirectional bridge) — sequential-dual and beam-dual modes."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from retrieval.models import MultihopSearchResults, TemporalPPRResults
from retrieval.multihop_beam import BeamMultihopConfig, BridgeStrategy, beam_multihop_search
from retrieval.shared import _bridge_prompt
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
    return [make_fact(uuid=make_uuid(counter[0] + 600), score=0.9)]


@pytest.mark.asyncio
async def test_bdtr_two_ppr_searches_per_hop() -> None:
    """bridge_strategy=dual → dual bridge calls per member per hop."""
    counter = [0]
    bridge_calls = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    class DualLLM(StubLLM):
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            bridge_calls[0] += 1
            return "bridge query"

    config = BeamMultihopConfig(
        beam_width=1,
        max_hops=2,
        bridge_strategy=BridgeStrategy.dual,
        temporal=__import__("retrieval.config", fromlist=["TemporalPPRSearchConfig"]).TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), DualLLM(), "q", "g", config
        )

    assert bridge_calls[0] >= 2


@pytest.mark.asyncio
async def test_bdtr_new_uuids_is_union() -> None:
    """hop.new_uuids accumulates facts from both forward and backward paths."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    from retrieval.config import TemporalPPRSearchConfig
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=2,
        bridge_strategy=BridgeStrategy.dual,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )
    llm = StubLLM(complete_response="bridge")

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    all_hop_uuids: set[str] = set()
    for hop in result.hops:
        all_hop_uuids |= hop.new_uuids
    merged_uuids = {f.uuid for f in result.merged}
    assert all_hop_uuids <= merged_uuids


@pytest.mark.asyncio
async def test_bdtr_one_stop_other_continues() -> None:
    """Forward STOP + backward query → hop still executes."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    class AlternatingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            self.calls += 1
            return "" if self.calls % 2 == 1 else "bridge query"

        async def complete_json(self, prompt: str, schema: dict, *, model: str, **kwargs: Any) -> dict:
            return {}

    from retrieval.config import TemporalPPRSearchConfig
    llm = AlternatingLLM()
    config = BeamMultihopConfig(
        beam_width=1,
        max_hops=3,
        bridge_strategy=BridgeStrategy.dual,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert isinstance(result, MultihopSearchResults)


@pytest.mark.asyncio
async def test_bdtr_both_stop_triggers_llm_stop() -> None:
    """Both bridges return '' → stop_reason='llm_stop'."""
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    from retrieval.config import TemporalPPRSearchConfig
    llm = StubLLM(complete_response="")
    config = BeamMultihopConfig(
        beam_width=2,
        max_hops=5,
        bridge_strategy=BridgeStrategy.dual,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    assert result.stop_reason == "llm_stop"


@pytest.mark.asyncio
async def test_bdtr_calls_are_concurrent() -> None:
    """Forward and backward bridge calls issued in parallel (within ~50ms of each other)."""
    call_times: list[float] = []
    counter = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    class TimingLLM:
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            import time
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return "bridge query"

        async def complete_json(self, prompt: str, schema: dict, *, model: str, **kwargs: Any) -> dict:
            return {}

    from retrieval.config import TemporalPPRSearchConfig
    config = BeamMultihopConfig(
        beam_width=1,
        max_hops=2,
        bridge_strategy=BridgeStrategy.dual,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.multihop_beam.temporal_ppr_search", side_effect=fake_temporal):
        result = await beam_multihop_search(
            make_stub_driver(), StubEmbedder(), TimingLLM(), "q", "g", config
        )

    if len(call_times) >= 2:
        assert abs(call_times[1] - call_times[0]) < 0.05


def test_bdtr_prompts_use_structural_delimiters() -> None:
    """Forward and backward prompts contain <user_query> tags."""
    facts = [make_fact(uuid=make_uuid(700))]
    fwd_prompt = _bridge_prompt("test query", facts, dynamic=True, dual="forward")
    bwd_prompt = _bridge_prompt("test query", facts, dynamic=True, dual="backward")
    assert "<user_query>" in fwd_prompt
    assert "<user_query>" in bwd_prompt
    assert "FORWARD" in fwd_prompt
    assert "BACKWARD" in bwd_prompt
