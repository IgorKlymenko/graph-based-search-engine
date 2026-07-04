"""Tests for retrieval.shared — FactMerger, StopCriterion, BridgeGenerator, EmbeddingFuser."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from retrieval.exceptions import LLMCallError
from retrieval.shared import BridgeGenerator, EmbeddingFuser, FactMerger, StopCriterion
from tests.conftest import StubLLM, make_fact, make_uuid

# ---------------------------------------------------------------------------
# FactMerger
# ---------------------------------------------------------------------------


def test_merger_new_uuids_on_first_add() -> None:
    merger = FactMerger()
    facts = [make_fact(uuid=make_uuid(1)), make_fact(uuid=make_uuid(2))]
    new = merger.add(facts)
    assert new == frozenset({make_uuid(1), make_uuid(2)})


def test_merger_dedup_keeps_best_score() -> None:
    merger = FactMerger()
    merger.add([make_fact(uuid=make_uuid(1), score=0.5, rerank_score=0.5)])
    merger.add([make_fact(uuid=make_uuid(1), score=0.5, rerank_score=0.9)])
    facts = merger.sorted_facts()
    assert len(facts) == 1
    assert facts[0].rerank_score == 0.9


def test_merger_sorted_descending() -> None:
    merger = FactMerger()
    merger.add([
        make_fact(uuid=make_uuid(1), score=0.3, rerank_score=0.3),
        make_fact(uuid=make_uuid(2), score=0.9, rerank_score=0.9),
        make_fact(uuid=make_uuid(3), score=0.6, rerank_score=0.6),
    ])
    scores = [f.rerank_score or f.score for f in merger.sorted_facts()]
    assert scores == sorted(scores, reverse=True)


def test_merger_new_uuids_incremental() -> None:
    merger = FactMerger()
    merger.add([make_fact(uuid=make_uuid(1))])
    new = merger.add([make_fact(uuid=make_uuid(2)), make_fact(uuid=make_uuid(1))])
    assert new == frozenset({make_uuid(2)})


# ---------------------------------------------------------------------------
# StopCriterion
# ---------------------------------------------------------------------------


def test_stop_criterion_budget() -> None:
    sc = StopCriterion(min_new_facts=1, max_hops=3)
    stop, reason = sc.should_stop(hop_idx=2, new_uuid_count=5, dynamic=True)
    assert stop is True
    assert reason == "budget"


def test_stop_criterion_convergence() -> None:
    sc = StopCriterion(min_new_facts=2, max_hops=5)
    stop, reason = sc.should_stop(hop_idx=2, new_uuid_count=1, dynamic=True)
    assert stop is True
    assert reason == "convergence"


def test_stop_no_trigger_at_hop_zero() -> None:
    sc = StopCriterion(min_new_facts=5, max_hops=10)
    stop, reason = sc.should_stop(hop_idx=0, new_uuid_count=0, dynamic=True)
    assert stop is False


# ---------------------------------------------------------------------------
# BridgeGenerator._validate_bridge_output (via single)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_gen_long_output_is_stop() -> None:
    llm = StubLLM(complete_response="x" * 501)
    gen = BridgeGenerator(llm, "gpt-4o-mini", timeout=5.0)
    result = await gen.single("query", [], dynamic=True)
    assert result == ""


@pytest.mark.asyncio
async def test_bridge_gen_stop_keyword_case_insensitive() -> None:
    for resp in ["STOP", "stop", "Stop"]:
        llm = StubLLM(complete_response=resp)
        gen = BridgeGenerator(llm, "gpt-4o-mini", timeout=5.0)
        result = await gen.single("query", [], dynamic=True)
        assert result == "", f"Expected '' for response {resp!r}"


@pytest.mark.asyncio
async def test_bridge_gen_nonprintable_is_stop() -> None:
    llm = StubLLM(complete_response="\x00\x01\x02")
    gen = BridgeGenerator(llm, "gpt-4o-mini", timeout=5.0)
    result = await gen.single("query", [], dynamic=True)
    assert result == ""


@pytest.mark.asyncio
async def test_bridge_gen_llm_timeout_returns_none() -> None:
    async def slow_complete(prompt: str, *, model: str, **kwargs: Any) -> str:
        await asyncio.sleep(10.0)
        return "never"

    class TimeoutLLM:
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            return await slow_complete(prompt, model=model, **kwargs)

        async def complete_json(self, prompt: str, schema: dict, *, model: str, **kwargs: Any) -> dict:
            return {}

    gen = BridgeGenerator(TimeoutLLM(), "gpt-4o-mini", timeout=0.01)
    result = await gen.single("query", [], dynamic=True)
    assert result is None


@pytest.mark.asyncio
async def test_bridge_gen_dual_fires_two_concurrent_calls() -> None:
    call_times: list[float] = []

    class TrackingLLM:
        async def complete(self, prompt: str, *, model: str, **kwargs: Any) -> str:
            import time
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return "bridge query"

        async def complete_json(self, prompt: str, schema: dict, *, model: str, **kwargs: Any) -> dict:
            return {}

    gen = BridgeGenerator(TrackingLLM(), "gpt-4o-mini", timeout=5.0)
    fwd, bwd = await gen.dual("query", [], dynamic=True)
    assert len(call_times) == 2
    # Both calls should have started within 50ms of each other (concurrent)
    assert abs(call_times[1] - call_times[0]) < 0.05


# ---------------------------------------------------------------------------
# EmbeddingFuser
# ---------------------------------------------------------------------------


def test_fuser_output_is_unit_norm() -> None:
    fuser = EmbeddingFuser(alpha=0.6)
    q = [0.6, 0.8, 0.0, 0.0]
    facts = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    result = fuser.fuse(q, facts)
    norm = sum(x * x for x in result) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_fuser_empty_facts_returns_query_emb() -> None:
    fuser = EmbeddingFuser(alpha=0.6)
    q = [0.6, 0.8, 0.0, 0.0]
    result = fuser.fuse(q, [])
    assert result == q


def test_fuser_alpha_zero_raises() -> None:
    with pytest.raises(ValueError):
        EmbeddingFuser(alpha=0.0)


def test_fuser_alpha_one_raises() -> None:
    with pytest.raises(ValueError):
        EmbeddingFuser(alpha=1.0)
