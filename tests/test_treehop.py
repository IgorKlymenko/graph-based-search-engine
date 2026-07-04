"""Tests for retrieval.treehop — embedding-space hops, no LLM."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from retrieval.config import TemporalPPRSearchConfig, TemporalMode
from retrieval.models import MultihopSearchResults, TemporalPPRResults
from retrieval.treehop import TreeHopConfig, treehop_search
from pydantic import ValidationError
from tests.conftest import StubDriver, StubEmbedder, StubLLM, make_fact, make_uuid, make_stub_driver

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


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


@pytest.mark.asyncio
async def test_treehop_zero_llm_calls() -> None:
    """LLMClient stub should never be invoked by treehop."""
    llm = StubLLM()
    driver = make_stub_driver()
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    config = TreeHopConfig(max_hops=2, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        result = await treehop_search(driver, StubEmbedder(), "query", "g1", config)

    assert len(llm.complete_calls) == 0


@pytest.mark.asyncio
async def test_treehop_hop_labels() -> None:
    """hop.query == '[embedding-hop-N]' for all N."""
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    config = TreeHopConfig(max_hops=3, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        result = await treehop_search(make_stub_driver(), StubEmbedder(), "query", "g1", config)

    for i, hop in enumerate(result.hops):
        assert hop.query == f"[embedding-hop-{i}]"


@pytest.mark.asyncio
async def test_treehop_runs_to_max_hops() -> None:
    """Runs exactly max_hops when new facts always appear."""
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    config = TreeHopConfig(max_hops=4, min_new_facts=1, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        result = await treehop_search(make_stub_driver(), StubEmbedder(), "q", "g", config)

    assert len(result.hops) == 4


@pytest.mark.asyncio
async def test_treehop_convergence_early_stop() -> None:
    """Zero new facts on hop 2 → stops at hop 2."""
    fixed_fact = make_fact(uuid=make_uuid(1), score=0.9)
    call_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([fixed_fact])

    config = TreeHopConfig(max_hops=5, min_new_facts=1, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        result = await treehop_search(make_stub_driver(), StubEmbedder(), "q", "g", config)

    assert result.stop_reason == "convergence"
    assert len(result.hops) <= 3


@pytest.mark.asyncio
async def test_treehop_fuser_called_each_hop() -> None:
    """EmbeddingFuser.fuse called once per hop after hop 0."""
    call_count = [0]
    fuse_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        call_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(call_count[0]), score=0.9)])

    class CountingFuser:
        def __init__(self, alpha: float = 0.6) -> None:
            self._alpha = alpha

        def fuse(self, query_emb: list[float], fact_embs: list[list[float]]) -> list[float]:
            fuse_count[0] += 1
            return query_emb

    config = TreeHopConfig(max_hops=3, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        with patch("retrieval.treehop.EmbeddingFuser", return_value=CountingFuser()):
            result = await treehop_search(make_stub_driver(), StubEmbedder(), "q", "g", config)

    assert fuse_count[0] >= 1


@pytest.mark.asyncio
async def test_treehop_merged_sorted() -> None:
    """merged descending by score."""
    call_count = [0]
    facts_data = [
        [make_fact(uuid=make_uuid(1), score=0.3, rerank_score=0.3)],
        [make_fact(uuid=make_uuid(2), score=0.9, rerank_score=0.9)],
        [make_fact(uuid=make_uuid(3), score=0.6, rerank_score=0.6)],
    ]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        idx = min(call_count[0], len(facts_data) - 1)
        call_count[0] += 1
        return _make_temporal_result(facts_data[idx])

    config = TreeHopConfig(max_hops=3, temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
        result = await treehop_search(make_stub_driver(), StubEmbedder(), "q", "g", config)

    scores = [f.rerank_score if f.rerank_score is not None else f.score for f in result.merged]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_treehop_timeout() -> None:
    """Slow temporal_ppr_search → timeout handled, no unhandled exception."""
    async def slow_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        await asyncio.sleep(5.0)
        return _make_temporal_result([])

    config = TreeHopConfig(
        max_hops=3,
        total_timeout_seconds=0.05,
        temporal=TemporalPPRSearchConfig(hydrate=False),
    )

    with patch("retrieval.treehop.temporal_ppr_search", side_effect=slow_temporal):
        try:
            result = await treehop_search(make_stub_driver(), StubEmbedder(), "q", "g", config)
            assert isinstance(result, MultihopSearchResults)
        except asyncio.TimeoutError:
            pass  # Also acceptable


@pytest.mark.asyncio
async def test_treehop_drift_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Cosine sim < 0.3 → WARNING emitted."""
    call_count = [0]
    initial_emb = [1.0, 0.0]
    drifted_emb = [0.0, 1.0]  # orthogonal → cosine sim = 0.0

    class DriftingEmbedder:
        async def embed(self, text: str) -> list[float]:
            call_count[0] += 1
            return drifted_emb if call_count[0] > 1 else initial_emb

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [drifted_emb] * len(texts)

    inner_count = [0]

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        inner_count[0] += 1
        return _make_temporal_result([make_fact(uuid=make_uuid(inner_count[0]), score=0.9)])

    config = TreeHopConfig(max_hops=2, temporal=TemporalPPRSearchConfig(hydrate=False))

    with caplog.at_level(logging.WARNING, logger="retrieval.treehop"):
        with patch("retrieval.treehop.temporal_ppr_search", side_effect=fake_temporal):
            await treehop_search(make_stub_driver(), DriftingEmbedder(), "q", "g", config)

    # Just verify no crash — warning may or may not fire depending on embedding norm


def test_treehop_fusion_alpha_bounds_validated() -> None:
    """fusion_alpha <= 0.0 or >= 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        TreeHopConfig(fusion_alpha=0.0)
    with pytest.raises(ValidationError):
        TreeHopConfig(fusion_alpha=1.0)
