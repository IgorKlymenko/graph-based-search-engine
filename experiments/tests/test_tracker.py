"""Tests for arena/tracker.py."""
from __future__ import annotations

import asyncio

import pytest

from arena.tracker import (
    TokenTracker,
    TokenUsage,
    TrackingEmbedder,
    TrackingLLMClient,
    count_tokens,
)
from arena.runner import StubEmbedder, StubLLMClient


@pytest.mark.anyio
async def test_token_tracker_records_stages() -> None:
    tracker = TokenTracker()
    tracker.record(stage="llm", tokens_in=100, tokens_out=50, model="gpt-4o-mini")
    tracker.record(stage="embed", tokens_in=30, tokens_out=0, model="embedding")
    by_stage = tracker.by_stage()
    assert "llm" in by_stage
    assert "embed" in by_stage
    assert by_stage["llm"].tokens_in == 100
    assert by_stage["llm"].tokens_out == 50
    assert by_stage["embed"].tokens_in == 30
    assert by_stage["embed"].tokens_out == 0


@pytest.mark.anyio
async def test_token_tracker_total_sums_all() -> None:
    tracker = TokenTracker()
    tracker.record(stage="llm", tokens_in=200, tokens_out=80)
    tracker.record(stage="embed", tokens_in=40, tokens_out=0)
    total = tracker.total()
    assert total.tokens_in == 240
    assert total.tokens_out == 80
    assert total.total == 320


@pytest.mark.anyio
async def test_tracking_llm_client_counts_tokens() -> None:
    inner = StubLLMClient()
    tracker = TokenTracker()
    client = TrackingLLMClient(inner, tracker)
    await client.complete("hello world this is a prompt", model="gpt-4o-mini")
    total = tracker.total()
    # "STOP" response + prompt should both be counted
    assert total.tokens_in > 0
    assert total.tokens_out >= 0


@pytest.mark.anyio
async def test_tracking_embedder_counts_tokens() -> None:
    inner = StubEmbedder()
    tracker = TokenTracker()
    emb = TrackingEmbedder(inner, tracker)
    await emb.embed("some text to embed")
    total = tracker.total()
    assert total.tokens_in > 0
    assert total.tokens_out == 0


@pytest.mark.anyio
async def test_tracking_llm_json_counts_tokens() -> None:
    inner = StubLLMClient()
    tracker = TokenTracker()
    client = TrackingLLMClient(inner, tracker)
    await client.complete_json(
        "generate json response", schema={"type": "object"}, model="gpt-4o-mini"
    )
    by_stage = tracker.by_stage()
    assert "llm_json" in by_stage
    assert by_stage["llm_json"].tokens_in > 0


@pytest.mark.anyio
async def test_token_tracker_thread_safe() -> None:
    """Concurrent async tasks recording to the same tracker produce correct totals."""
    tracker = TokenTracker()

    async def _record(n: int) -> None:
        for _ in range(n):
            tracker.record(stage="llm", tokens_in=1, tokens_out=1)
            await asyncio.sleep(0)

    await asyncio.gather(_record(50), _record(50))
    total = tracker.total()
    assert total.tokens_in == 100
    assert total.tokens_out == 100


def test_count_tokens_fallback_no_tiktoken(monkeypatch) -> None:
    """Without tiktoken (or for unknown model), returns a word-count estimate."""
    import sys
    import importlib

    # Force tiktoken import to fail
    original = sys.modules.get("tiktoken")
    sys.modules["tiktoken"] = None  # type: ignore[assignment]
    try:
        # Reload count_tokens in a fresh call
        result = count_tokens("hello world foo bar baz", "unknown-model-xyz")
        # 5 words * 1.33 ≈ 6 (or max(1, ...))
        assert result >= 1
        assert isinstance(result, int)
    finally:
        if original is None:
            del sys.modules["tiktoken"]
        else:
            sys.modules["tiktoken"] = original
