"""Token usage tracking, cost breakdown, timing, and wrapping clients."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from retrieval.interfaces import Embedder, LLMClient
from retrieval.models import Fact

if TYPE_CHECKING:
    pass


def count_tokens(text: str, model: str) -> int:
    """Count tokens using tiktoken; fall back to word-count estimate on any failure."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))


@dataclass(frozen=True)
class TokenUsage:
    tokens_in: int
    tokens_out: int

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(self.tokens_in + other.tokens_in, self.tokens_out + other.tokens_out)


@dataclass(frozen=True)
class CostBreakdown:
    llm_cost_usd: float
    embed_cost_usd: float

    @property
    def total_usd(self) -> float:
        return self.llm_cost_usd + self.embed_cost_usd


@dataclass
class StageTimings:
    graph_load_ms: float = 0.0
    seed_collection_ms: float = 0.0
    ppr_iteration_ms: float = 0.0
    rerank_ms: float = 0.0
    temporal_filter_ms: float = 0.0
    hydration_ms: float = 0.0
    bridge_total_ms: float = 0.0
    decompose_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class HopTrace:
    hop_index: int
    query: str
    ppr_seeds: list[str]
    nodes_loaded: int
    edges_loaded: int
    facts_before_temporal: int
    facts_after_temporal: int
    dropped_by_temporal: int
    new_fact_uuids: frozenset[str]
    all_fact_uuids: frozenset[str]
    bridge_queries: list[str]
    token_usage: TokenUsage
    timing_ms: float
    expansion_terms: list[str] = field(default_factory=list)
    hyde_answers: list[str] = field(default_factory=list)


@dataclass
class MethodResult:
    method_id: str
    query: str
    group_id: str
    facts: list[Fact]
    hops: list[HopTrace]
    stop_reason: str
    token_usage: TokenUsage
    cost: CostBreakdown
    timing: StageTimings
    error: str | None = None
    generated_answer: str | None = None

    @property
    def fact_uuids(self) -> frozenset[str]:
        return frozenset(f.uuid for f in self.facts)


class TokenTracker:
    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(
        self, *, stage: str, tokens_in: int, tokens_out: int, model: str = ""
    ) -> None:
        self._records.append(
            {"stage": stage, "in": tokens_in, "out": tokens_out, "model": model}
        )

    def total(self) -> TokenUsage:
        return TokenUsage(
            tokens_in=sum(r["in"] for r in self._records),
            tokens_out=sum(r["out"] for r in self._records),
        )

    def by_stage(self) -> dict[str, TokenUsage]:
        stages: dict[str, list[dict]] = defaultdict(list)
        for r in self._records:
            stages[r["stage"]].append(r)
        return {
            stage: TokenUsage(
                tokens_in=sum(r["in"] for r in records),
                tokens_out=sum(r["out"] for r in records),
            )
            for stage, records in stages.items()
        }


class TrackingLLMClient:
    """Wraps any LLMClient, recording token usage per call."""

    def __init__(self, inner: LLMClient, tracker: TokenTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    async def complete(self, prompt: str, *, model: str, **kwargs: object) -> str:
        result = await self._inner.complete(prompt, model=model, **kwargs)
        tokens_in = count_tokens(prompt, model)
        tokens_out = count_tokens(result, model)
        self._tracker.record(
            stage="llm", tokens_in=tokens_in, tokens_out=tokens_out, model=model
        )
        return result

    async def complete_json(
        self, prompt: str, schema: dict, *, model: str, **kwargs: object
    ) -> dict:
        result = await self._inner.complete_json(prompt, schema, model=model, **kwargs)
        tokens_in = count_tokens(prompt, model)
        tokens_out = count_tokens(str(result), model)
        self._tracker.record(
            stage="llm_json", tokens_in=tokens_in, tokens_out=tokens_out, model=model
        )
        return result


class TrackingEmbedder:
    """Wraps any Embedder, recording token usage per call."""

    def __init__(self, inner: Embedder, tracker: TokenTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    async def embed(self, text: str) -> list[float]:
        result = await self._inner.embed(text)
        self._tracker.record(
            stage="embed", tokens_in=count_tokens(text, "embedding"), tokens_out=0
        )
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = await self._inner.embed_batch(texts)
        total_in = sum(count_tokens(t, "embedding") for t in texts)
        self._tracker.record(stage="embed_batch", tokens_in=total_in, tokens_out=0)
        return results


@contextmanager
def timed(stage: str, timings: StageTimings) -> Iterator[None]:
    """Context manager that records elapsed ms for `stage` into a StageTimings object."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    setattr(timings, f"{stage}_ms", elapsed_ms)
