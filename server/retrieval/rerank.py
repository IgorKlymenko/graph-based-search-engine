from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from retrieval.config import RerankMode
from retrieval.exceptions import LLMCallError
from retrieval.interfaces import Embedder, LLMClient
from retrieval.models import Fact

log = logging.getLogger(__name__)

_CROSS_ENCODER_BATCH = 20
_DEFAULT_MIN_RELEVANCE: dict[RerankMode, float] = {
    RerankMode.embedding: 0.30,
    RerankMode.cross_encoder: 0.50,
    RerankMode.none: 0.0,
}


@dataclass
class _ScoredFact:
    fact: Fact
    embedding: list[float] | None = field(default=None)


async def rerank_facts(
    query: str,
    candidates: list[_ScoredFact],
    mode: RerankMode,
    embedder: Embedder,
    llm_client: LLMClient | None,
    min_relevance: float | None,
) -> list[Fact]:
    """Return facts with rerank_score set, filtered by min_relevance, sorted descending."""
    threshold = min_relevance if min_relevance is not None else _DEFAULT_MIN_RELEVANCE[mode]

    match mode:
        case RerankMode.embedding:
            return await _rerank_embedding(query, candidates, embedder, threshold)
        case RerankMode.cross_encoder:
            return await _rerank_cross_encoder(query, candidates, llm_client, threshold)
        case RerankMode.none:
            return _rerank_none(candidates, threshold)


async def _rerank_embedding(
    query: str,
    candidates: list[_ScoredFact],
    embedder: Embedder,
    threshold: float,
) -> list[Fact]:
    query_emb = await embedder.embed(query)
    results: list[tuple[Fact, float]] = []
    for sf in candidates:
        if sf.embedding is None:
            continue
        score = _dot(query_emb, sf.embedding)
        results.append((sf.fact, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [
        f.model_copy(update={"rerank_score": score})
        for f, score in results
        if score >= threshold
    ]


async def _rerank_cross_encoder(
    query: str,
    candidates: list[_ScoredFact],
    llm_client: LLMClient | None,
    threshold: float,
) -> list[Fact]:
    if llm_client is None:
        log.warning("cross_encoder rerank requested but no llm_client; falling back to none mode")
        return _rerank_none(candidates, threshold)

    scored: list[tuple[Fact, float]] = []
    for batch_start in range(0, len(candidates), _CROSS_ENCODER_BATCH):
        batch = candidates[batch_start : batch_start + _CROSS_ENCODER_BATCH]
        tasks = [_cross_encoder_call(query, sf.fact, llm_client) for sf in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sf, result in zip(batch, results):
            if isinstance(result, BaseException):
                log.warning("cross_encoder call failed: %s", result)
                score = 0.0
            else:
                score = result
            scored.append((sf.fact, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        f.model_copy(update={"rerank_score": score})
        for f, score in scored
        if score >= threshold
    ]


def _rerank_none(candidates: list[_ScoredFact], threshold: float) -> list[Fact]:
    results = [
        sf.fact.model_copy(update={"rerank_score": sf.fact.score})
        for sf in candidates
        if sf.fact.score >= threshold
    ]
    results.sort(key=lambda f: f.rerank_score or 0.0, reverse=True)
    return results


async def _cross_encoder_call(query: str, fact: Fact, llm_client: LLMClient) -> float:
    prompt = (
        "<instructions>\n"
        "Does the following fact directly answer or strongly support the query?\n"
        "Respond with exactly YES or NO — no other text.\n"
        "</instructions>\n"
        f"<user_query>{query}</user_query>\n"
        f"<retrieved_fact>{fact.fact}</retrieved_fact>"
    )
    try:
        raw = await llm_client.complete(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=4)
    except LLMCallError:
        return 0.0
    return 1.0 if raw.strip().upper() == "YES" else 0.0


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
