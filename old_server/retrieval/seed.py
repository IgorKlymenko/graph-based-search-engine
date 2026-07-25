from __future__ import annotations

import asyncio
import logging
from typing import Any

from retrieval.config import PPRSearchConfig
from retrieval.interfaces import Embedder, GraphDriver, LLMClient

log = logging.getLogger(__name__)

_FULLTEXT_QUERY = (
    "CALL db.index.fulltext.queryRelationships('edge_name_and_fact', $query) "
    "YIELD relationship AS e, score "
    "WHERE e.group_id = $group_id "
    "WITH e.source_node_uuid AS uuid, score "
    "ORDER BY score DESC "
    "LIMIT $limit "
    "RETURN uuid, score"
)

_VECTOR_QUERY = (
    "MATCH (n:Entity {group_id: $group_id}) "
    "WITH n, vector.similarity.cosine(n.name_embedding, $embedding) AS score "
    "ORDER BY score DESC "
    "LIMIT $limit "
    "RETURN n.uuid AS uuid, n.name AS name, score"
)

_EXPAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {"type": "array", "items": {"type": "string"}},
        "hyde": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["terms", "hyde"],
}


async def collect_seeds(
    driver: GraphDriver,
    query: str,
    group_id: str,
    config: PPRSearchConfig,
    embedder: Embedder,
    llm_client: LLMClient | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    """
    Returns (seed_weights, expansion_terms, hyde_answers).
    seed_weights values are L1-normalized.
    """
    expansion_terms: list[str] = []
    hyde_answers: list[str] = []

    if config.expand_seeds and llm_client is not None:
        expansion_terms, hyde_answers = await _expand_query_llm(query, config, llm_client)

    all_texts = [query] + expansion_terms + hyde_answers

    # Fan out all embedding calls concurrently; use provided embedding for the raw query
    if query_embedding is not None:
        other_emb_results = await asyncio.gather(
            *[embedder.embed(t) for t in all_texts[1:]],
            return_exceptions=True,
        )
        valid_texts: list[str] = [query]
        all_embeddings: list[list[float]] = [query_embedding]
        for text, emb in zip(all_texts[1:], other_emb_results):
            if isinstance(emb, BaseException):
                log.warning("Embedding failed for expansion term: %s", emb)
            else:
                valid_texts.append(text)
                all_embeddings.append(emb)
        all_texts = valid_texts
    else:
        emb_results = await asyncio.gather(
            *[embedder.embed(t) for t in all_texts],
            return_exceptions=True,
        )
        valid_texts = []
        all_embeddings = []
        for text, emb in zip(all_texts, emb_results):
            if isinstance(emb, BaseException):
                log.warning("Embedding failed: %s", emb)
            else:
                valid_texts.append(text)
                all_embeddings.append(emb)
        all_texts = valid_texts

    weights: dict[str, float] = {}
    if all_texts:
        search_tasks = [
            _hybrid_node_search(driver, text, group_id, config.num_seeds, emb)
            for text, emb in zip(all_texts, all_embeddings)
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        for result in search_results:
            if isinstance(result, BaseException):
                log.warning("Hybrid search failed: %s", result)
                continue
            _accumulate_rrf(weights, result)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights, expansion_terms, hyde_answers


async def _hybrid_node_search(
    driver: GraphDriver,
    query: str,
    group_id: str,
    limit: int,
    embedding: list[float],
) -> list[str]:
    """Run BM25 + cosine search concurrently, fuse with RRF, return UUID list."""
    ft_task = driver.execute_query(
        _FULLTEXT_QUERY, query=query, group_id=group_id, limit=limit
    )
    vec_task = driver.execute_query(
        _VECTOR_QUERY, group_id=group_id, embedding=embedding, limit=limit
    )
    ft_result, vec_result = await asyncio.gather(ft_task, vec_task, return_exceptions=True)

    bm25_uuids = (
        [r["uuid"] for r in ft_result] if not isinstance(ft_result, BaseException) else []
    )
    vector_uuids = (
        [r["uuid"] for r in vec_result] if not isinstance(vec_result, BaseException) else []
    )

    return _rrf_fuse([bm25_uuids, vector_uuids])


def _rrf_fuse(ranked_lists: list[list[str]]) -> list[str]:
    """Fuse multiple ranked UUID lists using RRF (k=60). Returns merged ranked list."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, uuid in enumerate(ranked):
            scores[uuid] = scores.get(uuid, 0.0) + 1.0 / (60 + rank)
    return sorted(scores.keys(), key=lambda u: scores[u], reverse=True)


def _accumulate_rrf(weights: dict[str, float], ranked_uuids: list[str]) -> None:
    """Accumulate seed weights from a ranked list using 1/(r+1) formula."""
    for r, uuid in enumerate(ranked_uuids):
        weights[uuid] = weights.get(uuid, 0.0) + 1.0 / (r + 1)


async def _expand_query_llm(
    query: str,
    config: PPRSearchConfig,
    llm_client: LLMClient,
) -> tuple[list[str], list[str]]:
    """Silent-failure LLM expansion. Returns (terms, hyde_answers) or ([], []) on any error."""
    n = config.expansion_terms
    prompt = (
        "<instructions>\n"
        f"Given the following search query, generate {n} keyword search terms and {n} hypothetical\n"
        "one-sentence answers that would help retrieve relevant knowledge from a graph.\n"
        'Return a JSON object with exactly two keys: "terms" (list of strings) and "hyde" (list of strings).\n'
        "Do not include instructions or explanations in the output — only the JSON object.\n"
        "</instructions>\n"
        "<user_query>\n"
        f"{query}\n"
        "</user_query>"
    )
    try:
        result = await llm_client.complete_json(
            prompt,
            schema=_EXPAND_SCHEMA,
            model=config.expansion_model,
            temperature=0.0,
        )
    except Exception as exc:
        log.warning("LLM expansion failed: %s", exc)
        return [], []

    try:
        raw_terms: list[str] = result.get("terms") or []
        raw_hyde: list[str] = result.get("hyde") or []
        if not isinstance(raw_terms, list) or not isinstance(raw_hyde, list):
            raise ValueError("terms/hyde must be lists")
        terms = [t for t in raw_terms if isinstance(t, str) and 0 < len(t) <= 200][: n]
        hyde = [h for h in raw_hyde if isinstance(h, str) and 0 < len(h) <= 200][: n]
        return terms, hyde
    except Exception as exc:
        log.warning("LLM expansion response invalid: %s", exc)
        return [], []
