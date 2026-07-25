from __future__ import annotations

import asyncio
import logging

from retrieval.config import MultihopSearchConfig, SearchFilters
from retrieval.exceptions import LLMCallError, validate_inputs
from retrieval.interfaces import Embedder, GraphDriver, LLMClient
from retrieval.models import Fact, MultihopHop, MultihopSearchResults
from retrieval.temporal import temporal_ppr_search

log = logging.getLogger(__name__)

_MAX_FACTS_IN_PROMPT = 50
_MAX_BRIDGE_LEN = 500


async def multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: MultihopSearchConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    """Sequential multihop search with LLM bridge queries."""
    validate_inputs(query, group_id)
    if config.min_new_facts == 0:
        log.warning("min_new_facts=0: convergence stopping is disabled; loop runs to max_hops")
    merged: dict[str, Fact] = {}
    hops: list[MultihopHop] = []
    stop_reason = "budget"

    async def _run() -> None:
        nonlocal stop_reason
        current_query = query
        hop_limit = config.fixed_hops if not config.dynamic else config.max_hops

        for hop_idx in range(hop_limit):
            temporal_result = await temporal_ppr_search(
                driver, embedder, llm_client, current_query, group_id, config.temporal, filters
            )

            before = set(merged.keys())
            for f in temporal_result.facts:
                prev = merged.get(f.uuid)
                if prev is None or (f.rerank_score or 0.0) > (prev.rerank_score or 0.0):
                    merged[f.uuid] = f

            new_uuids = set(merged.keys()) - before
            hops.append(
                MultihopHop(
                    query=current_query,
                    temporal=temporal_result,
                    new_uuids=frozenset(new_uuids),
                )
            )

            if hop_idx > 0 and len(new_uuids) < config.min_new_facts:
                stop_reason = "convergence"
                return

            if hop_idx == hop_limit - 1:
                stop_reason = "fixed" if not config.dynamic else "budget"
                return

            bridge = await _bridge_question(
                query=query,
                collected_facts=list(merged.values()),
                model=config.bridge_model,
                dynamic=config.dynamic,
                llm_client=llm_client,
                timeout=config.bridge_timeout_seconds,
            )

            if bridge is None:
                stop_reason = "llm_error"
                return
            if bridge == "":
                stop_reason = "llm_stop"
                return
            current_query = bridge

    try:
        await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)
    except asyncio.TimeoutError:
        stop_reason = "timeout"

    sorted_facts = sorted(
        merged.values(), key=lambda f: f.rerank_score if f.rerank_score is not None else f.score, reverse=True
    )
    return MultihopSearchResults(hops=hops, merged=sorted_facts, stop_reason=stop_reason)


async def _bridge_question(
    query: str,
    collected_facts: list[Fact],
    model: str,
    dynamic: bool,
    llm_client: LLMClient,
    timeout: float,
) -> str | None:
    """
    Returns: non-empty str = next query, '' = STOP or invalid, None = LLM error/timeout.
    """
    stop_instruction = (
        "- If the collected facts are sufficient to answer the original question, output exactly:\n"
        "  STOP\n"
        if dynamic
        else ""
    )
    truncated = collected_facts[:_MAX_FACTS_IN_PROMPT]
    formatted_facts = "\n".join(
        f"- {f.fact[:500]}  [valid: {f.valid_at} → {f.invalid_at or 'present'}]"
        for f in truncated
    )
    prompt = (
        "<instructions>\n"
        "You are a search assistant retrieving knowledge from a graph database.\n"
        "Your task is to generate ONE concise follow-up search query that retrieves\n"
        "information missing from the facts collected so far.\n"
        "\n"
        "Rules:\n"
        "- Target entities or relationships not already covered by the collected facts.\n"
        "- Do NOT restate the original question.\n"
        "- Output only the follow-up query string — no explanation, no preamble.\n"
        f"{stop_instruction}"
        "</instructions>\n"
        "\n"
        "<user_query>\n"
        f"{query}\n"
        "</user_query>\n"
        "\n"
        "<collected_facts>\n"
        f"{formatted_facts}\n"
        "</collected_facts>"
    )
    try:
        raw = await asyncio.wait_for(
            llm_client.complete(prompt, model=model, temperature=0.0, seed=42),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, LLMCallError):
        return None

    return _validate_bridge_output(raw)


def _validate_bridge_output(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped.upper() == "STOP":
        return ""
    if len(stripped) > _MAX_BRIDGE_LEN:
        return ""
    return stripped
