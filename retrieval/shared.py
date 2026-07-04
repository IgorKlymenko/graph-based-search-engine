from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from retrieval.config import RerankMode, TemporalMode
from retrieval.exceptions import LLMCallError
from retrieval.interfaces import LLMClient
from retrieval.models import Fact

log = logging.getLogger(__name__)

_MAX_BRIDGE_LEN = 500
_MAX_FACTS_IN_PROMPT = 50


# ---------------------------------------------------------------------------
# FactMerger
# ---------------------------------------------------------------------------


def _score(f: Fact) -> float:
    return f.rerank_score if f.rerank_score is not None else f.score


class FactMerger:
    """Accumulates facts across hops, keeping the best score per UUID."""

    def __init__(self) -> None:
        self._by_uuid: dict[str, Fact] = {}

    def add(self, facts: Iterable[Fact]) -> frozenset[str]:
        """Add facts; return the set of UUIDs that are new this call."""
        before = set(self._by_uuid.keys())
        for f in facts:
            prev = self._by_uuid.get(f.uuid)
            if prev is None or _score(f) > _score(prev):
                self._by_uuid[f.uuid] = f
        return frozenset(self._by_uuid.keys()) - before

    def sorted_facts(self) -> list[Fact]:
        return sorted(self._by_uuid.values(), key=_score, reverse=True)

    @property
    def all_uuids(self) -> frozenset[str]:
        return frozenset(self._by_uuid.keys())


# ---------------------------------------------------------------------------
# StopCriterion
# ---------------------------------------------------------------------------


@dataclass
class StopCriterion:
    """Encapsulates stopping logic shared by all multihop variants."""

    min_new_facts: int
    max_hops: int

    def should_stop(
        self,
        hop_idx: int,
        new_uuid_count: int,
        dynamic: bool,
    ) -> tuple[bool, str]:
        """Returns (stop, reason). Caller handles llm_stop/llm_error/timeout/fixed."""
        if hop_idx >= self.max_hops - 1:
            return True, "budget"
        if dynamic and hop_idx > 0 and new_uuid_count < self.min_new_facts:
            return True, "convergence"
        return False, ""


# ---------------------------------------------------------------------------
# BridgeGenerator
# ---------------------------------------------------------------------------


def _validate_bridge_output(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped.upper() == "STOP":
        return ""
    if len(stripped) > _MAX_BRIDGE_LEN:
        return ""
    if any(not c.isprintable() for c in stripped):
        return ""
    return stripped


def _bridge_prompt(
    original_query: str,
    collected_facts: list[Fact],
    dynamic: bool,
    dual: Literal[False, "forward", "backward"] = False,
) -> str:
    stop_instruction = (
        "- If the collected facts are sufficient to answer the original question, output exactly:\n"
        "  STOP\n"
        if dynamic
        else ""
    )
    if dual == "forward":
        task = (
            "Your task is to generate ONE concise FORWARD bridge query: given known facts,\n"
            "what entity or relationship would CONNECT to or EXTEND these toward an answer?"
        )
    elif dual == "backward":
        task = (
            "Your task is to generate ONE concise BACKWARD bridge query: working backwards\n"
            "from the original question, what PREREQUISITE fact or relationship is missing?"
        )
    else:
        task = (
            "Your task is to generate ONE concise follow-up search query that retrieves\n"
            "information missing from the facts collected so far."
        )

    truncated = collected_facts[:_MAX_FACTS_IN_PROMPT]
    formatted_facts = "\n".join(
        f"- {f.fact[:500]}  [valid: {f.valid_at} → {f.invalid_at or 'present'}]"
        for f in truncated
    )
    return (
        "<instructions>\n"
        "You are a search assistant retrieving knowledge from a graph database.\n"
        f"{task}\n"
        "\n"
        "Rules:\n"
        "- Target entities or relationships not already covered by the collected facts.\n"
        "- Do NOT restate the original question.\n"
        "- Output only the follow-up query string — no explanation, no preamble.\n"
        f"{stop_instruction}"
        "</instructions>\n"
        "\n"
        "<user_query>\n"
        f"{original_query}\n"
        "</user_query>\n"
        "\n"
        "<collected_facts>\n"
        f"{formatted_facts}\n"
        "</collected_facts>"
    )


class BridgeGenerator:
    """Single place for all LLM bridge query generation."""

    def __init__(self, llm_client: LLMClient, model: str, timeout: float) -> None:
        self._llm = llm_client
        self._model = model
        self._timeout = timeout

    async def single(
        self,
        original_query: str,
        collected_facts: list[Fact],
        dynamic: bool,
    ) -> str | None:
        """Returns: non-empty str = next query, '' = STOP, None = LLM error."""
        prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual=False)
        return await self._call(prompt)

    async def dual(
        self,
        original_query: str,
        collected_facts: list[Fact],
        dynamic: bool,
    ) -> tuple[str | None, str | None]:
        """BDTR: generate forward and backward bridge queries in parallel."""
        forward_prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual="forward")
        backward_prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual="backward")
        results = await asyncio.gather(
            self._call(forward_prompt),
            self._call(backward_prompt),
            return_exceptions=True,
        )
        fwd = results[0] if not isinstance(results[0], BaseException) else None
        bwd = results[1] if not isinstance(results[1], BaseException) else None
        return fwd, bwd

    async def decompose(
        self,
        query: str,
        model: str,
        llm_client: LLMClient,
    ) -> list[dict]:
        """
        DAG decomposition: returns list of {'id': int, 'question': str, 'depends_on': list[int]}.
        Raises LLMCallError on failure.
        """
        prompt = (
            "<instructions>\n"
            f"Decompose the following question into at most 6 sub-questions that together\n"
            "fully answer it. For each sub-question, specify which earlier sub-questions (by index)\n"
            "it depends on. Sub-questions with no dependencies can be answered in parallel.\n"
            "Do not restate the original question as a sub-question.\n"
            'Return JSON:\n'
            '{\n'
            '  "subqueries": [\n'
            '    {"id": 0, "question": "...", "depends_on": []},\n'
            '    {"id": 1, "question": "...", "depends_on": [0]},\n'
            '    ...\n'
            '  ]\n'
            '}\n'
            "</instructions>\n"
            "<user_query>\n"
            f"{query}\n"
            "</user_query>"
        )
        schema = {
            "type": "object",
            "properties": {
                "subqueries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "question": {"type": "string"},
                            "depends_on": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["id", "question", "depends_on"],
                    },
                }
            },
            "required": ["subqueries"],
        }
        try:
            result = await asyncio.wait_for(
                llm_client.complete_json(prompt, schema, model=model, temperature=0.0),
                timeout=self._timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            raise LLMCallError("Decomposition LLM call failed") from exc

        subqueries: list[dict] = result.get("subqueries") or []
        return subqueries

    async def _call(self, prompt: str) -> str | None:
        try:
            raw = await asyncio.wait_for(
                self._llm.complete(prompt, model=self._model, temperature=0.0, seed=42),
                timeout=self._timeout,
            )
        except (asyncio.TimeoutError, LLMCallError):
            return None
        return _validate_bridge_output(raw)


# ---------------------------------------------------------------------------
# EmbeddingFuser
# ---------------------------------------------------------------------------


class EmbeddingFuser:
    """Fuses a query embedding with retrieved fact embeddings for the next TreeHop query vector."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.5 < alpha < 0.95:
            raise ValueError("alpha must be in (0.5, 0.95)")
        self._alpha = alpha

    def fuse(
        self,
        query_emb: list[float],
        fact_embs: list[list[float]],
    ) -> list[float]:
        """next = normalize(alpha * query_emb + (1-alpha) * mean(fact_embs))."""
        if not fact_embs:
            return query_emb
        q = np.array(query_emb, dtype=np.float64)
        F = np.array(fact_embs, dtype=np.float64)
        mean_f = F.mean(axis=0)
        fused = self._alpha * q + (1.0 - self._alpha) * mean_f
        norm = float(np.linalg.norm(fused))
        if norm < 1e-10:
            return query_emb
        return (fused / norm).tolist()
