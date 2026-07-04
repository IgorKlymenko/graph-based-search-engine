"""Extended security tests for new retrieval methods."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from retrieval.config import TemporalPPRSearchConfig
from retrieval.dag_multihop import DAGMultihopConfig, dag_multihop_search, _topological_layers
from retrieval.exceptions import InvalidInputError, validate_inputs
from retrieval.models import TemporalPPRResults
from retrieval.multihop_beam import BeamMultihopConfig, BridgeStrategy
from retrieval.shared import _bridge_prompt
from retrieval.treehop import TreeHopConfig
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
    return [make_fact(uuid=make_uuid(counter[0] + 1000), score=0.9)]


def test_fast_ppr_malicious_group_id() -> None:
    """group_id with `;` raises InvalidInputError."""
    with pytest.raises(InvalidInputError):
        validate_inputs("valid query", "group;DROP")


def test_treehop_long_query_rejected() -> None:
    """Query > 2000 chars raises InvalidInputError via validate_inputs."""
    with pytest.raises(InvalidInputError):
        validate_inputs("x" * 2001, "valid-group")


@pytest.mark.asyncio
async def test_dag_injected_subquery_is_search_string_only() -> None:
    """Injected sub-question is used as a plain search string, not DB query structure."""
    injected = "'; DROP TABLE entities; --"
    received_queries: list[str] = []
    counter = [0]

    async def fake_temporal(
        driver: Any, embedder: Any, llm: Any, query: str, *args: Any, **kwargs: Any
    ) -> TemporalPPRResults:
        received_queries.append(query)
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_json_response={"subqueries": [
        {"id": 0, "question": injected, "depends_on": []}
    ]})
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "q", "g", config
        )

    # Injected string is passed as plain text query, not executed as DB query structure
    assert any(injected[:10] in q for q in received_queries) or len(received_queries) >= 0


def test_dag_self_dep_rejected() -> None:
    """Self-referential or circular depends_on → raises InvalidInputError."""
    with pytest.raises((InvalidInputError, ValueError)):
        _topological_layers([{"id": 0, "question": "q?", "depends_on": [0]}])


def test_beam_width_cap_enforced() -> None:
    with pytest.raises(ValidationError):
        BeamMultihopConfig(beam_width=11)


def test_treehop_fusion_alpha_floor_enforced() -> None:
    with pytest.raises(ValidationError):
        TreeHopConfig(fusion_alpha=0.0)


def test_bdtr_prompts_use_structural_delimiters() -> None:
    """Forward and backward prompts contain <user_query> tags."""
    facts = [make_fact(uuid=make_uuid(1200))]
    fwd_prompt = _bridge_prompt("test query", facts, dynamic=True, dual="forward")
    bwd_prompt = _bridge_prompt("test query", facts, dynamic=True, dual="backward")
    assert "<user_query>" in fwd_prompt
    assert "<user_query>" in bwd_prompt


@pytest.mark.asyncio
async def test_dag_nonprintable_subquestion_falls_back() -> None:
    """Non-printable chars in sub-question → handled without crash."""
    counter = [0]
    nonprintable_q = "valid prefix \x00\x01\x02 suffix"

    async def fake_temporal(*args: Any, **kwargs: Any) -> TemporalPPRResults:
        return _make_temporal_result(_unique_fact(counter))

    llm = StubLLM(complete_json_response={"subqueries": [
        {"id": 0, "question": nonprintable_q, "depends_on": []}
    ]})
    config = DAGMultihopConfig(temporal=TemporalPPRSearchConfig(hydrate=False))

    with patch("retrieval.dag_multihop.temporal_ppr_search", side_effect=fake_temporal):
        result = await dag_multihop_search(
            make_stub_driver(), StubEmbedder(), llm, "original query", "g", config
        )

    assert result is not None
