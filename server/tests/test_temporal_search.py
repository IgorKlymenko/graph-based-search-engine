"""Tests for retrieval.temporal — temporal layer with stub PPR."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from retrieval.config import TemporalMode, TemporalPPRSearchConfig
from retrieval.models import Fact, TemporalPPRResults
from retrieval.temporal import temporal_ppr_search
from tests.conftest import StubDriver, StubEmbedder, make_fact, make_uuid, make_stub_driver

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_PAST = datetime(2024, 1, 1, tzinfo=timezone.utc)
_FUTURE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Valid UUIDs for temporal test facts
_UUID_CURRENT = make_uuid(100)
_UUID_STALE = make_uuid(101)
_UUID_NODE_0 = "a0000000-0000-0000-0000-000000000000"
_UUID_NODE_1 = "a0000000-0000-0000-0000-000000000001"


def _make_current_fact_row(uid: str = _UUID_CURRENT) -> dict[str, Any]:
    return {
        "uuid": uid,
        "fact": "Current fact",
        "source_uuid": _UUID_NODE_0,
        "target_uuid": _UUID_NODE_1,
        "score": 0.9,
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "valid_at": None,
        "invalid_at": None,  # currently_true
        "expired_at": None,
        "episodes": [],
        "embedding": [0.6, 0.8] + [0.0] * 14,
    }


def _make_stale_fact_row(uid: str = _UUID_STALE) -> dict[str, Any]:
    return {
        "uuid": uid,
        "fact": "Stale fact",
        "source_uuid": _UUID_NODE_0,
        "target_uuid": _UUID_NODE_1,
        "score": 0.7,
        "created_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        "valid_at": None,
        "invalid_at": _PAST,  # already invalid
        "expired_at": None,
        "episodes": [],
        "embedding": [0.6, 0.8] + [0.0] * 14,
    }


def _make_two_node_driver(fact_rows: list[dict[str, Any]]) -> StubDriver:
    return StubDriver(
        node_rows=[
            {"uuid": _UUID_NODE_0, "name": "A", "summary": ""},
            {"uuid": _UUID_NODE_1, "name": "B", "summary": ""},
        ],
        edge_rows=[{"source_uuid": _UUID_NODE_0, "target_uuid": _UUID_NODE_1}],
        fact_rows=fact_rows,
        default_rows=[{"uuid": _UUID_NODE_0, "name": "A", "score": 1.0}],
    )


@pytest.mark.asyncio
async def test_stale_facts_are_dropped() -> None:
    driver = _make_two_node_driver([_make_current_fact_row(), _make_stale_fact_row()])
    config = TemporalPPRSearchConfig(
        mode=TemporalMode.currently_true,
        hydrate=False,
    )
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    uuids = [f.uuid for f in result.facts]
    assert _UUID_CURRENT in uuids
    assert _UUID_STALE not in uuids


@pytest.mark.asyncio
async def test_dropped_by_temporal_count() -> None:
    driver = _make_two_node_driver([_make_current_fact_row(), _make_stale_fact_row()])
    config = TemporalPPRSearchConfig(mode=TemporalMode.currently_true, hydrate=False)
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    assert result.dropped_by_temporal == 1


@pytest.mark.asyncio
async def test_relevant_count_is_pre_filter() -> None:
    driver = _make_two_node_driver([_make_current_fact_row(), _make_stale_fact_row()])
    config = TemporalPPRSearchConfig(mode=TemporalMode.currently_true, hydrate=False)
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    # relevant_count = total PPR facts before temporal filter
    assert result.relevant_count >= result.dropped_by_temporal + len(result.facts)


@pytest.mark.asyncio
async def test_as_of_defaults_to_now() -> None:
    driver = make_stub_driver(n_nodes=2, n_edges=1)
    config = TemporalPPRSearchConfig(mode=TemporalMode.all, as_of=None, hydrate=False)
    before = datetime.now(timezone.utc)
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    after = datetime.now(timezone.utc)
    assert before <= result.as_of <= after + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_max_facts_truncates() -> None:
    n_facts = 5
    fact_rows = [
        {
            "uuid": make_uuid(200 + i),
            "fact": f"Fact {i}",
            "source_uuid": _UUID_NODE_0,
            "target_uuid": _UUID_NODE_1,
            "score": 0.9 - i * 0.01,
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "valid_at": None,
            "invalid_at": None,
            "expired_at": None,
            "episodes": [],
            "embedding": [0.6, 0.8] + [0.0] * 14,
        }
        for i in range(n_facts)
    ]
    driver = StubDriver(
        node_rows=[
            {"uuid": _UUID_NODE_0, "name": "A", "summary": ""},
            {"uuid": _UUID_NODE_1, "name": "B", "summary": ""},
        ],
        edge_rows=[{"source_uuid": _UUID_NODE_0, "target_uuid": _UUID_NODE_1}],
        fact_rows=fact_rows,
        default_rows=[{"uuid": _UUID_NODE_0, "name": "A", "score": 1.0}],
    )
    config = TemporalPPRSearchConfig(
        mode=TemporalMode.all,
        max_facts=2,
        hydrate=False,
    )
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    assert len(result.facts) <= 2


@pytest.mark.asyncio
async def test_hydrate_false_skips_passage_fetch() -> None:
    fact_with_episode = {
        "uuid": make_uuid(300),
        "fact": "Fact with episode",
        "source_uuid": _UUID_NODE_0,
        "target_uuid": _UUID_NODE_1,
        "score": 0.9,
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
        "episodes": ["ep-uuid-1"],
        "embedding": [0.6, 0.8] + [0.0] * 14,
    }
    driver = _make_two_node_driver([fact_with_episode])
    config = TemporalPPRSearchConfig(mode=TemporalMode.all, hydrate=False)
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    # No episode query should have been issued
    episode_calls = [c for c in driver.calls if "e.uuid AS uuid, e.name AS name" in c[0]]
    assert len(episode_calls) == 0
    assert result.passages == []


@pytest.mark.asyncio
async def test_mode_forwarded_to_result() -> None:
    driver = make_stub_driver(n_nodes=2, n_edges=1)
    config = TemporalPPRSearchConfig(mode=TemporalMode.changed_since, hydrate=False)
    result = await temporal_ppr_search(
        driver, StubEmbedder(), None, "query", "g1", config
    )
    assert result.mode == TemporalMode.changed_since.value
