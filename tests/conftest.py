"""Shared stubs and helpers for the retrieval library test suite."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from retrieval.models import Fact


# ---------------------------------------------------------------------------
# UUID helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

_UUID_TEMPLATE = "00000000-0000-0000-0000-{:012x}"


def make_uuid(n: int = 1) -> str:
    """Return a deterministic valid UUID string for integer n."""
    return _UUID_TEMPLATE.format(n)


# Common UUID constants for tests
UUID_FACT_1 = make_uuid(1)
UUID_FACT_2 = make_uuid(2)
UUID_FACT_3 = make_uuid(3)
UUID_NODE_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID_NODE_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# make_fact helper
# ---------------------------------------------------------------------------


def make_fact(**overrides: Any) -> Fact:
    """Build a valid Fact with sensible defaults; keyword overrides applied."""
    defaults: dict[str, Any] = {
        "uuid": UUID_FACT_1,
        "fact": "Some fact text",
        "source_uuid": UUID_NODE_A,
        "target_uuid": UUID_NODE_B,
        "score": 0.8,
        "rerank_score": None,
        "created_at": _NOW,
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
        "episodes": [],
    }
    defaults.update(overrides)
    return Fact(**defaults)


# ---------------------------------------------------------------------------
# StubDriver
# ---------------------------------------------------------------------------

class StubDriver:
    """Deterministic in-memory graph driver stub.

    Returns ``node_rows`` for node queries, ``edge_rows`` for edge queries,
    ``fact_rows`` for fact queries, and ``[]`` for anything else unless
    ``default_rows`` is set.
    """

    def __init__(
        self,
        node_rows: list[dict[str, Any]] | None = None,
        edge_rows: list[dict[str, Any]] | None = None,
        fact_rows: list[dict[str, Any]] | None = None,
        default_rows: list[dict[str, Any]] | None = None,
        raise_on_query: Exception | None = None,
    ) -> None:
        self.node_rows: list[dict[str, Any]] = node_rows or []
        self.edge_rows: list[dict[str, Any]] = edge_rows or []
        self.fact_rows: list[dict[str, Any]] = fact_rows or []
        self.default_rows: list[dict[str, Any]] = default_rows or []
        self.raise_on_query = raise_on_query
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, _sql: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((_sql, params))
        if self.raise_on_query is not None:
            raise self.raise_on_query
        if "n.uuid AS uuid, n.name AS name, n.summary AS summary" in _sql:
            return self.node_rows
        # fact query also contains "e.source_node_uuid", so check facts first
        if "e.uuid AS uuid, e.fact AS fact" in _sql:
            return self.fact_rows
        if "e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid" in _sql:
            return self.edge_rows
        # fulltext / vector seed queries
        if "queryNodes" in _sql or "vector.similarity" in _sql:
            return self.default_rows
        # episode hydration
        if "e.uuid AS uuid, e.name AS name, e.content AS content" in _sql:
            return self.default_rows
        return self.default_rows


def _node_uuid(i: int) -> str:
    return f"a0000000-0000-0000-0000-{i:012x}"


def _edge_uuid(i: int) -> str:
    return f"b0000000-0000-0000-0000-{i:012x}"


def _fact_uuid(i: int) -> str:
    return f"c0000000-0000-0000-0000-{i:012x}"


def make_stub_driver(
    n_nodes: int = 3,
    n_edges: int = 2,
    with_facts: bool = True,
) -> StubDriver:
    """Convenience factory: build a StubDriver with synthetic node/edge/fact rows."""
    node_rows = [
        {"uuid": _node_uuid(i), "name": f"Node {i}", "summary": f"Summary {i}"}
        for i in range(n_nodes)
    ]
    edge_rows = [
        {"source_uuid": _node_uuid(i), "target_uuid": _node_uuid((i + 1) % n_nodes)}
        for i in range(min(n_edges, n_nodes))
    ]
    fact_rows: list[dict[str, Any]] = []
    if with_facts and n_nodes >= 2:
        fact_rows = [
            {
                "uuid": _fact_uuid(i),
                "fact": f"Fact text {i}",
                "source_uuid": _node_uuid(i % n_nodes),
                "target_uuid": _node_uuid((i + 1) % n_nodes),
                "score": 0.8,
                "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "valid_at": None,
                "invalid_at": None,
                "expired_at": None,
                "episodes": [],
                "embedding": [0.6, 0.8] + [0.0] * 14,
            }
            for i in range(min(3, n_nodes - 1))
        ]
    return StubDriver(node_rows=node_rows, edge_rows=edge_rows, fact_rows=fact_rows)


# ---------------------------------------------------------------------------
# StubEmbedder
# ---------------------------------------------------------------------------

_UNIT_VECTOR: list[float] = [0.6, 0.8] + [0.0] * 14  # 16-dim, L2-norm=1.0


class StubEmbedder:
    """Returns a fixed unit-norm vector for every text."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector if vector is not None else list(_UNIT_VECTOR)
        self.embed_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return list(self._vector)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        for t in texts:
            self.embed_calls.append(t)
        return [list(self._vector) for _ in texts]


# ---------------------------------------------------------------------------
# StubLLM
# ---------------------------------------------------------------------------

class StubLLM:
    """Returns a configurable string; records all calls."""

    def __init__(
        self,
        complete_response: str = "follow-up query",
        complete_json_response: dict[str, Any] | None = None,
        raise_on_call: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._response = complete_response
        self._json_response = complete_json_response or {"subqueries": []}
        self._raise = raise_on_call
        self._delay = delay_seconds
        self.complete_calls: list[tuple[str, str]] = []  # (prompt, model)
        self.complete_json_calls: list[tuple[str, str]] = []

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
    ) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.complete_calls.append((prompt, model))
        if self._raise is not None:
            raise self._raise
        return self._response

    async def complete_json(
        self,
        prompt: str,
        schema: dict,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.complete_json_calls.append((prompt, model))
        if self._raise is not None:
            raise self._raise
        return self._json_response


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_driver() -> StubDriver:
    return make_stub_driver()


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()
