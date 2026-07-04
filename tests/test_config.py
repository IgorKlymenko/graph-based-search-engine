"""Tests for retrieval.config — Pydantic validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from retrieval.config import (
    MultihopSearchConfig,
    PPRSearchConfig,
    SearchFilters,
)
from retrieval.exceptions import InvalidInputError
from tests.conftest import make_fact


def test_alpha_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        PPRSearchConfig(alpha=0.0)


def test_alpha_one_rejected() -> None:
    with pytest.raises(ValidationError):
        PPRSearchConfig(alpha=1.0)


def test_num_seeds_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        PPRSearchConfig(num_seeds=0)


def test_max_hops_over_limit() -> None:
    with pytest.raises(ValidationError):
        MultihopSearchConfig(max_hops=21)


def test_node_label_injection() -> None:
    # SearchFilters raises InvalidInputError (not ValidationError) from the validator
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(node_labels=["x`) DETACH DELETE n //"])


def test_edge_type_injection() -> None:
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(edge_types=["REL`]"])


def test_node_label_empty_string() -> None:
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(node_labels=[""])


def test_node_label_starts_with_digit() -> None:
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(node_labels=["1Node"])


def test_valid_node_label() -> None:
    f = SearchFilters(node_labels=["EntityPerson"])
    assert f.node_labels == ["EntityPerson"]


def test_fact_score_clamped_above_one() -> None:
    fact = make_fact(score=2.0)
    assert fact.score == 1.0


def test_fact_score_clamped_below_zero() -> None:
    fact = make_fact(score=-0.5)
    assert fact.score == 0.0
