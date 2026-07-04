"""Tests for retrieval.filters — temporal predicate logic."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from retrieval.config import TemporalMode
from retrieval.filters import make_aware, temporal_match
from tests.conftest import make_fact, UUID_FACT_1

_T = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_BEFORE = datetime(2024, 1, 1, tzinfo=timezone.utc)
_AFTER = datetime(2024, 12, 31, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# temporal_match
# ---------------------------------------------------------------------------

def test_all_mode_returns_everything() -> None:
    fact = make_fact(valid_at=_BEFORE, invalid_at=_BEFORE)
    assert temporal_match(fact, TemporalMode.all, _T) is True


def test_currently_true_with_null_invalid_at() -> None:
    fact = make_fact(invalid_at=None)
    assert temporal_match(fact, TemporalMode.currently_true, _T) is True


def test_currently_true_with_set_invalid_at() -> None:
    fact = make_fact(invalid_at=_AFTER)
    assert temporal_match(fact, TemporalMode.currently_true, _T) is False


def test_as_of_both_bounds_set() -> None:
    fact = make_fact(valid_at=_BEFORE, invalid_at=_AFTER)
    assert temporal_match(fact, TemporalMode.as_of, _T) is True


def test_as_of_both_bounds_set_outside() -> None:
    fact = make_fact(valid_at=_AFTER, invalid_at=None)
    assert temporal_match(fact, TemporalMode.as_of, _T) is False


def test_as_of_valid_at_null() -> None:
    """None valid_at means valid from the beginning of time."""
    fact = make_fact(valid_at=None, invalid_at=_AFTER)
    assert temporal_match(fact, TemporalMode.as_of, _T) is True


def test_as_of_boundary_inclusive_valid_at() -> None:
    """valid_at == as_of should return True."""
    fact = make_fact(valid_at=_T, invalid_at=_AFTER)
    assert temporal_match(fact, TemporalMode.as_of, _T) is True


def test_as_of_boundary_exclusive_invalid_at() -> None:
    """invalid_at == as_of should return False (strict: i > t required)."""
    fact = make_fact(valid_at=_BEFORE, invalid_at=_T)
    assert temporal_match(fact, TemporalMode.as_of, _T) is False


def test_changed_since_invalid_at_before_as_of() -> None:
    fact = make_fact(invalid_at=_BEFORE)
    assert temporal_match(fact, TemporalMode.changed_since, _T) is False


def test_changed_since_invalid_at_after_as_of() -> None:
    fact = make_fact(invalid_at=_AFTER)
    assert temporal_match(fact, TemporalMode.changed_since, _T) is True


def test_changed_since_invalid_at_null() -> None:
    fact = make_fact(invalid_at=None)
    assert temporal_match(fact, TemporalMode.changed_since, _T) is False


# ---------------------------------------------------------------------------
# make_aware
# ---------------------------------------------------------------------------

def test_make_aware_none() -> None:
    assert make_aware(None) is None


def test_make_aware_naive() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    result = make_aware(naive)
    assert result is not None
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc


def test_make_aware_already_aware() -> None:
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = make_aware(aware)
    assert result == aware


def test_make_aware_non_utc_timezone() -> None:
    """Non-UTC timezone is preserved unchanged."""
    eastern = timezone(timedelta(hours=-5))
    dt = datetime(2024, 6, 1, 10, 0, 0, tzinfo=eastern)
    result = make_aware(dt)
    assert result == dt
    assert result is not None
    assert result.tzinfo == eastern
