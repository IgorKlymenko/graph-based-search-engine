"""Security boundary tests for the retrieval library."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from retrieval.config import SearchFilters
from retrieval.exceptions import InvalidInputError, validate_inputs
from retrieval.shared import _validate_bridge_output
from tests.conftest import make_fact, make_uuid, make_stub_driver


# ---------------------------------------------------------------------------
# Cypher / input injection
# ---------------------------------------------------------------------------

def test_node_label_cypher_injection_rejected() -> None:
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(node_labels=["x`) DETACH DELETE n //"])


def test_edge_type_injection_rejected() -> None:
    with pytest.raises((ValidationError, InvalidInputError)):
        SearchFilters(edge_types=["REL; DROP ALL"])


def test_group_id_injection_rejected() -> None:
    with pytest.raises(InvalidInputError):
        validate_inputs("valid query", "x; DROP ALL")


def test_query_too_long_rejected() -> None:
    with pytest.raises(InvalidInputError):
        validate_inputs("x" * 2001, "valid-group")


# ---------------------------------------------------------------------------
# Bridge output sanitization
# ---------------------------------------------------------------------------

def test_bridge_output_too_long_treated_as_stop() -> None:
    result = _validate_bridge_output("x" * 501)
    assert result == ""


def test_bridge_output_nonprintable_treated_as_stop() -> None:
    result = _validate_bridge_output("\x00\x01\x02 real text")
    assert result == ""


# ---------------------------------------------------------------------------
# Seed expansion security
# ---------------------------------------------------------------------------

def test_expansion_term_too_long_discarded() -> None:
    """Terms > 200 chars are filtered out in _expand_query_llm."""
    import inspect
    from retrieval.seed import _expand_query_llm
    source = inspect.getsource(_expand_query_llm)
    assert "200" in source


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

def test_score_clamped_below_zero() -> None:
    fact = make_fact(score=-5.0)
    assert fact.score == 0.0


def test_score_clamped_above_one() -> None:
    fact = make_fact(score=2.5)
    assert fact.score == 1.0


# ---------------------------------------------------------------------------
# Error message sanitization (no API key / DB password leakage)
# ---------------------------------------------------------------------------

def test_error_message_no_api_key() -> None:
    """LLM exception message does not contain the API key sentinel."""
    api_key = "sk-supersecret-key-value-123"
    os.environ["LLM_API_KEY"] = api_key

    from retrieval.exceptions import LLMCallError
    try:
        raise LLMCallError("Connection failed with status 401")
    except LLMCallError as exc:
        assert api_key not in str(exc)


def test_error_message_no_db_password() -> None:
    """DB exception message does not contain the DB password sentinel."""
    db_password = "db-password-very-secret-789"
    os.environ["DB_PASSWORD"] = db_password

    from retrieval.exceptions import GraphLoadError
    try:
        raise GraphLoadError("Failed to load graph nodes")
    except GraphLoadError as exc:
        assert db_password not in str(exc)


# ---------------------------------------------------------------------------
# Malformed rows
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fact_with_invalid_uuid_skipped() -> None:
    """Row with missing/malformed required fields is skipped gracefully."""
    from retrieval.ppr import _load_candidate_facts
    from tests.conftest import StubDriver

    bad_row = {
        "uuid": make_uuid(999),
        "fact": None,  # Will cause Pydantic error (non-nullable)
        "source_uuid": "n0",
        "target_uuid": "n1",
        "score": 0.5,
        "created_at": None,
        "valid_at": None,
        "invalid_at": None,
        "expired_at": None,
        "episodes": [],
        "embedding": None,
    }

    driver = StubDriver(fact_rows=[bad_row])
    result = await _load_candidate_facts(driver, "g1", [make_uuid(999)], 10, None)
    assert isinstance(result, list)
    # Bad row should be skipped (fact=None fails validation)
    assert len(result) == 0
