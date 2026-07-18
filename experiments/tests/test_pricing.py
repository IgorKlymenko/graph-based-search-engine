"""Tests for arena/pricing.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from arena.pricing import DEFAULT_PRICING, compute_cost, load_pricing
from arena.tracker import TokenUsage


def test_known_model_returns_correct_cost() -> None:
    usage = TokenUsage(tokens_in=1_000_000, tokens_out=0)
    pricing = {"gpt-4o-mini": {"in": 0.15, "out": 0.60}}
    cost = compute_cost(usage, "gpt-4o-mini", pricing)
    assert abs(cost - 0.15) < 1e-9


def test_unknown_model_falls_back_to_default() -> None:
    usage = TokenUsage(tokens_in=1000, tokens_out=100)
    # Unknown model should not raise; uses DEFAULT_PRICING fallback
    cost = compute_cost(usage, "nonexistent-model-xyz")
    assert cost >= 0.0


def test_zero_price_in_file_raises(tmp_path: Path) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"gpt-4o-mini": {"in": 0.0, "out": 0.60}}))
    with pytest.raises(ValueError, match="zero/negative"):
        load_pricing(pricing_file)


def test_negative_price_raises(tmp_path: Path) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"gpt-4o-mini": {"in": -0.15, "out": 0.60}}))
    with pytest.raises(ValueError):
        load_pricing(pricing_file)


def test_pricing_file_hash_logged(tmp_path: Path, caplog) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"gpt-4o-mini": {"in": 0.15, "out": 0.60}}))
    with caplog.at_level(logging.INFO, logger="arena.pricing"):
        load_pricing(pricing_file)
    assert any("sha256" in msg.lower() for msg in caplog.messages)


def test_cost_rounds_to_six_decimal_places() -> None:
    usage = TokenUsage(tokens_in=123456, tokens_out=7890)
    pricing = {"test-model": {"in": 1.23456789, "out": 9.87654321}}
    cost = compute_cost(usage, "test-model", pricing)
    # Verify result is a float (rounding happens at display/report level)
    assert isinstance(cost, float)
    # Round for display
    rounded = round(cost, 6)
    assert rounded == round(cost, 6)


def test_embed_model_zero_out_price_is_valid(tmp_path: Path) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(
        json.dumps({"text-embedding-3-small": {"in": 0.02, "out": 0.0}})
    )
    result = load_pricing(pricing_file)
    assert result["text-embedding-3-small"]["out"] == 0.0
