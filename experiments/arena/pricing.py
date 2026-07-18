"""Pricing table and cost calculation for arena token tracking."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from arena.tracker import TokenUsage

log = logging.getLogger(__name__)

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
    "embedding": {"in": 0.02, "out": 0.0},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
}

_PRICING_PATH = Path(__file__).parent / "pricing.json"


def load_pricing(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load and validate pricing from JSON file. Raises ValueError on invalid prices."""
    target = path if path is not None else _PRICING_PATH

    if not target.exists():
        log.warning("pricing.json not found at %s; using defaults", target)
        return dict(DEFAULT_PRICING)

    content = target.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    log.info("Loading pricing from %s (sha256: %s)", target, sha)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning("pricing.json is invalid JSON: %s; using defaults", exc)
        return dict(DEFAULT_PRICING)

    pricing: dict[str, dict[str, float]] = {}
    for model, prices in data.items():
        in_price = float(prices.get("in", 0))
        out_price = float(prices.get("out", 0))
        if in_price <= 0:
            raise ValueError(
                f"pricing.json: model {model!r} has zero/negative 'in' price: {prices}"
            )
        if out_price < 0:
            raise ValueError(
                f"pricing.json: model {model!r} has negative 'out' price: {prices}"
            )
        pricing[model] = {"in": in_price, "out": out_price}

    return pricing


def compute_cost(
    usage: TokenUsage,
    model: str,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """Return USD cost for the given token usage and model."""
    table = pricing if pricing is not None else load_pricing()
    prices = table.get(model) or DEFAULT_PRICING.get(model) or DEFAULT_PRICING["gpt-4o-mini"]
    return (usage.tokens_in / 1_000_000) * prices["in"] + (
        usage.tokens_out / 1_000_000
    ) * prices["out"]
