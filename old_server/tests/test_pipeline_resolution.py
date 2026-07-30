"""validate_partition must yield a clean partition of exactly the input set."""

from __future__ import annotations

from pipeline.resolution import _pick_contexts, validate_partition


def test_omitted_members_repaired_as_singletons() -> None:
    concepts = ["Alice", "Bob", "Carol"]
    result = validate_partition(
        concepts,
        {"groups": [{"canonical": "Alice", "members": ["Alice"],
                     "confidence": "high", "reason": "x"}]},
        verbose=False,
    )
    covered = {m for g in result["groups"] for m in g["members"]}
    assert covered == set(concepts)
    assert result["repaired"] == 2


def test_hallucinated_member_dropped() -> None:
    concepts = ["Alice", "Bob"]
    result = validate_partition(
        concepts,
        {"groups": [{"canonical": "Alice",
                     "members": ["Alice", "Zebra", "Bob"],
                     "confidence": "high", "reason": "x"}]},
        verbose=False,
    )
    covered = {m for g in result["groups"] for m in g["members"]}
    assert covered == {"Alice", "Bob"}
    assert result["dropped"] == 1


def test_normalization_fallback_matches_quoted_variant() -> None:
    concepts = ["Neo4j"]
    result = validate_partition(
        concepts,
        {"groups": [{"canonical": "Neo4j", "members": ["'Neo4j' "],
                     "confidence": "high", "reason": "x"}]},
        verbose=False,
    )
    assert result["groups"][0]["members"] == ["Neo4j"]
    assert result["dropped"] == 0 and result["repaired"] == 0


def test_duplicate_member_kept_once() -> None:
    concepts = ["Alice", "Bob"]
    result = validate_partition(
        concepts,
        {"groups": [
            {"canonical": "Alice", "members": ["Alice", "Alice"],
             "confidence": "high", "reason": "x"},
            {"canonical": "Bob", "members": ["Bob"],
             "confidence": "high", "reason": "y"},
        ]},
        verbose=False,
    )
    all_members = [m for g in result["groups"] for m in g["members"]]
    assert sorted(all_members) == ["Alice", "Bob"]
    assert result["dropped"] == 1


def test_pick_contexts_spreads_selection() -> None:
    assert _pick_contexts(["a"], k=2) == ["a"]
    assert _pick_contexts(["a", "b", "c", "d"], k=2) == ["a", "d"]
    assert _pick_contexts(["a", "b", "c"], k=1) == ["a"]
