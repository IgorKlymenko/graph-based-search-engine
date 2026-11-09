"""Incremental linking must reproduce the full re-link while scoring fewer pairs."""

import numpy as np

from src.inject import pairing as PR
from src.inject.main import link

VECTOR = {
    "sarah": np.array([1.0, 0.0, 0.0]),
    "steven": np.array([0.0, 1.0, 0.0]),
    "okta": np.array([0.0, 0.0, 1.0]),
}
GOLD: dict[str, str] = {}


def rec(episode_id: str, entity: str, *surfaces: str) -> PR.ChainRecord:
    chain_id = f"{episode_id}:{surfaces[0]}"
    GOLD[chain_id] = entity
    return PR.ChainRecord(
        chain_id=chain_id,
        episode_id=episode_id,
        head=surfaces[0],
        surfaces=frozenset(surfaces),
        candidate_type="proper_name",
        name_vector=VECTOR[entity],
    )


def corpus() -> list[PR.ChainRecord]:
    """Six episodes; the last two only mention Steven and Okta, so the Sarah
    pairs are untouched by the new evidence and can be carried."""
    return [
        rec("e1", "sarah", "Sarah Chen"),
        rec("e1", "okta", "Okta"),
        rec("e2", "sarah", "Sarah Chen"),
        rec("e2", "steven", "Steven Chen"),
        rec("e3", "steven", "Steven Chen"),
        rec("e3", "okta", "Okta"),
        rec("e4", "sarah", "S. Chen"),
        rec("e4", "okta", "Okta"),
        rec("e5", "steven", "Steven Chen"),
        rec("e5", "okta", "Okta"),
        rec("e6", "okta", "Okta"),
        rec("e6", "steven", "Steven Chen"),
    ]


def adjudicate(a, b, model, evidence, passages, cache) -> dict:
    """Stub reviewer: answers from gold, so the review band resolves correctly."""
    return {
        "same_entity": GOLD[a.chain_id] == GOLD[b.chain_id],
        "confidence": 1.0,
        "reason": "stub",
    }


THRESHOLDS = PR.Thresholds(low=0.30, high=0.50)


def partition(assignment: dict[str, int]) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for chain_id, cluster in assignment.items():
        groups.setdefault(cluster, set()).add(chain_id)
    return {frozenset(g) for g in groups.values()}


def run(records, previous=None) -> dict:
    return link(
        records,
        model="stub",
        k=50,
        thresholds=THRESHOLDS,
        workers=2,
        passages={},
        cache={},
        previous=previous,
        adjudicate=adjudicate,
    )


def state_of(records, result) -> dict:
    return {
        "chains": [r.chain_id for r in records],
        "merges": [list(m) for m in result["merges"]],
    }


def test_incremental_matches_the_full_relink_and_scores_fewer_pairs() -> None:
    records = corpus()
    full = run(records)
    old = [r for r in records if r.episode_id not in {"e5", "e6"}]
    first = run(old)

    incremental = run(records, previous=state_of(old, first))

    assert partition(incremental["assignment"]) == partition(full["assignment"])
    assert incremental["pairs"] < full["pairs"]
    assert incremental["carried"] > 0
    gold = {
        frozenset(
            c
            for c, e in GOLD.items()
            if e == entity and c in {r.chain_id for r in records}
        )
        for entity in VECTOR
    }
    assert partition(full["assignment"]) == gold


def test_the_review_band_is_exercised_and_resolved_by_the_reviewer() -> None:
    """`S. Chen` against `Sarah Chen` with no alias evidence lands in the band."""
    result = run(corpus())
    assert result["reviewed"] > 0


def test_merges_naming_a_vanished_chain_are_dropped() -> None:
    records = corpus()
    full = run(records)
    fewer = [r for r in records if r.episode_id != "e1"]

    incremental = run(fewer, previous=state_of(records, full))  # no KeyError

    assert set(incremental["assignment"]) == {r.chain_id for r in fewer}


def test_reviews_are_checkpointed_as_they_arrive(monkeypatch) -> None:
    from src.inject import main as M

    monkeypatch.setattr(M, "CHECKPOINT_EVERY", 2)
    saves: list[int] = []
    result = link(
        corpus(),
        model="stub",
        k=50,
        thresholds=THRESHOLDS,
        workers=2,
        passages={},
        cache={},
        adjudicate=adjudicate,
        checkpoint=lambda: saves.append(1),
    )
    assert result["reviewed"] >= 2 and len(saves) == result["reviewed"] // 2
