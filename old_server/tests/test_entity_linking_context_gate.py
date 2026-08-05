"""The gated context feature: it must fire only where the names cannot decide."""

from __future__ import annotations

import numpy as np
import pytest

from entity_linking import pairing as PR


def chain(chain_id: str, episode: str, surfaces: list[str],
          context: np.ndarray | None = None) -> PR.ChainRecord:
    return PR.ChainRecord(
        chain_id=chain_id, episode_id=episode, head=surfaces[0],
        surfaces=frozenset(surfaces), candidate_type="proper_name",
        name_vector=np.array([1.0, 0.0]), context_vector=context)


ALIGNED = np.array([1.0, 0.0])
ORTHOGONAL = np.array([0.0, 1.0])


def test_gate_closed_for_different_names() -> None:
    """Below the gate the feature is silent, whatever the contexts look like."""
    a = chain("a", "ep1", ["Sarah Chen"], ALIGNED)
    b = chain("b", "ep2", ["Workvivo"], ALIGNED)
    assert PR.context_gated(a, b) == 0.0


def test_gate_open_for_identical_names() -> None:
    a = chain("a", "ep1", ["S. Chen"], ALIGNED)
    b = chain("b", "ep2", ["S. Chen"], ALIGNED)
    assert PR.context_gated(a, b) == pytest.approx(1.0)


def test_open_gate_separates_contexts() -> None:
    """The whole point: same string, different context -> a lower score."""
    a = chain("a", "ep1", ["S. Chen"], ALIGNED)
    same = chain("b", "ep2", ["S. Chen"], ALIGNED)
    other = chain("c", "ep3", ["S. Chen"], ORTHOGONAL)
    assert PR.context_gated(a, same) > PR.context_gated(a, other)


def test_missing_vector_is_neutral_not_negative() -> None:
    """An absent vector must never push a pair toward splitting.

    Chains whose mentions fall outside the token window get no context vector.
    Returning a low similarity there would penalise them for a pooling accident.
    """
    a = chain("a", "ep1", ["S. Chen"], ALIGNED)
    b = chain("b", "ep2", ["S. Chen"], None)
    assert PR.context_gated(a, b) == 0.0


def test_feature_is_scored_and_absent_context_does_not_change_the_verdict() -> None:
    evidence = PR.Evidence(same_chain=set(), split_in_episode=set())
    a = chain("a", "ep1", ["S. Chen"], None)
    b = chain("b", "ep2", ["S. Chen"], None)
    feats = PR.features(a, b, evidence)
    assert "context_gated" in feats
    assert feats["context_gated"] == 0.0

    with_context = PR.features(
        chain("a", "ep1", ["S. Chen"], ALIGNED),
        chain("b", "ep2", ["S. Chen"], ALIGNED), evidence)
    scorer = PR.PairScorer()
    assert scorer.score(with_context) > scorer.score(feats)
