"""Contested mentions must not become entity nodes of their own."""

from __future__ import annotations

from entity_linking import pairing as PR


def chain(chain_id: str, episode: str, surfaces: list[str]) -> PR.ChainRecord:
    return PR.ChainRecord(
        chain_id=chain_id, episode_id=episode, head=surfaces[0],
        surfaces=frozenset(surfaces), candidate_type="proper_name")


SARAH = chain("c1", "ep1", ["Sarah Chen", "S. Chen"])
STEVEN = chain("c2", "ep2", ["Steven Chen", "S. Chen"])
LONE = chain("c3", "ep3", ["S. Chen"])
CONTESTED = PR.Evidence(same_chain=set(), split_in_episode=set(),
                        ambiguous_surfaces=frozenset({"S. Chen"}))


def test_lone_contested_chain_is_withheld() -> None:
    """A chain whose only name is contested is a mention we failed to place."""
    records = [SARAH, STEVEN, LONE]
    assignment = {"c1": 0, "c2": 1, "c3": 2}
    pending = PR.unresolved(records, assignment, CONTESTED)
    assert set(pending) == {"c3"}
    assert pending["c3"] == [0, 1]  # both entities using the contested surface


def test_a_merged_chain_is_never_withheld() -> None:
    """Merging with something means the evidence already settled it."""
    records = [SARAH, STEVEN, LONE]
    assignment = {"c1": 0, "c2": 1, "c3": 0}  # c3 joined Sarah
    assert PR.unresolved(records, assignment, CONTESTED) == {}


def test_an_uncontested_surface_anchors_the_chain() -> None:
    """One clean name is enough to identify on, however contested the others."""
    both = chain("c3", "ep3", ["S. Chen", "Sarah Chen"])
    records = [SARAH, STEVEN, both]
    assignment = {"c1": 0, "c2": 1, "c3": 2}
    assert PR.unresolved(records, assignment, CONTESTED) == {}


def test_no_candidates_means_it_stays_an_entity() -> None:
    """With nothing to attach to it is unlinked, not ambiguous — keep the node.

    Withholding it would delete the only record that the mention exists.
    """
    records = [LONE]
    assert PR.unresolved(records, {"c3": 0}, CONTESTED) == {}


def test_nothing_is_withheld_without_contested_surfaces() -> None:
    clean = PR.Evidence(same_chain=set(), split_in_episode=set())
    records = [SARAH, STEVEN, LONE]
    assert PR.unresolved(records, {"c1": 0, "c2": 1, "c3": 2}, clean) == {}
