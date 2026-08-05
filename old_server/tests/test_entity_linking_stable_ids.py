"""Naming a cluster by its smallest chain id: what it guarantees and what it doesn't.

The failure cases are asserted too, deliberately. They are the reason this is an
interim measure rather than the answer, and a test that pins them stops the
weakness being rediscovered as a bug.
"""

from __future__ import annotations

from entity_linking import pairing as PR

# content-derived chain ids; sorted order is what the naming depends on
A, B, C, D = "aaa-1", "bbb-2", "ccc-3", "ddd-4"


def test_cluster_is_named_by_its_smallest_member() -> None:
    assignment = {A: 0, B: 0, C: 1}
    assert PR.stable_entity_ids(assignment) == {A: A, B: A, C: C}


def test_id_survives_a_re_run_that_relabels_clusters() -> None:
    """Positional labels shuffle between runs; the minimum does not."""
    first = PR.stable_entity_ids({A: 0, B: 0, C: 1, D: 1})
    second = PR.stable_entity_ids({A: 7, B: 7, C: 3, D: 3})  # same grouping
    assert first == second


def test_a_merge_reads_as_absorption_into_the_survivor() -> None:
    """The merged cluster keeps the global minimum, so one id lives on."""
    before = PR.stable_entity_ids({A: 0, B: 0, C: 1, D: 1})
    assert before[A] == A and before[C] == C

    after = PR.stable_entity_ids({A: 0, B: 0, C: 0, D: 0})
    assert set(after.values()) == {A}  # C's entity was absorbed into A's


def test_a_split_hands_the_id_to_whichever_half_holds_the_minimum() -> None:
    """Arbitrary by uuid order, not by which half is the continuation.

    Here the three-member half loses its id to the singleton, purely because
    A sorts first. Event sourcing is what fixes this.
    """
    after = PR.stable_entity_ids({A: 0, B: 1, C: 1, D: 1})
    assert after[A] == A
    assert after[B] == after[C] == after[D] == B  # the larger half was renamed


def test_one_new_chain_can_rename_an_otherwise_unchanged_entity() -> None:
    """An incremental sync adding a smaller id renames the whole entity."""
    before = PR.stable_entity_ids({B: 0, C: 0, D: 0})
    assert set(before.values()) == {B}

    after = PR.stable_entity_ids({A: 0, B: 0, C: 0, D: 0})
    assert set(after.values()) == {A}  # membership barely changed, id did not survive


def test_singletons_name_themselves() -> None:
    assert PR.stable_entity_ids({A: 0, B: 1}) == {A: A, B: B}
