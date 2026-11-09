"""Only what changed since the last upload is sent; no baseline sends everything."""

import copy

from src.inject import store as ST


def make_store() -> dict:
    return {
        "files": {
            "f1": {"file_id": "f1", "full_text_hash": "h1", "removed_at": None},
            "f2": {"file_id": "f2", "full_text_hash": "h2", "removed_at": None},
        },
        "episodes": {
            "e1": {
                "episode_id": "e1",
                "file_id": "f1",
                "text": "x",
                "removed_at": None,
            },
            "e2": {
                "episode_id": "e2",
                "file_id": "f2",
                "text": "y",
                "removed_at": None,
            },
        },
        "candidates": {
            "c1a": {
                "candidate_id": "c1a",
                "chain_id": "ch1",
                "text": "Sarah Chen",
                "is_head": True,
            },
            "c1b": {
                "candidate_id": "c1b",
                "chain_id": "ch1",
                "text": "she",
                "is_head": False,
            },
            "c2": {
                "candidate_id": "c2",
                "chain_id": "ch2",
                "text": "Okta",
                "is_head": True,
            },
        },
        "chains": {
            "ch1": {
                "chain_id": "ch1",
                "episode_id": "e1",
                "content_key": "k1",
                "entity_id": "E1",
            },
            "ch2": {
                "chain_id": "ch2",
                "episode_id": "e2",
                "content_key": "k2",
                "entity_id": "E2",
            },
        },
    }


def ids(rows, key):
    return sorted(r[key] for r in rows)


def test_no_baseline_sends_everything() -> None:
    todo = ST.plan(make_store(), None)
    assert ids(todo["files"], "file_id") == ["f1", "f2"]
    assert ids(todo["episodes"], "episode_id") == ["e1", "e2"]
    assert ids(todo["chains"], "chain_id") == ["ch1", "ch2"]
    assert todo["surfaces"] == ["Okta", "Sarah Chen"]


def test_a_snapshot_of_the_same_store_plans_nothing() -> None:
    store = make_store()
    todo = ST.plan(store, ST.snapshot(store))
    assert all(todo[k] == [] for k in todo), todo


def test_only_the_edited_rows_are_sent() -> None:
    store = make_store()
    baseline = ST.snapshot(copy.deepcopy(store))
    store["files"]["f1"]["full_text_hash"] = "h1b"
    store["episodes"]["e1"]["removed_at"] = "2026-09-17T00:00:00"
    store["episodes"]["e3"] = {
        "episode_id": "e3",
        "file_id": "f1",
        "text": "z",
        "removed_at": None,
    }
    store["chains"]["ch3"] = {
        "chain_id": "ch3",
        "episode_id": "e3",
        "content_key": "k3",
        "entity_id": "E1",
    }
    store["candidates"]["c3"] = {
        "candidate_id": "c3",
        "chain_id": "ch3",
        "text": "S. Chen",
        "is_head": True,
    }

    todo = ST.plan(store, baseline)

    assert ids(todo["files"], "file_id") == ["f1"]
    assert ids(todo["episodes"], "episode_id") == ["e1", "e3"]
    assert ids(todo["chains"], "chain_id") == ["ch3"]
    assert [r[0] for r in todo["chain_surfaces"]] == ["ch3"]
    assert todo["surfaces"] == ["S. Chen"]


def test_a_reclustered_chain_is_resent_with_its_surfaces_but_no_new_surface() -> None:
    store = make_store()
    baseline = ST.snapshot(copy.deepcopy(store))
    store["chains"]["ch2"]["entity_id"] = "E9"

    todo = ST.plan(store, baseline)

    assert ids(todo["chains"], "chain_id") == ["ch2"]
    assert [r[:2] for r in todo["chain_surfaces"]] == [("ch2", "Okta")]
    assert todo["surfaces"] == []
    assert todo["files"] == [] and todo["episodes"] == []


def test_pronouns_never_reach_chain_surfaces() -> None:
    rows = ST._rows_for_chain_surfaces(make_store())
    assert {surface for _, surface, _, _ in rows} == {"Sarah Chen", "Okta"}


def test_a_chain_whose_head_was_a_pronoun_gets_its_longest_surface_as_head() -> None:
    store = make_store()
    store["candidates"]["c1a"]["is_head"] = False
    store["candidates"]["c1b"]["is_head"] = True  # the model picked "she"
    rows = {
        (chain, surface): head
        for chain, surface, head, _ in ST._rows_for_chain_surfaces(store)
    }
    assert rows[("ch1", "Sarah Chen")] is True


def test_a_changed_verdict_resends_the_surface_and_nothing_else() -> None:
    store = make_store()
    store["names"] = {"Okta": True, "Sarah Chen": True}
    baseline = ST.snapshot(copy.deepcopy(store))
    store["names"]["Okta"] = False

    todo = ST.plan(store, baseline)

    assert todo["surfaces"] == ["Okta"]
    assert todo["chains"] == [] and todo["files"] == [] and todo["episodes"] == []


def test_a_pre_flag_snapshot_resends_every_surface_once() -> None:
    store = make_store()
    baseline = ST.snapshot(store)
    baseline["surfaces"] = sorted(baseline["surfaces"])   # the old list shape
    assert ST.plan(store, baseline)["surfaces"] == ["Okta", "Sarah Chen"]
    assert ST.plan(store, ST.snapshot(store))["surfaces"] == []


def test_the_head_moves_to_a_name_when_the_models_head_is_judged_junk() -> None:
    store = make_store()
    store["candidates"]["c1c"] = {"candidate_id": "c1c", "chain_id": "ch1",
                                  "text": "the model", "is_head": True}
    store["candidates"]["c1a"]["is_head"] = False
    store["names"] = {"the model": False, "Sarah Chen": True}
    rows = {(chain, surface): head for chain, surface, head, _ in
            ST._rows_for_chain_surfaces(store)}
    assert rows[("ch1", "Sarah Chen")] is True
    assert rows[("ch1", "the model")] is False
