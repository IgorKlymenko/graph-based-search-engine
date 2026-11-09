"""A call that never returns must not freeze the stage: abandon after silence, keep the rest."""

import threading

from src.inject import chains as CH


def test_all_results_arrive_when_calls_finish() -> None:
    got = dict(CH.completed(lambda n: n * 2, [1, 2, 3], workers=2, stall=5))
    assert got == {1: 2, 2: 4, 3: 6}


def test_a_hung_call_is_abandoned_after_the_stall_and_the_others_kept() -> None:
    release = threading.Event()

    def call(item: str) -> str:
        if item == "hung":
            release.wait(timeout=10)
        return item.upper()

    try:
        got = dict(
            CH.completed(call, ["a", "hung", "b"], workers=3, stall=0.3, label="tests")
        )
    finally:
        release.set()
    assert got == {"a": "A", "b": "B"}


def test_an_empty_batch_yields_nothing() -> None:
    assert list(CH.completed(lambda n: n, [], workers=2)) == []


def test_run_reports_abandoned_episodes(monkeypatch) -> None:
    release = threading.Event()
    store = {
        "episodes": {
            "e1": {"text": "x", "extracted_at": None, "removed_at": None},
            "e2": {"text": "y", "extracted_at": None, "removed_at": None},
        },
        "candidates": {},
        "chains": {},
    }

    def fake(store_, episode_id, *, model, retries=1):
        if episode_id == "e2":
            release.wait(timeout=10)
        return [], [], None

    monkeypatch.setattr(CH, "process_episode", fake)
    try:
        monkeypatch.setattr(CH, "STALL_TIMEOUT", 0.3)
        result = CH.run(store, model="stub", workers=2)
    finally:
        release.set()
    assert result["episodes"] == 1 and result["abandoned"] == 1
    assert (
        store["episodes"]["e1"]["extracted_at"]
        and not store["episodes"]["e2"]["extracted_at"]
    )
