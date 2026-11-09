"""The name judge must judge only what is new, survive a mangled batch, and leave the unanswered as names."""

from src.inject.utils import judge as JG


def store_with(*episodes: tuple[str, str, list[tuple[str, int, int]]]) -> dict:
    """(episode_id, text, [(surface, start, end)]) -> a minimal store."""
    store = {"episodes": {}, "candidates": {}}
    for eid, text, spans in episodes:
        store["episodes"][eid] = {"episode_id": eid, "text": text, "removed_at": None}
        for n, (surface, start, end) in enumerate(spans):
            cid = f"{eid}:{n}"
            store["candidates"][cid] = {
                "candidate_id": cid,
                "episode_id": eid,
                "text": surface,
                "start": start,
                "end": end,
                "chain_id": f"{eid}:c",
            }
    return store


def test_context_is_the_first_live_occurrence_window() -> None:
    text = "x" * 200 + "Okta rollout" + "y" * 200
    store = store_with(
        ("e1", text, [("Okta", 200, 204)]), ("e2", "Okta again", [("Okta", 0, 4)])
    )
    ctx = JG.contexts_for(store, {"Okta"})
    assert ctx["Okta"].startswith("x" * JG.CONTEXT_CHARS + "Okta")
    assert len(ctx["Okta"]) == JG.CONTEXT_CHARS * 2 + 4


def test_retired_episodes_give_no_context() -> None:
    store = store_with(("e1", "Okta", [("Okta", 0, 4)]))
    store["episodes"]["e1"]["removed_at"] = "2026-09-17"
    assert JG.contexts_for(store, {"Okta"}) == {}


def test_a_mangled_batch_is_retried_and_the_unanswerable_left_out() -> None:
    calls: list[int] = []

    def complete(*, system, user, model, schema, schema_name):
        lines = user.split("\n")
        calls.append(len(lines))
        if len(lines) > 1:
            return {"verdicts": [{"id": "bogus", "keep": True}]}  # the whole batch lost
        key, surface, _ = lines[0].split(" | ", 2)
        if surface == "the model":
            return {"verdicts": []}  # never answered
        return {"verdicts": [{"id": key, "keep": surface == "Okta"}]}

    verdicts = JG.judge_surfaces(
        {"Okta": "c", "the model": "c", "the results": "c"},
        models=("stub",),
        workers=2,
        complete=complete,
    )
    assert verdicts == {"Okta": True, "the results": False}
    assert calls[0] == 3 and 1 in calls  # batch first, then singles


def test_prompt_lines_never_contain_a_newline_from_the_surface() -> None:
    seen: list[str] = []

    def complete(*, system, user, model, schema, schema_name):
        seen.append(user)
        return {"verdicts": [{"id": "s0", "keep": False}]}

    JG.judge_surfaces(
        {"Ser\n```": "ctx"}, models=("stub",), workers=1, complete=complete
    )
    assert "\n" not in seen[0]


def test_the_cascade_lets_a_cheap_drop_stand_and_the_strong_model_decide_the_rest() -> (
    None
):
    seen: list[tuple[str, str]] = []

    def complete(*, system, user, model, schema, schema_name):
        out = []
        for line in user.split("\n"):
            key, surface, _ = line.split(" | ", 2)
            seen.append((model, surface))
            if model == "cheap":
                out.append({"id": key, "keep": surface != "the results"})
            else:
                out.append({"id": key, "keep": surface == "Okta"})
        return {"verdicts": out}

    verdicts = JG.judge_surfaces(
        {"Okta": "c", "the results": "c", "Custom": "c"},
        models=("cheap", "strong"),
        workers=1,
        complete=complete,
    )
    assert verdicts == {"Okta": True, "the results": False, "Custom": False}
    assert ("strong", "the results") not in seen  # dropped early, never escalated
    assert ("strong", "Okta") in seen and ("strong", "Custom") in seen


def test_progress_is_reported_with_the_verdicts_settled_so_far(monkeypatch) -> None:
    monkeypatch.setattr(JG, "BATCH", 1)
    monkeypatch.setattr(JG, "CHECKPOINT_BATCHES", 2)
    reports: list[dict] = []

    def complete(*, system, user, model, schema, schema_name):
        key, surface, _ = user.split(" | ", 2)
        return {"verdicts": [{"id": key, "keep": surface.startswith("K")}]}

    verdicts = JG.judge_surfaces(
        {"K1": "c", "D1": "c", "K2": "c", "D2": "c"},
        models=("stub",),
        workers=1,
        complete=complete,
        on_progress=reports.append,
    )
    assert verdicts == {"K1": True, "K2": True, "D1": False, "D2": False}
    assert reports and all(set(r.items()) <= set(verdicts.items()) for r in reports)
