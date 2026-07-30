"""Span location must follow each occurrence's own context sentence.

Regression: ``episode_text.index(concept)`` returns only the first match, so
every repeat occurrence of a concept inside one episode was pooled around the
first one — discarding the contextual difference span pooling exists to capture.
"""

from __future__ import annotations

from pipeline.embed_candidates import _locate_span, apply_window

EPISODE = (
    "Workvivo supports SCIM provisioning. "
    "Administrators configure Okta first. "
    "Later, Workvivo syncs the team names."
)


def test_repeated_concept_resolves_to_its_own_occurrence() -> None:
    first_ctx = "Workvivo supports SCIM provisioning."
    last_ctx = "Later, Workvivo syncs the team names."

    first = _locate_span(EPISODE, "Workvivo", first_ctx)
    last = _locate_span(EPISODE, "Workvivo", last_ctx)

    assert first == (0, True)
    assert last is not None
    start, exact = last
    assert exact
    assert start == EPISODE.rindex("Workvivo")
    assert start != 0, "second occurrence must not collapse onto the first"
    assert EPISODE[start : start + len("Workvivo")] == "Workvivo"


def test_span_slices_the_concept_exactly() -> None:
    for ctx in ("Administrators configure Okta first.",):
        located = _locate_span(EPISODE, "Okta", ctx)
        assert located is not None
        start, _ = located
        assert EPISODE[start : start + 4] == "Okta"


def test_falls_back_to_first_match_when_context_not_verbatim() -> None:
    located = _locate_span(EPISODE, "Workvivo", "a paraphrased sentence")
    assert located == (0, False)


def test_falls_back_when_concept_absent_from_its_context() -> None:
    # context is verbatim, but the extractor named a concept not inside it
    located = _locate_span(EPISODE, "Workvivo", "Administrators configure Okta first.")
    assert located == (0, False)


def test_returns_none_when_concept_absent_from_episode() -> None:
    assert _locate_span(EPISODE, "Slack", "Slack is unrelated.") is None


def test_window_keeps_a_late_span_inside_the_clip() -> None:
    # a span past the 2048-char threshold must survive, with offsets rebased
    episode = "x" * 4000 + "Neo4j" + "y" * 4000
    span_start = 4000
    clipped, new_start, new_end = apply_window(
        episode, span_start, span_start + len("Neo4j")
    )
    assert clipped[new_start:new_end] == "Neo4j"
    assert len(clipped) < len(episode)
