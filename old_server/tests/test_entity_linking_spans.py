"""Span inventory, offset-preserving normalisation, and the merge back."""

from __future__ import annotations

import pytest

from entity_linking import spans as SP

PROSE = (
    "Workvivo supports SCIM provisioning from Okta. "
    "The platform syncs team names, and it fails if a name does not match."
)


# ── normalisation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "plain sentence",
    "wrapped\nline",
    "para one\n\npara two",
    "# Heading\n\n- bullet\n- other\n",
    "| a | b |\n|---|---|\n",
    "arrows --> and -- dashes ** bold **",
    "nbsp and​zero width\ttab",
    "",
])
def test_normalisation_preserves_length(text: str) -> None:
    """The one invariant everything downstream rests on."""
    assert len(SP.normalize_for_parse(text)) == len(text)


def test_wrapped_line_joins_but_paragraph_breaks() -> None:
    joined = SP.normalize_for_parse("reach out to Workvivo\nSupport now")
    assert "Workvivo Support" in joined

    broken = SP.normalize_for_parse("A heading\n\nBody starts here")
    assert broken.startswith("A heading.")


def test_markdown_scaffolding_and_clause_markers_become_breaks() -> None:
    out = SP.normalize_for_parse("# Title\n\n- item one\n- item two")
    assert "#" not in out and "-" not in out
    assert "." in SP.normalize_for_parse("take the average --> but uniform is bad")


# ── parsing ───────────────────────────────────────────────────────────────────

def test_every_span_slices_the_original_text() -> None:
    for span in SP.parse_spans(PROSE):
        assert PROSE[span.start : span.end] == span.text


def test_nested_proper_noun_is_emitted_alongside_its_phrase() -> None:
    text = "Okta configures the Neo4j database."
    found = {s.text for s in SP.parse_spans(text)}
    assert "the Neo4j database" in found
    assert "Neo4j" in found  # the product is a node; the database is another


@pytest.mark.parametrize("text", [
    'the score was 42 -- "100%" of the time',
    "see ( 3 ) and [ 7 ] for details",
    '"the" = "Maks" weights are 0.5 vs 1.0',
])
def test_spans_without_letters_are_not_candidates(text: str) -> None:
    """Bare numerals and punctuation are values, never entity references."""
    spans = SP.parse_spans(text)
    assert spans  # the sentence does contain real candidates
    for span in spans:
        assert any(ch.isalpha() for ch in span.text), span.text


def test_candidate_types_are_derived_not_guessed() -> None:
    types = {s.text: s.candidate_type for s in SP.parse_spans(PROSE)}
    assert types.get("Workvivo") == SP.PROPER_NAME
    assert types.get("The platform") == SP.DEFINITE_DESCRIPTION
    assert types.get("it") == SP.PRONOUN


def test_offsets_survive_markdown_layout() -> None:
    md = "# Heading\n\nWorkvivo supports SCIM from\nOkta.\n\n- a bullet line\n"
    spans = SP.parse_spans(md)
    assert spans
    for span in spans:
        assert md[span.start : span.end] == span.text


# ── marking ───────────────────────────────────────────────────────────────────

def test_marked_copy_nests_and_leaves_the_original_alone() -> None:
    text = "Okta configures the Neo4j database."
    spans = SP.parse_spans(text)
    marked = SP.mark_text(text, spans)
    assert marked != text
    assert text[: len(text)] == text  # untouched
    assert marked.count(SP._OPEN) == marked.count(SP._CLOSE) == len(spans)
    for span in spans:  # every id appears exactly once
        assert marked.count(f"{SP._OPEN}{span.sid} ") == 1


# ── locating model-proposed spans ─────────────────────────────────────────────

def _propose(*texts: str) -> list[dict]:
    return [{"id": f"n{i}", "text": t, "candidate_type": None}
            for i, t in enumerate(texts)]


def test_exact_match_is_labelled_exact() -> None:
    kept, dropped = SP.locate_new_spans(PROSE, _propose("Okta"), set())
    assert not dropped
    assert kept[0].match == "exact"
    assert PROSE[kept[0].start : kept[0].end] == "Okta"


def test_reflowed_whitespace_still_matches_exactly() -> None:
    text = "reach out to Workvivo\nSupport before changing"
    kept, dropped = SP.locate_new_spans(text, _propose("Workvivo Support"), set())
    assert not dropped
    assert kept[0].text == "Workvivo\nSupport"
    assert kept[0].match.startswith("whitespace")


def test_typo_is_recovered_by_fuzzy_and_labelled_as_such() -> None:
    kept, _ = SP.locate_new_spans(PROSE, _propose("The platfrom"), set())
    assert kept and kept[0].text == "The platform"
    assert kept[0].match.startswith("fuzzy")


def test_fuzzy_can_be_disabled() -> None:
    kept, dropped = SP.locate_new_spans(
        PROSE, _propose("The platfrom"), set(), fuzzy=False
    )
    assert not kept and dropped[0]["reason"] == "not_found"


def test_invented_text_is_dropped() -> None:
    _, dropped = SP.locate_new_spans(PROSE, _propose("Zendesk Portal"), set())
    assert dropped[0]["reason"] == "not_found"


def test_repeated_text_is_ambiguous_not_first_match() -> None:
    text = "the context matters. the context is lost."
    kept, dropped = SP.locate_new_spans(text, _propose("the context"), set())
    assert not kept
    assert dropped[0]["reason"].startswith("ambiguous")


def test_ids_colliding_with_the_parser_namespace_are_rejected() -> None:
    proposed = [{"id": "c0", "text": "Okta", "candidate_type": None}]
    kept, dropped = SP.locate_new_spans(PROSE, proposed, {"c0"})
    assert not kept and dropped[0]["reason"] == "bad_or_duplicate_id"


# ── merging ───────────────────────────────────────────────────────────────────

def _span(sid, start, end, text, ctype=SP.PROPER_NAME, origin="parser"):
    return SP.Span(sid, start, end, text, ctype, origin)


def test_nested_spans_both_survive_the_merge() -> None:
    """Regression: collapsing containment fused two distinct entities.

    "Mean Pooling" nested inside "Span Mean Pooling" was aliased to its
    container, silently merging concepts the model had correctly separated.
    """
    outer = _span("c0", 0, 17, "Span Mean Pooling")
    inner = _span("c1", 5, 17, "Mean Pooling")
    table, collapsed = SP.merge_spans([outer, inner], [])
    assert collapsed == []
    assert table["c0"] is outer and table["c1"] is inner


def test_crossing_spans_collapse_to_the_longer() -> None:
    left = _span("c0", 0, 10, "aaaaaaaaaa")
    right = _span("c1", 5, 12, "bbbbbbb")
    table, collapsed = SP.merge_spans([left, right], [])
    assert len(collapsed) == 1
    assert table["c1"] is left  # the loser aliases to the survivor


def test_parser_span_wins_a_tie_against_a_model_span() -> None:
    parser = _span("c0", 0, 6, "abcdef")
    model = _span("n0", 3, 9, "defghi", origin="model")
    table, collapsed = SP.merge_spans([parser], [model])
    assert collapsed[0]["dropped"] == "n0"
    assert table["n0"] is parser


# ── head selection ────────────────────────────────────────────────────────────

def test_head_prefers_proper_name_then_description_then_length() -> None:
    pronoun = _span("c0", 0, 2, "it", SP.PRONOUN)
    described = _span("c1", 3, 15, "the platform", SP.DEFINITE_DESCRIPTION)
    named = _span("c2", 16, 24, "Workvivo", SP.PROPER_NAME)
    assert SP.pick_head([pronoun, described, named]) is named
    assert SP.pick_head([pronoun, described]) is described
    assert SP.pick_head([pronoun]) is pronoun
    assert SP.pick_head([]) is None


# ── deterministic union of identical proper names ─────────────────────────────

def test_identical_proper_names_are_unioned() -> None:
    """The model split two 'Span Mean Pooling' spans in a live run; within one
    episode an identical proper name is settled by the text, not by judgement."""
    a = _span("c2", 0, 17, "Span Mean Pooling")
    b = _span("c22", 40, 57, "Span Mean Pooling")
    merged = SP.union_identical_names([[a], [b]])
    assert len(merged) == 1
    assert {s.sid for s in merged[0]} == {"c2", "c22"}


def test_union_does_not_touch_descriptions_or_nominals() -> None:
    """'the average' of the whole is not 'the average' of the span."""
    a = _span("c12", 0, 11, "the average", SP.DEFINITE_DESCRIPTION)
    b = _span("c24", 40, 51, "the average", SP.DEFINITE_DESCRIPTION)
    c = _span("c6", 60, 70, "embeddings", SP.NOMINAL)
    d = _span("c7", 80, 90, "embeddings", SP.NOMINAL)
    assert len(SP.union_identical_names([[a], [b]])) == 2
    assert len(SP.union_identical_names([[c], [d]])) == 2


def test_union_requires_identical_text_not_containment() -> None:
    exact = _span("c0", 0, 12, "Mean Pooling")
    longer = _span("c10", 20, 42, "just one  Mean Pooling")
    assert len(SP.union_identical_names([[exact], [longer]])) == 2


def test_union_is_transitive_and_rechooses_a_head() -> None:
    pronoun = _span("c9", 0, 2, "it", SP.PRONOUN)
    name_a = _span("c0", 10, 18, "Workvivo")
    name_b = _span("c5", 30, 38, "Workvivo")
    described = _span("c3", 50, 62, "the platform", SP.DEFINITE_DESCRIPTION)
    merged = SP.union_identical_names([[pronoun, name_a], [described, name_b]])
    assert len(merged) == 1
    assert merged[0][0].candidate_type == SP.PROPER_NAME  # head, not the pronoun
    assert len(merged[0]) == 4
