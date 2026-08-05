"""Which surfaces can name something, and which are grammatical scaffolding."""

from __future__ import annotations

import pytest

from entity_linking import spans as SP


@pytest.mark.parametrize("text", [
    "Sarah Chen", "Okta", "GPT-4", "Q3",
    "the freeze", "the pilot tenant", "the identity migration",
    # thin-looking but real query keys: in a database's own documentation
    # "frozen" carries meaning and someone might well type it
    "frozen", "database", "the index",
])
def test_content_words_are_kept(text: str) -> None:
    assert SP.is_identifying(text)


@pytest.mark.parametrize("text", [
    "it", "they", "she", "them", "its", "themselves",
    "the", "a", "an", "this", "that", "those",
    "both", "all", "several", "each", "none",
    "three", "one", "2023", "42",
    "", "   ", "--",
])
def test_pure_scaffolding_is_dropped(text: str) -> None:
    assert not SP.is_identifying(text)


def test_a_function_word_beside_a_content_word_survives() -> None:
    """The rule is *every* token, not any: "the freeze" is not "the"."""
    assert SP.is_identifying("the three amigos")
    assert SP.is_identifying("all hands")
    assert not SP.is_identifying("all")


def test_case_and_punctuation_do_not_matter() -> None:
    assert not SP.is_identifying("The")
    assert not SP.is_identifying("It.")
    assert SP.is_identifying("Okta.")
