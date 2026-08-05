"""What a chain's fingerprint must and must not depend on."""

from __future__ import annotations

from entity_linking import chains as CH
from entity_linking import spans as SP

EPISODE = "11111111-1111-5111-8111-111111111111"


def span(text: str, start: int, kind: str = SP.PROPER_NAME) -> SP.Span:
    return SP.Span(sid=f"c{start}", start=start, end=start + len(text),
                   text=text, candidate_type=kind, origin="parser")


SARAH = span("Sarah Chen", 0)
SHORT = span("S. Chen", 40)
PRONOUN = span("She", 80, SP.PRONOUN)


def test_order_does_not_matter() -> None:
    assert CH.content_key([SARAH, SHORT]) == CH.content_key([SHORT, SARAH])


def test_which_span_leads_does_not_matter() -> None:
    """The property the head-offset derivation got wrong.

    The head is the model's opinion about which surface is canonical, not a
    property of the chain, and it can change between runs on its own.
    """
    head_first = CH.build_chains(EPISODE, [[SARAH, SHORT]])[0]
    head_second = CH.build_chains(EPISODE, [[SHORT, SARAH]])[0]
    assert head_first["content_key"] == head_second["content_key"]
    assert head_first["chain_id"] == head_second["chain_id"]
    assert head_first["head_surface"] != head_second["head_surface"]


def test_offsets_do_not_matter() -> None:
    """An edit above the chain moves it without changing what it contains."""
    moved = [span("Sarah Chen", 500), span("S. Chen", 540)]
    assert CH.content_key([SARAH, SHORT]) == CH.content_key(moved)


def test_gaining_a_surface_changes_the_key() -> None:
    """The whole point: a chain that changed must not look unchanged."""
    assert CH.content_key([SARAH]) != CH.content_key([SARAH, SHORT])


def test_the_hash_covers_only_what_gets_stored() -> None:
    """Pronouns are filtered out of chain_surfaces, so they must not be hashed.

    If they were, every chain in the corpus would look new the moment the filter
    changed, and every entity_id — which costs review calls — would be discarded.
    """
    assert CH.content_key([SARAH, SHORT]) == CH.content_key([SARAH, SHORT, PRONOUN])


def test_a_chain_of_pure_scaffolding_has_no_key_and_is_not_stored() -> None:
    assert CH.content_key([PRONOUN]) is None
    assert CH.build_chains(EPISODE, [[PRONOUN]]) == []
    assert CH.build_candidates(EPISODE, [[PRONOUN]]) == []


def test_candidates_and_chains_agree_on_the_id() -> None:
    """Two functions derive it independently; they must not drift apart."""
    resolved = [[SARAH, SHORT, PRONOUN], [span("Okta", 200)]]
    chains = CH.build_chains(EPISODE, resolved)
    candidates = CH.build_candidates(EPISODE, resolved)
    assert {c["chain_id"] for c in candidates} == {c["chain_id"] for c in chains}


def test_the_same_chain_in_two_episodes_is_two_chains() -> None:
    other = "22222222-2222-5222-8222-222222222222"
    assert (CH.build_chains(EPISODE, [[SARAH]])[0]["chain_id"]
            != CH.build_chains(other, [[SARAH]])[0]["chain_id"])
