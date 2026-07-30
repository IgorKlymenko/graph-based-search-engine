"""Episode offsets must exactly slice the source document (spec §8.1)."""

from __future__ import annotations

from pipeline.episodes import split_episodes


def test_episodes_are_exact_slices() -> None:
    doc = "First para.\n\nSecond para is here.\n\n\nThird after blank runs."
    episodes = split_episodes(doc, max_chars=1000)
    assert episodes, "expected at least one episode"
    for ep in episodes:
        assert doc[ep.char_start : ep.char_end] == ep.text


def test_merges_paragraphs_up_to_max_chars() -> None:
    para = "x" * 300
    doc = "\n\n".join([para] * 5)  # 5 paragraphs
    episodes = split_episodes(doc, max_chars=1000)
    # 300-char paras: ~3 fit per 1000-char episode.
    assert len(episodes) >= 2
    for ep in episodes:
        assert doc[ep.char_start : ep.char_end] == ep.text


def test_paragraph_larger_than_max_becomes_its_own_episode() -> None:
    big = "y" * 5000
    episodes = split_episodes(big, max_chars=1000)
    assert len(episodes) == 1
    assert episodes[0].text == big
    assert (episodes[0].char_start, episodes[0].char_end) == (0, 5000)


def test_empty_and_blank_docs_yield_no_episodes() -> None:
    assert split_episodes("") == []
    assert split_episodes("\n\n   \n\n") == []


def test_offsets_are_monotonic_and_nonoverlapping() -> None:
    doc = "\n\n".join(f"paragraph number {i}" for i in range(10))
    episodes = split_episodes(doc, max_chars=60)
    prev_end = -1
    for ep in episodes:
        assert ep.char_start > prev_end
        prev_end = ep.char_end
