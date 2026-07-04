"""Chunking offsets must exactly slice the source document (spec §8.1)."""

from __future__ import annotations

from pipeline.chunking import chunk_doc


def test_chunks_are_exact_slices() -> None:
    doc = "First para.\n\nSecond para is here.\n\n\nThird after blank runs."
    chunks = chunk_doc(doc, max_chars=1000)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert doc[c.char_start : c.char_end] == c.text


def test_merges_paragraphs_up_to_max_chars() -> None:
    para = "x" * 300
    doc = "\n\n".join([para] * 5)  # 5 paragraphs
    chunks = chunk_doc(doc, max_chars=1000)
    # 300-char paras: ~3 fit per 1000-char chunk.
    assert len(chunks) >= 2
    for c in chunks:
        assert doc[c.char_start : c.char_end] == c.text


def test_paragraph_larger_than_max_becomes_its_own_chunk() -> None:
    big = "y" * 5000
    chunks = chunk_doc(big, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].text == big
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 5000)


def test_empty_and_blank_docs_yield_no_chunks() -> None:
    assert chunk_doc("") == []
    assert chunk_doc("\n\n   \n\n") == []


def test_offsets_are_monotonic_and_nonoverlapping() -> None:
    doc = "\n\n".join(f"paragraph number {i}" for i in range(10))
    chunks = chunk_doc(doc, max_chars=60)
    prev_end = -1
    for c in chunks:
        assert c.char_start > prev_end
        prev_end = c.char_end
