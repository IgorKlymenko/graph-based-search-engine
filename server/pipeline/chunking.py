"""Document chunking with faithful character offsets.

The notebook's ``chunk_doc`` stripped each paragraph and re-joined with
``"\\n\\n"``, so a chunk's text no longer matched any slice of the source and
``doc.find(chunk)`` returned -1 for ~1/3 of chunks (spec §8.1). Here every
chunk is an exact slice ``doc[char_start:char_end]`` of the original document,
so its offsets are always valid and downstream stages never re-derive them.
"""

from __future__ import annotations

from typing import NamedTuple


class Chunk(NamedTuple):
    """A ~``max_chars`` slice of a document with its source offsets."""

    text: str
    char_start: int
    char_end: int


def _paragraph_spans(doc: str) -> list[tuple[int, int]]:
    """Offsets ``(start, end)`` of each non-blank paragraph in ``doc``.

    Splits on blank lines (``"\\n\\n"``), matching the original behavior, while
    tracking absolute offsets. Leading/trailing whitespace is trimmed from each
    span so ``doc[start:end]`` begins and ends on real content.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in doc.split("\n\n"):
        seg_start = cursor
        cursor = seg_start + len(part) + 2  # +2 for the removed "\n\n"
        if not part.strip():
            continue
        lead = len(part) - len(part.lstrip())
        trail = len(part) - len(part.rstrip())
        spans.append((seg_start + lead, seg_start + len(part) - trail))
    return spans


def chunk_doc(doc: str, max_chars: int = 1000) -> list[Chunk]:
    """Split ``doc`` into ~``max_chars`` chunks, greedily merging paragraphs.

    A single paragraph longer than ``max_chars`` becomes its own chunk. Each
    returned chunk's ``text`` equals ``doc[char_start:char_end]`` exactly.
    """
    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end in _paragraph_spans(doc):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= max_chars:
            cur_end = end
        else:
            chunks.append(Chunk(doc[cur_start:cur_end], cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        chunks.append(Chunk(doc[cur_start:cur_end], cur_start, cur_end))
    return chunks
