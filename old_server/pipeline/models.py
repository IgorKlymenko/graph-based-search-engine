"""Typed records shared across pipeline stages.

``RelationEdge`` and ``NodeFact`` are ported verbatim from
``relation_extraction.ipynb``; ``Candidate`` and ``ResolvedEntity`` formalize the
dict shapes the pipeline stages and ``graph_upload.py`` exchange via JSON caches.

Vocabulary: a *concept* is a candidate string as it appears in the source; an
*episode* is the ~1000-char document slice containing it (see episodes.py); a
*canonical entity* is the group of concepts that resolution judged to denote one
real-world thing (see resolution.py).
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

from pydantic import BaseModel

from pipeline.config import EXTRACTOR_VERSION

# ── fact taxonomy (relation extraction) ────────────────────────────────────────
FACT_TYPE_VALUES = [
    "assertion",  # checkable claim — edge if object resolves, node-fact otherwise
    "definition",  # copular/attributive — edge if entity object, else node-fact
    "opinion",  # personal-notes corpus: KEPT as node-fact, labeled
    "instruction",  # KEPT as node-fact, labeled
    "prediction",  # KEPT as node-fact, labeled
    "metaphor",  # log-only
    "other_non_fact",  # log-only
]

# fact types that carry retrievable content (edge or node-fact)
CONTENT_TYPES = {"assertion", "definition", "opinion", "instruction", "prediction"}
# fact types eligible to become PPR edges (when object resolves)
EDGE_TYPES = {"assertion", "definition"}


def unit_key(candidate: dict) -> str:
    """Stable id for one resolution unit: a ``(concept, context)`` in an episode.

    Stage 2 embeds one vector per unit and stage 4 resolves units, so both must
    derive the same id from a candidate dict alone — hence hashing ``context``
    rather than carrying a span offset across the stage boundary.

    Scoped to the episode on purpose: the span vector is a function of the whole
    episode, so the same sentence appearing in two documents is two units with
    two genuinely different vectors.
    """
    ctx = hashlib.sha1(candidate["context"].encode()).hexdigest()[:8]
    return "\x1f".join((
        candidate["concept"],
        candidate["source_file"],
        str(candidate["episode_idx"]),
        ctx,
    ))


class Candidate(BaseModel):
    """One entity-candidate occurrence in an episode.

    ``concept`` is the candidate string itself, copied verbatim from the source;
    ``episode`` is the full episode text containing it. ``episode_char_start`` /
    ``_end`` are the episode's offsets in the source document (exact slice — see
    episodes.py), which relation extraction reuses instead of re-deriving them.
    """

    concept: str
    type: str
    context: str
    episode_idx: int
    episode: str
    source_file: str
    episode_char_start: int
    episode_char_end: int
    char_offset: int | None = None  # offset of context within episode (gleaned)
    gleaned: bool = False


class ResolvedEntity(BaseModel):
    """A canonical entity after clustering + LLM resolution."""

    canonical: str
    members: list[str]
    confidence: str = "high"
    reason: str = ""
    source: str = "llm_resolved"  # or "singleton"
    source_files: list[str] = []


class RelationEdge(BaseModel):
    """Entity-to-entity RELATES_TO edge with a plain-English description."""

    relation_id: str
    subject_entity_id: str
    relation_description: str
    object_raw: str
    object_entity_id: str | None = None
    object_resolved: bool = False
    fact_type: str = "assertion"
    valid_from: str | None = None
    valid_to: str | None = None
    temporal_inferred: bool = False
    doc_id: str
    char_start: int
    char_end: int
    source_quote: str
    quote_verified: bool = False
    span_verified: bool = False
    confidence: float = 1.0
    extractor_version: str = EXTRACTOR_VERSION


class NodeFact(BaseModel):
    """Fact about ONE entity. Indexed for asymmetric fact ranking; creates NO
    edge. Sources: objectless statements, unresolved objects, class-valued
    definitions, and opinions/instructions/predictions (labeled by fact_type).
    """

    fact_id: str
    entity_id: str
    fact_description: str
    fact_type: str = "assertion"
    doc_id: str
    char_start: int
    char_end: int
    source_quote: str
    quote_verified: bool = False
    confidence: float = 1.0
    provenance: str = "extraction"
    extractor_version: str = EXTRACTOR_VERSION


class EpisodeResult(NamedTuple):
    edges: list[RelationEdge]
    node_facts: list[NodeFact]
    non_facts: list[dict]  # metaphor / other_non_fact only
    rejected_subjects: list[dict]
    uncovered: list[str]
