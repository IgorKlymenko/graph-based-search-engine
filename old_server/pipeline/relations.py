"""Stage 6 — entity->entity relation extraction.

Ported from ``relation_extraction.ipynb``. Schema, verification, and confidence
scoring are verbatim. Episodes are rebuilt from the candidate cache using the
character offsets recorded at split time (spec §8.1), so ``doc.find`` is never
used and offsets are valid for every episode whose source file exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from pipeline.config import (
    ENTITY_CACHE,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    RELATION_MODEL,
    RELATION_SAVE_EVERY,
    RELATION_WORKERS,
    RELATIONS_CACHE,
    RESOLVED_CACHE,
)
from pipeline.models import (
    CONTENT_TYPES,
    EDGE_TYPES,
    FACT_TYPE_VALUES,
    EpisodeResult,
    NodeFact,
    RelationEdge,
    unit_key,
)
from services.openai import complete_json_schema


class AliasIndex(NamedTuple):
    """Replay of resolution decisions, keyed by concept string.

    Resolution now works per unit, so one concept may denote several entities.
    These maps are therefore corpus-wide *fallbacks*; the accurate lookup is
    episode-scoped, via the ``episode_canonicals`` argument threaded through
    ``resolve_to_canonical``.
    """

    concept_to_canonicals: dict[str, list[str]]
    lower_map: dict[str, list[str]]
    canonical_aliases: dict[str, list[str]]


def build_alias_index(
    candidates: list[dict], unit_to_canonical: dict[str, str]
) -> AliasIndex:
    """Build the lookup structures relation extraction needs from resolution."""
    concept_to_canonicals: dict[str, list[str]] = defaultdict(list)
    for m in candidates:
        canon = unit_to_canonical.get(unit_key(m))
        if canon and canon not in concept_to_canonicals[m["concept"]]:
            concept_to_canonicals[m["concept"]].append(canon)

    canonical_aliases: dict[str, list[str]] = defaultdict(list)
    lower_map: dict[str, list[str]] = defaultdict(list)
    for concept, canons in concept_to_canonicals.items():
        for canon in canons:
            if concept != canon and concept not in canonical_aliases[canon]:
                canonical_aliases[canon].append(concept)
            if canon not in lower_map[concept.lower()]:
                lower_map[concept.lower()].append(canon)
    return AliasIndex(
        dict(concept_to_canonicals), dict(lower_map), dict(canonical_aliases)
    )


def resolve_to_canonical(
    name: str, index: AliasIndex, episode_canonicals: set[str] | None = None
) -> tuple[str | None, bool]:
    """Concept or canonical name -> (canonical_id, resolved).

    Dictionary replay of resolution decisions only — no fuzzy matching. When a
    concept denotes several entities corpus-wide, a canonical present in the
    current episode wins; otherwise the first is taken, which is the same
    single-answer behaviour as before unit-level resolution.
    """
    if not name:
        return None, False
    canons = (
        index.concept_to_canonicals.get(name)
        or index.lower_map.get(name.lower())
    )
    if not canons:
        return None, False
    if episode_canonicals:
        local = [c for c in canons if c in episode_canonicals]
        if local:
            return local[0], True
    return canons[0], True


def format_entities_with_aliases(
    entities: list[str], index: AliasIndex, max_aliases: int = 4
) -> str:
    lines = []
    for e in entities:
        aliases = index.canonical_aliases.get(e, [])[:max_aliases]
        tag = f"  (in text also as: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {e}{tag}")
    return "\n".join(lines)


def make_relation_id(
    doc_id: str, char_start: int, subject: str, object_raw: str, description: str
) -> str:
    desc_h = hashlib.sha1(description.encode()).hexdigest()[:8]
    key = f"{doc_id}|{char_start}|{subject}|{object_raw}|{desc_h}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def make_fact_id(doc_id: str, char_start: int, entity: str, description: str) -> str:
    desc_h = hashlib.sha1(description.encode()).hexdigest()[:8]
    key = f"{doc_id}|{char_start}|{entity}|{desc_h}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


_SYSTEM_PROMPT = """\
You read an episode of text and extract atomic facts about the given entities for a personal knowledge graph. Each fact is one sentence about a subject entity, optionally connecting it to an object entity.

CLASSIFY every statement the text makes about a listed entity — do not silently skip anything except pure scalars:
- assertion: a checkable claim about the subject. ("Alice works for Acme Corp as a data scientist.")
- definition: a copular or attributive statement saying what the subject IS — its class, role, or nature. ("Neo4j is a graph database.")
- opinion: a judgment or stance. ("Direct Mentorship is not really scalable.") These are the author's notes — emit them, they are valuable content.
- instruction: guidance or a rule. ("Never talk about the idea in the abstract.") Emit these too.
- prediction: a claim about the future. Emit.
- metaphor / other_non_fact: emit with the label so the pipeline can log them.

SKIP entirely (a separate stage extracts these): pure scalar attributes — headcounts, dates as standalone facts, money amounts, percentages, durations. "The team has 12 people" → skip. A date ANCHORING a fact goes in valid_from/valid_to, never in the description.

OBJECT RULES — the most important part:
- raw_object_str must be an ENTITY REFERENCE: a noun phrase naming a specific thing, person, organization, method, product, or concept — something that could itself be a node in the graph. It is NEVER a verb phrase, a clause, or the remainder of the sentence. Test before writing it: could this string plausibly be an item in the ENTITIES list?
- If a statement asserts something about the subject but has NO entity object — a property, behavior, or state ("Medoid is interpretable", "Saved data is messy") — set raw_object_str to "" and still emit the item. It becomes a fact attached to the subject alone.
- When the text supports it, prefer objects that appear in the ENTITIES list (under their canonical name or a listed alias). Copy object names VERBATIM as they appear in the text.

COVERAGE REQUIREMENT: every entity in ENTITIES must appear as subject or object of at least one emitted item, OR be listed in uncovered_entities. Entities may appear in the text under any of their listed aliases — check aliases before declaring one uncovered. Check definitions and appositions, not only verbs. An entity referred to purely in passing, with nothing asserted about it, belongs in uncovered_entities; that is a valid outcome.

RULES for each item:
- relation_description: ONE sentence, one claim, grounded in the text. Copy or lightly paraphrase; never infer beyond what is stated. Use the subject's canonical name, not a pronoun or alias.
- Direction: the agent/possessor/part-owner is the subject. Emit each fact once — never both A→B and B→A.
- source_quote: the VERBATIM sentence from the episode supporting this item, character-for-character. Two adjacent sentences if support spans both.
- subject_entity_id: copied VERBATIM from the ENTITIES list (canonical form, not an alias).
- valid_from / valid_to: ISO format, only if explicitly stated; partial dates for partial information ("March 2024" → "2024-03"); else null. temporal_inferred = true only if you derived a period not literally present.

You will receive TEXT (the episode) and ENTITIES (canonical entity IDs with the aliases they may appear under in the text)."""

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject_entity_id": {"type": "string"},
                    "raw_object_str": {"type": "string"},
                    "relation_description": {"type": "string"},
                    "fact_type": {"type": "string", "enum": FACT_TYPE_VALUES},
                    "source_quote": {"type": "string"},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                    "temporal_inferred": {"type": "boolean"},
                },
                "required": [
                    "subject_entity_id", "raw_object_str", "relation_description",
                    "fact_type", "source_quote", "valid_from", "valid_to",
                    "temporal_inferred",
                ],
            },
        },
        "uncovered_entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relations", "uncovered_entities"],
}


def extract_relations_from_episode(
    episode: dict, entities: list[str], index: AliasIndex, *, model: str
) -> dict:
    if not entities:
        return {"relations": [], "uncovered_entities": []}
    user_msg = (
        f"TEXT:\n{episode['text']}\n\n"
        "ENTITIES (canonical IDs — copy exactly as subject_entity_id; "
        "aliases show how they may appear in the text):\n"
        + format_entities_with_aliases(entities, index)
    )
    return complete_json_schema(
        system=_SYSTEM_PROMPT,
        user=user_msg,
        schema=_SCHEMA,
        schema_name="relations",
        model=model,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


_DIGITS_RE = re.compile(r"\d+")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_quote_and_span(
    raw_object_str: str, model_quote: str, episode_text: str,
    valid_from: str | None, valid_to: str | None,
) -> dict:
    quote_verified = (
        bool(model_quote) and _norm_ws(model_quote) in _norm_ws(episode_text)
    )
    haystack = model_quote if quote_verified else episode_text
    span_verified = (not raw_object_str) or raw_object_str.lower() in haystack.lower()
    temporal_inferred = False
    for date_val in filter(None, [valid_from, valid_to]):
        digits = _DIGITS_RE.findall(date_val)
        if digits and not all(d in haystack for d in digits):
            temporal_inferred = True
            break
    return {"quote_verified": quote_verified, "span_verified": span_verified,
            "temporal_inferred": temporal_inferred}


def score_confidence(
    object_resolved: bool, quote_verified: bool, span_verified: bool,
    temporal_inferred: bool, has_object: bool = True,
) -> float:
    score = 1.00
    if has_object and not object_resolved:
        score -= 0.20
    if not quote_verified:
        score -= 0.15
    if not span_verified:
        score -= 0.30
    if temporal_inferred:
        score -= 0.10
    return round(max(0.0, score), 2)


def process_episode(
    episode: dict, entities: list[str], index: AliasIndex, *, model: str
) -> EpisodeResult:
    raw = extract_relations_from_episode(episode, entities, index, model=model)
    edges: list[RelationEdge] = []
    node_facts: list[NodeFact] = []
    non_facts: list[dict] = []
    rejected_subjects: list[dict] = []
    entity_set = set(entities)

    for rr in raw["relations"]:
        fact_type = rr.get("fact_type", "assertion")
        raw_obj = rr["raw_object_str"].strip()
        desc = rr["relation_description"]
        quote = rr.get("source_quote", "")

        if fact_type not in CONTENT_TYPES:
            non_facts.append({
                "fact_type": fact_type, "subject_entity_id": rr["subject_entity_id"],
                "raw_object_str": raw_obj, "relation_description": desc,
                "source_quote": quote, "doc_id": episode["doc_id"],
            })
            continue

        subj = rr["subject_entity_id"]
        if subj not in entity_set:
            subj_canon, ok = resolve_to_canonical(subj, index, entity_set)
            if ok and subj_canon in entity_set:
                subj = subj_canon
            else:
                rejected_subjects.append({
                    "subject_attempted": rr["subject_entity_id"],
                    "raw_object_str": raw_obj, "relation_description": desc,
                    "doc_id": episode["doc_id"],
                })
                continue

        # entity_set is this episode's canonicals — scopes homographs correctly
        obj_canon, obj_resolved = resolve_to_canonical(raw_obj, index, entity_set)
        has_object = bool(raw_obj)
        verify = verify_quote_and_span(
            raw_obj, quote, episode["text"], rr.get("valid_from"), rr.get("valid_to")
        )
        temporal_inf = verify["temporal_inferred"] or rr.get("temporal_inferred", False)
        conf = score_confidence(
            obj_resolved, verify["quote_verified"], verify["span_verified"],
            temporal_inf, has_object,
        )

        if fact_type in EDGE_TYPES and obj_resolved:
            edges.append(RelationEdge(
                relation_id=make_relation_id(
                    episode["doc_id"], episode["char_start"], subj, raw_obj, desc),
                subject_entity_id=subj,
                relation_description=desc,
                object_raw=raw_obj,
                object_entity_id=obj_canon,
                object_resolved=True,
                fact_type=fact_type,
                valid_from=rr.get("valid_from"),
                valid_to=rr.get("valid_to"),
                temporal_inferred=temporal_inf,
                doc_id=episode["doc_id"],
                char_start=episode["char_start"],
                char_end=episode["char_end"],
                source_quote=quote,
                quote_verified=verify["quote_verified"],
                span_verified=verify["span_verified"],
                confidence=conf,
            ))
        else:
            if has_object and not obj_resolved:
                prov = "unresolved_object"
            elif not has_object:
                prov = "objectless"
            else:
                prov = fact_type
            node_facts.append(NodeFact(
                fact_id=make_fact_id(
                    episode["doc_id"], episode["char_start"], subj, desc),
                entity_id=subj,
                fact_description=desc,
                fact_type=fact_type,
                doc_id=episode["doc_id"],
                char_start=episode["char_start"],
                char_end=episode["char_end"],
                source_quote=quote,
                quote_verified=verify["quote_verified"],
                confidence=conf,
                provenance=prov,
            ))

    uncovered_raw = [e for e in raw.get("uncovered_entities", []) if e in entity_set]
    touched = {e.subject_entity_id for e in edges}
    touched |= {e.object_entity_id for e in edges if e.object_entity_id}
    touched |= {f.entity_id for f in node_facts}
    uncovered = sorted(((entity_set - touched) | set(uncovered_raw)) - touched)

    return EpisodeResult(
        edges=edges, node_facts=node_facts, non_facts=non_facts,
        rejected_subjects=rejected_subjects, uncovered=uncovered,
    )


def _build_episodes(
    candidates: list[dict], unit_to_canonical: dict[str, str]
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], list[str]]]:
    """Reconstruct per-episode text/offsets and canonical entity lists."""
    episodes: dict[tuple[str, str], dict] = {}
    episode_entities: dict[tuple[str, str], list[str]] = {}
    for m in candidates:
        key = (m["source_file"], str(m["episode_idx"]))
        if key not in episodes:
            episodes[key] = {
                "doc_id": m["source_file"],
                "episode_idx": str(m["episode_idx"]),
                "text": m["episode"],
                "char_start": m.get("episode_char_start", -1),
                "char_end": m.get("episode_char_end", -1),
            }
            episode_entities[key] = []
        canon = unit_to_canonical.get(unit_key(m))
        if canon and canon not in episode_entities[key]:
            episode_entities[key].append(canon)
    return episodes, episode_entities


def run(
    *,
    entity_cache: Path = ENTITY_CACHE,
    resolved_cache: Path = RESOLVED_CACHE,
    cache_path: Path = RELATIONS_CACHE,
    model: str = RELATION_MODEL,
    workers: int = RELATION_WORKERS,
    save_every: int = RELATION_SAVE_EVERY,
) -> dict:
    """Extract relations for every eligible episode and persist the cache."""
    candidates = json.loads(entity_cache.read_text())["candidates"]
    resolved = json.loads(resolved_cache.read_text())
    unit_to_canonical: dict[str, str] = resolved["unit_to_canonical"]
    index = build_alias_index(candidates, unit_to_canonical)

    episodes, episode_entities = _build_episodes(candidates, unit_to_canonical)
    eligible = [
        (key, episode)
        for key, episode in episodes.items()
        if episode["char_start"] >= 0 and episode_entities.get(key)
    ]
    bad_offsets = sum(1 for e in episodes.values() if e["char_start"] < 0)
    print(f"[6/6] relations: {len(eligible)} eligible episodes "
          f"({bad_offsets} bad offsets)")

    all_edges: list[dict] = []
    all_node_facts: list[dict] = []
    all_non_facts: list[dict] = []
    all_rejected: list[dict] = []
    all_uncovered: dict[str, list[str]] = {}
    failed_episodes: list[str] = []

    def _save() -> None:
        cache_path.write_text(json.dumps({
            "edges": all_edges,
            "node_facts": all_node_facts,
            "non_facts": all_non_facts,
            "rejected_subjects": all_rejected,
            "uncovered_by_episode": all_uncovered,
            "failed_episodes": failed_episodes,
        }, ensure_ascii=False, indent=1))

    def _process(item: tuple[tuple[str, str], dict]) -> tuple:
        key, episode = item
        result = process_episode(episode, episode_entities[key], index, model=model)
        return (
            str(key),
            [e.model_dump(mode="json") for e in result.edges],
            [f.model_dump(mode="json") for f in result.node_facts],
            result.non_facts,
            result.rejected_subjects,
            result.uncovered,
        )

    if eligible:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, item): item for item in eligible}
            for i, future in enumerate(as_completed(futures)):
                key, _ = futures[future]
                try:
                    k, eb, nfb, xfb, rjb, unc = future.result()
                    all_edges.extend(eb)
                    all_node_facts.extend(nfb)
                    all_non_facts.extend(xfb)
                    all_rejected.extend(rjb)
                    if unc:
                        all_uncovered[k] = unc
                except (AttributeError, KeyError, NameError, TypeError):
                    raise  # code bug — fix, don't skip
                except Exception as exc:  # noqa: BLE001 — API failure per episode
                    print(f"  ✗ episode {key} failed: {exc!r}")
                    failed_episodes.append(str(key))
                if (i + 1) % save_every == 0:
                    _save()

    _save()
    print(f"  {len(all_edges)} edges | {len(all_node_facts)} node-facts | "
          f"{len(failed_episodes)} failed -> {cache_path}")
    return {"edges": all_edges, "node_facts": all_node_facts,
            "failed_episodes": failed_episodes}
