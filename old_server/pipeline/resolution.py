"""Stage 4 — LLM entity resolution over HDBSCAN clusters.

Ported from ``entity_extraction.ipynb``. The system prompt, JSON schema, and
``validate_partition`` are verbatim. Two fixes vs. the notebook (spec §8):
the chat timeout is 90s (not 10s, which caused spurious all-singleton
fallbacks), and singletons come from the SAME HDBSCAN partition that was
resolved (clustering.partition), not a separate CC run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from pipeline import clustering
from pipeline.config import (
    ESCALATE_MODEL,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    CANDIDATE_EMBED_MODEL,
    RESOLVE_MODEL,
    RESOLVE_WORKERS,
    RESOLVED_CACHE,
)
from pipeline.models import unit_key
from services.openai import complete_json_schema

RESOLVE_SYSTEM = """You decide which of the given entity candidates denote the SAME real-world entity. These candidates were pre-clustered by embedding similarity, so similarity is NOT evidence of sameness — your job is to find the genuine identity boundaries inside a similar-looking set. The resulting entities become nodes in a knowledge graph traversed by PageRank: a wrong merge fuses two entities' neighborhoods and corrupts every traversal through them; an unmerged duplicate merely splits evidence and is cheap to fix later. Default to SEPARATE.

WHAT COUNTS AS EVIDENCE OF SAMENESS (merge only on these):
- Acronym and its expansion, especially when a context shows them introduced together ("Personalized PageRank (PPR)").
- Morphological variants of one name or term: casing, hyphenation, plural, spelling, punctuation ("Neo4j" / "neo4j"; "co-reference" / "coreference").
- Formal vs. short form of one name with compatible contexts ("Acme Corporation" / "Acme Corp" / "Acme", where contexts describe the same organization).
- An explicit alias statement in a context ("X, also known as Y").

WHAT IS NEVER SUFFICIENT: shared head noun, same type, overlapping topic, one being a substring of the other, or contexts about related subject matter. These are why the candidates were clustered together; they prove nothing.

MANDATORY SEPARATIONS:
- Different specificity levels: versions, editions, sub-units, and qualified variants are DISTINCT entities ("GPT-4" ≠ "GPT-4 Turbo"; "University of Toronto" ≠ "University of Toronto Mississauga") unless a context explicitly equates them.
- Different terms for related concepts: merge only terminological variants of ONE term, never two synonyms or near-synonyms ("entity resolution" ≠ "entity deduplication").
- Same concept form, different referents: contexts describing incompatible facts mean separate groups even for identical strings' variants.
- Generic or anaphoric candidates ("the platform", "the model", "this system"): keep as singletons unless a context from the SAME document explicitly ties one to a named member. Across documents, identical generic phrases denote different things.
- Conflicting types (e.g. person vs. organization): separate, unless the contexts make clear one type label is simply an error.
- Related but distinct things (a company and its product, a method and the paper introducing it): separate.

A candidate that matches nothing goes in its own single-member group. This is the expected outcome for most candidates, not a failure.

For each group output:
- canonical: the most complete and specific member name. Never generalize, abbreviate, or invent a name absent from the members.
- members: the input candidates in this group, copied VERBATIM, character-for-character, exactly as given.
- confidence: that ALL members denote one entity. "high" = direct evidence from the list above for every member; "medium" = evidence for the pattern but a member's context is uninformative; "low" = plausible but unverified — prefer splitting such a group into a smaller confident core plus singletons over emitting a low-confidence merge.
- reason: one sentence naming the evidence type used (or the boundary kept).

Every input candidate must appear in exactly one group. Do not invent, omit, reword, or duplicate any candidate."""

RESOLVE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical": {"type": "string"},
                    "members": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["canonical", "members", "confidence", "reason"],
            },
        }
    },
    "required": ["groups"],
}


def validate_partition(
    concepts: list[str], result: dict, verbose: bool = True
) -> dict:
    """Enforce 'every concept in exactly one group', with normalization fallback
    before dropping a member. Omissions restored as low-confidence singletons."""
    concept_set = set(concepts)

    def norm(s: str) -> str:
        return s.strip().strip("'\"").strip()

    norm_map: dict[str, str] = {}
    for s in concepts:
        norm_map.setdefault(norm(s), s)

    seen: set[str] = set()
    clean_groups: list[dict] = []
    dropped_examples: list[str] = []
    dropped = 0
    for g in result["groups"]:
        members = []
        for m in g["members"]:
            cand = m if m in concept_set else norm_map.get(norm(m))
            if cand is not None and cand not in seen:
                members.append(cand)
                seen.add(cand)
            else:
                dropped += 1
                if len(dropped_examples) < 5:
                    dropped_examples.append(m)
        if members:
            clean_groups.append({**g, "members": members})

    missing = sorted(concept_set - seen)
    for m in missing:
        clean_groups.append({
            "canonical": m, "members": [m],
            "confidence": "low",
            "reason": "repair: omitted or reworded by model",
        })

    if verbose and (dropped or missing):
        print(f"  ⚠ validate: dropped={dropped} repaired={len(missing)} "
              f"examples={dropped_examples!r}")

    result["groups"] = clean_groups
    result["repaired"] = len(missing)
    result["dropped"] = dropped
    return result


def _pick_contexts(ctxs: list[str], k: int = 2) -> list[str]:
    """Pick up to k contexts spread across the list (proxy for source diversity)."""
    if len(ctxs) <= k:
        return ctxs
    if k == 1:
        return [ctxs[0]]
    step = (len(ctxs) - 1) / (k - 1)
    return [ctxs[round(i * step)] for i in range(k)]


def resolve_cluster(
    concepts: list[str],
    contexts: dict[str, list[str]],
    types: dict[str, str] | None = None,
    counts: dict[str, int] | None = None,
    model: str = RESOLVE_MODEL,
) -> dict:
    """Ask the LLM to group one cluster's concepts by real-world identity."""
    lines = []
    for i, s in enumerate(concepts, 1):
        picked = _pick_contexts(contexts.get(s, []), k=2)
        ctx_str = " | ".join(f'"{c[:200].strip()}"' for c in picked) or "(no context)"
        meta = []
        if types and s in types:
            meta.append(types[s])
        if counts and s in counts:
            meta.append(f"×{counts[s]}")
        tag = f" [{', '.join(meta)}]" if meta else ""
        lines.append(f"{i}. {s}{tag}\n   contexts: {ctx_str}")
    user_msg = (
        "Entity candidates (copy each concept exactly as written after the number, "
        "without the number):\n" + "\n".join(lines)
    )

    result = complete_json_schema(
        system=RESOLVE_SYSTEM,
        user=user_msg,
        schema=RESOLVE_SCHEMA,
        schema_name="entity_groups",
        model=model,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )
    return validate_partition(concepts, result)


def _all_singleton_fallback(concepts: list[str], why: str) -> dict:
    return {
        "groups": [
            {"canonical": s, "members": [s], "confidence": "low", "reason": why}
            for s in concepts
        ],
        "repaired": 0, "dropped": 0, "failed": True,
    }


def _needs_escalation(result: dict) -> bool:
    return any(
        g["confidence"] == "low" and len(g["members"]) > 1 for g in result["groups"]
    )


class UnitInfo(NamedTuple):
    """What stage 4 needs to know about one resolution unit."""

    concept: str
    context: str
    type: str
    source_file: str
    occurrences: int


def _build_units(candidates: list[dict]) -> dict[str, UnitInfo]:
    """One entry per ``models.unit_key``, carrying how many candidates collapsed in."""
    counts: Counter[str] = Counter()
    info: dict[str, UnitInfo] = {}
    for m in candidates:
        key = unit_key(m)
        counts[key] += 1
        if key not in info:
            info[key] = UnitInfo(
                concept=m["concept"],
                context=m["context"],
                type=m["type"],
                source_file=m.get("source_file", ""),
                occurrences=0,
            )
    return {k: v._replace(occurrences=counts[k]) for k, v in info.items()}


class ClusterView(NamedTuple):
    """One cluster reduced to the distinct concepts the LLM must decide between."""

    concepts: list[str]
    contexts: dict[str, list[str]]
    types: dict[str, str]
    counts: dict[str, int]


def _cluster_view(cluster: list[str], units: dict[str, UnitInfo]) -> ClusterView:
    """Collapse a cluster's units to its distinct concepts.

    Units sharing a concept need no adjudication against each other — the LLM
    only ever decides across distinct strings — so the prompt lists each concept
    once with the contexts seen *in this cluster*, and the verdict is mapped back
    to every unit afterwards. This is what keeps prompts bounded now that a
    popular concept contributes hundreds of units instead of one.
    """
    contexts: dict[str, list[str]] = defaultdict(list)
    types: dict[str, str] = {}
    counts: Counter[str] = Counter()
    order: list[str] = []
    for key in cluster:
        u = units.get(key)
        if u is None:
            continue
        if u.concept not in types:
            order.append(u.concept)
        contexts[u.concept].append(u.context)
        types[u.concept] = u.type
        counts[u.concept] += u.occurrences
    return ClusterView(order, dict(contexts), types, dict(counts))


def _entity_id(members: list[str]) -> str:
    """Stable id for a resolved entity, derived from its unit membership.

    Entity identity cannot key on the canonical name any more: two clusters may
    both resolve to the name "Apple" while denoting different things, and a
    name-keyed uuid would silently fuse them again at upload.
    """
    joined = "\x1e".join(sorted(members))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def run(
    unit_vectors: dict[str, list[float]],
    candidates: list[dict],
    *,
    resolve_model: str = RESOLVE_MODEL,
    escalate_model: str = ESCALATE_MODEL,
    workers: int = RESOLVE_WORKERS,
    cache_path: Path = RESOLVED_CACHE,
) -> dict:
    """Group units into canonical entities; persist ``resolved_entities.json``.

    Clustering happens over units, so one concept can land in several clusters
    and resolve to several entities. Adjudication happens over the distinct
    concepts inside each cluster — a cluster holding one distinct concept is
    settled without an LLM call.
    """
    units = _build_units(candidates)
    multi_clusters, singletons = clustering.partition(unit_vectors)
    views = [_cluster_view(c, units) for c in multi_clusters]
    need_llm = [i for i, v in enumerate(views) if len(v.concepts) > 1]
    print(f"[4/6] resolve: {len(multi_clusters)} multi-clusters "
          f"({len(need_llm)} with >1 distinct concept -> LLM), "
          f"{len(singletons)} singleton units")

    resolved: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resolve_cluster, views[i].concepts, views[i].contexts,
                views[i].types, views[i].counts, resolve_model,
            ): i
            for i in need_llm
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                resolved[i] = future.result()
            except Exception as exc:  # noqa: BLE001 — API failure, keep pipeline alive
                print(f"  ✗ cluster {i} failed: {exc!r}")
                resolved[i] = _all_singleton_fallback(
                    views[i].concepts, f"api failure: {exc}"
                )

    escalate_idx = [i for i, r in resolved.items() if _needs_escalation(r)]
    if escalate_idx:
        print(f"  escalating {len(escalate_idx)} marginal cluster(s) to {escalate_model}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    resolve_cluster, views[i].concepts, views[i].contexts,
                    views[i].types, views[i].counts, escalate_model,
                ): i
                for i in escalate_idx
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    resolved[i] = future.result()
                except Exception as exc:  # noqa: BLE001 — keep the mini result
                    print(f"  ✗ escalation of cluster {i} failed: {exc!r}")

    def _emit(members: list[str], canonical: str, confidence: str, reason: str,
              source: str) -> None:
        if not members:
            return
        entities_out.append({
            "canonical": canonical,
            "members": members,
            "confidence": confidence,
            "reason": reason,
            "source": source,
        })

    entities_out: list[dict] = []
    for i, cluster in enumerate(multi_clusters):
        known = [k for k in cluster if k in units]
        if i not in resolved:
            # single distinct concept in this cluster — nothing to adjudicate
            if known:
                _emit(known, units[known[0]].concept, "high",
                      "one distinct concept in cluster — no resolution needed",
                      "single_concept_cluster")
            continue
        for g in resolved[i]["groups"]:
            group_cands = set(g["members"])
            _emit([k for k in known if units[k].concept in group_cands],
                  g["canonical"], g.get("confidence", "high"), g["reason"],
                  "llm_resolved")

    # Singleton units carry NO separation evidence, so splitting them would just
    # fragment one entity across its own candidates. A split needs a positive
    # signal — units of one concept landing in DIFFERENT clusters. So: fold a
    # concept's unclustered units into its cluster-derived entity when there is
    # exactly one; otherwise (none, or several — ambiguous) group them together.
    holders: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entities_out):
        for cand in {units[k].concept for k in e["members"]}:
            holders[cand].append(i)

    singleton_by_concept: dict[str, list[str]] = defaultdict(list)
    for cluster in singletons:
        key = cluster[0]
        if key in units:
            singleton_by_concept[units[key].concept].append(key)

    folded = 0
    for concept, keys in singleton_by_concept.items():
        owners = holders.get(concept, [])
        if len(owners) == 1:
            entities_out[owners[0]]["members"].extend(keys)
            folded += len(keys)
        else:
            _emit(keys, concept, "high",
                  "unclustered units — merged by concept, no evidence to split",
                  "singleton")

    # entity_id only now that membership is final
    for e in entities_out:
        e["entity_id"] = _entity_id(e["members"])
        e["member_concepts"] = sorted({units[k].concept for k in e["members"]})
        e["source_files"] = sorted(
            {units[k].source_file for k in e["members"] if units[k].source_file}
        )

    # keyed by UNIT, not concept — the same concept may map to two entities
    unit_to_canonical: dict[str, str] = {}
    unit_to_entity_id: dict[str, str] = {}
    for e in entities_out:
        for member in e["members"]:
            unit_to_canonical[member] = e["canonical"]
            unit_to_entity_id[member] = e["entity_id"]

    payload = {
        "embedding_model": CANDIDATE_EMBED_MODEL,
        "resolve_model": resolve_model,
        "resolution_unit": "concept+context+episode",
        "clustering": {
            "method": "hdbscan",
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
        },
        "total_entities": len(entities_out),
        "entities": entities_out,
        "unit_to_canonical": unit_to_canonical,
        "unit_to_entity_id": unit_to_entity_id,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"  {len(entities_out)} entities over {len(unit_to_canonical)} units "
          f"({folded} unclustered units folded in); "
          f"{_homograph_count(entities_out)} canonical names now denote >1 entity "
          f"-> {cache_path}")
    return payload


def _homograph_count(entities: list[dict]) -> int:
    """Canonical names shared by more than one entity — the point of unit-level."""
    counts = Counter(e["canonical"] for e in entities)
    return sum(1 for n in counts.values() if n > 1)
