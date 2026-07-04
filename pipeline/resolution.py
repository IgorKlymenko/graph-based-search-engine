"""Stage 4 — LLM entity resolution over HDBSCAN clusters.

Ported from ``entity_extraction.ipynb``. The system prompt, JSON schema, and
``validate_partition`` are verbatim. Two fixes vs. the notebook (spec §8):
the chat timeout is 90s (not 10s, which caused spurious all-singleton
fallbacks), and singletons come from the SAME HDBSCAN partition that was
resolved (clustering.partition), not a separate CC run.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline import clustering
from pipeline.config import (
    ESCALATE_MODEL,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    MENTION_EMBED_MODEL,
    RESOLVE_MODEL,
    RESOLVE_WORKERS,
    RESOLVED_CACHE,
)
from services.openai import complete_json_schema

RESOLVE_SYSTEM = """You decide which of the given entity mentions denote the SAME real-world entity. These mentions were pre-clustered by embedding similarity, so similarity is NOT evidence of sameness — your job is to find the genuine identity boundaries inside a similar-looking set. The resulting entities become nodes in a knowledge graph traversed by PageRank: a wrong merge fuses two entities' neighborhoods and corrupts every traversal through them; an unmerged duplicate merely splits evidence and is cheap to fix later. Default to SEPARATE.

WHAT COUNTS AS EVIDENCE OF SAMENESS (merge only on these):
- Acronym and its expansion, especially when a context shows them introduced together ("Personalized PageRank (PPR)").
- Morphological variants of one name or term: casing, hyphenation, plural, spelling, punctuation ("Neo4j" / "neo4j"; "co-reference" / "coreference").
- Formal vs. short form of one name with compatible contexts ("Acme Corporation" / "Acme Corp" / "Acme", where contexts describe the same organization).
- An explicit alias statement in a context ("X, also known as Y").

WHAT IS NEVER SUFFICIENT: shared head noun, same type, overlapping topic, one being a substring of the other, or contexts about related subject matter. These are why the mentions were clustered together; they prove nothing.

MANDATORY SEPARATIONS:
- Different specificity levels: versions, editions, sub-units, and qualified variants are DISTINCT entities ("GPT-4" ≠ "GPT-4 Turbo"; "University of Toronto" ≠ "University of Toronto Mississauga") unless a context explicitly equates them.
- Different terms for related concepts: merge only terminological variants of ONE term, never two synonyms or near-synonyms ("entity resolution" ≠ "entity deduplication").
- Same surface form, different referents: contexts describing incompatible facts mean separate groups even for identical strings' variants.
- Generic or anaphoric mentions ("the platform", "the model", "this system"): keep as singletons unless a context from the SAME document explicitly ties one to a named member. Across documents, identical generic phrases denote different things.
- Conflicting types (e.g. person vs. organization): separate, unless the contexts make clear one type label is simply an error.
- Related but distinct things (a company and its product, a method and the paper introducing it): separate.

A mention that matches nothing goes in its own single-member group. This is the expected outcome for most mentions, not a failure.

For each group output:
- canonical: the most complete and specific member name. Never generalize, abbreviate, or invent a name absent from the members.
- members: the input mentions in this group, copied VERBATIM, character-for-character, exactly as given.
- confidence: that ALL members denote one entity. "high" = direct evidence from the list above for every member; "medium" = evidence for the pattern but a member's context is uninformative; "low" = plausible but unverified — prefer splitting such a group into a smaller confident core plus singletons over emitting a low-confidence merge.
- reason: one sentence naming the evidence type used (or the boundary kept).

Every input mention must appear in exactly one group. Do not invent, omit, reword, or duplicate any mention."""

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


def validate_partition(entities: list[str], result: dict, verbose: bool = True) -> dict:
    """Enforce 'every mention in exactly one group', with normalization fallback
    before dropping a member. Omissions restored as low-confidence singletons."""
    entity_set = set(entities)

    def norm(s: str) -> str:
        return s.strip().strip("'\"").strip()

    norm_map: dict[str, str] = {}
    for e in entities:
        norm_map.setdefault(norm(e), e)

    seen: set[str] = set()
    clean_groups: list[dict] = []
    dropped_examples: list[str] = []
    dropped = 0
    for g in result["groups"]:
        members = []
        for m in g["members"]:
            cand = m if m in entity_set else norm_map.get(norm(m))
            if cand is not None and cand not in seen:
                members.append(cand)
                seen.add(cand)
            else:
                dropped += 1
                if len(dropped_examples) < 5:
                    dropped_examples.append(m)
        if members:
            clean_groups.append({**g, "members": members})

    missing = sorted(entity_set - seen)
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
    entities: list[str],
    contexts: dict[str, list[str]],
    types: dict[str, str] | None = None,
    counts: dict[str, int] | None = None,
    model: str = RESOLVE_MODEL,
) -> dict:
    """Ask the LLM to group entities in one cluster by real-world identity."""
    lines = []
    for i, e in enumerate(entities, 1):
        picked = _pick_contexts(contexts.get(e, []), k=2)
        ctx_str = " | ".join(f'"{c[:200].strip()}"' for c in picked) or "(no context)"
        meta = []
        if types and e in types:
            meta.append(types[e])
        if counts and e in counts:
            meta.append(f"×{counts[e]}")
        tag = f" [{', '.join(meta)}]" if meta else ""
        lines.append(f"{i}. {e}{tag}\n   contexts: {ctx_str}")
    user_msg = (
        "Entity mentions (copy the mention text exactly as written after the number, "
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
    return validate_partition(entities, result)


def _all_singleton_fallback(entities: list[str], why: str) -> dict:
    return {
        "groups": [
            {"canonical": e, "members": [e], "confidence": "low", "reason": why}
            for e in entities
        ],
        "repaired": 0, "dropped": 0, "failed": True,
    }


def _needs_escalation(result: dict) -> bool:
    return any(
        g["confidence"] == "low" and len(g["members"]) > 1 for g in result["groups"]
    )


def _build_indexes(
    mentions: list[dict],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, int], dict[str, set[str]]]:
    """Per-surface contexts, type, occurrence count, and source files."""
    from collections import defaultdict

    contexts: dict[str, list[str]] = defaultdict(list)
    types: dict[str, str] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for m in mentions:
        text = m["text"]
        contexts[text].append(m["context"])
        types[text] = m["type"]
        counts[text] += 1
        if m.get("source_file"):
            sources[text].add(m["source_file"])
    return contexts, types, dict(counts), sources


def run(
    vectors: dict[str, list[float]],
    mentions: list[dict],
    *,
    resolve_model: str = RESOLVE_MODEL,
    escalate_model: str = ESCALATE_MODEL,
    workers: int = RESOLVE_WORKERS,
    cache_path: Path = RESOLVED_CACHE,
) -> dict:
    """Resolve entities and persist ``resolved_entities.json``."""
    contexts, types, counts, sources = _build_indexes(mentions)
    multi_clusters, singletons = clustering.partition(vectors)
    print(f"[4/6] resolve: {len(multi_clusters)} multi-clusters, "
          f"{len(singletons)} singletons")

    resolved: list[tuple[list[str], dict]] = [([], {})] * len(multi_clusters)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                resolve_cluster, cluster, contexts, types, counts, resolve_model
            ): i
            for i, cluster in enumerate(multi_clusters)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                resolved[i] = (multi_clusters[i], future.result())
            except Exception as exc:  # noqa: BLE001 — API failure, keep pipeline alive
                print(f"  ✗ cluster {i} failed: {exc!r}")
                resolved[i] = (
                    multi_clusters[i],
                    _all_singleton_fallback(multi_clusters[i], f"api failure: {exc}"),
                )

    escalate_idx = [i for i, (_, r) in enumerate(resolved) if _needs_escalation(r)]
    if escalate_idx:
        print(f"  escalating {len(escalate_idx)} marginal cluster(s) to {escalate_model}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    resolve_cluster, multi_clusters[i], contexts, types, counts,
                    escalate_model,
                ): i
                for i in escalate_idx
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    resolved[i] = (multi_clusters[i], future.result())
                except Exception as exc:  # noqa: BLE001 — keep the mini result
                    print(f"  ✗ escalation of cluster {i} failed: {exc!r}")

    def _sources_for(members: list[str]) -> list[str]:
        return sorted({s for m in members for s in sources.get(m, set())})

    entities_out: list[dict] = []
    for _, result in resolved:
        for g in result["groups"]:
            entities_out.append({
                "canonical": g["canonical"],
                "members": g["members"],
                "confidence": g.get("confidence", "high"),
                "reason": g["reason"],
                "source": "llm_resolved",
                "source_files": _sources_for(g["members"]),
            })
    for cluster in singletons:
        surface = cluster[0]
        entities_out.append({
            "canonical": surface,
            "members": [surface],
            "confidence": "high",
            "reason": "singleton cluster — no resolution needed",
            "source": "singleton",
            "source_files": _sources_for([surface]),
        })

    mention_to_canonical: dict[str, str] = {}
    for e in entities_out:
        for member in e["members"]:
            mention_to_canonical[member] = e["canonical"]
    canonical_to_sources = {e["canonical"]: e["source_files"] for e in entities_out}

    payload = {
        "embedding_model": MENTION_EMBED_MODEL,
        "resolve_model": resolve_model,
        "clustering": {
            "method": "hdbscan",
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES,
        },
        "total_entities": len(entities_out),
        "entities": entities_out,
        "mention_to_canonical": mention_to_canonical,
        "canonical_to_sources": canonical_to_sources,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    n_merged = sum(1 for e in entities_out if len(e["members"]) > 1)
    print(f"  {len(entities_out)} canonical entities "
          f"({n_merged} merged, {len(mention_to_canonical)} surfaces) -> {cache_path}")
    return payload
