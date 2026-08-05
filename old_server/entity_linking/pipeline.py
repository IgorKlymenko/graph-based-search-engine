"""Run every stage: documents -> episodes -> chains -> vectors -> entities.

    uv run python -m entity_linking.pipeline --notes-dir ./notes \\
        --tenant-id <uuid> --store ./store.json
    uv run python -m entity_linking.pipeline --store ./store.json --dry-run

Each stage is resumable and writes into the same store, so re-running after a
document changes only redoes the work that document invalidated. The stages are
in ``1_episodes.py`` .. ``4_link.py``; this file is only orchestration.

Two model steps: one coref call per *new* episode, and one adjudication call per
chain pair the scorer cannot settle. Everything else runs locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entity_linking import chains as CH  # noqa: E402
from entity_linking import encoding as EN  # noqa: E402
from entity_linking import episodes as EP  # noqa: E402
from entity_linking import pairing as PR  # noqa: E402
from entity_linking import spans as SP  # noqa: E402
from entity_linking.prompts import (  # noqa: E402
    ENTITY_CHOICE_PROMPT,
    ENTITY_CHOICE_SCHEMA,
    MERGE_REVIEW_PROMPT,
    MERGE_REVIEW_SCHEMA,
)

REVIEW_WORKERS = 8


def sync(store: dict, notes_dir: Path, *, tenant_id: str, source: str) -> dict:
    """Stage 1 — bring the store in line with the corpus."""
    diffs = EP.sync_corpus(store, notes_dir, tenant_id=tenant_id, source=source)
    added = sum(len(d.added) for d in diffs)
    removed = sum(len(d.removed) for d in diffs)
    changed = sum(1 for d in diffs if d.status != "unchanged")
    print(f"[1/4] episodes: {len(diffs)} documents, {changed} changed "
          f"-> +{added} episodes, {removed} retired")
    return {"documents": len(diffs), "added": added, "removed": removed}


def extract(store: dict, *, model: str, workers: int, limit: int | None,
            store_path: Path | None) -> dict:
    """Stage 2 — coref chains for every episode that has none yet."""
    result = CH.run(store, model=model, workers=workers, limit=limit,
                    store_path=store_path)
    print(f"[2/4] chains: {result['episodes']} episodes extracted, "
          f"{result['candidates']} candidates in store")
    return result


def _name_vectors(strings: list[str]) -> dict[str, np.ndarray]:
    """Identity embedding: the head string alone, no episode context.

    Measured at AUC 0.936 for cross-episode identity on ECB+ against 0.745 for
    the contextual chain vector, so this is what the linker keys on. Context
    enters only through ``context_gated``, on pairs this cannot separate.
    """
    import torch

    tokenizer, model, device = EN._load()
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(strings), 64):
        batch = strings[i:i + 64]
        encoded = tokenizer([EN.PREFIX + s for s in batch], return_tensors="pt",
                            padding=True, truncation=True, max_length=64)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.cpu().numpy()
        mask = encoded["attention_mask"].cpu().numpy()[..., None]
        vectors = (hidden * mask).sum(1) / mask.sum(1)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        out.update(dict(zip(batch, vectors)))
    return out


def _context_vectors(store: dict, by_chain: dict[str, list[dict]]) -> dict:
    """One forward pass per episode -> a contextual vector per chain.

    Only consumed by ``context_gated``, which fires on pairs whose surfaces are
    the same string and where the name features carry no information. Uncentered:
    episode-centering measured at chance for identity, uncentered at 0.745.
    """
    by_episode: dict[str, list[str]] = {}
    for chain_id, group in by_chain.items():
        by_episode.setdefault(group[0]["episode_id"], []).append(chain_id)

    out: dict[str, np.ndarray] = {}
    for episode_id, chain_ids in by_episode.items():
        episode = store["episodes"].get(episode_id)
        if episode is None:
            continue
        encoded = EN.encode_episode(episode["text"])
        chains = [[EN.Span(sid=c["candidate_id"], start=c["start"], end=c["end"],
                           text=c["text"], candidate_type=c["candidate_type"],
                           origin=c["origin"])
                   for c in sorted(by_chain[cid], key=lambda c: c["start"])]
                  for cid in chain_ids]
        vectors = EN.encode_chains(episode["text"], chains, encoded=encoded)
        # encode_chains drops a chain whose mentions fell outside the token
        # window, so match on the head span rather than by position
        by_head = {(v.head.start, v.head.end): v.vector for v in vectors}
        for chain_id, chain in zip(chain_ids, chains):
            vector = by_head.get((chain[0].start, chain[0].end))
            if vector is not None:
                out[chain_id] = vector
    return out


def _head_of(group: list[dict]) -> dict:
    """The chain's head, re-picked when the model's choice was filtered out.

    Stage 2 may have made a pronoun the head; once it is gone the chain still
    needs a name, and the longest remaining surface is the same fallback
    ``spans.pick_head`` uses.
    """
    return next((c for c in group if c["is_head"]),
                max(group, key=lambda c: len(c["text"])))


def encode(store: dict, *, context: bool = True) -> list[PR.ChainRecord]:
    """Stage 3 — one ChainRecord per live chain, carrying both vector kinds."""
    live = [c for c in store["candidates"].values()
            if store["episodes"].get(c["episode_id"], {}).get("removed_at") is None]
    by_chain: dict[str, list[dict]] = {}
    for candidate in live:
        by_chain.setdefault(candidate["chain_id"], []).append(candidate)

    # A pronoun is a legitimate chain member -- it is how the model linked the
    # mentions -- but it is not a name, and letting it into `surfaces` makes two
    # chains that both say "it" score string_similarity 1.0 and lets
    # ("Sarah Chen", "she") count as an alias. Candidates keep them; identity
    # does not. `is_identifying` also drops "the", "three" and friends.
    identifying: dict[str, list[dict]] = {}
    dropped_chains = dropped_surfaces = 0
    for chain_id, group in by_chain.items():
        kept = [c for c in group if SP.is_identifying(c["text"])]
        dropped_surfaces += len(group) - len(kept)
        if kept:
            identifying[chain_id] = kept
        else:
            dropped_chains += 1  # nothing left that could name anything

    heads = sorted({_head_of(group)["text"] for group in identifying.values()})
    vectors = _name_vectors(heads) if heads else {}
    contexts = _context_vectors(store, by_chain) if context else {}

    records: list[PR.ChainRecord] = []
    for chain_id, group in identifying.items():
        head = _head_of(group)
        records.append(PR.ChainRecord(
            chain_id=chain_id, episode_id=head["episode_id"], head=head["text"],
            surfaces=frozenset(c["text"] for c in group),
            candidate_type=head["candidate_type"],
            name_vector=vectors.get(head["text"]),
            context_vector=contexts.get(chain_id)))
    print(f"[3/4] encode: {len(records)} chains, {len(heads)} distinct names, "
          f"{len(contexts)} context vectors "
          f"({dropped_surfaces} non-identifying surfaces dropped, "
          f"{dropped_chains} chains left with none)")
    return records


def link(records: list[PR.ChainRecord], *, model: str, k: int,
         thresholds: PR.Thresholds, workers: int,
         passages: dict[str, str] | None = None) -> dict:
    """Stage 4 — block, score, review the uncertain band, cluster."""
    evidence = PR.build_evidence(records)
    scorer = PR.PairScorer()
    candidates = (PR.ann_candidate_pairs(records, k=k) if len(records) > k
                  else PR.candidate_pairs(records))
    scored = [(a, b, scorer.score(PR.features(a, b, evidence)))
              for a, b in candidates]

    to_review = [(a, b) for a, b, p in scored
                 if PR.decide(p, thresholds) == PR.REVIEW]
    answers: dict[tuple[str, str], dict] = {}
    if to_review:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = dict(zip(
                [(a.chain_id, b.chain_id) for a, b in to_review],
                pool.map(lambda ab: _adjudicate(*ab, model, evidence, passages),
                         to_review)))

    merges: list[tuple[str, str, float]] = []
    for a, b, probability in scored:
        verdict = PR.decide(probability, thresholds)
        if verdict == PR.REVIEW:
            answer = answers[(a.chain_id, b.chain_id)]
            verdict = PR.MERGE if answer["same_entity"] else PR.SPLIT
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))

    assignment = PR.cluster([r.chain_id for r in records], merges)
    print(f"[4/4] link: {len(candidates)} pairs scored, {len(to_review)} reviewed, "
          f"{len(merges)} merged -> {len(set(assignment.values()))} entities")
    return {"assignment": assignment, "reviewed": len(to_review),
            "merges": len(merges), "pairs": len(candidates),
            "evidence": evidence}


def _profile(members: list[PR.ChainRecord], passages: dict[str, str],
             max_passages: int = 2) -> str:
    """What the graph already knows about one entity, for the reviewer to use."""
    surfaces = sorted({s for m in members for s in m.surfaces})
    episodes = sorted({m.episode_id for m in members})[:max_passages]
    seen = " | ".join(passages.get(e, "")[:300] for e in episodes)
    return f"names: {surfaces}\n  appears in: {seen}"


def resolve_against_graph(
    records: list[PR.ChainRecord], assignment: dict[str, int],
    pending: dict[str, list[int]], passages: dict[str, str], *, model: str,
    workers: int = REVIEW_WORKERS,
) -> dict[str, dict]:
    """Ask the model which known entity an ambiguous mention belongs to.

    The pairwise reviewer sees two passages and nothing else, which is why it
    scores 81% on the pairs the scorer could not settle. Here it sees the mention
    against a *profile* of each candidate entity assembled from the rest of the
    corpus — every name that entity goes by and the passages it appears in. The
    name is known to be uninformative, so the decision has to come from whether
    the mention's activity matches one candidate's established record.

    Returns ``chain_id -> {"cluster": int | None, "answer": {...}}``. A null
    cluster means it declined, and the mention stays unresolved: a wrong
    attachment asserts false facts about a real entity, an unplaced mention only
    stays unplaced. The raw answer is kept so callers can show the reasoning.
    """
    if not pending:
        return {}
    by_cluster: dict[int, list[PR.ChainRecord]] = {}
    by_id: dict[str, PR.ChainRecord] = {}
    for record in records:
        by_cluster.setdefault(assignment[record.chain_id], []).append(record)
        by_id[record.chain_id] = record

    def ask(item: tuple[str, list[int]]) -> tuple[str, dict]:
        chain_id, clusters = item
        record = by_id[chain_id]
        profiles = "\n\n".join(
            f"[{cluster}] {_profile(by_cluster[cluster], passages)}"
            for cluster in clusters)
        user = (f"Mention: {sorted(record.surfaces)}\n"
                f"Passage: {passages.get(record.episode_id, '')}\n\n"
                f"Candidate entities:\n\n{profiles}")
        answer = CH._complete_json(
            system=ENTITY_CHOICE_PROMPT, user=user, model=model,
            schema=ENTITY_CHOICE_SCHEMA, schema_name="entity_choice")
        try:
            picked = int(answer.get("entity_id"))
        except (TypeError, ValueError):
            picked = None
        if picked not in clusters:
            picked = None
        return chain_id, {"cluster": picked, "answer": answer,
                          "candidates": clusters, "prompt": user}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(ask, pending.items()))


def _adjudicate(a: PR.ChainRecord, b: PR.ChainRecord, model: str,
                evidence: PR.Evidence, passages: dict[str, str] | None) -> dict:
    contested = sorted((a.surfaces | b.surfaces) & evidence.ambiguous_surfaces)
    note = (f"\n\nNOTE: {contested} is used for more than one entity elsewhere."
            if contested else "")
    text = passages or {}
    user = (f"A names: {sorted(a.surfaces)}\n"
            f"A passage: {text.get(a.episode_id, '')}\n\n"
            f"B names: {sorted(b.surfaces)}\n"
            f"B passage: {text.get(b.episode_id, '')}{note}")
    return CH._complete_json(system=MERGE_REVIEW_PROMPT, user=user, model=model,
                             schema=MERGE_REVIEW_SCHEMA,
                             schema_name="merge_review")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--source", default=EP.UPLOAD_SOURCE)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--workers", type=int, default=CH.EPISODE_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract at most N episodes this run")
    parser.add_argument("--ann-k", type=int, default=50)
    # The prior weights top out at 0.50 for an exact name match, so a higher
    # merge cut point never fires. Do NOT replace these with choose_thresholds()
    # output: on ECB+ its precision target scored 0.634 CoNLL against 0.775 for
    # the cut chosen to maximise CoNLL directly. See DATAFLOW.md.
    parser.add_argument("--merge-at", type=float, default=0.50,
                        help="Score at or above which pairs merge unreviewed")
    parser.add_argument("--split-at", type=float, default=0.30,
                        help="Score below which pairs split unreviewed")
    parser.add_argument("--no-graph-resolve", action="store_true",
                        help="Skip the graph-context call for ambiguous mentions")
    parser.add_argument("--no-context", action="store_true",
                        help="Skip the per-episode forward pass; disables "
                             "context_gated, which only fires on same-string pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stages 1 and 3 only; no model calls")
    parser.add_argument("--entities-out", type=Path, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    store = EP.load_store(args.store)

    if args.notes_dir:
        if not args.tenant_id:
            sys.exit("--tenant-id is required with --notes-dir")
        if not args.notes_dir.is_dir():
            sys.exit(f"not a directory: {args.notes_dir}")
        sync(store, args.notes_dir, tenant_id=args.tenant_id, source=args.source)

    if args.dry_run:
        pending = CH.pending_episodes(store)
        print(f"[2/4] dry-run: {len(pending)} episodes pending extraction")
        EP.save_store(args.store, store)
        return

    extract(store, model=args.model, workers=args.workers, limit=args.limit,
            store_path=args.store)
    records = encode(store, context=not args.no_context)
    if len(records) < 2:
        print("  fewer than two chains; nothing to link")
        EP.save_store(args.store, store)
        return

    passages = {eid: ep["text"] for eid, ep in store["episodes"].items()}
    result = link(records, model=args.model, k=args.ann_k, workers=REVIEW_WORKERS,
                  thresholds=PR.Thresholds(low=args.split_at, high=args.merge_at),
                  passages=passages)

    assignment = result["assignment"]
    pending = PR.unresolved(records, assignment, result["evidence"])
    placed: dict[str, int] = {}
    if pending and not args.no_graph_resolve:
        decided = resolve_against_graph(records, assignment, pending, passages,
                                        model=args.model)
        placed = {cid: r["cluster"] for cid, r in decided.items()
                  if r["cluster"] is not None}
        for chain_id, cluster in placed.items():
            assignment[chain_id] = cluster
        pending = {k: v for k, v in pending.items() if k not in placed}
    if pending or placed:
        print(f"  ambiguous mentions: {len(placed)} placed by the graph, "
              f"{len(pending)} left unresolved (no node created)")

    # Name clusters by their smallest chain id rather than by position, so an
    # entity whose membership did not change keeps its id across re-runs.
    entity_id_of = PR.stable_entity_ids(assignment)
    entity_of_cluster = {assignment[c]: e for c, e in entity_id_of.items()}
    entities: dict[str, dict] = {}
    for record in records:
        if record.chain_id in pending:
            continue  # a contested name with nothing to anchor it is not an entity
        entity = entities.setdefault(entity_id_of[record.chain_id],
                                     {"surfaces": set(), "chain_ids": [],
                                      "episode_ids": set()})
        entity["surfaces"] |= record.surfaces
        entity["chain_ids"].append(record.chain_id)
        entity["episode_ids"].add(record.episode_id)
    store["entities"] = {
        entity_id: {"surfaces": sorted(entity["surfaces"]),
                    "chain_ids": sorted(entity["chain_ids"]),
                    "episode_ids": sorted(entity["episode_ids"])}
        for entity_id, entity in sorted(entities.items())
    }
    by_id = {r.chain_id: r for r in records}
    store["unresolved"] = [
        {"chain_id": chain_id, "episode_id": by_id[chain_id].episode_id,
         "surfaces": sorted(by_id[chain_id].surfaces),
         "candidates": sorted(entity_of_cluster[c] for c in clusters),
         "reason": "contested_surface"}
        for chain_id, clusters in sorted(pending.items())
    ]
    EP.save_store(args.store, store)

    print(f"\nstore -> {args.store}  "
          f"({len(store['files'])} documents, {len(store['episodes'])} episodes, "
          f"{len(store['candidates'])} candidates, {len(store['entities'])} entities)")
    print(f"wall clock {time.perf_counter() - started:.1f}s")
    if args.entities_out:
        args.entities_out.write_text(json.dumps(store["entities"], indent=2))
        print(f"entities -> {args.entities_out}")


if __name__ == "__main__":
    main()
