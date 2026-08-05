"""Run one pasted document through all four stages, dumping every mid-step.

    pbpaste | uv run python entity_linking/demos/demo_document.py
    uv run python entity_linking/demos/demo_document.py --file notes.md
    uv run python entity_linking/demos/demo_document.py --file notes.md --local

Reads the document from stdin (paste, then Ctrl-D) or from ``--file``, then runs
split -> extract -> embed -> link and writes one JSON holding the input and the
output of every stage, so each step can be inspected without re-running the
others.

``--local`` stops after the span inventory: no LLM calls, no model download, no
cost. That is the fastest way to see what the parser hands the model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from entity_linking import chains as CH  # noqa: E402
from entity_linking import episodes as EP  # noqa: E402
from entity_linking import pairing as PR  # noqa: E402
from entity_linking import pipeline as PL  # noqa: E402
from entity_linking import spans as SP  # noqa: E402

TENANT_ID = "00000000-0000-0000-0000-000000000001"
VECTOR_PREVIEW = 6


def read_document(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text()
    if sys.stdin.isatty():
        print("Paste the document, then Ctrl-D:\n", file=sys.stderr)
    return sys.stdin.read()


def split(store: dict, doc: str, *, external_id: str, max_chars: int) -> dict:
    """Stage 1 — document into content-addressed episodes."""
    diff = EP.sync_file(store, doc, tenant_id=TENANT_ID, source=EP.UPLOAD_SOURCE,
                        external_id=external_id, max_chars=max_chars)
    file_record = store["files"][diff.file_id]
    episodes = [store["episodes"][eid] for eid in file_record["episode_ids"]]
    print(f"[1/4] split: {len(episodes)} episodes "
          f"({min(len(e['text']) for e in episodes)}-"
          f"{max(len(e['text']) for e in episodes)} chars)")
    return {"file": file_record, "episodes": episodes}


def inventory(text: str) -> dict:
    """The local half of stage 2: what the parser finds, and what the model sees.

    Recomputed rather than captured from ``extract_chains`` so ``--local`` can
    show it without an API key. Both call the same two functions on the same
    text, so the result is identical.
    """
    parser_spans = SP.parse_spans(text)
    return {
        "normalized": SP.normalize_for_parse(text),
        "marked_text": SP.mark_text(text, parser_spans),
        "spans": [s._asdict() for s in parser_spans],
    }


def extract(store: dict, episodes: list[dict], *, model: str) -> dict:
    """Stage 2 — one LLM call per episode, keeping every intermediate."""
    per_episode = []
    for episode in episodes:
        episode_id, text = episode["episode_id"], episode["text"]
        step = {"episode_id": episode_id, "text": text, **inventory(text)}

        annotated, failure = None, None
        for attempt in range(2):  # same contract as CH.process_episode
            try:
                annotated = CH.extract_chains(text, model=model)
                break
            except CH.UnknownSpanIds as exc:
                failure = {"stage": "validate", "attempt": attempt,
                           "error": repr(exc)}
            except Exception as exc:  # noqa: BLE001 — one episode, not the run
                failure = {"stage": "chains", "attempt": attempt,
                           "error": repr(exc)}
                break

        if annotated is None:
            step["failure"] = failure
            per_episode.append(step)
            continue

        resolved = CH.resolve_chains(annotated)
        records = CH.build_candidates(episode_id, resolved)
        for record in records:
            store["candidates"][record["candidate_id"]] = record
        store["episodes"][episode_id]["extracted_at"] = "demo"

        step["model_raw"] = {
            "chains": annotated["raw_chains"],
            "new_spans": [s._asdict() for s in annotated["new_spans"]],
            "dropped_new_spans": annotated["dropped_new_spans"],
            "collapsed": annotated["collapsed"],
        }
        step["chains"] = [
            {"head": chain[0].text,
             "members": [s._asdict() for s in chain]}
            for chain in resolved
        ]
        step["candidates"] = records
        per_episode.append(step)

    failed = [s["episode_id"] for s in per_episode if "failure" in s]
    print(f"[2/4] extract: {len(per_episode) - len(failed)}/{len(per_episode)} "
          f"episodes, {len(store['candidates'])} candidates"
          + (f", {len(failed)} FAILED" if failed else ""))
    return {"episodes": per_episode, "failed": failed}


def encode(store: dict) -> tuple[list[PR.ChainRecord], dict]:
    """Stage 3 — one identity vector per chain (the context vector is unused)."""
    records = PL.encode(store)
    dump = [
        {"chain_id": r.chain_id, "episode_id": r.episode_id, "head": r.head,
         "surfaces": sorted(r.surfaces), "candidate_type": r.candidate_type,
         "name_vector_dim": None if r.name_vector is None else len(r.name_vector),
         "name_vector_head": None if r.name_vector is None else
         [round(float(x), 4) for x in r.name_vector[:VECTOR_PREVIEW]]}
        for r in records
    ]
    return records, {"chains": dump}


def link(records: list[PR.ChainRecord], passages: dict[str, str], *,
         model: str, thresholds: PR.Thresholds, review: bool) -> dict:
    """Stage 4 — evidence, scores, review, clusters, with the per-pair detail."""
    evidence = PR.build_evidence(records)
    scorer = PR.PairScorer()

    scored = []
    for a, b in itertools.combinations(records, 2):
        if a.episode_id == b.episode_id:
            continue
        feats = PR.features(a, b, evidence)
        scored.append((a, b, feats, scorer.score(feats)))

    reviews: dict[tuple[str, str], dict] = {}
    if review:
        for a, b, _feats, probability in scored:
            if PR.decide(probability, thresholds) == PR.REVIEW:
                reviews[(a.chain_id, b.chain_id)] = PL._adjudicate(
                    a, b, model, evidence, passages)

    pairs, merges = [], []
    for a, b, feats, probability in scored:
        verdict = PR.decide(probability, thresholds)
        answer = reviews.get((a.chain_id, b.chain_id))
        if verdict == PR.REVIEW:
            verdict = (PR.MERGE if answer and answer.get("same_entity")
                       else PR.SPLIT)
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))
        pairs.append({
            "a": {"chain_id": a.chain_id, "surfaces": sorted(a.surfaces)},
            "b": {"chain_id": b.chain_id, "surfaces": sorted(b.surfaces)},
            "features": {k: round(float(v), 4) for k, v in feats.items()},
            "probability": round(probability, 4),
            "verdict": verdict, "review": answer,
        })

    assignment = PR.cluster([r.chain_id for r in records], merges)
    pending = PR.unresolved(records, assignment, evidence)
    placed: dict[str, int] = {}
    if pending and review:
        decided = PL.resolve_against_graph(records, assignment, pending,
                                           passages, model=model)
        placed = {cid: r["cluster"] for cid, r in decided.items()
                  if r["cluster"] is not None}
        for chain_id, cluster in placed.items():
            assignment[chain_id] = cluster
        pending = {k: v for k, v in pending.items() if k not in placed}

    entities: dict[int, dict] = {}
    for record in records:
        if record.chain_id in pending:
            continue  # contested name, nothing to anchor it: not an entity
        entity = entities.setdefault(assignment[record.chain_id],
                                     {"surfaces": set(), "chain_ids": [],
                                      "episode_ids": []})
        entity["surfaces"] |= record.surfaces
        entity["chain_ids"].append(record.chain_id)
        entity["episode_ids"].append(record.episode_id)

    by_id = {r.chain_id: r for r in records}
    print(f"[4/4] link: {len(pairs)} pairs, {len(reviews)} reviewed, "
          f"{len(merges)} merged -> {len(entities)} entities")
    if pending or placed:
        print(f"      ambiguous mentions: {len(placed)} placed by the graph, "
              f"{len(pending)} left unresolved")
    return {
        "evidence": {
            "same_chain": sorted(map(list, evidence.same_chain)),
            "split_in_episode": sorted(map(list, evidence.split_in_episode)),
            "ambiguous_surfaces": sorted(evidence.ambiguous_surfaces),
        },
        "thresholds": {"split_below": thresholds.low,
                       "merge_at": thresholds.high},
        "pairs": pairs,
        "entities": {
            str(cluster): {"surfaces": sorted(entity["surfaces"]),
                           "chain_ids": entity["chain_ids"],
                           "episode_ids": sorted(set(entity["episode_ids"]))}
            for cluster, entity in sorted(entities.items())
        },
        "placed_by_graph": {cid: str(c) for cid, c in sorted(placed.items())},
        "unresolved": [
            {"chain_id": chain_id, "episode_id": by_id[chain_id].episode_id,
             "surfaces": sorted(by_id[chain_id].surfaces),
             "candidates": [str(c) for c in clusters],
             "reason": "contested_surface"}
            for chain_id, clusters in sorted(pending.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None,
                        help="Read the document from a file instead of stdin")
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/document_trace.json"))
    parser.add_argument("--external-id", default="pasted.md")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--merge-at", type=float, default=0.50)
    parser.add_argument("--split-at", type=float, default=0.30)
    parser.add_argument("--local", action="store_true",
                        help="Stages 1 and the span inventory only; no LLM")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip stage 4's LLM calls; the band splits")
    args = parser.parse_args()

    doc = read_document(args)
    if not doc.strip():
        sys.exit("empty document")

    started = time.perf_counter()
    store = EP.empty_store()
    trace: dict = {"input": {"chars": len(doc), "external_id": args.external_id,
                             "max_chars": args.max_chars, "text": doc}}

    trace["stage1_split"] = split(store, doc, external_id=args.external_id,
                                  max_chars=args.max_chars)
    episodes = trace["stage1_split"]["episodes"]

    if args.local:
        trace["stage2_extract"] = {"episodes": [
            {"episode_id": e["episode_id"], "text": e["text"],
             **inventory(e["text"])} for e in episodes]}
        total = sum(len(s["spans"]) for s in trace["stage2_extract"]["episodes"])
        print(f"[2/4] local: {total} parser spans; stopping before the LLM")
    else:
        trace["stage2_extract"] = extract(store, episodes, model=args.model)
        records, trace["stage3_encode"] = encode(store)
        if len(records) < 2:
            print("  fewer than two chains; nothing to link")
        else:
            passages = {e["episode_id"]: e["text"] for e in episodes}
            trace["stage4_link"] = link(
                records, passages, model=args.model, review=not args.no_review,
                thresholds=PR.Thresholds(low=args.split_at, high=args.merge_at))

    args.out.write_text(json.dumps(trace, indent=2, ensure_ascii=False,
                                   default=str))
    print(f"\ntrace -> {args.out}  ({args.out.stat().st_size // 1024} KB, "
          f"{time.perf_counter() - started:.1f}s)")
    for entity in trace.get("stage4_link", {}).get("entities", {}).values():
        print(f"  {entity['surfaces']}")


if __name__ == "__main__":
    main()
