"""IDF again, but over surfaces this pipeline actually produces.

    uv run python entity_linking/demos/eval_seeding_pipeline.py --root <ecb+>

``eval_seeding.py`` found IDF worth +0.002 MRR — but it indexed ECB+'s *gold*
mentions, a vocabulary with no junk in it. Our extractor emits "the freeze",
"the change", "the vendor": high-frequency generic surfaces attached to
singleton entities nobody would ever want seeded. Demoting exactly those is what
IDF is for, so the clean-vocabulary null result proves nothing about our data.

Here stage 2 runs for real over ECB+ passages. Predicted chains are aligned to
gold clusters by surface overlap; chains that align to nothing are kept in the
index as distractor entities, because they are the pollution under test.

Queries are held-out chains of aligned entities, so the target is always
reachable and only the ranking is being measured.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_ecbplus as ECB  # noqa: E402
import eval_seeding as SEED  # noqa: E402

from entity_linking import chains as CH  # noqa: E402
from entity_linking import spans as SP  # noqa: E402


def extract(passages: dict[str, str], *, model: str, workers: int) -> dict:
    """Stage 2 over every document; returns doc -> list of surface sets."""
    docs = sorted(passages)

    def one(doc: str):
        try:
            resolved = CH.resolve_chains(
                CH.extract_chains(passages[doc], model=model))
        except Exception:  # noqa: BLE001 — one document, not the run
            return []
        out = []
        for chain in resolved:
            surfaces = sorted({s.text for s in chain
                               if s.candidate_type != SP.PRONOUN})
            if surfaces:
                out.append(surfaces)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(docs, pool.map(one, docs)))


def align(predicted: dict, gold_by_doc: dict) -> list[dict]:
    """Give each predicted chain the gold cluster it overlaps most, or None.

    Overlap is on surface strings within the same document. A chain matching no
    gold mention is junk — kept, with a synthetic entity of its own, so it can
    pollute the surface table exactly as it would in production.
    """
    chains, junk = [], 0
    for doc, groups in predicted.items():
        gold = gold_by_doc.get(doc, {})
        for i, surfaces in enumerate(groups):
            overlaps = {cluster: len(set(surfaces) & texts)
                        for cluster, texts in gold.items()}
            best = max(overlaps, key=overlaps.get) if overlaps else None
            if best is None or overlaps[best] == 0:
                best, junk = f"junk:{doc}:{i}", junk + 1
            chains.append({"doc": doc, "entity": best, "surfaces": surfaces,
                           "is_junk": best.startswith("junk:")})
    print(f"  {len(chains)} predicted chains, {junk} aligned to no gold mention "
          f"({junk / max(len(chains), 1):.1%} junk)")
    return chains


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    mentions, passages = ECB.read_corpus(args.root)
    keep = sorted({m.topic for m in mentions}, key=int)[:args.topics]
    mentions = [m for m in mentions if m.topic in keep]
    passages = {m.doc_id: passages[m.doc_id] for m in mentions
                if m.doc_id in passages}

    gold_by_doc: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    for mention in mentions:
        gold_by_doc[mention.doc_id][mention.cluster].add(mention.text)

    print(f"extracting {len(passages)} documents with {args.model} …")
    predicted = extract(passages, model=args.model, workers=args.workers)
    chains = align(predicted, gold_by_doc)

    by_entity: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        by_entity[chain["entity"]].append(chain)

    held_out, indexed = [], []
    for group in by_entity.values():
        if len(group) < 2 or group[0]["is_junk"]:
            indexed.extend(group)  # junk stays in the index as a distractor
            continue
        held_out.append(group[0])
        indexed.extend(group[1:])

    index = SEED.Index(indexed)
    junk_entities = sum(1 for e in index.surfaces_of_entity if e.startswith("junk:"))
    print(f"  {len(indexed)} indexed, {len(held_out)} queries, "
          f"{len(index.surfaces)} distinct surfaces, "
          f"{junk_entities} junk entities in the index")

    df = Counter(len(index.episodes_of[s]) for s in index.surfaces)
    total = len(index.surfaces)
    print("  surface document frequency:")
    for k in sorted(df)[:5]:
        print(f"    in {k:>2} episode(s): {df[k]:>4} ({df[k] / total:.1%})")
    print(f"    max df {max(df)}\n")

    queries = {i: " ".join(c["surfaces"]) for i, c in enumerate(held_out)}
    texts = sorted({t for t in queries.values() if t})
    vectors = SEED.embed(index.surfaces + texts)
    matrix = np.vstack([vectors[s] for s in index.surfaces])

    results = {v: [0, 0, 0.0] for v in SEED.VARIANTS}
    junk_at_1 = {v: 0 for v in SEED.VARIANTS}
    asked = 0
    for i, chain in enumerate(held_out):
        text = queries.get(i)
        if not text:
            continue
        asked += 1
        sims = matrix @ vectors[text]
        top = np.argsort(-sims)[:args.top_k]
        hits = [(index.surfaces[j], float(sims[j])) for j in top]
        for variant in SEED.VARIANTS:
            ranked = SEED.rank_entities(index, hits, variant)
            gold = chain["entity"]
            results[variant][0] += int(bool(ranked) and ranked[0] == gold)
            results[variant][1] += int(gold in ranked[:5])
            results[variant][2] += (1.0 / (ranked.index(gold) + 1)
                                    if gold in ranked else 0.0)
            if ranked and ranked[0].startswith("junk:"):
                junk_at_1[variant] += 1

    print(f"{'variant':<28}{'R@1':>8}{'R@5':>8}{'MRR':>8}{'junk@1':>9}")
    for variant in SEED.VARIANTS:
        at1, at5, rr = results[variant]
        print(f"{variant:<28}{at1 / asked:>8.3f}{at5 / asked:>8.3f}"
              f"{rr / asked:>8.3f}{junk_at_1[variant] / asked:>9.3f}")


if __name__ == "__main__":
    main()
