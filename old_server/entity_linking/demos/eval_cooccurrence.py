"""Co-occurrence edges: how dense is the graph, and does walking it help?

    uv run python entity_linking/demos/eval_cooccurrence.py --root <ecb+>

Two entities get an edge when a chain of each sits in the same episode. That is
free — it falls out of ``chains`` with a self-join — but free is not the same as
useful, and the failure mode is specific: an episode with c chains produces
c(c-1)/2 edges, so a long passage connects everything to everything. PPR over a
near-complete graph ranks by degree rather than by relevance.

So this measures two things:

  1. density — degree distribution, and what edge thresholds do to it
  2. whether running PPR from the surface seeds *helps or hurts* finding the
     right entity, against not walking at all

Point 2 is the only thing ECB+ can score, and it is a floor rather than the real
question: PPR exists to reach *related* entities, which needs queries with gold
multi-entity answers that this corpus does not provide. If the walk degrades even
the seed ranking, that settles it early and cheaply.

Extraction is cached to --cache so repeat runs cost nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_ecbplus as ECB  # noqa: E402
import eval_seeding as SEED  # noqa: E402
import eval_seeding_pipeline as PIPE  # noqa: E402

from entity_linking import chains as CH  # noqa: E402


def load_chains(args) -> list[dict]:
    """Predicted chains with gold entity labels, cached across runs."""
    if args.cache.exists():
        chains = json.loads(args.cache.read_text())
        print(f"{len(chains)} chains from cache {args.cache}")
        return chains

    mentions, passages = ECB.read_corpus(args.root)
    keep = sorted({m.topic for m in mentions}, key=int)[:args.topics]
    mentions = [m for m in mentions if m.topic in keep]
    passages = {m.doc_id: passages[m.doc_id] for m in mentions if m.doc_id in passages}

    gold_by_doc: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    topic_of = {}
    for mention in mentions:
        gold_by_doc[mention.doc_id][mention.cluster].add(mention.text)
        topic_of[mention.doc_id] = mention.topic

    print(f"extracting {len(passages)} documents …")
    predicted = PIPE.extract(passages, model=args.model, workers=args.workers)
    chains = PIPE.align(predicted, gold_by_doc)
    for chain in chains:
        chain["topic"] = topic_of.get(chain["doc"], "?")
    args.cache.write_text(json.dumps(chains))
    print(f"cached -> {args.cache}")
    return chains


def build_graph(chains: list[dict], min_episodes: int = 1):
    """Entity co-occurrence: a chain of each in one episode. Weight = episodes."""
    by_doc: dict[str, set[str]] = defaultdict(set)
    for chain in chains:
        by_doc[chain["doc"]].add(chain["entity"])

    weights: Counter = Counter()
    for entities in by_doc.values():
        ordered = sorted(entities)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                weights[(a, b)] += 1
    return {pair: w for pair, w in weights.items() if w >= min_episodes}


def describe(edges: dict, entities: list[str], label: str) -> None:
    degree: Counter = Counter()
    for (a, b), w in edges.items():
        degree[a] += 1
        degree[b] += 1
    n = len(entities)
    possible = n * (n - 1) / 2
    degrees = sorted(degree.values())
    print(f"  {label:<22} {len(edges):>7} edges  "
          f"density {len(edges) / possible:6.4f}  "
          f"mean deg {np.mean(degrees) if degrees else 0:6.1f}  "
          f"median {degrees[len(degrees) // 2] if degrees else 0:>4}  "
          f"max {max(degrees) if degrees else 0:>4}")


def ppr(edges: dict, index: dict[str, int], seeds: dict[str, float],
        alpha: float = 0.15, iterations: int = 30) -> np.ndarray:
    """Power iteration with restart. Row-normalised, weights as given."""
    n = len(index)
    out = np.zeros(n)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b), w in edges.items():
        i, j = index[a], index[b]
        neighbours[i].append((j, w))
        neighbours[j].append((i, w))
        out[i] += w
        out[j] += w

    restart = np.zeros(n)
    for entity, weight in seeds.items():
        if entity in index:
            restart[index[entity]] = weight
    total = restart.sum()
    if total == 0:
        return restart
    restart /= total

    score = restart.copy()
    for _ in range(iterations):
        nxt = np.zeros(n)
        for i, row in enumerate(neighbours):
            if not row or score[i] == 0:
                continue
            share = score[i] / out[i]
            for j, w in row:
                nxt[j] += share * w
        score = alpha * restart + (1 - alpha) * nxt
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=Path,
                        default=Path("/tmp/ecb_chains_cache.json"))
    args = parser.parse_args()

    chains = load_chains(args)
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        by_entity[chain["entity"]].append(chain)

    held_out, indexed = [], []
    for group in by_entity.values():
        if len(group) < 2 or group[0]["is_junk"]:
            indexed.extend(group)
            continue
        held_out.append(group[0])
        indexed.extend(group[1:])

    index = SEED.Index(indexed)
    entities = sorted({c["entity"] for c in indexed})
    position = {e: i for i, e in enumerate(entities)}
    print(f"{len(indexed)} indexed chains, {len(entities)} entities, "
          f"{len(held_out)} queries\n")

    print("graph density:")
    graphs = {f"min_episodes={t}": build_graph(indexed, t) for t in (1, 2, 3)}
    for label, edges in graphs.items():
        describe(edges, entities, label)

    queries = {i: " ".join(c["surfaces"]) for i, c in enumerate(held_out)}
    texts = sorted({t for t in queries.values() if t})
    vectors = SEED.embed(index.surfaces + texts)
    matrix = np.vstack([vectors[s] for s in index.surfaces])

    print(f"\nranking (seeds = top-{args.seeds} by sqrt-normalised surface score):")
    print(f"{'method':<34}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    edges = graphs["min_episodes=1"]

    rows: list[tuple[str, list]] = [("seeds only, no walk", [])]
    for alpha in (0.15, 0.5, 0.85):
        rows.append((f"PPR alpha={alpha} (min_ep=1)", []))
    rows.append(("PPR alpha=0.5 (min_ep=2)", []))

    scores = {name: [0, 0, 0.0] for name, _ in rows}
    asked = 0
    for i, chain in enumerate(held_out):
        text = queries.get(i)
        if not text:
            continue
        asked += 1
        sims = matrix @ vectors[text]
        top = np.argsort(-sims)[:args.top_k]
        hits = [(index.surfaces[j], float(sims[j])) for j in top]
        ranked = SEED.rank_entities(index, hits, "+ sqrt only")
        seeds = {e: 1.0 / (r + 1) for r, e in enumerate(ranked[:args.seeds])}

        candidates = {"seeds only, no walk": ranked}
        for alpha in (0.15, 0.5, 0.85):
            vector = ppr(edges, position, seeds, alpha=alpha)
            candidates[f"PPR alpha={alpha} (min_ep=1)"] = [
                entities[j] for j in np.argsort(-vector) if vector[j] > 0]
        vector = ppr(graphs["min_episodes=2"], position, seeds, alpha=0.5)
        candidates["PPR alpha=0.5 (min_ep=2)"] = [
            entities[j] for j in np.argsort(-vector) if vector[j] > 0]

        for name, order in candidates.items():
            gold = chain["entity"]
            scores[name][0] += int(bool(order) and order[0] == gold)
            scores[name][1] += int(gold in order[:5])
            scores[name][2] += (1.0 / (order.index(gold) + 1)
                                if gold in order else 0.0)

    for name, _ in rows:
        at1, at5, rr = scores[name]
        print(f"{name:<34}{at1 / asked:>8.3f}{at5 / asked:>8.3f}{rr / asked:>8.3f}")


if __name__ == "__main__":
    main()
