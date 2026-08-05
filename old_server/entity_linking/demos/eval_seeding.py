"""Does IDF weighting actually help surface -> entity seeding? Measure before storing.

    uv run python entity_linking/demos/eval_seeding.py --root /tmp/ecbPlus-master

The retrieval design is: query -> top-k surfaces by cosine -> entities that own
those surfaces -> PPR seeds. ``surfaces.n_entities`` and ``surfaces.n_episodes``
exist to weight that middle step. This checks whether they earn their place.

Protocol: ECB+ gold clusters are the entities. For every entity with at least two
chains, one chain is held out as a query and the rest form the index — so the
target entity is always reachable, and the question is purely whether the ranking
finds it.

Two query modes, because they stress different things:
  surface  the held-out chain's own names — short, near-exact, the easy case
  passage  the held-out document's text — long, the asymmetric case where a
           paragraph is compared against two-word names

Embeddings are text-embedding-3-large at 1536 dims, matching the schema.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_ecbplus as ECB  # noqa: E402

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536


def embed(texts: list[str], batch: int = 256) -> dict[str, np.ndarray]:
    """One batched pass; returns L2-normalised vectors keyed by text."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        response = client.embeddings.create(
            model=EMBED_MODEL, input=chunk, dimensions=EMBED_DIM)
        for text, item in zip(texts[i:i + batch], response.data):
            vector = np.asarray(item.embedding, dtype=np.float32)
            out[text] = vector / np.linalg.norm(vector)
    return out


class Index:
    """The surfaces table and its bridge, built from the indexed chains only."""

    def __init__(self, chains: list[dict]) -> None:
        self.entities_of: dict[str, set[str]] = defaultdict(set)
        self.episodes_of: dict[str, set[str]] = defaultdict(set)
        self.surfaces_of_entity: dict[str, set[str]] = defaultdict(set)
        # (entity, surfaces in that chain) per surface -- only used to reproduce
        # what the fan-out SQL scores, never by the intended weighting
        self.chains_of: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for chain in chains:
            for surface in chain["surfaces"]:
                self.entities_of[surface].add(chain["entity"])
                self.episodes_of[surface].add(chain["doc"])
                self.surfaces_of_entity[chain["entity"]].add(surface)
                self.chains_of[surface].append(
                    (chain["entity"], len(chain["surfaces"])))
        self.surfaces = sorted(self.entities_of)
        self.n_episodes_total = len({c["doc"] for c in chains})

    def idf(self, surface: str) -> float:
        """Classic IDF over episodes: a surface in every passage says nothing."""
        df = len(self.episodes_of[surface])
        return math.log((1 + self.n_episodes_total) / (1 + df))

    def ambiguity(self, surface: str) -> float:
        """1/n_entities — a surface claimed by three entities identifies none."""
        return 1.0 / len(self.entities_of[surface])


VARIANTS = (
    "cosine only",
    "+ idf(n_episodes)",
    "+ ambiguity(1/n_entities)",
    "+ both",
    "+ sqrt only",
    "+ both, sqrt-normalised",
    "sqrt via fan-out SQL (bug)",
)


def rank_entities(index: Index, hits: list[tuple[str, float]],
                  variant: str) -> list[str]:
    """Score entities by the surfaces that matched, under one weighting scheme."""
    scores: dict[str, float] = defaultdict(float)
    if variant == "sqrt via fan-out SQL (bug)":
        # the DATAFLOW query joined chain_surfaces a second time to count
        # breadth, which duplicates every hit row once per surface of the chain:
        # score * n / sqrt(n) = score * sqrt(n), amplifying breadth instead of
        # damping it
        for surface, cosine in hits:
            for entity, chain_size in index.chains_of[surface]:
                scores[entity] += cosine * chain_size
        return [e for e, _ in sorted(
            ((e, s / math.sqrt(len(index.surfaces_of_entity[e])))
             for e, s in scores.items()), key=lambda kv: -kv[1])]
    for surface, cosine in hits:
        weight = cosine
        if variant in ("+ idf(n_episodes)", "+ both", "+ both, sqrt-normalised"):
            weight *= index.idf(surface)
        if variant in ("+ ambiguity(1/n_entities)", "+ both",
                       "+ both, sqrt-normalised"):
            weight *= index.ambiguity(surface)
        for entity in index.entities_of[surface]:
            scores[entity] += weight
    if variant in ("+ sqrt only", "+ both, sqrt-normalised"):
        # breadth should help, not dominate: an entity with 40 surfaces would
        # otherwise outrank the right answer on sheer surface count
        scores = {e: s / math.sqrt(len(index.surfaces_of_entity[e]))
                  for e, s in scores.items()}
    return [e for e, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def build(root: Path, topics: int) -> tuple[list[dict], dict[str, str]]:
    mentions, passages = ECB.read_corpus(root)
    keep = sorted({m.topic for m in mentions}, key=int)[:topics]
    mentions = [m for m in mentions if m.topic in keep]

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for mention in mentions:
        grouped[(mention.doc_id, mention.cluster)].append(mention)

    chains = [{"doc": doc, "entity": cluster,
               "surfaces": sorted({m.text for m in group})}
              for (doc, cluster), group in grouped.items()]
    return chains, passages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    chains, passages = build(args.root, args.topics)
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        by_entity[chain["entity"]].append(chain)

    held_out, indexed = [], []
    for entity, group in by_entity.items():
        if len(group) < 2:
            indexed.extend(group)  # nothing to hold out from a singleton
            continue
        held_out.append(group[0])
        indexed.extend(group[1:])

    index = Index(indexed)
    print(f"{len(chains)} chains, {len(by_entity)} entities, "
          f"{len(indexed)} indexed, {len(held_out)} queries")
    print(f"{len(index.surfaces)} distinct surfaces, "
          f"{sum(len(v) for v in index.entities_of.values()) / len(index.surfaces):.2f}"
          f" entities per surface on average")
    contested = sum(1 for s in index.surfaces if len(index.entities_of[s]) > 1)
    print(f"{contested} surfaces ({contested / len(index.surfaces):.1%}) "
          f"claimed by more than one entity")
    df = sorted(len(index.episodes_of[s]) for s in index.surfaces)
    import collections
    hist = collections.Counter(df)
    print("surface document frequency (what IDF has to work with):")
    for k in sorted(hist)[:6]:
        print(f"    in {k:>2} episode(s): {hist[k]:>4} surfaces "
              f"({hist[k] / len(df):.1%})")
    print(f"    max df {max(df)}, median {df[len(df) // 2]}\n")

    queries = {
        "surface": {i: " ".join(c["surfaces"]) for i, c in enumerate(held_out)},
        "passage": {i: passages.get(c["doc"], "") for i, c in enumerate(held_out)},
    }
    texts = sorted({t for mode in queries.values() for t in mode.values() if t})
    vectors = embed(index.surfaces + texts)
    matrix = np.vstack([vectors[s] for s in index.surfaces])

    for mode, texts_by_query in queries.items():
        print(f"=== query = {mode} ===")
        results = {v: [0, 0, 0.0] for v in VARIANTS}  # hits@1, hits@5, rr
        asked = 0
        for i, chain in enumerate(held_out):
            text = texts_by_query.get(i)
            if not text:
                continue
            asked += 1
            sims = matrix @ vectors[text]
            top = np.argsort(-sims)[:args.top_k]
            hits = [(index.surfaces[j], float(sims[j])) for j in top]
            for variant in VARIANTS:
                ranked = rank_entities(index, hits, variant)
                gold = chain["entity"]
                results[variant][0] += int(bool(ranked) and ranked[0] == gold)
                results[variant][1] += int(gold in ranked[:5])
                results[variant][2] += (1.0 / (ranked.index(gold) + 1)
                                        if gold in ranked else 0.0)
        print(f"{'variant':<28}{'R@1':>8}{'R@5':>8}{'MRR':>8}")
        for variant in VARIANTS:
            at1, at5, rr = results[variant]
            print(f"{variant:<28}{at1 / asked:>8.3f}{at5 / asked:>8.3f}"
                  f"{rr / asked:>8.3f}")
        print()


if __name__ == "__main__":
    main()
