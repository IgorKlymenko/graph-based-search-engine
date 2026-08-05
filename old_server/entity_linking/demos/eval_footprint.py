"""Footprint recall: "everything we know about X", not "the passage answering X".

    uv run python entity_linking/demos/eval_footprint.py --root <ecb+>

``eval_retrieval.py`` asked each system to find the one passage a question was
written from. Dense won, and it should — the query paraphrases the passage.

That is not what this pipeline is for. The goal is open-ended questions where no
single passage is the answer and the user cannot name what they are missing. The
graph's claim there is *aggregation*: an entity is a join point over every
passage mentioning it, including ones sharing no vocabulary with the query.

So: query with an entity's name, and score against **every** document ECB+ says
mentions that entity. Gold comes from ECB+'s own coreference annotation, not from
our extraction, so extraction and linking failures count against us.

The sharp case is reported separately — documents where the entity appears
**only** under a surface different from the queried name. That is precisely where
vocabulary matching has nothing to match and the graph should be perfect.

Linking is real, not gold: chains are clustered by ``4_link.py`` exactly as in
production, so the numbers include whatever the linker gets wrong.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_cooccurrence as CO  # noqa: E402
import eval_ecbplus as ECB  # noqa: E402
import eval_expansion as EX  # noqa: E402
import eval_seeding as SEED  # noqa: E402

from entity_linking import chains as CH  # noqa: E402
from entity_linking import pairing as PR  # noqa: E402
from entity_linking import pipeline as PL  # noqa: E402


def link(chains: list[dict]) -> dict[int, str]:
    """Run the real linker over the predicted chains; returns index -> entity id."""
    heads = sorted({max(c["surfaces"], key=len) for c in chains})
    vectors = PL._name_vectors(heads)
    records = [
        PR.ChainRecord(chain_id=str(i), episode_id=c["doc"],
                       head=max(c["surfaces"], key=len),
                       surfaces=frozenset(c["surfaces"]),
                       candidate_type="proper_name",
                       name_vector=vectors.get(max(c["surfaces"], key=len)))
        for i, c in enumerate(chains)
    ]
    evidence = PR.build_evidence(records)
    scorer = PR.PairScorer()
    thresholds = PR.Thresholds(low=0.30, high=0.50)
    merges = []
    for a, b in itertools.combinations(records, 2):
        if a.episode_id == b.episode_id:
            continue
        probability = scorer.score(PR.features(a, b, evidence))
        if PR.decide(probability, thresholds) == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))
    assignment = PR.cluster([r.chain_id for r in records], merges)
    stable = PR.stable_entity_ids(assignment)
    print(f"  linked {len(records)} chains -> {len(set(stable.values()))} entities "
          f"({len(merges)} merges)")
    return {int(cid): entity for cid, entity in stable.items()}


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

    mentions, passages = ECB.read_corpus(args.root)
    keep = sorted({m.topic for m in mentions}, key=int)[:args.topics]
    mentions = [m for m in mentions if m.topic in keep]
    passages = {m.doc_id: passages[m.doc_id] for m in mentions if m.doc_id in passages}

    # gold: which documents mention each entity, and under which surfaces
    gold_docs: dict[str, set[str]] = defaultdict(set)
    gold_surfaces: dict[tuple[str, str], set[str]] = defaultdict(set)
    names: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        gold_docs[mention.cluster].add(mention.doc_id)
        gold_surfaces[(mention.cluster, mention.doc_id)].add(mention.text)
        names[mention.cluster].add(mention.text)

    chains = CO.load_chains(args)
    chains = [c for c in chains if c["doc"] in passages]
    predicted = link(chains)
    linked = [{"doc": c["doc"], "entity": predicted[i], "surfaces": c["surfaces"]}
              for i, c in enumerate(chains)]

    index = SEED.Index(linked)
    docs_of_entity: dict[str, set[str]] = defaultdict(set)
    for chain in linked:
        docs_of_entity[chain["entity"]].add(chain["doc"])

    queries = []
    for cluster, docs in gold_docs.items():
        if len(docs) < 2:
            continue
        name = max(names[cluster], key=len)  # what a user would type
        alias_only = {d for d in docs
                      if name not in gold_surfaces[(cluster, d)]}
        queries.append({"name": name, "gold": docs, "alias_only": alias_only})
    print(f"{len(queries)} entities with >=2 documents, "
          f"{sum(len(q['gold']) for q in queries) / len(queries):.1f} documents each")
    print(f"{sum(len(q['alias_only']) for q in queries)} of "
          f"{sum(len(q['gold']) for q in queries)} gold documents are alias-only\n")

    episode_ids = sorted(passages)
    texts = sorted({*(passages[d] for d in episode_ids), *index.surfaces,
                    *(q["name"] for q in queries)})
    vectors = SEED.embed(texts)
    episode_matrix = np.vstack([vectors[passages[d]] for d in episode_ids])
    surface_matrix = np.vstack([vectors[s] for s in index.surfaces])
    edges = CO.build_graph(linked)                        # raw counts
    weighted = EX.weighted_edges(linked, "pmi x inverse_size")
    entities = sorted({c["entity"] for c in linked})
    position = {e: i for i, e in enumerate(entities)}

    def dense(name: str) -> list[str]:
        sims = episode_matrix @ vectors[name]
        return [episode_ids[j] for j in np.argsort(-sims)]

    def seeds_of(name: str) -> dict[str, float]:
        sims = surface_matrix @ vectors[name]
        top = np.argsort(-sims)[:args.top_k]
        hits = [(index.surfaces[j], float(sims[j])) for j in top]
        ordered = SEED.rank_entities(index, hits, "+ sqrt only")
        return {e: 1.0 / (r + 1) for r, e in enumerate(ordered[:args.seeds])}

    def by_entity(scores: dict[str, float]) -> list[str]:
        best: dict[str, float] = defaultdict(float)
        for entity, weight in scores.items():
            for doc in docs_of_entity.get(entity, ()):
                best[doc] = max(best[doc], weight)
        return [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]

    systems = {
        "dense over passages": dense,
        "graph (seeds)": lambda n: by_entity(seeds_of(n)),
        "graph + PPR raw, a=0.5": lambda n: by_entity(dict(zip(
            entities, CO.ppr(edges, position, seeds_of(n), alpha=0.5)))),
        "graph + PPR pmi, a=0.2": lambda n: by_entity(dict(zip(
            entities, CO.ppr(weighted, position, seeds_of(n), alpha=0.2)))),
    }

    for subset in ("all gold documents", "alias-only documents"):
        key = "gold" if subset.startswith("all") else "alias_only"
        items = [q for q in queries if q[key]]
        print(f"=== {subset} ({len(items)} entities) ===")
        print(f"{'system':<26}{'R@5':>8}{'R@10':>8}{'R@20':>8}{'recall':>9}")
        for label, run in systems.items():
            totals = defaultdict(float)
            for item in items:
                ranked = run(item["name"])
                gold = item[key]
                for k in (5, 10, 20):
                    totals[k] += len(gold & set(ranked[:k])) / len(gold)
                totals["all"] += len(gold & set(ranked)) / len(gold)
            n = len(items)
            print(f"{label:<26}{totals[5] / n:>8.3f}{totals[10] / n:>8.3f}"
                  f"{totals[20] / n:>8.3f}{totals['all'] / n:>9.3f}")
        print()


if __name__ == "__main__":
    main()
