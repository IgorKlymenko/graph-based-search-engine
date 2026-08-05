"""End-to-end retrieval: does the entity graph beat searching the passages directly?

    uv run python entity_linking/demos/eval_retrieval.py --root <ecb+>

Everything measured so far scores the *linker* — whether chains land in the right
entity. Nothing has scored the thing the linker exists for: given a question,
does this pipeline find the passage that answers it.

Three systems, same queries, same corpus:

  dense       embed the episodes, cosine against the query. No entities, no
              chains, no graph. This is the baseline the graph has to beat, and
              if it doesn't, the graph is not earning its cost.
  seeds       query -> surfaces -> entities -> those entities' episodes
  seeds+PPR   the same, then walk the co-occurrence graph before collecting
              episodes

Queries are generated from the passages themselves: one question per document
that the document answers, gold = that document. A second set asks about a whole
ECB+ topic, where gold is every document in it — that is the multi-document case
PPR expansion is supposed to serve.

Both the extraction and the generated queries are cached, so only the first run
costs anything.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_cooccurrence as CO  # noqa: E402
import eval_ecbplus as ECB  # noqa: E402
import eval_seeding as SEED  # noqa: E402

from entity_linking import chains as CH  # noqa: E402

QUESTION_PROMPT = """You write retrieval test questions.

Given a passage, write ONE question that this passage answers and that someone
who had not read it might plausibly ask. Use the specific names, organisations
and events the passage mentions — a question with no anchor in it is not a
retrieval test, it is a guessing game.

Do not include the answer. Do not refer to "the passage" or "this article".
Return the question only."""

QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

TOPIC_PROMPT = """You write retrieval test questions.

Given several passages that all report the SAME event, write ONE question about
that event which no single passage fully answers — it should need details that
are spread across more than one of them. Use the specific names and places.

Do not include the answer. Return the question only."""


def generate(passages: dict[str, str], topics: dict[str, list[str]], *,
             model: str, workers: int, per_doc: int, cache: Path) -> dict:
    if cache.exists():
        queries = json.loads(cache.read_text())
        print(f"{len(queries['doc'])} document + {len(queries['topic'])} topic "
              f"queries from cache")
        return queries

    rng = random.Random(11)
    chosen = rng.sample(sorted(passages), min(per_doc, len(passages)))

    def ask(prompt: str, text: str) -> str:
        try:
            return CH._complete_json(system=prompt, user=text, model=model,
                                     schema=QUESTION_SCHEMA,
                                     schema_name="question")["question"]
        except Exception:  # noqa: BLE001
            return ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        doc_qs = list(pool.map(lambda d: ask(QUESTION_PROMPT, passages[d]), chosen))
        topic_texts = {
            t: "\n\n---\n\n".join(passages[d][:1200] for d in docs[:4])
            for t, docs in topics.items()
        }
        topic_qs = dict(zip(
            topic_texts,
            pool.map(lambda t: ask(TOPIC_PROMPT, topic_texts[t]), topic_texts)))

    queries = {
        "doc": [{"question": q, "gold": [d]}
                for d, q in zip(chosen, doc_qs) if q],
        "topic": [{"question": q, "gold": topics[t]}
                  for t, q in topic_qs.items() if q],
    }
    cache.write_text(json.dumps(queries, indent=1))
    print(f"generated {len(queries['doc'])} document + {len(queries['topic'])} "
          f"topic queries -> {cache}")
    return queries


def score(ranked: list[str], gold: list[str], ks=(1, 5, 10)) -> dict:
    """Recall@k over the gold documents, plus reciprocal rank of the first hit."""
    goldset = set(gold)
    out = {f"R@{k}": len(goldset & set(ranked[:k])) / len(goldset) for k in ks}
    first = next((i for i, d in enumerate(ranked) if d in goldset), None)
    out["MRR"] = 1.0 / (first + 1) if first is not None else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--per-doc", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chain-cache", type=Path,
                        default=Path("/tmp/ecb_chains_cache.json"))
    parser.add_argument("--query-cache", type=Path,
                        default=Path("/tmp/ecb_queries.json"))
    args = parser.parse_args()

    mentions, passages = ECB.read_corpus(args.root)
    keep = sorted({m.topic for m in mentions}, key=int)[:args.topics]
    mentions = [m for m in mentions if m.topic in keep]
    passages = {m.doc_id: passages[m.doc_id] for m in mentions if m.doc_id in passages}
    topic_docs: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        if mention.doc_id not in topic_docs[mention.topic]:
            topic_docs[mention.topic].append(mention.doc_id)

    args.cache = args.chain_cache  # CO.load_chains reads args.cache
    chains = CO.load_chains(args)
    chains = [c for c in chains if c["doc"] in passages]
    queries = generate(passages, dict(topic_docs), model=args.model,
                       workers=args.workers, per_doc=args.per_doc,
                       cache=args.query_cache)

    index = SEED.Index(chains)
    entities = sorted({c["entity"] for c in chains})
    position = {e: i for i, e in enumerate(entities)}
    docs_of_entity: dict[str, set[str]] = defaultdict(set)
    for chain in chains:
        docs_of_entity[chain["entity"]].add(chain["doc"])
    edges = CO.build_graph(chains)
    print(f"{len(chains)} chains, {len(entities)} entities, {len(edges)} edges, "
          f"{len(passages)} passages\n")

    episode_ids = sorted(passages)
    all_texts = ([passages[d] for d in episode_ids] + index.surfaces
                 + [q["question"] for group in queries.values() for q in group])
    vectors = SEED.embed(sorted(set(all_texts)))
    episode_matrix = np.vstack([vectors[passages[d]] for d in episode_ids])
    surface_matrix = np.vstack([vectors[s] for s in index.surfaces])

    def rank_dense(question: str) -> list[str]:
        sims = episode_matrix @ vectors[question]
        return [episode_ids[j] for j in np.argsort(-sims)]

    def entity_scores(question: str) -> dict[str, float]:
        sims = surface_matrix @ vectors[question]
        top = np.argsort(-sims)[:args.top_k]
        hits = [(index.surfaces[j], float(sims[j])) for j in top]
        ordered = SEED.rank_entities(index, hits, "+ sqrt only")
        return {e: 1.0 / (r + 1) for r, e in enumerate(ordered[:args.seeds])}

    def docs_from(scores: dict[str, float]) -> list[str]:
        """Rank passages by the best-scoring entity that appears in them."""
        best: dict[str, float] = defaultdict(float)
        for entity, weight in scores.items():
            for doc in docs_of_entity.get(entity, ()):
                best[doc] = max(best[doc], weight)
        return [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]

    def rerank(docs: list[str], question: str, pool: int = 50) -> list[str]:
        """Order graph-retrieved passages by the query, as step 8 of the design.

        Without this the graph path ranks by a step function over the top-5 seed
        scores, so every passage containing the best entity ties and the order
        inside the tie is arbitrary. That measures the blocker, not the system.
        """
        head = docs[:pool]
        if not head:
            return docs
        sims = np.vstack([vectors[passages[d]] for d in head]) @ vectors[question]
        return [head[j] for j in np.argsort(-sims)] + docs[pool:]

    def ppr_docs(question: str) -> list[str]:
        return docs_from(dict(zip(
            entities, CO.ppr(edges, position, entity_scores(question),
                             alpha=0.5))))

    systems = {
        "dense over passages (baseline)": rank_dense,
        "seeds only": lambda q: docs_from(entity_scores(q)),
        "seeds + PPR": ppr_docs,
        "seeds + PPR, reranked": lambda q: rerank(ppr_docs(q), q),
        "dense, reranked (sanity)": lambda q: rerank(rank_dense(q), q),
    }

    for group, items in queries.items():
        if not items:
            continue
        print(f"=== {group} queries ({len(items)}) ===")
        print(f"{'system':<34}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
              f"{'cand.rec':>10}")
        for name, run in systems.items():
            totals = defaultdict(float)
            for item in items:
                ranked = run(item["question"])
                for metric, value in score(ranked, item["gold"]).items():
                    totals[metric] += value
                # what fraction of gold the candidate set contains at all --
                # the graph's actual job, before any ranking
                gold = set(item["gold"])
                totals["cand"] += len(gold & set(ranked)) / len(gold)
            n = len(items)
            print(f"{name:<34}{totals['R@1'] / n:>8.3f}{totals['R@5'] / n:>8.3f}"
                  f"{totals['R@10'] / n:>8.3f}{totals['MRR'] / n:>8.3f}"
                  f"{totals['cand'] / n:>10.3f}")
        print()


if __name__ == "__main__":
    main()
