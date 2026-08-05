"""Does the contextual chain vector help where the name vector cannot?

    uv run python entity_linking/demos/eval_context_vector.py
    uv run python entity_linking/demos/eval_context_vector.py --corpus large

`context_cosine` was dropped from the scorer after measuring AUC 0.505 over all
cross-episode pairs. That is a *marginal* statistic, dominated by pairs the name
vector already settles. This asks the conditional question instead: among pairs
whose surface strings are near-identical — where `name_cosine` is ~1.0 and
carries no information — does context separate same-entity from different-entity?

That subset is exactly where the linker is known to fail (`S. Chen` denoting two
people, `J. Park` denoting two more).

Chains are built from the gold surface map rather than from an LLM call, so
stage 2's errors are not a confound: this measures the vector, nothing else.
Every gold surface here is a proper name, which also controls for the
type-mixed-pooling confound (a pronoun pools span-only, a name pools span+8).

Both centerings are reported, since episode-centering removes the shared episode
topic and might be removing the cross-episode signal along with it.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from entity_linking import encoding as EN  # noqa: E402
from entity_linking import pipeline as PL  # noqa: E402
from entity_linking.spans import PROPER_NAME, Span  # noqa: E402

IDENTICAL = 0.999
NEAR_IDENTICAL = 0.90


class Chain:
    """One surface's occurrences inside one episode, with its gold entity."""

    __slots__ = ("episode_id", "surface", "entity", "spans", "topic",
                 "name_vector", "centered", "uncentered")

    def __init__(self, episode_id, surface, entity, spans, topic=None):
        self.episode_id = episode_id
        self.surface = surface
        self.entity = entity
        self.spans = spans
        self.topic = topic  # ECB+ only: pairs are drawn within a topic
        self.name_vector = self.centered = self.uncentered = None


def load_corpus(name: str) -> list[tuple[str, dict[str, str]]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_end_to_end as D

    return D.LARGE if name == "large" else D.build_xl()


def occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    """Every occurrence of ``surface``, so a chain pools all of its mentions."""
    out, start = [], text.find(surface)
    while start != -1:
        out.append((start, start + len(surface)))
        start = text.find(surface, start + 1)
    return out


def build_chains(corpus: list[tuple[str, dict[str, str]]]) -> list[Chain]:
    """One chain per (episode, surface). Vectors filled in by one pass each."""
    chains: list[Chain] = []
    for i, (text, mapping) in enumerate(corpus):
        episode_id = f"ep{i:03d}"
        found = []
        for surface, entity in mapping.items():
            spots = occurrences(text, surface)
            if spots:
                found.append(Chain(episode_id, surface, entity, [
                    Span(sid=f"c{j}", start=s, end=e, text=surface,
                         candidate_type=PROPER_NAME, origin="gold")
                    for j, (s, e) in enumerate(spots)
                ]))
        if not found:
            continue

        encoded = EN.encode_episode(text)
        spans = [chain.spans for chain in found]
        for attribute, center in (("centered", True), ("uncentered", False)):
            vectors = EN.encode_chains(text, spans, encoded=encoded,
                                       center=center)
            # encode_chains drops a chain whose mentions fell outside the token
            # window, so match on the head span rather than by position
            by_head = {(v.head.start, v.head.end): v.vector for v in vectors}
            for chain in found:
                head = (chain.spans[0].start, chain.spans[0].end)
                setattr(chain, attribute, by_head.get(head))
        chains.extend(found)

    names = PL._name_vectors(sorted({c.surface for c in chains}))
    for chain in chains:
        chain.name_vector = names[chain.surface]
    return [c for c in chains
            if c.centered is not None and c.uncentered is not None]


def build_chains_ecbplus(root: Path, topics: int) -> list[Chain]:
    """Same measurement on real text, where the hard subset is not synthetic.

    The synthetic corpora contain six same-string-different-entity pairs, which
    is far too few to measure anything. ECB+ has them by the hundred, in real
    contexts nobody wrote to make this experiment come out.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_ecbplus as ECB

    mentions, passages = ECB.read_corpus(root)
    keep = sorted({m.topic for m in mentions}, key=int)[:topics]
    mentions = [m for m in mentions if m.topic in keep]

    grouped: dict[tuple[str, str, str], list] = {}
    for mention in mentions:
        key = (mention.topic, mention.doc_id, mention.cluster)
        grouped.setdefault(key, []).append(mention)

    by_document: dict[str, list] = {}
    for (topic, doc_id, cluster), group in grouped.items():
        by_document.setdefault(doc_id, []).append((topic, cluster, group))

    chains: list[Chain] = []
    for doc_id, groups in by_document.items():
        text = passages[doc_id]
        found = []
        for topic, cluster, group in groups:
            head = max((m.text for m in group), key=len)
            spots = occurrences(text, head)
            if not spots:  # tokenised text does not always rejoin verbatim
                continue
            found.append(Chain(doc_id, head, cluster, [
                Span(sid=f"c{j}", start=s, end=e, text=head,
                     candidate_type=PROPER_NAME, origin="gold")
                for j, (s, e) in enumerate(spots)
            ], topic=topic))
        if not found:
            continue

        encoded = EN.encode_episode(text)
        spans = [c.spans for c in found]
        for attribute, center in (("centered", True), ("uncentered", False)):
            vectors = EN.encode_chains(text, spans, encoded=encoded,
                                       center=center)
            by_head = {(v.head.start, v.head.end): v.vector for v in vectors}
            for chain in found:
                key = (chain.spans[0].start, chain.spans[0].end)
                setattr(chain, attribute, by_head.get(key))
        chains.extend(found)

    names = PL._name_vectors(sorted({c.surface for c in chains}))
    for chain in chains:
        chain.name_vector = names[chain.surface]
    return [c for c in chains
            if c.centered is not None and c.uncentered is not None]


def auc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def build_pairs(chains: list[Chain]) -> list[dict]:
    rows = []
    for a, b in itertools.combinations(chains, 2):
        if a.episode_id == b.episode_id:
            continue
        if a.topic is not None and a.topic != b.topic:
            continue  # ECB+ annotates coreference only within a topic
        rows.append({
            "surfaces": (a.surface, b.surface),
            "label": int(a.entity == b.entity),
            "name_cosine": cosine(a.name_vector, b.name_vector),
            "string_similarity": difflib.SequenceMatcher(
                None, a.surface.lower(), b.surface.lower()).ratio(),
            "context_centered": cosine(a.centered, b.centered),
            "context_uncentered": cosine(a.uncentered, b.uncentered),
        })
    return rows


SIGNALS = ("name_cosine", "string_similarity",
           "context_centered", "context_uncentered")


def table(rows: list[dict], title: str) -> None:
    positives = sum(r["label"] for r in rows)
    print(f"\n{title}")
    print(f"  {len(rows)} pairs, {positives} same-entity "
          f"({positives / len(rows):.1%})" if rows else "  (empty)")
    if not rows or positives in (0, len(rows)):
        print("  only one class present — AUC undefined")
        return
    labels = [r["label"] for r in rows]
    for signal in SIGNALS:
        value = auc([r[signal] for r in rows], labels)
        mark = "  <-- chance" if abs(value - 0.5) < 0.03 else ""
        print(f"    {signal:<20} AUC {value:.3f}{mark}")


def combined(rows: list[dict], title: str) -> None:
    """Does adding context to name+string buy anything, cross-validated?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    labels = np.array([r["label"] for r in rows])
    if len(set(labels.tolist())) < 2 or min(np.bincount(labels)) < 5:
        print(f"\n{title}\n  too few of one class to cross-validate")
        return
    print(f"\n{title}")
    for name, features in (
        ("name + string", ("name_cosine", "string_similarity")),
        ("+ context (centered)", ("name_cosine", "string_similarity",
                                  "context_centered")),
        ("+ context (uncentered)", ("name_cosine", "string_similarity",
                                    "context_uncentered")),
        ("context alone", ("context_centered", "context_uncentered")),
    ):
        x = np.array([[r[f] for f in features] for r in rows])
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        scores = cross_val_score(model, x, labels, cv=5, scoring="roc_auc")
        print(f"    {name:<24} AUC {scores.mean():.3f} ± {scores.std():.3f}")


def veto(rows: list[dict]) -> None:
    """Is a very low context cosine a reliable 'different entity' signal?"""
    print("\nsplit-side veto: purity of the lowest-scoring decile")
    for signal in ("context_centered", "context_uncentered"):
        ordered = sorted(rows, key=lambda r: r[signal])
        decile = ordered[:max(1, len(ordered) // 10)]
        negatives = sum(1 for r in decile if not r["label"])
        base = sum(1 for r in rows if not r["label"]) / len(rows)
        print(f"    {signal:<20} {negatives / len(decile):.3f} "
              f"different-entity (base rate {base:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("large", "xl"), default="xl")
    parser.add_argument("--ecb", type=Path, default=None,
                        help="ECB+ root; real text instead of the synthetic corpus")
    parser.add_argument("--topics", type=int, default=10)
    args = parser.parse_args()

    if args.ecb:
        print(f"ECB+, first {args.topics} topics")
        chains = build_chains_ecbplus(args.ecb, args.topics)
    else:
        corpus = load_corpus(args.corpus)
        print(f"corpus: {len(corpus)} episodes")
        chains = build_chains(corpus)
    print(f"{len(chains)} chains, {len({c.entity for c in chains})} gold entities")

    rows = build_pairs(chains)
    table(rows, "ALL cross-episode pairs (what the 0.505 measured)")

    near = [r for r in rows
            if r["string_similarity"] >= NEAR_IDENTICAL]
    table(near, f"CONDITIONAL: string similarity >= {NEAR_IDENTICAL} "
                "(name vector uninformative)")

    same = [r for r in rows if r["surfaces"][0] == r["surfaces"][1]]
    table(same, "HARDEST: identical surface strings "
                "(the 'S. Chen is two people' case)")

    combined(near, f"combined scorer, string similarity >= {NEAR_IDENTICAL}")
    combined(same, "combined scorer, identical surfaces")
    veto(rows)


if __name__ == "__main__":
    main()
