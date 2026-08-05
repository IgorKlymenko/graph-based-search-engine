"""Fit and score the linking layer on ECB+, a real cross-document coref corpus.

    uv run python entity_linking/demos/eval_ecbplus.py --root /tmp/ecbPlus-master
    uv run python entity_linking/demos/eval_ecbplus.py --root … --live

Protocol: topics are split three ways, fit the scorer on **train**, pick the two
thresholds on **dev**, report on **test** only. Splits are disjoint by topic, so
no document — and no gold cluster, since ECB+ annotates coreference only within
a topic — is shared between them.

ECB+ supplies gold mention spans, so this exercises stages 3 and 4 only: identity
embedding, evidence features, pair scoring, review, clustering. The parser and
the coref call are bypassed, which is the point — it isolates the linking layer,
and it is the first corpus here that was not written alongside the algorithm.

Pairs are drawn within a topic and across documents, the standard ECB+ setting
with gold topic clustering. Cross-topic pairs are excluded: they are negative by
construction and would pad every precision figure with free correct splits.

Only *entity* markables count (ACTION_* and NEG_ACTION_* are events, a different
subtask), and only sentences ECB+ marks as validated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent))

from entity_linking import coref_metrics as M  # noqa: E402
from entity_linking import encoding as EN  # noqa: E402
from entity_linking import pairing as PR  # noqa: E402
from entity_linking import pipeline as PL  # noqa: E402

EVENT_PREFIXES = ("ACTION", "NEG_ACTION")
REVIEW_WORKERS = 8


class Mention:
    """One gold entity mention: its text, its document, its gold cluster."""

    __slots__ = ("doc_id", "topic", "text", "sentence", "cluster")

    def __init__(self, doc_id, topic, text, sentence, cluster):
        self.doc_id = doc_id
        self.topic = topic
        self.text = text
        self.sentence = sentence
        self.cluster = cluster


def validated_sentences(root: Path) -> set[tuple[str, str, str]]:
    """(topic, file, sentence) triples ECB+ marks as annotated."""
    path = root / "ECB+_LREC2014" / "ECBplus_coreference_sentences.csv"
    keep = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            keep.add((row["Topic"].strip(), row["File"].strip(),
                      row["Sentence Number"].strip()))
    return keep


def read_document(path: Path, keep: set) -> tuple[list[Mention], str]:
    """Entity mentions and validated-sentence text from one ECB+ document."""
    doc = ET.parse(path).getroot()
    doc_name = doc.get("doc_name", path.name)
    topic = path.parent.name
    stem = doc_name.replace(".xml", "").split("_", 1)[-1]

    tokens = {t.get("t_id"): (t.text or "", t.get("sentence"))
              for t in doc.findall("token")}
    live = {s for (tp, st, s) in keep if tp == topic and st == stem}

    sentences: dict[str, list[str]] = defaultdict(list)
    for word, sentence in tokens.values():
        if sentence in live:
            sentences[sentence].append(word)
    passage = " ".join(" ".join(w) for _, w in sorted(sentences.items(),
                                                      key=lambda kv: int(kv[0])))

    # markable id -> (surface, sentence), entity markables only
    spans: dict[str, tuple[str, str]] = {}
    markables = doc.find("Markables")
    if markables is None:
        return [], passage
    for markable in markables:
        if markable.tag.startswith(EVENT_PREFIXES):
            continue
        anchors = [a.get("t_id") for a in markable.findall("token_anchor")]
        if not anchors or anchors[0] not in tokens:
            continue
        words = [tokens[a][0] for a in anchors if a in tokens]
        if words:
            spans[markable.get("m_id")] = (" ".join(words), tokens[anchors[0]][1])

    # cross-document coreference: note= is the corpus-wide cluster id
    cluster_of: dict[str, str] = {}
    relations = doc.find("Relations")
    if relations is not None:
        for relation in relations:
            note = relation.get("note")
            if note:
                for source in relation.findall("source"):
                    cluster_of[source.get("m_id")] = note

    mentions = [
        Mention(doc_name, topic, text, sentence, cluster_of[m_id])
        for m_id, (text, sentence) in spans.items()
        if m_id in cluster_of and sentence in live
    ]
    return mentions, passage


def read_corpus(root: Path) -> tuple[list[Mention], dict[str, str]]:
    corpus = root / "ECB+_LREC2014" / "ECB+"
    keep = validated_sentences(root)
    mentions: list[Mention] = []
    passages: dict[str, str] = {}
    for topic in sorted(corpus.iterdir()):
        if not topic.is_dir():
            continue
        for path in sorted(topic.glob("*.xml")):
            found, passage = read_document(path, keep)
            mentions.extend(found)
            if found:
                passages[found[0].doc_id] = passage
    return mentions, passages


def name_vectors(strings: list[str]) -> dict[str, np.ndarray]:
    """One batched pass of the identity encoder over every distinct head."""
    import torch

    tokenizer, model, device = EN._load()
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(strings), 128):
        batch = strings[i:i + 128]
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


def context_vectors(groups: list[tuple[str, str, list]],
                    passages: dict[str, str]) -> dict[str, np.ndarray]:
    """One forward pass per document -> a contextual vector per chain.

    Feeds ``PR.context_gated``, which only fires on same-string pairs. Uncentered,
    matching the pipeline.
    """
    from entity_linking import encoding as EN
    from entity_linking.spans import PROPER_NAME, Span

    by_document: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for chain_id, doc_id, group in groups:
        by_document[doc_id].append((chain_id, max((m.text for m in group),
                                                  key=len)))

    out: dict[str, np.ndarray] = {}
    for doc_id, chains in by_document.items():
        text = passages.get(doc_id, "")
        spans, ids = [], []
        for chain_id, head in chains:
            spots = ECB_occurrences(text, head)
            if not spots:  # tokenised text does not always rejoin verbatim
                continue
            spans.append([Span(sid=f"c{j}", start=s, end=e, text=head,
                               candidate_type=PROPER_NAME, origin="gold")
                          for j, (s, e) in enumerate(spots)])
            ids.append(chain_id)
        if not spans:
            continue
        encoded = EN.encode_episode(text)
        vectors = EN.encode_chains(text, spans, encoded=encoded)
        by_head = {(v.head.start, v.head.end): v.vector for v in vectors}
        for chain_id, chain in zip(ids, spans):
            vector = by_head.get((chain[0].start, chain[0].end))
            if vector is not None:
                out[chain_id] = vector
    return out


def ECB_occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    out, start = [], text.find(surface)
    while start != -1:
        out.append((start, start + len(surface)))
        start = text.find(surface, start + 1)
    return out


def to_chain_records(mentions: list[Mention],
                     vectors: dict[str, np.ndarray],
                     passages: dict[str, str] | None = None
                     ) -> list[PR.ChainRecord]:
    """Same-document mentions sharing a gold cluster become one chain.

    That is what stage 2 would produce with perfect within-document coref, so
    only the cross-document layer is under test.
    """
    grouped: dict[tuple[str, str, str], list[Mention]] = defaultdict(list)
    for mention in mentions:
        grouped[(mention.topic, mention.doc_id, mention.cluster)].append(mention)

    triples = [(f"{topic}|{doc_id}|{cluster}", doc_id, group)
               for (topic, doc_id, cluster), group in grouped.items()]
    contexts = context_vectors(triples, passages) if passages else {}

    records = []
    for chain_id, doc_id, group in triples:
        head = max((m.text for m in group), key=len)
        records.append(PR.ChainRecord(
            chain_id=chain_id, episode_id=doc_id, head=head,
            surfaces=frozenset(m.text for m in group),
            candidate_type="proper_name",
            name_vector=vectors.get(head),
            context_vector=contexts.get(chain_id)))
    return records


def heads_of(mentions: list[Mention]) -> set[str]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mention in mentions:
        grouped[(mention.doc_id, mention.cluster)].append(mention.text)
    return {max(texts, key=len) for texts in grouped.values()}


def topic_of(chain_id: str) -> str:
    return chain_id.split("|", 1)[0]


def gold_of(chain_id: str) -> str:
    return chain_id.rsplit("|", 1)[1]


def eligible_pairs(records: list[PR.ChainRecord]):
    """Within topic, across documents — the standard ECB+ pairing."""
    by_topic: dict[str, list[PR.ChainRecord]] = defaultdict(list)
    for record in records:
        by_topic[topic_of(record.chain_id)].append(record)
    for group in by_topic.values():
        for a, b in itertools.combinations(group, 2):
            if a.episode_id != b.episode_id:
                yield a, b


def scored_pairs(records: list[PR.ChainRecord], scorer: PR.PairScorer):
    """(a, b, features, probability, label) for every eligible pair."""
    evidence = PR.build_evidence(records)
    rows = []
    for a, b in eligible_pairs(records):
        feats = PR.features(a, b, evidence)
        label = int(gold_of(a.chain_id) == gold_of(b.chain_id))
        rows.append((a, b, feats, scorer.score(feats), label))
    return rows, evidence


def auc(rows) -> float:
    """Rank quality of the scorer, independent of where the cut points land."""
    from sklearn.metrics import roc_auc_score

    labels = [r[4] for r in rows]
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, [r[3] for r in rows]))


def clustering_conll(records: list[PR.ChainRecord], rows, high: float) -> float:
    """CoNLL that a single merge cut point would produce on these records."""
    merges = [(a.chain_id, b.chain_id, p) for a, b, _f, p, _y in rows if p >= high]
    assignment = PR.cluster([r.chain_id for r in records], merges)
    gold = {r.chain_id: gold_of(r.chain_id) for r in records}
    return M.conll(M._as_clusters(gold), M._as_clusters(assignment))["CoNLL"].f1


def choose_high_by_conll(records: list[PR.ChainRecord], rows,
                         steps: int = 60) -> tuple[float, float]:
    """Pick the merge cut point that maximises CoNLL, not pairwise precision.

    ``choose_thresholds`` targets pairwise merge precision, which is the wrong
    objective once merges are closed transitively: a 95% target pushes the cut so
    high that recall collapses, and CEAFe punishes the resulting fragmentation
    far harder than the few extra wrong merges would have cost. The metric we
    report is the metric to tune against.
    """
    scores = np.array([r[3] for r in rows])
    grid = np.unique(np.quantile(scores, np.linspace(0.80, 1.0, steps)))
    best = max(((float(h), clustering_conll(records, rows, h)) for h in grid),
               key=lambda hf: hf[1])
    return best


def choose_band(rows, cut: float, fraction: float) -> PR.Thresholds:
    """Widen a window around the cut point until it holds ``fraction`` of pairs.

    The uncertain pairs sit on *both* sides of the cut, so the band is symmetric
    rather than the [low, high) that a precision target produces. Width is set by
    review budget, which is the thing actually being spent.
    """
    distance = np.sort(np.abs(np.array([r[3] for r in rows]) - cut))
    width = float(distance[min(int(len(distance) * fraction), len(distance) - 1)])
    return PR.Thresholds(low=max(0.0, cut - width), high=min(1.0, cut + width))


def oracle_conll(records: list[PR.ChainRecord], rows,
                 thresholds: PR.Thresholds) -> float:
    """CoNLL if every reviewed pair came back correct — the ceiling on review."""
    truth = {(a.chain_id, b.chain_id): bool(y) for a, b, _f, p, y in rows
             if PR.decide(p, thresholds) == PR.REVIEW}
    merges = []
    for a, b, _f, probability, _y in rows:
        verdict = PR.decide(probability, thresholds)
        if verdict == PR.REVIEW:
            verdict = (PR.MERGE if truth[(a.chain_id, b.chain_id)] else PR.SPLIT)
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))
    assignment = PR.cluster([r.chain_id for r in records], merges)
    gold = {r.chain_id: gold_of(r.chain_id) for r in records}
    return M.conll(M._as_clusters(gold), M._as_clusters(assignment))["CoNLL"].f1


def evaluate(records: list[PR.ChainRecord], rows, thresholds: PR.Thresholds,
             *, label: str, reviews: dict | None = None) -> dict:
    """Apply the decision rule, cluster, and print the CoNLL block."""
    cut = (thresholds.low + thresholds.high) / 2
    merges = []
    reviewed = 0
    for a, b, _feats, probability, _y in rows:
        verdict = PR.decide(probability, thresholds)
        if verdict == PR.REVIEW:
            reviewed += 1
            answer = (reviews or {}).get((a.chain_id, b.chain_id))
            if answer is None:  # unreviewed: fall back to the cut point
                answer = probability >= cut
            verdict = PR.MERGE if answer else PR.SPLIT
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))

    assignment = PR.cluster([r.chain_id for r in records], merges)
    gold = {r.chain_id: gold_of(r.chain_id) for r in records}
    scores = M.conll(M._as_clusters(gold), M._as_clusters(assignment))
    print(f"\n=== {label} ===")
    print(f"{len(merges)} merges, {reviewed} in the review band")
    print(M.report(gold, assignment))
    return {"conll": scores["CoNLL"].f1, "merges": len(merges),
            "reviewed": reviewed, "clusters": len(set(assignment.values()))}


def adjudicate(rows, thresholds: PR.Thresholds, evidence: PR.Evidence,
               passages: dict[str, str], *, model: str,
               cap: int | None) -> dict[tuple[str, str], bool]:
    """Send the uncertain band to the model, one call per pair."""
    cut = (thresholds.low + thresholds.high) / 2
    band = [(a, b) for a, b, _f, p, _y in rows
            if PR.decide(p, thresholds) == PR.REVIEW]
    if cap is not None and len(band) > cap:
        print(f"  review band is {len(band)} pairs; keeping the {cap} nearest "
              f"the cut")
        scores = {(a.chain_id, b.chain_id): p for a, b, _f, p, _y in rows}
        band.sort(key=lambda ab: abs(scores[(ab[0].chain_id, ab[1].chain_id)] - cut))
        band = band[:cap]
    if not band:
        return {}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=REVIEW_WORKERS) as pool:
        answers = list(pool.map(
            lambda ab: PL._adjudicate(ab[0], ab[1], model, evidence, passages),
            band))
    elapsed = time.perf_counter() - started

    out = {(a.chain_id, b.chain_id): bool(answer.get("same_entity"))
           for (a, b), answer in zip(band, answers)}
    agreed = sum(
        out[(a.chain_id, b.chain_id)] == bool(y)
        for a, b, _f, _p, y in rows
        if (a.chain_id, b.chain_id) in out
    )
    print(f"  {len(band)} review calls in {elapsed:.0f}s, "
          f"reviewer accuracy {agreed / len(band):.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.55)
    parser.add_argument("--dev", type=float, default=0.20)
    parser.add_argument("--merge-precision", type=float, default=0.95)
    parser.add_argument("--split-precision", type=float, default=0.95)
    parser.add_argument("--no-context", action="store_true",
                        help="Drop context_gated, for the ablation")
    parser.add_argument("--review-frac", type=float, default=0.01,
                        help="Fraction of pairs to route to review")
    parser.add_argument("--live", action="store_true",
                        help="Adjudicate the review band with real LLM calls")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-review", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    mentions, passages = read_corpus(args.root)
    topics = sorted({m.topic for m in mentions}, key=int)
    cut_a = int(len(topics) * args.train)
    cut_b = cut_a + int(len(topics) * args.dev)
    split_of = {"train": set(topics[:cut_a]), "dev": set(topics[cut_a:cut_b]),
                "test": set(topics[cut_b:])}

    print(f"{len(topics)} topics, {len({m.doc_id for m in mentions})} documents, "
          f"{len(mentions)} entity mentions")
    for name, group in split_of.items():
        print(f"  {name:<5} {len(group):>2} topics: "
              f"{' '.join(sorted(group, key=int))}")

    vectors = name_vectors(sorted(heads_of(mentions)))
    context = None if args.no_context else passages
    parts = {name: to_chain_records([m for m in mentions if m.topic in group],
                                    vectors, context)
             for name, group in split_of.items()}
    for name, records in parts.items():
        print(f"  {name:<5} {len(records):>4} chains, "
              f"{len({gold_of(r.chain_id) for r in records}):>3} gold entities")

    # ── fit on train ─────────────────────────────────────────────────────────
    train_rows, _ = scored_pairs(parts["train"], PR.PairScorer())
    scorer = PR.PairScorer().fit([r[2] for r in train_rows],
                                 [r[4] for r in train_rows])
    positives = sum(r[4] for r in train_rows)
    print(f"\nfitted on {len(train_rows)} train pairs "
          f"({positives} positive, {positives / len(train_rows):.1%})")
    print(f"  intercept {scorer.intercept:+.2f}")
    for name, weight in sorted(scorer.weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<18} {weight:+.2f}")

    # ── thresholds on dev ────────────────────────────────────────────────────
    dev_rows, _ = scored_pairs(parts["dev"], scorer)
    print(f"\ndev: {len(dev_rows)} pairs, AUC {auc(dev_rows):.4f}")

    by_precision = PR.choose_thresholds(
        [r[3] for r in dev_rows], [r[4] for r in dev_rows],
        merge_precision=args.merge_precision,
        split_precision=args.split_precision)
    print(f"  targeting {args.merge_precision:.0%} merge precision -> "
          f"merge at {by_precision.high:.3f}, dev CoNLL "
          f"{clustering_conll(parts['dev'], dev_rows, by_precision.high):.3f}")

    cut, dev_best = choose_high_by_conll(parts["dev"], dev_rows)
    print(f"  maximising dev CoNLL      -> merge at {cut:.3f}, dev CoNLL "
          f"{dev_best:.3f}")

    thresholds = choose_band(dev_rows, cut, args.review_frac)
    in_band = sum(1 for r in dev_rows if PR.decide(r[3], thresholds) == PR.REVIEW)
    print(f"  band [{thresholds.low:.3f}, {thresholds.high:.3f}) holds {in_band} "
          f"dev pairs ({in_band / len(dev_rows):.1%}) for review; "
          f"a perfect reviewer would take dev CoNLL {dev_best:.3f} -> "
          f"{oracle_conll(parts['dev'], dev_rows, thresholds):.3f}")

    # ── report on test ───────────────────────────────────────────────────────
    test = parts["test"]
    gold = {r.chain_id: gold_of(r.chain_id) for r in test}
    results = {}

    exact = {r.chain_id: f"{topic_of(r.chain_id)}|{r.head.lower()}" for r in test}
    baseline = M.conll(M._as_clusters(gold), M._as_clusters(exact))
    print(f"\n=== baseline: exact head match ===\n{M.report(gold, exact)}")
    results["exact_head"] = {"conll": baseline["CoNLL"].f1}

    prior_rows, _ = scored_pairs(test, PR.PairScorer())
    results["prior"] = evaluate(test, prior_rows, PR.Thresholds(0.30, 0.50),
                                label="prior weights, hand-set thresholds")

    test_rows, evidence = scored_pairs(test, scorer)
    print(f"\ntest AUC: prior {auc(prior_rows):.4f}, fitted {auc(test_rows):.4f}")
    results["fitted_precision_target"] = evaluate(
        test, test_rows, by_precision,
        label=f"fitted weights, {args.merge_precision:.0%}-precision thresholds")
    results["fitted"] = evaluate(test, test_rows, PR.Thresholds(cut, cut),
                                 label="fitted weights, dev-CoNLL cut (no review)")
    print(f"  a perfect reviewer over the band would reach "
          f"{oracle_conll(test, test_rows, thresholds):.3f}")

    if args.live:
        reviews = adjudicate(test_rows, thresholds, evidence, passages,
                             model=args.model, cap=args.max_review)
        results["fitted_llm"] = evaluate(test, test_rows, thresholds,
                                         label="fitted weights + LLM review",
                                         reviews=reviews)

    print("\n=== test summary ===")
    for name, result in results.items():
        print(f"  {name:<34} CoNLL {result['conll']:.3f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
