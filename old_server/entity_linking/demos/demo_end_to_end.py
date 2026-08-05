"""End-to-end: episodes -> coref chains -> pair scoring -> review -> entities.

    uv run python entity_linking/demos/demo_end_to_end.py
    uv run python entity_linking/demos/demo_end_to_end.py --corpus large
    uv run python entity_linking/demos/demo_end_to_end.py --dump /tmp/e2e.json

Real LLM calls at both model steps: one coref call per episode, then one
adjudication call per pair that lands in the review band. Everything else —
parsing, span marking, embedding, feature extraction, thresholding — is local.

The corpus has a hand-written gold assignment, so the output is scored rather
than described: pairwise precision/recall against the gold, and the final
clustering compared entity by entity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent      # entity_linking/
sys.path.insert(0, str(_PKG))
sys.path.insert(0, str(_PKG.parent))

from entity_linking import chains as S1  # noqa: E402  (2_chains.py)
from entity_linking import coref_metrics as M  # noqa: E402
from entity_linking import encoding as EN  # noqa: E402
from entity_linking import pairing as P  # noqa: E402
from entity_linking.prompts import (  # noqa: E402
    MERGE_REVIEW_PROMPT,
    MERGE_REVIEW_SCHEMA,
)

EPISODES = {
    # the alias is revealed here — the only place linking the two forms
    "ep1": "Sarah Chen leads the identity migration. S. Chen signed off the "
           "rollback plan before the freeze.",
    "ep2": "S. Chen reviewed the Okta configuration and approved the change.",
    # both Chens in one passage — the co-occurrence signal
    "ep3": "Sarah Chen and Steven Chen both joined the review. Steven Chen "
           "owns billing while Sarah Chen owns identity.",
    "ep4": "Steven Chen published the pricing note for next quarter.",
    "ep5": "Workvivo syncs team names from Okta on every scheduled run.",
    "ep6": "Provisioning in Workvivo can come from SCIM or from Okta.",
    # this alias is never revealed anywhere — nothing can link it
    "ep7": "Maria Lopez configured the directory sync for the pilot tenant.",
    "ep8": "M. Lopez closed the incident about the failed nightly sync.",
}

SMALL_GOLD_BY_SURFACE = {
    "Sarah Chen": "e_sarah", "S. Chen": "e_sarah", "Steven Chen": "e_steven",
    "Workvivo": "e_workvivo", "Okta": "e_okta", "SCIM": "e_scim",
    "Maria Lopez": "e_maria", "M. Lopez": "e_maria",
}
GOLD = {(e, s): ent for e in EPISODES for s, ent in SMALL_GOLD_BY_SURFACE.items()}

# ── larger corpus: ambiguous alias, shared surnames, unrevealed alias ─────────

LARGE = [
    ("Sarah Chen leads the identity migration. S. Chen signed off the rollback "
     "plan before the freeze.", {"Sarah Chen": "e_sarah", "S. Chen": "e_sarah"}),
    ("S. Chen reviewed the Okta configuration and approved the change.",
     {"S. Chen": "e_sarah", "Okta": "e_okta"}),
    ("Sarah Chen and Steven Chen both joined the review. Steven Chen owns "
     "billing while Sarah Chen owns identity.",
     {"Sarah Chen": "e_sarah", "Steven Chen": "e_steven"}),
    ("Steven Chen published the pricing note for next quarter.",
     {"Steven Chen": "e_steven"}),
    ("Steven Chen escalated the billing defect to the vendor.",
     {"Steven Chen": "e_steven"}),
    # Jina Park introduces "J. Park" ...
    ("Jina Park owns the search relevance work. J. Park shipped the reranker "
     "last month.", {"Jina Park": "e_jina", "J. Park": "e_jina"}),
    ("J. Park tuned the reranker weights after the evaluation.",
     {"J. Park": "e_jina"}),
    # ... and so does Joon Park, in a different passage. The alias is ambiguous.
    ("Joon Park manages the data platform. J. Park approved the retention "
     "policy.", {"Joon Park": "e_joon", "J. Park": "e_joon"}),
    ("J. Park signed off the retention policy for the warehouse.",
     {"J. Park": "e_joon"}),
    ("Jina Park and Joon Park disagreed about index refresh cadence.",
     {"Jina Park": "e_jina", "Joon Park": "e_joon"}),
    # Lopez pair: Maria's alias is never revealed anywhere
    ("Maria Lopez configured the directory sync for the pilot tenant.",
     {"Maria Lopez": "e_maria"}),
    ("M. Lopez closed the incident about the failed nightly sync.",
     {"M. Lopez": "e_maria"}),
    ("Diego Lopez drafted the onboarding guide. D. Lopez reviewed it.",
     {"Diego Lopez": "e_diego", "D. Lopez": "e_diego"}),
    ("D. Lopez updated the onboarding guide after the pilot.",
     {"D. Lopez": "e_diego"}),
    ("Maria Lopez and Diego Lopez presented the rollout plan together.",
     {"Maria Lopez": "e_maria", "Diego Lopez": "e_diego"}),
    # products
    ("Workvivo syncs team names from Okta on every scheduled run.",
     {"Workvivo": "e_workvivo", "Okta": "e_okta"}),
    ("Provisioning in Workvivo can come from SCIM or from Okta.",
     {"Workvivo": "e_workvivo", "SCIM": "e_scim", "Okta": "e_okta"}),
    ("SCIM defines a schema for users and groups.", {"SCIM": "e_scim"}),
    ("Neo4j stores the resulting graph for later queries.",
     {"Neo4j": "e_neo4j"}),
    ("Neo4j evaluates Cypher over the property graph.", {"Neo4j": "e_neo4j"}),
    ("The migration moved provisioning from SCIM to Okta last quarter.",
     {"SCIM": "e_scim", "Okta": "e_okta"}),
    ("Okta issues SAML assertions for every tenant.", {"Okta": "e_okta"}),
]


def build_xl() -> list[tuple[str, dict[str, str]]]:
    """A larger corpus with the hard cases at realistic density.

    Most entities are easy; a minority carry a contested initial, an alias that
    is never revealed, or a shared surname. That mix is the point — a corpus
    made entirely of hard cases overstates how often the linker is in trouble.
    """
    import random
    rng = random.Random(17)
    people = [
        ("Sarah", "Chen"), ("Steven", "Chen"),      # shared surname + initial
        ("Jina", "Park"), ("Joon", "Park"),         # contested "J. Park"
        ("Maria", "Lopez"), ("Diego", "Lopez"),
        ("Aisha", "Rahman"), ("Tomas", "Novak"), ("Rachel", "Weber"),
        ("Karim", "Haddad"), ("Sofia", "Marino"), ("Daniel", "Kovacs"),
        ("Elena", "Silva"), ("Marcus", "Reyes"), ("Priya", "Nair"),
        ("Robert", "Fischer"), ("Nadia", "Duarte"), ("Julia", "Larsen"),
    ]
    products = ["Workvivo", "Okta", "SCIM", "Neo4j", "Slack", "Postgres",
                "Airflow", "Snowflake"]
    solo = ["{x} reviewed the {t} and approved the release.",
            "{x} published the notes for the {t}.",
            "{x} filed an incident about the {t}.",
            "{x} owns the {t} this quarter.",
            "{x} closed the escalation after the {t} shipped."]
    topics = ["migration plan", "connector", "nightly sync", "billing rollout",
              "retention policy", "onboarding guide", "index refresh"]
    duo = ["{x} disagreed with {y} about the {t}.",
           "The audit covered {x} and {y} without blocking the {t}."]
    reveal = ["{x} opened the ticket. {y} closed it after the fix.",
              "{x} drafted the {t}. {y} reviewed it the same day."]
    prod = ["{p} syncs identities from {q} on every run.",
            "Provisioning in {p} can come from {q}.",
            "{p} stores the resulting graph for later queries."]

    out: list[tuple[str, dict[str, str]]] = []
    # a third of people never have their alias revealed
    revealed = {i for i in range(len(people)) if i % 3 != 2}
    for i, (first, last) in enumerate(people):
        canon = f"{first} {last}"
        alias = f"{first[0]}. {last}"
        entity = f"e_{first}_{last}"
        for template in rng.sample(solo, 2):
            out.append((template.format(x=canon, t=rng.choice(topics)),
                        {canon: entity}))
        out.append((rng.choice(solo).format(x=alias, t=rng.choice(topics)),
                    {alias: entity}))
        if i in revealed:
            out.append((rng.choice(reveal).format(x=canon, y=alias,
                                                  t=rng.choice(topics)),
                        {canon: entity, alias: entity}))
    for _ in range(14):  # co-occurrence, biased toward confusable pairs
        i, j = rng.sample(range(len(people)), 2)
        (f1, l1), (f2, l2) = people[i], people[j]
        out.append((rng.choice(duo).format(x=f"{f1} {l1}", y=f"{f2} {l2}",
                                           t=rng.choice(topics)),
                    {f"{f1} {l1}": f"e_{f1}_{l1}", f"{f2} {l2}": f"e_{f2}_{l2}"}))
    for _ in range(12):
        a, b = rng.sample(products, 2)
        out.append((rng.choice(prod).format(p=a, q=b), {a: f"e_{a}", b: f"e_{b}"}))
    return out


def name_vectors(strings: list[str]) -> dict[str, np.ndarray]:
    import torch

    tokenizer, model, device = EN._load()
    batch = tokenizer([EN.PREFIX + s for s in strings], return_tensors="pt",
                      padding=True, truncation=True, max_length=64)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state.cpu().numpy()
    mask = batch["attention_mask"].cpu().numpy()[..., None]
    vectors = (hidden * mask).sum(1) / mask.sum(1)
    return dict(zip(strings, vectors / np.linalg.norm(vectors, axis=1,
                                                      keepdims=True)))


def gold_of(record: P.ChainRecord) -> str | None:
    """Which gold entity a chain denotes, or None if it is not one we track.

    Keyed per episode, not per surface: a short form like "J. Park" can denote
    different people in different passages, which is the case the linker most
    needs to get right and a global surface->entity map cannot express.
    """
    labels = {GOLD[(record.episode_id, s)] for s in record.surfaces
              if (record.episode_id, s) in GOLD}
    return labels.pop() if len(labels) == 1 else None


def adjudicate(a: P.ChainRecord, b: P.ChainRecord, model: str,
               ambiguous: frozenset[str] = frozenset()) -> dict:
    """One LLM call for a pair the scorer abstained on.

    Contested short forms are named explicitly. Without that the reviewer sees
    two plausible passages about "J. Park" and merges them; it has no way to
    know the corpus uses that form for two different people.
    """
    shared = sorted((a.surfaces | b.surfaces) & ambiguous)
    warning = (f"\n\nNOTE: {shared} is used for more than one entity elsewhere "
               f"in this corpus. Do not treat it as identifying on its own."
               if shared else "")
    user = (f"A names: {sorted(a.surfaces)}\nA passage: {EPISODES[a.episode_id]}\n\n"
            f"B names: {sorted(b.surfaces)}\nB passage: {EPISODES[b.episode_id]}"
            + warning)
    return S1._complete_json(system=MERGE_REVIEW_PROMPT, user=user, model=model,
                             schema=MERGE_REVIEW_SCHEMA, schema_name="merge_review")


def components(records: list[P.ChainRecord], merges: set[frozenset]) -> dict[str, int]:
    """Union-find over accepted merges -> a cluster id per chain."""
    parent = {r.chain_id: r.chain_id for r in records}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in merges:
        x, y = tuple(pair)
        parent[find(x)] = find(y)
    roots = {}
    return {r.chain_id: roots.setdefault(find(r.chain_id), len(roots))
            for r in records}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("small", "large", "xl"),
                        default="small")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()
    if args.corpus in ("large", "xl"):
        corpus = LARGE if args.corpus == "large" else build_xl()
        EPISODES.clear()
        GOLD.clear()
        for i, (text, mapping) in enumerate(corpus):
            episode_id = f"ep{i:02d}"
            EPISODES[episode_id] = text
            for surface, entity in mapping.items():
                GOLD[(episode_id, surface)] = entity
    calls = {"coref": 0, "review": 0}
    started = time.perf_counter()

    print(f"=== 1. coref, one LLM call per episode ({len(EPISODES)} calls) ===")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        per_episode = dict(zip(EPISODES, pool.map(
            lambda t: S1.resolve_chains(S1.extract_chains(t, model=args.model)),
            EPISODES.values())))
    calls["coref"] = len(per_episode)
    records: list[P.ChainRecord] = []
    for episode_id, chains in per_episode.items():
        vectors = name_vectors([c[0].text for c in chains])
        shown = []
        for i, chain in enumerate(chains):
            record = P.ChainRecord(
                chain_id=f"{episode_id}-{i}", episode_id=episode_id,
                head=chain[0].text,
                surfaces=frozenset(s.text for s in chain),
                candidate_type=chain[0].candidate_type,
                name_vector=vectors[chain[0].text], context_vector=None)
            records.append(record)
            if gold_of(record):
                shown.append("{" + ", ".join(sorted(record.surfaces)) + "}")
        if args.corpus != "xl":
            print(f"  {episode_id}: " + (" ".join(shown) or "(none tracked)"))

    tracked = [r for r in records if gold_of(r)]
    print(f"\n  {len(records)} chains, {len(tracked)} carry a gold entity")

    evidence = P.build_evidence(records)
    print(f"  evidence: {len(evidence.same_chain)} alias pairs, "
          f"{len(evidence.split_in_episode)} kept-apart pairs, "
          f"{len(evidence.ambiguous_surfaces)} contested surfaces "
          f"{sorted(evidence.ambiguous_surfaces)}")
    for pair in sorted(evidence.same_chain):
        print(f"     alias: {pair[0]!r} = {pair[1]!r}")

    print("\n=== 2. score every cross-episode pair (local) ===")
    scorer = P.PairScorer()
    pairs, rows = [], []
    for a, b in itertools.combinations(tracked, 2):
        if a.episode_id == b.episode_id:
            continue
        feats = P.features(a, b, evidence)
        pairs.append((a, b, scorer.score(feats), feats))
        rows.append(int(gold_of(a) == gold_of(b)))
    thresholds = P.choose_thresholds([p for _, _, p, _ in pairs], rows,
                                     merge_precision=1.0, split_precision=1.0)
    print(f"  {len(pairs)} pairs, {sum(rows)} true merges")
    print(f"  band = [{thresholds.low:.3f}, {thresholds.high:.3f}]")

    print("\n=== 3. decide; review band goes to the LLM ===")
    accepted: list[tuple[str, str, float]] = []
    cannot_link: set[frozenset] = set()
    decisions = []
    to_review = [(a, b) for (a, b, p, _) in pairs
                 if P.decide(p, thresholds) == P.REVIEW]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        answers = dict(zip(
            [(a.chain_id, b.chain_id) for a, b in to_review],
            pool.map(lambda ab: adjudicate(ab[0], ab[1], args.model,
                                           evidence.ambiguous_surfaces),
                     to_review)))
    calls["review"] = len(answers)
    for (a, b, probability, feats), label in zip(pairs, rows):
        verdict = P.decide(probability, thresholds)
        note = ""
        if verdict == P.REVIEW:
            answer = answers[(a.chain_id, b.chain_id)]
            same = bool(answer["same_entity"])
            note = f"  LLM: {'same' if same else 'different'} " \
                   f"({answer['confidence']:.2f}) — {answer['reason'][:60]}"
            verdict = P.MERGE if same else P.SPLIT
        if verdict == P.MERGE:
            accepted.append((a.chain_id, b.chain_id, probability))
        else:
            # kept for inspection only; not fed to the clusterer (see P.cluster)
            cannot_link.add(frozenset((a.chain_id, b.chain_id)))
        correct = (verdict == P.MERGE) == bool(label)
        decisions.append({"a": sorted(a.surfaces), "b": sorted(b.surfaces),
                          "p": probability, "verdict": verdict,
                          "gold": label, "correct": correct})
        if (not correct or note) and args.corpus != "xl":
            mark = "ok " if correct else "WRONG"
            gold_txt = "same" if label else "diff"
            print(f"  [{mark}] {sorted(a.surfaces)} ~ {sorted(b.surfaces)} "
                  f"p={probability:.2f} -> {verdict} (gold={gold_txt})")
            if note:
                print(note)

    print("\n=== 4. compare with the gold ===")
    tp = sum(1 for d in decisions if d["verdict"] == P.MERGE and d["gold"])
    fp = sum(1 for d in decisions if d["verdict"] == P.MERGE and not d["gold"])
    fn = sum(1 for d in decisions if d["verdict"] != P.MERGE and d["gold"])
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    print(f"  pairwise: precision {precision:.0%}  recall {recall:.0%}  "
          f"(tp={tp} fp={fp} fn={fn})")

    predicted = P.cluster([r.chain_id for r in tracked], accepted)
    by_cluster: dict[int, list[P.ChainRecord]] = {}
    for record in tracked:
        by_cluster.setdefault(predicted[record.chain_id], []).append(record)
    gold_entities = {gold_of(r) for r in tracked}
    print(f"  entities: {len(by_cluster)} predicted vs {len(gold_entities)} gold")
    for cluster, members in sorted(by_cluster.items()):
        labels = {gold_of(m) for m in members}
        surfaces = sorted({s for m in members for s in m.surfaces})
        flag = "" if len(labels) == 1 else f"   <-- MIXED {sorted(labels)}"
        print(f"     {surfaces}{flag}")

    print("\n=== 5. coreference scores ===")
    print(M.summarise({r.chain_id: gold_of(r) for r in tracked}, predicted))

    print(f"\n  LLM calls: {calls['coref']} coref + {calls['review']} review "
          f"= {sum(calls.values())};  wall clock {time.perf_counter()-started:.1f}s")
    import pickle
    pickle.dump({
        "chain_ids": [r.chain_id for r in tracked],
        "gold": {r.chain_id: gold_of(r) for r in tracked},
        "merges": accepted,
        "cannot_link": cannot_link,
        "pairs": [(a.chain_id, b.chain_id, p) for a, b, p, _ in pairs],
        "labels": rows,
        "surfaces": {r.chain_id: sorted(r.surfaces) for r in tracked},
    }, open("/tmp/state.pkl", "wb"))
    print("  state -> /tmp/state.pkl")

    if args.dump:
        args.dump.write_text(json.dumps(decisions, indent=2, default=float))
        print(f"  wrote decisions -> {args.dump}")


if __name__ == "__main__":
    main()
