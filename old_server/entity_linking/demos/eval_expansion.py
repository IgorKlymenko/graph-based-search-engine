"""Does expansion reach the *right* neighbours, or just the busiest ones?

    uv run python entity_linking/demos/eval_expansion.py
    uv run python entity_linking/demos/eval_expansion.py --episodes 600

Co-occurrence gives ~8 edges per node on real data, and nothing so far has
checked whether those edges point anywhere useful. On ECB+ they cannot be
checked: relatedness has no gold there, and every document in a topic is about
one event so everything is related to everything.

Here relatedness is true by construction. Entities belong to **projects**, and
two entities are genuinely related exactly when they share one. The corpus is
then generated with three kinds of episode:

  focused    a few entities from one project. Honest evidence.
  crosstalk  entities from two projects in one passage — a status meeting.
             Real co-occurrence, no real relationship. This is the trap.
  noisy      one project plus several noise entities that appear everywhere,
             which is how hubs form.

Extraction and linking are skipped: chains are built directly from the generated
mentions. That is deliberate — this isolates the graph layer, so anything measured
here is the edges and the walk, not the extractor.

Scored per seed entity: of the entities expansion reaches, how many share the
seed's project (precision), how much of the project is reached (recall), and how
much of what came back is pure noise.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_cooccurrence as CO  # noqa: E402

PROJECTS = {
    "identity": ["Sarah Chen", "Okta", "SCIM", "Workvivo", "Entra ID"],
    "billing": ["Steven Chen", "Stripe", "the invoice service", "Chargebee"],
    "search": ["Jina Park", "the reranker", "OpenSearch", "Elastic"],
    "data": ["Joon Park", "Snowflake", "Airflow", "the warehouse", "dbt"],
    "mobile": ["Maria Lopez", "the iOS client", "TestFlight", "Firebase"],
    "infra": ["Diego Lopez", "Kubernetes", "Terraform", "the mesh", "Istio"],
    "docs": ["Aisha Rahman", "Notion", "the handbook", "Confluence"],
    "support": ["Tomas Novak", "Zendesk", "the escalation queue", "Intercom"],
}
NOISE = ["the weekly sync", "the roadmap", "Q3", "the design review",
         "the all-hands", "the postmortem", "the standup", "the retro"]


def bridged_pairs() -> dict[str, tuple[str, str]]:
    """One pair per project that is never allowed in the same episode.

    They share a project, so they are genuinely related, but no direct edge can
    ever form between them. Reaching one from the other requires walking through
    a third member -- which is the whole reason to run PPR instead of just
    listing a seed's neighbours.
    """
    return {name: (members[0], members[-1]) for name, members in PROJECTS.items()}


def generate(episodes: int, seed: int = 7) -> list[dict]:
    """Chains, as stage 2 would leave them, with the project known per entity."""
    rng = random.Random(seed)
    names = sorted(PROJECTS)
    forbidden = bridged_pairs()
    out: list[dict] = []

    def drop_bridge(chosen: list[str], project: str) -> list[str]:
        a, z = forbidden[project]
        if a in chosen and z in chosen:
            # pick the victim once: putting rng.random() inside the
            # comprehension re-rolls per element and keeps both
            victim = a if rng.random() < 0.5 else z
            chosen = [e for e in chosen if e != victim]
        return chosen
    for i in range(episodes):
        doc = f"ep{i:04d}"
        roll = rng.random()
        if roll < 0.55:                                   # focused
            project = rng.choice(names)
            members = rng.sample(PROJECTS[project],
                                 rng.randint(2, min(4, len(PROJECTS[project]))))
            entities = drop_bridge(list(members), project)
        elif roll < 0.80:                                 # crosstalk
            a, b = rng.sample(names, 2)
            entities = (drop_bridge(rng.sample(PROJECTS[a], rng.randint(1, 2)), a)
                        + drop_bridge(rng.sample(PROJECTS[b], rng.randint(1, 2)), b))
        else:                                             # noisy
            project = rng.choice(names)
            entities = (drop_bridge(
                            rng.sample(PROJECTS[project], rng.randint(2, 3)),
                            project)
                        + rng.sample(NOISE, rng.randint(2, 4)))
        if rng.random() < 0.4:
            entities.append(rng.choice(NOISE))
        for entity in dict.fromkeys(entities):
            out.append({"doc": doc, "entity": entity, "surfaces": [entity]})
    return out


def project_of() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for project, members in PROJECTS.items():
        for entity in members:
            out[entity] = project
    for entity in NOISE:
        out[entity] = None
    return out


# ── edge weighting schemes ────────────────────────────────────────────────────

def weighted_edges(chains: list[dict], scheme: str) -> dict[tuple[str, str], float]:
    """Co-occurrence weights under one scheme.

    ``inverse_size`` is the interesting one: a pair seen in a passage naming
    three entities is stronger evidence than the same pair in a status meeting
    naming twenty, so each episode contributes 1/(c-1) rather than 1.

    ``pmi`` asks whether two entities co-occur *more than chance*, which is what
    demotes an entity that co-occurs with everything.
    """
    by_doc: dict[str, list[str]] = defaultdict(list)
    for chain in chains:
        if chain["entity"] not in by_doc[chain["doc"]]:
            by_doc[chain["doc"]].append(chain["entity"])

    pairs: Counter = Counter()
    weighted: dict[tuple[str, str], float] = defaultdict(float)
    appears: Counter = Counter()
    for entities in by_doc.values():
        ordered = sorted(entities)
        for entity in ordered:
            appears[entity] += 1
        contribution = 1.0 / max(len(ordered) - 1, 1)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pairs[(a, b)] += 1
                weighted[(a, b)] += contribution

    total = len(by_doc)
    if scheme == "raw":
        return dict(pairs)
    if scheme == "binary":
        return {pair: 1.0 for pair in pairs}
    if scheme == "min2":
        return {pair: c for pair, c in pairs.items() if c >= 2}
    if scheme == "inverse_size":
        return dict(weighted)
    if scheme in ("pmi", "pmi x inverse_size"):
        out = {}
        for (a, b), count in pairs.items():
            expected = appears[a] * appears[b] / total
            pmi = math.log(max(count, 1e-9) / max(expected, 1e-9))
            if pmi <= 0:
                continue  # co-occur no more than chance: not evidence
            out[(a, b)] = pmi * (weighted[(a, b)] / count
                                 if scheme.endswith("inverse_size") else 1.0)
        return out
    raise ValueError(scheme)


SCHEMES = ("raw", "binary", "min2", "inverse_size", "pmi", "pmi x inverse_size")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    chains = generate(args.episodes)
    belongs = project_of()
    entities = sorted({c["entity"] for c in chains})
    position = {e: i for i, e in enumerate(entities)}
    print(f"{args.episodes} episodes, {len(entities)} entities "
          f"({len(NOISE)} of them noise), {len(PROJECTS)} projects\n")

    bridges = bridged_pairs()
    partner = {}
    for project, (a, z) in bridges.items():
        partner[a], partner[z] = z, a

    def evaluate(edges, alpha, one_hop=False):
        """Precision/recall over each seed's project, plus bridge reach."""
        totals = defaultdict(float)
        asked = bridge_asked = 0
        for entity in entities:
            project = belongs[entity]
            if project is None:
                continue
            asked += 1
            if one_hop:
                # no walk at all: a seed's direct neighbours, by edge weight
                scores = np.zeros(len(entities))
                for (a, b), w in edges.items():
                    if a == entity:
                        scores[position[b]] = w
                    elif b == entity:
                        scores[position[a]] = w
            else:
                scores = CO.ppr(edges, position, {entity: 1.0}, alpha=alpha)
            order = [entities[j] for j in np.argsort(-scores)
                     if entities[j] != entity and scores[j] > 0]
            family = set(PROJECTS[project]) - {entity}
            top = order[:len(family)]
            if top:
                totals["P"] += sum(1 for e in top
                                   if belongs[e] == project) / len(top)
            totals["R"] += len(family & set(top)) / len(family) if family else 0.0
            beyond = order[:10]
            if beyond:
                totals["noise"] += sum(1 for e in beyond
                                       if belongs[e] is None) / len(beyond)
            mate = partner.get(entity)
            if mate is not None:
                bridge_asked += 1
                totals["bridge_fam"] += float(mate in top)
                totals["bridge_10"] += float(mate in beyond)
        return ({k: v / asked for k, v in totals.items() if not k.startswith("bridge")}
                | {k: v / max(bridge_asked, 1) for k, v in totals.items()
                   if k.startswith("bridge")})

    print("=== edge weighting (alpha = %.2f) ===" % args.alpha)
    print(f"{'scheme':<22}{'edges':>7}{'deg':>7}{'P@fam':>8}{'recall':>8}"
          f"{'noise@10':>10}{'bridge@fam':>12}")
    graphs = {}
    for scheme in SCHEMES:
        edges = graphs[scheme] = weighted_edges(chains, scheme)
        degree: Counter = Counter()
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        r = evaluate(edges, args.alpha)
        mean_degree = np.mean(list(degree.values())) if degree else 0.0
        print(f"{scheme:<22}{len(edges):>7}{mean_degree:>7.1f}{r['P']:>8.3f}"
              f"{r['R']:>8.3f}{r['noise']:>10.3f}{r['bridge_fam']:>12.3f}")

    forbidden = sum(1 for (a, z) in bridges.values()
                    if ((a, z) if a < z else (z, a)) not in graphs["raw"])
    print(f"\n{len(bridges)} bridged pairs planted (same project, never in one "
          f"episode); {forbidden} have no direct edge, as intended")

    print("\n=== how far to walk (PMI edges) ===")
    print(f"{'alpha':<22}{'P@fam':>8}{'recall':>8}{'noise@10':>10}"
          f"{'bridge@fam':>12}{'bridge@10':>11}")
    pmi = graphs["pmi"]
    r = evaluate(pmi, 1.0, one_hop=True)
    print(f"{'1-hop, no walk':<22}{r['P']:>8.3f}{r['R']:>8.3f}{r['noise']:>10.3f}"
          f"{r['bridge_fam']:>12.3f}{r['bridge_10']:>11.3f}")
    for alpha in (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95):
        r = evaluate(pmi, alpha)
        print(f"{'restart ' + format(alpha, '.2f'):<22}{r['P']:>8.3f}"
              f"{r['R']:>8.3f}{r['noise']:>10.3f}{r['bridge_fam']:>12.3f}"
              f"{r['bridge_10']:>11.3f}")

    print("\nP@fam       scored at k = the seed's project size; 1.000 is perfect")
    print("bridge@fam  the partner it can only reach through a third entity")
    print("alpha       restart probability: high = stay near the seed")


if __name__ == "__main__":
    main()
