"""Standard coreference scores: MUC, B-cubed, CEAFe, and their CoNLL average.

Pairwise precision/recall flatters a fragmented clustering. Splitting one entity
into five clusters loses only the pairs *between* the fragments, while the many
correct pairs *inside* each fragment still count — so a system can report high
pairwise precision while producing nonsense clusters. These three metrics look
at the clustering itself, from three different angles:

* **MUC** counts merge links. Cheapest to satisfy: it rewards any link that
  joins two mentions of one entity, so it barely punishes over-merging and is
  nearly blind to singleton errors.
* **B-cubed** scores each mention by how much of its cluster is correct. It
  punishes over-splitting and over-merging symmetrically, per mention, so a
  large wrong cluster hurts every mention in it.
* **CEAFe** aligns predicted clusters to gold one-to-one and scores the
  alignment. Because each gold entity can claim only one predicted cluster,
  fragmentation is punished hardest here.

CoNLL F1 is the mean of the three, which is what the literature reports.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

Cluster = frozenset[str]


class Score(NamedTuple):
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def _as_clusters(assignment: dict[str, object]) -> list[Cluster]:
    grouped: dict[object, set[str]] = {}
    for mention, key in assignment.items():
        grouped.setdefault(key, set()).add(mention)
    return [frozenset(v) for v in grouped.values()]


def _muc_side(truth: list[Cluster], other: list[Cluster]) -> float:
    """Fraction of the links inside ``truth`` that ``other`` also makes."""
    numerator = denominator = 0
    for cluster in truth:
        if len(cluster) < 2:
            continue  # MUC ignores singletons entirely
        pieces = {frozenset(cluster & o) for o in other}
        pieces = {p for p in pieces if p}
        numerator += len(cluster) - len(pieces)
        denominator += len(cluster) - 1
    return numerator / denominator if denominator else 0.0


def muc(gold: list[Cluster], system: list[Cluster]) -> Score:
    return Score(precision=_muc_side(system, gold), recall=_muc_side(gold, system))


def b_cubed(gold: list[Cluster], system: list[Cluster]) -> Score:
    gold_of = {m: c for c in gold for m in c}
    system_of = {m: c for c in system for m in c}
    mentions = set(gold_of) & set(system_of)
    if not mentions:
        return Score(0.0, 0.0)
    precision = recall = 0.0
    for mention in mentions:
        overlap = len(gold_of[mention] & system_of[mention])
        precision += overlap / len(system_of[mention])
        recall += overlap / len(gold_of[mention])
    return Score(precision / len(mentions), recall / len(mentions))


def ceafe(gold: list[Cluster], system: list[Cluster]) -> Score:
    """One-to-one alignment scored by the phi4 similarity."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not gold or not system:
        return Score(0.0, 0.0)
    similarity = np.zeros((len(gold), len(system)))
    for i, g in enumerate(gold):
        for j, s in enumerate(system):
            if g & s:
                similarity[i, j] = 2 * len(g & s) / (len(g) + len(s))
    rows, cols = linear_sum_assignment(-similarity)
    total = float(similarity[rows, cols].sum())
    return Score(precision=total / len(system), recall=total / len(gold))


def conll(gold: list[Cluster], system: list[Cluster]) -> dict[str, Score]:
    """All three plus their mean F1, which is the headline CoNLL number."""
    scores = {"MUC": muc(gold, system), "B3": b_cubed(gold, system),
              "CEAFe": ceafe(gold, system)}
    mean = sum(s.f1 for s in scores.values()) / 3
    scores["CoNLL"] = Score(mean, mean)
    return scores


def report(
    gold_assignment: dict[str, object], system_assignment: dict[str, object]
) -> str:
    gold, system = _as_clusters(gold_assignment), _as_clusters(system_assignment)
    lines = [f"{'metric':<8}{'P':>8}{'R':>8}{'F1':>8}", "-" * 32]
    for name, score in conll(gold, system).items():
        if name == "CoNLL":
            lines.append("-" * 32)
            lines.append(f"{name:<8}{'':>8}{'':>8}{score.f1:>8.3f}")
        else:
            lines.append(f"{name:<8}{score.precision:>8.3f}"
                         f"{score.recall:>8.3f}{score.f1:>8.3f}")
    lines.append(f"\nclusters: {len(system)} predicted vs {len(gold)} gold")
    return "\n".join(lines)


def pairwise(
    gold_assignment: dict[str, object], system_assignment: dict[str, object]
) -> Score:
    """The metric we had been reporting, for comparison."""
    mentions = sorted(set(gold_assignment) & set(system_assignment))
    tp = fp = fn = 0
    for i, a in enumerate(mentions):
        for b in mentions[i + 1:]:
            same_gold = gold_assignment[a] == gold_assignment[b]
            same_system = system_assignment[a] == system_assignment[b]
            tp += same_gold and same_system
            fp += same_system and not same_gold
            fn += same_gold and not same_system
    return Score(tp / (tp + fp) if tp + fp else 0.0,
                 tp / (tp + fn) if tp + fn else 0.0)


def summarise(gold: dict[str, object], system: dict[str, object]) -> str:
    pair = pairwise(gold, system)
    return (f"{report(gold, system)}\n"
            f"pairwise  P {pair.precision:.3f}  R {pair.recall:.3f}  "
            f"F1 {pair.f1:.3f}   <- what we had been reporting")


def _iter_check(clusters: Iterable[Cluster]) -> None:  # pragma: no cover
    seen: set[str] = set()
    for cluster in clusters:
        assert not (seen & cluster), "clusters must be disjoint"
        seen |= cluster
