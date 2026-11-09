"""Score whether two coreference chains from different episodes are one entity.

Cosine alone cannot do this. Two people at one company are described in the same
language, appear in the same kinds of episode, and sit at nearly the same point
in embedding space — "Sarah Chen" vs "S. Chen" and "Sarah Chen" vs "Sarah Kim"
differ by 0.03 cosine, and no threshold separates them.

The features that *can* separate them are not similarity at all, they are
decisions an LLM already made while reading an episode:

* ``alias_hit`` — some episode's chain contained both surface forms, so a model
  that read the surrounding text asserted they corefer.
* ``cooccurrence`` — some episode contained both surfaces in *different* chains,
  so a model looked at both and kept them apart. Strong negative evidence, and
  exactly the signal cosine gets backwards: two people at one company co-occur
  constantly, which drives their embeddings together while proving they differ.

Those signals come free with chains — no extra model calls — because level 1
already produced them per episode. This module aggregates them corpus-wide and
combines everything into one calibrated probability.
"""

from __future__ import annotations

import difflib
import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

FEATURE_NAMES = (
    "name_cosine",
    "string_similarity",
    "alias_hit",
    "cooccurrence",
    "context_gated",
)
# Dropped after measurement, not on principle:
#   type_agreement  — zero variance in a single-type corpus, so it shifts every
#                     score by a constant and discriminates nothing.

# Gate above which the name features stop carrying information and the context
# vector is allowed to speak. Below it, `name_cosine` already separates at AUC
# 0.936 and context only adds noise.
CONTEXT_GATE = 0.90

# Prior weights, used until you fit on labelled pairs. Chosen to reproduce the
# worked example: an alias hit carries a merge on its own, a co-occurrence
# blocks one, and the embedding contributes far less than its magnitude suggests.
PRIOR_WEIGHTS: dict[str, float] = {
    "name_cosine": 2.0,
    "string_similarity": 1.5,
    "alias_hit": 3.5,
    "cooccurrence": -1.2,
    "context_gated": 1.0,
}
PRIOR_INTERCEPT = -3.5  # absorbs the constant that type_agreement used to add

# Below this many chains per neighbour, blocking is exhaustive instead of ANN.
# With k neighbours each, ANN yields ~n*k/2 pairs against n*n/2 exhaustive, so
# the saving is k/n -- nothing until n greatly exceeds k. At 55 chains and k=50
# it cut 1485 pairs to 1364 (8%) while asking a 55-item index for 51 neighbours,
# which is the regime where an approximate index is least reliable: the last two
# neighbours of 16 rows differed between runs on identical input, *with* a fixed
# seed. Those pairs then score, possibly merge, and reshuffle entity ids for no
# reason anything upstream can see. Exhaustive is exact and deterministic; ANN
# is an optimisation and should only run where it is actually optimising.
ANN_MIN_FACTOR = 5


class ChainRecord(NamedTuple):
    """One chain, as level 1 and level 2 leave it."""

    chain_id: str
    episode_id: str
    head: str
    surfaces: frozenset[str]
    candidate_type: str
    name_vector: np.ndarray | None = None
    context_vector: np.ndarray | None = None
    key: str | None = None  # what a connector says this chain is (a qualified name)


def keys_settle(a: ChainRecord, b: ChainRecord) -> bool:
    """Is this pair decided by connector keys, so the reviewer is never asked?

    A seeded chain carries the qualified name its connector resolved. Two
    chains with different keys are two things, and a chain with a key beside
    one with none is a symbol beside prose. Both were sent to the reviewer
    before this rule existed and it merged them: measured on the author's
    vault plus this repository, 903 review merges joined distinct symbols
    (``PairScorer`` with ``PairScorer.fit`` and ``PairScorer.__init__``) and
    288 joined a symbol to a prose chain ("Entity Linking" with
    ``entity_linking.4_link.Evidence``), gluing 853 chains into one entity.
    The reviewer decides what the passages leave uncertain; a key is not
    uncertain. A pair the scorer merges on its own still merges -- an alias
    such as ``episodes.empty_store`` for ``1_episodes.empty_store`` is real
    evidence that no key contradicts.
    """
    if a.key is None and b.key is None:
        return False
    return a.key != b.key


class Evidence(NamedTuple):
    """Corpus-wide coreference decisions, indexed for lookup.

    ``same_chain`` holds unordered surface pairs some episode put in one chain.
    ``split_in_episode`` holds pairs some episode kept in separate chains.

    Surfaces are not entity identifiers, and treating them as such makes the
    alias signal transitively unsafe: if one episode reveals "Sarah Chen" is
    "S. Chen" and another reveals "Steven Chen" is "S. Chen", the shared surface
    would otherwise link the two Sarahs. ``build_evidence`` therefore lets
    co-occurrence veto a contradicted alias — an episode that saw both surfaces
    and kept them apart outranks an alias inferred through a third string.
    """

    same_chain: set[tuple[str, str]]
    split_in_episode: set[tuple[str, str]]
    ambiguous_surfaces: frozenset[str] = frozenset()


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_evidence(chains: Iterable[ChainRecord]) -> Evidence:
    """Aggregate every per-episode coreference decision into pair-level evidence."""
    by_episode: dict[str, list[ChainRecord]] = defaultdict(list)
    same: set[tuple[str, str]] = set()
    for chain in chains:
        by_episode[chain.episode_id].append(chain)
        surfaces = sorted(chain.surfaces)
        for i, a in enumerate(surfaces):
            for b in surfaces[i + 1:]:
                same.add(_key(a, b))

    split: set[tuple[str, str]] = set()
    for episode_chains in by_episode.values():
        for i, one in enumerate(episode_chains):
            for other in episode_chains[i + 1:]:
                for a in one.surfaces:
                    for b in other.surfaces:
                        if a != b:
                            split.add(_key(a, b))

    # A short form can be claimed by two different entities: one episode reveals
    # "Sarah Chen" is "S. Chen", another reveals "Steven Chen" is "S. Chen", and
    # a third keeps the two Sarahs apart. No single alias pair is contradicted —
    # the contradiction is transitive, through the shared surface — so a direct
    # ``same - split`` misses it entirely. Drop every alias edge on a surface
    # that bridges two surfaces some episode kept apart: it cannot be trusted to
    # link anything, and a bad merge is far more costly than a missed one.
    partners: dict[str, set[str]] = defaultdict(set)
    for a, b in same:
        partners[a].add(b)
        partners[b].add(a)
    ambiguous = {
        surface for surface, linked in partners.items()
        if any(_key(x, y) in split
               for x, y in itertools.combinations(sorted(linked), 2))
    }
    same = {pair for pair in same if not (set(pair) & ambiguous)}
    # Dropping the edges is not enough: those pairs then fall to the reviewer,
    # which sees two plausible passages and merges them. The set is exported so
    # the reviewer can be told the short form is contested.
    return Evidence(same_chain=same, split_in_episode=split,
                    ambiguous_surfaces=frozenset(ambiguous))


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denom if denom else 0.0


def _string_similarity(a: ChainRecord, b: ChainRecord,
                      ambiguous: frozenset[str] = frozenset()) -> float:
    """Best match over surface forms, not just the heads.

    A chain's head is one of several ways it was written; "S. Chen" may be a
    member rather than the head, and comparing heads alone would miss it.

    ``ambiguous`` surfaces are excluded from the comparison. Dropping the alias
    edge is not enough on its own: a contested short form still matches itself at
    1.0, so the similarity features re-derive the merge the veto just refused —
    measured fusing Sarah Chen, Steven Chen and S. Chen into one entity *after*
    the veto had correctly fired. A surface that cannot identify anything must not
    identify anything through the back door either.

    Returns 0.0 when every surface on one side is contested: there is nothing
    trustworthy left to match on, which is the honest answer and leaves the chain
    a singleton for ``unresolved`` to route.
    """
    best = 0.0
    for x in a.surfaces:
        if x in ambiguous:
            continue
        for y in b.surfaces:
            if y in ambiguous:
                continue
            ratio = difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()
            best = max(best, ratio)
    return best


def context_gated(a: ChainRecord, b: ChainRecord) -> float:
    """Context similarity, but only where the name features have nothing to say.

    On pairs whose surfaces are the *same string* — one short form denoting two
    people — ``name_cosine`` is ~1.0 and ``string_similarity`` is exactly 1.0, so
    every such pair scores identically and merges or splits as an undifferentiated
    block. That is the mechanism behind "S. Chen" fusing two people.

    Measured on ECB+, 2466 same-string cross-document pairs: name + string score
    AUC 0.500 there (no information at all), and adding this feature takes the
    combined scorer to 0.690. Over *all* pairs the same signal is worth nothing —
    ``name_cosine`` is already at 0.936 — which is why it is gated rather than
    scored everywhere.

    Uncentered on purpose. The earlier "context is chance" reading (AUC 0.505)
    measured the episode-*centered* vector; uncentered, the same corpus gives
    0.898. Centering removes the episode's shared component, and enough of the
    entity's own signal with it.

    Returns 0.0 — the neutral value, not a low similarity — when the gate is
    closed or either vector is missing, so an absent context vector never pushes
    a pair toward splitting.
    """
    if _string_similarity(a, b) < CONTEXT_GATE:
        return 0.0
    if a.context_vector is None or b.context_vector is None:
        return 0.0
    return _cosine(a.context_vector, b.context_vector)


def features(a: ChainRecord, b: ChainRecord, evidence: Evidence) -> dict[str, float]:
    """The five numbers scored for one candidate pair."""
    alias = any(_key(x, y) in evidence.same_chain
                for x in a.surfaces for y in b.surfaces if x != y)
    split = any(_key(x, y) in evidence.split_in_episode
                for x in a.surfaces for y in b.surfaces if x != y)
    return {
        "name_cosine": _cosine(a.name_vector, b.name_vector),
        "string_similarity": _string_similarity(a, b,
                                                evidence.ambiguous_surfaces),
        "alias_hit": float(alias),
        "cooccurrence": float(split),
        "context_gated": context_gated(a, b),
    }


def candidate_pairs(
    chains: list[ChainRecord], *, min_string: float = 0.35, min_cosine: float = 0.5
) -> list[tuple[ChainRecord, ChainRecord]]:
    """Pairs worth scoring: different episodes, and not obviously unrelated.

    Blocking keeps this out of O(n^2) scoring while staying deliberately loose —
    the whole point is that the decision is made by the evidence features, so the
    blocker must not quietly pre-decide it on similarity.
    """
    out = []
    for i, a in enumerate(chains):
        for b in chains[i + 1:]:
            if a.episode_id == b.episode_id:
                continue
            if (_string_similarity(a, b) >= min_string
                    or _cosine(a.name_vector, b.name_vector) >= min_cosine):
                out.append((a, b))
    return out


def ann_candidate_pairs(
    chains: list[ChainRecord], *, k: int = 50, ef: int = 200
) -> list[tuple[ChainRecord, ChainRecord]]:
    """Blocking by approximate nearest neighbour over the name vectors.

    ``candidate_pairs`` compares everything to everything, which is fine at a few
    hundred chains and impossible at a hundred thousand. Here each chain asks the
    index for its ``k`` nearest names and only those pairs are scored, turning
    O(n^2) into O(n log n).

    The name vector is the right key: it was measured at AUC 0.98+ for identity
    while the context vector reaches 0.745, so neighbours in name space are the
    ones worth examining. ``k`` bounds recall — a true merge whose partner falls
    outside the k nearest can never be scored, so k trades recall for cost and
    should be checked against labelled data rather than guessed.
    """
    import hnswlib

    usable = [c for c in chains if c.name_vector is not None]
    if len(usable) < 2:
        return []
    matrix = np.vstack([c.name_vector for c in usable]).astype("float32")
    index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
    # Seeded: HNSW construction is randomised, and an unseeded index returned
    # a different neighbour set on identical input -- measured, 2 pairs of
    # 1364 differed between runs. Those pairs then score, possibly merge, and
    # reshuffle entity ids for no reason anything upstream can see.
    index.init_index(max_elements=len(usable), ef_construction=ef, M=16,
                     random_seed=100)
    index.add_items(matrix, np.arange(len(usable)))
    index.set_ef(max(ef, k + 1))

    neighbours, _ = index.knn_query(matrix, k=min(k + 1, len(usable)))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[ChainRecord, ChainRecord]] = []
    for i, row in enumerate(neighbours):
        for j in row:
            j = int(j)
            if i == j or usable[i].episode_id == usable[j].episode_id:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            out.append((usable[key[0]], usable[key[1]]))
    return out


class PairScorer:
    """Logistic regression over the five features; a prior until it is fitted.

    Unfitted it uses ``PRIOR_WEIGHTS`` so the pipeline runs from day one. Fitting
    on a few hundred hand-labelled pairs is what makes the output a *calibrated*
    probability rather than a plausible-looking number.
    """

    def __init__(self, weights: dict[str, float] | None = None,
                 intercept: float = PRIOR_INTERCEPT) -> None:
        self.weights = dict(weights or PRIOR_WEIGHTS)
        self.intercept = intercept
        self.fitted = False

    def score(self, feats: dict[str, float]) -> float:
        z = self.intercept + sum(
            self.weights.get(name, 0.0) * feats.get(name, 0.0)
            for name in FEATURE_NAMES
        )
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, rows: list[dict[str, float]], labels: list[int]) -> PairScorer:
        """Fit on hand-labelled pairs. Needs both classes present."""
        from sklearn.linear_model import LogisticRegression

        x = np.array([[r.get(n, 0.0) for n in FEATURE_NAMES] for r in rows])
        y = np.asarray(labels)
        if len(set(y.tolist())) < 2:
            raise ValueError("need both positive and negative labels to fit")
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(x, y)
        self.weights = dict(zip(FEATURE_NAMES, model.coef_[0].tolist()))
        self.intercept = float(model.intercept_[0])
        self.fitted = True
        return self

    def explain(self, feats: dict[str, float]) -> list[tuple[str, float]]:
        """Per-feature contribution to the logit, largest magnitude first."""
        parts = [(n, self.weights.get(n, 0.0) * feats.get(n, 0.0))
                 for n in FEATURE_NAMES]
        return sorted(parts, key=lambda kv: -abs(kv[1]))


def score_pairs(
    chains: list[ChainRecord], scorer: PairScorer | None = None, **blocking
) -> list[tuple[ChainRecord, ChainRecord, float, dict[str, float]]]:
    """Block, extract features, score. Returns pairs sorted most-likely first."""
    scorer = scorer or PairScorer()
    evidence = build_evidence(chains)
    scored = []
    for a, b in candidate_pairs(chains, **blocking):
        feats = features(a, b, evidence)
        scored.append((a, b, scorer.score(feats), feats))
    return sorted(scored, key=lambda row: -row[2])


# ── three-way decision ────────────────────────────────────────────────────────

MERGE, REVIEW, SPLIT = "merge", "review", "split"


class Thresholds(NamedTuple):
    """Where the automatic decisions stop and review begins."""

    low: float   # below this, split without asking
    high: float  # above this, merge without asking

    @property
    def band(self) -> float:
        return self.high - self.low


def decide(probability: float, thresholds: Thresholds) -> str:
    """Merge, split, or hand to an LLM.

    A single threshold forces a decision on pairs the features genuinely cannot
    settle — an ambiguous short form claimed by two people looks the same either
    way. Two thresholds let the scorer abstain, which is the honest answer, and
    routes only the genuinely uncertain pairs to a model.
    """
    if probability >= thresholds.high:
        return MERGE
    if probability < thresholds.low:
        return SPLIT
    return REVIEW


def choose_thresholds(
    probabilities: list[float],
    labels: list[int],
    *,
    merge_precision: float = 1.0,
    split_precision: float = 1.0,
) -> Thresholds:
    """Pick the two cut points from labelled pairs.

    ``high`` is the lowest score at which merging still hits the target
    precision; ``low`` is the highest score at which splitting does. Everything
    between is what neither side can claim — the review band is an output of the
    data, not a knob. A wide band means the features are not separating and more
    labelling or a better feature will pay off more than tuning will.
    """
    scored = sorted(zip(probabilities, labels), key=lambda r: -r[0])
    if not scored:
        return Thresholds(0.0, 1.0)

    # Scan candidate cut points, never splitting a tie: many pairs share a score
    # exactly (an alias hit with identical strings lands them all on the same
    # value), and admitting half a tied block would report a precision the
    # threshold cannot actually deliver.
    def cut(order: list[tuple[float, int]], wanted: int, target: float) -> float:
        seen = correct = 0
        best = None
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and order[j][0] == order[i][0]:
                seen += 1
                correct += int(order[j][1] == wanted)
                j += 1
            if correct / seen >= target:
                best = order[i][0]
            i = j
        return best

    high = cut(scored, 1, merge_precision)
    low = cut(sorted(zip(probabilities, labels), key=lambda r: r[0]), 0,
              split_precision)
    if high is None:
        high = 1.0 + 1e-9  # nothing can be merged automatically
    if low is None:
        low = 0.0          # nothing can be split automatically
    return Thresholds(low=min(low, high), high=max(low, high))


def cluster(
    chain_ids: list[str],
    merges: list[tuple[str, str, float]],
    cannot_link: set[frozenset] | None = None,
) -> dict[str, int]:
    """Group chains by merge decisions. Transitive closure by default.

    ``cannot_link`` defaults to off, and that is a measured choice rather than a
    simplification. Blocking merges across explicit splits scored 0.774 CoNLL
    against 0.945 for plain union-find on the same pairwise decisions — the
    constraints cost 0.17 F1 and produced 45 clusters where 26 were correct.

    The cause is that the decision source contradicts itself: 17 surface pairs
    ("Sarah Chen" ~ "S. Chen" among them) were merged in one episode pairing and
    split in another. A veto lets one dissent overrule every agreeing merge,
    while the closure lets the merges accumulate. Hard constraints are the wrong
    shape for an inconsistent source; a confidence-weighted vote would be the
    principled version if the protection is wanted back.
    """
    cannot_link = cannot_link or set()
    parent = {cid: cid for cid in chain_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members: dict[str, set[str]] = {cid: {cid} for cid in chain_ids}
    for a, b, _ in sorted(merges, key=lambda m: -m[2]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        blocked = any(frozenset((x, y)) in cannot_link
                      for x in members[ra] for y in members[rb])
        if blocked:
            continue
        parent[ra] = rb
        members[rb] |= members[ra]
    roots: dict[str, int] = {}
    return {cid: roots.setdefault(find(cid), len(roots)) for cid in chain_ids}


def unresolved(
    chains: list[ChainRecord], assignment: dict[str, int], evidence: Evidence
) -> dict[str, list[int]]:
    """Chains that should not become entities: contested name, nothing to anchor it.

    A chain whose *only* names are contested surfaces, and which merged with
    nothing, is not a discovered entity — it is a mention we failed to place. Left
    alone it becomes a node like ``["S. Chen"]``: a person who does not exist,
    indistinguishable in the graph from a real one.

    Both conditions are required. A chain that merged with something had its
    identity settled by evidence. A chain with any uncontested surface can carry
    identity on that surface alone.

    Returns ``chain_id -> candidate cluster ids`` — the entities that already use
    the contested surface, i.e. the ones it is plausibly a mention of. A chain
    with no candidates is left as an entity: nothing to attach it to, so it is an
    unlinked mention rather than an ambiguous one.

    Suppressing these breaks the partition that coreference metrics assume, so a
    score computed after this step is not comparable to one computed before it.
    """
    by_cluster: dict[int, list[ChainRecord]] = defaultdict(list)
    for chain in chains:
        by_cluster[assignment[chain.chain_id]].append(chain)

    holders: dict[str, set[int]] = defaultdict(set)
    for cluster, members in by_cluster.items():
        for member in members:
            for surface in member.surfaces:
                holders[surface].add(cluster)

    out: dict[str, list[int]] = {}
    for cluster, members in by_cluster.items():
        # Every member contested, not just a lone one. Two bare "S. Chen"
        # chains merge with each other -- identical heads, so the context gate
        # opens and the reviewer sees two plausible passages -- and the pair
        # then looked "settled by evidence" purely because the cluster had grown
        # to two. Measured on a ten-document corpus: it produced a third
        # S. Chen entity beside Sarah's and Steven's, and nothing was routed to
        # the graph resolver at all. Merging with something equally unanchored
        # is not evidence.
        if not all(m.surfaces <= evidence.ambiguous_surfaces for m in members):
            continue  # somebody here has a name that identifies on its own
        contested = {s for m in members for s in m.surfaces}
        candidates = sorted(
            {c for surface in contested for c in holders[surface]} - {cluster}
        )
        if candidates:
            for member in members:
                out[member.chain_id] = candidates
    return out


def stable_entity_ids(assignment: dict[str, int]) -> dict[str, str]:
    """Name each cluster by the smallest chain id in it. Cheap id stability.

    Cluster labels out of ``cluster`` are positional, so they shuffle whenever
    anything about the clustering changes and every downstream reference breaks.
    The minimum member id is not positional, and that buys most of what event
    sourcing would, for one line:

    * chain ids are **content-derived** (``uuid5`` over episode + surfaces), so a
      cluster whose membership did not change has the same minimum, and its
      entity id survives the re-run untouched.
    * a merge behaves correctly without any event machinery: the merged cluster
      contains both old minima and keeps the global one, so it reads as the other
      entity having been absorbed into the survivor.

    Where it does not hold, and these are exactly the interesting cases:

    * a **split** moves the minimum into one half arbitrarily — the half that
      keeps the id is decided by uuid ordering, not by which half is the
      continuation.
    * an **incremental sync** that adds one chain with a smaller id renames an
      entity whose membership barely changed.

    So: stable whenever nothing interesting happened, no protection when
    something did. It is the 80% version, not the answer — the answer is
    recording merge and split as events.

    Requires chain ids to stay content-derived. A random ``chain_id`` makes the
    minimum arbitrary and this degrades to the positional labels it replaces.
    """
    members: dict[int, list[str]] = defaultdict(list)
    for chain_id, cluster in assignment.items():
        members[cluster].append(chain_id)
    smallest = {cluster: min(group) for cluster, group in members.items()}
    return {chain_id: smallest[cluster] for chain_id, cluster in assignment.items()}
