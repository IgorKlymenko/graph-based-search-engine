"""The walk and the cutoff, without a database."""

import random

import numpy as np

from src.query import retrieve as R


def loop_ppr(edges, seeds, *, alpha, iterations):
    """The pure-Python implementation the sparse walk replaced, kept as the oracle."""
    nodes = sorted({n for pair in edges for n in pair} | set(seeds))
    index = {n: i for i, n in enumerate(nodes)}
    neighbours = [[] for _ in nodes]
    out = np.zeros(len(nodes))
    for (a, b), w in edges.items():
        i, j = index[a], index[b]
        neighbours[i].append((j, w))
        neighbours[j].append((i, w))
        out[i] += w
        out[j] += w
    restart = np.zeros(len(nodes))
    for entity, weight in seeds.items():
        if entity in index:
            restart[index[entity]] = weight
    restart /= restart.sum()
    score = restart.copy()
    for _ in range(iterations):
        nxt = np.zeros(len(nodes))
        for i, row in enumerate(neighbours):
            if not row or score[i] == 0 or out[i] == 0:
                continue
            for j, w in row:
                nxt[j] += score[i] / out[i] * w
        score = alpha * restart + (1 - alpha) * nxt
    return {nodes[i]: float(s) for i, s in enumerate(score) if s > 0}


def test_sparse_walk_matches_the_loop() -> None:
    rng = random.Random(1)
    edges = {}
    while len(edges) < 200:
        a, b = rng.sample(range(60), 2)
        edges[(f"n{min(a, b)}", f"n{max(a, b)}")] = rng.uniform(0.1, 3.0)
    seeds = {"n3": 1.0, "n17": 0.5}
    fast = R.ppr(edges, seeds, alpha=0.5, iterations=30)
    slow = loop_ppr(edges, seeds, alpha=0.5, iterations=30)
    assert set(fast) == set(slow)
    assert all(abs(fast[n] - slow[n]) < 1e-12 for n in slow)


def test_prune_keeps_seeds_and_drops_the_tail() -> None:
    ranked = {"seed": 0.5, "near": 0.2, "far": 0.05, "noise": 0.01}
    assert R.prune(ranked, {"seed": 1.0}, ratio=0.2) == {
        "seed": 0.5,
        "near": 0.2,
        "far": 0.05,
    }
    assert R.prune(ranked, {"seed": 1.0}, ratio=0.0) == ranked


def test_enter_requires_exactly_one_way_in() -> None:
    class Cur:
        def __init__(self):
            self.sql = []

        def execute(self, q, params=None):
            self.sql.append((q, params))

    cur = Cur()
    R.enter(cur, connector="slack", workspace="T1")
    assert cur.sql[0][0] == "SET LOCAL ROLE app_connector"
    assert ("T1",) in [p for _, p in cur.sql]
    cur = Cur()
    R.enter(cur, tenant="t")
    assert cur.sql[0][0] == "SET LOCAL ROLE app_ingest"
    import pytest

    with pytest.raises(ValueError):
        R.enter(Cur(), tenant="t", workspace="T1", connector="slack")
    with pytest.raises(ValueError):
        R.enter(Cur())


def test_lexical_hits_are_word_bounded_case_insensitive_and_longest_first() -> None:
    surfaces = ["ACID", "Postgres", "the Postgres Schema", "ace", "id", "space"]
    got = R.lexical_hits("What are the four ACID guarantees postgres gives?", surfaces)
    assert got == ["Postgres", "ACID"]  # not "ace"/"id" inside words, not "space"


def test_fusion_puts_a_surface_found_both_ways_first() -> None:
    fused = R.fuse([("Neo4j", 0.9), ("ACID", 0.5), ("graph", 0.4)], ["ACID"])
    assert [s for s, _ in fused][:2] == ["ACID", "Neo4j"]


def test_no_lookup_answers_from_the_conversation_without_touching_the_database(
    monkeypatch,
) -> None:
    seen = {}

    def fake_answer(question, passages, **kw):
        seen.update(question=question, passages=passages, **kw)
        return {
            "stance": "answer",
            "answer": "Shorter: use Postgres.",
            "reframe": None,
            "clarify": None,
            "sources": [],
        }

    monkeypatch.setattr(R.AN, "answer", fake_answer)
    monkeypatch.setattr(
        R,
        "embed_texts",
        lambda texts: (_ for _ in ()).throw(AssertionError("no embedding expected")),
    )
    verdict = R.answer_question(
        "Shorten that",
        dsn="postgresql://nowhere",
        connector="slack",
        workspace="T",
        conversation=[("user", "igor", "x")],
        latest="shorter pls",
        needs_lookup=False,
    )
    assert (
        seen["question"] == "Shorten that"
        and seen["passages"] == []
        and seen["latest"] == "shorter pls"
    )
    assert (
        verdict["retrieval"]["lookup"] is False
        and verdict["answer"] == "Shorter: use Postgres."
    )


def test_text_tokens_keep_identifiers_and_their_parts() -> None:
    assert R.text_tokens("keys_settle(a, b) and PairScorer.fit 0.898") == [
        "keys_settle",
        "keys",
        "settle",
        "and",
        "pairscorer.fit",
        "pair",
        "scorer",
        "fit",
        "0.898",
    ]


def test_bm25_prefers_the_rare_term_over_the_common_one() -> None:
    docs = [
        R.text_tokens(d)
        for d in (
            "chains chains chains stall abandoned",
            "chains chains",
            "chains warning",
        )
    ]
    scores = R.bm25(R.text_tokens("stall abandoned chains"), docs)
    assert scores[0] > scores[1] > 0 and scores[0] > scores[2] > 0


def test_fuse_ranks_puts_an_episode_found_both_ways_first() -> None:
    assert R.fuse_ranks(["a", "b", "c"], ["c", "d"]) == ["c", "a", "b", "d"]
    assert R.fuse_ranks(["a"], []) == ["a"]


def test_format_passages_reads_like_a_tool_result() -> None:
    found = R.Retrieval(
        passages=[("earlier/scalability.md", "Neo4j is unscalable.", 0.5, None)],
        contested={"run": ["pkg.jobs.run", "pkg.cli.run"]},
        hits=[],
        edges=0,
        reached=0,
        kept=0,
    )
    out = R.format_passages(found)
    assert out.startswith("[1] scalability (score 0.50)\nNeo4j is unscalable.")
    assert out.endswith(
        "Names that mean more than one thing here: run: pkg.jobs.run, pkg.cli.run"
    )
    assert R.format_passages(found._replace(passages=[], contested={})).startswith(
        "Nothing"
    )
