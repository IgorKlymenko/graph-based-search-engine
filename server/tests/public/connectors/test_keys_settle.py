"""A connector key is not uncertainty: keyed pairs never reach the reviewer."""

import numpy as np

from src.inject import chains as CH
from src.inject import main as MAIN
from src.inject import pairing as PR


def rec(cid, head, key=None, episode=None):
    return PR.ChainRecord(chain_id=cid, episode_id=episode or cid, head=head,
                          surfaces=frozenset({head}), candidate_type=PR.PROPER_NAME
                          if hasattr(PR, "PROPER_NAME") else "proper_name",
                          name_vector=np.ones(4) / 2, key=key)


def test_two_keys_or_a_key_beside_prose_are_settled() -> None:
    a = rec("a", "PairScorer", "m.PairScorer")
    b = rec("b", "PairScorer.fit", "m.PairScorer.fit")
    c = rec("c", "the scorer")
    d = rec("d", "PairScorer", "m.PairScorer")
    assert PR.keys_settle(a, b)
    assert PR.keys_settle(a, c)
    assert not PR.keys_settle(a, d)
    assert not PR.keys_settle(c, rec("e", "the scorer"))


def test_link_splits_keyed_pairs_without_asking() -> None:
    asked = []

    def adjudicate(a, b, model, evidence, passages, cache):
        asked.append((a.chain_id, b.chain_id))
        return {"same_entity": True}

    a = rec("a", "PairScorer", "m.PairScorer")
    b = rec("b", "PairScorer.fit", "m.PairScorer.fit")
    result = MAIN.link([a, b], model="stub", k=50,
                       thresholds=PR.Thresholds(low=0.0, high=1.01), workers=1,
                       adjudicate=adjudicate)
    assert asked == []
    assert result["assignment"]["a"] != result["assignment"]["b"]


def test_encode_reads_the_key_from_the_seeds() -> None:
    text = "function fit (m.PairScorer.fit)"
    store = {
        "episodes": {"e1": {"episode_id": "e1", "text": text, "removed_at": None,
                             "seeds": [{"key": "m.PairScorer.fit",
                                        "spans": [[9, 12, "fit"], [14, 30, "m.PairScorer.fit"]]}]}},
        "candidates": {
            "c1": {"candidate_id": "c1", "episode_id": "e1", "start": 9, "end": 12,
                   "text": "fit", "candidate_type": "proper_name",
                   "origin": CH.SEED_ORIGIN, "chain_id": "ch1", "is_head": False},
            "c2": {"candidate_id": "c2", "episode_id": "e1", "start": 14, "end": 30,
                   "text": "m.PairScorer.fit", "candidate_type": "proper_name",
                   "origin": CH.SEED_ORIGIN, "chain_id": "ch1", "is_head": True},
            "c3": {"candidate_id": "c3", "episode_id": "e1", "start": 0, "end": 8,
                   "text": "function", "candidate_type": "nominal",
                   "origin": "parser", "chain_id": "ch2", "is_head": True},
        },
    }
    cache = {"model": "x", "names": {"m.PairScorer.fit": np.ones(4), "function": np.ones(4)},
             "contexts": {}}
    records = {r.chain_id: r for r in MAIN.encode(store, context=False, cache=cache)}
    assert records["ch1"].key == "m.PairScorer.fit"
    assert records["ch2"].key is None
