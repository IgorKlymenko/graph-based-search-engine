"""A withheld mention's candidate list must survive the resolver emptying a cluster.

Regression: ``pending`` is computed before ``resolve_against_graph`` runs, and a
placed chain can empty the cluster it came from. Indexing ``entity_of_cluster``
with that stale cluster id raised ``KeyError`` on the first full-vault run and
discarded every review answer of the run.
"""

from src.inject import pairing as PR
from src.inject.main import unresolved_report


def _record(chain_id: str, episode_id: str, *surfaces: str) -> PR.ChainRecord:
    return PR.ChainRecord(
        chain_id=chain_id,
        episode_id=episode_id,
        head=surfaces[0],
        surfaces=frozenset(surfaces),
        candidate_type="proper_name",
    )


def test_a_candidate_cluster_emptied_by_the_resolver_is_skipped_not_raised():
    records = [_record("c1", "e1", "S. Chen"), _record("c2", "e2", "S. Chen")]
    # c1 is still pending; cluster 7 was c2's and is gone because the resolver
    # placed c2 elsewhere, so it no longer maps to an entity. Cluster 3 remains.
    pending = {"c1": [7, 3]}
    entity_of_cluster = {3: "entity-3"}

    report = unresolved_report(records, pending, entity_of_cluster)

    assert report == [
        {
            "chain_id": "c1",
            "episode_id": "e1",
            "surfaces": ["S. Chen"],
            "candidates": ["entity-3"],
            "reason": "contested_surface",
        }
    ]


def test_the_report_is_ordered_by_chain_id_and_empty_when_nothing_is_pending():
    records = [_record("b", "e1", "J. Park"), _record("a", "e2", "J. Park")]
    pending = {"b": [1], "a": [1]}

    report = unresolved_report(records, pending, {1: "entity-1"})

    assert [row["chain_id"] for row in report] == ["a", "b"]
    assert unresolved_report([], {}, {}) == []
