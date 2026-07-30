"""Stage 3 — HDBSCAN clustering of concept vectors.

Blocking, not matching: clusters are concept groups for stage 4 to adjudicate,
so recall matters here and precision is the LLM's job (see resolution.py).

Density-based clustering with no threshold to tune. Concepts that fit no dense
region are HDBSCAN "noise" (label -1); we treat every noise point as its own
singleton so the resolution stage sees one coherent partition (spec §8.2). The
exploratory consensus correlation-clustering code from the notebook is
intentionally NOT ported.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import (
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
)


def hdbscan_clustering(
    vectors: dict[str, list[float]],
    *,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
) -> tuple[list[list[str]], list[str]]:
    """Cluster ``{concept: vector}``.

    Returns ``(clusters, noise)`` where every input concept appears in exactly
    one multi/size-1 cluster or in ``noise``. ``clusters`` is sorted largest
    first. An empty input yields ``([], [])``.
    """
    from collections import defaultdict

    from sklearn.cluster import HDBSCAN

    concepts = list(vectors)
    if not concepts:
        return [], []

    matrix = np.array([vectors[s] for s in concepts])
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    ).fit_predict(matrix)

    cluster_map: dict[int, list[str]] = defaultdict(list)
    noise: list[str] = []
    for concept, label in zip(concepts, labels):
        if label == -1:
            noise.append(concept)
        else:
            cluster_map[int(label)].append(concept)

    clusters = sorted(cluster_map.values(), key=len, reverse=True)
    return clusters, noise


def partition(
    vectors: dict[str, list[float]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Full partition for resolution: ``(multi_clusters, singletons)``.

    Singletons are size-1 clusters AND noise points from the SAME HDBSCAN run —
    the notebook's bug was drawing singletons from a different (CC) partition and
    dropping HDBSCAN noise entirely.
    """
    clusters, noise = hdbscan_clustering(vectors)
    multi = [c for c in clusters if len(c) > 1]
    singletons = [c for c in clusters if len(c) == 1] + [[n] for n in noise]
    return multi, singletons
