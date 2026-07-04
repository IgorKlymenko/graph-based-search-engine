"""Stage 5 — per-canonical-entity vectors for retrieval.

Two vectors per canonical entity, saved to ``resolved_entity_vectors.npz``:

- ``cluster_matrix`` — mean of the member surfaces' mxbai span vectors (symmetric,
  entity-vs-entity), zeros when no member was embedded.
- ``query_matrix`` — ``services.e5`` "passage: " embedding of the canonical name,
  paired at query time with a "query: " embedding of the user's question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.config import ENTITY_VECTOR_CACHE


def run(
    entities: list[dict],
    surface_vectors: dict[str, np.ndarray],
    *,
    cache_path: Path = ENTITY_VECTOR_CACHE,
) -> None:
    """Write cluster + e5 query matrices keyed by canonical name."""
    from services.e5 import embed_e5_sync

    if not entities:
        print("[5/6] entity vectors: no entities — nothing to write")
        return

    canonicals = [e["canonical"] for e in entities]

    cluster_dim = (
        next(iter(surface_vectors.values())).shape[0] if surface_vectors else 0
    )
    cluster_matrix = np.zeros((len(entities), cluster_dim), dtype=np.float32)
    for i, e in enumerate(entities):
        member_vecs = [
            surface_vectors[m] for m in e["members"] if m in surface_vectors
        ]
        if member_vecs:
            cluster_matrix[i] = np.mean(member_vecs, axis=0)

    query_matrix = np.array(
        embed_e5_sync(canonicals, is_query=False), dtype=np.float32
    )

    np.savez(
        cache_path,
        canonicals=np.array(canonicals),
        cluster_matrix=cluster_matrix,
        query_matrix=query_matrix,
    )
    print(f"[5/6] entity vectors: {len(canonicals)} entities -> {cache_path}  "
          f"(cluster {cluster_matrix.shape}, query {query_matrix.shape})")
