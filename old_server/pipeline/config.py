"""Central configuration for the extraction pipeline.

Constants only. Secrets stay in the environment (loaded from .env at the CLI
entry point): OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
NEO4J_DATABASE.
"""

from __future__ import annotations

from pathlib import Path

# ── artifact paths (repo root, frozen names read by graph_upload.py) ───────────
ENTITY_CACHE = Path("entity_cache.json")
VECTOR_CACHE = Path("vector_cache.npz")
RESOLVED_CACHE = Path("resolved_entities.json")
ENTITY_VECTOR_CACHE = Path("resolved_entity_vectors.npz")
RELATIONS_CACHE = Path("relations_cache.json")

# ── stage 1: candidate extraction ────────────────────────────────────────────────
EXTRACTION_MODEL = "gpt-4.1-mini"
ENABLE_GLEANING = True
EPISODE_MAX_CHARS = 1000
FILE_WORKERS = 4
EPISODE_WORKERS = 8

# ── stage 2: concept embedding (mxbai span-pool, symmetric) ────────────────────
CANDIDATE_EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_BATCH_SIZE = 16
EMBED_MAX_TOKENS = 512
CHARS_PER_TOKEN = 4
EMBED_SAVE_EVERY = 10  # checkpoint every N batches

# ── stage 3: clustering (HDBSCAN) ──────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = "cosine"

# ── stage 4: LLM entity resolution ─────────────────────────────────────────────
RESOLVE_MODEL = "gpt-4.1-mini"
ESCALATE_MODEL = "gpt-4.1"
RESOLVE_WORKERS = 8

# ── stage 5: entity query embeddings (e5, asymmetric — via services.e5) ────────
ENTITY_QUERY_EMBED_MODEL = "intfloat/e5-base-v2"

# ── stage 6: relation extraction ───────────────────────────────────────────────
RELATION_MODEL = "gpt-4.1-mini"
EXTRACTOR_VERSION = "relation-extraction-v1/gpt-4.1-mini"
RELATION_WORKERS = 8
RELATION_SAVE_EVERY = 50

# ── LLM call defaults ──────────────────────────────────────────────────────────
LLM_TIMEOUT = 90.0
LLM_MAX_RETRIES = 2
