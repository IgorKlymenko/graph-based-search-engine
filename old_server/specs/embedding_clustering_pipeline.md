# Embedding & Entity Clustering Pipeline

> Built iteratively in `embeddings.ipynb` during a single session. This doc captures what was built, why each decision was made, and what to watch out for.

---

## What This Is

A pipeline for extracting named entities from unstructured notes, embedding them with contextual awareness, deduplicating them, and clustering them into semantically meaningful groups — all without specifying the number of clusters upfront.

Primary use case: processing meeting notes / knowledge base documents to build a knowledge graph of entities and their relationships.

---

## Pipeline Overview

```
Document
  → chunk_doc()           # split into ~1000 char chunks
  → extract_entities()    # OpenAI structured output → typed entities
  → embed_span()          # MiniLM/mpnet span pooling per chunk
  → deduplicate()         # union-find: same type + string overlap + embedding sim
  → hdbscan_clustering()  # density-based, no k needed
  → process_new_doc_hdbscan()  # online assignment of new docs
```

---

## Key Components

### 1. Embeddings

**OpenAI `text-embedding-3-small`** — used for static/global semantic similarity (word arithmetic experiments). 1536 dims.

**`sentence-transformers/all-mpnet-base-v2`** — used for contextual span embeddings. 768 dims. Upgraded from `all-MiniLM-L6-v2` for better accuracy (~420MB vs ~90MB).

### 2. Span Pooling

Instead of embedding a word in isolation, the full sentence/chunk is fed through the transformer. Only the token embeddings overlapping with the entity's character span are pooled (mean). This gives a **contextual** vector — "Apple" in a tech context gets a different vector than "apple" in a fruit context.

```python
embed_span(text, span_start, span_end)
```

- Input: full chunk text + character offsets of the entity
- The transformer runs self-attention over the whole chunk first
- Then only the entity's token embeddings are extracted and mean-pooled
- Chunking window: 1024 chars centered on span if text > 2048 chars (to stay under 512 tokens)

**Why not sentence-level embeddings?** Sentence embeddings lose entity focus. Span pooling gives per-entity vectors that still encode surrounding context.

### 3. Entity Extraction (OpenAI Structured Outputs)

```python
class EntityType(str, Enum):
    PERSON, ROLE, ORG, CONCEPT, TOOL, LOCATION, EVENT

class Entity(BaseModel):
    text: str       # exact string as it appears in the source
    type: EntityType
```

Using `client.beta.chat.completions.parse()` with `response_format=Entities`. The enum enforces standardized types — the model cannot hallucinate a new type.

**Important:** entities are extracted per-chunk so each entity is embedded with its local context only.

### 4. Deduplication

Three-gate merge condition — **all three must be true**:
1. Same `EntityType`
2. String overlap > threshold (word Jaccard; containment = 1.0)
3. Cosine similarity > threshold (default 0.85)

Uses union-find. Canonical name = longest string. Vector = mean of merged members.

**Why all three?**
- `COO` vs `CTO`: same type ✓, no string overlap ✗ → kept separate (correct)
- `"Late Seed"` vs `"Late Seed startup"`: same type ✓, containment ✓, high sim ✓ → merged (correct)
- `"apple"` (fruit) vs `"Apple"` (company): different types ✗, low sim ✗ → kept separate (correct)

### 5. Correlation Clustering (Pivot Algorithm)

Simple, fast baseline. No k needed. 3-approximation guarantee.

**How it works (human terms):** Pick a random entity. Pull in everyone similar to it (cosine sim > threshold). Remove them all. Repeat with what's left.

**Consensus clustering on top:** Run pivot N times (it's random), count how often each pair co-occurs in the same cluster. Use co-occurrence frequency as a new similarity score. Run one final pivot pass on those scores. Much more stable than a single run.

```python
correlation_clustering(vectors, threshold=0.6)
consensus_clustering(vectors, threshold=0.7, n_runs=50)
```

### 6. HDBSCAN Clustering

Density-based, hierarchical. Better than pivot for finding clusters of varying density. Explicitly marks outliers as noise (label = -1) rather than forcing every entity into a cluster.

```python
hdbscan_clustering(dedup_vectors, min_cluster_size=2, min_samples=1, metric="cosine")
# returns (clusters, noise)
```

**Key params:**
- `min_cluster_size`: minimum entities to form a cluster
- `min_samples`: how conservative — higher = more noise points
- Outliers in `noise` list are entities that didn't fit anywhere

### 7. Online Assignment (New Documents)

```python
process_new_doc_hdbscan(new_doc, hdb_clusters, dedup_vectors, threshold=0.35)
```

Centroids are **frozen before the doc is processed** — computed once from the original clusters, never updated mid-doc. This prevents the gravity well bug (see below).

Each new entity is assigned to the nearest frozen centroid above threshold, or creates a new singleton cluster.

---

## Bugs & Lessons Learned

### The Gravity Well Bug

**Symptom:** Every entity from a new doc gets pulled into the same cluster, even clearly unrelated ones.

**Cause 1 (original):** New entities created a new cluster immediately, then subsequent entities compared against that new cluster and matched it (because all entities in a doc have similar embedding space from shared context).

**Fix 1:** Freeze cluster count before processing — only compare against `hdb_clusters[:n_original]`.

**Cause 2 (residual):** Centroid was recomputed after each entity was added. CPO shifts Cluster 1's centroid → Notion matches → Confluence matches → etc.

**Fix 2:** Precompute all centroids before the loop. Never update them mid-doc.

```python
centroids = [np.mean([vectors[m] for m in cluster], axis=0) for cluster in hdb_clusters[:n_original]]
# then compare against these frozen centroids throughout
```

---

## Algorithms Compared

| Algorithm | k needed | Handles noise | Stable | Notes |
|---|---|---|---|---|
| Pivot (single run) | No | No | Low | Fast, simple |
| Consensus clustering | No | No | High | Best of pivot runs |
| HDBSCAN | No | Yes | High | Best overall; needs sklearn |

---

## Models Used

| Model | Dims | Size | Use |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | API | Global semantic similarity |
| `all-MiniLM-L6-v2` | 384 | 90MB | Initial span pooling (replaced) |
| `all-mpnet-base-v2` | 768 | 420MB | Span pooling (current) |
| `gpt-4o-mini` | — | API | Entity extraction |

---

## Implicit Context & Design Intentions

- The doc used for testing was a **sales meeting note** (meeting with Tess, a potential client). Entities like `CTO`, `COO`, `SharePoint`, `Tess`, `GraphRAG`, `ZDR` came from that real doc.
- The goal is to build a **knowledge graph** — entities are nodes, clustering informs which nodes are the same concept or closely related.
- **Online processing matters**: new documents should update the entity graph incrementally without rerunning the full pipeline.
- The user explored sentence-level vs span-level embeddings and intuitively understood why span pooling is better for entity-level tasks.
- HDBSCAN was preferred over correlation clustering because it naturally handles outliers — irrelevant entities (e.g. `"Today"`, `"Tmrw"`, `"Screenshot..."`) become noise rather than polluting clusters.
- The user's attention was high on: what goes INTO an embedding, why context matters, centroid drift bugs, and deduplication logic.

---

## Open Questions / Future Work

- **Dedup threshold tuning**: `sim_threshold=0.85` was set conservatively. Nothing merged in the test doc — may need lowering.
- **Online HDBSCAN refit**: The current online approach uses centroid assignment (approximate). A periodic full HDBSCAN refit on accumulated vectors would give cleaner clusters.
- **Cross-doc entity resolution**: Same entity appearing in two docs (e.g. "Tess" and "Tess Smith") — currently only deduplicated within a doc.
- **Mahalanobis distance**: Explored but deferred. Captures cluster shape but needs many members per cluster to be stable with 768-dim vectors.
- **Cluster naming**: Clusters are anonymous lists. Could use the most representative entity or an LLM to generate a label.
