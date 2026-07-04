# Retrieval Library — Part 1: Foundation (Models, Interfaces, Config)

<!-- AGENT CONTEXT — read this first -->
> This file is one of five focused slices of a single implementation spec.
> All five files describe the same **retrieval/** Python library.
> The canonical source is `spec/ppr-temporal-multihop-reimplementation.md`.
>
> **Seven retrieval methods:**
> `ppr` · `fast-ppr` · `treehop` · `beam` · `beam-dual` · `dag` · `sequential-dual`
>
> **Module layout (brief):**
> ```
> retrieval/
> ├── interfaces.py   # GraphDriver, Embedder, LLMClient protocols
> ├── models.py       # Fact, Passage, RankedNode, *Results types
> ├── config.py       # All config dataclasses + enums
> ├── exceptions.py   # RetrievalError hierarchy
> ├── shared.py       # FactMerger, StopCriterion, BridgeGenerator, EmbeddingFuser
> ├── filters.py      # temporal_match() pure function
> ├── graph_loader.py # load_group() → (nodes, edges)
> ├── ppr.py          # Baseline power-iteration PPR
> ├── ppr_approx.py   # FAST-PPR approximate algorithm
> ├── seed.py         # Hybrid seed collection + LLM expansion
> ├── rerank.py       # Embedding / cross-encoder rerank
> ├── temporal.py     # Temporal PPR search + passage hydration
> ├── multihop.py     # Sequential multihop (single + BDTR dual bridge)
> ├── multihop_beam.py# Beam-search multihop
> ├── treehop.py      # Embedding-space hops (no LLM)
> └── dag_multihop.py # DAG query decomposition multihop
> ```
>
> **Key types:** `Fact`, `Passage`, `RankedNode`, `PPRResults`, `TemporalPPRResults`,
> `MultihopHop`, `MultihopSearchResults` — all frozen Pydantic models defined in `models.py`.
>
> **External I/O protocols** in `interfaces.py`: `GraphDriver.execute_query()`,
> `Embedder.embed()`, `LLMClient.complete()` / `complete_json()`.
<!-- END AGENT CONTEXT -->


**This file covers**: purpose, module layout, external interfaces, data models, configuration, custom exceptions, error handling policy, dependencies, and open questions. Implement these files first — every other module depends on them.

---

## 1. Purpose and Scope

This document specifies a standalone implementation of knowledge-graph search features across two tiers:

**Baseline methods:**
- **Personalized PageRank (PPR)** — graph-diffusion ranking seeded from query-relevant entities
- **Temporal PPR search** — bi-temporal filtering layered on top of PPR
- **Multihop search (sequential LLM-bridge)** — LLM writes follow-up queries hop by hop

**Extended methods (added after baseline):**
- **FAST-PPR** — approximate PPR via bidirectional random walk; sublinear in graph size (Option A)
- **TreeHop** — LLM-free multihop via embedding-space query fusion (Option H)
- **Beam-search multihop** — parallel hypothesis exploration across hops (Option I)
- **BDTR dual-bridge** — complementary forward+backward reasoning queries per hop (Option J)
- **DAG multihop** — query decomposed into dependency graph, parallel sub-query resolution (Option K)

The implementation targets a new, independent repository with its own code style, dependencies, and abstractions. It does not import from, extend, or depend on any existing graphiti codebase. All interfaces, models, and helpers are defined here from scratch.

---

## 2. Background and Motivation

### 2.1 What these features do

| Layer | Core idea |
|---|---|
| PPR | Seed from hybrid node search → power-iterate a transition matrix → gate candidates by embedding or cross-encoder relevance |
| Temporal | Run PPR → post-filter facts by a bi-temporal predicate → optionally hydrate source passages |
| Multihop | Run temporal-PPR → LLM writes a follow-up query → repeat until budget or convergence |

The three layers compose: multihop calls temporal, temporal calls PPR, PPR calls graph loading and seed search. Each layer is independently testable.

### 2.2 Design goals

- Clean abstraction boundary between algorithm and storage — graph loading is behind a protocol, not a concrete driver.
- All algorithms are pure functions or thin async wrappers — no global state.
- Unit tests require no database and no live LLM — all external I/O is behind injectable interfaces.
- Temporal filtering is a pure Python predicate over in-memory facts, not a database query, so it is backend-agnostic.
- The PPR power iteration is CPU-bound and must not block an async event loop for large graphs.

### 2.3 Behavioural contract to preserve

- Four temporal modes: `all`, `currently_true`, `as_of`, `changed_since`.
- Bi-temporal fact fields: `valid_at`, `invalid_at`, `created_at`, `expired_at`.
- PPR power iteration: configurable alpha (teleport probability), convergence tolerance, and max iterations.
- Seed selection: BM25 + cosine similarity via Reciprocal Rank Fusion (RRF), with optional LLM expansion (keyword terms + HyDE hypothetical answers).
- Rerank modes: `embedding`, `cross_encoder`, `none`.
- Multihop modes: `dynamic` (LLM decides when to stop) and `fixed` (exact N hops).

---

## 3. Module Layout

```
retrieval/
├── __init__.py
├── exceptions.py        # All custom exception types
├── models.py            # Fact, Passage, RankedNode, all result containers
├── config.py            # Config dataclasses for every search variant
├── interfaces.py        # Protocols: GraphDriver, Embedder, LLMClient
├── shared.py            # Reusable helpers: FactMerger, BridgeGenerator, StopCriterion, EmbeddingFuser
├── filters.py           # Temporal predicate helpers (pure functions, no I/O)
├── graph_loader.py      # Loads nodes + edges from a GraphDriver implementation
├── ppr.py               # Baseline PPR: transition matrix + power iteration
├── ppr_approx.py        # FAST-PPR: bidirectional random walk approximation (Option A)
├── seed.py              # Seed collection: hybrid search + optional LLM expansion
├── rerank.py            # Embedding cosine rerank and cross-encoder LLM rerank
├── temporal.py          # Temporal PPR search: PPR + filter + passage hydration
├── multihop.py          # Sequential multihop: LLM bridge, BDTR dual-bridge (Options J baseline)
├── multihop_beam.py     # Beam-search multihop: parallel hypotheses (Option I)
├── treehop.py           # TreeHop: embedding-space hops, no LLM bridge (Option H)
└── dag_multihop.py      # DAG multihop: decompose query → dependency graph → parallel resolve (Option K)
```

### 3.1 Dependency graph

```
multihop.py / multihop_beam.py / treehop.py / dag_multihop.py
    └── temporal.py
            ├── ppr.py / ppr_approx.py
            │     ├── graph_loader.py   (uses interfaces.GraphDriver)
            │     ├── seed.py           (uses interfaces.Embedder, interfaces.LLMClient)
            │     └── rerank.py         (uses interfaces.Embedder, interfaces.LLMClient)
            └── filters.py
shared.py       ← multihop.py, multihop_beam.py, treehop.py, dag_multihop.py (FactMerger, BridgeGenerator, StopCriterion, EmbeddingFuser)
config.py       ← all modules (read only)
models.py       ← all modules (read only)
exceptions.py   ← all modules (read only)
interfaces.py   ← graph_loader, seed, rerank, all multihop variants
```

---

## 4. External Interfaces (`interfaces.py`)

All external I/O is accessed through these `typing.Protocol` definitions. Implementations are provided by the caller; the retrieval library never instantiates them directly.

### 4.1 GraphDriver

```python
from typing import Any, Protocol

class GraphDriver(Protocol):
    async def execute_query(
        self,
        query: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized database query and return rows as dicts."""
        ...
```

- The `query` string is always a static literal with named `$param` placeholders.
- All variable values are passed via `**params` — never interpolated into the query string.
- The driver is responsible for connection pooling, retry, and serializing Python types (datetime, UUID) into database-native types.

### 4.2 Embedder

```python
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Return a unit-norm embedding vector for the given text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call where the backend supports it."""
        ...
```

- Implementations may call a remote API (OpenAI, Voyage, Cohere) or a local model.
- Vectors are assumed to be L2-normalized. Cosine similarity reduces to dot product.

### 4.3 LLMClient

```python
class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Return the model's text completion for the given prompt."""
        ...

    async def complete_json(
        self,
        prompt: str,
        schema: dict,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Return a JSON-structured completion validated against `schema`."""
        ...
```

---

## 5. Data Models (`models.py`)

All models are immutable Pydantic v2 `BaseModel` instances. Construction raises `ValidationError` on type mismatch or constraint violation, so callers never receive partially-valid objects.

### 5.1 Fact

A single directional relationship between two graph entities, extracted from source text.

```python
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    fact: str                           # Natural-language statement of the relationship
    source_uuid: str                    # UUID of the source entity node
    target_uuid: str                    # UUID of the target entity node
    score: float                        # PPR endpoint mass (0–1)
    rerank_score: float | None = None   # Post-PPR relevance gate score (0–1)
    created_at: datetime                # Wall-clock ingestion time (UTC)
    valid_at: datetime | None = None    # Event time: when fact became true
    invalid_at: datetime | None = None  # Event time: when fact stopped being true
    expired_at: datetime | None = None  # Explicit manual retraction time
    episodes: list[str] = Field(default_factory=list)  # Source passage UUIDs

    @field_validator('score', 'rerank_score', mode='before')
    @classmethod
    def clamp_score(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))
```

### 5.2 Passage

A source document excerpt that one or more facts were extracted from.

```python
class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    name: str
    content: str
    valid_at: datetime | None = None
    source_description: str = ''
    supports_fact_uuids: list[str] = Field(default_factory=list)
    best_relevance: float | None = None
```

### 5.3 RankedNode

An entity node with its PPR score after graph diffusion.

```python
class RankedNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    name: str
    summary: str
    score: float
```

### 5.4 PPRResults

```python
class PPRResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: list[RankedNode]              # Top-K entity nodes by PPR score
    facts: list[Fact]                    # Candidate facts after rerank gate
    seed_uuids: list[str]                # Entity UUIDs used as PPR seeds
    expansion_terms: list[str] = Field(default_factory=list)
    hyde_answers: list[str] = Field(default_factory=list)
    num_nodes: int                       # Total nodes loaded for this group
    num_edges: int                       # Total edges loaded for this group
```

### 5.5 TemporalPPRResults

```python
class TemporalPPRResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    facts: list[Fact]
    mode: str
    as_of: datetime
    relevant_count: int                  # Facts before temporal filter
    dropped_by_temporal: int = 0         # Facts excluded by temporal predicate
    num_seeds: int
    num_nodes: int
    num_edges: int
    passages: list[Passage] = Field(default_factory=list)
```

### 5.6 MultihopHop

```python
class MultihopHop(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    temporal: TemporalPPRResults
    new_uuids: frozenset[str]            # Fact UUIDs first seen in this hop
```

### 5.7 MultihopSearchResults

```python
class MultihopSearchResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    hops: list[MultihopHop]
    merged: list[Fact]                   # Deduplicated, best score per UUID, sorted descending
    stop_reason: str                     # 'budget' | 'convergence' | 'llm_stop' | 'llm_error' | 'timeout' | 'fixed'
```

---

## 6. Configuration (`config.py`)

All configs use Pydantic v2 with field validators. No mutable defaults (`Field(default_factory=...)` for nested objects).

### 6.1 RerankMode

```python
from enum import Enum

class RerankMode(str, Enum):
    embedding = 'embedding'
    cross_encoder = 'cross_encoder'
    none = 'none'
```

### 6.2 TemporalMode

```python
class TemporalMode(str, Enum):
    all = 'all'
    currently_true = 'currently_true'
    as_of = 'as_of'
    changed_since = 'changed_since'
```

### 6.3 SearchFilters

Optional pre-filter applied to graph queries before PPR to narrow the candidate set.

```python
from pydantic import BaseModel, Field, field_validator
import re

_SAFE_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def _validate_identifiers(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    for v in values:
        if not _SAFE_IDENTIFIER.match(v):
            raise ValueError(
                f'{field_name} must match [A-Za-z_][A-Za-z0-9_]*, got: {v!r}'
            )
    return values

class SearchFilters(BaseModel):
    node_labels: list[str] | None = None    # Restrict to entity nodes with these labels
    edge_types: list[str] | None = None     # Restrict to edges of these types
    valid_at_after: datetime | None = None
    valid_at_before: datetime | None = None
    created_at_after: datetime | None = None
    created_at_before: datetime | None = None
    edge_uuids: list[str] | None = None     # Restrict to specific fact UUIDs

    @field_validator('node_labels')
    @classmethod
    def validate_node_labels(cls, v: list[str] | None) -> list[str] | None:
        return _validate_identifiers(v, 'node_labels')

    @field_validator('edge_types')
    @classmethod
    def validate_edge_types(cls, v: list[str] | None) -> list[str] | None:
        return _validate_identifiers(v, 'edge_types')
```

### 6.4 PPRSearchConfig

```python
class PPRSearchConfig(BaseModel):
    num_seeds: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=0.15, gt=0.0, lt=1.0)
    max_iter: int = Field(default=100, ge=10, le=500)
    tol: float = Field(default=1e-6, gt=0.0)
    top_nodes: int = Field(default=10, ge=1, le=200)
    top_facts: int = Field(default=15, ge=1, le=500)
    candidate_facts: int = Field(default=50, ge=1, le=1000)
    rerank: RerankMode = RerankMode.embedding
    min_relevance: float | None = None   # None = auto-select per rerank mode
    expand_seeds: bool = False
    expansion_terms: int = Field(default=6, ge=0, le=20)
    expansion_model: str = 'gpt-4o-mini'
    max_nodes: int = Field(default=100_000, ge=1)
    max_edges: int = Field(default=500_000, ge=1)
    ppr_timeout_seconds: float = Field(default=10.0, gt=0.0)
```

### 6.5 TemporalPPRSearchConfig

```python
class TemporalPPRSearchConfig(BaseModel):
    mode: TemporalMode = TemporalMode.currently_true
    as_of: datetime | None = None        # None = UTC now at call time
    max_facts: int = Field(default=15, ge=1, le=500)
    hydrate: bool = True
    max_passages: int = Field(default=5, ge=0, le=50)
    ppr: PPRSearchConfig = Field(default_factory=PPRSearchConfig)
```

### 6.6 MultihopSearchConfig

```python
class MultihopSearchConfig(BaseModel):
    dynamic: bool = True
    max_hops: int = Field(default=4, ge=1, le=20)
    fixed_hops: int = Field(default=2, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    bridge_model: str = 'gpt-4o-mini'
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=120.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
```

---

## 7. Custom Exceptions (`exceptions.py`)

```python
class RetrievalError(Exception):
    """Base for all retrieval errors."""

class GraphLoadError(RetrievalError):
    """Database query failed during graph loading."""

class GraphTooLargeError(RetrievalError):
    """Group graph exceeds configured node or edge limits."""
    def __init__(self, num_nodes: int, num_edges: int, limits: dict[str, int]) -> None:
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.limits = limits
        super().__init__(
            f'Graph too large: {num_nodes} nodes, {num_edges} edges '
            f'(limits: {limits})'
        )

class LLMCallError(RetrievalError):
    """LLM API call failed."""

class EmbeddingError(RetrievalError):
    """Embedding API call failed."""

class SearchTimeoutError(RetrievalError):
    """Search exceeded its wall-clock deadline."""

class InvalidInputError(RetrievalError):
    """Caller-supplied input failed validation."""
```

---

## 13. Error Handling Policy

| Scenario | Behavior |
|---|---|
| `group_id` fails validation | `InvalidInputError` raised immediately, no I/O |
| `node_labels` / `edge_types` fail allowlist | `ValidationError` at Pydantic construction, no I/O |
| `query` too long | `InvalidInputError` raised immediately |
| Graph database query fails | `GraphLoadError` raised, original exc chained |
| Graph exceeds size limits | `GraphTooLargeError` raised before matrix build |
| PPR power iteration times out | `SearchTimeoutError` raised |
| Zero seeds after hybrid search | Uniform personalization vector; return top PPR nodes by mass |
| LLM expansion fails | Silent fallback to raw query; log WARNING |
| Cross-encoder LLM call fails | Fall back to `embedding` rerank; log WARNING |
| Bridge question LLM call fails | `stop_reason='llm_error'`; return partial results |
| Bridge question LLM times out | Same as above |
| Bridge output fails validation | Treat as STOP; `stop_reason='llm_stop'` |
| Individual fact datetime parse fails | Skip fact; log DEBUG |
| Individual fact UUID invalid | Skip fact; log WARNING |
| Passage hydration DB call fails | Return result without passages; log WARNING |
| Multihop total timeout | Return partial `MultihopSearchResults` with `stop_reason='timeout'` |

---

## 16. Dependencies

### Required

| Package | Version | Purpose |
|---|---|---|
| `scipy` | `>=1.10` | Sparse CSR matrix for PPR |
| `numpy` | `>=1.24` | Vector operations in power iteration |
| `pydantic` | `>=2.0` | Model validation and serialization |
| `openai` | `>=1.0` | Default LLM client implementation (async) |

### Optional

| Package | Version | Purpose |
|---|---|---|
| `sentence-transformers` | `>=2.0` | Local embedding implementation |
| `opentelemetry-api` | `>=1.0` | Tracing (no-op if absent) |
| `prometheus_client` | `>=0.17` | Metrics (no-op if absent) |
| `psutil` | `>=5.9` | Memory estimation before matrix build |
| `hypothesis` | `>=6.0` | Property-based tests (test-only) |
| `pytest-benchmark` | `>=4.0` | Performance benchmarks (test-only) |
| `pytest-asyncio` | `>=0.23` | Async test support (test-only) |

No graph database driver is a direct dependency of the retrieval library. Callers supply a `GraphDriver` implementation.

---

## 17. Open Questions

1. **Target database**: the graph queries in this spec use Neo4j Cypher syntax as a reference. The actual target database for the reimplementation must be specified. If it is not Neo4j (e.g. Memgraph, FalkorDB, ArangoDB, or a vector-graph hybrid), the queries in `graph_loader.py` and `seed.py` need to be adapted. The protocol-based `GraphDriver` interface isolates this change to one file.

2. **Embedding storage on facts**: the embedding reranker assumes a pre-computed `fact_embedding` vector is stored on each relationship in the graph and returned alongside `uuid`, `fact`, `valid_at`, etc. Confirm this is populated at ingestion time. If not, the reranker must call `embedder.embed(fact.fact)` at query time (latency and cost impact).

3. **Fulltext index**: hybrid seed search depends on a fulltext index over entity names and summaries. Confirm this index is created as part of the graph schema setup. The spec assumes the index name `entity_fulltext`.

4. **Multi-group semantics**: the public API may need to support `group_ids: list[str]`. If so, specify the merge strategy across groups: run separate PPR per group (separate matrices), then merge fact lists by score; or build one combined matrix across groups (requires cross-group edges). This spec assumes the former.

5. **True cross-encoder vs LLM yes/no**: the current cross-encoder rerank uses a yes/no LLM prompt (one call per fact). A true cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers) would be faster and cheaper. Consider a fourth `RerankMode.cross_encoder_model` option backed by a local model.

6. **Seed failure**: if hybrid search returns zero results for all queries (entity names do not match anything), the personalization vector degenerates to uniform. The spec returns the top-K nodes by PPR mass, which equals the highest-degree nodes. Decide whether this is useful or whether zero seeds should return an empty result.

7. **Code style**: this spec uses Python 3.11+ syntax (`match/case`, `str | None`). Confirm the target repository's minimum Python version before implementation.

