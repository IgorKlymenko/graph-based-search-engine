# PPR + Temporal Search + Multihop Search — Reimplementation Spec

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

## 8. Algorithm Specifications

### 8.1 Graph Loader (`graph_loader.py`)

```python
from typing import NamedTuple
from retrieval.interfaces import GraphDriver
from retrieval.exceptions import GraphLoadError, GraphTooLargeError
from retrieval.config import PPRSearchConfig

class GraphNode(NamedTuple):
    uuid: str
    name: str
    summary: str

class GraphEdge(NamedTuple):
    source_uuid: str
    target_uuid: str

async def load_group(
    driver: GraphDriver,
    group_id: str,
    config: PPRSearchConfig,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    ...
```

Expected queries (Neo4j Cypher shown as reference; adapt to target database):

```cypher
MATCH (n:Entity {group_id: $group_id})
RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary
```

```cypher
MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]->(m:Entity {group_id: $group_id})
RETURN e.source_node_uuid AS source_uuid, e.target_node_uuid AS target_uuid
```

**Size guard** (before returning):
```python
if len(nodes) > config.max_nodes or len(edges) > config.max_edges:
    raise GraphTooLargeError(
        num_nodes=len(nodes),
        num_edges=len(edges),
        limits={'max_nodes': config.max_nodes, 'max_edges': config.max_edges},
    )
```

All query exceptions from the driver must be caught and re-raised as `GraphLoadError` with a generic message. The original exception is chained (`raise GraphLoadError(...) from original_exc`) for debugging but must not expose Cypher text or driver internals.

### 8.2 PPR Algorithm (`ppr.py`)

#### 8.2.1 Transition matrix construction

1. Collect all unique node UUIDs. Sort lexicographically and assign integer indices `idx: dict[str, int]`. Sorting is required for determinism — the same graph must produce the same matrix across calls.
2. Allocate a `(N, N)` COO sparse matrix. For each undirected edge `(u, v)`, add weight 1 at `(idx[v], idx[u])` and `(idx[u], idx[v])`.
3. Convert to CSR and column-normalize: divide each column by its column sum. Columns with zero sum (dangling nodes) remain zero.

#### 8.2.2 Personalization vector

```python
s = np.zeros(N, dtype=np.float64)
for uuid, weight in seed_weights.items():
    if uuid in idx:
        s[idx[uuid]] = weight
if s.sum() == 0:
    s[:] = 1.0 / N    # uniform fallback when no seeds match loaded nodes
else:
    s /= s.sum()       # L1-normalize
```

#### 8.2.3 Power iteration

```python
p = s.copy()
for _ in range(config.max_iter):
    p_new = (1.0 - config.alpha) * M @ p + config.alpha * s
    # Dangling mass redistribution (uniform, not back to seeds)
    leaked = 1.0 - p_new.sum()
    if abs(leaked) > 1e-10:
        p_new += leaked / N
    delta = np.abs(p_new - p).sum()
    p = p_new
    if delta < config.tol:
        break
```

**Event loop note**: If `N > 10_000`, run the iteration in a thread pool executor:
```python
loop = asyncio.get_event_loop()
p = await loop.run_in_executor(None, _run_power_iteration, M, s, config)
```

Wrap the executor call in `asyncio.wait_for(..., timeout=config.ppr_timeout_seconds)`. On `asyncio.TimeoutError`, raise `SearchTimeoutError`.

#### 8.2.4 Empty graph

If `len(nodes) == 0`, return `PPRResults` immediately with all list fields empty and `num_nodes=0, num_edges=0`.

### 8.3 Seed Collection (`seed.py`)

```python
async def collect_seeds(
    driver: GraphDriver,
    query: str,
    group_id: str,
    config: PPRSearchConfig,
    embedder: Embedder,
    llm_client: LLMClient | None = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    """
    Returns (seed_weights, expansion_terms, hyde_answers).
    seed_weights: {node_uuid: unnormalized weight}
    """
```

**Step 1 — raw query seeds**:
Run hybrid node search (BM25 fulltext + embedding cosine, fused via RRF) on `query`. For each result at rank `r` (0-indexed): `weights[uuid] += 1.0 / (r + 1)`.

**Step 2 — LLM expansion** (only if `config.expand_seeds` and `llm_client is not None`):
Call `_expand_query_llm(query, config, llm_client)` which returns `(terms: list[str], hyde: list[str])`.
For each term and each HyDE answer, run hybrid search, accumulate weights with the same RRF formula.

**Step 3 — normalize**:
Divide all weights by their sum. Return the normalized dict.

**Hybrid node search** runs these two queries and fuses results:

```cypher
// BM25 fulltext (index must exist on Entity.name, Entity.summary)
CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid AS uuid, node.name AS name, score
LIMIT $limit
```

```cypher
// Cosine similarity via vector index
MATCH (n:Entity {group_id: $group_id})
WITH n, vector.similarity.cosine(n.name_embedding, $embedding) AS score
ORDER BY score DESC
LIMIT $limit
RETURN n.uuid AS uuid, n.name AS name, score
```

RRF fusion: final_score(uuid) = Σ `1 / (60 + rank_in_list)` across lists.

**LLM expansion prompt**:
```
<instructions>
Given the following search query, generate {n} keyword search terms and {n} hypothetical
one-sentence answers that would help retrieve relevant knowledge from a graph.
Return a JSON object with exactly two keys: "terms" (list of strings) and "hyde" (list of strings).
Do not include instructions or explanations in the output — only the JSON object.
</instructions>
<user_query>
{query}
</user_query>
```

Validate the JSON response against schema `{"terms": [str], "hyde": [str]}`. Discard terms or answers longer than 200 characters. Cap list lengths at `config.expansion_terms`. On any failure (API error, parse error, schema mismatch), log a WARNING and return `([], [])`.

### 8.4 Reranking (`rerank.py`)

```python
async def rerank_facts(
    query: str,
    facts: list[Fact],
    mode: RerankMode,
    embedder: Embedder,
    llm_client: LLMClient | None,
    min_relevance: float | None,
) -> list[Fact]:
    """Return facts with rerank_score set, filtered by min_relevance, sorted descending."""
```

**embedding mode**:
```python
query_emb = await embedder.embed(query)
# fact_embedding is fetched from the database alongside the fact
scored = [(f, dot(query_emb, f._embedding)) for f in facts]
```

The fact embedding is stored in the database and returned by the edge query. It is not part of the `Fact` model (which is the public result type) but is held in an internal `_ScoredFact` intermediate struct during reranking, then discarded.

Default `min_relevance` for embedding mode: **0.30**.

**cross_encoder mode**:
One LLM call per candidate, batched 20 at a time concurrently:

```
<instructions>
Does the following fact directly answer or strongly support the query?
Respond with exactly YES or NO — no other text.
</instructions>
<query>{query}</query>
<fact>{fact.fact}</fact>
```

Parse: `YES` → 1.0, `NO` → 0.0, anything else → 0.0. Default `min_relevance`: **0.50**.

**none mode**: Use raw PPR `score` as `rerank_score`. Default `min_relevance`: **0.0**.

### 8.5 Temporal Filtering (`filters.py`)

Pure functions; no I/O.

```python
from datetime import datetime, timezone
from retrieval.models import Fact
from retrieval.config import TemporalMode

def make_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

def temporal_match(fact: Fact, mode: TemporalMode, as_of: datetime) -> bool:
    v = make_aware(fact.valid_at)
    i = make_aware(fact.invalid_at)
    t = make_aware(as_of)   # caller guarantees non-None

    match mode:
        case TemporalMode.all:
            return True
        case TemporalMode.currently_true:
            return i is None
        case TemporalMode.as_of:
            return (v is None or v <= t) and (i is None or i > t)
        case TemporalMode.changed_since:
            return i is not None and i > t
```

**Edge cases**:
- `valid_at=None` in `as_of` mode → treated as valid from the beginning of time (matches if `invalid_at` condition passes).
- `valid_at == as_of` → inclusive (`v <= t`).
- `invalid_at == as_of` → exclusive (`i > t` is False, so fact is NOT returned).
- Naive datetimes from the database → assumed UTC.

### 8.6 Temporal PPR Search (`temporal.py`)

```python
async def temporal_ppr_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient | None,
    query: str,
    group_id: str,
    config: TemporalPPRSearchConfig,
    filters: SearchFilters | None = None,
) -> TemporalPPRResults:
    as_of = make_aware(config.as_of) or datetime.now(timezone.utc)

    ppr_result = await ppr_search(driver, embedder, llm_client, query, group_id, config.ppr, filters)

    relevant_count = len(ppr_result.facts)
    filtered = [f for f in ppr_result.facts if temporal_match(f, config.mode, as_of)]
    dropped = relevant_count - len(filtered)

    passages: list[Passage] = []
    if config.hydrate and filtered:
        passages = await _hydrate_passages(driver, filtered, config.max_passages)

    return TemporalPPRResults(
        facts=filtered[:config.max_facts],
        mode=config.mode.value,
        as_of=as_of,
        relevant_count=relevant_count,
        dropped_by_temporal=dropped,
        num_seeds=len(ppr_result.seed_uuids),
        num_nodes=ppr_result.num_nodes,
        num_edges=ppr_result.num_edges,
        passages=passages,
    )
```

### 8.7 Passage Hydration

```python
async def _hydrate_passages(
    driver: GraphDriver,
    facts: list[Fact],
    max_passages: int,
) -> list[Passage]:
```

Algorithm:
1. Collect episode UUIDs from `facts[*].episodes`. Preserve order of first appearance.
2. For each episode UUID, compute `best_score = max(f.rerank_score or f.score for f in facts if ep_uuid in f.episodes)`.
3. Sort episode UUIDs by `best_score` descending. Take top `max_passages`.
4. Fetch episode rows from the database:
   ```cypher
   MATCH (e:Episode)
   WHERE e.uuid IN $uuids
   RETURN e.uuid AS uuid, e.name AS name, e.content AS content,
          e.valid_at AS valid_at, e.source_description AS source_description
   ```
5. Build `Passage` objects with `supports_fact_uuids` = UUIDs of all facts that reference each episode.

On any database error during hydration, log WARNING and return `[]`. The caller still receives a valid `TemporalPPRResults` without passages.

### 8.8 Multihop Search (`multihop.py`)

```python
async def multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: MultihopSearchConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
```

Wrap the entire function body in `asyncio.wait_for(..., timeout=config.total_timeout_seconds)`. On `asyncio.TimeoutError`, return the partial `MultihopSearchResults` built so far with `stop_reason='timeout'` rather than raising — partial results are more useful than an error.

```python
merged: dict[str, Fact] = {}
hops: list[MultihopHop] = []
current_query = query
stop_reason = 'budget'
hop_limit = config.fixed_hops if not config.dynamic else config.max_hops

for hop_idx in range(hop_limit):
    temporal_result = await temporal_ppr_search(
        driver, embedder, llm_client, current_query, group_id, config.temporal, filters
    )

    before = set(merged.keys())
    for f in temporal_result.facts:
        prev = merged.get(f.uuid)
        if prev is None or (f.rerank_score or 0.0) > (prev.rerank_score or 0.0):
            merged[f.uuid] = f

    new_uuids = set(merged.keys()) - before
    hops.append(MultihopHop(
        query=current_query,
        temporal=temporal_result,
        new_uuids=frozenset(new_uuids),
    ))

    if hop_idx > 0 and len(new_uuids) < config.min_new_facts:
        stop_reason = 'convergence'
        break

    if hop_idx == hop_limit - 1:
        if not config.dynamic:
            stop_reason = 'fixed'
        break

    bridge = await _bridge_question(
        query=query,
        collected_facts=list(merged.values()),
        model=config.bridge_model,
        dynamic=config.dynamic,
        llm_client=llm_client,
        timeout=config.bridge_timeout_seconds,
    )

    if bridge is None:          # LLM API error
        stop_reason = 'llm_error'
        break
    if bridge == '':            # LLM said STOP
        stop_reason = 'llm_stop'
        break
    current_query = bridge

sorted_facts = sorted(merged.values(), key=lambda f: f.rerank_score or f.score, reverse=True)
return MultihopSearchResults(hops=hops, merged=sorted_facts, stop_reason=stop_reason)
```

#### 8.8.1 Bridge question generation

```python
async def _bridge_question(
    query: str,
    collected_facts: list[Fact],
    model: str,
    dynamic: bool,
    llm_client: LLMClient,
    timeout: float,
) -> str | None:
    """
    Returns:
      str (non-empty)  — the follow-up query
      ''               — LLM indicated search is complete (STOP)
      None             — LLM call failed
    """
```

Prompt template:
```
<instructions>
You are a search assistant retrieving knowledge from a graph database.
Your task is to generate ONE concise follow-up search query that retrieves
information missing from the facts collected so far.

Rules:
- Target entities or relationships not already covered by the collected facts.
- Do NOT restate the original question.
- Output only the follow-up query string — no explanation, no preamble.
{stop_instruction}
</instructions>

<original_question>
{query}
</original_question>

<collected_facts>
{formatted_facts}
</collected_facts>
```

Where `stop_instruction` (only included when `dynamic=True`):
```
- If the collected facts are sufficient to answer the original question, output exactly:
  STOP
```

And `formatted_facts` is:
```
- {fact.fact}  [valid: {fact.valid_at} → {fact.invalid_at or 'present'}]
```
(one line per fact, truncated to 50 facts maximum to avoid context overflow)

Parse:
- If the response (after stripping whitespace) is `STOP` → return `''`.
- If response length > 500 chars → treat as STOP (output validation), return `''`.
- If response is empty → return `''`.
- Otherwise → return the stripped response string.

Wrap in `asyncio.wait_for(..., timeout=timeout)`. On `asyncio.TimeoutError` or `LLMCallError` → return `None`.

---

## 9. Shared Utilities (`shared.py`)

All four multihop variants share the same merge logic, bridge generation, stopping criteria, and embedding fusion. These live in `shared.py` so no variant duplicates them.

### 9.1 FactMerger

Accumulates facts across hops, keeping the best score per UUID.

```python
class FactMerger:
    def __init__(self) -> None:
        self._by_uuid: dict[str, Fact] = {}

    def add(self, facts: Iterable[Fact]) -> frozenset[str]:
        """Add facts; return set of UUIDs that are new (not seen before this call)."""
        before = set(self._by_uuid.keys())
        for f in facts:
            prev = self._by_uuid.get(f.uuid)
            if prev is None or _score(f) > _score(prev):
                self._by_uuid[f.uuid] = f
        return frozenset(self._by_uuid.keys()) - before

    def sorted_facts(self) -> list[Fact]:
        return sorted(self._by_uuid.values(), key=_score, reverse=True)

    @property
    def all_uuids(self) -> frozenset[str]:
        return frozenset(self._by_uuid.keys())

def _score(f: Fact) -> float:
    return f.rerank_score if f.rerank_score is not None else f.score
```

### 9.2 StopCriterion

Encapsulates all stopping logic so every multihop variant applies identical rules.

```python
@dataclass
class StopCriterion:
    min_new_facts: int
    max_hops: int

    def should_stop(
        self,
        hop_idx: int,
        new_uuid_count: int,
        dynamic: bool,
    ) -> tuple[bool, str]:
        """
        Returns (stop: bool, reason: str).
        reason is one of: 'budget', 'convergence'.
        Caller handles 'llm_stop', 'llm_error', 'timeout', 'fixed'.
        """
        if hop_idx >= self.max_hops - 1:
            return True, 'budget'
        if dynamic and hop_idx > 0 and new_uuid_count < self.min_new_facts:
            return True, 'convergence'
        return False, ''
```

### 9.3 BridgeGenerator

Single place for all LLM bridge query generation, shared by `multihop.py`, `multihop_beam.py`, and `dag_multihop.py`.

```python
class BridgeGenerator:
    def __init__(self, llm_client: LLMClient, model: str, timeout: float) -> None:
        self._llm = llm_client
        self._model = model
        self._timeout = timeout

    async def single(
        self,
        original_query: str,
        collected_facts: list[Fact],
        dynamic: bool,
    ) -> str | None:
        """
        Returns: non-empty str = next query, '' = STOP, None = LLM error.
        """
        prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual=False)
        return await self._call(prompt)

    async def dual(
        self,
        original_query: str,
        collected_facts: list[Fact],
        dynamic: bool,
    ) -> tuple[str | None, str | None]:
        """
        BDTR: generate forward and backward bridge queries in parallel.
        Returns (forward_query, backward_query) — each is str | '' | None.
        """
        forward_prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual='forward')
        backward_prompt = _bridge_prompt(original_query, collected_facts, dynamic, dual='backward')
        results = await asyncio.gather(
            self._call(forward_prompt),
            self._call(backward_prompt),
            return_exceptions=True,
        )
        return (
            results[0] if not isinstance(results[0], BaseException) else None,
            results[1] if not isinstance(results[1], BaseException) else None,
        )

    async def decompose(
        self,
        query: str,
        model: str,
        llm_client: LLMClient,
    ) -> list[dict]:
        """
        DAG: decompose query into sub-questions with dependency indices.
        Returns list of {'id': int, 'question': str, 'depends_on': list[int]}.
        Raises LLMCallError on failure.
        """
        ...

    async def _call(self, prompt: str) -> str | None:
        try:
            raw = await asyncio.wait_for(
                self._llm.complete(prompt, model=self._model, temperature=0.0, seed=42),
                timeout=self._timeout,
            )
        except (asyncio.TimeoutError, LLMCallError):
            return None
        return _validate_bridge_output(raw)
```

The `_bridge_prompt` function takes a `dual` argument:
- `dual=False` → standard single-perspective prompt (same as original §8.8.1)
- `dual='forward'` → "Given known facts, what entity or relationship would CONNECT to or EXTEND these toward an answer?"
- `dual='backward'` → "Working backwards from the original question, what PREREQUISITE fact or relationship is missing?"

`_validate_bridge_output(raw: str) -> str`:
- Strip whitespace
- If equals `STOP` (case-insensitive) → return `''`
- If `len > 500` or contains non-printable chars → return `''`
- If empty → return `''`
- Otherwise return stripped string

### 9.4 EmbeddingFuser (for TreeHop)

```python
class EmbeddingFuser:
    """Fuses a query embedding with retrieved fact embeddings to produce the next hop's query vector."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError('alpha must be in (0, 1)')
        self._alpha = alpha

    def fuse(
        self,
        query_emb: list[float],
        fact_embs: list[list[float]],
    ) -> list[float]:
        """
        next = normalize(alpha * query_emb + (1-alpha) * mean(fact_embs))
        Returns unit-norm vector of same dimension.
        """
        if not fact_embs:
            return query_emb
        q = np.array(query_emb, dtype=np.float64)
        F = np.array(fact_embs, dtype=np.float64)
        mean_f = F.mean(axis=0)
        fused = self._alpha * q + (1.0 - self._alpha) * mean_f
        norm = np.linalg.norm(fused)
        if norm < 1e-10:
            return query_emb
        return (fused / norm).tolist()
```

---

## 10. Extended Algorithm Specifications

### 10.1 FAST-PPR — Approximate Personalized PageRank (`ppr_approx.py`)

**Source**: Lofgren et al. 2014 (KDD). Average-case `O(√d / δ)` vs power iteration's `O(edges × max_iter)`.

#### 10.1.1 Algorithm overview

FAST-PPR estimates the PPR score from the seed distribution `s` to each target node using a bidirectional approach:

1. **Forward push** (from seeds): Run a sparse push operation from seed nodes, accumulating approximate PPR mass in a residual vector. Nodes whose residual exceeds `push_threshold` are expanded.
2. **Backward random walk** (from top-K targets): For each of the top-K candidate target nodes, run Monte Carlo random walks *backward* from the target, estimating how much PPR mass reaches it.
3. **Combine**: PPR estimate = forward residual contribution + backward walk contribution.

For retrieval purposes, we only need the top-K scores, not all N scores — this is the key saving vs power iteration.

#### 10.1.2 Config

```python
class FastPPRConfig(BaseModel):
    num_seeds: int = Field(default=10, ge=1, le=100)
    top_k: int = Field(default=50, ge=1, le=500)       # Only estimate PPR for top-K candidates
    push_threshold: float = Field(default=1e-5, gt=0.0) # Forward push expansion threshold
    num_walks: int = Field(default=200, ge=10, le=5000) # Backward Monte Carlo walks per target
    alpha: float = Field(default=0.15, gt=0.0, lt=1.0)
    rerank: RerankMode = RerankMode.embedding
    min_relevance: float | None = None
    max_nodes: int = Field(default=100_000, ge=1)
    max_edges: int = Field(default=500_000, ge=1)
    ppr_timeout_seconds: float = Field(default=10.0, gt=0.0)
```

#### 10.1.3 Forward push

```python
def _forward_push(
    adj: dict[str, list[str]],   # adjacency list: node → out-neighbours
    seeds: dict[str, float],     # seed → weight (L1-normalized)
    alpha: float,
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns (estimates, residuals).
    estimates[v] ≈ PPR(seeds, v) accumulated so far.
    residuals[v] = unprocessed probability mass at v.
    """
    estimates: dict[str, float] = defaultdict(float)
    residuals: dict[str, float] = dict(seeds)
    queue = list(seeds.keys())

    while queue:
        v = queue.pop()
        r = residuals.pop(v, 0.0)
        if r < threshold:
            residuals[v] = r
            continue
        estimates[v] += alpha * r
        push_mass = (1.0 - alpha) * r
        neighbours = adj.get(v, [])
        if not neighbours:
            # dangling: redistribute uniformly (handled in post-step)
            continue
        per_neighbour = push_mass / len(neighbours)
        for u in neighbours:
            prev = residuals.get(u, 0.0)
            residuals[u] = prev + per_neighbour
            if prev < threshold <= residuals[u]:
                queue.append(u)

    return estimates, residuals
```

#### 10.1.4 Backward Monte Carlo walks

```python
def _backward_walks(
    rev_adj: dict[str, list[str]],   # reverse adjacency: node → in-neighbours
    target: str,
    num_walks: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """
    Estimate the probability that a random walk starting at any node reaches `target`.
    Used to compute the backward contribution to PPR(seeds, target).
    """
    hits = 0
    for _ in range(num_walks):
        node = target
        while True:
            if rng.random() < alpha:
                break   # teleport — this walk contributes if it started from a seed
            parents = rev_adj.get(node, [])
            if not parents:
                break
            node = rng.choice(parents)
        # A hit means: the random walk visited `target` on a path from the seeds
        # (simplified counting; full FAST-PPR uses importance sampling)
        hits += 1
    return hits / num_walks
```

**Implementation note**: The full FAST-PPR algorithm uses importance-sampling correction to combine forward and backward estimates. For the retrieval use case, a simplified version — forward push for high-mass nodes + random walks for the top-K candidates by forward estimate — is sufficient and avoids the complexity of the full estimator.

#### 10.1.5 Public interface

```python
async def fast_ppr_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient | None,
    query: str,
    group_id: str,
    config: FastPPRConfig,
    filters: SearchFilters | None = None,
) -> PPRResults:
    """Drop-in replacement for ppr_search(). Same return type."""
```

The rest of the pipeline (seed collection via `seed.py`, reranking via `rerank.py`, fact loading) is identical to baseline PPR. Only the score computation step differs.

**Shared**: uses `seed.collect_seeds`, `rerank.rerank_facts`, `graph_loader.load_group` — no duplication.

---

### 10.2 TreeHop — Embedding-Space Multihop (`treehop.py`)

**Source**: TreeHop, arXiv 2504.20114. 99% lower latency than LLM-bridge methods; no LLM calls between hops.

#### 10.2.1 Config

```python
class TreeHopConfig(BaseModel):
    max_hops: int = Field(default=4, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    fusion_alpha: float = Field(default=0.6, gt=0.0, lt=1.0)  # weight on original query emb
    top_k_facts_for_fusion: int = Field(default=5, ge=1, le=50)  # fact embs to fuse per hop
    total_timeout_seconds: float = Field(default=60.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
```

#### 10.2.2 Algorithm

```python
async def treehop_search(
    driver: GraphDriver,
    embedder: Embedder,
    query: str,
    group_id: str,
    config: TreeHopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    """No LLMClient parameter — TreeHop makes zero LLM calls."""
    merger = FactMerger()
    hops: list[MultihopHop] = []
    stop_criterion = StopCriterion(config.min_new_facts, config.max_hops)
    fuser = EmbeddingFuser(config.fusion_alpha)
    stop_reason = 'budget'

    # Embed original query once; reuse/fuse across hops
    current_emb: list[float] = await embedder.embed(query)

    async def _run():
        nonlocal stop_reason, current_emb
        for hop_idx in range(config.max_hops):
            # Run temporal-PPR with the current (fused) embedding as the query signal
            # Pass embedding directly to seed collection to bypass text re-embedding
            temporal_result = await temporal_ppr_search(
                driver, embedder, None, query, group_id,
                config.temporal, filters,
                _query_embedding_override=current_emb,  # internal parameter
            )

            new_uuids = merger.add(temporal_result.facts)
            hops.append(MultihopHop(
                query=f'[embedding-hop-{hop_idx}]',   # no text query
                temporal=temporal_result,
                new_uuids=new_uuids,
            ))

            stop, reason = stop_criterion.should_stop(hop_idx, len(new_uuids), dynamic=True)
            if stop:
                stop_reason = reason
                break

            # Fuse: take top-K retrieved fact embeddings + original query embedding
            top_facts = merger.sorted_facts()[:config.top_k_facts_for_fusion]
            fact_embs = [f._embedding for f in top_facts if f._embedding is not None]
            if not fact_embs:
                stop_reason = 'convergence'
                break
            current_emb = fuser.fuse(current_emb, fact_embs)

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)
    return MultihopSearchResults(hops=hops, merged=merger.sorted_facts(), stop_reason=stop_reason)
```

**Implementation note**: `_query_embedding_override` is an internal keyword argument on `temporal_ppr_search` that bypasses the `embedder.embed(query)` call in seed collection and uses the supplied vector directly. This avoids re-embedding on every hop while allowing the embedding to evolve.

**Shared**: uses `FactMerger`, `StopCriterion`, `EmbeddingFuser` from `shared.py`, and `temporal_ppr_search` from `temporal.py`. Does not use `BridgeGenerator`.

---

### 10.3 Beam-Search Multihop (`multihop_beam.py`)

**Source**: Think-on-Graph 2.0 (ICLR 2025). Maintains B parallel reasoning hypotheses; prunes to top-B after each hop.

#### 10.3.1 Config

```python
class BeamMultihopConfig(BaseModel):
    beam_width: int = Field(default=3, ge=1, le=10)
    max_hops: int = Field(default=4, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    bridge_model: str = 'gpt-4o-mini'
    bridge_strategy: BridgeStrategy = BridgeStrategy.single
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=180.0, gt=0.0)
    beam_score_fn: str = 'mean_rerank'   # 'mean_rerank' | 'max_rerank' | 'fact_count'
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
```

```python
class BridgeStrategy(str, Enum):
    single = 'single'    # one bridge query per beam member (baseline)
    dual = 'dual'        # BDTR: forward + backward per beam member (Option J)
```

#### 10.3.2 Beam state

```python
@dataclass
class BeamMember:
    query: str
    merger: FactMerger
    hops: list[MultihopHop]
    score: float = 0.0      # beam scoring signal
```

#### 10.3.3 Algorithm

```python
async def beam_multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: BeamMultihopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    bridge_gen = BridgeGenerator(llm_client, config.bridge_model, config.bridge_timeout_seconds)
    stop_criterion = StopCriterion(config.min_new_facts, config.max_hops)
    beam: list[BeamMember] = [BeamMember(query=query, merger=FactMerger(), hops=[])]
    global_merger = FactMerger()
    stop_reason = 'budget'

    async def _run():
        nonlocal stop_reason

        for hop_idx in range(config.max_hops):
            # --- Fan out: run all beam members in parallel ---
            hop_tasks = [
                temporal_ppr_search(driver, embedder, llm_client, m.query, group_id, config.temporal, filters)
                for m in beam
            ]
            hop_results: list[TemporalPPRResults] = await asyncio.gather(*hop_tasks)

            # --- Update each beam member ---
            for member, result in zip(beam, hop_results):
                new_uuids = member.merger.add(result.facts)
                global_merger.add(result.facts)
                member.hops.append(MultihopHop(
                    query=member.query,
                    temporal=result,
                    new_uuids=new_uuids,
                ))
                member.score = _beam_score(member, config.beam_score_fn)

            # --- Prune to top beam_width ---
            beam.sort(key=lambda m: m.score, reverse=True)
            beam[:] = beam[:config.beam_width]

            # --- Check stopping ---
            total_new = len(global_merger.all_uuids)  # rough proxy
            stop, reason = stop_criterion.should_stop(hop_idx, total_new, dynamic=True)
            if stop:
                stop_reason = reason
                break

            if hop_idx == config.max_hops - 1:
                break

            # --- Generate next queries for each beam member in parallel ---
            if config.bridge_strategy == BridgeStrategy.dual:
                bridge_tasks = [
                    bridge_gen.dual(query, m.merger.sorted_facts(), dynamic=True)
                    for m in beam
                ]
                dual_results = await asyncio.gather(*bridge_tasks)
                new_beam: list[BeamMember] = []
                for member, (fwd, bwd) in zip(beam, dual_results):
                    for q in [fwd, bwd]:
                        if q:   # None = error, '' = STOP
                            new_beam.append(BeamMember(
                                query=q,
                                merger=FactMerger(),   # fresh per-member merger
                                hops=[],
                            ))
                # Re-seed each new member's merger from global merged facts
                for m in new_beam:
                    m.merger.add(global_merger.sorted_facts())
                beam = new_beam[:config.beam_width] if new_beam else beam
                if not new_beam:
                    stop_reason = 'llm_stop'
                    break
            else:
                bridge_tasks = [
                    bridge_gen.single(query, m.merger.sorted_facts(), dynamic=True)
                    for m in beam
                ]
                bridge_results: list[str | None] = await asyncio.gather(*bridge_tasks)
                new_beam = []
                for member, bridge in zip(beam, bridge_results):
                    if bridge:
                        new_beam.append(BeamMember(query=bridge, merger=FactMerger(), hops=[]))
                        new_beam[-1].merger.add(global_merger.sorted_facts())
                if not new_beam:
                    stop_reason = 'llm_stop'
                    break
                beam = new_beam[:config.beam_width]

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)

    # Flatten all hops from all beam members for the result
    all_hops = [h for m in beam for h in m.hops]
    return MultihopSearchResults(
        hops=all_hops,
        merged=global_merger.sorted_facts(),
        stop_reason=stop_reason,
    )

def _beam_score(member: BeamMember, fn: str) -> float:
    facts = member.merger.sorted_facts()
    if not facts:
        return 0.0
    if fn == 'mean_rerank':
        scores = [f.rerank_score or f.score for f in facts]
        return sum(scores) / len(scores)
    if fn == 'max_rerank':
        return max(f.rerank_score or f.score for f in facts)
    if fn == 'fact_count':
        return float(len(facts))
    return 0.0
```

**Shared**: uses `FactMerger`, `StopCriterion`, `BridgeGenerator` from `shared.py`, `temporal_ppr_search` from `temporal.py`.

---

### 10.4 BDTR Dual-Bridge for Sequential Multihop

When `bridge_strategy=BridgeStrategy.dual` is set on `MultihopSearchConfig` (the sequential multihop), the existing `multihop_search()` uses `BridgeGenerator.dual()` instead of `BridgeGenerator.single()`. Both bridge queries are searched in parallel; results are merged into the hop's fact set using `FactMerger`.

Add `BridgeStrategy` enum to `config.py`:

```python
class BridgeStrategy(str, Enum):
    single = 'single'   # default: one bridge query per hop
    dual = 'dual'       # BDTR: forward + backward bridge queries per hop
```

Update `MultihopSearchConfig`:

```python
class MultihopSearchConfig(BaseModel):
    dynamic: bool = True
    max_hops: int = Field(default=4, ge=1, le=20)
    fixed_hops: int = Field(default=2, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    bridge_model: str = 'gpt-4o-mini'
    bridge_strategy: BridgeStrategy = BridgeStrategy.single   # NEW
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=120.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
```

When `bridge_strategy=dual`, the sequential loop generates two queries per hop, runs both temporal-PPR searches in parallel, and merges the combined fact set before calling `StopCriterion`. The `new_uuids` in the `MultihopHop` is the union from both searches. `stop_reason` remains `'llm_stop'` if both bridge queries return STOP or error.

---

### 10.5 DAG Query Decomposition Multihop (`dag_multihop.py`)

**Source**: PankRAG / LogicRAG pattern. Decomposes a compound query into a dependency DAG, resolves parallelizable sub-queries concurrently.

#### 10.5.1 Config

```python
class DAGMultihopConfig(BaseModel):
    decompose_model: str = 'gpt-4o-mini'
    decompose_timeout_seconds: float = Field(default=20.0, gt=0.0)
    max_subqueries: int = Field(default=6, ge=1, le=20)
    bridge_model: str = 'gpt-4o-mini'
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=180.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
```

#### 10.5.2 Decomposition prompt

```
<instructions>
Decompose the following question into at most {max_subqueries} sub-questions that together
fully answer it. For each sub-question, specify which earlier sub-questions (by index)
it depends on. Sub-questions with no dependencies can be answered in parallel.
Do not restate the original question as a sub-question.
Return JSON:
{
  "subqueries": [
    {"id": 0, "question": "...", "depends_on": []},
    {"id": 1, "question": "...", "depends_on": [0]},
    ...
  ]
}
</instructions>
<original_question>
{query}
</original_question>
```

Validate the JSON response:
- `subqueries` must be a list of 1–`max_subqueries` items.
- Each `question` must be a non-empty string ≤ 500 chars.
- `depends_on` must reference only IDs with lower index (no cycles).
- If validation fails, fall back to `[{"id": 0, "question": query, "depends_on": []}]` (single-node DAG = sequential).

#### 10.5.3 Algorithm

```python
async def dag_multihop_search(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    config: DAGMultihopConfig,
    filters: SearchFilters | None = None,
) -> MultihopSearchResults:
    bridge_gen = BridgeGenerator(llm_client, config.bridge_model, config.bridge_timeout_seconds)
    global_merger = FactMerger()
    hops: list[MultihopHop] = []
    stop_reason = 'fixed'

    async def _run():
        nonlocal stop_reason

        # Step 1: Decompose query into DAG
        subqueries = await _decompose_query(query, config, llm_client)

        # Step 2: Topological resolution
        results_by_id: dict[int, TemporalPPRResults] = {}

        for layer in _topological_layers(subqueries):
            # All sub-queries in this layer have no unresolved dependencies
            layer_tasks = [
                temporal_ppr_search(
                    driver, embedder, llm_client,
                    _enrich_subquery(sq, results_by_id),   # inject dependency context
                    group_id, config.temporal, filters
                )
                for sq in layer
            ]
            layer_results = await asyncio.gather(*layer_tasks)

            for sq, result in zip(layer, layer_results):
                results_by_id[sq['id']] = result
                new_uuids = global_merger.add(result.facts)
                hops.append(MultihopHop(
                    query=sq['question'],
                    temporal=result,
                    new_uuids=new_uuids,
                ))

        stop_reason = 'fixed'  # DAG always runs to completion (no early stop)

    await asyncio.wait_for(_run(), timeout=config.total_timeout_seconds)
    return MultihopSearchResults(hops=hops, merged=global_merger.sorted_facts(), stop_reason=stop_reason)
```

Helper functions:

```python
def _topological_layers(subqueries: list[dict]) -> list[list[dict]]:
    """
    Group sub-queries into layers where all dependencies in a layer are resolved.
    Each layer can run in parallel.
    """
    resolved: set[int] = set()
    remaining = list(subqueries)
    layers = []
    while remaining:
        ready = [sq for sq in remaining if all(d in resolved for d in sq['depends_on'])]
        if not ready:
            raise ValueError('Circular dependency in sub-query DAG')
        layers.append(ready)
        resolved.update(sq['id'] for sq in ready)
        remaining = [sq for sq in remaining if sq['id'] not in resolved]
    return layers

def _enrich_subquery(sq: dict, results_by_id: dict[int, TemporalPPRResults]) -> str:
    """
    Prepend facts from dependency results to the sub-query string so the retriever
    has context from prior resolved sub-queries.
    Cap context at 10 facts × 200 chars each.
    """
    dep_facts = []
    for dep_id in sq['depends_on']:
        dep_result = results_by_id.get(dep_id)
        if dep_result:
            dep_facts.extend(dep_result.facts[:5])
    if not dep_facts:
        return sq['question']
    context = ' | '.join(f.fact[:200] for f in dep_facts[:10])
    return f"Context: {context}\n\nQuestion: {sq['question']}"
```

**Shared**: uses `FactMerger` from `shared.py`, `BridgeGenerator.decompose()` for the decomposition step, `temporal_ppr_search` from `temporal.py`. No `StopCriterion` needed (DAG always runs to completion).

---

## 11. Security Risks and Mitigations

### 11.1 Graph Query Injection (CRITICAL)

**Risk**: Any string from user input interpolated into a database query can break query structure and allow arbitrary operations — deleting nodes, exfiltrating data, creating malicious relationships.

**Attack vectors**: `group_id`, `node_labels`, `edge_types`, and property names in `SearchFilters`. Also any fact or passage content if it is ever embedded in a query predicate rather than a parameter.

**Why it is non-trivial**: `node_labels` and `edge_types` in graph query languages like Cypher cannot be supplied as parameters — they must be embedded literally in the query string. This is a structural limitation of the query language, not a driver bug.

**Mitigations**:
1. **Parameterize all scalar values** (`group_id`, UUIDs, dates, embedding vectors). Never use f-strings or string concatenation for values that the driver can accept as a parameter.
2. **Allowlist node labels and edge types** at Pydantic model construction time (see `SearchFilters` in §6.3). Pattern `^[A-Za-z_][A-Za-z0-9_]*$` — enforced before any query is constructed. Reject with `InvalidInputError` rather than silently dropping.
3. **Validate `group_id`** at the public API boundary. Reject values containing `` ` ``, `'`, `"`, `;`, `{`, `}`, or whitespace. Pattern: `^[A-Za-z0-9_\-]+$`.
4. **Never construct query strings from fact or passage content**. Fact text, episode content, and LLM outputs are never embedded in queries — they travel only as parameters.
5. **Defense in depth**: the database user account used by `GraphDriver` should have read-only permissions on the entity/episode data and no permission to modify schema or drop indices.

### 11.2 LLM Prompt Injection (HIGH)

**Risk**: User-supplied query text and graph-stored fact content are both included verbatim in LLM prompts. A malicious actor can craft a query or inject content during ingestion that hijacks the LLM into generating harmful bridge queries, leaking graph schema, or triggering denial-of-service via computationally expensive follow-up queries.

**Attack vectors**:
- `query` parameter in `multihop_search()`, `_bridge_question()`, and `collect_seeds()`.
- Fact text returned from the database (originally from user-controlled ingestion).
- Passage content in hydration used as part of an LLM prompt (if added in future).

**Mitigations**:
1. **Structural delimiters**: all user-supplied content and database-sourced content must be wrapped in XML-like tags (`<user_query>`, `<collected_facts>`) that are explicitly identified as untrusted in the system/instruction section of the prompt. Instructions appear separately in `<instructions>` tags.
2. **Input length limits**: enforce `len(query) <= 2000` at the public API boundary. Raise `InvalidInputError` for violations. This limits token consumption and reduces attack surface for injection.
3. **Output validation on bridge questions**: any response > 500 chars is treated as STOP. Non-printable characters trigger STOP. This prevents a successful injection from generating an arbitrarily large or structured malicious query.
4. **Output validation on LLM expansion**: the `complete_json` call enforces a strict schema `{"terms": [str], "hyde": [str]}`. Individual strings > 200 chars are discarded. The list is capped at `config.expansion_terms`.
5. **LLM outputs are never executed as database queries**: the bridge question and expansion terms are used only as search query strings passed to hybrid node search — they become parameter values in subsequent calls, not query structure.
6. **Fact truncation in prompts**: when formatting collected facts for the bridge prompt, cap at 50 facts and truncate each fact string to 500 chars. This limits exposure of graph content to the LLM and bounds token cost.

### 11.3 Data Exfiltration via External APIs (MEDIUM)

**Risk**: Embedding and LLM calls send fact text, passage content, and user queries to third-party remote APIs. This is a data residency and privacy risk when the graph contains PII, trade secrets, or regulated data.

**Mitigations**:
1. **Document the data flow explicitly**: embedding calls (seed search, reranking) and LLM calls (expansion, bridge, cross-encoder) all send text to external APIs. This must appear in the deployment documentation.
2. **Local-only mode**: the `Embedder` and `LLMClient` protocols can be implemented with local models (e.g. `sentence-transformers`, `ollama`). Provide at least a reference local implementation.
3. **Zero-external-call mode**: `rerank=none` + `expand_seeds=False` + `hydrate=False` in the config sends no text to any external API beyond the database. Document this explicitly as the "air-gapped" config.
4. **No logging of query or fact text at INFO level or above**: structured log entries must not include raw query strings or fact content. Use DEBUG level if needed for development. In production, log only UUIDs and scores.

### 11.4 Denial of Service via Large Graph Loading (HIGH)

**Risk**: PPR loads all entity nodes and edges for a `group_id` into Python memory to build a scipy CSR matrix. A group with millions of nodes or edges will exhaust process memory and either OOM-crash or cause extreme GC pressure.

**Mitigations**:
1. **Hard limits in config**: `PPRSearchConfig.max_nodes` and `max_edges` (defaults: 100k and 500k). `load_group` checks these after fetching and raises `GraphTooLargeError` before matrix construction.
2. **Memory estimation before allocation**: estimate CSR memory as `edges * 3 * 8` bytes. If estimate exceeds 80% of available memory (via `psutil.virtual_memory().available`), raise `GraphTooLargeError`. (Make `psutil` an optional dependency; skip the check if unavailable.)
3. **Subgraph fallback** (separate code path, not silent): when the full group is too large, offer a k-hop subgraph PPR rooted at the seeds. The caller opts in explicitly via `PPRSearchConfig(subgraph_fallback=True, subgraph_hops=2)`. This is not an automatic silent fallback.
4. **PPR timeout**: the power iteration runs in a thread pool executor wrapped in `asyncio.wait_for(timeout=config.ppr_timeout_seconds)`. Default: 10 seconds. Raises `SearchTimeoutError`.

### 11.5 Denial of Service via Unbounded Multihop (MEDIUM)

**Risk**: Each hop triggers an LLM bridge call plus a full PPR graph load. At `max_hops=20` with a moderately large graph, a single `multihop_search` call can run for several minutes and consume significant LLM token budget.

**Mitigations**:
1. **Pydantic-enforced cap**: `max_hops: int = Field(default=4, ge=1, le=20)`. Cannot exceed 20 at the type level.
2. **Total wall-clock timeout**: the entire `multihop_search` coroutine is wrapped in `asyncio.wait_for(timeout=config.total_timeout_seconds)`. Default: 120s. On timeout, return partial results with `stop_reason='timeout'` rather than raising — partial results are more useful than an exception.
3. **Per-bridge timeout**: each `_bridge_question` call has its own `asyncio.wait_for(timeout=config.bridge_timeout_seconds)`. Default: 15s. On timeout, treat as STOP (`stop_reason='llm_error'`).
4. **Convergence guardrail**: `min_new_facts=0` disables convergence stopping. Log a WARNING when this is set, as the loop will run to full `max_hops`.
5. **Token budget logging**: log `len(merged)` facts and approximate token count at each hop at DEBUG level. Allows operators to tune `max_hops` per deployment.

### 11.6 Insecure Deserialization of Database Data (LOW-MEDIUM)

**Risk**: Data returned from the graph database is implicitly trusted. A compromised database, misconfigured permissions, or driver bug can return malformed data — negative scores, null UUIDs, invalid datetimes, injection strings — which can cause crashes, incorrect results, or secondary injection.

**Mitigations**:
1. **All database rows go through Pydantic models before use**. The `Fact` model's `clamp_score` validator (§5.1) ensures `score` and `rerank_score` are in `[0.0, 1.0]`. Type annotations ensure UUIDs are strings and datetimes are `datetime`.
2. **Datetime parsing**: all datetime values from the database must pass through a single `parse_db_date(v: Any) -> datetime` helper that handles ISO 8601 strings, Unix timestamps, and Neo4j `DateTime` objects. A parse failure on an individual fact logs DEBUG and skips the fact rather than crashing.
3. **UUID validation**: UUIDs are accepted as strings but validated by regex `^[0-9a-f-]{36}$` in a `field_validator`. Invalid UUIDs from the database are logged at WARNING and the row is skipped.

### 11.7 Information Leakage via Error Messages (LOW)

**Risk**: Exceptions from the database driver, LLM API, or matrix library can expose internal schema details, file paths, API keys, or graph structure if they propagate to the caller unmodified.

**Mitigations**:
1. **Re-raise with generic messages**: all exceptions from `GraphDriver.execute_query` are caught at the `graph_loader.py` boundary and re-raised as `GraphLoadError("Graph query failed")` with `__cause__` set for debugging. The original message is logged at DEBUG only.
2. **LLM exceptions**: all exceptions from `LLMClient.complete` / `complete_json` are caught in `seed.py`, `rerank.py`, and `multihop.py` and re-raised as `LLMCallError("LLM call failed")`.
3. **No raw query text in exceptions**: the Cypher queries used internally are static strings (no user input embedded), so they are safe to include in DEBUG logs but must not appear in ERROR-level messages or in exception messages propagated to callers.
4. **Credentials must never appear in exceptions**: validate in tests by setting `LLM_API_KEY` and `DB_PASSWORD` env vars to sentinel values and asserting that no raised exception or log message contains those sentinels.

### 11.8 Temporal Access Control (`as_of` parameter) (LOW)

**Risk**: The `as_of` parameter lets callers query historical graph states, which may include facts that were intentionally retracted. If this API is exposed over a network, any caller can query any point-in-time state for any `group_id` they can supply.

**Clarification**: `as_of` is an intended feature, not a vulnerability. The risk is in exposure without authorization.

**Mitigations**:
1. The retrieval library does not enforce authorization — this is the responsibility of the API layer that wraps it.
2. Document clearly: any HTTP or RPC endpoint exposing these functions must validate that the caller is authorized to read the requested `group_id` before calling the library.
3. In `changed_since` mode, consider an optional `max_lookback` duration config. Facts invalidated more than `max_lookback` ago can optionally be excluded to prevent unlimited historical archaeology.

### 11.9 Race Conditions in Concurrent PPR Loads (LOW)

**Risk**: If `multihop_search` or `temporal_ppr_search` is called concurrently for the same `group_id`, multiple full graph loads happen simultaneously, potentially overloading the database.

**Mitigation**: Implement a module-level `asyncio.Semaphore` with a configurable limit (default: 4 concurrent graph loads across all group IDs). Wrap the `load_group` call:
```python
_LOAD_SEMAPHORE = asyncio.Semaphore(4)

async with _LOAD_SEMAPHORE:
    nodes, edges = await load_group(driver, group_id, config)
```
For multi-process deployments, this semaphore is per-process; distributed rate-limiting requires an external mechanism (e.g. Redis).

### 11.10 Beam Search Amplifies LLM Token Cost (MEDIUM — beam-specific)

**Risk**: `beam_width=10, max_hops=20` generates 200 LLM bridge calls per request. With `bridge_strategy=dual` this doubles to 400. A single request can consume hundreds of thousands of tokens.

**Mitigations**:
1. Hard cap `beam_width ≤ 10` via Pydantic validator.
2. `total_timeout_seconds` (default 180s) provides a wall-clock backstop.
3. Log cumulative token count at each hop (DEBUG). Expose it in `MultihopSearchResults.token_usage` so callers can alert on overruns.
4. The arena comparison tool (see arena spec) exposes token counts prominently, making cost visible before production use.

### 11.11 DAG Decomposition Prompt Injection (MEDIUM — DAG-specific)

**Risk**: The decomposition LLM call returns a JSON structure with sub-question strings. A malicious query can attempt to inject sub-questions with harmful content or create a deeply nested dependency structure causing `_topological_layers` to iterate excessively.

**Mitigations**:
1. Sub-questions from the LLM are only ever used as retrieval query strings — never executed as graph queries or code.
2. Validate that `len(subqueries) ≤ max_subqueries` and each `question` is ≤ 500 chars.
3. Cycle detection in `_topological_layers`: raise `InvalidInputError` (not crash) if no valid topological ordering exists.
4. Fallback: if decomposition validation fails for any reason, fall back to single-node DAG (original query, no decomposition). Log WARNING.

### 11.12 TreeHop Embedding Drift (LOW — TreeHop-specific)

**Risk**: After several hops of embedding fusion, the query vector drifts far from the original query's semantic space. Later hops retrieve facts unrelated to the original question — a silent correctness degradation.

**Mitigation**:
1. `fusion_alpha` (default 0.6) keeps 60% weight on the original query embedding per hop, bounding drift geometrically: after N hops, original query contributes at least `alpha^N`.
2. Pydantic bounds `(0.5, 0.95)` — values below 0.5 allow drift to dominate within 2-3 hops.
3. Log `cosine_similarity(original_emb, current_emb)` at each hop at DEBUG. If similarity drops below 0.3, log WARNING.

---

## 12. Testing Plan

### 12.1 Unit tests (no database, no live LLM)

All unit tests run without any database or API credentials. LLM and database calls are replaced by simple stubs (not mocks — stubs with deterministic returns). Target: full suite completes in < 30 seconds.

#### `tests/test_filters.py` — temporal predicate logic

| Test | Verifies |
|---|---|
| `test_all_mode_returns_everything` | `TemporalMode.all` returns True for any combination of fields |
| `test_currently_true_with_null_invalid_at` | Returns True when `invalid_at=None` |
| `test_currently_true_with_set_invalid_at` | Returns False when `invalid_at` is any datetime |
| `test_as_of_both_bounds_set` | Returns True only when `valid_at <= as_of < invalid_at` |
| `test_as_of_valid_at_null` | Treats as valid from the beginning of time |
| `test_as_of_boundary_inclusive_valid_at` | `valid_at == as_of` → True |
| `test_as_of_boundary_exclusive_invalid_at` | `invalid_at == as_of` → False |
| `test_changed_since_invalid_at_before_as_of` | Returns False |
| `test_changed_since_invalid_at_after_as_of` | Returns True |
| `test_changed_since_invalid_at_null` | Returns False |
| `test_make_aware_none` | Returns None |
| `test_make_aware_naive` | Attaches UTC timezone |
| `test_make_aware_already_aware` | Returns unchanged |
| `test_make_aware_non_utc_timezone` | Preserves the original timezone |

#### `tests/test_ppr_algorithm.py` — pure matrix math

| Test | Verifies |
|---|---|
| `test_empty_graph_returns_empty_results` | No nodes → empty PPRResults immediately |
| `test_single_node_score_is_one` | One node, seed on it → score = 1.0 |
| `test_two_nodes_mass_flows_to_neighbor` | Node A → B: seeding A gives B non-zero score |
| `test_dangling_node_no_mass_sink` | Dangling node (no out-edges) does not accumulate all mass |
| `test_scores_sum_to_one` | Output vector sums to 1.0 ± 1e-6 after convergence |
| `test_deterministic_index_mapping` | Same node list → same `idx` dict on repeated calls |
| `test_column_stochastic_matrix` | Each column of M sums to ≤ 1.0 |
| `test_convergence_within_max_iter` | 100-node ring graph converges before `max_iter=100` |
| `test_alpha_near_one_stays_near_seed` | `alpha=0.99` → top node is the seed |
| `test_no_seed_matches_graph` | All seeds have unknown UUIDs → uniform distribution |

#### `tests/test_config.py` — Pydantic validation

| Test | Verifies |
|---|---|
| `test_alpha_zero_rejected` | `alpha=0.0` raises `ValidationError` |
| `test_alpha_one_rejected` | `alpha=1.0` raises `ValidationError` |
| `test_num_seeds_zero_rejected` | `num_seeds=0` raises `ValidationError` |
| `test_max_hops_over_limit` | `max_hops=21` raises `ValidationError` |
| `test_node_label_injection` | `node_labels=['x\`) DETACH DELETE n //']` raises `ValidationError` |
| `test_edge_type_injection` | `edge_types=['REL\`]']` raises `ValidationError` |
| `test_node_label_empty_string` | `node_labels=['']` raises `ValidationError` |
| `test_node_label_starts_with_digit` | `node_labels=['1Node']` raises `ValidationError` |
| `test_valid_node_label` | `node_labels=['EntityPerson']` constructs successfully |
| `test_fact_score_clamped_above_one` | `score=2.0` is stored as `1.0` |
| `test_fact_score_clamped_below_zero` | `score=-0.5` is stored as `0.0` |

#### `tests/test_temporal_search.py` — temporal layer with stub PPR

| Test | Verifies |
|---|---|
| `test_stale_facts_are_dropped` | PPR returns current + stale fact; only current in result |
| `test_dropped_by_temporal_count` | `dropped_by_temporal` reflects exact count |
| `test_relevant_count_is_pre_filter` | `relevant_count` equals total PPR facts before filter |
| `test_as_of_defaults_to_now` | `config.as_of=None` → `as_of` is close to `datetime.now(UTC)` |
| `test_max_facts_truncates` | 20 facts pass filter, `max_facts=5` → 5 returned |
| `test_hydrate_false_skips_passage_fetch` | Stub DB not called for episodes when `hydrate=False` |
| `test_mode_forwarded_to_result` | `result.mode` matches `config.mode.value` |

#### `tests/test_multihop.py` — multihop orchestration with stub temporal

| Test | Verifies |
|---|---|
| `test_fixed_hops_exact_count` | `dynamic=False, fixed_hops=3` → exactly 3 hops |
| `test_fixed_stop_reason` | `stop_reason == 'fixed'` |
| `test_dynamic_llm_stop` | Bridge returns `''` on hop 2 → `stop_reason='llm_stop'`, 2 hops |
| `test_dynamic_convergence` | Zero new facts on hop 2 → `stop_reason='convergence'` |
| `test_dynamic_budget_exhausted` | All hops produce new facts, bridge never stops → `stop_reason='budget'` |
| `test_llm_error_stop_reason` | Bridge returns `None` → `stop_reason='llm_error'` |
| `test_merge_best_rerank_score_wins` | Same UUID from two hops → higher `rerank_score` kept |
| `test_merge_sorted_descending` | `merged` list is sorted by `rerank_score` descending |
| `test_new_uuids_disjoint_across_hops` | `hop.new_uuids` is disjoint from all prior hops |
| `test_bridge_called_between_hops` | Bridge stub called `hops - 1` times |
| `test_bridge_output_over_500_chars_is_stop` | 501-char bridge response → `stop_reason='llm_stop'` |
| `test_original_query_on_first_hop` | `hops[0].query == original_query` |

#### `tests/test_security.py` — security boundary checks

| Test | Verifies |
|---|---|
| `test_node_label_cypher_injection_rejected` | Backtick injection in `node_labels` raises `ValidationError` |
| `test_edge_type_injection_rejected` | Semicolon in `edge_types` raises `ValidationError` |
| `test_group_id_injection_rejected` | `group_id='x; DROP ALL'` raises `InvalidInputError` |
| `test_query_too_long_rejected` | Query > 2000 chars raises `InvalidInputError` |
| `test_bridge_output_too_long_treated_as_stop` | > 500 chars → empty string returned |
| `test_bridge_output_nonprintable_treated_as_stop` | `\x00\x01\x02` in bridge output → empty string |
| `test_expansion_term_too_long_discarded` | Term > 200 chars in JSON response → discarded |
| `test_score_clamped_below_zero` | Negative score from DB → stored as 0.0 |
| `test_score_clamped_above_one` | Score > 1.0 from DB → stored as 1.0 |
| `test_error_message_no_api_key` | LLM exception does not propagate API key value |
| `test_error_message_no_db_password` | DB exception does not propagate password value |
| `test_fact_with_invalid_uuid_skipped` | Row with UUID `not-a-uuid` is skipped, not crash |

### 12.2 Integration tests (require database)

These tests require a running graph database instance. They are marked with `_int` in the function name and excluded from the default test run.

| Test | Verifies |
|---|---|
| `test_load_group_returns_nodes_edges_int` | Synthetic group with 5 nodes and 4 edges loads correctly |
| `test_ppr_on_live_graph_int` | Seed UUIDs appear in top-ranked nodes after PPR |
| `test_temporal_currently_true_int` | Two versions of a fact (one invalidated) → only active returned |
| `test_temporal_as_of_historical_int` | `as_of` between two insertions → historical fact returned |
| `test_temporal_changed_since_int` | Only facts invalidated after reference time returned |
| `test_passage_hydration_int` | Facts with episode UUIDs → correct `Passage` objects returned |
| `test_multihop_chain_facts_int` | Graph with A→B→C fact chain → multihop retrieves all three |
| `test_graph_too_large_raises_int` | Group exceeding `max_nodes` raises `GraphTooLargeError` |
| `test_concurrent_ppr_semaphore_int` | 8 concurrent calls → semaphore caps simultaneous loads at 4 |

### 12.3 Property-based tests (Hypothesis)

```python
# tests/test_ppr_properties.py

@given(
    n_nodes=st.integers(min_value=1, max_value=300),
    n_edges=st.integers(min_value=0, max_value=800),
    alpha=st.floats(min_value=0.01, max_value=0.99),
    n_seeds=st.integers(min_value=1, max_value=5),
)
def test_ppr_scores_sum_to_one(n_nodes, n_edges, alpha, n_seeds):
    """PPR output vector sums to 1.0 ± 1e-4 for any valid random graph."""

@given(facts=st.lists(fact_strategy(), min_size=0, max_size=200))
def test_temporal_all_is_identity(facts):
    """mode='all' returns every fact unchanged."""

@given(facts=st.lists(fact_strategy(), min_size=0, max_size=200))
def test_temporal_currently_true_no_invalid_at(facts):
    """mode='currently_true' returns only facts where invalid_at is None."""

@given(hops=st.integers(min_value=1, max_value=20))
def test_multihop_stop_reason_valid(hops):
    """stop_reason is always one of the six valid values."""

@given(score=st.floats(min_value=-100.0, max_value=100.0))
def test_fact_score_always_in_range(score):
    """Fact.score is always in [0.0, 1.0] after construction."""
```

### 12.4 Performance benchmarks

```
tests/bench/
├── bench_ppr.py        # pytest-benchmark tests
├── bench_temporal.py
└── bench_multihop.py
```

Targets (on a single CPU core, in-process, no database):

| Benchmark | Target |
|---|---|
| PPR transition matrix, 1k nodes, 5k edges | < 50ms |
| PPR power iteration, 1k nodes, 5k edges | < 200ms |
| PPR power iteration, 10k nodes, 50k edges | < 2s |
| FAST-PPR top-50, 10k nodes, 50k edges | < 500ms |
| Temporal filter, 10k facts, any mode | < 10ms |
| Multihop sequential overhead, 4 hops (stub I/O) | < 50ms |
| Beam width=3, 4 hops parallel (stub I/O) | < 80ms overhead vs sequential |
| TreeHop 4 hops, embedding fusion only (stub embedder) | < 20ms overhead vs hop-0 |
| DAG: decompose + 3-layer parallel (stub I/O) | < 60ms orchestration overhead |

### 12.5 Unit tests for new methods

#### `tests/retrieval/test_shared.py`

| Test | Verifies |
|---|---|
| `test_merger_new_uuids_on_first_add` | First call returns all added UUIDs as new |
| `test_merger_dedup_keeps_best_score` | Same UUID added twice → higher score kept |
| `test_merger_sorted_descending` | `sorted_facts()` is descending by score |
| `test_merger_new_uuids_incremental` | Second call returns only UUIDs not seen before |
| `test_stop_criterion_budget` | `hop_idx >= max_hops - 1` → stops with `'budget'` |
| `test_stop_criterion_convergence` | `new_count < min_new_facts` and `hop_idx > 0` → `'convergence'` |
| `test_stop_no_trigger_at_hop_zero` | `hop_idx == 0` never triggers convergence regardless of new count |
| `test_bridge_gen_long_output_is_stop` | > 500 chars → returns `''` |
| `test_bridge_gen_stop_keyword_case_insensitive` | `STOP` / `stop` / `Stop` → `''` |
| `test_bridge_gen_nonprintable_is_stop` | Non-printable chars → `''` |
| `test_bridge_gen_llm_timeout_returns_none` | Timeout → `None` |
| `test_bridge_gen_dual_fires_two_concurrent_calls` | Stub records 2 simultaneous LLM calls |
| `test_fuser_output_is_unit_norm` | Output L2 norm = 1.0 ± 1e-6 |
| `test_fuser_empty_facts_returns_query_emb` | No fact embeddings → query embedding unchanged |
| `test_fuser_alpha_zero_raises` | `alpha=0.0` raises `ValueError` |
| `test_fuser_alpha_one_raises` | `alpha=1.0` raises `ValueError` |

#### `tests/retrieval/test_fast_ppr.py`

| Test | Verifies |
|---|---|
| `test_fast_ppr_empty_graph` | Zero nodes → empty `PPRResults` |
| `test_fast_ppr_single_seed_node` | Single node seeded → score ≈ 1.0 |
| `test_fast_ppr_top_k_only` | Only `top_k` candidates scored; rest absent |
| `test_fast_ppr_non_negative_scores` | All scores ≥ 0 |
| `test_fast_ppr_returns_ppr_results_type` | Return type identical to baseline `ppr_search` |
| `test_fast_ppr_dangling_no_mass_sink` | Dangling node does not monopolize all mass |
| `test_forward_push_residuals_below_threshold_not_expanded` | Residual < threshold → left unexpanded |
| `test_backward_walks_count_matches_config` | `num_walks` random walk traces produced per target |

#### `tests/retrieval/test_treehop.py`

| Test | Verifies |
|---|---|
| `test_treehop_zero_llm_calls` | `LLMClient` stub never invoked |
| `test_treehop_hop_labels` | `hop.query == '[embedding-hop-N]'` for all N |
| `test_treehop_runs_to_max_hops` | Runs exactly `max_hops` when new facts always appear |
| `test_treehop_convergence_early_stop` | Zero new facts on hop 2 → stops at hop 2 |
| `test_treehop_fuser_called_each_hop` | `EmbeddingFuser.fuse` called once per hop after hop 0 |
| `test_treehop_merged_sorted` | `merged` descending by score |
| `test_treehop_timeout` | Stub sleeps → `stop_reason='timeout'`, partial results |
| `test_treehop_drift_warning` | Cosine sim drops below 0.3 → WARNING emitted |
| `test_treehop_fusion_alpha_bounds_validated` | `fusion_alpha ≤ 0.5` raises `ValidationError` |

#### `tests/retrieval/test_beam_multihop.py`

| Test | Verifies |
|---|---|
| `test_beam_width_one_matches_sequential_stop_reason` | Width=1 → stop reasons identical to sequential |
| `test_beam_parallel_ppr_calls` | Width=3 → 3 concurrent stubs called per hop |
| `test_beam_pruned_to_width_after_score` | Exactly `beam_width` members after each hop |
| `test_beam_score_mean_rerank` | Higher mean rerank → ranks first |
| `test_beam_dual_two_calls_per_member` | `dual` strategy → 2 bridge calls per member |
| `test_beam_all_stop_triggers_llm_stop` | All bridges return `''` → `'llm_stop'` |
| `test_beam_global_merger_has_all_facts` | All beam members' facts in final `merged` |
| `test_beam_timeout_returns_partial` | Timeout → partial `MultihopSearchResults`, no exception |
| `test_beam_width_gt_10_rejected` | `beam_width=11` raises `ValidationError` |

#### `tests/retrieval/test_bdtr.py`

| Test | Verifies |
|---|---|
| `test_bdtr_two_ppr_searches_per_hop` | `bridge_strategy=dual` → 2 temporal-PPR calls per hop |
| `test_bdtr_new_uuids_is_union` | `hop.new_uuids` = union from both forward and backward |
| `test_bdtr_one_stop_other_continues` | Forward STOP + backward query → hop still executes |
| `test_bdtr_both_stop_triggers_llm_stop` | Both `''` → `'llm_stop'` |
| `test_bdtr_calls_are_concurrent` | Forward and backward bridge calls issued in parallel |
| `test_bdtr_user_content_wrapped_in_tags` | Prompts wrap query/facts in `<user_query>` / `<collected_facts>` |

#### `tests/retrieval/test_dag_multihop.py`

| Test | Verifies |
|---|---|
| `test_dag_single_no_dep` | 1-node DAG → 1 hop, `stop_reason='fixed'` |
| `test_dag_parallel_independent` | Two independent sub-queries run concurrently |
| `test_dag_dependency_ordering` | Sub-query 1 depends on 0 → executes after 0 |
| `test_dag_context_prefix_injected` | Dependent sub-query question prefixed with dep facts |
| `test_dag_cycle_raises_invalid_input` | Self-referential `depends_on` → `InvalidInputError` |
| `test_dag_decomp_failure_fallback` | Invalid JSON from LLM → fallback to single-node DAG |
| `test_dag_question_too_long_discarded` | Question > 500 chars → sub-question dropped |
| `test_dag_stop_reason_always_fixed` | `stop_reason` is always `'fixed'` |
| `test_dag_all_facts_in_merged` | Facts from all sub-queries present in `merged` |
| `test_dag_timeout_partial_results` | Timeout mid-layer → partial hops returned |
| `test_dag_max_subqueries_enforced` | `max_subqueries=21` raises `ValidationError` |

#### `tests/retrieval/test_security_extended.py`

| Test | Verifies |
|---|---|
| `test_fast_ppr_malicious_group_id` | `group_id` with `;` raises `InvalidInputError` |
| `test_treehop_long_query_rejected` | Query > 2000 chars raises `InvalidInputError` |
| `test_dag_injected_subquery_is_search_string_only` | Injected sub-question never reaches DB as query structure |
| `test_dag_self_dep_rejected` | `depends_on: [0]` for id=0 raises `InvalidInputError` |
| `test_beam_width_cap_enforced` | `beam_width > 10` raises `ValidationError` |
| `test_treehop_fusion_alpha_floor_enforced` | `fusion_alpha < 0.5` raises `ValidationError` |
| `test_bdtr_prompts_use_structural_delimiters` | Forward and backward prompts contain `<user_query>` |
| `test_dag_nonprintable_subquestion_falls_back` | Non-printable chars in sub-question → fallback, no crash |

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

## 14. Observability

### 12.1 Structured logging

Use `logging.getLogger(__name__)` in each module. Log at DEBUG for per-item events, INFO for per-call summaries, WARNING for degraded-mode fallbacks, ERROR for unexpected failures.

```python
# graph_loader.py
logger.debug('graph_loader.loaded', extra={
    'group_id': group_id, 'num_nodes': len(nodes),
    'num_edges': len(edges), 'duration_ms': elapsed_ms,
})

# temporal.py
logger.info('temporal_ppr.complete', extra={
    'group_id': group_id, 'mode': config.mode.value,
    'relevant_count': relevant_count, 'dropped_by_temporal': dropped,
    'num_facts': len(filtered), 'duration_ms': elapsed_ms,
})

# multihop.py
logger.info('multihop.complete', extra={
    'group_id': group_id, 'num_hops': len(hops),
    'stop_reason': stop_reason, 'merged_facts': len(merged),
    'duration_ms': elapsed_ms,
})
```

**Never log**: query text, fact text, passage content, embedding vectors, or API keys.

### 12.2 OpenTelemetry spans

Each major operation creates an OTEL span using the `opentelemetry-api` package (optional dependency; no-op if not installed):

| Span name | Key attributes |
|---|---|
| `retrieval.graph_load` | `group_id`, `num_nodes`, `num_edges` |
| `retrieval.seed_collection` | `group_id`, `num_seeds`, `expanded` |
| `retrieval.ppr.iteration` | `group_id`, `num_nodes`, `iterations_ran` |
| `retrieval.rerank` | `mode`, `candidate_count`, `passed_count` |
| `retrieval.temporal.filter` | `mode`, `relevant_count`, `dropped_count` |
| `retrieval.temporal.hydrate` | `passage_count` |
| `retrieval.multihop.hop` | `hop_index`, `new_fact_count` |
| `retrieval.multihop.bridge` | `hop_index`, `stop_signal` |

### 12.3 Metrics

If `prometheus_client` is installed:

```python
from prometheus_client import Histogram, Counter, Gauge

PPR_GRAPH_NODES = Histogram('retrieval_ppr_nodes', 'Nodes loaded per PPR call',
    buckets=[100, 1_000, 10_000, 50_000, 100_000])
PPR_DURATION = Histogram('retrieval_ppr_duration_seconds', 'PPR wall time',
    buckets=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
TEMPORAL_DROP_RATE = Histogram('retrieval_temporal_drop_fraction',
    'Fraction of PPR facts dropped by temporal filter',
    buckets=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
MULTIHOP_HOPS = Histogram('retrieval_multihop_hops',
    'Number of hops per multihop call', buckets=[1, 2, 3, 4, 5, 10, 20])
MULTIHOP_STOP_REASON = Counter('retrieval_multihop_stop_total',
    'Multihop stop reason', ['reason'])
```

---

## 15. Concurrency and Async Design

- All public API functions are `async def`.
- The PPR power iteration is synchronous NumPy/SciPy code. For `len(nodes) > 10_000`, run it in a thread pool via `asyncio.get_event_loop().run_in_executor(None, ...)`.
- Seed collection fans out embedding calls concurrently: one `asyncio.gather` over the original query plus all expansion terms and HyDE answers.
- Cross-encoder reranking: gather at most 20 LLM calls concurrently, sequentially between batches.
- Multihop hops are sequential — each hop's query depends on the previous hop's results. No pipelining between hops.
- The global PPR semaphore (`asyncio.Semaphore`) is initialized at module import time. The default limit is configurable via an environment variable `RETRIEVAL_PPR_CONCURRENCY` (default: 4).

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
