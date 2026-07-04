# Retrieval Library — Part 2: Baseline Algorithms

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


**This file covers**: the six baseline retrieval modules (`graph_loader`, `ppr`, `seed`, `rerank`, `filters`, `temporal`, `multihop`). Implement these after Part 1 (foundation). The extended methods in Part 3 depend on `temporal.py` and `seed.py` from this file.

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

