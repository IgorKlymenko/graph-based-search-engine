# Retrieval Library — Part 3: Extended Algorithms

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


**This file covers**: `shared.py` (FactMerger, StopCriterion, BridgeGenerator, EmbeddingFuser) and five extended search methods: FAST-PPR, TreeHop, Beam multihop, BDTR dual-bridge, and DAG multihop. Implement after Part 1 and Part 2.

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

## 15. Concurrency and Async Design

- All public API functions are `async def`.
- The PPR power iteration is synchronous NumPy/SciPy code. For `len(nodes) > 10_000`, run it in a thread pool via `asyncio.get_event_loop().run_in_executor(None, ...)`.
- Seed collection fans out embedding calls concurrently: one `asyncio.gather` over the original query plus all expansion terms and HyDE answers.
- Cross-encoder reranking: gather at most 20 LLM calls concurrently, sequentially between batches.
- Multihop hops are sequential — each hop's query depends on the previous hop's results. No pipelining between hops.
- The global PPR semaphore (`asyncio.Semaphore`) is initialized at module import time. The default limit is configurable via an environment variable `RETRIEVAL_PPR_CONCURRENCY` (default: 4).

---

