# Retrieval Library — Part 4: Security

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


**This file covers**: all 12 security risks and their mitigations across every module and method. Read before implementing any module — the mitigations affect design decisions in `config.py`, `graph_loader.py`, `seed.py`, `multihop.py`, `ppr_approx.py`, `treehop.py`, `multihop_beam.py`, and `dag_multihop.py`.

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

