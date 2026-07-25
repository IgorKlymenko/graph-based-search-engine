# Retrieval Library — Part 5: Testing

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


**This file covers**: the complete testing plan — unit tests, integration tests, property-based tests, and benchmarks for all 7 retrieval methods and all shared utilities. Use this alongside the algorithm specs in Parts 2 and 3.

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

