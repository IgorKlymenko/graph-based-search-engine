# Arena — Method Comparison Tool Spec

## 1. Purpose

Arena is a CLI tool for running any combination of retrieval methods side-by-side on the same query and graph, then rendering a detailed comparison of:

- Tokens consumed (in / out / total) and their monetary cost
- Wall-clock time per method and per stage
- Facts retrieved, how they were scored, and which hops surfaced them
- Intermediate reasoning steps (bridge queries, beam states, DAG layers)

It is primarily a developer/researcher tool, not a production API. The output is terminal-rendered or written to JSON/Markdown. No persistent state beyond the output files.

---

## 2. Module Layout

```
arena/
├── __init__.py
├── cli.py           # Entry point: Click commands (run, duel, compare)
├── runner.py        # Executes a single method with full instrumentation
├── tracker.py       # TokenTracker, CostTracker, TimingTracker
├── display.py       # Rich-based terminal rendering: tables, diffs, panels
├── duel.py          # Side-by-side two-method comparison with fact diff
├── report.py        # JSON + Markdown report serialization
└── pricing.py       # Model pricing table and cost calculation
```

Arena imports from the `retrieval/` library (§3 of the main spec) for all search logic. It does not reimplement any retrieval algorithm.

---

## 3. Supported Methods

| Method ID | Class | Description |
|---|---|---|
| `ppr` | Baseline PPR + sequential multihop | Power-iteration, single bridge query per hop |
| `fast-ppr` | FAST-PPR + sequential multihop | Bidirectional random walk, single bridge |
| `treehop` | TreeHop | Embedding-space hops, no LLM bridge |
| `beam` | Beam-search multihop | Parallel hypotheses, single bridge per member |
| `beam-dual` | Beam-search + BDTR | Parallel hypotheses, forward+backward bridge per member |
| `dag` | DAG multihop | LLM decomposes query into sub-DAG |
| `sequential-dual` | Sequential + BDTR | Single chain, forward+backward bridge per hop |

Methods are referenced by ID on the CLI. New methods are registered by adding an entry to `runner.METHOD_REGISTRY`.

---

## 4. CLI Commands

All commands are exposed under the `arena` entry point.

### 4.1 `arena run`

Run one or more methods on a single query and print a summary.

```
arena run \
  --query "Who led the Apollo 11 mission and what were their prior roles?" \
  --group-id my-graph \
  --methods ppr,treehop,dag \           # default: all
  --output json \                       # json | markdown | terminal (default)
  --out-file results.json \             # optional; prints to stdout if omitted
  --temporal-mode currently_true \      # temporal filter mode
  --as-of "2024-01-01T00:00:00Z" \     # optional; used with as_of mode
  --max-hops 4 \
  --beam-width 3 \
  --llm-model gpt-4o-mini \
  --embed-model text-embedding-3-small \
  --no-hydrate                          # skip passage hydration
```

Output (terminal default):

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Arena Run  ·  Query: "Who led the Apollo 11 mission…"                  │
│  Group: my-graph  ·  Temporal: currently_true                           │
└─────────────────────────────────────────────────────────────────────────┘

Method       │ Time    │ Tokens In │ Tokens Out │ Cost      │ Facts │ Hops │ Stop
─────────────┼─────────┼───────────┼────────────┼───────────┼───────┼──────┼──────────
ppr          │  1.23s  │     1,204 │        312 │ $0.00041  │    12 │    3 │ convergence
fast-ppr     │  0.41s  │     1,204 │        312 │ $0.00041  │    10 │    3 │ convergence
treehop      │  0.08s  │       312 │          0 │ $0.000031 │     9 │    4 │ convergence
dag          │  2.11s  │     2,108 │        589 │ $0.00089  │    15 │    4 │ fixed

Cheapest:  treehop  ($0.000031)
Fastest:   treehop  (0.08s)
Most facts: dag     (15)
```

### 4.2 `arena duel`

Run exactly two methods, render a rich side-by-side comparison of retrieved facts, intermediate steps, and scores.

```
arena duel \
  --query "What changed in the company leadership after 2022?" \
  --group-id my-graph \
  --method-a ppr \
  --method-b dag \
  --temporal-mode changed_since \
  --as-of "2022-01-01T00:00:00Z"
```

Output layout (terminal, using Rich panels):

```
╔══════════════════════════════════╦══════════════════════════════════╗
║  ppr  ·  3 hops  ·  1.23s       ║  dag  ·  4 hops  ·  2.11s       ║
║  12 facts  ·  $0.00041           ║  15 facts  ·  $0.00089           ║
╠══════════════════════════════════╬══════════════════════════════════╣
║  HOP 1: "What changed in the…"  ║  [DECOMPOSE]                     ║
║  Seeds: Neil Armstrong, …        ║  Sub-q 0: "Who was CEO in 2022?" ║
║  Nodes loaded: 4,231             ║  Sub-q 1: "Who replaced them?"   ║
║                                  ║  Sub-q 2: "When did change?"     ║
║  FACTS (hop 1):                  ║  [parallel: 0, 2] → [1]          ║
║  [0.92] Neil Armstrong led…     ║                                  ║
║  [0.87] Command module pilot…   ║  LAYER 1 FACTS:                  ║
║                                  ║  [0.95] CEO was John Smith…      ║
║  HOP 2 bridge: "What were…"     ║  [0.91] Company founded in…      ║
║  [NEW] [0.81] Prior to Apollo…  ║                                  ║
║  [DUP] [0.87] Command module…   ║  LAYER 2 FACTS (context-enriched)║
║                                  ║  [0.89] John Smith replaced by…  ║
╠══════════════════════════════════╬══════════════════════════════════╣
║  UNIQUE TO PPR:                  ║  UNIQUE TO DAG:                  ║
║  · Buzz Aldrin backup role       ║  · Board vote details            ║
║  · Mission Control handoff       ║  · Interim appointment           ║
╠══════════════════════════════════╬══════════════════════════════════╣
║  SHARED FACTS (10):              ║  Neil Armstrong led…, etc.       ║
╚══════════════════════════════════╩══════════════════════════════════╝

Token breakdown:
  ppr    seeds: 412 in / 0 out  │  bridge×2: 792 in / 312 out
  dag    decompose: 584 in / 189 out  │  sub-queries: 1524 in / 400 out
```

### 4.3 `arena compare`

Like `run` but outputs a structured JSON report suitable for automated comparison, CI regression testing, or logging to a metrics store.

```
arena compare \
  --query-file queries.jsonl \        # one query per line: {"query": "...", "group_id": "..."}
  --methods all \
  --out-file comparison.json \
  --parallel 4                        # run N queries concurrently
```

---

## 5. Data Models (`tracker.py`)

### 5.1 TokenUsage

```python
@dataclass(frozen=True)
class TokenUsage:
    tokens_in: int
    tokens_out: int

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(self.tokens_in + other.tokens_in, self.tokens_out + other.tokens_out)
```

### 5.2 CostBreakdown

```python
@dataclass(frozen=True)
class CostBreakdown:
    llm_cost_usd: float    # LLM prompt + completion cost
    embed_cost_usd: float  # Embedding API cost

    @property
    def total_usd(self) -> float:
        return self.llm_cost_usd + self.embed_cost_usd
```

### 5.3 StageTimings

```python
@dataclass
class StageTimings:
    graph_load_ms: float = 0.0
    seed_collection_ms: float = 0.0
    ppr_iteration_ms: float = 0.0
    rerank_ms: float = 0.0
    temporal_filter_ms: float = 0.0
    hydration_ms: float = 0.0
    bridge_total_ms: float = 0.0
    decompose_ms: float = 0.0          # DAG only
    total_ms: float = 0.0
```

### 5.4 HopTrace

Full trace of a single hop — what query was used, what came back, what was new.

```python
@dataclass
class HopTrace:
    hop_index: int
    query: str                          # text query or '[embedding-hop-N]'
    ppr_seeds: list[str]                # entity names used as PPR seeds
    nodes_loaded: int
    edges_loaded: int
    facts_before_temporal: int
    facts_after_temporal: int
    dropped_by_temporal: int
    new_fact_uuids: frozenset[str]
    all_fact_uuids: frozenset[str]
    bridge_queries: list[str]           # 1 for single, 2 for dual, 0 for treehop/dag leaf
    token_usage: TokenUsage
    timing_ms: float
```

### 5.5 MethodResult

Top-level result for one method on one query.

```python
@dataclass
class MethodResult:
    method_id: str
    query: str
    group_id: str
    facts: list[Fact]                   # final merged fact list
    hops: list[HopTrace]
    stop_reason: str
    token_usage: TokenUsage
    cost: CostBreakdown
    timing: StageTimings
    error: str | None = None            # set if method raised an exception

    @property
    def fact_uuids(self) -> frozenset[str]:
        return frozenset(f.uuid for f in self.facts)
```

---

## 6. Token and Cost Tracking (`tracker.py`, `pricing.py`)

### 6.1 TokenTracker

Wraps `LLMClient` and `Embedder` at the arena level to count tokens transparently. The retrieval library itself is unmodified.

```python
class TrackingLLMClient:
    """Wraps any LLMClient, recording token usage per call."""

    def __init__(self, inner: LLMClient, tracker: 'TokenTracker') -> None:
        self._inner = inner
        self._tracker = tracker

    async def complete(self, prompt: str, *, model: str, **kwargs) -> str:
        result = await self._inner.complete(prompt, model=model, **kwargs)
        tokens_in = count_tokens(prompt, model)
        tokens_out = count_tokens(result, model)
        self._tracker.record(stage='llm', tokens_in=tokens_in, tokens_out=tokens_out, model=model)
        return result

    async def complete_json(self, prompt: str, schema: dict, *, model: str, **kwargs) -> dict:
        result = await self._inner.complete_json(prompt, schema, model=model, **kwargs)
        tokens_in = count_tokens(prompt, model)
        tokens_out = count_tokens(str(result), model)
        self._tracker.record(stage='llm_json', tokens_in=tokens_in, tokens_out=tokens_out, model=model)
        return result


class TrackingEmbedder:
    """Wraps any Embedder, recording token usage per call."""

    def __init__(self, inner: Embedder, tracker: 'TokenTracker') -> None:
        self._inner = inner
        self._tracker = tracker

    async def embed(self, text: str) -> list[float]:
        result = await self._inner.embed(text)
        self._tracker.record(stage='embed', tokens_in=count_tokens(text, 'embedding'), tokens_out=0)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = await self._inner.embed_batch(texts)
        total_in = sum(count_tokens(t, 'embedding') for t in texts)
        self._tracker.record(stage='embed_batch', tokens_in=total_in, tokens_out=0)
        return results


class TokenTracker:
    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, *, stage: str, tokens_in: int, tokens_out: int, model: str = '') -> None:
        self._records.append({'stage': stage, 'in': tokens_in, 'out': tokens_out, 'model': model})

    def total(self) -> TokenUsage:
        return TokenUsage(
            tokens_in=sum(r['in'] for r in self._records),
            tokens_out=sum(r['out'] for r in self._records),
        )

    def by_stage(self) -> dict[str, TokenUsage]:
        stages: dict[str, list[dict]] = defaultdict(list)
        for r in self._records:
            stages[r['stage']].append(r)
        return {
            stage: TokenUsage(
                tokens_in=sum(r['in'] for r in records),
                tokens_out=sum(r['out'] for r in records),
            )
            for stage, records in stages.items()
        }
```

### 6.2 Token counting (`count_tokens`)

```python
def count_tokens(text: str, model: str) -> int:
    """
    Use tiktoken for OpenAI models. Fall back to len(text.split()) * 1.33 for others.
    Never raise — return 0 on any error (token counting is best-effort in the arena).
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text.split()) * 1.33))
```

### 6.3 Pricing table (`pricing.py`)

```python
# USD per 1M tokens. Kept in a config file so it can be updated without code changes.
# Loaded from arena/pricing.json at startup; falls back to these defaults.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    'gpt-4o':              {'in': 2.50,  'out': 10.00},
    'gpt-4o-mini':         {'in': 0.15,  'out': 0.60},
    'gpt-4.1':             {'in': 2.00,  'out': 8.00},
    'gpt-4.1-mini':        {'in': 0.40,  'out': 1.60},
    'text-embedding-3-small': {'in': 0.02, 'out': 0.0},
    'text-embedding-3-large': {'in': 0.13, 'out': 0.0},
    'embedding':           {'in': 0.02,  'out': 0.0},   # generic fallback
}

def compute_cost(usage: TokenUsage, model: str) -> float:
    """Returns USD cost for the given token usage and model."""
    pricing = _load_pricing().get(model, DEFAULT_PRICING.get('gpt-4o-mini', {'in': 0.15, 'out': 0.60}))
    return (usage.tokens_in / 1_000_000) * pricing['in'] + \
           (usage.tokens_out / 1_000_000) * pricing['out']
```

The pricing file `arena/pricing.json` is loaded at startup and can be overridden by the user without editing code:

```json
{
  "gpt-4o-mini": {"in": 0.15, "out": 0.60},
  "claude-sonnet-4-6": {"in": 3.00, "out": 15.00}
}
```

---

## 7. Runner (`runner.py`)

### 7.1 Method registry

```python
METHOD_REGISTRY: dict[str, Callable] = {
    'ppr':            run_ppr_sequential,
    'fast-ppr':       run_fast_ppr_sequential,
    'treehop':        run_treehop,
    'beam':           run_beam,
    'beam-dual':      run_beam_dual,
    'dag':            run_dag,
    'sequential-dual': run_sequential_dual,
}
```

Each `run_*` function has the same signature:

```python
async def run_ppr_sequential(
    driver: GraphDriver,
    embedder: Embedder,
    llm_client: LLMClient,
    query: str,
    group_id: str,
    cli_args: ArenaArgs,
    tracker: TokenTracker,
) -> MethodResult:
```

### 7.2 Timing instrumentation

Each `run_*` function uses a `TimingContext` context manager from `tracker.py`:

```python
@contextmanager
def timed(stage: str, timings: StageTimings) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    setattr(timings, f'{stage}_ms', elapsed_ms)
```

Stages: `graph_load`, `seed_collection`, `ppr_iteration`, `rerank`, `temporal_filter`, `hydration`, `bridge_total`, `decompose`.

### 7.3 Hop trace construction

Inside each `run_*` function, the runner converts the library's `MultihopHop` (or `TreeHop`'s equivalent) into a `HopTrace`, capturing the bridge queries actually sent to the LLM and the token snapshot at that point.

For TreeHop: `bridge_queries=[]`, `query='[embedding-hop-N]'`.
For DAG: the "hops" map to sub-queries in topological layer order; `bridge_queries=[]` (the decomposition prompt is tracked as `decompose_ms`).

### 7.4 Error wrapping

If a method raises any exception:
1. Log the full exception at ERROR level.
2. Return `MethodResult(error=str(e), facts=[], hops=[], ...)` with zero metrics.
3. Do not propagate the exception to the caller — other methods in the same `run` should still execute.

---

## 8. Display (`display.py`, `duel.py`)

Rendering uses the `rich` library. All display functions accept `MethodResult` objects and produce terminal output or return `str` (for Markdown mode).

### 8.1 Summary table (`display.summary_table`)

Renders the per-method comparison table shown in §4.1. Columns:

| Column | Source |
|---|---|
| Method | `result.method_id` |
| Time | `result.timing.total_ms` |
| Tokens In | `result.token_usage.tokens_in` |
| Tokens Out | `result.token_usage.tokens_out` |
| Cost | `result.cost.total_usd` (formatted as `$0.000XX`) |
| Facts | `len(result.facts)` |
| Hops | `len(result.hops)` |
| Stop | `result.stop_reason` |

Highlight rows: cheapest in green, fastest in blue, most facts in yellow.

### 8.2 Hop detail (`display.hop_detail`)

Expandable per-hop view (enabled by `--verbose`):

```
HOP 1  ·  "original query"  ·  120ms  ·  412in/112out tokens
  Seeds:       Neil Armstrong (0.92), Michael Collins (0.87)
  Nodes/Edges: 4,231 / 18,904
  PPR → 50 candidates → 12 passed rerank (threshold 0.30)
  Temporal:    15 → 12 (dropped 3 by currently_true)
  NEW facts:   9  DUP facts: 3

  [0.92] Neil Armstrong served as commander of Apollo 11
  [0.87] Michael Collins piloted the command module
  ...

HOP 2  bridge: "What were their roles before Apollo 11?"  ·  180ms
  ...
```

### 8.3 Duel view (`duel.render`)

Renders two `MethodResult` objects side by side using Rich columns. Sections:
1. **Header**: method ID, hops, time, cost, facts count.
2. **Per-hop detail**: synchronized by hop index; if methods have different hop counts, shorter side shows `—`.
3. **Fact diff**: facts unique to A, facts unique to B, facts shared by both (matched by UUID).
4. **Token breakdown**: per-stage token table for each method.
5. **Verdict row**: labels the winner on time, cost, and fact count.

The fact diff section shows each unique fact with its score and the hop it was first retrieved in.

### 8.4 Output modes

| Flag | Behavior |
|---|---|
| `--output terminal` | Rich-rendered to stdout (default) |
| `--output json` | `MethodResult` (or list of them) serialized to JSON via Pydantic `.model_dump()` |
| `--output markdown` | Plain Markdown table + fenced code blocks |

JSON output is the canonical machine-readable format. All float fields are rounded to 6 decimal places. Datetimes are ISO 8601 UTC strings.

---

## 9. Report Serialization (`report.py`)

### 9.1 JSON schema for a single run

```json
{
  "arena_version": "1.0",
  "query": "...",
  "group_id": "...",
  "temporal_mode": "currently_true",
  "as_of": "2024-01-01T00:00:00Z",
  "results": [
    {
      "method_id": "ppr",
      "facts": [
        {"uuid": "...", "fact": "...", "score": 0.92, "rerank_score": 0.85,
         "valid_at": null, "invalid_at": null, "first_hop": 0}
      ],
      "hops": [
        {
          "hop_index": 0,
          "query": "...",
          "ppr_seeds": ["Neil Armstrong", "..."],
          "nodes_loaded": 4231,
          "edges_loaded": 18904,
          "facts_before_temporal": 15,
          "facts_after_temporal": 12,
          "dropped_by_temporal": 3,
          "new_fact_count": 9,
          "bridge_queries": ["What were their roles before Apollo 11?"],
          "tokens_in": 412,
          "tokens_out": 112,
          "timing_ms": 120.4
        }
      ],
      "stop_reason": "convergence",
      "tokens_in": 1204,
      "tokens_out": 312,
      "llm_cost_usd": 0.000368,
      "embed_cost_usd": 0.000042,
      "total_cost_usd": 0.00041,
      "total_ms": 1230.0,
      "stage_timings": {
        "graph_load_ms": 45.2,
        "seed_collection_ms": 210.4,
        "ppr_iteration_ms": 380.1,
        "rerank_ms": 120.3,
        "temporal_filter_ms": 2.1,
        "bridge_total_ms": 472.0
      },
      "error": null
    }
  ]
}
```

### 9.2 JSONL batch report

When `--query-file` is used, each line of the output JSONL is one run (above schema). The file also includes a header line:

```json
{"type": "header", "arena_version": "1.0", "methods": ["ppr", "dag"], "query_count": 50, "generated_at": "..."}
```

---

## 10. Security Risks and Mitigations

### 10.1 CLI Argument Injection (HIGH)

**Risk**: The `--query`, `--group-id`, `--methods`, and `--as-of` arguments are supplied on the command line. A script that constructs CLI calls from user-supplied data (e.g. a CI pipeline with user-submitted queries) could pass arguments that attempt shell injection or graph injection.

**Mitigations**:
1. **Query and group-id validation**: Apply the same validators as the retrieval library (`len(query) ≤ 2000`, `group_id` matches `^[A-Za-z0-9_\-]+$`). Raise `InvalidInputError` with a clear message before any network call.
2. **Method name allowlist**: `--methods` is validated against `METHOD_REGISTRY.keys()`. Unknown method IDs raise `InvalidInputError`, never passed to `eval` or `__import__`.
3. **`--as-of` datetime parsing**: Parse strictly via `datetime.fromisoformat()`. Reject anything that doesn't parse. Do not pass the raw string to any query.
4. **No shell expansion of arguments**: All subprocess calls (if any — e.g. to open a file) use `subprocess.run([...], shell=False)`.

### 10.2 Pricing File Tampering (MEDIUM)

**Risk**: `arena/pricing.json` is loaded from the filesystem. If a CI/CD pipeline or shared environment allows this file to be modified, pricing figures can be set to zero, hiding actual costs from developers.

**Mitigations**:
1. Log the pricing file path and a SHA-256 hash of its contents at startup (INFO level).
2. If the pricing file is absent, fall back to hardcoded defaults and log WARNING. Do not silently use $0.00 prices.
3. Validate that all price values are positive floats. A zero or negative price in the file raises `ValueError` at load time.
4. For production cost tracking, the arena should write to an append-only log that is separate from the pricing file, so tampered prices don't alter historical records.

### 10.3 Output File Path Traversal (MEDIUM)

**Risk**: `--out-file` accepts an arbitrary file path. If the arena is invoked programmatically from a web service, a path like `../../etc/cron.d/pwned` could be supplied.

**Mitigations**:
1. Resolve `--out-file` to an absolute path with `Path.resolve()`.
2. Reject paths that resolve outside the current working directory and its descendants (unless `--allow-absolute-path` flag is explicitly set).
3. Reject paths that do not end in `.json`, `.jsonl`, or `.md` (allowlist on extension).
4. Write to a temp file first, then `os.rename()` atomically. Never partially-write to the target.

### 10.4 Query File Injection (`--query-file`) (MEDIUM)

**Risk**: Each line of the `--query-file` JSONL is parsed and its `query` and `group_id` fields are used. A maliciously crafted JSONL file could inject queries designed to probe the graph, exfiltrate facts, or consume LLM budget.

**Mitigations**:
1. Parse each line with `json.loads()`. Any parse failure skips the line with a WARNING (no crash).
2. Apply the same per-query validations as `arena run` to each parsed query (length, group-id regex).
3. Cap the number of queries per file at a configurable `MAX_QUERIES_PER_FILE = 1000`. Reject files with more lines.
4. `--parallel` is capped at `min(parallel, 16)` regardless of user input.

### 10.5 Token Cost DoS (MEDIUM)

**Risk**: An operator running `arena compare --methods all --query-file large_file.jsonl` with `beam-dual` at `beam_width=10, max_hops=20` could generate thousands of LLM calls per query, resulting in unexpected API bills.

**Mitigations**:
1. **Per-run cost cap**: `--max-cost-usd` flag (default: `$1.00`). If cumulative cost for a single run exceeds this cap, abort remaining methods and print a warning. Already-completed methods are still reported.
2. **Per-file cost cap**: `--max-total-cost-usd` flag (default: `$10.00`). Abort batch run if cumulative cost exceeds this.
3. **Dry-run estimation**: `arena run --dry-run` estimates token count based on query length and hop count without making any API calls. Prints estimated cost range before prompting user to confirm.
4. **Cost reported prominently**: The summary table always shows cost as the third column. It is never hidden.

### 10.6 Sensitive Data in Output Files (MEDIUM)

**Risk**: Output JSON/Markdown files contain fact text extracted from the graph, which may include PII or confidential information. These files could be shared, committed to git, or uploaded to issue trackers unintentionally.

**Mitigations**:
1. Add a `--redact-facts` flag that replaces fact text with `[REDACTED-uuid]` in all output formats while preserving scores and structure for debugging.
2. Log a WARNING at the start of any run that writes to a file: "Output file will contain graph fact content."
3. Default output file name includes a timestamp (`arena_results_20260626_143022.json`) rather than a generic name like `results.json`, making accidental overwrites and commits less likely.
4. `--out-file` paths ending in `.git/` or containing hidden directories are rejected.

### 10.7 LLM Response Logging (LOW)

**Risk**: If bridge questions or LLM responses are logged at INFO level, they appear in shared log aggregators (Datadog, CloudWatch, etc.), exposing query intent and graph content.

**Mitigation**:
1. LLM prompts and completions are logged only at DEBUG level. Production deployments should set `LOG_LEVEL=INFO`.
2. Token counts (not text) are logged at INFO level.
3. The `--verbose` flag enables DEBUG logging for the arena process only, scoped to the arena logger (`logging.getLogger('arena')`). It does not change the retrieval library's log level.

### 10.8 Concurrent Run Data Leakage (LOW)

**Risk**: `arena compare --parallel 4` runs multiple queries concurrently. A shared mutable `TokenTracker` could accumulate tokens across queries, making per-query cost figures inaccurate and potentially leaking one query's facts into another's context.

**Mitigation**: Each query run gets its own `TokenTracker` instance. The `TrackingLLMClient` and `TrackingEmbedder` wrappers take the tracker as a constructor argument, not a global. Concurrent runs share the same underlying `LLMClient` connection pool but each has its own accounting.

---

## 11. Testing Plan

### 11.1 Unit tests (no database, no live LLM)

All tests use stub drivers, embedders, and LLM clients. No network calls.

#### `tests/arena/test_tracker.py`

| Test | Verifies |
|---|---|
| `test_token_tracker_records_stages` | `by_stage()` groups records correctly |
| `test_token_tracker_total_sums_all` | `total()` = sum of all records |
| `test_tracking_llm_client_counts_tokens` | Wrapping a stub LLM client records tokens on each call |
| `test_tracking_embedder_counts_tokens` | Wrapping a stub embedder records tokens |
| `test_tracking_llm_json_counts_tokens` | `complete_json` records both prompt and response tokens |
| `test_token_tracker_thread_safe` | Concurrent async tasks recording to same tracker produce correct totals |
| `test_count_tokens_fallback_no_tiktoken` | Without tiktoken, returns estimate rather than raising |

#### `tests/arena/test_pricing.py`

| Test | Verifies |
|---|---|
| `test_known_model_returns_correct_cost` | `gpt-4o-mini` 1M in tokens = $0.15 |
| `test_unknown_model_falls_back_to_default` | Unknown model uses fallback pricing, no crash |
| `test_zero_price_in_file_raises` | `{"in": 0.0, "out": 0.6}` in pricing.json raises `ValueError` |
| `test_negative_price_raises` | Negative price raises `ValueError` |
| `test_pricing_file_hash_logged` | Loading pricing.json logs its SHA-256 hash at INFO |
| `test_cost_rounds_to_six_decimal_places` | Floating point cost is rounded in output |

#### `tests/arena/test_runner.py`

| Test | Verifies |
|---|---|
| `test_run_ppr_returns_method_result` | Stub PPR returns `MethodResult` with correct `method_id='ppr'` |
| `test_run_treehop_zero_tokens_out` | TreeHop produces zero output tokens (no LLM) |
| `test_run_method_error_returns_error_field` | Exception during run → `result.error` set, no exception propagates |
| `test_run_multiple_methods_independent` | One method error does not affect other methods' results |
| `test_hop_trace_populated` | `HopTrace` has correct `nodes_loaded`, `facts_before_temporal`, etc. |
| `test_hop_trace_treehop_no_bridge` | TreeHop `HopTrace.bridge_queries == []` |
| `test_hop_trace_dag_subquery_as_query` | DAG `HopTrace.query` equals the sub-question string |
| `test_timing_stages_populated` | `StageTimings` has non-zero values for relevant stages |
| `test_all_methods_registered` | `METHOD_REGISTRY` contains all 7 method IDs |

#### `tests/arena/test_cli.py`

| Test | Verifies |
|---|---|
| `test_run_command_valid_invocation` | `arena run --query "..." --group-id x` exits 0 |
| `test_run_command_invalid_method` | `--methods unknown-method` exits non-zero with error message |
| `test_run_command_query_too_long` | Query > 2000 chars exits non-zero |
| `test_run_command_invalid_group_id` | `--group-id "x;DROP"` exits non-zero |
| `test_run_command_output_json` | `--output json` produces valid JSON to stdout |
| `test_run_command_out_file_created` | `--out-file /tmp/test.json` creates the file |
| `test_run_command_out_file_path_traversal_rejected` | `--out-file ../../etc/evil` exits non-zero |
| `test_run_command_out_file_bad_extension` | `--out-file result.exe` exits non-zero |
| `test_duel_command_requires_two_methods` | Missing `--method-b` exits non-zero with clear error |
| `test_compare_command_query_file_parsed` | 3-query JSONL file → 3 result sets in output |
| `test_compare_command_max_queries_enforced` | File with 1001 queries → error |
| `test_compare_command_parallel_cap` | `--parallel 100` capped at 16 |
| `test_dry_run_no_api_calls` | `--dry-run` prints estimate, makes zero LLM/embed calls |
| `test_max_cost_cap_aborts_run` | `--max-cost-usd 0.0001` with expensive method → aborted with message |
| `test_as_of_invalid_format_rejected` | `--as-of "not-a-date"` exits non-zero |

#### `tests/arena/test_display.py`

| Test | Verifies |
|---|---|
| `test_summary_table_renders_all_methods` | 3 methods → 3 rows in table |
| `test_summary_table_highlights_cheapest` | Cheapest row has green markup |
| `test_hop_detail_new_vs_dup_labels` | `[NEW]` and `[DUP]` prefixes appear on correct facts |
| `test_duel_unique_facts_correct` | Facts unique to method A not in method B's UUID set |
| `test_duel_shared_facts_correct` | Shared facts appear in both UUID sets |
| `test_markdown_output_no_rich_markup` | Markdown output contains no `[bold]` or ANSI escapes |
| `test_json_output_valid_schema` | JSON output parses and matches schema |
| `test_redact_facts_replaces_text` | `--redact-facts` → all `fact` fields are `[REDACTED-uuid]` |

### 11.2 Integration tests (require database and API keys)

| Test | Verifies |
|---|---|
| `test_run_all_methods_live_int` | All 7 methods run on a synthetic 100-node graph |
| `test_duel_fact_diff_accurate_int` | Fact UUIDs in diff match those actually retrieved |
| `test_compare_batch_int` | 5-query batch with 2 methods → 10 result sets in output |
| `test_cost_cap_aborts_live_int` | `max_cost_usd=0.001` stops an expensive run mid-way |
| `test_token_counts_match_api_response_int` | Token count from tracker matches API usage field |

### 11.3 Property-based tests

```python
@given(
    tokens_in=st.integers(min_value=0, max_value=1_000_000),
    tokens_out=st.integers(min_value=0, max_value=100_000),
    model=st.sampled_from(list(DEFAULT_PRICING.keys())),
)
def test_cost_always_non_negative(tokens_in, tokens_out, model):
    usage = TokenUsage(tokens_in=tokens_in, tokens_out=tokens_out)
    assert compute_cost(usage, model) >= 0.0

@given(results=st.lists(method_result_strategy(), min_size=1, max_size=7))
def test_summary_table_never_crashes(results):
    """display.summary_table() must not raise for any valid MethodResult list."""
    display.summary_table(results)

@given(
    a=method_result_strategy(),
    b=method_result_strategy(),
)
def test_duel_unique_plus_shared_equals_union(a, b):
    unique_a, unique_b, shared = duel.fact_diff(a, b)
    assert unique_a | unique_b | shared == a.fact_uuids | b.fact_uuids
```

---

## 12. Concurrency Design

- `arena compare --parallel N` uses `asyncio.gather` over the query list, limited by a semaphore of size `min(N, 16)`.
- Within a single query run, all configured methods run sequentially by default. Use `--parallel-methods` to run methods concurrently (they share the same graph driver connection pool).
- Each method run has its own `TokenTracker` and `StageTimings` — no shared mutable state across concurrent runs.
- The `TimingContext` context manager uses `time.perf_counter()` which is thread-safe and monotonic.

---

## 13. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `click` | `>=8.0` | CLI argument parsing |
| `rich` | `>=13.0` | Terminal rendering (tables, panels, syntax highlighting) |
| `tiktoken` | `>=0.6` | Token counting for OpenAI models |
| `pydantic` | `>=2.0` | Report serialization and config validation |

Optional:
| `pytest-asyncio` | test only | Async test support |
| `hypothesis` | test only | Property-based tests |

The arena has no direct dependency on `scipy`, `numpy`, or any graph database driver — those come through the `retrieval/` library.

---

## 14. Open Questions

1. **HTML report**: Should `--output html` be supported for a richer visual report with collapsible hop detail? Could use `rich`'s HTML export or a Jinja2 template.

2. **Persistent history**: Should the arena store results in a local SQLite database so users can compare across runs over time (e.g. "did adding BDTR improve fact recall on these 10 queries")? This would make the arena a lightweight evaluation harness.

3. **LLM-as-judge scoring**: Should the arena optionally call an LLM to score each method's retrieved facts for relevance to the query, producing a `judge_score` per method? This enables automated quality ranking, not just quantity and cost comparison.

4. **Config file**: Should `arena run` accept a `--config arena.yaml` file for specifying all parameters (methods, models, hop counts, etc.) so complex runs can be reproduced without long CLI strings?

5. **Token counting for non-OpenAI models**: `tiktoken` only covers OpenAI models. For Claude or Gemini, the word-count estimator is used. Should the arena accept `--token-counter anthropic` to use the Anthropic token-count API endpoint for accurate counts?
