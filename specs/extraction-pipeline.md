# Spec: Extraction Pipeline as Python Modules

Convert `entity_extraction.ipynb` and `relation_extraction.ipynb` into typed,
testable Python modules that run as ONE command, ending with the existing
`graph_upload.py`. The notebooks are the reference implementation for prompts,
schemas, and algorithms; this spec is the authority on structure, naming, and
the bug fixes below. `graph_upload.py` is NOT rewritten — it is the final stage
and its input contracts are frozen.

## 1. Goal and non-goals

**Goal.** `uv run python -m pipeline.run --notes-dir /Applications/reink`
executes: mention extraction → mention embedding → clustering → LLM entity
resolution → entity vectors → relation extraction → `graph_upload.py`.
Each stage persists its artifact and is individually resumable/skippable.

**Non-goals.**
- **No Fact nodes.** `(:Fact)` / `HAS_FACT` upload is out of scope (§4,
  Stage 7). Relation extraction still writes `node_facts` into the cache
  (they fall out of the same LLM call and stay useful for audit), but they
  are not embedded and not uploaded.
- Do not change `graph_upload.py` beyond the `--skip-facts` flag (§4, Stage 7).
- Do not port dead notebook cells: `%history` cells, commented-out embedding
  loops, the `relations_cache_v2.json` salvage cell (that logic already lives
  in `load_relations_cache` in `graph_upload.py`), or the notebook demo cells.
- Do not port the notebook Stage-4 fact embedding (`relation_vectors.npz`) —
  `graph_upload.py` embeds facts itself with `text-embedding-3-small`.
- No new retrieval features; `retrieval/` is untouched.

## 2. Artifact contract (frozen names, repo root)

These names are what `graph_upload.py` and `retrieval/` already read. The
notebooks' `*_v1.json` variants are drift — do not use them.

| Artifact | Producer stage | Schema |
|---|---|---|
| `entity_cache.json` | 1 | `{"processed_files": [str], "mentions": [Mention]}` |
| `vector_cache.npz` | 2 | `entities` (unique surface texts), `types`, `matrix` (float32, one averaged span vector per surface) |
| `resolved_entities.json` | 3–4 | `{"embedding_model", "resolve_model", "clustering", "total_entities", "entities": [ResolvedEntity], "mention_to_canonical": {surface: canonical}, "canonical_to_sources": {canonical: [file]}}` |
| `resolved_entity_vectors.npz` | 5 | `canonicals`, `cluster_matrix` (mxbai, averaged member vectors), `query_matrix` (e5 passage vectors) |
| `relations_cache.json` | 6 | dict: `{"edges": [RelationEdge], "node_facts": [NodeFact], "non_facts": [...], "rejected_subjects": [...], "uncovered_by_chunk": {...}, "failed_chunks": [...]}` |

`Mention` = `{text, type, context, episode_idx, episode, source_file,
char_offset?, gleaned?}`. `RelationEdge` / `NodeFact` are the Pydantic models
in `relation_extraction.ipynb`, copied verbatim.

## 3. Package layout

```
pipeline/
  __init__.py
  config.py        # all knobs: models, thresholds, workers, paths (constants)
  models.py        # Mention, ResolvedEntity, RelationEdge, NodeFact, ChunkResult
  chunking.py      # chunk_doc with real char offsets (§8.1)
  mentions.py      # stage 1: extraction + gleaning prompts and calls
  embed_mentions.py# stage 2: mxbai span-pool embedding (absorb embed_cache.py)
  clustering.py    # stage 3: HDBSCAN
  resolution.py    # stage 4: LLM resolution, validate_partition, escalation
  entity_vectors.py# stage 5: cluster_matrix + e5 query_matrix
  relations.py     # stage 6: relation extraction, verification, confidence
  run.py           # CLI orchestrator; last step invokes graph_upload.py
```

Reuse, don't duplicate:
- `services/e5.py` for ALL e5 embedding (it already enforces the asymmetric
  `"query: "`/`"passage: "` prefixes — both required; a missing `"passage: "`
  prefix is the known silent bug with this model).
- One shared OpenAI client helper (look at `services/openai.py` first; extend
  it rather than instantiating clients per module). `timeout=90.0,
  max_retries=2` for chat calls.
- `embed_cache.py`'s device selection (mps/cuda/cpu) and checkpointing move
  into `embed_mentions.py`; delete `embed_cache.py` after absorbing it.

## 4. Stage behavior

### Stage 1 — mention extraction (`mentions.py`)
Port from `entity_extraction.ipynb`:
- `SYSTEM_PROMPT`, `GLEANING_SYSTEM_PROMPT`, `MENTION_SCHEMA` verbatim.
- Model: `gpt-4.1-mini` default (config), strict JSON-schema output, temp 0.
- Per file: chunk → first pass → gleaning pass (config `ENABLE_GLEANING=True`);
  gleaned mentions get `gleaned: True` and `char_offset`.
- Concurrency: files × episodes via `ThreadPoolExecutor` (defaults 4 × 8).
- Resumable: skip files already in `processed_files`; save cache after each
  file under a lock. `--token-budget` (tiktoken count, default: unlimited)
  limits how many pending files are processed per run.

### Stage 2 — mention embedding (`embed_mentions.py`)
Port `embed_spans_batch` (mxbai-embed-large-v1, token-level span pooling over
the mention's char range inside its episode, window-clipped to 512 tokens).
Average vectors across occurrences of the same surface text. Skip mentions
whose `text` is not found verbatim in the episode (count and report).
Checkpoint to `vector_cache.npz` every N batches; resume by skipping
already-embedded surfaces when the model name matches.

### Stage 3 — clustering (`clustering.py`)
HDBSCAN only — do NOT port the consensus correlation-clustering code (it was
exploratory). `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=2,
metric="cosine")` over the surface vectors.
Output: multi-clusters + singletons. **Singletons = size-1 clusters AND
HDBSCAN noise points, from the SAME partition used for resolution** (§8.2).

### Stage 4 — LLM resolution (`resolution.py`)
Port verbatim: `RESOLVE_SYSTEM`, `RESOLVE_SCHEMA`, `resolve_cluster`,
`validate_partition`, `_pick_contexts`, the all-singleton API-failure
fallback, and the escalation pass (low-confidence merges re-run on
`gpt-4.1`). Fix the chat-call timeout: 90s, not the notebook's 10s (§8.3).
Persist `resolved_entities.json` per §2, with `"clustering"` recording the
method + params used. Singleton entities get
`source: "singleton"`, `confidence: "high"`.

### Stage 5 — entity vectors (`entity_vectors.py`)
Per canonical entity: `cluster_matrix[i]` = mean of member surface vectors
from `vector_cache.npz` (zeros if none); `query_matrix` = `services.e5`
passage embeddings of canonical names. Save `resolved_entity_vectors.npz`.

### Stage 6 — relation extraction (`relations.py`)
Port from `relation_extraction.ipynb` verbatim: `_SYSTEM_PROMPT`, `_SCHEMA`,
`FACT_TYPE_VALUES`, `_CONTENT_TYPES`, `_EDGE_TYPES`, `RelationEdge`,
`NodeFact`, `make_relation_id`, `make_fact_id`, `resolve_to_canonical`,
`format_entities_with_aliases`, `verify_quote_and_span`, `score_confidence`,
`process_chunk`, and the bulk runner (8 workers, incremental saves every 50
chunks, code-bug exceptions re-raised, API failures recorded in
`failed_chunks`). Model `gpt-4.1-mini`, `EXTRACTOR_VERSION =
"relation-extraction-v1/gpt-4.1-mini"`.

Chunk reconstruction: build chunks from `entity_cache.json` mentions keyed by
`(source_file, episode_idx)`; chunk-scoped entity list = canonicals of the
mentions in that chunk (as the notebook does). `char_start`/`char_end` come
from the stored mention `episode` located in the source doc — after the §8.1
fix these must be valid for ~100% of chunks whose source files still exist.

### Stage 7 — upload (`run.py` + minimal `graph_upload.py` change)
Invoke `graph_upload.py` as a subprocess:
`uv run python graph_upload.py [--saga X --group-id Y --dry-run ...]`,
forwarding relevant CLI flags. Non-zero exit fails the pipeline.

**No Fact nodes:** add a `--skip-facts` flag to `graph_upload.py` (the one
permitted change) that skips `upload_node_facts`, skips embedding node-fact
texts (saves the OpenAI calls), and skips creating the `fact_embedding_idx` /
`fact_text` indexes. The pipeline always passes `--skip-facts`. Default off
so direct invocations keep today's behavior.

## 5. CLI (`pipeline/run.py`)

```
uv run python -m pipeline.run --notes-dir /Applications/reink \
    [--from STAGE] [--only STAGE] [--force] \
    [--extraction-model gpt-4.1-mini] [--no-gleaning] \
    [--token-budget N] \
    [--dry-run] [--skip-upload] \
    [--saga NAME] [--group-id ID] [--min-confidence F]
```

- Stage names: `mentions, embed, cluster+resolve, entity-vectors, relations,
  upload` (cluster and resolve run together — clustering output is not
  persisted separately).
- Default skip logic: a stage is skipped if its output artifact exists and is
  newer than its input artifact; `--force` reruns from `--from` onward.
- `--dry-run` propagates to `graph_upload.py`; earlier stages have no writes
  to guard beyond their own cache files.
- Print a per-stage summary line (counts, skipped/ran, elapsed).

## 6. Config (`pipeline/config.py`)

Constants only (no env parsing beyond what exists): model names, thresholds
(`HDBSCAN_MIN_CLUSTER_SIZE=5`, `HDBSCAN_MIN_SAMPLES=2`), worker counts,
batch sizes, artifact `Path`s, prompt version strings. Env stays in `.env` (`OPENAI_API_KEY`, `NEO4J_*`), loaded via
`dotenv` at CLI entry only.

## 7. Frozen interfaces

- `graph_upload.py` reads `entity_cache.json` (`mentions` with `text`,
  `context`, `episode`, `episode_idx`, `source_file`),
  `resolved_entities.json` (`entities` with `canonical`, `members`, `reason`;
  `mention_to_canonical`), `relations_cache.json` (dict form). Anything the
  pipeline writes must keep those keys.
- `retrieval/` reads `resolved_entity_vectors.npz` key names as listed in §2.

## 8. Bugs to fix during the port (do NOT reproduce notebook behavior)

1. **Chunk char offsets.** `chunk_doc` strips paragraphs and re-joins with
   `"\n\n"`, so the chunk text often differs from the source slice and
   `doc.find(episode)` returns −1 (62/182 chunks in the last run). Rewrite
   chunking to compute each chunk's `(char_start, char_end)` against the
   original document while building it, and store both in the mention records
   so relation extraction never re-derives offsets via `find`.
   Keep behavior otherwise: split on blank lines, merge to ~1000 chars.
2. **Mixed partitions.** The notebook resolved HDBSCAN multi-clusters but
   persisted singletons from the CC partition, dropping HDBSCAN noise points.
   Fix: one partition end-to-end; singletons = size-1 clusters + noise from
   the partition that was resolved.
3. **Resolution timeout.** `client.with_options(timeout=10.0)` causes
   spurious all-singleton fallbacks on big clusters. Use 90s.
4. **Cache filenames.** Write the unversioned names (§2) everywhere.
5. **Uncovered-entity set arithmetic.** In `process_chunk` the expression
   `(entity_set - touched) | set(uncovered) - touched` relies on operator
   precedence; keep the (correct) semantics but parenthesize explicitly:
   `((entity_set - touched) | set(uncovered)) - touched`.

## 9. Security & consistency review (required deliverable)

While porting, audit the touched code paths and report findings in the final
summary (fix in-scope ones; list out-of-scope ones). Known starting points:

- **Credential handling.** `graph_upload.py` falls back to
  `NEO4J_PASSWORD="password"` — a silent default that can write to the wrong
  DB. Prefer failing fast when `NEO4J_PASSWORD` is unset. Never log secrets;
  confirm `.env` is gitignored.
- **Prompt injection surface.** Note files are untrusted input fed verbatim
  into LLM prompts; extraction output is constrained by strict JSON schemas
  and post-validated (`validate_partition`, subject/entity-set checks,
  verbatim-quote verification). Keep all of those checks; never `eval`/`exec`
  or interpolate model output into Cypher — parameterized queries only (as
  `graph_upload.py` already does).
- **Path handling.** `source_file` values from caches are used to read files
  (`_doc_text`) and build Obsidian URIs. Constrain reads to `--notes-dir`
  (resolve + prefix check) so a tampered cache can't read arbitrary paths.
- **Subprocess use.** Stage 7 must build the `graph_upload.py` argv as a
  list (no `shell=True`, no string interpolation of user args).
- **Cache trust.** All `json.loads` of local caches should tolerate missing
  keys with clear errors, not crash deep in a stage; sizes/counters printed
  so silent truncation is visible.
- **Inconsistencies.** Flag any further drift you find between notebook
  logic, `graph_upload.py`, and `retrieval/` expectations (naming, schema
  keys, embedding models/dims) beyond those in §8.

## 10. Quality bar (CLAUDE.md applies)

- uv only; `uv run ruff format .`, `uv run ruff check .`, `uv run pyright`
  all clean on the new package. 88-char lines. Type hints everywhere;
  docstrings on public functions. No new dependencies expected — everything
  needed is already in `pyproject.toml`.
- Tests (`uv run pytest`, no network, no model downloads):
  - `chunking.py`: offsets exactly slice the source doc; merging respects
    max_chars; degenerate inputs (empty doc, single huge paragraph).
  - `resolution.validate_partition`: omission repair, hallucinated member
    drop, normalization fallback, duplicate member handling.
  - `relations.verify_quote_and_span` + `score_confidence`: quote hit/miss,
    objectless vacuous span pass, temporal-digit inference, score floors.
  - `relations.resolve_to_canonical`: exact, case-insensitive, miss.
  - `clustering`: HDBSCAN wrapper returns (clusters, noise) with every input
    surface in exactly one of the two, on a toy vector set.
  - Stage skip logic in `run.py` (artifact mtime comparison) with tmp files.
  - LLM calls mocked; embedding models never loaded in tests.
- End-to-end verification (manual, cheap): run the pipeline on 2–3 small
  files with `--token-budget 3000`, then
  `uv run python graph_upload.py --dry-run --skip-embeddings` must complete
  with zero "no uuid mapping" skips.

## 11. Acceptance checklist

- [ ] `uv run python -m pipeline.run --notes-dir <dir> --skip-upload` produces
      all five artifacts from a clean slate.
- [ ] Re-running the same command is a no-op (all stages skipped).
- [ ] `--from relations --force` reruns only relations (+ upload unless
      skipped).
- [ ] Bad-offset chunk count is 0 on the test corpus.
- [ ] `graph_upload.py --dry-run --skip-facts` consumes the artifacts without
      warnings and reports no Fact-node step.
- [ ] ruff, pyright, pytest all pass.
- [ ] Security/consistency findings reported (§9).
- [ ] Notebooks are left in place untouched (they become exploratory docs).
