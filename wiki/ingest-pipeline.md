# Ingest pipeline (`server/public/src/inject`)

Documents in, graph rows out. One JSON store holds all state; every stage is
resumable; two stages cost money. Canonical narrative with the measurements is
`old_server/entity_linking/DATAFLOW.md`; setup traps are in
`old_server/entity_linking/RUNNING.md`. This page is the compressed reference.

## Stages

| # | File (alias) | In | Out | Cost | Key functions |
|---|---|---|---|---|---|
| 1 | `1_episodes.py` (`episodes`) | directory of `.md`, or one rendered document | `store.files`, `store.episodes`, `FileDiff` | local | `sync_corpus`, `sync_file`, `retire_file`, `split_episodes`, `build_records`, `file_id_for`, `episode_id_for`, `stable_uuid`, `sha1_text`, `strip_markup`, `load_store`, `save_store` |
| 2 | `2_chains.py` (`chains`) | episodes with `extracted_at is None` | `store.candidates`, `store.chains`; sets `extracted_at` | **1 LLM call per episode** | `run`, `process_episode`, `extract_chains`, `resolve_chains`, `seeded_chains`, `merge_seeded`, `content_key`, `chain_id_for`, `build_chains`, `build_candidates`, `pending_episodes`, `_complete_json` |
| 2b | `utils/judge.py` (`main.judge`) | live identifying surfaces without a verdict (connector seeds arrive with theirs: `connectors.base.name_seeds`, ADR connectors 12) | `store.names` `{surface: bool}` | **1 mini call per 40 new surfaces, then 1 gpt-4.1 call per 40 survivors** | `contexts_for`, `judge_surfaces`; ADR tables 30 |
| 3 | `3_encode.py` (`encoding`) | live chains | `ChainRecord` per chain, in memory | local; one nomic forward pass per episode + one batch over head names | `encode_episode`, `encode_chains`, `chain_positions`, `pool`, `_load` |
| 4 | `4_link.py` (`pairing`) | `ChainRecord`s (with `key` from seeds, `main._seed_key`) | cluster assignment → `store.entities`, `store.unresolved` | **1 LLM call per pair in the review band** (+ 1 per contested mention) | `build_evidence`, `features`, `PairScorer`, `candidate_pairs`, `ann_candidate_pairs`, `decide`, `keys_settle`, `cluster`, `unresolved`, `stable_entity_ids`, `choose_thresholds` |
| 5 | `5_store.py` (`store`) | the store | rows in Postgres, one transaction | local | `upload`, `match_entities`, `assign_entities`, `_rows_for_chain_surfaces` |

Orchestration and the two review prompts live in `main.py`: `ingest()`, `sync`, `extract`, `encode`, `link`, `resolve_against_graph`, `_adjudicate`, `_ask_cached`, `_name_vectors`, `_context_vectors`.

## Identity (content-addressed at every level)

```
NS           = 6ba7b810-9dad-11d1-80b4-00c04fd430c8      # == uuid.NAMESPACE_DNS; there must only ever be one
file_id      = uuid5(NS, "file:{tenant_id}|{source}|{external_id}")     # path-based: survives edits
episode_id   = uuid5(NS, "episode:{file_id}|{sha1(text)}")              # content-based: survives moves
candidate_id = uuid5(NS, "candidate:{episode_id}|{start}|{end}")        # offsets are safe: episode text is immutable
content_key  = sha1("|".join(sorted(surfaces that pass is_identifying)))
chain_id     = uuid5(NS, "chain:{episode_id}|{content_key}")            # what the chain contains, not which span leads it
entity_id    = min(chain_id in cluster)   →  then store.match_entities carries prior ids forward
```

Why each level is keyed this way: `ADRs/supabase/tables.md` entry 25. Why `chains` in the DB uses a surrogate PK with `content_key` as a separate unique: same entry, and `chains.chain_id_for` docstring.

## The store

```
store = {
  "files":      {file_id: {file_id, tenant_id, source, external_id, title, url, full_text_hash, render_version, episode_ids[], synced_at, removed_at}},
  "episodes":   {episode_id: {episode_id, file_id, text, extracted_at, removed_at,
                              seeds?: [{key, spans: [[start, end, text], …]}]}},   # connector-resolved chains
  "candidates": {candidate_id: {candidate_id, episode_id, start, end, text, candidate_type, origin, chain_id, is_head}},
  "chains":     {chain_id: {chain_id, episode_id, content_key, head_surface, entity_id}},
  "names":      {surface: bool},                           # the name judge's verdicts; delete to re-judge
  "reviews":    {sha1(prompt): answer},                    # review + entity-choice cache, survives runs
  "linking":    {"chains": [...], "merges": [[a, b, p], ...]},   # last run's state; feeds incremental linking
  "uploaded":   {"files": {}, "episodes": {}, "chains": {}, "surfaces": []},   # what the DB holds; feeds the delta upload
  "entities":   {entity_id: {surfaces[], chain_ids[], episode_ids[]}},
  "unresolved": [{chain_id, episode_id, surfaces[], candidates[], reason}],
  "connectors": {"github": {repo_id: {full_name, head, render_version, blobs{path: sha}}}},   # each connector's cursor
}
```

Rules that make it resumable:

- An episode is done when `extracted_at` is set. A failed episode keeps it unset and is retried next run; the failure is printed as a warning, **not raised**.
- Stage 1 short-circuits on unchanged `full_text_hash` + `render_version`. Editing one paragraph re-extracts only episodes whose text changed.
- Removed episodes are stamped `removed_at`, never deleted; liveness of candidates is derived from their episode.
- The store is checkpointed every `SAVE_EVERY = 25` episodes during stage 2, and at the end of every stage.
- **Not safe to run concurrently on one store path.** One ingest per tenant at a time (`main.py` docstring).
- Vectors live in a sidecar, `<store>.vectors.npz` (`main.load_vector_cache`): name vectors by string, context vectors by chain and span set, discarded if `MODEL_NAME` changes. `--no-vector-cache` recomputes.
- Stage 4 scores only pairs that could have changed since `store["linking"]` was written: pairs touching a chain not linked last time, and old pairs whose chains share a surface with one. `--full-relink` scores everything.
- Stage 5 sends only rows that differ from `store["uploaded"]`; `--full-upload` re-sends all, and a tenant with no `files` rows forces that automatically.

## Stage 2 protocol (the part most likely to be broken by a "small" prompt change)

1. `spans.parse_spans` builds a deterministic span inventory (spaCy `en_core_web_sm`: entities, noun chunks, pronouns, nested proper-noun runs) and issues ids `c0, c1, …`. Types are derived, never asked of the model.
2. `spans.mark_text` renders a copy with `⟦c0 Sarah Chen⟧` markers, nesting preserved. The clean text is never modified; `normalize_for_parse` is length-preserving by assertion.
3. The model answers in **ids only** (`CHAIN_SCHEMA`, strict JSON). Spans it adds come back as `n*` with verbatim text and are re-found exact → whitespace-loose → fuzzy (`FUZZY_CUTOFF 0.86`, `FUZZY_MARGIN 0.04`); anything not matched exactly once is dropped and counted.
4. An id that was never issued is a protocol violation: retry once, then fail the episode (`UnknownSpanIds`).
5. If the episode carries `seeds` (a connector's pre-resolved mentions, [connectors.md](connectors.md)), `chains.seeded_chains` makes them proper-name spans with origin `connector` and `chains.merge_seeded` folds them in: a seed absorbs every model chain sharing a span's offsets or text, and on identical offsets the seed's span wins. A seed span that does not match the text is dropped with a warning.
6. Chains with an identical proper name are unioned deterministically (`spans.union_identical_names`). A chain whose every surface fails `is_identifying` is not stored at all.

## Stage 4 decision

```
features = name_cosine, string_similarity, alias_hit, cooccurrence, context_gated
score    = sigmoid(PRIOR_INTERCEPT + Σ PRIOR_WEIGHTS[f]·f)        # unfitted by default
score <  split_at (0.30)          → split, no LLM
split_at ≤ score < merge_at (0.50) → review: MERGE_REVIEW_PROMPT with both passages (+ contested-surface note); never when pairing.keys_settle(a, b) — different connector keys, or a key beside prose, split instead (ADR connectors 13)
score ≥ merge_at                  → merge, no LLM
cluster = union-find over merges (cannot-link OFF, measured)
unresolved = chains whose every surface is contested and which merged with nothing → ENTITY_CHOICE_PROMPT against entity profiles; null keeps entity_id NULL
```

`alias_hit` and `cooccurrence` are free: they are the per-episode coref decisions from stage 2 aggregated corpus-wide (`Evidence`). A surface that bridges two surfaces some episode kept apart is `ambiguous` and stops carrying alias evidence *and* string similarity.

## Knobs and constants

| Name | Value | Where | Change only after reading |
|---|---|---|---|
| `max_chars` | 1000 | `episodes.split_episodes` | ADR tables 26 |
| `EXTRACTION_MODEL` | `gpt-4.1-mini` | `chains` | — |
| `EPISODE_WORKERS` / `REVIEW_WORKERS` | 8 / 8 | `chains`, `main` | — |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES` | 90 s, 2 | `chains` | [gotchas.md](gotchas.md) § Run-on sentences |
| `CHECKPOINT_EVERY`, `CHECKPOINT_BATCHES` | 200 review answers, 10 judge batches | `main`, `utils/judge` | the store is saved that often during the paid stages, so a kill keeps what was answered |
| `STALL_TIMEOUT` | 300 s of nothing finishing | `chains.completed` | every LLM stage runs through this pool; abandoned calls stay pending. A full-vault run sat an hour in `SSLSocket.read` before it existed |
| `SAVE_EVERY` | 25 episodes | `chains` | — |
| `MODEL_NAME`, `PREFIX`, `MAX_TOKENS` | nomic-embed-text-v1.5, `search_document: `, 2048 | `encoding` | `encoding` docstring |
| `WINDOW_TOKENS` | proper 8, definite 3, nominal 3, pronoun 0 | `encoding` | `encoding` docstring |
| `CONTEXT_GATE` | 0.90 string similarity | `pairing` | DATAFLOW § 4c |
| `PRIOR_WEIGHTS` / `PRIOR_INTERCEPT` | cosine 2.0, string 1.5, alias 3.5, cooccur −1.2, context 1.0 / −3.5 | `pairing` | RUNNING § PairScorer |
| `merge_at` / `split_at` | 0.50 / 0.30 | `main.ingest` | **Do not** replace with `choose_thresholds()` output: 0.634 vs 0.775 CoNLL on ECB+ |
| `ann_k`, `ANN_MIN_FACTOR` | 50, 5 (ANN only above 250 chains) | `pairing` | `pairing` comment on determinism |
| HNSW | `M=16`, `ef=200`, `random_seed=100` | `pairing.ann_candidate_pairs` | seeded because unseeded differed between runs |
| `candidate_pairs` blocking | `min_string 0.35` or `min_cosine 0.5`, never same episode | `pairing` | — |
| `_name_vectors` batch | 64 strings, `max_length=64` | `main` | — |
| `graph_resolve`, `context` | on, on | `main.ingest` | `--no-graph-resolve`, `--no-context` |
| `judge_names`, `judge_models` | on, `("gpt-4.1-mini", "gpt-4.1")` | `main.ingest` | `--no-judge`, `--judge-models a,b`; ADR tables 30, [experiments.md](experiments.md) § Junk filtering |
| `vector_cache`, `full_relink`, `full_upload` | on, off, off | `main.ingest` | `--no-vector-cache`, `--full-relink`, `--full-upload`; [experiments.md](experiments.md) E2, E3, E4 |

## Models and what each is for

| Model | Used by | Why this one |
|---|---|---|
| `gpt-4.1-mini`, temperature 0, strict JSON schema | coref (stage 2), merge review, entity choice (stage 4) | cost; answers are ids/booleans, not prose |
| `gpt-4.1-mini` then `gpt-4.1`, strict JSON schema | name judge (stage 2b) | cascade F1 0.82 against hand labels (mini alone 0.53, 4.1 alone 0.80); forty surfaces per call; the mini pass drops 56 % before the strong model |
| `nomic-ai/nomic-embed-text-v1.5` | name vectors (identity, AUC 0.936) and context vectors (gated) | mean-pooled natively, so span pooling is its own idiom; needs `trust_remote_code` and the prefix; loads lazily, picks mps → cuda → cpu |
| `en_core_web_sm` 3.8.0 (pinned wheel) | span inventory | reproducible image build |
| `text-embedding-3-large`, 1536 dims | `surfaces.embedding` (query-time seeding, not linking) | pgvector index limit; backfilled by `ask.py embed`, left NULL by stage 5 |

## What costs money on a re-run

- Zero extraction calls if no episode text changed.
- Review calls are cached in `store["reviews"]` by `sha1(prompt)`; unchanged corpus → zero review calls. Delete that key to force re-review.
- Name and context vectors come from the sidecar when present; only new strings and changed chains are encoded.
- Pair scoring is limited to pairs that could have changed; the rest of last run's merges are carried. Blocking (ANN or exhaustive) and evidence still run over every chain, both cheap.
- The upload sends the delta against the last successful upload.

## Outputs of `ingest()`

Returns per-table counts when `dsn` is given: `files, episodes, surfaces, entities, chains, chain_surfaces, carried, renamed, emptied, full`. Counts are rows *sent* (the delta); `full` says whether everything was sent. `renamed > 0` means an entity id changed because a cluster's smallest chain changed; `emptied` is the number of orphaned entities swept.

## Where the tests are

Stage tests for the old copy: `old_server/tests/test_entity_linking_*.py` (content_key 8, context_gate 5, episodes 30, identifying 4, markup 16, matching 8, review_cache 4, spans 24, stable_ids 6, unresolved 5). Tests for what only the live copy has: `server/tests/public/inject/` (unresolved report, strip_markup port, vector cache, incremental link, upload plan and pronoun filter). See [testing.md](testing.md).
