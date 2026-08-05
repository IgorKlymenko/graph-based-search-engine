# Running the pipeline

Orientation for anyone picking this up. `DATAFLOW.md` explains *what* each stage
produces; this file is *how to run it* and what will trip you up.

## Prerequisites

```bash
uv sync
uv run python -m spacy download en_core_web_sm   # not a declared dependency
export OPENAI_API_KEY=...                        # read by openai.OpenAI()
```

`nomic-embed-text-v1.5` downloads on first use (~550 MB) and needs
`trust_remote_code=True`, which is already set in `3_encode.py`. Both models load
lazily — importing a module never fetches one, so tests and `--dry-run` stay
offline.

## Run it

```bash
# see what would be extracted, no model calls, no cost
uv run python -m entity_linking.pipeline --store ./store.json \
    --notes-dir ./notes --tenant-id <uuid> --dry-run

# full run
uv run python -m entity_linking.pipeline --store ./store.json \
    --notes-dir ./notes --tenant-id <uuid> --entities-out ./entities.json

# re-link without re-extracting (store already has candidates)
uv run python -m entity_linking.pipeline --store ./store.json
```

Omit `--notes-dir` to skip stage 1 and work on whatever is already in the store.
`--limit N` caps how many episodes get extracted this run, which is the safe way
to try a change against a large corpus.

Useful flags: `--model` (default `gpt-4.1-mini`), `--workers` (episode
extraction, default 8), `--ann-k` (blocking width, default 50), `--merge-at` /
`--split-at`.

## The store is the state

Everything lives in one JSON file, and every stage is resumable. A crashed or
killed run loses nothing: re-running redoes only what is missing.

```
store = {"files": {...}, "episodes": {...}, "candidates": {...}, "entities": {...}}
```

An episode is "done" when its `extracted_at` is set. If extraction fails —
including the hard failure on an invalid span id — `extracted_at` stays unset on
purpose, so the next run retries that episode and only that episode.

Stage 1 short-circuits on an unchanged `full_text_hash`, so re-running against an
unchanged corpus is nearly free. Editing one document re-extracts only the
episodes whose text actually changed; unchanged paragraphs keep their ids because
episode identity is content-addressed.

## What costs money

Two model steps, both proportional to *new* work:

- one extraction call per episode with no `extracted_at`
- one review call per chain pair landing in the `[--split-at, --merge-at)` band

Everything else — span parsing, embedding, ANN blocking, scoring, clustering —
is local. A re-run over an unchanged corpus makes zero extraction calls but will
re-score and re-review, because linking is global and has no per-pair cache.

## Gotchas

**Digit-prefixed module names.** `import entity_linking.4_link` is a syntax
error. `__init__.py` loads the stages by path and re-exports them under ordinary
names, so always go through those:

```python
from entity_linking import episodes as EP   # 1_episodes.py
from entity_linking import chains as CH     # 2_chains.py
from entity_linking import encoding as EN   # 3_encode.py
from entity_linking import pairing as PR    # 4_link.py
```

`spans`, `prompts` and `coref_metrics` are support modules, not stages, and
import normally.

**Do not replace the thresholds with `choose_thresholds()` output.** It targets
pairwise merge precision, which is the wrong objective once merges close
transitively — it fragments entities badly. Tune the cut against the clustering
metric instead. Measured, with numbers, in `DATAFLOW.md`.

**`PairScorer` is unfitted by default** and its prior weights saturate at 0.50
for an identical name, which is why `--merge-at` defaults to 0.50 and not
higher. Raising it without fitting produces zero merges.

**`cluster()` ignores cannot-link by default**, deliberately. The docstring
records why.

**The context vector is uncentered, and only reaches the scorer through
`context_gated`** — a feature that fires solely on pairs whose surfaces are the
same string. Computing it costs one forward pass per episode; `--no-context`
skips that. It measured flat end to end on ECB+ (0.774 vs 0.775) for a reason
specific to that corpus; `DATAFLOW.md` §4c has the numbers before you spend time
tuning it.

## Files

| file | role |
|---|---|
| `pipeline.py` | orchestration and the CLI; no logic of its own |
| `1_episodes.py` | documents → episodes, change detection, the store |
| `2_chains.py` | span inventory + LLM coref, one call per episode |
| `3_encode.py` | nomic forward pass, one per episode → chain vectors |
| `4_link.py` | evidence, features, scoring, review band, clustering |
| `spans.py` | normalisation, spaCy span inventory, fuzzy re-location |
| `prompts.py` | the two system prompts and their JSON schemas |
| `coref_metrics.py` | MUC / B³ / CEAFe / CoNLL |
| `demos/` | runnable examples; `demo_end_to_end.py` is the fullest |

`demos/*.py` dump intermediate JSON so you can inspect a single stage without
running the whole thing.

## Known open issues

- **Entity ids are positional** and change whenever clustering changes. Nothing
  guarantees stability across re-runs, which matters as soon as the output feeds
  a persistent graph. Unresolved.
- **Definite descriptions get promoted to entities** — `['the directory sync']`,
  `['the incident']`. The WHAT-COUNTS taxonomy in `prompts.py` is dead code and
  is not applied anywhere; that is the intended fix.
- **`4_link.py` has no unit tests**, and it holds the two subtlest pieces of
  logic in the package: the transitive alias veto in `build_evidence` and tie
  handling in `choose_thresholds`.
- **Thresholds are not fitted to your corpus.** The defaults are hand-set.
