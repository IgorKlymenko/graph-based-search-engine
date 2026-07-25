# Claude Code task — typed Fact-extraction pipeline (new notebook)

## Objective
Build a typed information-extraction pipeline that turns source text + already-resolved
entities into **reified Fact records** ready to write as Fact nodes (`HAS_FACT` edges to
their subject entity) in our bi-temporal Neo4j graph. The pipeline has four stages:

1. **Schema-guided LLM extraction** — identify each fact, its subject entity, predicate,
   and a **verbatim value string copied from the source**. The LLM does NOT compute the
   typed number.
2. **Deterministic normalizer** — pure-Python parsing of the verbatim string into a typed
   value (money → integer minor units + currency; counts → int; dates → ISO). No LLM here.
3. **Span re-verification** — for every numeric/temporal fact, confirm the value actually
   appears in the cited source span. Ungrounded facts are flagged/rejected.
4. **Confidence** — a rule-based score written onto each Fact record.

Deliver this as a **new Jupyter notebook** that consumes cached entities/chunks.

## Before you write anything — INSPECT FIRST
Do not assume the environment. Open the current notebook and the cache and report back what
you find before generating the new notebook:
- Where are cached entities stored, and do they carry **canonical/resolved entity IDs**, or
  only raw mentions? (This is a hard gate — see Stop-and-ask.)
- Are the **source chunks** the entities came from cached, and do they have character
  offsets / a stable `doc_id`? Fact provenance and span verification need these.
- What **LLM client** does the existing notebook already use (Anthropic SDK, OpenAI,
  instructor, etc.)? Reuse it; do not introduce a new one.
- Is there a live **Neo4j connection** and a known Fact-node schema already? If not, the
  graph write is out of scope for this notebook (stage the records instead).

## Caching protocol (important)
This new notebook should be fast and reproducible — it must read inputs from cache, not
recompute upstream stages inline. If a required input is **not** cached (e.g. resolved
entities aren't persisted, or chunks lack offsets):
- **Add the caching code to the CURRENT notebook**, run it to produce the cache, and only
  then write the new notebook that consumes it.
- Do not bake expensive upstream recomputation into the new notebook.

## The Fact record contract (Pydantic v2)
Define this as the single source of truth. Adapt field names to match the existing graph
schema you find, but keep these slots:

- `subject_entity_id: str`  — MUST be a canonical resolved ID from the cache, not a mention.
- `predicate: str`          — from a **closed vocabulary** (enum below); unknowns go to
  `"other"` with the raw predicate preserved in `predicate_raw`, never silently dropped.
- `value_type: Literal["money","count","percentage","date","duration","text"]`
- `raw_value_str: str`      — verbatim, copied from source exactly. Always kept.
- Typed value (by `value_type`):
  - money → `amount_minor: int` (integer minor units) + `currency: str` (ISO 4217). **Never float.**
  - count → `count: int`
  - percentage → `basis_points: int` (exact; avoid float)
  - date/duration → ISO 8601 string(s)
  - text → no typed value, qualitative fact
- `valid_from: str | None`, `valid_to: str | None` — world-time validity (bi-temporal).
- `temporal_inferred: bool` — True if valid-time was implied, not stated in source.
- `doc_id: str`, `char_start: int`, `char_end: int`, `source_quote: str` — provenance.
  Store the pointer + span, NOT the raw document blob.
- `confidence: float`       — 0..1, derived per stage 4.
- `extractor_version: str`  — stamp the model + schema version for future migrations.

Starter predicate enum ({{REVIEW — this is domain-specific, owner must confirm}}):
`revenue, funding_raised, valuation, headcount, price, growth_rate, count_of, duration,
date_event, other`.

## Stage detail

### Stage 1 — schema-guided LLM extraction
- Input: a cached chunk + the resolved entities present in it (pass the entity IDs/names in
  as context so the model maps facts onto KNOWN entities, not free-floating ones).
- Use structured output (tool/function calling or JSON-schema-constrained) bound to the
  Pydantic contract. `instructor` is a clean option if it's already available; otherwise use
  the notebook's existing client's structured-output mode.
- The model fills semantic slots and emits `raw_value_str` **verbatim**. It must NOT output a
  parsed/typed number as the value — leave typed fields empty at this stage.
- Optionally let the model emit a self-rated confidence, but treat it as weak signal only.

### Stage 2 — deterministic normalizer (no LLM)
- Money: parse `raw_value_str` with `price-parser` (returns a `Decimal` amount + currency
  symbol) → resolve ISO currency → convert to minor units using the currency's exponent via
  `babel.numbers.get_currency_precision(currency)` (USD→2, JPY→0). Use `decimal.Decimal`
  throughout; cast to `int` only at the end. Never `float`.
- Counts: integer parse, handling "12", "300 ppl", "~12" (strip approximators, record that it
  was approximate).
- Percentages: → integer basis points.
- Dates/durations: `dateparser` (natural language) or `python-dateutil` → ISO 8601. Mention
  Duckling as an optional upgrade if a service is acceptable, but default to the Python libs.
- On any parse failure: do not guess — leave typed value null, mark normalization failed
  (drives confidence down), keep `raw_value_str`.

### Stage 3 — span re-verification
- Deterministically locate `raw_value_str` (and its digits) within `source_quote` /
  the chunk at `[char_start:char_end]`. If absent → the LLM likely hallucinated or warped it;
  set a verification flag false and drop confidence sharply (or route to a reject bucket).
- Apply the SAME check to the temporal slot — dates are the highest-risk, least-caught slot.
  If valid-time was inferred rather than found in the span, set `temporal_inferred=True`.

### Stage 4 — confidence
- Rule-based composite (document the formula in a markdown cell). Suggested: start 1.0;
  subtract for normalization fallback/failure, for failed span verification, for
  `temporal_inferred`, for `predicate == "other"`. Note explicitly that this is NOT model
  logprob-based and is a heuristic. Write the result to `confidence`.

## Hard constraints (do not violate — these are settled decisions)
- Facts key onto **canonical** entity IDs. If the cache only has unresolved mentions, STOP.
- LLM finds the fact; **deterministic code types the value.** Never trust the LLM's number.
- **No float for money.** Integer minor units + ISO currency, always with the unit attached.
- Closed predicates preferred; unknowns bucketed to `other` with raw text kept, never dropped.
- Provenance is a pointer + offsets, never the raw document stored in the node.
- Identical output contract regardless of source genre (chat / docs / transcripts).
- Stamp `extractor_version` on every record.

## Stop and ask the user if
- Cached entities are unresolved mentions with no canonical IDs (resolution must precede this).
- Chunks have no offsets / no stable `doc_id` (provenance + span verification impossible).
- No existing LLM client to reuse, or the predicate vocabulary needs domain sign-off.

## Deliverable / done
- New notebook, cells in pipeline order, each stage a small testable function, with brief
  markdown explaining each stage.
- Pydantic models for the contract.
- A **small demo run** at the end over a handful of cached chunks: print the typed Fact
  records with their confidence and verification flags. Do NOT run the full corpus — leave
  scaling to the user.
- Graph write: only if a Neo4j connection + Fact schema already exist, and as a clearly
  separated, guarded final cell. Otherwise stage the records and flag the write as next step.
