# Conventions

What the codebase already does well, written down so it keeps doing it. Rules a
model must obey are in `CLAUDE.md`; this page is the *house style* observed in
the live code. When the two disagree, `CLAUDE.md` wins.

## Tooling

- `uv` only. Each of `server/public`, `old_server`, `experiments` is its own project with its own lock; run commands from inside the project directory. Never `pip`.
- Ruff, line length 88; `old_server` selects `E, F, I`. Run `uv run ruff check .` before committing. Current debt is listed in [testing.md](testing.md).
- Python ≥ 3.13 in `server/public`, ≥ 3.12 elsewhere. `from __future__ import annotations` at the top of every module. `X | None`, not `Optional[X]`.
- Type hints everywhere; `NamedTuple` for records that cross a stage boundary (`Episode`, `FileDiff`, `Span`, `ChainRecord`, `Evidence`, `Thresholds`, `Encoded`, `ChainVector`).

## Package layout

- **Digit-prefixed stage modules.** Stages are `1_episodes.py … 5_store.py` so a directory listing is the run order. Python cannot import them, so `_stages.load(filename, alias)` registers them in `sys.modules` and `__init__` re-exports them. Import stages **only** through the aliases:

  ```python
  from src.inject import episodes as EP, chains as CH, encoding as EN, pairing as PR, store as ST
  ```
  A stage that needs another stage uses the same `load(...)` call (see `2_chains.py` top). Support modules (`utils/spans.py`, `utils/prompts.py`) have ordinary names.
- Orchestration has no logic of its own (`main.py`, `pipeline.py`). Defaults for `source`, `model`, `workers` are resolved in `ingest()` from the stage constants so each is defined once.
- Every stage module is also a CLI (`if __name__ == "__main__": main()`), and every CLI is a thin wrapper over one function the API can call.
- Heavy imports (`torch`, `hnswlib`, `psycopg`, `sklearn`) happen **inside** the function that needs them. Importing a module never fetches a model or opens a connection; tests and `--dry-run` stay offline.
- Process-wide clients are lazy singletons (`chains._get_client`, `encoding._load`, `spans._nlp`).

## Naming

- `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE` constants, handler functions prefixed `handle_` (`ask.handle_embed`, `ask.handle_ask`).
- Private helpers start with `_`. Stage aliases are two capitals (`EP`, `CH`, `EN`, `PR`, `ST`, `SP`).
- Ids are named by what they identify: `file_id`, `episode_id`, `chain_id`, `entity_id`, `candidate_id`; never bare `id` outside SQL.
- Vocabulary is fixed — see [glossary.md](glossary.md). "Surface" is a string; "chain" is per-episode; "entity" is corpus-wide. Do not introduce "mention" as a stored object (ADR tables 27).

## Docstrings and comments

The distinctive practice here: **comments carry evidence**. A non-obvious choice is annotated with what was measured, what the alternative scored, and what breaks otherwise. Examples worth copying:

- `pairing.cluster` — why `cannot_link` is off (0.774 vs 0.945 CoNLL).
- `store.upload` — why `created_at` is `LEAST(now(), removed_at)`.
- `ask.collect_episodes` — why mean, not sum or max (25th / 10th / 7th).
- `20260812141500_grant_extensions_usage.sql` — the misleading error text and the residual exposure.

Rules:

- Module docstring says what the module produces, what it costs, and the one invariant that must hold. Public functions get a docstring; the *why* goes there, not inline.
- When a choice was measured, write the number and the rig. When something was tried and removed, say so and say when to bring it back (`entities` table comment on `merged_into`; `stable_entity_ids` docstring).
- Prompts are tuned artifacts: long lines, `# ruff: noqa: E501`, never rewrapped, edited as behaviour changes.
- Mark the unverified: "*(unverified)*" in ADRs, "not measured" in code.
- Comments explain, they do not narrate. No "increment i".

## SQL and migrations

- Comments in migrations explain the reason and the failure mode that would otherwise be silent. Follow the existing files.
- Every statement in application code is an upsert keyed the way the schema is keyed, so re-running is a no-op. Conflict target is the **natural** key where one exists (`chains` on `(tenant_id, episode_id, content_key)`).
- Composite FKs carry `tenant_id` so a cross-tenant reference is unrepresentable.
- Read names with `entity_display_name()`; write `display_name` only for a human correction.
- Migration checklist and role rules: [database.md](database.md) § Migrations.

## Tests

- Test names are sentences stating the invariant: `test_the_other_tenant_is_invisible_not_merely_filtered`, `test_forgetting_set_role_fails_loudly`. Module docstring states what property the file protects.
- Regressions pin the failure they fix, including the *known weaknesses* of an interim design (`test_entity_linking_stable_ids.py` asserts the cases where id stability does **not** hold, on purpose).
- Database tests skip, not fail, when nothing is listening on `TEST_DATABASE_URL`; each test is one transaction shaped like a request (`run()` helper).
- Offline by construction: no test downloads a model or calls OpenAI. Stubs live in `old_server/tests/conftest.py`.
- New feature → tests; bug fix → regression test (`CLAUDE.md`).

## ADRs

- Format: numbered entry, status (**Accepted / Deferred / Open / Superseded**), then **Context, Decision, Why, Cost**, optionally **Revisit when**. One screen per decision.
- Supersede, never edit, once shipped. Keep superseded entries for the reasoning.
- Every log ends with **What did not work** (a table) and **Open items**. Measured claims are marked *measured*; third-party limits are marked *unverified*.
- New decision about schema → `ADRs/supabase/tables.md`; about roles/isolation → `security.md`; about the Slack app → `slack/connector.md`. Add a one-liner to [decisions.md](decisions.md).

## Commits and PRs

- Imperative summary; trailers `Reported-by:` / `Github-Issue:#n` when applicable; the attribution line the session specifies. Never mention the tool that wrote it.
- PR description: problem, then how it is solved; reviewer `ArthurClune`.
- Schema, server, tests and docs for one change go in one commit (ADR tables 23).
- Update the relevant wiki page in the same commit as the code it describes.
