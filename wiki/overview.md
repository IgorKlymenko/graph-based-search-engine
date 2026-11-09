# Overview

## The product in five lines

A Slack-native team knowledge base. A workspace installs the app, its public
channels are rendered to text and indexed into an **entity graph in Postgres**,
and questions are answered by walking that graph (surfaces → entities →
Personalized PageRank → passages → LLM). Isolation between customers is enforced
by the database (row-level security keyed on a per-transaction tenant), not by
application code. Notion and Drive are planned as additional sources with the
same trust model. Nothing is in production yet; there is no staging.

## Repo map with status

| Path | Status | What it is |
|---|---|---|
| `CLAUDE.md` | live | Rules for models. The **Database access** section is the single most important text in the repo. |
| `wiki/` | live | This wiki. |
| `ADRs/supabase/tables.md` | live | Schema decisions, 29 entries. |
| `ADRs/supabase/security.md` | live | Roles and tenant isolation, 12 entries plus "what did not work". |
| `ADRs/slack/connector.md` | live | Slack app design, 15 entries. |
| `ADRs/connectors/github.md` | live | The connector contract, seeded mentions, the GitHub and code decisions, 10 entries. |
| `ADRs/brainstorming.md` | live | Open questions before more Slack code is written; blockers first. |
| `server/public/` | **live, canonical** | FastAPI Slack ingress (`src/app.py`), signing helper (`src/security/slack.py`), the ingest blackbox (`src/inject/`), the query path (`src/query/`: retrieval, stance, Slack rendering) and the connectors package (`src/connectors/`: contract, code parsers, GitHub). Python ≥ 3.13, own `uv` environment. |
| `server/tests/public/` | live | 121 tests: signature verification, health, and the live copy's own stage tests under `inject/`. |
| `supabase/` | live | 24 migrations (the only write path to schema), `config.toml`, ad-hoc `snippets/`. |
| `slack/` | live | `manifest.dev.yaml`, one example event payload. |
| `old_server/entity_linking/` | older copy | The same 5 stages as `src/inject` plus `pipeline.py` (orchestrator), `slack.py` (connector, still the only history puller), `ask.py` / `answer.py` (superseded by `src/query/`), `coref_metrics.py`, `demos/` (evals), `DATAFLOW.md`, `RUNNING.md`. |
| `old_server/tests/` | live | 325 collected tests. All stage tests and the 38 tenant-isolation tests are here, not under `server/`. |
| `old_server/pipeline/`, `graph_upload.py`, `query.py`, `reembed_entities.py` | legacy | Neo4j-era extraction: HDBSCAN clustering + LLM resolution + relation extraction. See [legacy.md](legacy.md). |
| `old_server/retrieval/`, `services/` | legacy | Neo4j retrieval library: PPR, approximate PPR, temporal, multihop, beam, DAG, treehop. Spec'd in `old_server/specs/`. |
| `old_server/mcp-server/` | legacy | MCP server over Neo4j for Railway. |
| `old_server/artifacts/` | gitignored | Pipeline outputs. |
| `experiments/arena/` | experimental | Retrieval-method comparison CLI over `old_server/retrieval`. Its `uv` source path points at `../server`, which no longer holds that package. |

## Which copy is canonical

`server/public/src/inject/` is the packaged form of `old_server/entity_linking/`
stages 1–5. `src/inject/main.py` is `entity_linking/pipeline.py` with an
`ingest()` function on top. Same code except:

- imports are `src.inject.*` instead of `entity_linking.*`;
- `1_episodes.py` in `src/inject` has **no `strip_markup`** (Obsidian/Markdown cleanup). The tests for it (`old_server/tests/test_entity_linking_markup.py`, 13 tests) only cover the old copy;
- `2_chains.py` in `src/inject` carries a stale module docstring (still says "Stage 1 … gleaning" and names `1_get_candidates.py`).
- `main.py` in `src/inject` has the resolver crash guards as `unresolved_report` with a test under `server/tests/public/inject/`; the old copy has them inline.

Edit `server/public` for anything that ships. Port a fix to `old_server/entity_linking` only if a test there covers it, until the tests move.

## Working-tree caveat

`git status` at the time of writing shows a restructure in flight: `server/src/*` and `server/tests/auth/*` deleted, `server/public/` and `server/tests/public/` added, `old_server/entity_linking/{5_store,ask,slack}.py`, four test files, `ADRs/slack/`, `ADRs/brainstorming.md` and three migrations **untracked**. `git log` does not describe the current layout. Commit or stash before trusting history.

## Environments

| Thing | Where | Notes |
|---|---|---|
| Postgres | local Supabase, `127.0.0.1:54322`, user `postgres`/`postgres` | `supabase start`; Studio on 54323, API on 54321. Pooler is **disabled** locally, so transaction pooling has never been exercised (open item in every ADR). |
| LLM | OpenAI | `gpt-4.1-mini` for coref and review, `gpt-4.1` for answers, `text-embedding-3-large@1536` for surfaces. |
| Local models | Hugging Face | `nomic-ai/nomic-embed-text-v1.5` (~550 MB, `trust_remote_code`), `en_core_web_sm` (pinned wheel). |
| Slack | dev app "Loupy" (bot `@loupy`, app `A0BKNP93EPL`, workspace Acme Inc) via ngrok | URLs in `slack/manifest.dev.yaml`. |
| Cloud Tasks / Cloud Run | designed, not built | ADR tables 7, 9. |
| Railway | legacy MCP server only | `old_server/mcp-server`. |
