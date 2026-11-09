# Wiki — the model-facing map of this codebase

Purpose: get a person or a model to the right file with the right context in as
few tokens as possible, and stop the practices this codebase already follows
from eroding. Pages are dense on purpose. Read the routing table, open only the
pages it names, then open code.

Companion to `CLAUDE.md` (rules a model must follow) and `ADRs/` (why things
are the way they are). This wiki is *what and where*; it never re-argues a
decision — it links the ADR entry number instead.

## Routing table

| You are about to… | Read first | Then open |
|---|---|---|
| Touch anything that talks to Postgres | [database.md](database.md), `CLAUDE.md` § Database access | `server/public/src/inject/src/5_store.py`, `supabase/migrations/20260806*` |
| Change extraction, linking, or entity ids | [ingest-pipeline.md](ingest-pipeline.md), [gotchas.md](gotchas.md) | `server/public/src/inject/` |
| Write or tune a query, seeding, or PPR | [retrieval.md](retrieval.md), [measurements.md](measurements.md) | `old_server/entity_linking/ask.py`, `old_server/entity_linking/DATAFLOW.md` § Query |
| Slack ingress, OAuth, manifest, rendering | [slack.md](slack.md) | `server/public/src/app.py`, `old_server/entity_linking/slack.py`, `ADRs/slack/connector.md` |
| Add or edit a migration | [database.md](database.md) § Migrations, [decisions.md](decisions.md) | `supabase/migrations/` |
| Add a connector (Notion, Drive, …) | [connectors.md](connectors.md), [architecture.md](architecture.md) § Adding a source | `server/public/src/connectors/base.py`, `1_episodes.py` `RENDER_VERSIONS`, `files.source` CHECK |
| Index code, add a language parser, touch the GitHub sync | [connectors.md](connectors.md) | `server/public/src/connectors/{code,github}/`, `ADRs/connectors/github.md` |
| Write or run tests | [testing.md](testing.md), [conventions.md](conventions.md) | `old_server/tests/`, `server/tests/public/` |
| Run anything locally | [runbook.md](runbook.md) | — |
| Understand why something is the way it is | [decisions.md](decisions.md) | `ADRs/` |
| Look up a term | [glossary.md](glossary.md) | — |
| Work in `old_server/pipeline`, `retrieval`, `mcp-server`, `experiments` | [legacy.md](legacy.md) | — |
| Propose or evaluate a change | [PROPOSALS.md](PROPOSALS.md), [experiments.md](experiments.md) | `old_server/entity_linking/demos/eval_*.py` |
| Park a product idea, or pick one up | [IDEAS.md](IDEAS.md) | [PROPOSALS.md](PROPOSALS.md) once it is scoped |

## Pages

| Page | One line |
|---|---|
| [overview.md](overview.md) | What the product is, repo map with live/legacy status, working-tree caveats |
| [architecture.md](architecture.md) | End-to-end flow, trust paths, built-vs-designed, adding a source |
| [ingest-pipeline.md](ingest-pipeline.md) | Stages 1–5, identity formulas, store shape, knobs, models, costs |
| [database.md](database.md) | Schema, keys, roles, RLS, functions, limits, migration list and rules |
| [retrieval.md](retrieval.md) | Surfaces → entities → PPR → passages → answer; knobs and their evidence |
| [slack.md](slack.md) | Ingress endpoint, signature, manifest, connector rendering, doc kinds |
| [connectors.md](connectors.md) | The connector contract, seeded mentions, the code parsers, the GitHub connector |
| [conventions.md](conventions.md) | Code, comment, docstring, test, ADR and commit conventions actually in use |
| [testing.md](testing.md) | Where tests live, how to run them, what needs a database, lint state |
| [runbook.md](runbook.md) | Copy-paste commands for every local workflow |
| [decisions.md](decisions.md) | One-line index of every ADR entry with status and pointer |
| [measurements.md](measurements.md) | Every measured number in one place, with the rig it came from |
| [gotchas.md](gotchas.md) | Traps that have already bitten once; read before changing the named code |
| [IDEAS.md](IDEAS.md) | Product ideas before they are proposals: the why, the signals already available, what could go wrong |
| [glossary.md](glossary.md) | Surface, chain, episode, evidence, wall, door, … |
| [legacy.md](legacy.md) | What the Neo4j-era code is, and whether to touch it |
| [experiments.md](experiments.md) | Mini-experiments run against the proposals on 2026-09-17: method, numbers, caveats |
| [PROPOSALS.md](PROPOSALS.md) | Efficiency, customer-feature and security proposals with evidence and measured status |

## Rules for this wiki

- One fact, one place. If an ADR or a docstring already states it, link; do not restate.
- Every page is derived from the repo as of **2026-09-17**. When code moves, fix the pointer in the same commit.
- Numbers come from measured results (`ADRs/`, `old_server/entity_linking/DATAFLOW.md`, docstrings). Never invent one; never round one.
- `PROPOSALS.md` is the only page that holds opinions. Everything else is fact plus a pointer.
- Paths are repo-relative. Function names are `module.function`. Stage modules are referred to by their alias (`episodes`, `chains`, `encoding`, `pairing`, `store`), see [conventions.md](conventions.md) § Digit-prefixed modules.
- Add a row to the routing table before adding a page. If a page cannot be routed to, it is not needed.
