# Architecture

Read this for the shape. Details per box are on the linked pages.

## End-to-end flow

```
Slack ──POST /slack/event──▶ ingress  server/public/src/app.py
        verify HMAC over raw bytes + 5-min replay window
        ▶ url_verification handshake (returns the challenge body)
        ▶ dedupe on event_id            [today: in-memory set; designed: internal.events INSERT … ON CONFLICT DO NOTHING as app_ack]
        ▶ drop bot_id / bot_message
        ▶ enqueue                       [today: FastAPI BackgroundTasks; designed: Cloud Tasks → POST /work]
        ▶ 200 within Slack's 3-second budget
                                                 │
worker (app_connector) ── [designed: claim event with one conditional UPDATE; today: BackgroundTasks in the ingress]
        ▶ resolve token via public.current_token_ref()   [designed]
        ▶ render thread/day documents   old_server/entity_linking/slack.py   [CLI today]
        ▶ ingest()                      server/public/src/inject/main.py
        ▶ answer_question()             server/public/src/query/retrieve.py  (as app_connector for team_id)
        ▶ answer / reframe / clarify    render_slack → chat.postMessage in thread
                                                 │
src/inject   1_episodes ─▶ 2_chains (LLM) ─▶ 3_encode (nomic) ─▶ 4_link (LLM on the review band) ─▶ 5_store (ONE transaction)
                                                 ▼
Postgres     files → episodes → surfaces, entities → chains → chain_surfaces        (FK order; RLS walls on every one)
                                                 │
query        question ─embed─▶ surfaces (HNSW; is_name) ─▶ chain_surfaces ─▶ chains ─▶ entities
             ─PPR over PMI-weighted co-occurrence (self-join on chains)─▶ cutoff ─▶ episodes ─▶ stance + answer with citations
                                                 server/public/src/query/{retrieve,answer}.py
```

## Two trust paths, three roles

| Path | Role | Declares | Tenant comes from |
|---|---|---|---|
| Anything with a workspace (Slack, later Notion/Drive) | `app_connector` | `app.connector`, `app.workspace_id` | `connector_installations`, resolved by the database |
| Uploads, backfills, the query CLI today | `app_ingest` | `app.tenant_id` | Claimed by the server; isolation here is application logic |
| Public webhook ACK | `app_ack` | `app.connector`, `app.workspace_id` | Can only insert into `internal.events` |

The connector role **cannot** claim a tenant; the ack role **cannot** read the corpus. Full table in [database.md](database.md) § Roles; reasoning in `ADRs/supabase/security.md` entries 1–3, 6.

## Built vs designed

| Piece | State | Evidence |
|---|---|---|
| Signature verification, replay window, handshake, bot filter, holding reply | built, tested | `app.py`, `server/tests/public/auth/` |
| Event dedupe in Postgres, worker claim | schema built, app not wired | `internal.events` migration; `app.py` uses `_seen_event_ids` |
| Cloud Tasks enqueue, `/work` endpoint, OIDC | designed only | ADR tables 7; security open items |
| OAuth install → tenant + subscription + installation | designed only | ADR slack 1–3; no callback route exists |
| Slack history → documents | built, CLI only | `old_server/entity_linking/slack.py` |
| Ingest stages 1–5 | built, tested in `old_server/tests` | `src/inject/` |
| Entity-id carry-forward across runs | built | `store.match_entities` |
| Surface embedding backfill | built, manual | `ask.py embed` |
| Query path (seed → PPR → cutoff → passages → stance + answer) | built; wired to the Slack ingress and a CLI | `src/query/retrieve.py`, `app.handle_event` |
| Managed PPR service, precomputed edge payload | designed only | ADR tables 24 |
| JIT provisioning of Slack users, memberships | designed only | ADR tables 14 |
| `connector_channels`, progressive backfill, silent mode | designed only | ADR slack 11–13 |
| Retention, deletion, per-channel ACL | open | ADR slack open items A, B |

## Component table

| Component | File | Entry point | Talks to |
|---|---|---|---|
| Ingress | `server/public/src/app.py` | `POST /slack/event`, `GET /health` | Slack Web API (`chat.postMessage`) |
| Signing | `server/public/src/security/slack.py` | `slack_sign(body, secret, ts)` | — |
| Ingest blackbox | `server/public/src/inject/main.py` | `ingest(store_path=…, notes_dir=…, tenant_id=…, dsn=…)` | OpenAI, nomic, Postgres |
| Stage loader | `server/public/src/inject/_stages.py` | `load("1_episodes", "src.inject.episodes")` | — |
| Span inventory | `server/public/src/inject/utils/spans.py` | `parse_spans`, `mark_text`, `locate_new_spans` | spaCy |
| Prompts | `server/public/src/inject/utils/prompts.py` | `CHAIN_SYSTEM_PROMPT`, `MERGE_REVIEW_PROMPT`, `ENTITY_CHOICE_PROMPT` + schemas | — |
| Slack connector | `old_server/entity_linking/slack.py` | `python -m entity_linking.slack` | Slack Web API (read) |
| Connector contract | `server/public/src/connectors/base.py` | `sync(store, connector, tenant_id=…)` | the store |
| Code parsers | `server/public/src/connectors/code/` | `parser_for(path).parse(path, text, tree=…)`, `render(parsed, path=…)` | — |
| GitHub connector | `server/public/src/connectors/github/` | `python -m src.connectors.github` | GitHub REST API (read) |
| Query | `server/public/src/query/retrieve.py`, `answer.py` | `answer_question(...)`; `python -m src.query.retrieve embed|ask` | OpenAI, Postgres (as `app_connector` for a workspace or `app_ingest` for a tenant) |
| Schema | `supabase/migrations/*.sql` | `supabase db reset` | — |

## Adding a source (Notion, Drive, …)

Per `ADRs/supabase/security.md` entry 4 it is "a row, not a role". The code side is a class satisfying `connectors.base.Connector` fed to `base.sync` ([connectors.md](connectors.md)); the GitHub connector is the worked example. The checklist that actually falls out of the code:

1. Install path writes one row to `connector_installations (source, external_id, tenant_id, token_ref)`. The token goes to the secret store; only the handle is stored.
2. Add the source name to every `CHECK (source IN (…))`: `connector_installations`, `files`, `internal.events`, by recreating the constraints in one migration (see `20260918120000_add_github_source.sql`). Today all three list `slack`, `gdrive`, `notion`, `github`; `files` also allows `upload`.
3. Add an entry to `RENDER_VERSIONS` in `1_episodes.py`. Bump it whenever the renderer's output changes; the bump is what invalidates `full_text_hash` for that source only.
4. Write a renderer that emits **one blank-line-separated paragraph per appendable unit** (see `slack.render_message`) and resolves all markup **before** the text is hashed (ADR slack 9). Pick an `external_id` that is the source's own stable id, not a path or title.
5. Yield `Document`s from `documents()` and gone ids from `removed()`; `base.sync` calls `episodes.sync_file` and `episodes.retire_file` for you. Push-based sources must retire documents from explicit delete events; only enumerable sources may infer deletion from absence, and only from a whole listing (`Connector.removed` docstring). A structured source attaches `mentions` and they become seeds (ADR connectors 2).
6. Run under `app_connector` with `app.connector = '<source>'` and `app.workspace_id = <external_id>`. No new role, no new policy.
7. Add a row to `old_server/tests/test_db_tenant_isolation.py` seed like `NOTION_WS` and assert `test_a_second_source_needs_no_new_role_or_policy` still holds.
