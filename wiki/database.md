# Database

Postgres on Supabase. Schema changes only through `supabase/migrations/`. The
non-negotiable client rules (one transaction, `SET LOCAL`, parameterised
`set_config`, FK write order, webhook dedupe and claim) are in `CLAUDE.md`
§ Database access and are **not repeated here**. This page is the map.

## Tables

| Table | PK | Tenant column | Wall | Server may | Notes |
|---|---|---|---|---|---|
| `tenants` | `id` | `id` | yes | read | `name` non-blank. |
| `users` | `id` | — | none; **grant revoked** | nothing | `auth_user_id` nullable unique → `auth.users`; `email` lowercase + partial unique; `email_source ∈ {slack, oidc}` present iff email is. ADR tables 11, 13. |
| `memberships` | `id` | `tenant_id` | yes | read | `UNIQUE (user_id, tenant_id)`; `role ∈ {owner, admin, member}`. ADR tables 12, 19. |
| `connector_installations` | `(source, external_id)` | `tenant_id` | none; **grant revoked** | only via `current_tenant_id()` / `current_token_ref()` | `token_ref` is a secret-store handle; `uninstalled_at` revokes instantly; `UNIQUE (source, external_id, tenant_id)` is the FK target for `slack_users`. ADR security 4, 8, 9. |
| `slack_users` | `(slack_team_id, slack_user_id)` | — | none; **grant revoked** | nothing | `source` fixed to `'slack'` by CHECK so it can FK the composite key. |
| `subscriptions` | `id` | `tenant_id` | yes | read | `sub_status ∈ {active, cancelled, trial, past_due, expired}`; one `active`/`trial` per tenant (partial unique). |
| `files` | `id` (uuid5 from code) | `tenant_id` | yes | rw | `source ∈ {slack, gdrive, notion, upload, github}`; `title`, `url` filled by `5_store.upload` from the store when a connector supplied them; `UNIQUE (tenant_id, source, external_id)`; `UNIQUE (id, tenant_id)` for composite FKs; `changed_at` trigger fires only when `full_text_hash` or `render_version` changes; `removed_at` cascades to episodes via trigger. |
| `episodes` | `id` | `tenant_id` | yes | rw | FK `(file_id, tenant_id)` → files; `expired_at ≥ created_at` CHECK; no ordinal (ADR tables 26). |
| `surfaces` | `(tenant_id, surface)` | `tenant_id` | yes | rw | `embedding vector(1536)` NULL until backfilled; HNSW cosine index; `pg_trgm` GIN on `surface`; `embed_model` default `text-embedding-3-large@1536`; `is_name boolean` (NULL = not judged, read as a name; ADR tables 30). No `n_entities`/`n_episodes` (ADR tables 29). |
| `entities` | `(tenant_id, id)` | `tenant_id` | yes | rw | `display_name` is a **human override only**; no `merged_into` (ADR tables 28). |
| `chains` | `(tenant_id, id)` | `tenant_id` | yes | rw | `entity_id` nullable, FK → entities **ON DELETE RESTRICT**; FK `(tenant_id, episode_id)` → episodes CASCADE; `UNIQUE (tenant_id, episode_id, content_key)` is the upsert key. |
| `chain_surfaces` | `(tenant_id, chain_id, surface)` | `tenant_id` | yes | rw | `is_head` (exactly one per chain, partial unique), `n_mentions` (the one count not re-derivable). FK → surfaces has no ON DELETE, so a surface with chains cannot be deleted. |
| `internal.events` | `(source, external_id, event_id)` | `tenant_id` nullable, **not the key** | walled on `(source, external_id)` = declared connector + workspace | `app_ack` insert; `app_user` select/update | `status ∈ {queued, processing, done, failed}`, `retry_num`, `payload jsonb`, `error`, timestamps. In schema `internal` so PostgREST never exposes it. |

Relationship spine (every query walks it one way or the other):

```
files ──(id,tenant_id)──▶ episodes ──(tenant_id,id)──▶ chains ──(tenant_id,id)──▶ chain_surfaces ──(tenant_id,surface)──▶ surfaces
                                                         │ entity_id
                                                         ▼
                                                      entities
edges  = chains ⋈ chains on episode_id, a.entity_id < b.entity_id   (no edge table; ADR tables 24)
```

## Functions, views, triggers

| Object | Kind | Runs as | What |
|---|---|---|---|
| `public.current_tenant_id()` | SQL, STABLE, SECURITY DEFINER, `search_path=''` | owner (`postgres`, has BYPASSRLS — load-bearing) | `app_ingest` → `app.tenant_id`; anyone else → lookup in `connector_installations` where `uninstalled_at IS NULL`. Branches on `current_setting('role')`, never `current_user`. |
| `public.current_token_ref()` | same | same | The one `token_ref` for the declared workspace. Execute granted to `app_connector` only. |
| `public.current_tenant_ids()`, `public.has_role_in(t, roles[])` | same | same | End-user helpers over `auth.uid()`. **Unused**; granted to `authenticated` for later. |
| `public.entity_display_name(tenant, entity)` | SQL, STABLE, security invoker | caller | `display_name` override, else most frequent `is_head` surface (ties: longer, then lexical). The only sanctioned way to read a name. |
| `public.entity_names` | VIEW, `security_invoker` | caller | id, name, `is_override`, chain and episode counts. For browsing, not hot loops. |
| `cascade_file_removal()` | trigger AFTER UPDATE on files | — | when `removed_at` flips from NULL, sets `episodes.expired_at`. |
| `extensions.moddatetime` | trigger | — | `updated_at` on tenants, users, memberships, subscriptions, surfaces, entities, chains; `changed_at` on files (conditional). |

## Roles

| Role | Login | Member of | Declares | Ceilings | Purpose |
|---|---|---|---|---|---|
| `app_user` | no | — | — | 30 s / 10 s (belt and braces) | Privilege bundle. Policies name it. Never assumed directly. |
| `app_connector` | no | `app_user` | `app.connector`, `app.workspace_id` | 30 s / 10 s | Derives tenant. Cannot claim. Only role with `current_token_ref()`. |
| `app_ingest` | no | `app_user` | `app.tenant_id` | 30 s / 10 s | Claims tenant. Used by `5_store.upload` and `ask.py`. |
| `app_ack` | no | **not** `app_user` | `app.connector`, `app.workspace_id` | 5 s / 5 s | Insert into `internal.events` and nothing else. |
| `app_connector_login` | yes, NOINHERIT | `app_connector` | — | 30 s / 10 s, 30 connections | Credential. Must `SET LOCAL ROLE` or every query is `permission denied`. |
| `app_ingest_login` | yes, NOINHERIT | `app_ingest` | — | 30 s / 10 s, 10 connections | Credential. |
| `app_ack_login` | yes, NOINHERIT | `app_ack` | — | 5 s / 5 s, 40 connections | Credential for the public endpoint. |
| `postgres` | yes | all four bundles | — | — | Migrations and tests only. Carries BYPASSRLS: walls apply only *after* `SET ROLE`. |

Ceilings are `statement_timeout` / `idle_in_transaction_session_timeout`. They only take effect on the role that logs in (ADR security 12). Login roles are created **without a password**; set it out of band. `app_ingest` and `app_connector` (not `app_user`) have USAGE on schema `extensions` for the `vector` type — a missing grant reports as `type "vector" does not exist`, and so does a `search_path` without `extensions`, which is the login roles' default: set it with `SET LOCAL` inside the transaction (`query.retrieve.enter`). Keep network calls out of the transaction: the 10-second idle ceiling is real and terminates the session.

## Policies in one line

Every tenant-scoped table: `ENABLE` + `FORCE ROW LEVEL SECURITY`, one **RESTRICTIVE** wall `tenant_id = public.current_tenant_id()` (`id =` on tenants) `FOR ALL TO app_user`, plus permissive doors: `FOR ALL USING (true)` on content tables, `FOR SELECT` on tenants/memberships/subscriptions. Unset context → `NULL = NULL` → every wall denies. Reasoning: ADR tables 18, security 2, 10.

Invariant to re-check when adding a table — every table is either walled or ungranted, never grantable-and-unwalled. Query in `ADRs/supabase/security.md` entry 10.

## Which code path uses which role today

| Code | Role | Context |
|---|---|---|
| `src/inject/src/5_store.upload` | `app_ingest` | `app.tenant_id` |
| `src/query/retrieve.answer_question` from the Slack ingress | `app_connector` | `app.connector = 'slack'`, `app.workspace_id = team_id` |
| `src/query/retrieve` CLI (`embed`, `ask --tenant`) and the old `ask.py` | `app_ingest` | `app.tenant_id` |
| `old_server/tests/test_db_tenant_isolation.py` | all of them, via `SET LOCAL ROLE` from `postgres` | per test |
| `server/public/src/app.py` | via `answer_question` above, from `CONNECTOR_DATABASE_URL` | — |

## Migrations (26, in order)

| File | Does |
|---|---|
| `20260728210208` | `moddatetime` extension |
| `20260728210225` | `tenants` |
| `20260728210306` | `users` |
| `20260728210721` | `memberships` |
| `20260728214757` | `connector_installations` |
| `20260728214812` | `slack_users` |
| `20260729145256` | `subscriptions` |
| `20260730172836` | `files` + `changed_at` trigger |
| `20260730173432` | `episodes` + `cascade_file_removal` |
| `20260804231459` | `vector` extension |
| `20260804231802` | `surfaces` + HNSW |
| `20260804232245` | `entities` |
| `20260804234008` | `chains` |
| `20260805000110` | `chain_surfaces` |
| `20260805001000` | `entity_display_name()` |
| `20260805001500` | `entity_names` view (no grant of its own; the roles migration's `ON ALL TABLES` covers views) |
| `20260805120000` | `pg_trgm` + GIN on surfaces |
| `20260806183012` | resolver and helper functions |
| `20260806183343` | `app_user`, `app_connector`, `app_ingest`, `app_ack`; grants and revokes |
| `20260806190227` | all walls and doors |
| `20260806190500` | login roles, ceilings, connection limits |
| `20260806190506` | intentionally empty; it duplicated `20260806190227` |
| `20260806200000` | `internal.events` with workspace wall and role-split doors |
| `20260812141500` | USAGE on `extensions` to `app_ingest`, `app_connector`; documents the `pg_stat_statements` residual |
| `20260917150000` | `surfaces.is_name` — the name judge's verdict; applied by `supabase db reset` on 2026-09-19 |
| `20260918120000` | `github` added to the three `source` CHECKs (constraints recreated by their generated names); applied by `supabase db reset` on 2026-09-19, isolation suite 39/39 |

Both files were corrected on 2026-09-17 but a clean `supabase db reset` has not been run since (Docker unavailable). Run it before trusting a fresh environment; see [PROPOSALS.md](PROPOSALS.md) S4.

### Rules for a new migration

- One purpose per file, so a policy review is a policy diff (ADR tables 22). Forward-only; restore from backup is the rollback.
- `text` + CHECK, never `varchar(n)`; `timestamptz`, never `timestamp` (ADR tables 21).
- New tenant-scoped table: `tenant_id`, `ENABLE` + `FORCE RLS`, the restrictive wall, a permissive door, and — if it should be read-only or unreachable — a `REVOKE`, not a missing policy.
- New `SECURITY DEFINER` function: `SET search_path = ''`, every name schema-qualified, `REVOKE EXECUTE … FROM PUBLIC`, then grant to exactly the role that needs it.
- Never put a password, token, or tenant id in a migration.
- After applying, run `old_server/tests/test_db_tenant_isolation.py` (39 tests; some skip without login passwords).

## Limits to design around

`statement_timeout` 30 s (5 s on `app_ack`), `idle_in_transaction_session_timeout` 10 s (5 s), connection limits 30 / 10 / 40, PostgREST `max_rows` 1000, pgvector indexable dimension ≤ 2000 (hence 1536). Chunk long operations; hitting a cap is a bug, not load.

## Unverified

Behaviour under the **transaction pooler**. `config.toml` has `[db.pooler] enabled = false`, so it has never run locally. `SET LOCAL` is the correct primitive for it; it is untested.
