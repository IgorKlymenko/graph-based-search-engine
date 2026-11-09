# Decisions index

One line per ADR entry. Status: **A** accepted · **D** deferred · **O** open ·
**S** superseded. Read the entry before changing what it governs; do not
re-argue it here.

## `ADRs/connectors/github.md`

| # | S | Decision |
|---|---|---|
| 1 | A | A connector yields `Document`s; `base.sync` is the only door into the store; `removed()` after `documents()`. |
| 2 | A | Structured sources carry references as `mentions` → per-episode `seeds`; stage 2 folds them into the model's chains (`merge_seeded`); the model still reads the prose. |
| 3 | A | Seeds are proper names; the head is the longest, i.e. the qualified name. |
| 4 | A | A GitHub file is `repo/{repo_id}/{path}`; title `owner/name/path`; url `…/blob/{ref}/{path}`. |
| 5 | A | Cursor per repo = head commit + `{path: blob sha}` + render version; same head → 1 request; measured 28 → 0 fetches. |
| 6 | A | Python via stdlib `ast`; languages behind `PARSERS` by extension; markdown via `strip_markup`; the rest skipped. |
| 7 | A | References only where the file resolves them; builtins, stdlib, calls on locals dropped. |
| 8 | A | One paragraph per symbol, no line numbers or bodies; the module paragraph is the index. |
| 9 | A | Import root = first ancestor without `__init__.py`, from the tree; never by directory name. |
| 10 | A/O | Token from `GITHUB_TOKEN`, CLI as `app_ingest`; App install (installation id as workspace, `app_connector`) is open B. |
| 11 | A | Ambiguous short names render qualified as references and plain text on definitions (`render.Style`); measured on four repos, false merges 303 → 1, 2039 → 2, 2564 → 12, 2180 → 12. |
| 12 | A | Seeded surfaces are names by construction (`base.name_seeds`); the judge rejected 7 in 10 identifiers. |
| 13 | A | `ChainRecord.key`; a pair with different keys, or a key beside prose, never reaches the reviewer (`pairing.keys_settle`). |

## `ADRs/supabase/tables.md`

| # | S | Decision |
|---|---|---|
| 1 | A | Postgres holds the graph; Neo4j is not used. RLS at storage, edges commit with documents, one ops surface. Revisit when per-tenant graphs outgrow RAM. |
| 2 | S→24 | PPR in application memory over a CSR snapshot. Reasoning kept: iterative PPR inside Postgres is 100–1000× slower. |
| 3 | S→24 | CSR blobs built at ingestion, immutable per `(tenant, graph_version)`; upload blob **before** pointer row. |
| 4 | S→24 | `graph_version` bumped in the same transaction as the edge write. |
| 5 | S→24 | `graphs` is a snapshot pointer table, nodes reference `tenant_id` not `graph_id`. |
| 6 | A | Two-exchange Slack shape: ACK fast, answer later via `chat.postMessage`. Dedupe is load-bearing. |
| 7 | A | Cloud Tasks carries post-ACK work; `/work` must be idempotent. |
| 8 | A | Verify HMAC over raw bytes and dedupe with `ON CONFLICT DO NOTHING` before any work. |
| 9 | A | Scale-to-zero; `min-instances=1` is a reversible knob; heavy imports lazy. |
| 10 | S | Typing-based warmup rejected: `user_typing` is RTM-only. |
| 11 | A | `users.id` is independent; `auth_user_id` nullable link. Slack-only users exist. |
| 12 | A | `memberships` bridge table; role is a fact about the link. |
| 13 | A | Email is the merge key with `email_source`; auto-merge only on trustworthy sources. |
| 14 | A | Just-in-time provisioning of Slack users on first event; ignore `bot_id`; `ON CONFLICT DO NOTHING`. |
| 15 | A | Slack identity is workspace-scoped: PK `(slack_team_id, slack_user_id)`; installs in `connector_installations`. |
| 16 | A | Bot tokens in a secret manager; `token_ref` only; signing secret in env. |
| 17 | A | Uninstall is soft (`uninstalled_at`); reinstall reuses the row (open: history). |
| 18 | A | Tenant isolation is a RESTRICTIVE wall; business rules are PERMISSIVE doors; `FORCE RLS`. |
| 19 | A/D | RBAC `owner/admin/member` now; groups, spaces, per-resource grants deferred; hedge: nullable `space_id`. |
| 20 | O | Permission-aware retrieval: derived data, tenant-wide walk, prompt as last boundary. Enforce at enrichment query; accept scoring leak knowingly. |
| 21 | A | `text` + CHECK, never `varchar(n)`; `timestamptz` always. |
| 22 | A | Migrations are the only write path to schema; forward-only; small and single-purpose. |
| 23 | A | Monorepo; asymmetric CI; local + production only until a design partner is live. |
| 24 | A | Graph derived from `chains` self-join; no edges table; PMI weights; walk in a managed service. |
| 25 | A | Content-addressed identity at every level; `chains` PK is a surrogate with `content_key` unique. |
| 26 | A | Episodes carry no position; recover order with `doc.find(text)` at query time. |
| 27 | A | Chains are the linking unit, surfaces the lookup; `chain_surfaces` is the mention table with `n_mentions`. |
| 28 | A | Entities are thin and disposable; emptied ones deleted; no `merged_into` until something holds ids across syncs. |
| 29 | A | Seeding is vector + lexical; `surfaces.embedding` 1536 + trigram; IDF and `1/n_entities` measured out; divide by `sqrt(n_surfaces)`. |
| 31 | A | A BM25 text channel over episode texts is fused with the walk's passage ranking (reciprocal rank, `TEXT_CANDIDATES` = 10). Graph-only had the answering passage for 14/28 agent questions, fused 24/28, nothing lost, +0.05 s. |
| 30 | A | Surfaces are judged for name-hood at ingest (LLM cascade mini→4.1, cached); `surfaces.is_name`; non-names never seed, are dropped from chain identity, and a chain with no name is withheld. Rules and a coref-prompt paragraph measured and rejected. |

What did not work (schema side): `ordinal`, `merged_into`, `n_entities`/`n_episodes`, head-offset chain ids, stored edges, `slack_teams`, context vector as identity (a centering bug). Open items: `facts` table, `documents`/`node_sources`, bi-temporal columns, blob location, subscription history, canonical name lock, entity id stability, extraction filter (WHAT-COUNTS taxonomy applied nowhere), transaction pooler.

## `ADRs/supabase/security.md`

| # | S | Decision |
|---|---|---|
| 1 | A | The database derives the tenant from a declared workspace; it cannot verify the request. Cloud Tasks OIDC is the other half and is not built. |
| 2 | A | One tenant per transaction, scalar equality, `SET LOCAL`; unset context denies. Batch jobs loop, never widen the wall. |
| 3 | A | Roles split by how the tenant is established: `app_connector` derives, `app_ingest` claims; both inherit `app_user`. |
| 4 | A | Adding a source is a row in `connector_installations`, not a role. |
| 5 | S→4 | `slack_teams` removed. |
| 6 | A | `NOINHERIT` login roles without passwords are the authentication floor; forgetting `SET ROLE` fails loudly. |
| 7 | A | Resolver branches on `current_setting('role')`, not on which settings exist. |
| 8 | A | Uninstall revokes immediately. |
| 9 | A | The token never enters the database; `current_token_ref()` for `app_connector` only. |
| 10 | A | Deny by revoking privilege, not by omitting a policy — RLS filters, it does not raise. Invariant query included. |
| 11 | A | `SECURITY DEFINER` functions: `search_path=''`, qualified names, `REVOKE FROM PUBLIC`; owner must have BYPASSRLS. |
| 12 | A | Resource ceilings on the login roles; per-role settings are not inherited. |

What did not work: gating the server on `auth.uid()`; `COALESCE` in the resolver; `current_user` inside a definer; ceilings on `NOLOGIN` roles; relying on RLS to deny writes; `GRANT ON ALL TABLES` + default privileges; `app_user NOLOGIN` with no members. Open: Cloud Tasks OIDC, replay window in prod code, idempotency on `event_id` (schema done), `SET LOCAL` in prod code, parameterised `set_config`, transaction pooler, walls for `authenticated`, audit trail, `app_ingest` can claim any tenant, `pg_stat_statements` residual (recorded in migration `20260812141500`).

## `ADRs/slack/connector.md`

| # | S | Decision |
|---|---|---|
| 1 | A | OAuth install auto-provisions tenant, subscription (`trialing` — note the CHECK allows `trial`), installation in one transaction. |
| 2 | A | Reinstall reuses the tenant via `ON CONFLICT DO UPDATE`. |
| 3 | A | Token never lands in Postgres (inherits security 9). |
| 4 | A | "Never DMs" is a scope decision: no `im:history`/`mpim:history`. |
| 5 | D | Private channels out of scope for v1 (tenant-level isolation would leak them). |
| 6 | A | A thread is a file: `thread/{channel}/{thread_ts}`. |
| 7 | A | Loose messages bundle into `day/{channel}/{YYYY-MM-DD}`; promotion rule open. |
| 8 | A | One message per paragraph so appends re-extract one episode (measured). |
| 9 | A | Mentions and markup resolved before hashing. |
| 10 | A | Attachments are their own files keyed on Slack `file_id`; `url_private_download` needs the bearer token. |
| 11 | A | Progressive indexing: recent first, backfill in background, notify. Rate limits *unverified*. |
| 12 | A | A `connector_channels` table is required (shape open). |
| 13 | A | Proactive answering opt-in per channel, ships silent; launch on `@mention` + slash command. |
| 14 | A | Cheap heuristics gate the classifier. |
| 15 | A | Ambiguity is answered with a question in thread; state location undecided. |
| 17 | A | A thread Nazar has answered in is a conversation: replies there are answered without a mention, the thread is read back as context, the follow-up is condensed to a standalone question (or answered from the thread alone). `message.channels` subscribed; everything else dropped. |
| 16 | A | The answer step decides a stance first — answer, reframe (with cited grounds), clarify — and may say the question is the wrong one. Measured 7/7 controls untouched, 3/5 bad premises caught, stable across runs. |

What did not work: "wait 20 minutes"; answer anything question-shaped; one file per message; LLM-classify every message; fold attachments into threads. Open: A retention/deletion, B per-channel ACL, C `connector_channels` shape, D rate limits, E canvases/posts, F vision on images, G edits/deletes, H bot self-citation.

## `ADRs/brainstorming.md`

Not decisions — a triage list. Four blockers before more Slack code: verify rate limits; Enterprise Grid tenancy; day-file promotion rule; where clarification state lives. Then retention + ACL, then cost modelling, then product surface.
