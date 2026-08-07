# Decision Log — Security & tenant isolation

Decisions taken while building the access-control layer: which role the server
connects as, how a tenant is established, and what the database enforces versus
what it has to trust.

Companion to `tables.md`, which covers schema. Entry 18 there states the shape
(restrictive walls, permissive doors); this log records how it is actually
wired and what had to be thrown away first.

Each entry: what was decided, why, and what it costs. Supersede rather than edit
once shipped.

Status key: **Accepted** (decided and being built) · **Deferred** (decided not
to build yet) · **Open** (identified, not yet decided) · **Superseded**

Every claim marked *measured* was verified against a local Postgres; the
assertions live in `old_server/tests/test_db_tenant_isolation.py` (29 tests).

---

## The trust boundary

### 1. The database derives the tenant; it cannot verify the request — *Accepted*

**Context.** Slack calls the server, the server verifies the signature, ACKs,
and enqueues. A worker later opens a database connection. By then the signature
proof is gone — the worker holds a task payload naming a workspace.

**Decision.** The server declares a *workspace*; the database resolves which
tenant that is, from `connector_installations`. The server never names a tenant
on the connector path.

**Why.** It moves the mapping out of application code into one place that
cannot be got wrong by a stale cache or a mixed-up variable. A worker that
computes the wrong tenant from the right workspace is now caught.

**Cost.** It does *not* verify the workspace itself. If anything can post to the
worker endpoint, it names any workspace and reads that tenant. The database will
faithfully resolve it. Cloud Tasks OIDC on that endpoint is therefore not an
optimisation — it is the other half of this decision, and it is not built yet.

---

### 2. One tenant per transaction, not a set — *Accepted*

**Context.** The wall could compare against a set of tenants (`IN`) or a single
one (`=`).

**Decision.** Scalar equality against `public.current_tenant_id()`, declared per
transaction with `SET LOCAL`.

**Why.** Two properties fall out. An unset context makes the function `NULL`,
`tenant_id = NULL` is `NULL` rather than true, and **every wall denies** —
forgetting to declare a tenant locks the server out instead of opening the
corpus up. And `SET LOCAL` is transaction-scoped, so a pooled connection cannot
carry one request's tenant into the next.

**Cost.** No batch job can span tenants in one transaction. That is a real
constraint on future backfills; the answer is a loop, not a wider wall.

---

## Roles

### 3. Roles split by how the tenant is established, not by connector — *Accepted*

**Context.** Two ways a tenant becomes known: derived from a verified
installation, or claimed outright by the server.

**Decision.** `app_connector` derives and *cannot* claim. `app_ingest` claims,
for uploads and backfills that have no workspace. Both inherit one privilege
bundle, `app_user`, so grants live in one place.

**Why.** Structural rather than advisory. With one role holding both paths, a
connector worker that set `app.tenant_id` instead of `app.workspace_id` would
silently fall back to the weaker model — no error, no signal. Separated, the
resolver never reads `app.tenant_id` for that role, so the mistake denies.

**Cost.** Two connection pools instead of one. `app_ingest` can still claim any
tenant — inherent, since an upload has nothing external to verify, so that
path's isolation is application logic rather than database enforcement.

---

### 4. Adding a source is a row, not a role — *Accepted*

**Context.** Notion and Drive will follow Slack.

**Decision.** `connector_installations (source, external_id, tenant_id,
token_ref, uninstalled_at)`. One resolver lookup, keyed on both columns.

**Why.** Notion and Drive share Slack's trust model exactly: an OAuth install
yields a verifiable external id. Same role, same policy, same resolver — only a
row. A `CASE` per connector would grow a branch on the hot path of every policy
on every query.

**Cost.** Source-specific detail has nowhere to live in this table. It only
holds what resolution needs.

---

### 5. `slack_teams` removed — *Superseded by 4*

It held nothing `connector_installations` does not: team id, tenant, token_ref,
installed/uninstalled. `slack_users` now carries a `source` column fixed to
`'slack'` by CHECK so it can reference the composite key. Keeping both would
have meant two tables that must agree about which workspace belongs to whom.

---

### 6. `NOINHERIT` login roles are the authentication floor — *Accepted*

**Context.** Nothing could assume the bundle at first: `app_user` was `NOLOGIN`
with no members, so `SET ROLE` failed outright. The interim fix granted it to
`postgres`, which carries `BYPASSRLS`.

**Decision.** `app_connector_login` and `app_ingest_login`, `LOGIN NOINHERIT`,
granted the matching bundle, created without a password.

**Why.** Three things at once. Resource ceilings apply (entry 12). No
`BYPASSRLS`, so the worst case stops being "sees every tenant". And `NOINHERIT`
means the role can do nothing but connect until it names one — so a code path
that forgets `SET ROLE` **fails loudly** instead of running unwalled. *Measured:*
`permission denied for table files` without the switch, correct access with it.

**Cost.** The password must be set out of band — never in a migration, which
lands in git and the statement log. Until then the roles are inert, and
`postgres` remains the only way in for migrations and tests.

---

## Resolution

### 7. The resolver branches on the caller's role, not on which settings exist — *Accepted*

**Decision.** `current_setting('role')` selects the branch: `app_ingest` claims,
everything else derives.

**Why.** A worker cannot reach the weaker path by setting a different variable,
only by *being* a different role — a deployment decision rather than a line of
code. Deriving is the default, so a role added later lands on the strict branch.

**Cost.** Couples the resolver to a role name. Renaming `app_ingest` silently
changes behaviour rather than erroring.

---

### 8. Uninstall revokes immediately — *Accepted*

`uninstalled_at IS NULL` in both resolvers. Removing the app stops access at the
same moment, not merely stopping new events. *Measured.*

---

### 9. The token never enters the database — *Accepted*

`connector_installations.token_ref` is a pointer into the secret store. The
worker fetches it through `public.current_token_ref()`, a `SECURITY DEFINER`
function returning the row for the workspace it has already proven, granted only
to `app_connector`.

**Why.** A `SELECT` grant on the table would let any worker enumerate every
workspace and its tenant — the customer list. A database that never holds the
token cannot leak it.

---

## Enforcement

### 10. Privilege revocation, not policy absence — *Accepted*

**Context.** Tenancy and billing should be read-only to the server; identity
tables and `connector_installations` unreachable entirely.

**Decision.** `REVOKE` the privilege. Do not rely on omitting a permissive
policy.

**Why.** **RLS filters; it does not raise.** With the grant left in place,
`DELETE FROM tenants` succeeds and silently removes zero rows — and the day
someone adds a `FOR ALL` policy it starts removing real ones. *Measured:* the
test expecting an error initially failed for exactly this reason.

**Invariant.** Every table is either walled or ungranted; none is both grantable
and unwalled. Worth re-checking whenever a table is added:

```sql
SELECT c.relname, has_table_privilege('app_user', c.oid, 'SELECT') AS grantable,
       EXISTS (SELECT 1 FROM pg_policies p WHERE p.tablename = c.relname
                AND p.permissive = 'RESTRICTIVE') AS walled
FROM pg_class c
WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'r';
```

---

### 11. `SECURITY DEFINER` functions are locked down — *Accepted*

`search_path = ''` with every name qualified — a definer function resolving
names through a caller-controlled path is a privilege hole. `REVOKE EXECUTE
FROM PUBLIC`, which is the default otherwise.

**Fragile dependency.** The definer read of `connector_installations` works only
because the owner (`postgres`) has `BYPASSRLS`: the table is `FORCE ROW LEVEL
SECURITY`, which strips the owner's usual exemption. A definer function owned by
a role *without* `BYPASSRLS` reads zero rows, returns NULL, and silently denies
everything. Check ownership first if that ever happens — the failure is total and
reads as a configuration problem.

---

### 12. Resource ceilings belong on the role that logs in — *Accepted*

`statement_timeout = 30s`, `idle_in_transaction_session_timeout = 10s`.

`ALTER ROLE ... SET` is applied **at login**, and per-role settings are not
inherited through membership. Set on the login roles they hold, and `SET ROLE`
afterwards does not reset them. *Measured:* `30s` on the login role, `0` via
`SET ROLE` into a `NOLOGIN` bundle. Also set on the bundles as belt and braces,
in case one is ever given `LOGIN` directly.

---

## What did not work

Recorded because each looked correct until it was run, and several would have
failed silently in production.

| Attempt | What happened |
| --- | --- |
| **Gating the server on `auth.uid()`** | The first wall derived tenants from `memberships` via the JWT. A server on its own Postgres role has no JWT, so `auth.uid()` is NULL and the function returned zero rows. *Measured:* files visible 0, tenants visible 0, every INSERT rejected. The wall was built for the access model we were explicitly **not** using. The membership helpers survive, unused, for when `authenticated` gets direct access. |
| **`COALESCE` in the resolver** | Falling back from workspace to a bare tenant claim meant an *unknown or uninstalled* workspace fell through to whatever the server also happened to claim — precisely the case that must deny. Replaced with `CASE`: once a workspace is declared it is the only authority. |
| **`current_user` to detect the caller** | Inside `SECURITY DEFINER`, `current_user` is the **owner**, not the caller. The ingest branch never fired and every upload silently resolved to NULL. `session_user` is no better — it is the login role, not the one `SET ROLE` selected. `current_setting('role')` is what reflects the choice. |
| **Ceilings on `NOLOGIN` roles** | Configured, inert. `statement_timeout` read back as `0` inside a `SET ROLE` session for a year of production before anyone would have noticed. Fixed by entry 6. |
| **Relying on RLS to deny writes** | See entry 10. RLS filters rather than raising, so a `DELETE` on a read-only table succeeded and removed nothing. |
| **`GRANT ... ON ALL TABLES` plus `ALTER DEFAULT PRIVILEGES`** | Auto-granted `connector_installations` — the workspace→tenant map — to `app_user` the moment it was created. RLS returned zero rows rather than raising, so the hole read as "working". Every over-grant is now revoked explicitly, with the reasoning attached. |
| **`app_user NOLOGIN` with no members** | Nothing could assume it: `permission denied to set role`. A privilege bundle with no path to it is not a security boundary, it is a typo. |

---

## Open items

| Item | Note |
| --- | --- |
| Cloud Tasks OIDC on the worker endpoint | The largest gap. Everything in entry 1 assumes the workspace id is genuine; nothing today stops an arbitrary POST from naming one. |
| Slack replay window | The signed timestamp must be rejected beyond ~5 minutes, or a captured request replays indefinitely. |
| Idempotency on `event_id` | ACK-then-enqueue makes Slack retries the normal case; without dedupe one event processes several times. See the `events` table row in `tables.md`. |
| `SET LOCAL`, not `SET`, in production code | Asserted in tests; the application must match. Session-level settings survive a pooled connection and leak the previous request's tenant. |
| Parameterised `set_config` | The workspace id is an external string. Never interpolate it. |
| Transaction pooler untested | `SET LOCAL` is correct for it, but only direct sessions have been exercised — and session-vs-transaction pooling is exactly where a leak appears in production and not in dev. |
| Walls for `authenticated` | It has no direct access yet, so nothing is exposed. RESTRICTIVE policies bind only the roles they name, so a permissive policy added there without a wall would have no isolation at all. |
| Audit trail | No record of which workspace wrote what. Not a vulnerability; an inability to answer questions after one. |
| `app_ingest` can claim any tenant | Inherent to entry 3. Worth revisiting if the upload path ever becomes customer-facing rather than operator-driven. |
