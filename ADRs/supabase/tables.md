# Decision Log

Decisions taken while designing the Slack query path, the graph storage
architecture, and the identity/tenancy schema.

Each entry: what was decided, why, and what it costs. Format is deliberately
short — one screen per decision. Supersede entries rather than editing them
once they have shipped.

Status key: **Accepted** (decided and being built) · **Deferred** (decided not
to build yet) · **Open** (identified, not yet decided)

---

## Storage & retrieval

### 1. Postgres holds the graph; Neo4j is not used — *Accepted*

**Context.** The graph could live in a native graph store (Neo4j Aura) or in
the Postgres instance already used for tenancy and documents.

**Decision.** Postgres, with denormalized `tenant_id` throughout.

**Amended (24).** The tables are `entities`, `chains`, `chain_surfaces` and
`surfaces`, and there is no `edges` table at all — adjacency is derived from a
self-join on `chains`. The reasoning below is unaffected; only the nouns are.

**Why.** Three properties Neo4j can't offer here: RLS enforces tenant
isolation at the storage layer instead of in application code; edges commit in
the same transaction as the documents they were derived from, eliminating a
sync pipeline; and one system means one set of ops, backups, and credentials.

**Cost.** Adjacency stops being a physical pointer and becomes a B-tree probe.
The GDS algorithm library (community detection, betweenness) is gone — anything
beyond PPR is code we write. Cypher ergonomics are lost until SQL/PGQ reaches
managed Postgres (PG19 GA late 2026, managed availability later).

**Revisit when.** Per-tenant graphs stop fitting in application memory, or
cross-tenant analytics becomes a product requirement.

---



### 2. PPR runs in application memory over a CSR snapshot — *Superseded by 24*

> The walk moved to a managed service and the edge set became derived rather
> than stored, which removes the snapshot, its version counter and its pointer
> table together. Entries 2–5 are kept for the reasoning, not the conclusion.

**Context.** Iterative PPR inside Postgres (driver loop or recursive CTE) is
100–1000× slower than in-memory — each round joins the whole edge table with no
frontier sparsity.

**Decision.** Approximate seeded PPR (push-based) runs in the application over
a CSR adjacency snapshot: three flat arrays (`indptr`, `indices`, `weights`)
plus the UUID↔int mapping.

**Why.** CSR makes "neighbours of node i" a contiguous slice — cache-friendly,
prefetchable. It rebuilds in RAM the index-free adjacency lost at the storage
layer. It is also the format `scipy.sparse` consumes directly.

**Cost.** CSR is frozen — insertion means rebuilding, so the snapshot is
immutable per `graph_version`. RLS guards the extraction query, not each hop of
the walk.

**Note.** The reranker and LLM call dominate end-to-end latency. PPR itself is
not on the critical path and should not be optimized further without a
measurement saying otherwise.

---



### 3. CSR snapshots are built at ingestion and stored as immutable blobs — *Superseded by 24*

**Context.** The serving path runs on scale-to-zero instances. A cache built at
query time dies with the instance, and the build is CPU-bound work that cannot
happen after a request returns.

**Decision.** The ingestion pipeline builds the CSR and uploads it to object
storage under an immutable key `(tenant_id, graph_version)`. The serving path
downloads it (~150 ms) or uses a resident copy.

**Why.** The build needs a long-lived context; ingestion already has one.
Serving becomes stateless-friendly: fetch a blob, use it, die.

**Ordering invariant.** Upload the blob **before** updating the pointer row.
The reverse order can leave a row promising a blob that does not exist — every
query for that tenant fails. Upload-first can only leave an orphan blob, which
is harmless and cleanable.

**Cost.** Write amplification on graph updates; a retention rule is required
(keep the last two versions per tenant).

---



### 4. `graph_version` is the cache-invalidation signal, bumped in-transaction — *Superseded by 24*

**Decision.** Ingestion bumps a per-tenant version counter in the *same
transaction* as the edge write, ideally via an `AFTER` trigger. Consumers
compare their snapshot's version against the current one before trusting it.

**Why.** The safety invariant is *changed graph ⇒ changed version*. As two
separate operations, a crash between them produces new data under an old
version — silent staleness, the worst failure mode because nothing errors.
Atomicity makes that state unrepresentable. The trigger moves enforcement from
N ingestion call sites to one.

**Consequence.** RAM residency is treated as luck; the blob plus the version
comparison is the guarantee. Any query path must be able to load the blob
itself.

---



### 5. `graphs` is a snapshot pointer table, not a container — *Superseded by 24*

**Decision.** `graphs` holds `(tenant_id, graph_version, blob_path, built_at)`.
Nodes and edges reference `tenant_id`, **not** `graph_id`.

**Why.** If nodes belonged to a version row, every rebuild would duplicate the
entire node set — storage grows multiplicatively and "current graph" becomes a
join instead of a filter.

**Naming.** Rename to `graph_snapshots` so the table cannot be misread as
holding graph data. The earlier misreading is what produced the erroneous
`graph_id` columns.

**Open.** Whether to keep one row per tenant (overwrite) or one row per
snapshot (append-only). Append-only is preferred: in-flight tasks can still
resolve an older version, and retention becomes an explicit delete.

---



## Slack request path



### 6. Two-exchange Slack shape: ACK fast, answer later — *Accepted*

**Context.** Slack expects a 200 within 3 seconds; the real work (blob load,
PPR, enrichment, LLM) takes longer.

**Decision.** The inbound webhook is acknowledged immediately. The answer is
delivered later as a **new outbound request** to Slack's Web API
(`chat.postMessage`, bot token).

**Consequence.** Slack retries on missed deadlines, so dedupe is load-bearing
rather than hygiene.

---



### 7. Cloud Tasks carries post-ACK work — *Accepted*

**Context.** The platform allocates CPU while a request is in flight and
throttles the instance otherwise. Work started before returning does not
progress — it is suspended indefinitely with no error.

**Decision.** `/event` verifies, dedupes, enqueues, and ACKs. Cloud Tasks
delivers the work as a fresh HTTP POST to `/work`, which runs with full CPU for
its whole duration.

**Why beyond CPU.** The obligation becomes durable (survives instance death and
deploys), retries with backoff come free (return non-2xx), and the queue is an
explicit backpressure buffer with a configurable dispatch rate.

**Consequence.** Delivery is at-least-once, so `/work` must be idempotent —
check "already answered?" before posting, or a redelivered task double-posts.

---



### 8. Signature verification and dedupe happen before any work — *Accepted*

**Decision.** `/event` verifies Slack's HMAC over `v0:{timestamp}:{raw_body}`
with the app signing secret, rejects timestamps older than ~5 minutes, then
inserts the event ID with `ON CONFLICT DO NOTHING`. The insert's result is the
branch: duplicates ACK without enqueuing.

**Why.** Verification is the security boundary and is local — no network call.
Warmup and work consume real resources (extraction scan, RAM, pool slot), so
they are privileged actions. The unique constraint is the dedupe mechanism:
engine-enforced, not remembered.

**Detail.** Verification runs on raw bytes before JSON parsing; reserializing
changes the signature.

---



### 9. Scale-to-zero, with `min-instances=1` as a reversible knob — *Accepted*

**Decision.** Run scale-to-zero. `min-instances=1` may be enabled later for
reliable ACKs and a persistent RAM cache.

**Why it is only a knob.** Instances are replaced on deploys and recycled by
the platform, so residency is a probability improvement layered on an
architecture that must already be correct without it. The blob path and the
task queue are the correctness layer.

**Mitigation for cold start.** Heavy libraries (numpy, scipy, PPR modules) are
imported lazily inside the functions that need them, keeping the ACK path's
import graph small.

---



### 10. Typing-based warmup was rejected — *Superseded / not viable*

**Investigated.** Pre-warming the graph while a user types.

**Finding.** Slack's `user_typing` event is RTM-only and is not delivered to
HTTP webhook apps. RTM requires a persistent WebSocket connection, which
contradicts scale-to-zero.

**Also rejected.** Listening to all channel messages to predict intent —
requires history scopes (a security-review liability for a product whose
positioning is data isolation), costs scale with workspace chattiness, and
warming is structurally weak under autoscaling because the warmed instance may
not be the serving one.

**Instead.** Remove the need: precomputed blobs make cold instances fast enough
that prediction stops mattering. If an assistant-thread surface is adopted
later, `assistant_thread_started` is a legitimate pre-question signal.

---



## Identity & tenancy



### 11. `users` has its own primary key; `auth_user_id` is a nullable link — *Accepted*

**Context.** Keying `users.id` to `auth.users(id)` is the standard Supabase
pattern, but a Slack user who has never signed into the web app has no
`auth.users` row and therefore could not exist.

**Decision.** `users.id` is independently generated. `auth_user_id` is a
nullable unique reference to `auth.users`.

**Why.** Internal identity stays independent of any one credential provider.
Each new ingress type (Slack today, Teams or API keys later) is a link table,
not a migration of primary keys. Slack-only users are representable.

**Consequence.** The FE routes through the application server rather than
querying Supabase directly, so the server injects `users.id` as the session
claim — both ingress paths converge on one identity and RLS applies identically
to both.

---



### 12. Memberships are a bridge table, not a column on users — *Accepted*

**Decision.** `memberships(user_id, tenant_id, role)` with
`UNIQUE (user_id, tenant_id)`.

**Why.** A person can belong to several tenants (a consultant in a client
workspace; the founder in their own plus a design partner's). A `tenant_id`
column on `users` forecloses that, and retrofitting it means migrating every
user row and rewriting every policy — the expensive-to-reverse category.

**Also.** `role` is a fact about the *link*, not about the person — someone can
be an owner in one tenant and a member in another.

**Index.** The unique constraint doubles as the index for
`WHERE user_id = ...`, the hottest lookup in the database.

---



### 13. Email is the account-merge key, with a recorded source — *Accepted*

**Decision.** `email` (nullable, lowercase-enforced, partial unique index where
not null) plus `email_source` ∈ {`slack`, `oidc`}, constrained to be present or
absent together.

**Why.** Without it, the same human arriving through Slack and through web
sign-in becomes two irreconcilable rows.

**Security constraint.** Matching on email is an account-takeover path when the
email is unverified. Auto-merge only on sources that are trustworthy (workspace
admin-controlled Slack profiles; OIDC with a verified claim). Anything
ambiguous requires explicit confirmation.

**Known duplication.** Email is held in both `auth.users` and `public.users`,
justified because Slack-only users have no `auth.users` row. On conflict,
`auth.users` wins for anyone with `auth_user_id` set.

---



### 14. Just-in-time provisioning of Slack users — *Accepted*

**Decision.** On the first event from an unknown `slack_user_id` in a known
workspace, create the `users` row, the `slack_users` link, and the membership,
then proceed. No "connect your account" gate.

**Why.** Slack events carry the sender's user ID and team ID. Requiring an
OAuth flow before the first question is significant friction for a Slack-native
product.

**What this implicitly decides.** Workspace membership grants tenant access.
Coherent while everyone in a tenant sees everything — but it stops being
sufficient the moment document-level permissions exist, at which point
auto-provisioned users need a default that is not "everything."

**Guards.** Ignore events carrying `bot_id` and the app's own bot user ID
(otherwise the bot's replies retrigger the handler). Treat a missing `user`
field as "not a question." Use `ON CONFLICT DO NOTHING` — two simultaneous
first messages race.

---



### 15. Slack identity is workspace-scoped — *Accepted*

**Decision.** `slack_users` has composite primary key
`(slack_team_id, slack_user_id)`.

**Amended.** `slack_teams` is gone; installs live in `connector_installations`,
keyed `(source, external_id)` so Notion and Drive need no new table. `slack_users`
carries a `source` column fixed to `'slack'` by CHECK, purely so it can reference
that composite key. See `security.md` entries 4–5.

**Why.** Slack user IDs are scoped to a workspace. The same human in two
workspaces has two Slack IDs and gets two rows pointing at one `user_id`.

---



### 16. Bot tokens live in a secret manager; only a reference is stored — *Accepted*

**Decision.** `connector_installations.token_ref` holds a secret-manager path.
The token itself is never in Postgres, and the workers cannot read the column
directly — only through a scoped `SECURITY DEFINER` function returning the row
for the workspace they have already proven. See `security.md` entry 9.

**Why.** A bot token grants full API access to a customer's workspace. In a
column it is exposed to every backup, snapshot, database dump, log line, and
any future RLS mistake. In the secret manager, exposure narrows to IAM, access
is logged, and versioning gives a clean rotation path.

**Cost.** A fetch (~20–50 ms) on the work path — cached in process with a short
TTL so rotation still propagates.

**Not per-install.** The Slack *signing secret* belongs to the app, not to each
installation, and lives in server environment config. It must be usable before
any database lookup, since signature verification precedes trusting the
payload's `team_id`.

---



### 17. Uninstall is soft — *Accepted*

**Decision.** `connector_installations.uninstalled_at` marks removal rather than
deleting the row, preserving history. Both tenant resolvers test
`uninstalled_at IS NULL`, so access stops the moment the app is removed.

**Open.** Reinstalls collide with the `(source, external_id)` primary key.
Simplest resolution: reuse the row (clear `uninstalled_at`, update `token_ref`).
Alternative if full install history is wanted: surrogate PK plus
`UNIQUE (id) WHERE uninstalled_at IS NULL`.

---



## Access control



### 18. Tenant isolation is a restrictive wall; business rules are permissive doors — *Accepted*

**Decision.** Every tenant-scoped table gets `ENABLE` **and** `FORCE ROW LEVEL SECURITY`, a restrictive policy testing tenant membership, and permissive
policies for business rules.

**Why** `FORCE`**.** Table owners bypass RLS by default. Migrations and any
owner-privileged path would sail through, and isolation tests would pass while
enforcing nothing.

**Why the split.** Permissive policies OR together — adding one can only widen
access. A restrictive policy ANDs, so it holds regardless of what doors are
added later. Writing an invariant in permissive syntax is the classic RLS
disaster: it becomes another door.

**Also.** Grants gate objects, RLS gates rows, in that order. RLS refines
access and never creates it — a policy on a table with no grant is inert.

**Built out in `security.md`.** How the tenant is established, which role the
server connects as, and what the database can and cannot verify. That log also
records the several versions of this that did not work.

---



### 19. RBAC now; groups, spaces, and per-resource grants deferred — *Accepted / Deferred*

**Decision.** `memberships.role` ∈ {`owner`, `admin`, `member`}, enforced by a
check constraint and resolved through a single `stable` helper function rather
than scattered comparisons.

**What this buys.** Administrative permissions — who may invite users, connect
Slack, delete documents, manage billing.

**What it explicitly does not buy.** Content visibility. RBAC answers "what
actions may this user perform," not "which documents may this user see." Within
a tenant, every member sees everything.

**Deferred.** `groups`, `group_members`, `grants`, `spaces` — the per-resource
ACL layer. Justified while the product premise is a fully shared team knowledge
base.

**Cheap hedge.** Add a nullable `space_id` to `documents` now as the future
attachment point. Adding a nullable column later is easy; retrofitting the
*concept* of a container after documents exist is not.

---



### 20. Permission-aware retrieval is an open problem — *Open*

**Identified, not resolved.** Once content permissions exist, three enforcement
gaps appear that no RLS policy closes:

- **Derived data.** Entities, edges, facts, and embeddings derived from a
restricted document must inherit its restriction. Each derived artifact needs
a pointer back to its sources, and multi-source facts need a stated rule —
most-restrictive-wins is the defensible default.
- **The CSR snapshot is tenant-wide.** PPR walks it with no notion of who is
asking. Post-filtering results leaks through scoring (restricted nodes still
propagate mass); pre-filtering means the cache key becomes
`(tenant, permission set)`, changing the caching economics; filtering during
the walk puts an authorization lookup in the hot loop.
- **The prompt is the last boundary.** Anything placed in the context window
can appear in the answer, so the enrichment query is the real enforcement
point.

**Current position.** Enforce hard at the enrichment query; accept the scoring
leak knowingly and document it.

---



## Entity & graph layer

Added after building the linker. Everything here is measured; the evals live in
`old_server/entity_linking/demos/` and the numbers in `DATAFLOW.md`.

### 24. The graph is derived from `chains`; there is no edges table — *Accepted*

**Context.** Entries 2–5 assumed edges are stored and the walk runs in-process
over a CSR snapshot rebuilt at ingestion.

**Decision.** Two entities are joined when a chain of each sits in the same
episode — a self-join on `chains`. The walk runs in a managed PPR service, fed a
payload regenerated from that join.

**Why.** Merges and splits stop being maintenance. Re-clustering updates one
column, `chains.entity_id`, and every edge follows: nothing to migrate, nothing
to invalidate, no second source of truth to drift. A stored edge table would be
keyed on entity ids that churn every time clustering changes.

**Weights are PMI, not counts.** Measured on a synthetic corpus where
relatedness is true by construction: raw counts reach P@family 0.786 and *never*
bridge two entities connected only through a third; `PMI × 1/(episode size)`
reaches 0.983 with 87.5% bridging, on 58% fewer edges. Binary weights score
**0.000** — with every edge equal the walk goes to the highest-degree node, which
is reliably the least informative entity in the corpus.

**Cost.** The payload is recomputed per push rather than incrementally
maintained. `graph_version`, the CSR blob and the `graphs` pointer table all
become moot; there is no in-process snapshot left to invalidate.

---



### 25. Identity is content-addressed at every level — *Accepted*

**Decision.**

```
file_id    = uuid5(tenant_id | source | external_id)   -- path
episode_id = uuid5(file_id | sha1(text))               -- content
chain_id   = uuid5(episode_id | content_key)           -- content
```

**Why.** A document is a container that gets revised, so it keeps its identity
across edits. An episode is a unit of content, so it keeps its identity when an
edit *above* it moves it — keying on offsets would renumber every episode below
any insertion, and re-extract paragraphs whose text never changed.

**Cost, and the one place it is refused.** A content-derived key churns the
moment content changes, and anything referencing it dangles. `chains` therefore
uses a **surrogate** primary key with `content_key` as a separate unique: adding
one surface to a chain updates the fingerprint in place and keeps the row id,
its `chain_surfaces` children, and the `entity_id` that cost LLM review calls to
determine. The in-memory store derives the id instead, because a dict key *is*
its deduplication and there is no constraint to do that job.

---



### 26. Episodes carry no position — *Accepted*

**Decision.** No `ordinal`, and no ordered `episode_ids` array on `files`.

**Why.** Nothing in the pipeline reads order, so storing it meant maintaining a
version-dependent field on every sync for a consumer that does not exist.

**Cost, stated plainly.** Reading order is *not* recoverable from the database.
Recovering it means re-splitting the document and matching on text — episodes are
verbatim slices, so `doc.find(text)` is exact. One document read at query time,
which is the right place to pay it. Do not reinstate an ordering column without
a consumer that needs one.

---



### 27. Chains are the linking unit; surfaces are the lookup — *Accepted*

**Decision.** `chains` (one per episode's worth of coreferent mentions) is what
stage 4 clusters and what carries `entity_id`. `chain_surfaces` is the 1:N side
and simultaneously the evidence source and the surface→entity bridge. `surfaces`
holds one embedding per distinct string.

**Why.** Every query goes through `chains` in one direction or the other:
seeding walks surface → chain → entity, passages walk entity → chain → episode.
One hinge, no denormalised bridge to keep in sync when `entity_id` moves.

**Mentions are not stored.** Without offsets, a mention row's natural key
collapses to `(chain_id, surface)` — so `chain_surfaces` *is* the mention table,
with `n_mentions` for the frequency that would otherwise be lost. That count is
the one thing here that cannot be re-derived later without re-extracting.

**Cost.** Offsets are gone, so highlighting or recomputing a context vector needs
the episode text and a re-find.

---



### 28. Entities are thin and disposable — *Accepted*

**Decision.** `entities` holds an id, a canonical name and timestamps. No
surfaces array, no episode list, no counts, no `merged_into` tombstone. Emptied
entities are deleted.

**Why.** Everything else is derivable from `chains ⋈ chain_surfaces`, and a
derived value stored twice is a value that can disagree with itself. The
tombstone was dropped once it was clear nothing outside the graph holds an
entity id across syncs: the PPR payload is regenerated on every update, so a
merge is just chains changing `entity_id`.

**Cost.** A deleted id is unresolvable — there is no way to discover where it
went. Reinstate `merged_into` the moment anything caches an entity id, saves a
link, or otherwise outlives one sync.

---



### 29. Seeding is vector + lexical, and IDF was measured out — *Accepted*

**Decision.** `surfaces.embedding` is `vector(1536)` (text-embedding-3-large,
truncated) with an HNSW index, plus a `pg_trgm` GIN index on the surface itself.
No `n_entities`, no `n_episodes`.

**Why both halves.** For short strings, exact match beats cosine outright, and
`S. Chen` should hit `S. Chen` at rank 1 without depending on an embedding. 1536
also keeps the vector inside pgvector's indexable dimension limit, which full
3072 exceeds.

**Why the counts were dropped.** Measured on ECB+ with this pipeline's own
extraction: IDF over document frequency cost 0.017 MRR and `1/n_entities` cost
0.097. Both point the wrong way for structural reasons — junk surfaces are
*rare*, not frequent (75.7% appear in one episode), so IDF up-weights them; and
junk entities are singletons, so `1/n_entities` hands them the maximum. What
does work is dividing an entity's score by `sqrt(its surface count)`: MRR 0.689
→ 0.749, and it needs no stored column at all.

---



## Conventions



### 21. `text` with check constraints, never `varchar(n)` — *Accepted*

Postgres stores `text` and `varchar(n)` identically; the only difference is a
length check. A check constraint expresses the same rule, can combine length
with format, and is cheap to change — where altering a column type is not. Most
length limits are arbitrary anyway. Constrain what the domain actually
requires.

`timestamptz` everywhere, never `timestamp`.

### 22. Migrations are the only write path to schema — *Accepted*

The dashboard is for looking; the repo is for changing. `supabase db diff`
captures drift when the rule is broken while exploring. Forward-only — restore
from backup is the rollback, since untested down-migrations are worse than
none. Migration files may be edited freely until applied somewhere that cannot
be reset; after that, only new forward migrations.

Each migration is small and single-purpose so that a policy review is a policy
diff.

### 23. Monorepo; local + production environments for now — *Accepted*

One repo, because schema changes touch migrations, server, FE types, and tests
in one atomic commit. Research folders (`notebooks/`, `experiments/`) coexist
with production folders under **asymmetric CI** — production gated hard,
research gated barely.

Two environments while there are no customers: production *is* staging, because
mistakes there cost only time. Add staging the day a design partner is live —
that is when prod becomes precious and migrations start meeting lived-in data.

---



## What did not work

Schema-side counterparts to the list in `security.md`. Each was built, measured,
and removed.

| Attempt | What happened |
| --- | --- |
| **`ordinal` on episodes** | A version-dependent integer maintained on every sync, refreshed whenever an insertion moved anything. Nothing ever read it. Removing it cost the ability to recover reading order from the database — accepted knowingly (entry 26). |
| **`merged_into` tombstones** | Built to make a merged-away entity id resolvable. Then established that nothing outside the graph holds an id across syncs, which makes the pointer a column maintained for a reader that does not exist. Removed with a note on exactly when to bring it back. |
| **`n_entities` / `n_episodes` on surfaces** | Added as IDF and ambiguity weights before measuring. Both measured negative (entry 29). The lesson is the ordering: they were columns first and hypotheses second. |
| **Head-offset chain ids** | `uuid5(episode_id \| head.start \| head.end)` keyed identity to *which span the model called canonical* — an opinion that can change between runs while the chain does not. Replaced by `content_key`, which is a property of the chain. |
| **Storing edges** | A table keyed on entity ids, which churn on every re-cluster. Deriving the same edges from `chains` made merges and splits automatic instead of a migration (entry 24). |
| **Keeping `slack_teams` beside `connector_installations`** | Two tables that must agree about which workspace belongs to whom. The specific one held nothing the general one lacked. |
| **The context vector as an identity signal** | Dropped at AUC 0.505 — chance. The reading turned out to be an artifact of episode-centering; uncentered the same signal measures 0.745. It now enters only through a feature gated on same-string pairs, where the name features carry no information at all. A negative result that was really a bug in the measurement. |

---



## Open items


| Item                                | Note                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `facts` table and source bridges    | Enrichment returns facts with attribution; reified facts are what make answers explicable and permission-checkable. Not yet in the schema. This is also the labelled-edge layer entry 24 defers: co-occurrence tells the walk which entities are near each other, never why.                                        |
| `documents`, `node_sources`         | Document↔entity is many-to-many; `chains` is the bridge today (entity → chain → episode → file). Whether that is enough for permission propagation is untested.                                                                                                                                                     |
| `events` table shape                | Dedupe minimum is `(slack_event_id PK, received_at)`. Worth adding `payload jsonb` (replay), `status`/`responded_at` (idempotency at `/work`), `error`. `tenant_id` must be nullable — it is resolved in `/work`, not at ACK time. Needs a retention rule. Consider an `internal` schema so it is never API-exposed. |
| ~~Composite-FK tenant pinning~~     | **Shipped.** `files`, `episodes`, `chains` and `chain_surfaces` all reference `(id, tenant_id)`, so a cross-tenant reference is unrepresentable rather than merely denied.                                                                                                                                          |
| Bi-temporal columns and constraints | `valid_at`/`invalid_at`/`created_at`/`expired_at` on edges and facts, with a check that validity ordering holds and an exclusion constraint against overlapping intervals per fact.                                                                                                                                  |
| Blob storage location               | Object storage co-located with compute is faster for a latency-path download; single-vendor is operationally simpler. Measure a 20 MB cross-cloud fetch before committing.                                                                                                                                           |
| Subscription history                | Currently modelled one-to-one with tenants. One-to-many with a status column preserves trial→paid→churn history.                                                                                                                                                                                                     |
| Canonical name for an entity        | `entities.canonical_name` is `NOT NULL`, and nothing computes it — `pick_head` runs *within* a chain, never across a cluster. The first write path has to choose one, and re-clustering will silently overwrite a human correction unless a lock column is added.                                                    |
| Entity id stability across re-runs  | Ids are minted once and carried forward by chain-membership overlap — but that matching step does not exist yet, and the interim naming (smallest member chain id) renames an entity roughly 1 in *n+1* times when a chain joins it. Fine for full re-runs, wrong for incremental sync.                             |
| Extraction filter                   | Roughly a quarter of top-1 retrieval seeds land on junk entities (`the freeze`, `the change`) under every weighting scheme tried. No retrieval-side weight fixes it; the WHAT-COUNTS taxonomy in `prompts.py` is written and applied nowhere.                                                                        |
| Transaction-pooler behaviour        | `SET LOCAL` is correct for it, but only direct sessions have been exercised. Session-vs-transaction pooling is exactly where a tenant leak appears in production and not in dev.                                                                                                                                    |
