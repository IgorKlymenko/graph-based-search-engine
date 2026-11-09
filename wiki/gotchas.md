# Gotchas

Things that already bit once. Each names the code it guards. Read the row
before editing that code.

## Database

- **`SET`, not `SET LOCAL`** leaks the previous request's tenant on a pooled connection. Works in dev (session pooling), fails in prod (transaction pooling). `CLAUDE.md`.
- **Connecting as `postgres`** bypasses every wall until `SET ROLE`. The app must use a login role. `.env.example`.
- **Forgetting `SET LOCAL ROLE`** on a login role: `permission denied` on every table — by design (`NOINHERIT`). Do not "fix" with `INHERIT`.
- **`current_user` inside `SECURITY DEFINER`** is the owner. The resolver uses `current_setting('role')`. Renaming `app_ingest` silently changes behaviour.
- **A definer function owned by a role without `BYPASSRLS`** reads zero rows from `connector_installations` and denies everything, looking like a config error.
- **RLS filters, it does not raise.** `DELETE FROM tenants` "succeeds" with 0 rows. Deny with `REVOKE`.
- **`type "vector" does not exist`** has two causes and neither is a missing extension: a missing `USAGE ON SCHEMA extensions` (migration `20260812141500`), or a `search_path` that does not list `extensions`. The login roles default to `"$user", public`, so every walled transaction that touches embeddings must `SET LOCAL search_path = public, extensions`; `query.retrieve.enter` does. The `extra_search_path` in `config.toml` applies to the API roles only.
- **Partial writes are silent wrong answers.** `chains` without `chain_surfaces` makes `alias_hit` say no. One transaction, FK order: files → episodes → surfaces → entities → chains → chain_surfaces. `5_store.py`.
- **`episodes.created_at` after `expired_at`** trips the CHECK when a document is edited twice between uploads; hence `LEAST(now(), removed_at)`. `5_store.py`.
- **A chain's head moving between uploads** tripped `UNIQUE (tenant_id, chain_id) WHERE is_head`: the upsert reaches the new head row before the old one, and the partial index is checked per row. `store.upload` clears `is_head` for every chain in the delta first. Regression: `server/tests/public/inject/test_upload_head_moves.py` (needs `TEST_DATABASE_URL`).
- **Deleting an entity that still has chains** raises (`ON DELETE RESTRICT`); the orphan sweep runs last for that reason.
- **Reading `entities.display_name` directly** returns NULL almost always. Use `entity_display_name()`.
- **`e.expired_at IS NULL` in passage queries** is load-bearing; retired episodes keep their chains.
- **Breadth counted by re-joining `chain_surfaces`** amplifies instead of damping (MRR 0.718 → 0.658). Separate aggregates, `DISTINCT`. DATAFLOW SQL.
- **`20260806190506` is an intentionally empty migration** (it duplicated `20260806190227`). Do not put anything in it; add a new file instead.

## Ingest

- **`import src.inject.4_link`** is a syntax error. Use the aliases via `_stages.load`. [conventions.md](conventions.md).
- **Two uuid5 namespaces** would split every id. `_UUID_NS` in `1_episodes.py` equals the one in `graph_upload.py`; never add another.
- **`rel_path`** normalises absolute vs relative roots so `file_id` does not depend on how the CLI was invoked.
- **Resolve markup before hashing.** Slack mentions, links, Obsidian `[[x]]`, `==x==`: cleaning after storage breaks the verbatim-context invariant silently. `strip_markup` lives in `1_episodes.py` (both copies as of 2026-09-18); the GitHub connector runs it on `.md` files.
- **Front matter in `.md` corpora** is fed to the extractor as text; put metadata in a `manifest.json` sidecar instead.
- **Run-on sentences** (a whole table in one line) hit `LLM_TIMEOUT` (90 s). One item per line is ~16× faster; `strip_markup` keeps list markers for this reason.
- **`normalize_for_parse` must be length-preserving**; the assertion is the guard. Lose one character and every span points somewhere wrong.
- **The per-request timeout does not bound a call.** It resets on every byte, so a server that holds the socket parks the worker indefinitely (measured: nine threads in `SSLSocket.read` for an hour). `chains.completed` abandons a stage after `STALL_TIMEOUT` of silence and closes the shared HTTP client; the work stays pending, an abandoned review drops `store["linking"]` so the next run scores everything. Keep new LLM loops on that pool.
- **Model-returned ids are checked.** An unissued id is a protocol violation → retry once → fail the episode, which stays pending (`extracted_at` unset). Failures are printed, not raised; check the warning line.
- **`n*` spans must match exactly once**; duplicates and near-misses are dropped and counted, never guessed.
- **A pronoun-only chain is not stored**, and pronouns never reach `surfaces` (`is_identifying`), or `("Sarah Chen", "she")` becomes an alias and two "it" chains score 1.0.
- **`chain_id` keyed on head offsets** churned when the model changed its mind about the head; it is keyed on `content_key`. Do not go back.
- **`PairScorer` is unfitted**; prior weights saturate at 0.50 for an identical name, so `merge_at > 0.50` produces zero merges.
- **`choose_thresholds()` output** fragments entities (0.634 vs 0.775 CoNLL). Tune the cut on the clustering metric.
- **`cluster(cannot_link=…)`** is off on purpose (0.774 vs 0.945). The decision source contradicts itself.
- **Centering the context vector** made it read as chance. It is uncentered, and gated to same-string pairs.
- **Unseeded HNSW** returned different neighbours on identical input; `random_seed=100`, and exhaustive below `ANN_MIN_FACTOR · k` chains.
- **Review answers are cached by prompt hash** (`store["reviews"]`). Any change to what the reviewer sees is a new question; to force re-review delete the key.
- **Three more caches, three more flags.** `<store>.vectors.npz` (`--no-vector-cache`), `store["linking"]` (`--full-relink`) and `store["uploaded"]` (`--full-upload`). After a database reset the upload notices an empty tenant and goes full on its own; after changing scorer weights or thresholds you must `--full-relink`, because carried merges were decided under the old ones.
- **The judge's verdicts are cached by surface in `store["names"]`.** Changing `JUDGE_PROMPT` or the model does not re-judge; delete the key. A surface with no verdict counts as a name. Deleting the key also drops the verdicts connectors wrote for their seeds (`name_seeds`); re-sync the connector, or the judge rejects most code identifiers again (ADR connectors 12).
- **Pronouns must never reach `chain_surfaces`.** The filter lives in `store._rows_for_chain_surfaces`; it was missing once and "she" became a seed.
- **Entity ids rename** when a cluster's smallest chain changes (~1 in n+1 joins). `store.match_entities` carries prior ids forward by chain overlap; a split renames the smaller half. Nothing outside the graph may hold an entity id across syncs yet (ADR tables 28).
- **Two contested-only chains merging with each other** looked "settled by evidence"; `unresolved` checks that *every* member is contested.
- **One ingest per store path at a time.** The store is a file.
- **The paid stages checkpoint themselves** (every 200 reviews, every 10 judge batches, every 25 episodes). A killed run resumes from the last save; nothing answered is re-bought.
- **Save the store before shaping.** `pending` is computed before the graph resolver runs, and placing a chain can empty a candidate cluster. `main.unresolved_report` skips such clusters and `ingest()` saves the store first; 10,274 review answers were lost once when neither was true. Keep both.
- **Stage 3 is the memory peak, and the dev laptop has 8 GB.** Loading nomic under torch takes gigabytes; macOS killed a full-vault run there once, and killed the next one together with `supabase start` when both ran at once. Never run the pipeline and Docker's VM together on this machine; start only the database container (`supabase start -x …`, see [runbook.md](runbook.md)), and run encode/link passes with Docker stopped. The store checkpoints after every stage and during the paid ones, so a kill costs only the step in flight.
- **`einops`** is imported by nomic's remote code and fails at first encode, not at import; it is pinned in `pyproject.toml` for that reason.
- **The spaCy model is a pinned wheel** in `server/public`; `RUNNING.md`'s `spacy download` line applies to `old_server` only.

## Connectors

- **`removed()` after `documents()`.** A connector that lists a tree only knows deletions once the listing is done; `base.sync` fixes the order. Report only deletions positively observed — a partial listing read as absence retires live documents. The GitHub client refuses a truncated tree for this reason.
- **A seed span that does not match the episode text is dropped**, with a warning line. The usual cause is a renderer that changed output without `render.py` and `RENDER_VERSIONS["github"]` moving together.
- **Seeds that change while text does not** clear `extracted_at`; the episode is re-extracted on the next run. Expect one call per episode after improving a parser.
- **Two model chains for one identifier** would make it contested. `merge_seeded` absorbs on offsets *or* text, whatever type the parser gave the span; do not narrow it to proper names.
- **`src/` is a package in this repo and a layout directory in most.** `PythonParser.module_name` decides from the tree (`__init__.py` walk-up), never from the directory name. With no tree every directory is a package.
- **Unresolved calls are dropped on purpose.** Keeping `cur.execute` as written makes `execute` a hub surface in every database file.
- **`LocalRepo.head` is a hash of the working tree, not the commit.** Keyed on `rev-parse HEAD` it made uncommitted edits invisible to a re-sync (a function written the night before was not in the graph). Citations come from the `origin` remote when it is GitHub; a changed citation base re-emits every file (`cursor["html_url"]`), which updates `files.url` without re-extracting.
- **The cursor's `include` is part of "unchanged".** Narrowing `--include` on the same head must re-list the tree, or nothing outside the new scope is retired. Retirement is judged against every path the last sync rendered, whatever render version did it; a version bump alone must not orphan files.
- **The reviewer merges a class with its methods.** Shown two symbol paragraphs it says "same" for `PairScorer` and `PairScorer.fit`; 903 such merges made one 853-chain entity. `keys_settle` keeps keyed pairs out of the review band; do not route them back in with a prompt tweak.
- **A colliding short name is plain text on its definition.** If a code entity seems to lack the bare name as a surface, check `cursor["ambiguous"]`; that is the rule from ADR connectors 11, not a parser miss. A name made ambiguous only by a reference through a re-export the parser cannot follow (`from src.inject import pairing` for `4_link.py`) is a bug in `render._dangling`, which exists for exactly that case.
- **`GITHUB_TOKEN` unset** means unauthenticated, public-only, ~60 requests an hour *(unverified)*; a sync of a private repo then fails with 404, not 401.

## Connectors

- **`removed()` after `documents()`.** A connector that lists a tree only knows deletions once the listing is done; `base.sync` fixes the order. Report only deletions positively observed — a partial listing read as absence retires live documents. The GitHub client refuses a truncated tree for this reason.
- **A seed span that does not match the episode text is dropped**, with a warning line. The usual cause is a renderer that changed output without `render.py` and `RENDER_VERSIONS["github"]` moving together.
- **Seeds that change while text does not** clear `extracted_at`; the episode is re-extracted on the next run. Expect one call per episode after improving a parser.
- **Two model chains for one identifier** would make it contested. `merge_seeded` absorbs on offsets *or* text, whatever type the parser gave the span; do not narrow it to proper names.
- **`src/` is a package in this repo and a layout directory in most.** `PythonParser.module_name` decides from the tree (`__init__.py` walk-up), never from the directory name. With no tree every directory is a package.
- **Unresolved calls are dropped on purpose.** Keeping `cur.execute` as written makes `execute` a hub surface in every database file.
- **`GITHUB_TOKEN` unset** means unauthenticated, public-only, ~60 requests an hour *(unverified)*; a sync of a private repo then fails with 404, not 401.

## Slack

- **Verify on raw bytes.** Re-serialising JSON changes the signature.
- **`url_verification` must return the challenge as the body**, before the ACK path.
- **The bot answering itself** loops; filter `bot_id` and `bot_message`.
- **ngrok Replay** resends the original timestamp and fails the 5-minute window; `VERIFY_TIMESTAMP=false` is for that and nothing else.
- **`url_private_download` without the bearer token** returns an HTML login page with status 200, which naive code stores as document text.
- **Token prefixes**: `xoxb-` is the bot token; `xapp-`/`xoxp-` are not. `slack.main` checks.
- **Reinstalling the app mints a new token** (`invalid_auth` mid-backfill).
- **Slack is append-only and immortal**; a thread file is never finished. Edits shift downstream episodes; accepted.
- **Scopes are the DM guarantee.** The manifest matches ADR slack 4; adding `im:history` or `message.im` silently re-enables DM reading.

## Retrieval

- **`ALPHA` and `CUTOFF_RATIO` move together** (0.50 / 0.2 in `ask.py`; 0.2 / 0.3 on the synthetic rig). A cutoff tuned for one alpha drops bridged pairs at another. Removing junk hubs is what makes the walk stop caring (experiments J2).
- **PMI on once-seen pairs collapses to its maximum**; `MIN_CO_OCCURRENCE = 2` on the vault. The synthetic rig scored `count ≥ 2` no better than raw. Corpus-dependent again.
- **`max` passage scoring ties every episode holding the top entity**; `sum` rewards long notes; use mean.
- **Nearest surfaces first, then join to chains**; joining first spends the LIMIT on duplicates and defeats the HNSW index.
- **psycopg returns `uuid.UUID`**, the graph code uses `str`; mixed types in one node set fail to sort. `str()` at the boundary, `::uuid[]` going back.
- **No embeddings → "no surfaces matched"**; run `ask.py embed` after every ingest until it is automated.
