# Testing

## Where and how many (collected 2026-09-17)

| Project | Command | Collected | Covers |
|---|---|---|---|
| `server/public` | `cd server/public && uv run pytest` | 165 + 1 skipped (2026-09-19) | `auth/` (7): signature verification, health. `connectors/` (71): Python parser, code render, sync and seeds, seeded chains, GitHub connector with a fake client (incl. single-file include), GitHub client over `httpx.MockTransport`, local checkout (tracked + untracked files), keyed review, head move on upload (skipped without `TEST_DATABASE_URL`). `inject/` (36): unresolved report, strip_markup, vector cache, incremental link, upload plan and name-flag delta, name judge, stall watchdog. `query/` (37): stance and Slack rendering, conversation turns and condensing, thread routing end to end with retrieval stubbed, sparse walk, cutoff, wall entry, lexical and BM25 text channel, passage formatting. `scripts/` (13): the agent-token harness's pure parts (grep simulation, sufficiency, tool result, passage→file mapping through a connector-rendered document, question file validity) and the agent-loop harness's tool kit (glob/grep/read over a corpus, kits, billing, bootstrap). `testpaths = ["../tests/public"]`, `asyncio_mode = auto`. The auth conftest sets `SLACK_SIGNING_SECRET=mysecret`. |
| `old_server` | `cd old_server && uv run python -m pytest` | 338 | everything else, including bench files (`bench_*.py` are collected as tests). **Not `uv run pytest`**: that launcher's shebang points at the pre-rename `server/.venv`, so it runs on anaconda's site-packages and 8 span-parser tests fail there. `uv sync` rebuilds it. |

`old_server/tests` by area:

| Files | Count | Needs |
|---|---|---|
| `test_entity_linking_{episodes,spans,markup,content_key,context_gate,identifying,matching,review_cache,stable_ids,unresolved}.py` | 110 | spaCy model (offline) |
| `test_slack_render.py` | 18 | nothing |
| `test_ask_ppr.py` | 5 | nothing (the walk imports without psycopg) |
| `test_answer.py` | 5 | nothing (stubbed completion) |
| `test_db_tenant_isolation.py` | 38 | Postgres with migrations applied; skips otherwise. The `login_url` fixture **sets** `app_connector_login`'s password to `localdev-only` with `ALTER ROLE` and skips if the login roles are absent — never point `TEST_DATABASE_URL` at a database you care about. |
| `test_pipeline_*.py` | 16 | nothing (legacy extraction) |
| `test_{ppr_algorithm,fast_ppr,multihop,beam_multihop,dag_multihop,treehop,temporal_search,filters,shared,config,security,security_extended,bdtr}.py`, `bench/` | rest | stubs in `conftest.py` (legacy retrieval library) |

The stage tests import `entity_linking.*`, i.e. the **old copy**. `server/public/src/inject` has no stage tests of its own. Until they move, run `old_server` tests after touching either copy and port the change.

## Database tests

```
supabase start                                   # 127.0.0.1:54322, migrations applied by `supabase db reset`
cd old_server && uv run pytest tests/test_db_tenant_isolation.py -v
# or point elsewhere:
TEST_DATABASE_URL=postgresql://postgres:postgres@host:5432/postgres uv run pytest tests/test_db_tenant_isolation.py
```

Seeds two tenants (`ACME`, `GLOBEX`), a live Slack install, an uninstalled one, and a Notion install, as `postgres` (outside the wall); every assertion runs inside a `SET LOCAL ROLE` transaction. The four login-role tests connect as `app_connector_login` with the password the fixture just set; the `internal.events` tests use `ack()` and `claim()` helpers that are the reference shape for the production ACK and worker statements.

## Lint

`uv run ruff check .` — state at time of writing:

| Project | Findings |
|---|---|
| `server/public` | none (2026-09-18). |
| `old_server/entity_linking` | 4: `E501` in `demos/demo_chains.py`, `demos/demo_pairing.py` (prompt-like long lines). |

`uv run ruff format .` then `uv run ruff check . --fix` clears the fixable ones. Type checking (`uv run pyright`) is declared in `old_server` dev deps and is not part of any current workflow.

## Writing a new test

- Put stage tests beside the existing `test_entity_linking_*.py` until the suite moves; name the file after the invariant, not the function.
- Database invariant → a new `test_…` in `test_db_tenant_isolation.py` using `run(connection, sql, role=…, tenant=… | connector=…, workspace=…)`. Assert from inside the wall.
- Anything that would call OpenAI: inject a stub. `chains._complete_json` is the single choke point; `main._ask_cached` accepts a `call` thunk.
- Anything that would load nomic: pass vectors in (`ChainRecord(name_vector=…)`) rather than encoding.
- Anything that would call GitHub: fake the `connector.Client` protocol (four methods) or hand `client.GitHub` an `httpx.MockTransport` and a recording `sleep`.
- Async tests use `anyio`, not `asyncio` (`CLAUDE.md`).
