# Runbook

Copy-paste commands. Every command runs from the project directory named in the
comment. Secrets come from `.env` files that are gitignored; keys are listed in
`server/public/.env.example` and `.env.example`.

## Database

```bash
supabase start                      # repo root; CLI 2.109.1; DB 127.0.0.1:54322, Studio :54323, API :54321
# on the 8 GB laptop start the database only (everything the pipeline and the query path need):
supabase start -x gotrue,realtime,storage-api,imgproxy,kong,mailpit,inbucket,postgrest,pgadmin-schema-diff,postgres-meta,studio,edge-runtime,logflare,vector,supavisor
supabase db reset                   # drop + replay all migrations + seed.sql (if present)
supabase migration new <name>       # creates supabase/migrations/<ts>_<name>.sql
supabase db diff                    # capture drift if you edited in Studio (you should not have)
psql postgresql://postgres:postgres@127.0.0.1:54322/postgres      # as owner; walls apply only after SET ROLE
```

Passwords for the login roles are never in a migration (locally `localdev-only`, which the isolation tests also set):

```sql
ALTER ROLE app_ingest_login    PASSWORD '…';
ALTER ROLE app_connector_login PASSWORD '…';
ALTER ROLE app_ack_login       PASSWORD '…';
```

Ad-hoc queries inside the wall (pattern from `supabase/snippets/`):

```sql
BEGIN;
SET LOCAL ROLE app_ingest;
SELECT set_config('app.tenant_id', '<uuid>', true);
SELECT * FROM public.entity_names ORDER BY chains DESC LIMIT 20;
COMMIT;
```

## Ingest (live copy)

```bash
cd server/public
uv sync
uv run python -m src.inject.main --store ./store.json --notes-dir ./notes --tenant-id <uuid> --dry-run     # no model calls
uv run python -m src.inject.main --store ./store.json --notes-dir ./notes --tenant-id <uuid> \
    --dsn "postgresql://app_ingest_login:PASSWORD@127.0.0.1:54322/postgres"                              # full run + upload
uv run python -m src.inject.main --store ./store.json                                                    # re-link only
# flags: --limit N  --model  --workers  --ann-k  --merge-at  --split-at  --no-context  --no-graph-resolve  --entities-out  --source
#        --no-vector-cache  --full-relink  --full-upload   (caches: <store>.vectors.npz, store["linking"], store["uploaded"])
#        --no-judge  --judge-models gpt-4.1-mini,gpt-4.1     (the name judge cascade; verdicts cached in store["names"])
```

From Python: `from src.inject.main import ingest; ingest(store_path=…, notes_dir=…, tenant_id=…, dsn=…)`.

First run downloads `nomic-embed-text-v1.5` (~550 MB). The spaCy model is a pinned wheel in `pyproject.toml`, no download step.

## Ingest (old copy, with connector and query)

```bash
cd old_server
uv sync
set -a && . ../.env && set +a                                  # OPENAI_API_KEY etc.
uv run python -m entity_linking.pipeline --store /tmp/s.json --notes-dir /tmp/notes --tenant-id <uuid> --dsn postgresql://…
export SLACK_BOT_TOKEN=xoxb-…
uv run python -m entity_linking.slack --tenant <uuid> --store /tmp/slack.json --days 30 [--channels general eng]
uv run python -m entity_linking.pipeline --store /tmp/slack.json --tenant-id <uuid> --dsn postgresql://…
uv run python -m entity_linking.ask embed --tenant <uuid>                          # superseded: use src.query below
uv run python -m entity_linking.ask ask   --tenant <uuid> "who owns the Okta rollout?" --explain [--cutoff 0.2] [--alpha 0.5] [--no-reframe]
```

`ask.py` defaults to `TEST_DATABASE_URL` or the local `postgres` DSN and does `SET LOCAL ROLE app_ingest` itself.

## Query path (live copy)

```bash
cd server/public
uv run python -m src.query.retrieve embed --tenant <uuid>                                   # embed name surfaces (app_ingest)
uv run python -m src.query.retrieve ask --tenant <uuid> "who owns the Okta rollout?" --explain
uv run python -m src.query.retrieve ask --workspace T0123 "…"                                # as the Slack worker: app_connector, DB resolves the tenant
# flags: --k --seeds --passages --alpha --cutoff --min-co --no-ppr --no-reframe --model
```

## Slack ingress

```bash
cd server/public
cp .env.example .env && $EDITOR .env    # CONNECTOR_DATABASE_URL for the worker; a connector_installations row for your team id
uv run uvicorn src.app:app --port 8000 --reload
ngrok http 8000 && curl -s http://127.0.0.1:4040/api/tunnels | python3 -c 'import json,sys;print([t["public_url"] for t in json.load(sys.stdin)["tunnels"]])'
# paste the URL into slack/manifest.dev.yaml (request_url, redirect_urls) → api.slack.com → reinstall
curl -s localhost:8000/health
```

Replaying a captured request through ngrok fails the 5-minute window; set `VERIFY_TIMESTAMP=false` locally and never anywhere else.

## Tests and lint

```bash
cd server/public && uv run pytest                     # 7
cd old_server     && uv run python -m pytest                    # 333; `uv run pytest` uses a stale launcher (see testing.md)
cd old_server     && uv run pytest tests/test_db_tenant_isolation.py -v
uv run ruff format . && uv run ruff check . --fix     # in each project
```

## Evals (old copy)

`old_server/entity_linking/demos/`: `eval_retrieval.py` (single-passage), `eval_footprint.py` (coverage), `eval_expansion.py` (edge weights, synthetic), `eval_seeding*.py`, `eval_context_vector.py`, `eval_cooccurrence.py`, `eval_ecbplus.py`; `demo_end_to_end.py` is the fullest runnable example; `graph_view.py` renders the graph. Each dumps intermediate JSON.

Offline mini-experiments from 2026-09-17 (see `wiki/experiments.md`): `eval_sparse_ppr.py`, `eval_store_format.py`, `eval_score_cutoff.py`, `eval_junk_rig.py` need nothing; `eval_linking_mini.py` needs nomic and `OPENAI_API_KEY`; `eval_vault_markup.py --root <vault>` needs spaCy; `eval_junk_filter.py`, `eval_junk_judge_agreement.py`, `eval_junk_prompt.py`, `eval_junk_gold.py` (all `--root <vault>`) call OpenAI and cache verdicts under `old_server/artifacts/`; `eval_reframe.py [--repeat 3]` scores the answer stance against the vault store there.

## Legacy (Neo4j)

```bash
cd old_server
uv run python -m pipeline.run --notes-dir ./notes [--group-id g] [--from relations --force] [--skip-upload]
uv run dotenv run -- python query.py "what tools does the team use?" --group-id g
uv run python reembed_entities.py --dry-run
cd mcp-server && uv run python -m app.main            # needs MCP_AUTH_TOKEN, NEO4J_*
```
