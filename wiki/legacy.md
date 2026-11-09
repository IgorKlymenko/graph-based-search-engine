# Legacy code

Everything here predates the Postgres design (ADR tables 1, 24) and targets
Neo4j. It is kept because parts are reusable and because `experiments/arena`
depends on it. It is **not** where new work goes.

| Path | What it is | Reusable? |
|---|---|---|
| `old_server/pipeline/` | Six-stage extraction: candidates (with a "gleaning" second pass) → mxbai span embeddings → HDBSCAN blocking → LLM resolution → entity vectors → relation extraction. Artifacts as JSON/npz in `artifacts/<group_id>/`. `run.py` orchestrates with `--from/--force`. | The prompts in `candidates.py` are the ancestor of `utils/prompts.py` `SYSTEM_PROMPT`; the WHAT-COUNTS taxonomy there is still unapplied in the live path. The `Fact`/`RelationEdge` models are the starting point for the open `facts` table. |
| `old_server/graph_upload.py` | Writes Saga → Episode → Entity → RELATES_TO/HAS_FACT into Neo4j with `group_id` partitioning; entity name embeddings via OpenAI `text-embedding-3-large` (switched from e5 on 2026-07-08 for Railway cold starts). `stable_uuid` uses the same namespace as `1_episodes.py`. | The uuid namespace must stay identical; nothing else. |
| `old_server/query.py`, `reembed_entities.py` | Neo4j RAG query (BM25 + vector seeds → PPR → source files → LLM) and a re-embed utility. | Pattern only. |
| `old_server/retrieval/` | Typed library: `ppr.py`, `ppr_approx.py` (push-based, sparse), `temporal.py` (bi-temporal filters), `multihop.py`, `multihop_beam.py`, `dag_multihop.py`, `treehop.py`, `rerank.py` (embedding and cross-encoder), `seed.py`, `config.py` (identifier validation), `exceptions.py`, `interfaces.py` (`GraphDriver`, `Embedder`, `LLMClient` protocols). Specified in `old_server/specs/01–05` and audited in `specs/security-audit-report.md`. | `ppr_approx.py` and `rerank.py` are the natural replacements for `ask.ppr` and the missing rerank stage; the protocols make them testable with `conftest.py` stubs. Drivers and Cypher are not reusable. |
| `old_server/services/` | `neo4j.py` driver, `openai.py` client and embedder, `e5.py` (asymmetric prefixes; commented out in favour of OpenAI). | `openai.py` async client pattern. |
| `old_server/mcp-server/` | MCP server on Railway exposing `search_notes`, `list_group_ids`, `set_group_id` over Neo4j; bearer or capability-URL auth; Dockerfile with `uv sync --frozen`. | The Dockerfile layering and the auth middleware. The kg-search MCP endpoint in this session's config is this server, currently unreachable. |
| `old_server/specs/` | Specs for the retrieval library, the arena, the extraction pipeline port, a typed fact schema (v0, hypothesis), and a Claude Code task prompt. | Read `fact_predicate_schema_v0.md` before designing `facts`. |
| `experiments/arena/` | Click + Rich CLI comparing retrieval methods with token/cost/timing instrumentation (`pricing.json`, `tracker.py`, `duel.py`, `report.py`). 5 test files. | Instrument the Postgres query path the same way once it has more than one method. **Its `pyproject.toml` points `retrieval` at `../server`, which no longer contains that package (`old_server/pyproject.toml` is named `retrieval`).** |

Rules of engagement:

- Do not spend tokens reading `retrieval/` or `pipeline/` unless the task names them. Their tests (about 160 of the 325 in `old_server`) still run and should keep passing, but a failure there is not a regression in the product.
- Do not delete them in a change that is about something else; removal is its own decision (the repo already did one prune, commit `8da111e`, and kept an `archive/pre-cleanup-2026-07-18` branch).
- Anything that mentions `group_id`, `Saga`, `RELATES_TO`, Cypher, or `NEO4J_*` env vars is legacy.
