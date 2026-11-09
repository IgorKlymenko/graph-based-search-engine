# Connectors

How a source system becomes documents, and how a structured source (code)
hands its already-resolved references to the graph. Design decisions:
`ADRs/connectors/github.md`; the Slack connector predates this package and is
on [slack.md](slack.md).

## Contract (`server/public/src/connectors/base.py`)

| Thing | What |
|---|---|
| `Document(external_id, text, title, url, mentions)` | One rendered source object. `text` is what `sync_file` hashes; `title`/`url` go to `files.title`/`files.url` for citations. |
| `Mention(start, end, text, key)` | One occurrence of a name in `text`; mentions with one `key` corefer. `text[start:end] == text` always. |
| `Connector` | `source`, `documents()` (new or changed), `removed()` (gone; consumed **after** `documents`). |
| `sync(store, connector, tenant_id=…)` | The only door into the store: `sync_file` per document, `retire_file` per removed id, `attach_seeds` when a document has mentions. Returns `SyncCounts`. |
| `attach_seeds` | Re-splits with `split_episodes`, cuts mentions per episode, stores them as `episodes[*].seeds = [{key, spans: [[start, end, text], …]}]`. Changed seeds on unchanged text clear `extracted_at`. |
| `name_seeds` | Writes `store["names"][text] = True` for every seed mention that is not a bare lower-case word (bare words only when unjudged), so the prose judge never sees a connector's names (ADR connectors 12: it rejected 7 in 10). |

Stage 2 reads seeds in `chains.seeded_chains` (proper-name spans, origin `connector`) and folds them into the model's chains in `chains.merge_seeded`; see [ingest-pipeline.md](ingest-pipeline.md) § Stage 2. Stage 3 stamps the seed's key on the `ChainRecord`; stage 4 never asks the reviewer about a pair whose keys differ or where only one side has a key (`pairing.keys_settle`, ADR connectors 13).

## Adding a source

The checklist in [architecture.md](architecture.md) § Adding a source, plus: implement `Connector`, call `base.sync`. Push-based sources yield deltas from events and report deletions from delete events; enumerable sources diff a listing and may infer deletion from absence only when the listing was whole.

## Code (`server/public/src/connectors/code/`)

| File | What |
|---|---|
| `model.py` | `Symbol(kind, name, qualname, signature, doc, references, children)`, `Parsed(module, symbols)`, the `Parser` protocol, `PARSERS` by extension, `parser_for(path)`. |
| `python.py` | `PythonParser` on stdlib `ast`. `module_name(path, tree)` walks up while `__init__.py` exists (ADR connectors 9). References: imports, same-module defs, `self.m` for methods the class defines; builtins, stdlib and calls on locals dropped (ADR 7). `MAX_SIGNATURE` 300. |
| `render.py` | `render(parsed, path=…, style=Style(...)) -> (text, mentions)`. One paragraph per symbol: `kind name (qualname)`, signature, `defines:` / `calls:` / `imports:` / `uses:`, docstring. No line numbers (ADR 8). Module paragraph also carries `file x.py at path` (`package at …` for `__init__.py`, never a mention). `Style.ambiguous` (from `ambiguous_from` / `ambiguous_names`): those short names render qualified as references and as plain text on their own definitions (ADR 11, measured). `names_of(parsed)` is the per-path map the connector keeps; `index_document(maps, ambiguous)` is the measured, off-by-default evidence document. |

Every identifier the text shows is a mention keyed by qualified name; references render as the short name (`calls: sync_file`) with the qualified key, modules by dotted path. Prose is left to the model.

A new language: a class with `extensions` and `parse(path, text, tree=…)`, added to `_registry()` in `model.py`. Nothing else changes.

Measuring a rendering change: `uv run python scripts/eval_code_collisions.py <repo roots…>` from `server/public` (ground truth = qualified keys; see [experiments.md](experiments.md)).

## GitHub (`server/public/src/connectors/github/`)

```
export GITHUB_TOKEN="$(gh auth token)"        # after `gh auth login`; or a fine-grained PAT with Contents: read
cd server/public                              # no token: public repos only, unauthenticated rate
uv run python -m src.connectors.github --tenant <uuid> --store ./store.json \
    --repos owner/name [owner/other] [--ref main] [--include server/public ADRs]
uv run python -m src.connectors.github --local /path/to/checkout --tenant <uuid> --store ./store.json \
    --repos owner/name --include server/public ADRs CLAUDE.md   # working tree, no token; the legacy copy is out (2026-09-19)
uv run python -m src.inject.main --store ./store.json --tenant-id <uuid> --dsn ...
```

Verified 2026-09-19 against the private `IgorKlymenko/corev1` through the API with a `gh` token: 44 files in scope, 271 episodes, second sync 0 fetches, citation URLs `…/blob/main/<path>`. A local checkout goes through `github.local.LocalRepo` instead.

| Rule | Where | Why |
|---|---|---|
| `external_id = repo/{repo_id}/{path}`; `title = owner/name/path`; `url = …/blob/{ref}/{path}` | `connector.external_id_for` | ADR connectors 4 |
| Cursor per repo: `store["connectors"]["github"][repo_id] = {full_name, head, render_version, include, blobs{path: sha}, symbols{path: names_of}, ambiguous[], index}` | `GitHubConnector._repo_documents`, `cursors_in` | same head + version + include → 1 request; else tree + changed blobs, plus unchanged files whose names flipped ambiguity (ADR 5, 11) |
| `--include prefix…` narrows a monorepo; paths outside are never rendered and retire by absence | `GitHubConnector(include=…)` | the legacy half of a repo is not current knowledge | Measured 2026-09-19: dropping `old_server/entity_linking` retired 304 episodes and re-rendered 17 current files (100 episodes) whose short names stopped being ambiguous. |
| `LocalRepo(root)` stands in for the API over a git checkout (`git ls-files -co --exclude-standard`: tracked plus untracked-not-ignored, git blob shas); head = hash of the working tree; `html_url` from the `origin` remote when it is GitHub. CLI: `--local <checkout>` instead of the API, `--repos` names it | `github/local.py`, `connector.main` | offline runs, tests, private repos without a token; uncommitted edits and not-yet-added modules are indexed |
| `--include CLAUDE.md` matches the file itself, not only paths under `CLAUDE.md/` | `GitHubConnector._included` | before 2026-09-19 a single-file include silently indexed nothing |
| Paths gone from a whole tree are `removed()` | same | enumerable source; a truncated tree is refused by `GitHub.tree` |
| `wanted(path, size)`: parser extension or `PROSE_SUFFIXES`, not under `SKIP_DIRS`, ≤ `MAX_BLOB_BYTES` (200 000) | `connector` | vendored and generated code is not the company's knowledge |
| Binary (NUL in the first 8 KB) or unparseable → `unparsed`, not remembered, retried next sync | `render_file` | a Python 2 file should not fail a repo |
| 403/429 with `retry-after` or `x-ratelimit-remaining: 0` → sleep and retry; 5xx → 3 retries with backoff; other 4xx → `GitHubError` | `client.GitHub.get` | limits *unverified* (ADR open D) |
| No token → no `Authorization` header, `/rate_limit` as preflight | `client.GitHub` | try it on a public repo without setup |

`RENDER_VERSIONS["github"] = 3` (`1_episodes.py`); bump it when `render.py` or `strip_markup` output changes, and the next sync re-fetches every blob. `names_of(parsed, package=…)` records the module and whether it is an `__init__`, which `ambiguous_from` needs to tell a re-export from a real second definition (ADR 11, amended).

Role: the CLI writes through `5_store.upload` as `app_ingest` (tenant claimed). The App install path (`app_connector`, installation id as workspace) is ADR connectors open B. Migration `20260918120000` adds `github` to the three source CHECKs.

Tests: `server/tests/public/connectors/` — parser (13), render (8), sync and seeds (7), seeded chains (7), connector with a fake client (9), client with `httpx.MockTransport` (8).
