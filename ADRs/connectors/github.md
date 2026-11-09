# Decision Log — Connectors and the GitHub connector

Decisions taken while adding the second source. The first half is about the
connector contract every source shares; the second is GitHub and code. Slack's
own log is `../slack/connector.md`; entry 4 of `../supabase/security.md`
("a row, not a role") is assumed throughout and not re-argued.

Each entry: what was decided, why, and what it costs. Supersede rather than edit
once shipped.

Status key: **Accepted** · **Deferred** · **Open** · **Superseded**

Claims marked *measured* were verified against this repository on 2026-09-18.
Claims about GitHub's own limits are *unverified* and must be checked against
current API docs before they are relied on.

---

## The contract

### 1. A connector yields documents; `base.sync` is the only door — *Accepted*

**Decision.** `src/connectors/base.py`: a `Connector` is `source`,
`documents()` and `removed()`. `sync(store, connector, tenant_id=…)` feeds
every document through `episodes.sync_file` and retires every removed id
through `episodes.retire_file`. Connectors never touch the store otherwise.

**Why.** The rules that make a sync cheap and correct -- file identity,
render versions, content-addressed episodes, retire-not-delete -- are paid for
once, in one place. The Slack connector wrote to the store directly because
it was first; the second source is where duplication would have started.

**Cost.** `removed()` must be consumed after `documents()`; a connector that
computes deletions from a listing only knows them once the listing is done.
The order is fixed in `sync`, and a connector that reports a deletion it did
not positively observe retires live documents.

### 2. Structured sources carry their references as seeds — *Accepted*

**Context.** The coreference model reads prose. Code has already resolved its
references: `sync_file` in one module is one function, and no passage needs to
"make the identity clear".

**Decision.** `Document.mentions` are offset-addressed occurrences grouped by a
`key`. `base.attach_seeds` cuts them per episode into `episodes[*].seeds`
(episode-relative offsets, sorted by key). Stage 2 turns them into
`proper_name` spans with origin `connector` and folds them into the model's
chains with `chains.merge_seeded`: a seed absorbs any model chain that shares a
span's offsets or text. The model still runs over the episode -- docstrings are
prose, and that is where "tenant", "Okta", "the RLS wall" appear.

**Why.** Two chains in one episode that both hold `ingest` would read
downstream as a name the passage kept apart (`pairing.build_evidence` marks
the surface contested and stops trusting it). Folding rather than appending
keeps the symbol's identity in one chain, with the qualified name as a second
surface so alias evidence can link a Slack `ingest` to the code's
`src.inject.main.ingest` -- the linker's ordinary path (`alias_hit`, weight
3.5), no code-specific linking rule.

**Cost.** One coref call per code episode, as for prose. A seed whose text no
longer matches (a connector bug) is dropped with a warning, not stored. Seeds
that change while the text does not (a better parser) clear `extracted_at`
so the episode is re-extracted -- the honest cost, one call.

### 3. Seeds are proper names; the head is the longest — *Accepted*

**Decision.** Every seed span is `proper_name`. `spans.pick_head` therefore
names the chain by its longest proper name, the qualified one
(`itsdangerous.signer.Signer`), so `entity_display_name()` shows it.

**Why.** The short name is what people type and is kept as a surface; the
qualified name is what makes two `run`s different entities. A display name
that says which `run` is worth more than a shorter one.

**Cost.** Name-vector cosine between `run` (a Slack chain's head) and
`pkg.jobs.run` is lower than between two `run`s; alias evidence through the
shared surface carries the pair instead. *Not measured.*

---

## GitHub

### 4. A file is a document, keyed by repo id and path — *Accepted*

**Decision.**

```
external_id   repo/{repo_id}/{path}
title         {owner}/{name}/{path}
url           {html_url}/blob/{ref}/{path}
```

**Why.** The numeric repository id survives renames and transfers; the path
is the only identity a file has inside it (a blob sha is its *content*, which
is what `full_text_hash` already tracks). Mirrors `thread/{channel_id}/{ts}`:
the source's stable container id, then the object.

**Cost.** A moved file is a retirement plus a new document; its episodes are
content-addressed, so extraction is not repeated, but chain ids (keyed by
episode, keyed by file) are. Same as Slack's cost for a renamed channel.

### 5. The cursor is the head commit and every path's blob sha — *Accepted, measured*

**Decision.** Per repo, the store keeps `{head, render_version, blobs: {path:
sha}}` under `connectors.github`. Same head and version: one request (the
commit). New head: one tree call, then blobs only for changed shas, and
absence from the tree retires the path. A render-version bump empties the
map so everything is re-fetched.

**Measured** on `pallets/itsdangerous` (28 rendered files): first sync 28
blob fetches and 83 episodes; second sync 0 fetches, 0 changes.

**Why.** Trees and blobs are content-addressed on GitHub's side, so the diff
is a dictionary comparison and the cost of a sync is proportional to what
changed, not to the repo. `connector_channels` (ADR slack 12) is the same
idea for Slack and does not exist yet; this one lives in the store because
the store is where the Slack CLI keeps its state too, and the schema question
(where cursors live once there is a worker) is Open item A.

**Cost.** A truncated tree *(unverified: past 100,000 entries or 7 MB)* would
read as mass deletion; the client refuses it (`GitHub.tree`), and syncing by
subtree is not implemented.

### 6. Python by `ast`; other languages behind a registry — *Accepted*

**Decision.** `src/connectors/code/model.py` defines `Symbol`, `Parsed`, the
`Parser` protocol and `PARSERS` by extension. `python.py` is the one parser,
on the standard library. Markdown and text files go through
`episodes.strip_markup` with no seeds. Everything else is skipped.

**Why.** The interpreter that runs the ingest already parses Python exactly;
a tree-sitter grammar per language is a dependency, a version and a wheel to
pin, and the first customer corpus is this one. The registry is the seam:
a second language is a class, not a change to the connector.

**Cost.** A file the running Python cannot parse is counted as `unparsed` and
skipped, and is not remembered in the cursor so it is retried next sync.
Non-Python repos yield only their prose.

### 7. References are what the file can resolve; nothing else — *Accepted*

**Decision.** A call is kept if its head resolves through an import, a
same-module definition, or `self.m` where `m` is defined on the class.
Builtins and the standard library are dropped. Calls on local variables are
dropped.

**Why.** Kept as written, `cur.execute` in every database module becomes one
surface `execute` with no referent, and `json`/`len` become the biggest hubs
in the graph. The judge (ADR tables 30) would catch some of it later at a
cost per surface; not emitting it is free.

**Cost.** Duck-typed calls (`client.users()` where `client: Slack`) are lost.
Type inference is the only way back, and it is not on the list.

### 8. One paragraph per symbol, nothing positional — *Accepted, measured*

**Decision.** `render.py` emits, per symbol: kind and name with the qualified
name in parentheses, the signature, `defines:`/`calls:`/`imports:`/`uses:`
lines, then the docstring. No line numbers, no offsets, no bodies. The
module paragraph lists what the module defines and imports.

**Why.** Same reasoning as one Slack message per paragraph (ADR slack 8):
`split_episodes` packs paragraphs and episodes are content hashes, so an edit
inside one function re-extracts its episode and the packing after it, not the
file. A line number changes on every edit above it and would churn every
symbol below an inserted blank line.

**Measured.** *(test)* Appending a function changes the module paragraph (the
index) and the tail; the symbols between are byte-identical.

**Cost.** Bodies are not indexed: a question about a variable name inside a
function finds the function only if the docstring mentions it. The module
paragraph re-extracts on every added or removed top-level name.

### 9. The import root is the first directory without `__init__.py` — *Accepted*

**Decision.** `PythonParser.module_name(path, tree)` walks up from the file
while the directory holds an `__init__.py`; the first that does not is the
import root. `src/pkg/mod.py` is `pkg.mod` in a src-layout repo and
`src.pkg.mod` in one where `src` is a package (this repository). With no tree
every directory is a package.

**Why.** The qualified name is what keeps two `run`s apart and what people
would type; a heuristic on directory names (`src`, `lib`) was tried first and
got this repository wrong.

**Cost.** Namespace packages (no `__init__.py`) are read as layout
directories, so their modules lose the prefix.

### 10. Token from the environment; the App install path is not built — *Accepted, interim*

**Decision.** The CLI reads `GITHUB_TOKEN` and runs the upload path as
`app_ingest` with a claimed tenant, exactly as the Slack CLI does. No token
means unauthenticated access, public repositories only. A GitHub App
installation -- installation id as `connector_installations.external_id`,
installation token fetched through `token_ref`, worker as `app_connector` --
is the production shape and is Open item B.

**Why.** Same order as Slack: what is written is settled; how the token
arrives is routing and changes nothing here.

**Cost.** Nothing verifies that a token's holder may see the repos it names
beyond GitHub itself; the install path is where that becomes the database's
job.

### 11. A short name that means two things is never a mention on its own — *Accepted, measured*

**Context.** Open item C, measured. Short names collide constantly in real
code: 8–25 % of a repo's short names denote more than one qualified thing
(this repo's live source, `old_server/entity_linking`, httpx, fastapi, openai,
spacy), and 6–22 % of rendered in-repo references name one of them. Rendered
bare, `run` in two modules gives two chains one shared surface, and alias
evidence -- the strongest prior weight -- fuses them.

**Rig.** `server/public/scripts/eval_code_collisions.py`: render a repo under
a variant, build seed-only chains (no model call), encode heads with nomic,
run the linker's blocking, evidence and prior scorer as stage 4 does. Ground
truth is free: every chain carries the qualified name it came from, so a
merge across keys is false and a split within a key is a miss. Review-band
pairs are counted, not sent.

**Measured** (automatic merges true/false; pairs sent to review; entities
mixing two keys):

| Variant | live source (701 chains) | `entity_linking` (1070) | httpx (1257) | fastapi (1575) |
|---|---|---|---|---|
| baseline | 501 / **303**, 817 reviews, 71 impure | 1462 / **2039**, 630, 81 | 1371 / **2564**, 909, 75 | 3379 / **2180**, 2024, 126 |
| qualify ambiguous references only | 502 / 68, 1976, 15 | 1522 / 1206, 3151, 27 | 1388 / 645, 2235, 59; 22 missed | 3495 / 239, 5025, 45; 16 missed |
| qualify every reference | 515 / 74, 2990, 14 | 1543 / 1244, 3844, 26 | 1455 / 618, 2769, 54 | 3526 / 255, 6217, 44 |
| **qualify + bare name on ambiguous definitions is plain text** (`defs`) | 506 / **1**, 2125, 1, 0 missed | 1523 / **2**, 3261, 1, 0 missed | 1421 / **12**, 3088, 2, 0 missed | 3515 / **12**, 5440, 5, 0 missed |
| methods as `Class.method` | 504 / 244, 983, 63 | 1464 / 2026, 657, 80 | 1446 / 49, 2136, 6 | 3424 / 658, 3622, 101 |
| evidence index document only | 510 / 160, 1378, 57; 67 missed, 28 fragmented | 1377 / 410, 901, 56; 290 missed | 1354 / 117, 3949, 62; 408 missed | 3358 / 429, 4067, 117; 276 missed |
| qualify + index | 578 / 1, 2522, 1; 8 missed | 1726 / 2, 3245, 1; 9 missed | 1836 / 28, 3208, 9; 38 missed | 3787 / 24, 5998, 8; 36 missed |
| defs + `Class.method` | 510 / 1, 2125, 1 | 1523 / 2, 3312, 1 | 1466 / 0, 3058, 0 | 3559 / 10, 5454, 4 |
| defs + `Class.method` + index | 590 / 1, 2642, 1 | 1735 / 2, 3355, 1 | 1986 / 0, 3803, 0 | 3884 / 4, 6380, 2; 1 missed |

`Class.method` on top of `defs` removes the last 12 false merges on httpx
(whose collisions are `Client` / `AsyncClient` method pairs) and 2 of 12 on
fastapi, at the same review count. It is not the default because it removes
the bare method name as a surface for *every* method, ambiguous or not, and
the rig cannot see what that costs: a Slack message says "the send method",
not `Client.send`. Turn it on per repo when method collisions dominate.

**Decision.** `render.Style` defaults: a short name in `ambiguous_names` (any
name with more than one qualified referent across the repo, definitions and
references counted alike) renders qualified wherever it is referenced, and on
its own definition the bare name is plain text beside the qualified mention.
The index document and `Class.method` stay as measured options, off.
`RENDER_VERSIONS["github"]` is 2.

**Why.** Qualifying references alone is not enough: the definitions themselves
share the bare surface, and on a repo with many same-named functions across
files (`entity_linking`: `ppr`, `extract`, `embed` in demos and stages) that
leaves 1206 false merges. Making the bare name plain text removes the shared
surface at its source; the model may still span the word in prose, but then
it is the model's chain, absorbed by `merge_seeded` only through a seed that
exists. Contesting by evidence (the index) works on the linker's own terms but
is blunter: a contested surface stops matching *itself*, so it also splits
what it should have joined (67 and 290 missed pairs).

**Amended 2026-09-19.** A reference into the repo's own package tree that
no module file defines -- `src.inject.pairing.keys_settle`, the stage
loader's alias for `4_link.py` -- is a re-export the parser cannot follow and
no longer counts toward ambiguity (`render._dangling`: walking the dotted
prefix up, a module file means a member and is resolvable; a package first
means a missing child). Measured on the Slack test: every stage function in
this repository had been "ambiguous", so their definitions carried no bare
name, the model spanned the bare word as prose and the judge marked
`keys_settle` a non-name; the first `@Nazar` question about it found
nothing. After the amendment 6 of 118 ambiguous names cleared, 7 files
re-rendered, and the bare names became seeds. External references still
count. `RENDER_VERSIONS["github"]` is 3.

**Cost.** Review-band pairs grow 2.5–5× under `defs` (817 → 2125, 630 →
3261, 909 → 3088, 2024 → 5440): pairs the bare surface used to settle as a
merge now reach the reviewer with the qualified names in view. That is the
honest cost of no longer guessing, and it is paid once per pair
(`store["reviews"]`). The connector renders repo-wide: the cursor keeps `names_of` per
path, and a file whose names flipped ambiguity is fetched and rendered again
although its content did not change (measured on a synthetic three-file
repo: two re-fetches by known sha, none for the file that shows no flipped
name). Slack's `ingest` still reaches the code's `ingest` when the name is
unambiguous; when it is not, the two things called `run` are two entities,
and a question about `run` is the clarify stance's job (ADR slack 16). The
rig's reviews are counted from the prior scorer alone; review outcomes with
the model are not measured.

### 12. A seeded surface is a name; the judge is not asked — *Accepted, measured*

**Context.** Stage 2b (ADR tables 30) sends every new identifying surface to
a prose judge: would the author look it up by this string, does it denote one
specific thing. Run over this repository's live source, `entity_linking`,
the ADRs and `CLAUDE.md` synced into the author's vault store (294 code
episodes, 8,548 code chains), the judge rejected 421 of the 605 seed
surfaces it saw -- `candidate_pairs`, `unresolved_report`,
`entity_linking.pairing.cluster`, `src.inject.utils.spans.mark_text` -- and
714 seeded chains were withheld with no name left. Only 11 of 117
cross-source entities were anchored on a code symbol.

**Decision.** `base.name_seeds`, called from `attach_seeds`: every seed
mention that is not a bare lower-case word is written to `store["names"]` as
`True`, overriding any earlier verdict; a bare word (`work`, `run`) is
written only when no verdict exists. `main.judge` then never sends them,
since it judges only surfaces without a verdict.

**Why.** The judge decides whether a string names something. A connector
already resolved the mention to a qualified symbol; the question is answered
by construction, and paying a model to answer it again yields a worse answer
(it reads `function candidate_pairs (…)` as prose and says no). A bare word
keeps the prose verdict because in a docstring it is a common noun, and the
chain is still known by its qualified name.

**Cost.** Applied to the same store: 442 verdicts written, 358 of them
flipping a judge's `False`. Re-linking then re-scores every pair those
chains touch (the second run below). A surface that is both a code name and
a prose junk word stays junk, so a Slack question naming a function by its
bare name reaches it through string similarity rather than a surface hit.

### 13. Connector keys settle a pair; the reviewer is not asked — *Accepted, measured*

**Context.** With entry 12 in place the code chains had names and entered
linking. One entity then held 853 chains (`chains`, `ChainRecord`,
`PairScorer`, `Thresholds`, 530 of them seed-only symbol chains). Of the
4,708 merge edges inside it, 2,057 were the reviewer's "same" verdicts;
without them the largest component was 71 chains. Classified by what the
two sides were: 903 reviewer merges joined two symbols with *different*
keys (`PairScorer` with `PairScorer.fit` and `PairScorer.__init__`), 288
joined a symbol to a prose chain ("Entity Linking" with
`entity_linking.4_link.Evidence`), 950 joined two prose chains (`chain`,
`chains`, `CHAINS` -- the concept, and right). The alternative hypothesis,
the model's own chains grouping several identifiers from a signature line
and feeding alias evidence, was refuted: 6 such chains in 7,342.

**Decision.** `ChainRecord.key` carries the connector key (`main._seed_key`,
from the episode's seeds by offsets). `pairing.keys_settle(a, b)` is true
when the keys differ or exactly one side has a key; `main.link` neither
sends such a pair to the reviewer nor merges it on an abandoned review. A
pair the prior scorer merges on its own still merges.

**Why.** The reviewer exists for what the passages leave uncertain. A
qualified name resolved by a parser is not uncertain, and the reviewer,
shown two symbol paragraphs, reads a class and its method as one thing. Keys
are not made a hard cannot-link because the scorer's own merges across
different keys are the real aliases -- `episodes.empty_store` for
`1_episodes.empty_store` is the same function under the loader's alias, and
no key knows that.

**Cost.** A symbol and a prose description of it ("the ingest function") no
longer meet in the review band; they link only when a shared surface or the
name vector settles it. *Not measured* how often that path carried a right
merge before.

**Measured**, vault (301 episodes) plus this repository's live source, ADRs
and `CLAUDE.md` (293 code episodes), three consecutive runs on one store:

| Run | Entities | Cross-source | Anchored on a symbol | Largest entity |
|---|---|---|---|---|
| seeds judged by the prose judge | 1,930 | 117 | 11 | 61 chains |
| entry 12 (seeds are names) | 1,802 | 103 | 11 | **853** chains (`chains` + `PairScorer` + `Thresholds` + …) |
| entry 13 (keys settle), full re-link | 2,046 | 107 | 10 | 78 chains (`chains`, `chain`, `CHAINS`: the concept) |

Cross-source entities are concept-level (`Postgres`, `PPR`, `Supabase`,
`Neo4j`, `Cloud Tasks`, `RLS`, `Slack`, `Notion`, `OpenAI`, `slack_users`);
a symbol reaches the vault only through an exact surface (`entity_linking` /
"Entity Linking", `chains`). The vault does not name functions, so ten is
the honest number, not a ceiling to tune toward.

---

## What did not work

| Idea | Why it was dropped |
|---|---|
| Skip the model for code episodes, store seeds as the only chains | Docstrings are the prose that names the business concepts a codebase shares with Slack; losing them loses the cross-source edges the connector exists for. |
| Keep unresolved calls as written (`cur.execute`) | One surface per common method name across every file, all hubs, none an answer to anything. Entry 7. |
| Strip `src`/`lib` by name | Wrong for this repository, where `src` is the package. Entry 9. |
| Line numbers in the symbol header | Every edit above a symbol would re-hash it. Entry 8. |
| Blob sha as `external_id` | It is the content's identity, not the file's; every edit would be a new document. Entry 4. |
| `__init__.py` as a filename mention | Every package has one; the surface aliased all packages to each other. Entry 11 rig, baseline. |
| Contest colliding names by evidence alone (the index document) | Halves false merges but a contested surface no longer matches itself: 67 / 290 same-key pairs split. Entry 11. |
| Qualify every reference | No better than qualifying only ambiguous ones, at 1.5× the review-band pairs. Entry 11. |

## Open items

**A. Where cursors live once there is a worker.** The store is a file per
tenant today; a `connector_cursors(source, external_id, key, value)` table is
the obvious shape and would also hold Slack's `connector_channels` (ADR slack 12).

**B. GitHub App install.** OAuth callback, installation id → tenant,
installation token through `token_ref`, `push` webhooks into
`internal.events` so a sync is one event rather than a schedule.

**C. Short-name collisions across a repo.** *Closed by entry 11.* What
remains open is the model's side: a bare colliding name in a docstring is now
a model chain with no seed to join, and whether the reviewer then merges it
with the right definition is not measured.

**D. Rate limits — verify.** 60/hour unauthenticated, 5,000/hour with a
token, secondary limits with `retry-after`: all from memory.

**E. Which files are prose.** `.md`, `.markdown`, `.rst`, `.txt` today;
notebooks, YAML with comments, and `pyproject.toml` are not read.
