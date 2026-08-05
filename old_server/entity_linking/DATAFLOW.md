# Data flow

What object each stage receives, what it produces, and what it costs.

```
documents ──1──▶ episodes ──2──▶ chains ──3──▶ chain vectors ──4──▶ entities
             store          LLM×n          local             LLM×m + local
```

Every stage reads and writes one JSON store. Nothing is passed in memory between
runs, so any stage can be re-entered after a crash and will redo only what is
missing.

```
store = {
  "files":      {file_id:      {...}},   # stage 1
  "episodes":   {episode_id:   {...}},   # stage 1
  "candidates": {candidate_id: {...}},   # stage 2
  "entities":   {cluster_id:   [...]},   # stage 4
}
```

---



## Stage 1 — `1_episodes.py` : documents → episodes

**In:** a directory of `.md` (or any connector's rendered text), a `tenant_id`.
**Out:** `store["files"]`, `store["episodes"]`, and a `FileDiff` per document.
**Cost:** local only.

Splits each document on blank lines, greedily merging paragraphs up to
`max_chars`. Identity is content-addressed:

```python
file_id    = uuid5(NS, f"file:{tenant_id}|{source}|{external_id}")
episode_id = uuid5(NS, f"episode:{file_id}|{sha1(text)}")
```

```json
// store["files"][file_id]
{
  "file_id": "…", 
  "tenant_id": "…", 
  "source": "upload",
  "external_id": "notes/a.md", 
  "full_text_hash": "…", 
  "render_version": 1,
  "episode_ids": ["…"], 
  "synced_at": "…", 
  "removed_at": null
 }

// store["episodes"][episode_id]
{
  "episode_id": "…", 
  "file_id": "…", 
  "text": "Sarah Chen leads the identity migration. …",
  "extracted_at": null, "removed_at": null
 }
```

`file_id` is **path**-based so a document keeps its identity when edited;
`episode_id` is **content**-based so an unchanged paragraph keeps its identity
when an edit above it moves it. Keying episodes on offsets instead would
renumber everything below any insertion.

The episode carries no position, deliberately — there was an `ordinal` column and
it was dropped. Nothing in the pipeline reads order, so storing it meant
maintaining a version-dependent field on every sync to serve a consumer that does
not exist yet.

If order is ever needed, re-split the document and match on text: episodes are
verbatim slices, so `doc.find(text)` recovers position exactly. That costs one
document read at query time, which is the right place to pay it. Don't reinstate
an ordering column without a consumer that actually needs one.

`sync_file` returns `FileDiff(status, unchanged, added, removed)`. An unchanged
`full_text_hash` short-circuits before splitting.

---



## Stage 2 — `2_chains.py` : episodes → chains

**In:** episodes where `extracted_at is None`.
**Out:** `store["candidates"]`; sets `extracted_at` on each episode.
**Cost:** one LLM call per episode.

The model never sees raw text and never hands back rewritten text. Instead:

1. `spans.parse_spans` finds every span a parser can justify — named entities,
  noun chunks, pronouns, and proper-noun runs nested inside a chunk — and issues
   `c0, c1, …`. Types are **derived** (`proper_name` / `definite_description` /
   `pronoun` / `nominal`), never asked of the model.
2. `spans.mark_text` renders a *copy* with those ids inlined, nesting preserved:
  ```
   ⟦c0 Sarah Chen⟧ leads ⟦c1 the ⟦c2 identity⟧ migration⟧.
  ```
3. The model returns **ids only**, plus any span the parser missed as `n`* with
  verbatim text.
4. `locate_new_spans` re-finds each `n*` in the clean text — exact, then
  whitespace-insensitive, then fuzzy — and records which tier matched. A span
   that does not match *exactly once* is dropped and counted.
5. An id that was never issued is a protocol violation: retry once, then fail the
  episode. `extracted_at` stays unset so the next run picks it up.

```json
// store["candidates"][candidate_id]
{
  "candidate_id": "…", 
  "episode_id": "…",
  "start": 0, 
  "end": 10, 
  "text": "Sarah Chen",
  "candidate_type": "proper_name", 
  "origin": "parser",
  "chain_id": "…", 
  "is_head": true
 }
```

Offsets are safe to store here — the episode text is content-addressed, so a
given `episode_id` always denotes the same string.

Chains sharing an identical **proper name** are then unioned deterministically;
the model splits those often enough that it is not worth asking twice.

---



## Stage 3 — `3_encode.py` : chains → vectors

**In:** live chains (whose episode has no `removed_at`).
**Out:** in-memory `ChainRecord` per chain.
**Cost:** local; one forward pass per episode, plus one batch over the names.

Two vectors, because identity and disambiguation want opposite things:


| vector      | how                                          | good for                            |
| ----------- | -------------------------------------------- | ----------------------------------- |
| **name**    | head string alone, `search_document:` prefix | cross-episode identity — AUC 0.936  |
| **context** | span-pooled over the episode, **uncentered** | same-string pairs the name can't    |


The name vector decides identity. The context vector reaches stage 4 only through
`context_gated`, which fires on pairs whose surfaces are the same string — the
one case where the name features carry no information. See 4c for the numbers and
for why the vector is uncentered.

`--no-context` skips the per-episode forward pass entirely.

```python
ChainRecord(chain_id, episode_id, head, surfaces, candidate_type,
            name_vector, context_vector)
```

---



## Stage 4 — `4_link.py` : vectors → entities

**In:** `ChainRecord`s from every episode.
**Out:** `store["entities"]`.
**Cost:** one LLM call per pair in the review band; everything else local.

### 4a. Evidence — free, from stage 2

Per-episode coref decisions become corpus-wide pair evidence:

```python
Evidence(
  same_chain,          # some episode put these two surfaces in one chain
  split_in_episode,    # some episode kept them in different chains
  ambiguous_surfaces,  # a surface bridging two surfaces kept apart elsewhere
)
```

`ambiguous_surfaces` exists because surfaces are not entity identifiers. One
episode reveals `Sarah Chen ≡ S. Chen`, another reveals `Steven Chen ≡ S. Chen`;
without the veto, the shared short form fuses two people. Alias precision went
73% → 100% once contested surfaces stopped carrying evidence.

### 4b. Blocking

`ann_candidate_pairs(records, k=50)` — hnswlib over name vectors. At 444 chains
it cut 98,286 pairs to 14,376 (**85% fewer, 99.9% recall, 5.6× faster**). It does
*not* reduce LLM cost: it removes far-apart pairs, which the scorer would have
split confidently anyway.

### 4c. Scoring — five features

```
name_cosine        embedding similarity of the two head strings
string_similarity  best match over all surface forms
alias_hit          some episode asserted these surfaces corefer
cooccurrence       some episode saw both and kept them apart
context_gated      context similarity, but ONLY when string_similarity >= 0.90
```

Combined by logistic regression into a calibrated probability. Cosine alone
cannot do this: `Sarah Chen`/`S. Chen` and `Sarah Chen`/`Sarah Kim` differ by
0.03 cosine. The evidence features separate them, 0.94 vs 0.10.

Dropped after measurement: `type_agreement` (zero variance in a single-type
corpus).

#### The context gate

`context_cosine` was originally dropped at AUC 0.505 — chance. That reading was
an artifact of **episode-centering**. Uncentered, the same signal measures 0.898
on the synthetic corpus and 0.745 on ECB+. The centered vector subtracts the
episode's shared component and takes the entity's own signal with it.

The gate exists because the signal is conditional. Over all pairs, `name_cosine`
is already at 0.936 and context adds noise. On pairs whose surfaces are the
**same string** — one short form denoting two people — the name features carry
*no information at all*, and context is the only signal left:

| ECB+, 2466 same-string cross-document pairs | AUC |
|---|---|
| `name_cosine` | 0.414 (constant; noise) |
| `string_similarity` | 0.500 (chance by construction) |
| name + string combined, cross-validated | **0.500** |
| + `context_gated` | **0.690** |

**End to end it changed nothing: 0.774 against 0.775.** The fitted weight came
out at +0.13, and dev AUC moved 0.9752 → 0.9754.

The reason is worth recording, because it is about ECB+ and not about the
feature: same-string pairs there are **91.8% coreferent**, so the subset where
context helps is one where "merge" is already the right answer almost always, and
union-find merges it regardless of ranking. A better ordering inside a subset
that should mostly merge buys no CoNLL.

It is kept because it is nearly free once the vector exists, it costs nothing
off-gate, and the case it targets — a contested short form denoting two people —
is dense in enterprise corpora and rare in newswire. Under the *prior* weights it
does help (0.716 → 0.742). Treat that as untested on real contested data rather
than as a result.

Cost: one nomic forward pass per episode, which did not previously happen at all.
`--no-context` skips it.

### 4d. Three-way decision

```
score <  low          → split      no LLM
low ≤ score < high    → review     one LLM call, with the passages
score ≥ high          → merge      no LLM
```

Two thresholds, not one ± a range: the split side is easy and the merge side is
hard, so a symmetric band wastes review budget. `choose_thresholds` derives both
from labelled pairs at a target precision, so band width is an **output** — a
wide band means the features are not separating.

### 4e. Clustering

Plain union-find over accepted merges. Constraining it with cannot-link edges
scored **0.774 CoNLL against 0.945** on the same decisions, because the decision
source contradicts itself — 17 surface pairs were merged in one pairing and
split in another, and a veto lets one dissent overrule every agreeing merge.

```json
// store["entities"][cluster_id]
["M. Lopez", "Maria Lopez"]
```

---



## Where each identity comes from


| object    | key                     | stable across           |
| --------- | ----------------------- | ----------------------- |
| file      | `uuid5(tenant           | source                  |
| episode   | `uuid5(file_id          | sha1(text))`            |
| candidate | `uuid5(episode_id       | start                   |
| chain     | `uuid5(chain:episode_id | head span)`             |
| entity    | cluster index           | **nothing** — see below |


Entity ids are positional and change whenever clustering changes. Fragmentation
falls as a corpus grows and more aliases are revealed, so entities *should*
consolidate over time — which means identity cannot be a hash of current
membership if the graph is to survive re-runs. Unresolved.

## Measured end to end

92 episodes, 26 gold entities, one LLM call per episode plus 236 reviews:

```
MUC    P 0.977  R 0.977  F1 0.977
B3     P 0.912  R 0.972  F1 0.941
CEAFe  P 0.892  R 0.892  F1 0.892
CoNLL                    F1 0.937      26 predicted vs 26 gold
```

Two of 26 entities were fused, both on contested short forms (`S. Chen`,
`J. Park`). Report **CEAFe and B³, not pairwise P/R** — pairwise reported 0.971
precision on a run that produced 45 clusters for 26 entities.

These numbers come from a synthetic corpus written alongside the algorithm, with contested initials at a density real data will not have. ECB+ would give a figure not contaminated that way.
---

## Query — surfaces → entities → PPR → passages

Indexing ends at entities. Retrieval runs the same graph backwards, and `chains`
is the hinge in both directions:

```
surfaces ──chain_surfaces──▶ chains ──entity_id──▶ entities
                               │
                          episode_id
                               ▼
                            episodes ──file_id──▶ files
```

```
query
  1. expand ─────────────► terms[]            LLM, optional
  2. embed ──────────────► vector(1536)       text-embedding-3-large
  3. surface lookup ─────► [(surface, score)] top-k, vector + trigram, RRF
  4. bridge ─────────────► [(entity_id, score)]
  5. seed scoring ───────► {entity_id: weight} top-m, L1-normalised
  6. PPR ────────────────► [(entity_id, ppr_score)]   managed service
  7. entity → passages ──► [(episode_id, text)]
  8. rerank ─────────────► ranked episodes
  9. answer + citations    files.title / files.url
```

**Seeding** walks right, **passages** walk down — both are indexed joins over two
tables, so nothing is denormalised and re-clustering (`chains.entity_id`) changes
both answers at once with no cache to invalidate.

```sql
-- 4+5: surface hits -> weighted entity seeds
-- The hit sum and the breadth count MUST be separate aggregates. Joining
-- chain_surfaces a second time to count breadth duplicates every hit row once
-- per surface of the chain, giving score * n / sqrt(n) = score * sqrt(n) --
-- amplifying breadth instead of damping it. Measured: MRR 0.718 -> 0.658, the
-- whole gain gone. DISTINCT matters for the same reason one level down: a
-- surface can sit in several chains of one entity and must still count once.
WITH hits(surface, score) AS (
  SELECT * FROM unnest($surfaces::text[], $scores::float8[])
),
matched AS (
  SELECT DISTINCT c.entity_id, h.surface, h.score
  FROM hits h
  JOIN chain_surfaces cs ON cs.tenant_id = $t AND cs.surface = h.surface
  JOIN chains c ON (c.tenant_id, c.id) = (cs.tenant_id, cs.chain_id)
  WHERE c.entity_id IS NOT NULL
),
breadth AS (
  SELECT c.entity_id, count(DISTINCT cs.surface) AS n_surfaces
  FROM chains c
  JOIN chain_surfaces cs ON (cs.tenant_id, cs.chain_id) = (c.tenant_id, c.id)
  WHERE c.tenant_id = $t AND c.entity_id IN (SELECT entity_id FROM matched)
  GROUP BY 1
)
SELECT m.entity_id, sum(m.score) / sqrt(b.n_surfaces) AS weight
FROM matched m JOIN breadth b USING (entity_id)
GROUP BY m.entity_id, b.n_surfaces
ORDER BY weight DESC LIMIT $m;

-- 7: ranked entities -> live passages
SELECT e.text FROM chains c
JOIN episodes e ON (e.id, e.tenant_id) = (c.episode_id, c.tenant_id)
WHERE c.tenant_id = $t AND c.entity_id = ANY($ranked)
  AND e.expired_at IS NULL
ORDER BY $rank_of_entity LIMIT $n;
```

`e.expired_at IS NULL` is load-bearing: retired episodes keep their chains, so
without it a deleted document still serves passages.

### What the seeding weight is, and what it is not

`sum(score) / sqrt(n_surfaces)` — breadth should help without letting an entity
with forty surfaces outrank the right answer on surface count alone. Measured on
ECB+ with this pipeline's own extraction: **MRR 0.689 → 0.749, R@1 0.546 → 0.639.**

Absolute numbers drift ~0.03 between runs — stage 2 is a non-deterministic LLM
call and produces slightly different chains each time — but the ordering of the
variants is stable. A second run gave 0.653 → 0.718 for the same comparison.

Two weights that look obvious and are **not** used, both measured negative:

| weight | MRR | why it fails |
|---|---|---|
| `idf(n_episodes)` | −0.017 | junk surfaces are *rare*, not frequent (75.7% appear in one episode), so IDF up-weights them |
| `1/n_entities` | −0.097 | junk entities are singletons, so this hands them the maximum |

`n_entities` is kept on `surfaces` as the ambiguity signal — `> 1` routes a
mention to clarification — never as a seeding weight.

### Ambiguity at query time

A surface claimed by several entities splits its mass across them rather than
replicating it. Past a fan-out threshold, drop the surface or return a
clarification built from the candidates' `canonical_name`. Deferring here is
deliberate: indexing must commit to a partition, retrieval has the query as extra
evidence and does not have to.

### One decision left open

Passages from **seed** entities, from **walked** entities, or from the **edges
traversed**. Seeds-only is precise and wastes the multi-hop walk; walked-entities
is what PPR is for; edge provenance gives the passage that justified each hop and
needs an `entity_edge_episodes` table that does not exist yet.

### Measured: retrieval

Two evals, opposite verdicts, and the difference is the point. ECB+, 277
passages, 12 topics, real extraction and real linking throughout.

**Single-passage queries** — a question generated from one passage, gold is that
passage. `demos/eval_retrieval.py`:

| system | R@1 | R@10 | MRR | candidate recall |
|---|---|---|---|---|
| dense over passages | 0.725 | 0.992 | **0.815** | 1.000 |
| seeds only | 0.108 | 0.550 | 0.250 | 0.733 |
| seeds + PPR | 0.108 | 0.600 | 0.273 | 1.000 |
| seeds + PPR, reranked | 0.683 | 0.950 | 0.776 | 1.000 |

Dense wins. Note the rerank: without it the graph path orders passages by a
five-valued step function over the seed scores, so everything containing the top
entity ties — that is stage 8 of the flow above, and skipping it costs 0.53 MRR.

**Footprint queries** — query with an entity's name, gold is *every* document
ECB+ says mentions it. `demos/eval_footprint.py`:

| system | R@5 | R@20 | total recall |
|---|---|---|---|
| dense over passages | 0.423 | 0.745 | 1.000 |
| graph (seeds) | 0.451 | 0.629 | **0.633** |
| graph + PPR, raw edges, alpha 0.5 | 0.449 | 0.787 | 0.999 |
| graph + PPR, PMI edges, alpha 0.2 | 0.440 | **0.795** | 0.999 |

Restricted to **alias-only** documents — where the entity appears solely under a
surface different from the queried name — the gap widens: dense 0.665 at R@20
against 0.769 for the tuned graph.

The tuning is worth less here than on the synthetic rig (0.787 → 0.795) because
ECB+ has little for it to exploit: every document in a topic covers one event, so
entities co-occur near-uniformly and there are few genuinely bridged pairs.

The ordering flips because the tasks differ. Precision retrieval of one passage
is what dense is for. Coverage of everything known about an entity is what the
graph is for, and it is the shape open-ended questions actually have.

Two things worth keeping in view:

* **Seeds alone cap at 0.631 total recall.** A third of an entity's documents are
  unreachable at any depth without expansion. PPR takes it to 0.999 — the first
  measurement in which the walk earns anything.
* **The alias number is a floor, not an estimate.** Every ECB+ document in a topic
  reports the same event, so a passage saying only `S. Chen` still shares the
  whole event vocabulary with a query naming `Sarah Chen`, and dense rides that.
  A corpus where one person appears in genuinely unrelated contexts — the
  enterprise case — removes that rescue.

Expansion *quality* needed its own rig — see "Edge weights" below. Short version:
with PMI-weighted edges the walk reaches the right neighbours and declines to
follow incidental co-occurrence, so the density worry does not materialise.

### Edge weights — PMI, not counts

The graph PPR walks is entity co-occurrence: two entities are joined when a chain
of each sits in the same episode. That falls out of `chains` with a self-join, so
there is no edge table to maintain — a merge or split changes `chains.entity_id`
and the edges follow.

**Weight the edges by PMI**, not by how often the pair co-occurs:

```sql
WITH docs AS (
  SELECT DISTINCT episode_id, entity_id
  FROM chains WHERE tenant_id = $t AND entity_id IS NOT NULL
),
total AS (SELECT count(DISTINCT episode_id)::float AS n FROM docs),
freq  AS (SELECT entity_id, count(*)::float AS df FROM docs GROUP BY 1),
pairs AS (
  SELECT a.entity_id AS ea, b.entity_id AS eb, count(*)::float AS together
  FROM docs a JOIN docs b USING (episode_id)
  WHERE a.entity_id < b.entity_id
  GROUP BY 1, 2
)
SELECT p.ea, p.eb, ln(p.together * t.n / (fa.df * fb.df)) AS weight
FROM pairs p CROSS JOIN total t
JOIN freq fa ON fa.entity_id = p.ea
JOIN freq fb ON fb.entity_id = p.eb
WHERE p.together * t.n > fa.df * fb.df;   -- PMI > 0: more than chance
```

PMI asks whether a pair co-occurs *more than its two frequencies alone predict*.
An entity that appears everywhere co-occurs with everything — but only as much as
chance, so all of its edges collapse to zero and the walk stops routing through
it. The `WHERE` prunes those pairs entirely.

#### Measured

`demos/eval_expansion.py`. A synthetic corpus is required here: relatedness has no
gold in ECB+, where every document in a topic covers one event so everything is
related to everything. Entities are assigned to **projects**, two entities are
related exactly when they share one, and the generator plants the trap on
purpose — *crosstalk* episodes name two projects at once (a status meeting), which
is genuine co-occurrence and no real relationship, and *noise* entities appear
everywhere to build hubs.

400 episodes, 43 entities, 8 projects. The generator also plants one **bridged
pair** per project: two members that share a project but are never allowed in the
same episode, so no direct edge between them can exist and reaching one from the
other requires walking through a third. Scored at k = the seed's project size, so
1.000 is perfect:

| scheme | edges | mean degree | P@fam | recall | noise@10 | bridge@fam |
|---|---|---|---|---|---|---|
| raw counts | 515 | 24.0 | 0.786 | 0.786 | 0.674 | 0.000 |
| binary | 515 | 24.0 | **0.000** | 0.000 | 0.769 | 0.000 |
| count ≥ 2 | 346 | 16.1 | 0.786 | 0.786 | 0.654 | 0.000 |
| 1/(episode size) | 515 | 24.0 | 0.831 | 0.831 | 0.569 | 0.000 |
| PMI | 218 | 10.1 | 0.931 | 0.931 | 0.523 | 0.500 |
| **PMI × 1/(episode size)** | **218** | **10.1** | **0.983** | **0.983** | 0.457 | **0.875** |

* **PMI is most of the win, and a 58% smaller graph.** Raw counts reach 0.786 and
  never bridge. Multiplying by 1/(episode size) — a pair seen in a passage naming
  three entities is better evidence than the same pair in a status meeting naming
  twenty — takes P@fam to 0.983 and bridging to 0.875.
* **Binary weights are catastrophic: P@fam 0.000.** With every edge equal, PPR
  walks to the highest-degree nodes, which are exactly the entities that mean
  nothing. The danger is not density, it is *undifferentiated* density.
* **Expansion does not leak into the wrong project.** The generator plants
  crosstalk episodes naming two projects at once; PMI declines to walk those edges.

### How far to walk

Restart probability, PMI × 1/(size) edges:

| alpha | P@fam | bridge@fam | bridge@10 |
|---|---|---|---|
| 1-hop, no walk | 0.862 | **0.000** | **0.000** |
| 0.05 | 0.921 | 0.500 | 1.000 |
| **0.15 – 0.30** | **0.983** | **0.875** | 1.000 |
| 0.50 | 0.931 | 0.500 | 1.000 |
| 0.85 | 0.862 | 0.000 | 1.000 |

**This is what justifies PPR over listing a seed's neighbours.** A bridged partner
has no edge to the seed, so 1-hop scores 0.000 at any depth — it cannot be found
by expansion that only looks one step out. The walk finds 87.5% of them in the
top few slots.

Both ends fail differently: too little restart (0.05) diffuses past the project,
too much (0.85) degenerates to 1-hop. Use **alpha ≈ 0.2**.

`noise@10` near 0.5 is not the walk going astray: projects hold only 3–4 other
members, so past the family there is no correct answer left and the remaining
slots fill with whatever is nearest. The fix is a score cutoff rather than a fixed
k — stop when PPR mass falls off instead of always returning ten.
