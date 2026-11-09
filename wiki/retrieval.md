# Retrieval (query path)

Lives in `server/public/src/query/` (`retrieve.py` for seeding, the walk and passages; `answer.py` for the stance and rendering) and is called by the Slack ingress and by `python -m src.query.retrieve`. The SQL reference versions of each step, with the measurements behind them, are in `old_server/entity_linking/DATAFLOW.md` § Query. Enters the wall as `app_connector` (a declared workspace, the Slack worker) or `app_ingest` (a claimed tenant, the CLI); `old_server/entity_linking/ask.py` is the superseded CLI copy.

## Flow

```
question
  1. embed              text-embedding-3-large, 1536 dims               ask.embed_texts
  2. surface hits       HNSW cosine neighbours (k=40) ∪ word-bounded exact mentions of name surfaces, reciprocal-rank fused   retrieve.seed_entities / lexical_hits / fuse  (is_name = false never seeds)
  3. bridge             chain_surfaces → chains → entity_id (non-NULL)
  4. seed weights       1/(rank+1) over the first `seeds=8` distinct entities   (rank, not raw cosine: cosines are not calibrated across queries)
  5. graph              chains self-join per episode → PMI edges, min co-occurrence 2      ask.build_graph
  6. PPR                sparse power iteration, alpha=0.50, 30 iterations   ask.ppr   (14x the old loop, identical scores)
  6b. cutoff            keep seeds + walked entities >= 0.2 x best non-seed  ask.prune (--cutoff; 0 disables)
  7. passages          episodes of ranked entities, expired_at IS NULL, scored by MEAN of entity scores; ties → more ranked entities, then sum   retrieve.rank_episodes
  7b. text channel     BM25 over the tenant's live episode texts (identifiers split into parts: keys_settle → keys, settle), top `text` episode ids, reciprocal-rank fused with step 7; off when text=0   retrieve.text_hits / bm25 / fuse_ranks
  7c. take             at most 2 per source document; top `passages` (8)   retrieve.take_passages
  8. stance + answer    gpt-4.1, answer.ANSWER_PROMPT: `answer` | `reframe` (why, better question, evidence, then the answer) | `clarify` (which entity); --no-reframe pins it   answer.answer / answer.render / answer.render_slack
```

`--explain` prints seeds, edge count, entities reached, contested names and the chosen passages. `--no-ppr` is the seeds-only baseline. `search` (same knobs, no answer model) prints the passages as `format_passages` renders them: `[n] title (score)` then the text, then any contested names. That string is what a search tool hands an agent; `search_passages()` is the function behind it. `seed_entities` also returns the matched surfaces that fan out to several entities, with display names, which is what lets the answer step ask "which one?" (ADR slack 15, 16).

## Knobs and the evidence behind each

| Knob | Value | Evidence | Where measured |
|---|---|---|---|
| Seed weight | rank-based | cosine magnitudes are not comparable across queries | `seed_entities` docstring |
| Seeding weight (DB version) | `sum(score) / sqrt(n_surfaces)` | MRR 0.689 → 0.749, R@1 0.546 → 0.639 on ECB+ | DATAFLOW § seeding weight |
| Rejected seeding weights | `idf(n_episodes)` −0.017 MRR; `1/n_entities` −0.097 | junk surfaces are rare, junk entities are singletons | DATAFLOW; ADR tables 29 |
| Edge weight | PMI, `WHERE together·n > df_a·df_b` | P@family 0.983 vs 0.786 raw counts vs **0.000** binary; 58 % fewer edges | `demos/eval_expansion.py`, ADR tables 24 |
| Edge weight variant | PMI × 1/(episode size) | best on the synthetic rig; not in `ask.py` | DATAFLOW § edge weights |
| `MIN_CO_OCCURRENCE` | 2 | on the 144-note vault 94.8 % of pairs co-occur once and PMI then collapses to its maximum, so the graph was effectively unweighted | `ask.py` comment |
| `ALPHA` | **0.50** in `ask.py`; **≈0.2** recommended by the synthetic rig | vault: alpha 0.15 left the top 20 entities with 18 % of mass across 3346 of 3354 entities; synthetic: 0.15–0.30 best, 0.50 loses half the bridged pairs | `ask.py` comment vs DATAFLOW § how far to walk. **Corpus-dependent; re-measure per deployment.** |
| Passage score | mean of entity scores in the episode | `max` ties every episode holding the top entity; `sum` rewards long notes (answering notes ranked 25th sum, 10th max, 7th mean) | `collect_episodes` comment |
| `CUTOFF_RATIO` | **0.2** with `ALPHA = 0.5`; use 0.3 with alpha ≈ 0.2 | rig: P 0.959 / noise 0.03 at 3.6 entities vs P 0.343 / noise 0.40 for fixed k = 10; at alpha 0.5 a 0.3 ratio loses half the bridged pairs | [experiments.md](experiments.md) E13 |
| `k`, `seeds`, `passages` | 40, 8, 8 | defaults, untuned | `ask.main` |
| `text` (`TEXT_CANDIDATES`) | **10** (ADR tables 31); `--text 0` = graph only | BM25 candidates fused at the passage level; what reaches docstring continuations that name no symbol and notes whose entities the question does not mention | `scripts/eval_agent_tokens.py` |

## What the graph is for, and what dense is for

Two evals on ECB+ (277 passages, 12 topics, real extraction and linking):

- Single-passage questions: dense over passages MRR **0.815**; seeds + PPR + rerank 0.776; without rerank 0.273. Rerank is stage 8 and is not implemented in `ask.py`.
- Footprint questions (every document mentioning an entity): graph + PPR total recall **0.999** vs seeds-only **0.633**; alias-only documents R@20 0.769 graph vs 0.665 dense.

Precision retrieval of one passage is dense's job. Coverage of everything about an entity is the graph's, and PPR is what makes bridged (never co-mentioned) entities reachable: 1-hop scores 0.000 on bridge recall at any depth.

## Open decision (ADR tables 24 note, DATAFLOW § one decision left open)

Passages from **seed** entities, **walked** entities, or the **edges traversed** (needs an `entity_edge_episodes` table that does not exist). `ask.py` uses walked entities.

## Ambiguity at query time

A surface claimed by several entities splits its mass. Past a fan-out threshold, drop it or ask a clarifying question built from candidates' `entity_display_name` (ADR slack 15). Not implemented.

## Embedding backfill

`ask.py embed --tenant <uuid>` fills `surfaces.embedding` where NULL and `is_name` is not false, 256 per OpenAI call, and stamps `embed_model`. Stage 5 leaves the column NULL on purpose (different model, different budget). Until backfilled, only the trigram half of seeding can work — and `ask.py` currently uses only the vector half, so an unembedded tenant returns "no surfaces matched".

## Legacy retrieval library

`old_server/retrieval/` implements seven methods over Neo4j (PPR, approximate push PPR, temporal, multihop, beam, DAG, treehop) with a reranker and a security-audited config. None of it reads Postgres. Reuse the algorithms (`ppr_approx.py` is a push-based PPR with sparse matrices; `rerank.py` has embedding and cross-encoder modes), not the drivers. See [legacy.md](legacy.md).
