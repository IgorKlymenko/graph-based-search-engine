# Measurements

Every number the code and ADRs rely on, with the rig it came from. Rigs:
**synthetic** = generated corpus with planted contested initials and bridged
pairs (92–400 episodes); **ECB+** = 277 passages, 12 topics, real extraction
and linking; **vault** = the author's 144-note Obsidian vault (253 episodes,
3354 entities). Absolute numbers drift ~0.03 between runs because stage 2 is
non-deterministic; orderings are stable.

## Linking

| What | Result | Rig | Source |
|---|---|---|---|
| Name vector for cross-episode identity | AUC **0.936** | ECB+ | `main._name_vectors` |
| Context vector, uncentered | 0.745 (ECB+), 0.898 (synthetic) | both | DATAFLOW § 4c |
| Context vector, episode-centered | 0.505 (chance) — a measurement bug, not a result | synthetic | ADR tables "what did not work" |
| Same-string pairs (2466): name / string / both / + `context_gated` | 0.414 / 0.500 / 0.500 / **0.690** AUC | ECB+ | DATAFLOW § 4c |
| `context_gated` end to end | 0.774 vs 0.775 CoNLL (no change); fitted weight +0.13; dev AUC 0.9752 → 0.9754; under prior weights 0.716 → 0.742 | ECB+ | DATAFLOW § 4c |
| Same-string pairs that are coreferent | 91.8 % | ECB+ | DATAFLOW § 4c |
| Alias precision with the ambiguous-surface veto | 73 % → **100 %** | synthetic | DATAFLOW § 4a |
| Cannot-link constraints in clustering | **0.774 vs 0.945** CoNLL; 45 clusters for 26 gold; 17 contradicted surface pairs | synthetic | `pairing.cluster` |
| `choose_thresholds()` vs hand-set cut | 0.634 vs **0.775** CoNLL | ECB+ | `main.ingest` docstring |
| Pairwise reviewer on the review band | 81 % | — | `main.resolve_against_graph` docstring |
| End to end: 92 episodes, 26 gold, 236 reviews | MUC 0.977 · B³ 0.941 · CEAFe 0.892 · **CoNLL 0.937**; 2 entities fused on `S. Chen`, `J. Park` | synthetic | DATAFLOW § measured |
| ANN blocking at 444 chains, k=50 | 98,286 → 14,376 pairs (−85 %), 99.9 % recall, 5.6× faster; no LLM saving | synthetic | DATAFLOW § 4b |
| ANN at 55 chains, k=50 | 1485 → 1364 (−8 %); 2 pairs differed between runs unseeded | vault-sized | `pairing.ANN_MIN_FACTOR` comment |
| Entity id churn under smallest-chain naming | one joining chain renames ~1 in n+1; one added document orphaned an entity on first try | vault | `store.match_entities` |
| Unanchored contested chains merging with each other | produced a third `S. Chen` beside Sarah's and Steven's | 10-doc corpus | `pairing.unresolved` |
| Vault run (from `ADRs/brainstorming.md` log) | 5772 chains, 4384 names, 220,013 pairs, 10,274 reviewed, 6888 merged → 3439 entities; 138 ambiguous placed, 40 withheld | vault | log |
| Extraction throughput | 253 episodes ≈ 10 min | vault | ADR slack 11 |

| Code short-name collisions, seed-only chains, prior scorer | false merges 303 → **1** (live source), 2039 → **2** (`entity_linking`), 2564 → **12** (httpx), 2180 → **12** (fastapi) with `Style` defaults; 0 same-key pairs missed; review-band pairs 2.5–5× | code rig, four repos | ADR connectors 11, `scripts/eval_code_collisions.py` |
| Prevalence of colliding short names | 8.2–25.2 % of names; 6.5–21.9 % of rendered in-repo references | six repos | [experiments.md](experiments.md) |

## Seeding

| What | Result | Rig | Source |
|---|---|---|---|
| `sum(score)/sqrt(n_surfaces)` vs plain sum | MRR 0.689 → **0.749**, R@1 0.546 → 0.639 (second run 0.653 → 0.718) | ECB+ | DATAFLOW § seeding |
| `idf(n_episodes)` | −0.017 MRR | ECB+ | same |
| `1/n_entities` | −0.097 MRR | ECB+ | same |
| Junk surfaces appearing in exactly one episode | 75.7 % | ECB+ | same |
| Double-joining `chain_surfaces` for breadth | MRR 0.718 → 0.658 (the whole gain) | ECB+ | DATAFLOW SQL comment |
| Top-1 seeds landing on junk entities | ≈ 25 % under every weighting | ECB+ | ADR tables open items |

## Retrieval

Single-passage questions (gold = the passage the question came from), ECB+:

| System | R@1 | R@10 | MRR | cand. recall |
|---|---|---|---|---|
| dense over passages | 0.725 | 0.992 | **0.815** | 1.000 |
| seeds only | 0.108 | 0.550 | 0.250 | 0.733 |
| seeds + PPR | 0.108 | 0.600 | 0.273 | 1.000 |
| seeds + PPR + rerank | 0.683 | 0.950 | 0.776 | 1.000 |

Footprint questions (gold = every document mentioning the entity), ECB+:

| System | R@5 | R@20 | total recall |
|---|---|---|---|
| dense over passages | 0.423 | 0.745 | 1.000 |
| graph seeds | 0.451 | 0.629 | 0.633 |
| graph + PPR, raw edges, α 0.5 | 0.449 | 0.787 | 0.999 |
| graph + PPR, PMI edges, α 0.2 | 0.440 | **0.795** | 0.999 |

Alias-only documents R@20: dense 0.665 vs tuned graph **0.769**.

## Agents and tokens (28 questions about this project, 2026-09-19)

| Strategy | Tokens per question | Answer in front of the agent | Graded correct / partial / wrong |
|---|---|---|---|
| grep three terms, open top 3 files | 16,607 | 22/28 | 16 / 8 / 4 |
| wiki README + routed page | 5,565 | 12/28 | 13 / 8 / 7 |
| graph search, 8 passages, no text channel | 1,785 | 12/28 | 8 / 10 / 10 |
| **graph search + BM25 text channel, 8 passages** | **1,800** | **27/28** | **20 / 8 / 0** |
| oracle: the right files | 3,996 | 28/28 | 24 / 4 / 0 |

Source: `server/public/scripts/eval_agent_tokens.py`, [experiments.md](experiments.md) § Agents and tokens. Retrieval 0.24 s per question fused.

Real agent (gpt-4.1-mini with tools, 116 questions of which 60 model-written, graded by gpt-4.1; `scripts/eval_agent_loop.py`):

| Kit | Billed tokens per question (mean / median) | Correct | Wrong |
|---|---|---|---|
| glob, grep, read | 10,416 / 8,607 | 48 % | 22 |
| plus search | 12,857 / 12,000 | 57 % | 13 |
| search, read | 14,003 / 8,990 | 64 % | 8 |
| search, read, told the passages are evidence | 11,451 / **3,333** | **69 %** | **4** |

The search raises correctness and cuts wrong answers. Told to trust the passages, the agent answers from one search in 59 of 116 runs; the typical run is 2.6× cheaper than grep-and-read and a correct answer costs 23 % fewer tokens, while the mean is level because the runs where the search misses are long.

## Edge weights (synthetic, 400 episodes, 43 entities, 8 projects, k = project size)

| Scheme | edges | mean degree | P@fam | noise@10 | bridge@fam |
|---|---|---|---|---|---|
| raw counts | 515 | 24.0 | 0.786 | 0.674 | 0.000 |
| binary | 515 | 24.0 | **0.000** | 0.769 | 0.000 |
| count ≥ 2 | 346 | 16.1 | 0.786 | 0.654 | 0.000 |
| 1/(episode size) | 515 | 24.0 | 0.831 | 0.569 | 0.000 |
| PMI | 218 | 10.1 | 0.931 | 0.523 | 0.500 |
| **PMI × 1/(episode size)** | 218 | 10.1 | **0.983** | 0.457 | **0.875** |

Restart probability on the same rig: 1-hop 0.862 / bridge 0.000; α 0.05 → 0.921; **α 0.15–0.30 → 0.983, bridge 0.875**; α 0.50 → 0.931; α 0.85 → 0.862.

Vault, by contrast (`ask.py` comments): α 0.15 left the top 20 entities with 18 % of the mass spread across 3346 of 3354 entities; α 0.50 → 53 %. 94.8 % of entity pairs co-occur exactly once, so 75k of 79k PMI edges had the same weight; hence `MIN_CO_OCCURRENCE = 2`. Passage scoring for "who do we sell to?": answering notes ranked 25th (sum), 10th (max), 7th (mean); under max the top 8 episodes had exactly one distinct score.

## Database

38 tenant-isolation tests (`old_server/tests/test_db_tenant_isolation.py`). Measured facts: `statement_timeout` reads `0` inside a `SET ROLE` session from a `NOLOGIN` role and `30s` from the login role; `permission denied for table files` without `SET ROLE`; `DELETE FROM tenants` succeeded with zero rows before privileges were revoked; `auth.uid()`-gated walls returned zero rows for the server.

## Mini-experiments on the proposals

Run 2026-09-17 on the synthetic rigs and the real vault; all numbers in [experiments.md](experiments.md). Headlines: sparse PPR 14× faster with identical output; incremental linking scores 23 % of pairs for 11 % new episodes with an identical partition; prior thresholds merge at 0.906 precision vs 0.975 fitted; nomic beats OpenAI embeddings on hard pairs (0.956 vs 0.898 AUC); 3.7 % of the vault's surfaces carry markup on the raw path; 59 % of identifying spans are descriptions; a `score ≥ 0.3 × top` cutoff beats fixed k = 10 (P 0.959 vs 0.343); the coref prompt is under the prompt-caching minimum. Junk: ~60 % of stored surfaces are not entities by either of two LLM judges (kappa 0.55 between them); no cheap rule or prompt addendum fixes it; removing junk hubs cuts 70 % of edges.

## Unverified but load-bearing

Slack rate limits for non-Marketplace apps ("reportedly ~1 request/minute"); Marketplace approval effects; transaction-pooler behaviour of `SET LOCAL`; `context_gated` on real contested enterprise data; cross-cloud blob fetch latency.
