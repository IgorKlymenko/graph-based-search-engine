# Experiments — mini-versions of the proposals (2026-09-17)

Small, reproducible runs that test whether a proposal in [PROPOSALS.md](PROPOSALS.md)
is worth doing. Numbers only; the verdicts live in the proposals table. Scripts
are in `old_server/entity_linking/demos/` and run from `old_server/` with
`uv run python entity_linking/demos/<script>`.

**What was available.** No vault store, no ECB+ corpus and no database survived
locally, and Docker was down. So: the synthetic gold corpora already in
`demos/` (offline, nomic cached), the author's real Obsidian vault at
`/Applications/reink` (154 documents, counts only, nothing left the machine),
and a few hundred OpenAI embedding calls. Every synthetic number carries the
usual caveat: the corpus was written alongside the algorithm. Runtimes are on
the author's laptop with the `mps` device.

## Reproduction of the existing rig

`eval_expansion.py` (400 episodes, 43 entities) reproduced DATAFLOW exactly:
PMI × 1/size P@family 0.983, bridge 0.875; binary 0.000; alpha 0.15–0.30 best.
Everything below that uses the rig starts from a known state.

## E9 — prompt caching (`tiktoken`, `o200k_base`)

| Prompt | Tokens |
|---|---|
| `CHAIN_SYSTEM_PROMPT` | 573 |
| `CHAIN_SCHEMA` (JSON) | 187 |
| `MERGE_REVIEW_PROMPT` | 259 |
| `ENTITY_CHOICE_PROMPT` | 280 |

The coref call's static prefix is ~760 tokens. OpenAI's documented minimum for
automatic prompt caching is 1,024 tokens at the time of writing (verify against
current docs), so **no call is cached today**. Padding the static prefix past the
threshold (few-shot examples) would make the system prompt, which is most of
each call's input, cache-eligible for every episode.

## E6 — sparse PPR (`eval_sparse_ppr.py`)

Synthetic graphs with a skewed degree distribution; identical seeds; 30 iterations, alpha 0.5.

| Graph | `ask.ppr` (pure Python) | `scipy.sparse` | Speed-up | Top-20 overlap | max abs diff |
|---|---|---|---|---|---|
| vault-sized, 3,400 nodes / 79k edges | 1.61 s | 0.115 s | 14× | 20/20 | 1.7e-18 |
| 10× workspace, 10k nodes / 400k edges | 7.37 s | 0.554 s | 13× | 20/20 | 1.7e-18 |

Pair counting for `build_graph` at vault scale (253 episodes × 23 entities): 34 ms. The Python walk, not the counting, is the per-query cost; the SQL fetch of all chains was not measurable without a database.

## E5 — store format (`eval_store_format.py`)

Synthetic stores at vault size (144 files, 253 episodes, 7,073 candidates, 5,772 chains, 10,274 reviews) and 10×.

| Scale | JSON save | JSON load | size | saves per run | checkpoint total | SQLite full write | SQLite 25-episode delta |
|---|---|---|---|---|---|---|---|
| 1× | 0.13 s | 0.04 s | 4.7 MB | ~15 | 2.0 s | 0.11 s | 11 ms |
| 10× | 1.69 s | 0.49 s | 47.9 MB | ~106 | **179 s** | 0.79 s | 80 ms |

Checkpointing is free at vault scale and about three minutes per run at workspace scale.

## E2 — vector cache (`eval_linking_mini.py`)

Synthetic XL corpus: 92 episodes, 115 gold chains, 42 distinct surfaces.

| Step | Cost |
|---|---|
| nomic model load (per process) | 14.5 s |
| name vectors, 42 strings | 3.79 s (first batch, includes device warm-up) |
| episode forward passes | 63 ms per episode |
| `.npz` save / load of all name vectors | 13 ms / 11 ms |

## E1 — identity embedding, nomic vs OpenAI (`eval_linking_mini.py`)

6,532 cross-episode chain pairs, 236 positive, gold labels. AUC of cosine between head-surface vectors.

| Vectors | all pairs | different head strings (n=6,370) | hard: 0.5 ≤ string sim < 0.999 (n=334) |
|---|---|---|---|
| nomic-embed-text-v1.5 (768) | 0.999 | 0.998 | **0.956** |
| text-embedding-3-large @1536 | 0.998 | 0.996 | 0.898 |
| text-embedding-3-small @1536 | 0.998 | 0.995 | 0.868 |
| `string_similarity` alone (no embedding) | 0.999 | 0.997 | 0.930 |

On this corpus the easy pairs are separated by any signal, string similarity included. On the hard subset nomic keeps a 0.06 AUC lead over the OpenAI model and is the only vector that beats plain string similarity. Embedding cost: 2.7 s (large) / 0.8 s (small) for 42 strings, network included.

## E10 — review band, prior vs fitted scorer (`eval_linking_mini.py`)

Same corpus, gold chains, `candidate_pairs` blocking (5,863 pairs), thresholds `(0.30, 0.50)` for the prior.

| Scorer | review band | merge precision | split precision |
|---|---|---|---|
| prior weights, hand thresholds | 0.7 % (41 pairs) | **0.906** | 1.000 |
| fitted, 5-fold, thresholds at 0.98 precision on train | 2.3 % ± 0.4 % | 0.975 | 0.989 |

Fitted weights (last fold): `string_similarity` 10.8, `name_cosine` 7.3, `alias_hit` 1.8, `cooccurrence` −1.35, `context_gated` 0 (never fires: no context vectors here). On the vault the review band was 4.7 % (10,274 of 220,013 pairs), so band width is corpus-dependent; what is stable is that the prior thresholds trade ~9 % wrong automatic merges for a narrow band.

## E3 — incremental linking (`eval_linking_mini.py`)

Last 10 of 92 episodes treated as new. Old episodes linked first with evidence from old chains only; then only pairs touching a new chain are scored, with evidence rebuilt over everything, and union-find is seeded with the old merges.

| | Full re-link | Incremental |
|---|---|---|
| pairs scored | 5,863 | 1,320 (23 %) |
| clusters | 30 | 30 |
| pairwise agreement with full re-link | — | 1.0000 |
| old-pair verdicts changed by new evidence | — | 0 |

The delta fraction is roughly twice the new-episode fraction (a pair touches two chains). On real data a new alias can flip an old pair; re-score old pairs whose surfaces gained evidence (cheap to detect) rather than assuming zero.

## B5 — markup in surfaces, real vault (`eval_vault_markup.py --root /Applications/reink`)

154 documents, spaCy only.

| Path | episodes | spans | spans with markup | distinct identifying surfaces with markup | parse cost |
|---|---|---|---|---|---|
| raw text (what `src/inject` does today) | 311 | 11,051 | 245 (2.2 %) | **213 of 5,753 (3.7 %)**, e.g. `A==`, `TLS==`, `[[Day` | 44 ms/episode |
| `strip_markup` (old copy) | 301 | 10,853 | 10 (0.1 %) | 9 of 5,616 (0.2 %), all fenced code (```` ``` ````) | 34 ms/episode |

`strip_markup` does not remove fenced code blocks; those nine are a gap in it, recorded as B11.

## Junk share, real vault (same script)

Identifying spans by derived type, raw path: `proper_name` 3,400 · `nominal` 3,698 · `definite_description` 1,344 · `pronoun` 136 (mis-typed by the parser). **About 59 % of the spans that can become surfaces are descriptions, not names.** This is before the coref call, so it is an upper bound on what reaches `surfaces`, and it is the pool the "the freeze" seeds come from.

## E4 — sync vs upload delta (same script, on a temp copy)

First sync of 154 files: 0.08 s, 302 episodes. Second sync: 0 changed. After editing one paragraph in the largest note (51 paragraphs): 1 file changed, 1 episode added, 1 retired. `store.upload` would still upsert all 154 files and 302 episodes, plus every chain and surface row, for those 2 changed episodes.

## E13 — score cutoff vs fixed k (`eval_score_cutoff.py`)

Expansion rig, PMI × 1/size edges, one seed per non-noise entity. `P` is precision over what was returned, `R` recall of the seed's project, `noise` share of returned entities that are noise hubs, `bridge` whether the never-co-mentioned partner was returned, `avg n` entities returned.

Alpha 0.2:

| Rule | P | R | noise | bridge | avg n |
|---|---|---|---|---|---|
| fixed k = 10 | 0.343 | 1.000 | 0.400 | 1.000 | 10.0 |
| fixed k = 5 | 0.686 | 1.000 | 0.229 | 1.000 | 5.0 |
| oracle k = family size | 1.000 | 1.000 | 0.000 | 1.000 | 3.4 |
| score ≥ 0.5 × top | 1.000 | 0.919 | 0.000 | 0.438 | 3.2 |
| **score ≥ 0.3 × top** | **0.959** | **1.000** | **0.034** | **1.000** | 3.6 |
| score ≥ 0.2 × top | 0.804 | 1.000 | 0.158 | 1.000 | 4.6 |
| largest relative drop (knee) | 0.993 | 0.974 | 0.000 | 0.812 | 3.4 |
| cumulative mass 60 % | 0.715 | 1.000 | 0.230 | 1.000 | 5.3 |

Alpha 0.5: the ratio rules lose the bridged partners (0.3 × top → bridge 0.438; 0.5 × top → 0.000) because a higher restart keeps mass on direct neighbours; `score ≥ 0.2 × top` still reaches P 0.887, R 1.0, bridge 1.0 at 4.0 returned. The cutoff and the restart probability have to be chosen together.

## Junk filtering (second session, same day)

Four runs on the question "how do we keep descriptions out of the graph". Scripts: `eval_junk_filter.py`, `eval_junk_judge_agreement.py`, `eval_junk_rig.py`, `eval_junk_prompt.py`. Judge verdicts are cached under `old_server/artifacts/` (gitignored) so re-runs are free and the numbers reproduce.

### J1 — cheap rules vs an LLM judge (`eval_junk_filter.py --root <vault>`)

5,620 distinct identifying surfaces on the real vault (2,037 `proper_name`, 2,596 `nominal`, 962 `definite_description`, 25 mis-typed `pronoun`). A stratified sample of 410 was judged by `gpt-4.1-mini` with the WHAT-COUNTS rules from `prompts.SYSTEM_PROMPT`, each surface shown in its own context.

Judge keep rate by parser type: proper_name **0.49**, nominal 0.32, definite_description 0.38, pronoun 0.20. Half of what the parser calls a proper name is not an entity (capitalised words, non-English fragments, "87 days", "ONLY").

| Rule | precision | recall | F1 | keeps (whole vault) |
|---|---|---|---|---|
| R0 `is_identifying` (today) | 0.385 | 1.000 | 0.556 | 100 % |
| R1 proper_name only | 0.492 | 0.405 | 0.444 | 36 % |
| R2 proper or capitalised | 0.513 | 0.500 | 0.506 | 40 % |
| R3 proper or in 2+ docs | 0.417 | 0.494 | 0.452 | 46 % |
| R4 not a generic head noun | 0.417 | 0.905 | 0.571 | 86 % |
| R5 R2 and R4 | 0.542 | 0.487 | 0.513 | 37 % |
| R6 drop definite descriptions | 0.389 | 0.690 | 0.498 | 83 % |
| R7 R6 and R4 | 0.428 | 0.658 | 0.519 | 72 % |

Against this judge no cheap rule moves F1 by more than 0.015.

### J4 — is the judge trustworthy? (`eval_junk_judge_agreement.py`)

Same 410 surfaces judged by `gpt-4.1`. Agreement **0.802, Cohen's kappa 0.552**. Keep rates: mini 0.385, gpt-4.1 **0.241**. By type (mini / 4.1 / agreement): proper_name 0.49 / 0.50 / 0.88; nominal 0.32 / 0.18 / 0.82; definite_description 0.38 / **0.06** / 0.68. Both judges keep 21.5 % of the sample and both drop 58.8 %. The stronger judge is stricter on descriptions and the two disagree most exactly there.

Rules re-scored against `gpt-4.1`: today's filter precision **0.241** (F1 0.389); R5 precision 0.500, recall 0.717, **F1 0.589**; R2 0.577; R1 0.568; R6 recall 0.919 at precision 0.325. So the verdict on cheap rules depends on which judge is the reference; what does not depend on it is that roughly **60 % of stored surfaces are junk by either judge**.

### J2 — what filtering buys the walk (`eval_junk_rig.py`)

Expansion rig, PMI × 1/size. "Perfect" removes the planted noise entities; "type filter" removes every entity starting with "the …", which includes seven legitimate project members (`the invoice service`, `the reranker`, …).

| Scenario | entities | edges | alpha 0.2, cutoff 0.3: P / recall / noise / bridge | alpha 0.5, cutoff 0.3: P / recall / bridge |
|---|---|---|---|---|
| no filter | 43 | 218 | 0.959 / 1.000 / 0.034 / 1.000 | 0.968 / 0.921 / 0.438 |
| perfect junk filter | 35 | **66** | **1.000 / 1.000 / 0.000 / 1.000** | 0.993 / 1.000 / **1.000** |
| type filter (drop "the …") | 29 | 52 | 0.889 / **0.750** / 0.045 / 1.000 | 0.952 / 0.613 / 0.250 |

Noise hubs carry 70 % of the edges. Removing them makes the walk insensitive to alpha (bridge reach 1.0 at 0.5 where it was 0.438); removing descriptions by type costs a quarter of recall and, at alpha 0.5, most of the bridges.

### J3 — a stricter coref prompt (`eval_junk_prompt.py --root <vault>`)

40 vault episodes, the current `CHAIN_SYSTEM_PROMPT` vs the same prompt plus a "look-up test" paragraph telling the model to omit chains with no name or established term. Surfaces from both runs judged by `gpt-4.1-mini`.

| Variant | chains | per episode | distinct surfaces | judge keep | recall vs judged union | description-only chains |
|---|---|---|---|---|---|---|
| current | 977 | 24.4 | 923 | 0.341 | 0.981 | 278 |
| strict | 946 | 23.6 | 902 | 0.345 | 0.969 | 257 |

The instruction is almost ignored under the ids-only protocol: 2 % fewer surfaces, no precision gain, and 10 judge-approved surfaces lost. The parser pre-marks every span and the model is asked to group them; "leave it out" is not a move it makes.

### J5 — hand labels, judge prompts and the cascade (`eval_junk_gold.py --root <vault>`)

All 410 sampled surfaces labelled by Claude under one written policy (kept with the labels in `old_server/artifacts/junk_gold_labels.json`, local-only since the surfaces are note fragments): **69 keep, 341 drop (16.8 %)**. Gold keep rate by parser type: proper_name 0.31, nominal 0.09, definite_description 0.11.

| Judge / rule | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| `is_identifying` (today) | 0.168 | 1.000 | 0.288 | 0.168 |
| best cheap rule (R5) | 0.331 | 0.681 | 0.445 | 0.715 |
| gpt-4.1-mini, v1 / v2 / **v3** prompt | 0.399 / 0.415 / 0.357 | 0.913 / 0.986 / **1.000** | 0.555 / 0.584 / 0.526 | 0.754 / 0.763 / 0.710 |
| gpt-4.1, v1 / v2 / **v3** prompt | 0.616 / 0.663 / **0.684** | 0.884 / 0.897 / **0.970** | 0.726 / 0.762 / **0.802** | 0.888 / 0.906 / 0.919 |
| **cascade v3: mini then gpt-4.1** | **0.718** | 0.968 | **0.824** | **0.933** |

v2 is the policy written as a prompt; v3 adds "normalise determiners and markup before judging" and "capitalisation is not a name". The cascade works because mini v3 has recall 1.0 on gold: it drops 56 % of surfaces outright and 44 % reach the strong model, so the cost is about 60 % of gpt-4.1 alone. Remaining gpt-4.1 v3 false keeps are compound or broken spans (`Preindexed <-> Raw`, `Distinction #1 - CSV`) and misses are `Day 3`, `Spanish`. gpt-4.1 v3 left 13 of 410 unanswered even after single-item retries; the production judge treats those as names.

Shipped as ADR tables 30: `utils/judge.py` with the v3 prompt and the `("gpt-4.1-mini", "gpt-4.1")` cascade.

## Agents and tokens — is the search cheaper than reading files? (`server/public/scripts/eval_agent_tokens.py`, 2026-09-19)

**Question.** A coding or research agent needs a fact about this project. Does one call to the graph search put the answer in front of it for fewer tokens than the ways it has today?

**Rig.** 28 questions in `scripts/agent_questions.json` (16 about the code, 8 about the vault, 2 that need both, 2 *stale*: answered only by files the index had not been re-synced with). Each carries gold strings that must all appear in what the agent read (*sufficient*: no second round trip needed), a gold answer, and the files an oracle would open; every gold string was checked on disk and in the index before the run. Strategies, all against the vault-plus-code tenant and the same corpus on disk:

| Strategy | What the agent reads |
|---|---|
| `read_gold` | exactly the right files (oracle; the floor for any file-reading agent) |
| `grep_read` | grep output for the question's three most specific terms (identifiers first), then the top 3 files whole |
| `wiki` | `wiki/README.md`, then the page(s) gpt-4.1-mini routes to |
| `search_k` | one search call, top k passages, graph only (`--text 0`) |
| `hybrid_k` | the same with the BM25 text channel fused in (`--text 10`) |
| `*_open` | the search, then the whole file behind the top passage |

Tokens are o200k counts of the material plus 40 per tool call. Retrieval time is per question on the 8 GB laptop.

**Index before the re-sync** (44 code files, 293 code episodes; the working tree's `src/query`, `src/connectors` and `scripts` not yet indexed), string sufficiency:

| condition | mean tok | median | p90 | sufficient | tok per sufficient |
|---|---|---|---|---|---|
| read_gold | 3,974 | 3,804 | 8,360 | 28/28 | 3,974 |
| grep_read | 16,388 | 17,716 | 22,233 | 22/28 | 20,858 |
| wiki | 5,525 | 5,020 | 8,660 | 12/28 | 12,891 |
| search_3 | 702 | 664 | 971 | 10/28 | 1,965 |
| search_5 | 1,167 | 1,154 | 1,401 | 14/28 | 2,334 |
| search_8 | 1,856 | 1,816 | 2,262 | 14/28 | 3,711 |
| hybrid_5 | 1,191 | 1,174 | 1,512 | 20/28 | 1,668 |
| **hybrid_8** | **1,860** | 1,844 | 2,215 | **24/28** | **2,170** |
| hybrid_open | 5,794 | 6,690 | 9,475 | 23/28 | 7,053 |

By category (mean tokens, sufficient): code — grep 17.3k 13/16, hybrid_5 1.2k 11/16; vault — grep 15.4k 6/8, hybrid_5 1.2k 7/8; cross — grep 11.3k 1/2, hybrid_5 1.0k 2/2; stale — hybrid 0/2, as designed. Retrieval 0.11 s graph-only, 0.17 s hybrid.

**Why the graph alone missed half.** `--explain` on the misses: (1) a long docstring is split into episodes and the continuation names no symbol, so `keys_settle`'s rule ("two chains with different keys are two things") and `merge_seeded`'s ("same offsets or the same text") are unreachable from their entities; (2) generic seeds — "chains" and "event" resolve to concept entities with dozens of chains, and the walk lands on the wrong ones; (3) the vault's `Neo4j` is several entities and the scalability note ranks below Day notes that mention it more. The text channel (BM25 over episode texts with identifiers split into parts, reciprocal-rank fused at the passage level) reaches all three: hybrid_8 answers 10 questions graph-only search_8 did not, and loses none.

**What grep costs and why.** Grep is not bad at finding the file (22/28); it is expensive because the agent then reads whole modules — `4_link.py` alone is 6k tokens — and because a common term (`chains`, `role`) pulls in three unrelated files. The wiki route is cheap but answers detail questions poorly (12/28): the wiki is a map, not the text.

**Graded** (same index; gpt-4.1-mini answers from the context alone, gpt-4.1 grades against the gold answer as correct / partial / wrong):

| condition | mean tok | correct | partial | wrong |
|---|---|---|---|---|
| read_gold (oracle) | 3,994 | 24 | 4 | 0 |
| grep_read | 16,486 | 16 | 9 | 3 |
| wiki | 5,563 | 13 | 8 | 7 |
| search_5 (graph only) | 1,167 | 12 | 10 | 6 |
| hybrid_5 | 1,191 | 18 | 8 | 2 |
| **hybrid_8** | **1,860** | **22** | 5 | 1 |
| hybrid_open | 5,794 | 19 | 8 | 1 |

Grading agrees with the string check: hybrid_8 is correct on 22 of 28 against 16 for grep-and-read, at one ninth of the tokens; the oracle itself only reaches 24, so 22 is close to the ceiling this grader allows. The four hybrid_8 misses on the string check are the two stale questions, C7 (the head fallback, whose rule sits in a `spans.py` paragraph the question's words do not reach) and V5 (a Day note that says "mean pooling" once, in a sentence the model's chains do not cover).

**After the re-sync** (working tree synced with `--local`, 85 code files / 494 code episodes including `src/query`, `src/connectors`, `scripts`; 247 of 250 new episodes extracted, 3 left pending; re-link 190,002 pairs, 7,871 reviewed, 2,915 entities; 1,350 new surfaces embedded). Same questions, graded:

| condition | mean tok | sufficient | correct | partial | wrong |
|---|---|---|---|---|---|
| read_gold (oracle) | 3,996 | 28/28 | 24 | 4 | 0 |
| grep_read | 16,607 | 22/28 | 16 | 8 | 4 |
| wiki | 5,565 | 12/28 | 13 | 8 | 7 |
| search_5 (graph only) | 1,125 | 11/28 | 6 | 10 | 12 |
| search_8 (graph only) | 1,785 | 12/28 | 8 | 10 | 10 |
| hybrid_5 | 1,160 | 22/28 | 17 | 9 | 2 |
| **hybrid_8** | **1,800** | **27/28** | **20** | 8 | **0** |
| hybrid_open | 5,454 | 24/28 | 18 | 8 | 2 |

The two stale questions now answer (hybrid_5 2/2). hybrid_8's one string miss (C5, `keys_settle`) still grades correct: the passage paraphrases the rule. Retrieval 0.18 s graph-only, 0.24 s hybrid on the larger graph.

**Graph-only got worse as the corpus grew** (search_5 on code questions 8/16 → 4/16). `--explain` shows why: the repository holds two copies of the pipeline (`old_server/entity_linking` and `server/public/src/inject`) with near-identical docstrings, plus the eval scripts' symbols, and the re-link (2,915 entities against 1,994) spreads a question's surfaces over more, smaller entities; the walk then lands on the legacy copy, `RUNNING.md` or a demo, and the per-document cap of two fills the eight slots before the current module's paragraph. BM25 is indifferent to that: the exact identifier saturates on both copies and fusion keeps the current one in range. The lesson for a deployment is the same as ADR connectors 11's: index the current tree, not its history (`--include`), and expect the graph channel alone to degrade with duplicates while the fused result does not.

**Decision.** `TEXT_CANDIDATES` is 10 by default (ADR tables 31): the Slack path and the CLI fuse the text channel unless `--text 0`. The harness passes `text=0` explicitly for its graph-only rows, so the comparison stays measurable.

**Honest limits.** Sufficiency is a string check, so a paraphrasing passage counts as a miss; the graded run below is the fairer number. The `stale` pair shows the cost of an out-of-date index: a search cannot answer about code it has not seen, so an agent's index must follow the working tree (`--local` sync, now including untracked files). Question authoring by the same person who wrote the code favours identifier-bearing questions, which is the text channel's best case.

## Agents and tokens, second pass: a real agent and questions nobody on the team wrote (`server/public/scripts/eval_agent_loop.py`, 2026-09-19)

The first pass measured what each strategy *would* put in front of an agent. This one lets the agent decide. gpt-4.1-mini gets a question and a tool kit, calls tools until it answers (at most 8 rounds, then it must answer), and every token it is billed for is counted: the prompt of every turn (the whole context is re-sent each call), completions, and the tool results inside them. gpt-4.1 grades each answer against the question's reference as correct / partial / wrong. Runs are cached by their inputs (`eval_agent_loop.cache.json`), temperature 0.

| Kit | Tools | Stands for |
|---|---|---|
| `files` | `glob`, `grep` (regex, 50 lines), `read` (numbered lines, 400-line cut, ranges) | a coding agent today |
| `both` | the same plus `search` | the graph search offered as one more tool |
| `search` | `search`, `read` | search first, open a file only to confirm |

`search` is `retrieve.search_passages` with the text channel, eight passages, rendered by `format_passages`. The corpus the file tools see is exactly what the index holds after the legacy copy was retired: `server/public`, `ADRs`, `CLAUDE.md`, and the vault under `vault/`.

| Set | n | Where the questions come from |
|---|---|---|
| hand | 28 | `agent_questions.json`, written by the harness author, identifier-heavy |
| para | 28 | the same questions rewritten by gpt-4.1 in plain words: no identifiers, file names or numbers, the project's vocabulary kept (`make_agent_questions.py`) |
| gen | 60 | random live passages from the index (30 code, 30 notes, seed 7) turned by gpt-4.1 into the question a colleague would ask that the passage answers, plus a two-sentence reference; a second gpt-4.1 call keeps only questions the passage supports, that are project-specific and natural |

Bootstrap 95% intervals (2,000 resamples) on mean billed tokens. `tok/correct` is total billed tokens divided by correct answers: what one right answer costs.

**Results** (348 runs; billed = every prompt and completion token across the agent's turns; unique = tool results plus completions, what a run costs if every re-sent prefix is a cache hit):

| set | kit | n | billed mean | 95% CI | median | calls | correct | partial | wrong |
|---|---|---|---|---|---|---|---|---|---|
| hand | files | 28 | 7,918 | 6,084–9,997 | 6,622 | 5.9 | 13 | 11 | 4 |
| hand | both | 28 | 11,673 | 8,338–15,306 | 9,336 | 5.6 | 18 | 6 | 4 |
| hand | search | 28 | 9,240 | 6,238–12,654 | **2,992** | 3.1 | **22** | 6 | **0** |
| para | files | 28 | 9,971 | 8,057–11,991 | 9,338 | 6.6 | 12 | 10 | 6 |
| para | both | 28 | 14,894 | 11,736–18,043 | 13,585 | 6.4 | 14 | 9 | 5 |
| para | search | 28 | 19,752 | 13,478–27,452 | 17,296 | 4.7 | 15 | 10 | 3 |
| gen | files | 60 | 11,789 | 9,505–14,073 | 8,799 | 6.0 | 31 | 17 | 12 |
| gen | both | 60 | 12,460 | 10,299–14,719 | 11,840 | 5.2 | 34 | 22 | 4 |
| gen | search | 60 | 13,543 | 10,351–16,660 | 7,180 | 4.0 | 37 | 18 | 5 |
| **all** | files | 116 | 10,416 | 9,034–11,869 | 8,607 | 6.1 | 56 (48 %) | 38 | 22 |
| **all** | both | 116 | 12,857 | 11,253–14,578 | 12,000 | 5.6 | 66 (57 %) | 37 | 13 |
| **all** | search | 116 | 14,003 | 11,473–16,738 | 8,990 | 3.9 | **74 (64 %)** | 34 | **8** |

Unique tokens: files 1,992, both 2,500, search 3,267. Cost per correct answer (billed): 21.6k, 22.6k, 22.0k. Paired on the same question: search correct where files was not on 33 questions, the reverse on 15. On the generated code questions search is 21/30 correct with 0 wrong against files 16/30 with 5 wrong; on generated note questions 16/30 against 15/30.

**What the first pass got right and wrong.** Right: the search makes the agent *more correct* (+16 points, wrong answers cut by two thirds) and, on identifier-bearing questions, cheaper (hand set median 3.0k against 6.6k). Wrong: the 9× token saving does not survive a real agent on average. Billed tokens are the same or higher, for two reasons the static harness could not see. First, the agent does not stop: in 66 of 116 search-kit runs it followed the search with `read` calls to confirm, and those runs cost 22.7k billed on average against 2.6k for the 50 runs that answered from the search alone (which were also *more* correct: 35/50 against 39/66). Second, every eight-passage result (about 1.9k tokens) is re-sent on every later turn, so a fat tool result is paid for as many times as there are turns after it. Third, on paraphrased questions the search kit is no better than files (15 against 12 correct) and twice as expensive: without identifiers the text channel loses its edge, and the agent searches again.

**The `trust` kit** (same two tools; the system prompt adds that search passages are verbatim evidence to answer from and cite, and that a file is opened only when they leave the question unanswered):

| set | n | billed mean | 95% CI | median | calls | correct | partial | wrong |
|---|---|---|---|---|---|---|---|---|
| hand | 28 | 8,288 | 5,344–11,663 | 2,946 | 2.5 | 22 | 6 | 0 |
| para | 28 | 16,416 | 11,393–21,767 | 10,158 | 4.2 | 18 | 9 | 1 |
| gen | 60 | 10,610 | 7,689–13,958 | 3,176 | 3.0 | 40 | 17 | 3 |
| **all** | 116 | 11,451 | 9,208–13,806 | **3,333** | 3.2 | **80 (69 %)** | 32 | **4** |

Against files only: correct 69 % against 48 %, wrong 4 against 22, median billed 3.3k against 8.6k, mean 11.5k against 10.4k (intervals overlap), billed tokens per correct answer 16.6k against 21.6k. Paired: trust wins 35 questions, files 11. 59 of 116 runs answered from a single search, 42 of them correctly. Generated code questions 23/30 with none wrong; generated note questions 17/30. The paraphrased set stays the expensive one (10.2k median): when the first search misses, the agent searches and reads on.

**The reportable result.** A real agent given the search as a tool, and told to trust what it returns, answers 69 % of questions about this project correctly against 48 % with grep and file reads, with wrong answers down from 22 to 4, at 3.3k billed tokens in the typical case against 8.6k and 23 % fewer tokens per correct answer. Mean cost is not lower, because the runs where the search misses are long; that tail, not the search call, is where the next token saving is.

**What this says about the tool design.** The search's cost is not the call but what the agent does after it. The levers, in the order they should be tried: tell the agent the passages are evidence it may cite (the `trust` kit below); return fewer, shorter passages by default for tool use and let the agent ask for more; keep the answer step's reranker idea (ADR tables 24) for the tool too; and note that prompt caching, which every provider now offers, discounts exactly the re-sent prefix that dominates the bill: under a 90 % cache-hit assumption the kits cost 2.8k, 3.5k and 4.3k.

## Code symbols and short-name collisions (`server/public/scripts/eval_code_collisions.py`, 2026-09-18)

Runs from `server/public`; nomic cached; no model calls. Ground truth is the
qualified name each seeded chain carries. Full table and decision: ADR
connectors 11. The short form:

| Repo | Baseline false merges | `defs` false merges | Reviews baseline → defs |
|---|---|---|---|
| `server/public/src` (701 chains) | 303 of 804 | 1 | 817 → 2125 |
| `old_server/entity_linking` (1070) | 2039 of 3501 | 2 | 630 → 3261 |
| httpx (1257) | 2564 of 3935 | 12 (0 with `Class.method`) | 909 → 3088 |
| fastapi (1575) | 2180 of 5559 | 12 (10 with `Class.method`) | 2024 → 5440 |

Static prevalence (`measure_collisions`, same day): colliding short names 8.2 % (this repo) to 25.2 % (openai); ambiguous rendered in-repo references 6.5 % to 21.9 %; external calls clashing with a repo name 71 in this repo's live source.

Caveats: seed-only chains, so the model's own spans are absent; the prior scorer's review band is counted, not adjudicated. Full table with every variant: ADR connectors 11.

## Code into the vault graph (2026-09-19)

This repository (`server/public`, `old_server/entity_linking`, `ADRs`,
`CLAUDE.md`: 44 files, 293 episodes, 122 seeded) synced by the GitHub
connector from the local checkout into a copy of the vault store, then stages
2–4 without a database. Three runs on one store; numbers in ADR connectors
12–13. What each run found:

| Run | Finding |
|---|---|
| 1 (32 min, network drop mid-judge, resumed) | Judge rejected 421 of 605 seed surfaces; 714 seeded chains withheld; 11 of 117 cross-source entities anchored on a symbol. → `base.name_seeds`. |
| 2 (22 min) | Seeds named; one 853-chain hub. 2,057 of its 4,708 edges were reviewer verdicts; 903 joined distinct symbols, 288 a symbol to prose. Signature-junk hypothesis refuted (6 multi-identifier model chains of 7,342). → `ChainRecord.key`, `pairing.keys_settle`. |
| 3 (4 min, `--full-relink`, reviews cached) | Largest entity 78 chains (the `chains` concept); 107 cross-source, 10 symbol-anchored; 992 code-only entities. |

End to end (2026-09-19, Docker back): `supabase db reset` applied all 26 migrations; the isolation suite passed 39/39; stage 5 uploaded the store as `app_ingest` (295 files, 1,393 episodes, 9,772 surfaces, 2,020 entities, 15,681 chains, 16,542 chain_surfaces, `full=True`); `retrieve embed` embedded 2,619 name surfaces; two questions through `retrieve ask`. "What does merge_seeded do in the chains stage, and why does it exist?" was answered correctly from `2_chains.py` with the GitHub file (`files.title`) as the citation. "Why did we move from Neo4j to Postgres, and where in the code is tenant isolation enforced?" was answered from three vault notes (`tables`, `PostgreSQL`, `database`); the code half of the question drew no code passage, so the answer named RLS from the notes rather than pointing at a migration or `5_store.py`. The store used is `old_server/artifacts/vault/store_with_code.json`.

Side observations: the judge re-sent 744 and then 421 surfaces on runs with no new text — surfaces no model answered are left without a verdict and re-asked every run (`judge_surfaces` docstring); the original vault store on disk was re-judged by another session at 11:28 on 2026-09-18, so 82 vault verdicts differ between the copy and the original. Cost of the three runs: roughly 300 coref calls, ~1,200 judge surfaces, ~13,500 reviews on `gpt-4.1-mini` (exact spend not read from the account).

## Full vault run through the live pipeline (2026-09-18)

`src.inject.main` over a copy of the 154-note vault (`old_server/artifacts/vault/`), gpt-4.1-mini for coref and review, 4 extraction workers. Baseline = `--no-judge`.

| | baseline (no judge) | judged (cascade, `--full-relink`) |
|---|---|---|
| episodes / candidates / chains | 301 / 8,504 / 7,092 | *(running)* |
| pairs scored / reviewed / merged | 266,269 / 13,611 (5.1 %) / 9,044 | |
| entities / contested mentions withheld | 3,942 / 105 | |
| wall clock | 2,846 s, including two restarts (a stall, then a memory kill during encode) | |

The stall is what produced the watchdog (B14); the memory kill produced the checkpointing inside the paid stages. The judged column and the incremental-edit run follow in the same table when they finish.

## Reframe — may the answer say it is the wrong question? (`eval_reframe.py`)

Twelve questions over the vault store, passages chosen by keyword (a stand-in for seeding + PPR, which needs Postgres), `gpt-4.1`, three runs per question. Expected stances written before running.

| | v1 prompt | v2 prompt (adds "provisional choice presented as settled") |
|---|---|---|
| fine questions left alone | 7/7 | **7/7** |
| bad premises reframed | 3/5 | **3/5** (HDBSCAN now caught; "always-up DB" answers with the warmup-endpoint reframe in the body; "run MCP locally" is arguably answerable — the notes did it once) |
| stance flips across 3 runs | — | **0/12** |

Shipped as ADR slack 16: `old_server/entity_linking/answer.py`, wired into `ask.py`, `--no-reframe` to pin the stance.

### Slack format rules (2026-09-19, `server/public/scripts/eval_reframe.py`)

Same twelve questions through the live module after adding the FORMAT section to the prompt (lead with the answer, under 120 words, plain sentences, bullets, no Markdown Slack cannot render, a reframe must not restate its preface) and the Block Kit renderer.

| | before | after |
|---|---|---|
| stance accuracy | 10/12 | **11/12** (the "always-up database" case now reframes) |
| fine questions left alone | 7/7 | 7/7 |
| answer length | paragraphs, often 150+ words | **mean 54 words, max 90, none over 120** |
| citations | file paths (`earlier/r87/days/Day 38.md`) | note titles (`Day 38`), linked when the file has a URL |

One residual: the reframed answers still repeated the better question inside the body about half the time despite the rule, so the renderer now drops sentences that quote the better question or open by explaining the question is off (`answer._without_preface`). Verified live the same day: a synthetic mention produced a three-block reply (preface, answer, `Sources: Day 23, Day 25`) in the channel in 7 s.

### Live path (Postgres, `app_connector`, real seeding) — 2026-09-18

Vault loaded as a tenant, `T0BBWPQ8BEC` installed, name surfaces embedded. Four of the eval questions through `src.query.retrieve ask --workspace`:

| Question | before the seeding fix | after (lexical half + rank fusion + per-file cap) |
|---|---|---|
| scale the Neo4j Aura instance | `answer` on eight passages, four from one note; `scalability.md` not retrieved | **`reframe`** citing `scalability.md`; also delivered in Slack as a threaded reply |
| why HDBSCAN as the final approach | **`reframe`** (Day 23, Day 25) | same |
| finish facts extraction before the demo | `answer`; Day 26 ("no burning need") not retrieved | still not retrieved |
| the four ACID guarantees | `answer`: "passages do not contain it"; eight passages tied | still tied: the vault's `## ACID` heading was fused into the surface `Schema\nACID`, which merged into the `Schema` entity (B15) |

Seeding through the live path also showed the rank-based seed weights letting the top surface dominate, and the baseline store (no judge) seeding on descriptions like `the graph database`, `my graphDB`.

## Fixes landed this session

The name judge (stage 2b, ADR tables 30), a stall watchdog for every LLM pool (`chains.completed`; a full-vault run had sat an hour in `SSLSocket.read`), sparse PPR and the score cutoff (`ask.py`, `CUTOFF_RATIO = 0.2` beside `ALPHA = 0.5`), the vector cache sidecar `<store>.vectors.npz`, incremental linking with `store["linking"]`, the delta upload with `store["uploaded"]`, `strip_markup` ported with the code-fence rule, and the pronoun leak into `chain_surfaces` (B12). 29 tests in `server/public`, 5 + 3 new in `old_server`. Each carries the measurement it rests on in its docstring.

## Not run, and why

| Proposal | Reason | What it would take |
|---|---|---|
| E11 reranker | needs ECB+ and LLM-generated queries | `eval_retrieval.py --root <ecb+>` after downloading ECB+ |
| E8 Batch API | needs a real backfill | one workspace backfill with a submit/poll loop |
| S5 transaction pooler, B2 migration replay, E6 SQL side, E7 | Docker/Postgres unavailable | `supabase start`, then the isolation suite through port 54329 |
| E10 Slack backfill | not run against Slack; static estimate from `slack.sync` | calls = 1 + ⌈users/200⌉ + ⌈channels/200⌉ + Σ⌈messages/200⌉ + one per threaded message, × 1.2 s pause. A 30-day window with 2,000 threads is ≥ 40 min per sync, every sync, until a cursor exists |
