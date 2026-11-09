# Glossary

Terms as this codebase uses them. Use these words; do not coin synonyms.

| Term | Meaning | Where it lives |
|---|---|---|
| **tenant** | One customer. The isolation unit; every content row carries `tenant_id`. | `tenants` |
| **workspace** | A source's own id for an install (Slack `team_id`, Notion workspace). What the connector role declares; the DB maps it to a tenant. | `connector_installations.external_id` |
| **connector / source** | Where documents come from: `slack`, `gdrive`, `notion`, `upload`. | `files.source`, `RENDER_VERSIONS` |
| **installation** | One `(source, external_id)` row with a token handle; soft-deleted by `uninstalled_at`. | `connector_installations` |
| **claim vs derive** | `app_ingest` *claims* a tenant by setting `app.tenant_id`; `app_connector` *derives* it from the declared workspace. | `current_tenant_id()` |
| **wall / door** | RESTRICTIVE policy that says whose rows (AND) / PERMISSIVE policy that says what the role may do (OR). | migrations `20260806190227` |
| **ACK path / work path** | The public webhook handler that only inserts an event and returns 200 / the worker that claims the event and does everything else. | `app_ack` vs `app_connector` |
| **file / document** | One source object rendered to text, identified by path-like `(tenant, source, external_id)`. Keeps identity across edits. | `files`, `episodes.sync_file` |
| **render / `render_version`** | The connector's object → text step. Bumping the version invalidates every hash for that source. | `1_episodes.py` |
| **episode** | A ~1000-char verbatim slice of a document, split on blank lines, identified by its content hash. The unit of extraction and of retrieval. | `episodes` |
| **span / candidate** | One occurrence of a name-like string in an episode, with offsets, a derived type (`proper_name`, `definite_description`, `pronoun`, `nominal`) and an origin (`parser` or model `n*`). | `store.candidates`, `spans.Span` |
| **surface** | A distinct string that names something (`"Sarah Chen"`, `"S. Chen"`). Not an entity identifier. One embedding per surface per tenant. | `surfaces`, `chain_surfaces.surface` |
| **identifying** | A surface with at least one content word; pronouns, determiners, numerals alone are not. | `spans.is_identifying` |
| **name / non-name** | The judge's verdict on an identifying surface: would the author look it up by this string, and does it denote one specific thing. Non-names never seed and never carry chain identity. | `utils/judge.py`, `surfaces.is_name`, ADR tables 30 |
| **chain** | All spans in one episode that the model said corefer. The linking unit; carries `entity_id`. Identified by `(episode_id, content_key)`. | `chains` |
| **head** | The chain's best name (proper name > descriptive phrase > longest). One `is_head` surface per chain. | `chain_surfaces.is_head`, `spans.pick_head` |
| **`content_key`** | `sha1` of the chain's sorted identifying surfaces; what the chain contains. | `chains.content_key` |
| **entity** | A cluster of chains across episodes judged to be one real-world thing. Thin row; name is derived. | `entities`, `entity_display_name()` |
| **display name** | Derived: most frequent head surface. `entities.display_name` is a human override only. | `entity_display_name()` |
| **evidence** | Corpus-wide aggregation of per-episode coref decisions: `same_chain`, `split_in_episode`, `ambiguous_surfaces`. | `pairing.Evidence` |
| **alias_hit / cooccurrence** | Features: some episode put the two surfaces in one chain / some episode saw both and kept them apart. | `pairing.features` |
| **ambiguous / contested surface** | A surface bridging two surfaces some episode kept apart (`S. Chen` for Sarah and Steven). Carries no alias evidence and no string similarity. | `pairing.build_evidence` |
| **context gate** | `context_gated` fires only when string similarity ≥ 0.90, i.e. where names cannot decide. | `pairing.context_gated` |
| **review band** | Scores in `[split_at, merge_at)`; each pair costs one LLM adjudication, cached by prompt hash. | `pairing.decide`, `main._ask_cached` |
| **unresolved / withheld** | A chain whose every surface is contested and which merged with nothing; keeps `entity_id NULL` rather than becoming a fake entity. | `pairing.unresolved`, `chains.entity_id` |
| **graph resolve** | Asking the model which existing entity profile an unresolved mention matches, or none. | `main.resolve_against_graph` |
| **store** | The one JSON file holding files, episodes, candidates, chains, reviews, entities, unresolved. Resumable state. | `episodes.load_store` |
| **edge** | Two entities with chains in the same episode. Derived by self-join; never stored. Weighted by PMI. | ADR tables 24 |
| **PMI** | `ln(together · n / (df_a · df_b))`; positive means more than chance. Pairs at or below chance are dropped. | `ask.build_graph` |
| **seed** | An entity given restart mass at query time, reached from a surface hit. | `ask.seed_entities` |
| **PPR / alpha** | Personalized PageRank; alpha is the restart probability (higher stays nearer the seeds). | `ask.ppr` |
| **passage** | An episode returned for a query, scored by the mean PPR score of its entities. | `ask.collect_episodes` |
| **stance** | The answer step's first decision: `answer`, `reframe` (the passages show a better question; cite them, ask it, answer it) or `clarify` (a name means several things). | `answer.answer`, ADR slack 16 |
| **footprint** | Every document that mentions an entity; the coverage task the graph is for. | `demos/eval_footprint.py` |
| **thread file / day file** | Slack document kinds: a replied-to message with its replies / one channel-day of loose messages. | `slack.documents` |
| **ECB+** | Event Coreference Bank Plus, the newswire coreference benchmark used for real-extraction evals. | `demos/eval_ecbplus.py` |
| **CoNLL F1** | Mean of MUC, B³, CEAFe. Report CEAFe and B³, never pairwise P/R. | `coref_metrics.py` |
| **HNSW / trigram** | pgvector approximate-nearest-neighbour index on embeddings / `pg_trgm` GIN index for lexical matches on short strings. | `surfaces` indexes |
| **turn / thread context** | One earlier message of a Slack thread (`role, name, text`), Nazar's own marked as assistant; the last 12 are read back before answering a reply in a thread he is in. | `query/thread.py` |
| **condense** | Rewriting a follow-up ("and the pool model?") into a standalone question plus a `needs_lookup` flag, before retrieval. gpt-4.1-mini. | `thread.condense` |
