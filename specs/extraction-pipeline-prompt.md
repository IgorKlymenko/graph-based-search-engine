# Prompt for Fable 5 — implement the extraction pipeline

Paste the following into a fresh Claude Code (Fable 5) session in this repo:

---

Implement `specs/extraction-pipeline.md` end to end. That spec is the
authority on scope, structure, and naming; read it first and follow it
exactly, including its bug-fix list (§8) and the security/consistency
review (§9).

Reference material, in reading order:

1. `specs/extraction-pipeline.md` — the spec (binding).
2. `entity_extraction.ipynb` — reference implementation for mention
   extraction, gleaning, embedding, clustering, and LLM resolution. Copy the
   system prompts and JSON schemas verbatim; port only the cells the spec
   lists as in scope (skip demos, `%history`, commented-out code, and the
   consensus correlation-clustering code — clustering is HDBSCAN only).
3. `relation_extraction.ipynb` — reference implementation for relation
   extraction, object resolution, quote/span verification, and confidence
   scoring. Same rule: prompts, schemas, and Pydantic models verbatim.
4. `graph_upload.py` — the frozen final stage. Its only permitted change is
   the `--skip-facts` flag described in the spec (no Fact-node creation or
   upload in this pipeline).
5. `services/e5.py`, `services/openai.py`, `embed_cache.py` — existing code
   to reuse or absorb rather than duplicate.

Constraints:

- Follow `CLAUDE.md`: uv only (never pip), type hints everywhere, 88-char
  lines, ruff + pyright clean, pytest with anyio for async.
- All e5 embedding goes through `services/e5.py` — both `"query: "` and
  `"passage: "` prefixes are required; a missing passage prefix is the known
  silent bug with this model.
- Tests must run offline: mock LLM calls, never load embedding models.
- Do not modify anything under `retrieval/`, and leave the notebooks
  untouched.

Work through the spec stage by stage, committing nothing until the
acceptance checklist (§11) passes. Verify at the end by running the pipeline
on 2–3 small files with `--token-budget 3000 --skip-upload`, then
`uv run python graph_upload.py --dry-run --skip-embeddings --skip-facts`.
In your final summary, report: files created, test results, the end-to-end
dry-run output, and your §9 security/consistency findings (fixed vs.
flagged).

---
