# Security Audit Report — retrieval/ library

Audited against spec §11 (12 risks). Each risk lists status, affected files/lines, and fix applied.

---

## 11.1 Graph Query Injection — PARTIALLY MITIGATED → FIXED

**Gaps found:**
- `config.py:19`: `_validate_identifiers` raised `ValueError`; spec requires `InvalidInputError`.
- All public entry-point functions (`ppr_search`, `temporal_ppr_search`, etc.): `group_id` not validated against `^[A-Za-z0-9_\-]+$`.
- No `query` length cap at API boundary.

**Fixes applied:**
- `exceptions.py`: added `validate_inputs(query, group_id)` helper raising `InvalidInputError`.
- `config.py`: `_validate_identifiers` now raises `InvalidInputError` (imported from exceptions).
- `ppr.py`, `temporal.py`, `multihop.py`, `ppr_approx.py`, `treehop.py`, `multihop_beam.py`, `dag_multihop.py`: call `validate_inputs()` at entry.

All DB `execute_query` calls already use `$param` placeholders — no f-string interpolation of user values found. MITIGATED.

---

## 11.2 LLM Prompt Injection — PARTIALLY MITIGATED → FIXED

**Gaps found:**
- `rerank.py:117-118`: used `<query>` / `<fact>` tags; spec requires `<user_query>` / `<retrieved_fact>`.
- `multihop.py` and `shared.py`: used `<original_question>` instead of `<user_query>`.
- `shared.py` / `multihop.py`: individual fact strings not truncated to 500 chars in bridge prompts.
- No `len(query) <= 2000` enforcement (now part of `validate_inputs`).

**Fixes applied:**
- `rerank.py`: renamed tags to `<user_query>` / `<retrieved_fact>`.
- `multihop.py`, `shared.py`: renamed `<original_question>` to `<user_query>`.
- `shared.py` / `multihop.py`: added `f.fact[:500]` truncation in prompt formatting.
- `exceptions.py` → `validate_inputs`: rejects `len(query) > 2000`.

---

## 11.3 Data Exfiltration via External APIs — MITIGATED (design)

Protocol-based `Embedder` / `LLMClient` interfaces allow local implementations. `rerank=none` + `expand_seeds=False` + `hydrate=False` sends zero text to external APIs. No code fix needed.

---

## 11.4 DoS via Large Graph Loading — PARTIALLY MITIGATED → FIXED

**Gap:** Memory estimation before matrix allocation not implemented.

**Fix applied:**
- `ppr.py`: added optional `psutil` memory check before matrix build. Raises `GraphTooLargeError` if estimated CSR size exceeds 80% of available RAM. Skipped gracefully if `psutil` unavailable.

Hard limits (`max_nodes`, `max_edges`) checked in `graph_loader.py` before return. PPR timeout already wraps power iteration. MITIGATED.

---

## 11.5 DoS via Unbounded Multihop — PARTIALLY MITIGATED → FIXED

**Gap:** No `WARNING` when `min_new_facts=0` (convergence guard disabled).

**Fix applied:**
- `multihop.py`: logs `WARNING` at entry when `config.min_new_facts == 0`.
- `multihop_beam.py`: same.

All timeout wraps, Pydantic caps, and per-bridge timeouts already in place. MITIGATED.

---

## 11.6 Insecure Deserialization — PARTIALLY MITIGATED → FIXED

**Gap:** `Fact.uuid` had no regex format validator; malformed UUIDs from DB would silently propagate.

**Fix applied:**
- `models.py`: added `@field_validator("uuid")` on `Fact` checking `^[0-9a-fA-F-]{36}$`. Raises `ValueError` (caught by Pydantic → `ValidationError`); `_load_candidate_facts` already catches and logs + skips.

Score clamping already in place. Datetime parsing with skip-on-failure already in `ppr.py:_parse_dt`. MITIGATED.

---

## 11.7 Information Leakage via Error Messages — MITIGATED

All DB exceptions re-raised as `GraphLoadError("Failed to load …")`. All LLM exceptions caught and re-raised as `LLMCallError`. No Cypher, credentials, or internal paths in exception messages. No code fix needed.

---

## 11.8 Temporal Access Control — MITIGATED (design)

Authorization is the responsibility of the API layer. Library documented as requiring caller-level auth. No code fix needed.

---

## 11.9 Race Conditions in Concurrent PPR Loads — NOT MITIGATED → FIXED

**Fix applied:**
- `graph_loader.py`: added module-level `_LOAD_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("RETRIEVAL_PPR_CONCURRENCY", "4")))`. `load_group` acquires the semaphore before issuing DB queries.

---

## 11.10 Beam Search Token Cost — MITIGATED

`beam_width` already capped at `le=10` via Pydantic. Total timeout already in place. No code fix needed.

---

## 11.11 DAG Decomposition Prompt Injection — MITIGATED

`_topological_layers` raises `InvalidInputError` on cycles. Sub-question length capped at 500 chars. Decomposition failure falls back to single-node DAG with `WARNING` log. No code fix needed.

---

## 11.12 TreeHop Embedding Drift — PARTIALLY MITIGATED → FIXED

**Gap:** `EmbeddingFuser` and `TreeHopConfig.fusion_alpha` used bounds `(0.0, 1.0)`; spec requires `(0.5, 0.95)` to bound drift.

**Fixes applied:**
- `shared.py EmbeddingFuser.__init__`: changed guard to `not 0.5 < alpha < 0.95`.
- `treehop.py TreeHopConfig.fusion_alpha`: changed `Field(gt=0.0, lt=1.0)` to `Field(default=0.6, gt=0.5, lt=0.95)`.

Cosine similarity logging at DEBUG / WARNING already implemented in `treehop.py`. MITIGATED.

---

## Summary

| Risk | Status after fixes |
|---|---|
| 11.1 Graph query injection | MITIGATED |
| 11.2 LLM prompt injection | MITIGATED |
| 11.3 Data exfiltration | MITIGATED (design) |
| 11.4 DoS large graph | MITIGATED |
| 11.5 DoS unbounded multihop | MITIGATED |
| 11.6 Insecure deserialization | MITIGATED |
| 11.7 Information leakage | MITIGATED |
| 11.8 Temporal access control | MITIGATED (design) |
| 11.9 Race conditions | MITIGATED |
| 11.10 Beam token cost | MITIGATED |
| 11.11 DAG prompt injection | MITIGATED |
| 11.12 TreeHop drift | MITIGATED |
