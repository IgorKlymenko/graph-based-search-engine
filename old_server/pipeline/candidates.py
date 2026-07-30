"""Stage 1 — high-recall entity candidate extraction with gleaning.

Ported from ``entity_extraction.ipynb``. Prompts and the JSON schema follow the
notebook apart from naming the extracted span ``concept`` rather than ``text``;
the plumbing is threaded (files x episodes) and resumable via
``entity_cache.json``.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

from pipeline.config import (
    ENTITY_CACHE,
    EPISODE_MAX_CHARS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from pipeline.episodes import Episode, split_episodes
from pipeline.models import Candidate
from services.openai import complete_json_schema

SYSTEM_PROMPT = """You are an entity candidate extractor for knowledge-graph construction. This is Stage 1 of a pipeline: each candidate you emit is later embedded (exact span + surrounding sentence) and resolved into a graph node. You extract entity ENDPOINTS only — never relations, never scalar values.

TASK
Extract every span that names or specifically refers to an entity. List each occurrence separately, even when two candidates refer to the same thing. Never merge, normalize, or deduplicate — resolution is a downstream stage.

WHAT COUNTS AS A MENTION
1. Proper names: people, organizations, places, facilities, products, software, systems, events, works (papers, books, reports, standards), datasets. Numerals inside names are part of the name ("HippoRAG 2", "GPT-4", "World War II").
2. Named technical terms: terms that function as the established name of a concept, method, or artifact in the domain ("entity resolution", "personalized PageRank", "contrastive loss") — as opposed to ad-hoc descriptions.
3. Specific-referent nominals: a definite or demonstrative noun phrase ("the company", "this model", "our pipeline") counts ONLY if a reader of this document alone could identify the unique entity it refers to (it has an in-document antecedent or a unique referent). If it denotes a kind or class, exclude it.

WHAT IS NEVER A MENTION
- Pronouns and bare demonstratives: "it", "they", "he", "this"/"that" without a noun head, relative pronouns.
- Generic common nouns and bare plurals ("users", "data", "the results", "systems", "an approach"), even domain-flavored ones, unless they are part of a longer specific term.
- Scalar values and attributes: numbers, money, counts, dates, times, percentages, durations, measurements. A separate stage handles these. ("2023" is never a candidate; "GPT-4" is — the numeral is part of the name.)
- Relations or facts of any kind.

SPAN BOUNDARIES
- Extract the maximal span that functions as a single name or term. Never extract a proper-name substring of a longer proper name: "University of Toronto" yields exactly one candidate, not also "Toronto".
- DO separately extract a name that makes its own referential contribution inside a larger phrase — as a modifier of a common-noun head, in a possessive, or in coordination: "the Neo4j database" yields "Neo4j"; "the universities of Toronto and Waterloo" yields "Toronto" and "Waterloo".
- Acronym introduced alongside its expansion, e.g. "Personalized PageRank (PPR)": emit TWO candidates — "Personalized PageRank" and "PPR" — sharing the same context. Do not emit the combined parenthesized span. Extract later standalone occurrences of the acronym normally, one per occurrence.
- Exclude leading articles from proper names ("the" in "the University of Toronto" is not part of the candidate) but keep the determiner when the candidate is a nominal ("the company", "this model") — it carries the definiteness signal.

VERBATIM REQUIREMENTS
- "concept": copied character-for-character from the source, original casing and punctuation intact. It must be findable by exact string search inside its "context".
- "context": the complete sentence containing the candidate, copied verbatim. If the candidate appears in a heading, list item, table cell, or other fragment without sentence punctuation, use the entire line or cell. If the candidate spans a sentence boundary, include every sentence it touches. Never paraphrase, trim, or repair the source text.

TYPE FIELD
Use a lowercase singular noun. Prefer one of these anchor types whenever it fits:
person, organization, location, facility, product, software, method, concept, event, work, dataset, field
Tie-breaks: a runnable code artifact, library, or system → software (over product, over tool); a procedure, algorithm, or technique → method; an abstract idea that is not a procedure → concept; a paper, book, report, or standard → work; an academic or industry discipline → field. Mint a new type only when no anchor fits, and make it a single generic lowercase noun that would be reused verbatim in unrelated documents ("gene", "chemical", "regulation") — never a document-specific compound.

POSTURE
Favor recall on names and established terms — a missed candidate is unrecoverable downstream, a spurious one is filtered later. The single exception: generic nominals and class-level terms, where precision wins, because unresolvable generic nodes corrupt graph traversal. When unsure whether a span is an established term or a mere description, ask: is this a name someone would query the graph for, or a phrase describing something? Names and terms in; descriptions out."""

GLEANING_SYSTEM_PROMPT = """You are auditing an entity candidate extraction for completeness. This is the second pass of Stage 1 in a knowledge-graph pipeline: each candidate you emit is later embedded (exact span + surrounding sentence) and resolved into a graph node.

You will receive the original text, followed by the (concept, context) pairs already extracted by the first pass.
Your task: emit ONLY candidate occurrences the first pass MISSED. A candidate occurrence is a specific span in a specific sentence — if a name already in the list occurs AGAIN in a sentence not covered by the list, that new occurrence WAS missed and you must emit it. Do not re-emit any (concept, context) pair already listed.

Apply the same extraction rules as the first pass:

IN SCOPE
- Proper names: people, organizations, places, facilities, products, software, systems, events, works, datasets. Numerals inside names are part of the name ("HippoRAG 2").
- Named technical terms that function as the established name of a concept, method, or artifact — not ad-hoc descriptions.
- Definite/demonstrative nominals ("the company", "this model") ONLY if a reader of this text alone can identify the unique entity referred to. Kind-level or generic uses are out.

NEVER IN SCOPE
- Pronouns and bare demonstratives ("it", "they", "this" without a noun head).
- Generic common nouns and bare plurals ("users", "the results", "systems"), unless part of a longer specific term.
- Scalar values: numbers, money, counts, dates, times, percentages, durations. A separate stage handles these.
- Relations or facts.

SPAN AND CONTEXT RULES
- Maximal name span; never a proper-name substring of a longer proper name ("University of Toronto", not also "Toronto"). But DO extract an embedded name making its own referential contribution ("Neo4j" in "the Neo4j database").
- Acronym introduced with its expansion: two candidates sharing one context, never the combined parenthesized span.
- "concept": verbatim, character-for-character, findable by exact string search inside its "context".
- "context": the complete sentence, verbatim; the whole line/cell for headings and fragments; every sentence touched if the candidate crosses a boundary.

TYPE FIELD
Lowercase singular noun. Prefer: person, organization, location, facility, product, software, method, concept, event, work, dataset, field. Runnable artifact → software; procedure/algorithm → method; abstract non-procedure → concept; paper/report/standard → work. Mint a new type only if no anchor fits, as a single generic noun reusable across unrelated documents.

POSTURE
Common first-pass misses to check for deliberately: repeated occurrences of already-found names; candidates inside headings, lists, tables, captions, and parentheticals; embedded modifier names; acronym or expansion halves where only one was caught; specific-referent nominals; multiword technical terms. Favor recall on names and established terms. Do NOT use this pass to pad the list with generic nouns the first pass correctly excluded — an entity the first pass skipped and an exclusion the first pass applied correctly look similar; re-check against the rules above before emitting. If nothing was missed, return an empty candidates list."""

CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "concept": {"type": "string"},
                    "type": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["concept", "type", "context"],
            },
        }
    },
    "required": ["candidates"],
}


def extract_candidates(episode_text: str, *, model: str) -> list[dict]:
    """First-pass candidate extraction for a single episode (high recall)."""
    result = complete_json_schema(
        system=SYSTEM_PROMPT,
        user=episode_text,
        schema=CANDIDATE_SCHEMA,
        schema_name="entity_candidates",
        model=model,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )
    return result.get("candidates", [])


def glean_episode(
    episode_text: str, first_pass: list[dict], *, model: str
) -> list[dict]:
    """Second pass on one episode: returns only net-new candidate occurrences."""
    first_pairs = {(m["concept"], m["context"]) for m in first_pass}
    already = json.dumps(
        [{"concept": cand, "context": ctx} for cand, ctx in sorted(first_pairs)],
        ensure_ascii=False,
    )
    user_msg = (
        f"TEXT:\n{episode_text}\n\n"
        f"ALREADY EXTRACTED (do not re-emit these concept+context pairs):\n"
        f"{already}\n\n"
        "Which candidate occurrences were missed?"
    )
    raw = complete_json_schema(
        system=GLEANING_SYSTEM_PROMPT,
        user=user_msg,
        schema=CANDIDATE_SCHEMA,
        schema_name="entity_candidates",
        model=model,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    ).get("candidates", [])

    new_candidates: list[dict] = []
    for m in raw:
        if (m["concept"], m["context"]) in first_pairs:
            continue
        try:
            m["char_offset"] = episode_text.index(m["context"])
        except ValueError:
            m["char_offset"] = -1
        new_candidates.append(m)
    return new_candidates


def _tag(raw: dict, episode_idx: int, episode: Episode, source_file: str, *,
         gleaned: bool) -> dict:
    """Attach episode/document metadata to a raw candidate and validate its shape."""
    return Candidate(
        concept=raw["concept"],
        type=raw["type"],
        context=raw["context"],
        episode_idx=episode_idx,
        episode=episode.text,
        source_file=source_file,
        episode_char_start=episode.char_start,
        episode_char_end=episode.char_end,
        char_offset=raw.get("char_offset") if gleaned else None,
        gleaned=gleaned,
    ).model_dump()


def process_file(
    path: Path,
    text: str,
    *,
    model: str,
    enable_gleaning: bool,
    episode_workers: int,
) -> tuple[list[dict], list[dict]]:
    """Extract candidates from every episode of one file, in episode order.

    Returns the candidates plus a record of any episode whose API call failed. A
    failed episode is skipped instead of aborting the run; the record keeps the
    loss visible, since the file is marked processed and never retried.
    """
    episodes = split_episodes(text, EPISODE_MAX_CHARS)
    failures: list[dict] = []

    def _one(item: tuple[int, Episode]) -> tuple[int, list[dict]]:
        idx, episode = item
        try:
            first = extract_candidates(episode.text, model=model)
        except Exception as exc:  # noqa: BLE001 — API failure per episode
            failures.append({"source_file": str(path), "episode_idx": idx,
                             "stage": "extract", "error": repr(exc)})
            return idx, []
        candidates = [_tag(m, idx, episode, str(path), gleaned=False) for m in first]
        if enable_gleaning:
            try:
                gleaned = glean_episode(episode.text, first, model=model)
            except Exception as exc:  # noqa: BLE001 — keep the first-pass result
                failures.append({"source_file": str(path), "episode_idx": idx,
                                 "stage": "glean", "error": repr(exc)})
                gleaned = []
            candidates += [
                _tag(m, idx, episode, str(path), gleaned=True) for m in gleaned
            ]
        return idx, candidates

    results: dict[int, list[dict]] = {}
    if episodes:
        workers = min(len(episodes), episode_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, candidates in executor.map(_one, enumerate(episodes)):
                results[idx] = candidates

    ordered: list[dict] = []
    for idx in sorted(results):
        ordered.extend(results[idx])
    return ordered, failures


def _count_tokens(text: str, model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def load_cache(path: Path) -> dict:
    if path.exists():
        cache = json.loads(path.read_text())
        cache.setdefault("failed_episodes", [])  # caches written before this key
        return cache
    return {"processed_files": [], "candidates": [], "failed_episodes": []}


def _save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def run(
    notes_dir: Path,
    *,
    model: str,
    enable_gleaning: bool,
    token_budget: int | None,
    file_workers: int,
    episode_workers: int,
    cache_path: Path = ENTITY_CACHE,
) -> dict:
    """Extract candidates from pending .md files under ``notes_dir``.

    Resumable: files already in ``processed_files`` are skipped. ``token_budget``
    (when set) caps how many pending files are processed this run.
    """
    cache = load_cache(cache_path)
    processed = set(cache["processed_files"])
    pending = [p for p in sorted(notes_dir.rglob("*.md")) if str(p) not in processed]

    selected: list[tuple[Path, str]] = []
    cumulative = 0
    for path in pending:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        if token_budget is not None:
            n_tokens = _count_tokens(text, model)
            if cumulative + n_tokens > token_budget:
                break
            cumulative += n_tokens
        selected.append((path, text))

    print(
        f"[1/6] candidates: {len(pending)} pending, {len(selected)} selected "
        f"this run (already cached: {len(processed)})"
    )
    if not selected:
        return cache

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(len(selected), file_workers)) as executor:
        futures = {
            executor.submit(
                process_file,
                path,
                text,
                model=model,
                enable_gleaning=enable_gleaning,
                episode_workers=episode_workers,
            ): path
            for path, text in selected
        }
        for future in as_completed(futures):
            path = futures[future]
            candidates, failures = future.result()
            with lock:
                cache["candidates"].extend(candidates)
                cache["failed_episodes"].extend(failures)
                cache["processed_files"].append(str(path))
                _save_cache(cache_path, cache)
            failed = f" ({len(failures)} episodes failed)" if failures else ""
            print(f"  {path.name}: {len(candidates)} candidates{failed}")

    print(f"  total: {len(cache['candidates'])} candidates "
          f"from {len(cache['processed_files'])} files")
    if cache["failed_episodes"]:
        print(f"  WARNING: {len(cache['failed_episodes'])} episodes failed and were "
              f"skipped — see 'failed_episodes' in {cache_path}")
    return cache
