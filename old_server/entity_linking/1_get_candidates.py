"""Stage 1 — high-recall entity candidate extraction with gleaning.

Reads episodes from the store (``episodes.py``) and writes candidates back into
it. The unit of work is the *episode*, not the file: an episode carries
``extracted_at``, so a document edit re-extracts only the episodes whose text
actually changed, and a crashed run resumes without redoing finished episodes.

    uv run python entity_linking/1_get_candidates.py --notes-dir ./notes \\
        --store ./store.json --tenant-id <uuid>
    uv run python entity_linking/1_get_candidates.py --store ./store.json --dry-run

Prompts and the JSON schema follow ``entity_extraction.ipynb``, apart from
naming the extracted span ``concept``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

sys.path.insert(0, str(Path(__file__).parent))

import episodes as E  # noqa: E402  (needs the path shim above)
from prompts import (  # noqa: E402
    CANDIDATE_SCHEMA,
    GLEANING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

EXTRACTION_MODEL = "gpt-4.1-mini"
EPISODE_WORKERS = 8
SAVE_EVERY = 25  # checkpoint the store every N episodes
LLM_TIMEOUT = 90.0
LLM_MAX_RETRIES = 2

# ── LLM ───────────────────────────────────────────────────────────────────────

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    """Process-wide client (reads OPENAI_API_KEY); thread-safe by design."""
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def _complete_json(*, system: str, user: str, model: str) -> dict[str, Any]:
    """Chat completion constrained to CANDIDATE_SCHEMA; returns parsed JSON."""
    client = _get_client().with_options(
        timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "entity_candidates",
                "strict": True,
                "schema": CANDIDATE_SCHEMA,
            },
        },
    )
    return json.loads(response.choices[0].message.content or "{}")


def extract_candidates(episode_text: str, *, model: str) -> list[dict]:
    """First-pass candidate extraction for a single episode (high recall)."""
    return _complete_json(
        system=SYSTEM_PROMPT, user=episode_text, model=model
    ).get("candidates", [])


def glean_episode(
    episode_text: str, first_pass: list[dict], *, model: str
) -> list[dict]:
    """Second pass on one episode: returns only net-new candidate occurrences."""
    seen = {(c["concept"], c["context"]) for c in first_pass}
    already = json.dumps(
        [{"concept": concept, "context": ctx} for concept, ctx in sorted(seen)],
        ensure_ascii=False,
    )
    user = (
        f"TEXT:\n{episode_text}\n\n"
        f"ALREADY EXTRACTED (do not re-emit these concept+context pairs):\n"
        f"{already}\n\n"
        "Which candidate occurrences were missed?"
    )
    raw = _complete_json(
        system=GLEANING_SYSTEM_PROMPT, user=user, model=model
    ).get("candidates", [])
    return [c for c in raw if (c["concept"], c["context"]) not in seen]


# ── candidate records ─────────────────────────────────────────────────────────

def candidate_id_for(episode_id: str, concept: str, context: str) -> str:
    """Identity of one candidate occurrence, content-addressed like an episode.

    Keyed on the sentence rather than a character offset so it survives the
    document moving, and so two extraction runs over unchanged text produce the
    same ids. Repeating a concept in one sentence collapses to one candidate.
    """
    digest = E.sha1_text(f"{concept}\x1f{context}")
    return E.stable_uuid(f"candidate:{episode_id}|{digest}")


def build_candidates(
    episode_id: str, raw: list[dict], *, gleaned: bool
) -> list[dict]:
    """Turn raw model output into candidate records, dropping malformed items.

    A candidate whose concept is not literally inside its own context is
    unusable downstream — the span cannot be located — so it is dropped here
    rather than corrupting the embedding stage.
    """
    records: list[dict] = []
    for item in raw:
        concept, context = item.get("concept", ""), item.get("context", "")
        if not concept or not context or concept not in context:
            continue
        records.append({
            "candidate_id": candidate_id_for(episode_id, concept, context),
            "episode_id": episode_id,
            "concept": concept,
            "type": item.get("type", "concept"),
            "context": context,
            "gleaned": gleaned,
        })
    return records


def pending_episodes(store: dict) -> list[str]:
    """Live episodes with no extraction yet — the work list, resumable."""
    return [
        eid for eid, ep in store["episodes"].items()
        if ep.get("removed_at") is None and ep.get("extracted_at") is None
    ]


def live_candidates(store: dict) -> list[dict]:
    """Candidates whose episode is still present in its document.

    Liveness is derived rather than stored: a candidate belongs to exactly one
    episode, so duplicating ``removed_at`` onto it would just be state that can
    fall out of sync.
    """
    episodes = store["episodes"]
    return [
        c for c in store["candidates"].values()
        if episodes.get(c["episode_id"], {}).get("removed_at") is None
    ]


# ── extraction run ────────────────────────────────────────────────────────────

def process_episode(
    store: dict, episode_id: str, *, model: str, enable_gleaning: bool
) -> tuple[list[dict], dict | None]:
    """Extract one episode. Returns ``(records, failure)`` — never raises.

    A failed episode leaves ``extracted_at`` unset so the next run retries it,
    which is the behaviour the old per-file cache could not offer.
    """
    text = store["episodes"][episode_id]["text"]
    try:
        first = extract_candidates(text, model=model)
    except Exception as exc:  # noqa: BLE001 — API failure for this episode only
        return [], {"episode_id": episode_id, "stage": "extract",
                    "error": repr(exc)}

    records = build_candidates(episode_id, first, gleaned=False)
    if not enable_gleaning:
        return records, None
    try:
        gleaned = glean_episode(text, first, model=model)
    except Exception as exc:  # noqa: BLE001 — keep the first-pass result
        return records, {"episode_id": episode_id, "stage": "glean",
                         "error": repr(exc)}
    return records + build_candidates(episode_id, gleaned, gleaned=True), None


def run(
    store: dict,
    *,
    model: str = EXTRACTION_MODEL,
    enable_gleaning: bool = True,
    workers: int = EPISODE_WORKERS,
    limit: int | None = None,
    store_path: Path | None = None,
) -> dict:
    """Extract every pending episode, writing candidates into ``store``.

    Checkpoints to ``store_path`` every ``SAVE_EVERY`` episodes so a long run
    survives interruption with its finished work intact.
    """
    todo = pending_episodes(store)
    if limit is not None:
        todo = todo[:limit]
    print(f"[1] candidates: {len(todo)} episodes to extract "
          f"({len(store['episodes'])} in store, "
          f"{len(pending_episodes(store))} pending total)")
    if not todo:
        return {"episodes": 0, "candidates": 0, "failures": []}

    failures: list[dict] = []
    lock = threading.Lock()
    done = 0

    with ThreadPoolExecutor(max_workers=min(len(todo), workers)) as executor:
        futures = {
            executor.submit(process_episode, store, eid, model=model,
                            enable_gleaning=enable_gleaning): eid
            for eid in todo
        }
        for future in as_completed(futures):
            episode_id = futures[future]
            records, failure = future.result()
            with lock:
                for record in records:
                    store["candidates"][record["candidate_id"]] = record
                if failure is None:
                    store["episodes"][episode_id]["extracted_at"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                else:
                    failures.append(failure)
                done += 1
                if store_path and done % SAVE_EVERY == 0:
                    E.save_store(store_path, store)

    if store_path:
        E.save_store(store_path, store)
    extracted = len(todo) - len(failures)
    print(f"  {extracted}/{len(todo)} episodes extracted, "
          f"{len(store['candidates'])} candidates in store")
    if failures:
        print(f"  WARNING: {len(failures)} episodes failed and stay pending "
              f"— they will be retried on the next run")
    return {"episodes": extracted, "candidates": len(store["candidates"]),
            "failures": failures}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", type=Path, required=True,
                   help="Store JSON; created if absent")
    p.add_argument("--notes-dir", type=Path, default=None,
                   help="Sync this directory into the store before extracting")
    p.add_argument("--tenant-id", default=None,
                   help="Required with --notes-dir; scopes document identity")
    p.add_argument("--source", default=E.UPLOAD_SOURCE,
                   help=f"Connector name for --notes-dir (default: {E.UPLOAD_SOURCE})")
    p.add_argument("--model", default=EXTRACTION_MODEL)
    p.add_argument("--no-gleaning", action="store_true",
                   help="Skip the second-pass gleaning call")
    p.add_argument("--workers", type=int, default=EPISODE_WORKERS)
    p.add_argument("--limit", type=int, default=None,
                   help="Extract at most N episodes this run")
    p.add_argument("--dry-run", action="store_true",
                   help="Sync and report the work list without calling the LLM")
    args = p.parse_args()

    store = E.load_store(args.store)

    if args.notes_dir:
        if not args.tenant_id:
            sys.exit("--tenant-id is required with --notes-dir")
        if not args.notes_dir.is_dir():
            sys.exit(f"not a directory: {args.notes_dir}")
        diffs = E.sync_corpus(store, args.notes_dir, tenant_id=args.tenant_id,
                              source=args.source)
        added = sum(len(d.added) for d in diffs)
        removed = sum(len(d.removed) for d in diffs)
        changed = sum(1 for d in diffs if d.status != "unchanged")
        print(f"[0] sync: {len(diffs)} documents, {changed} changed "
              f"-> {added} episodes added, {removed} retired")

    if args.dry_run:
        todo = pending_episodes(store)
        print(f"[1] dry-run: {len(todo)} episodes pending extraction")
        for eid in todo[:10]:
            preview = store["episodes"][eid]["text"][:60].replace("\n", " ")
            print(f"  {eid[:8]}  {preview}…")
        if len(todo) > 10:
            print(f"  … and {len(todo) - 10} more")
    else:
        run(store, model=args.model, enable_gleaning=not args.no_gleaning,
            workers=args.workers, limit=args.limit, store_path=args.store)

    E.save_store(args.store, store)
    print(f"\nstore -> {args.store}  "
          f"({len(store['files'])} documents, {len(store['episodes'])} episodes, "
          f"{len(live_candidates(store))} live candidates)")


if __name__ == "__main__":
    main()
