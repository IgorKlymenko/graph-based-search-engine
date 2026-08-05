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
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entity_linking import spans as SP  # noqa: E402
from entity_linking._stages import load  # noqa: E402
from entity_linking.prompts import CHAIN_SCHEMA, CHAIN_SYSTEM_PROMPT  # noqa: E402

E = load("1_episodes", "entity_linking.episodes")

load_dotenv()

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


def _complete_json(
    *, system: str, user: str, model: str, schema: dict, schema_name: str
) -> dict[str, Any]:
    """Chat completion constrained to a strict JSON schema; returns parsed JSON."""
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
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content or "{}")


class UnknownSpanIds(ValueError):
    """The model referenced span ids that were never issued."""


def extract_chains(episode_text: str, *, model: str) -> dict:
    """One call: coreference chains over pre-indexed spans, answered in ids.

    Raises ``UnknownSpanIds`` when the response names ids we did not issue —
    the caller retries once and then fails the episode. Using ids instead of
    generated text is only worth anything if the ids are actually checked.
    """
    parser_spans = SP.parse_spans(episode_text)
    marked = SP.mark_text(episode_text, parser_spans)
    raw = _complete_json(
        system=CHAIN_SYSTEM_PROMPT, user=marked, model=model,
        schema=CHAIN_SCHEMA, schema_name="coreference_chains",
    )

    issued = {s.sid for s in parser_spans}
    new_spans, dropped_new = SP.locate_new_spans(
        episode_text, raw.get("new_spans", []), issued
    )
    declared = issued | {s.sid for s in new_spans}
    # ids the model declared but we could not verify: referencing them is not a
    # protocol error, so those members are dropped rather than failing the call
    unverified = {d["id"] for d in dropped_new if d.get("id")}

    unknown = sorted({
        sid
        for chain in raw.get("chains", [])
        for sid in [chain.get("head", "")] + list(chain.get("spans", []))
        if sid and sid not in declared and sid not in unverified
    })
    if unknown:
        raise UnknownSpanIds(f"undeclared span ids: {unknown[:8]}")

    table, collapsed = SP.merge_spans(parser_spans, new_spans)
    return {
        "marked_text": marked,
        "parser_spans": parser_spans,
        "new_spans": new_spans,
        "dropped_new_spans": dropped_new,
        "collapsed": collapsed,
        "table": table,
        "raw_chains": raw.get("chains", []),
    }


def resolve_chains(annotated: dict) -> list[list[SP.Span]]:
    """Turn id-chains into span-chains, deduplicated, each with a usable head.

    A chain whose head did not survive validation falls back within the chain
    (proper name -> definite description -> longest) rather than being discarded:
    the grouping is still information even when the model's head choice is not.

    Chains sharing an identical proper name are then unioned deterministically —
    the model splits those often enough that it is not worth trusting it with a
    judgement the text already settles.
    """
    table = annotated["table"]
    chains: list[list[SP.Span]] = []
    for raw in annotated["raw_chains"]:
        members: dict[tuple[int, int], SP.Span] = {}
        for sid in raw.get("spans", []):
            span = table.get(sid)
            if span is not None:  # absent = declared but unverified, drop it
                members[(span.start, span.end)] = span
        if not members:
            continue
        chain = sorted(members.values(), key=lambda s: s.start)
        head = table.get(raw.get("head", ""))
        if head is None or (head.start, head.end) not in members:
            head = SP.pick_head(chain)
        chains.append([head] + [s for s in chain if s is not head])
    return SP.union_identical_names(chains)


def build_candidates(episode_id: str, chains: list[list[SP.Span]]) -> list[dict]:
    """One record per span, addressed by offsets into the immutable episode text.

    Offsets are safe to store here where they were not on a file: episode text
    is content-addressed, so a given episode_id always denotes the same string.
    """
    records: list[dict] = []
    for chain in chains:
        head = chain[0]
        chain_id = E.stable_uuid(f"chain:{episode_id}|{head.start}|{head.end}")
        for span in chain:
            records.append({
                "candidate_id": E.stable_uuid(
                    f"candidate:{episode_id}|{span.start}|{span.end}"
                ),
                "episode_id": episode_id,
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "candidate_type": span.candidate_type,
                "origin": span.origin,
                "chain_id": chain_id,
                "is_head": span is head,
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
    store: dict, episode_id: str, *, model: str, retries: int = 1
) -> tuple[list[dict], dict | None]:
    """Annotate one episode. Returns ``(records, failure)`` — never raises.

    An undeclared span id is a protocol violation, not bad luck: retry once,
    then fail the episode loudly. A failed episode leaves ``extracted_at``
    unset, so the next run picks it up again rather than losing it silently.
    """
    text = store["episodes"][episode_id]["text"]
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            annotated = extract_chains(text, model=model)
        except UnknownSpanIds as exc:
            last = exc
            continue  # the model broke the id contract — one more chance
        except Exception as exc:  # noqa: BLE001 — API failure for this episode
            return [], {"episode_id": episode_id, "stage": "chains",
                        "attempt": attempt, "error": repr(exc)}
        records = build_candidates(episode_id, resolve_chains(annotated))
        return records, None
    return [], {"episode_id": episode_id, "stage": "validate",
                "attempts": retries + 1, "error": repr(last)}


def run(
    store: dict,
    *,
    model: str = EXTRACTION_MODEL,
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
            executor.submit(process_episode, store, eid, model=model): eid
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
        run(store, model=args.model, workers=args.workers,
            limit=args.limit, store_path=args.store)

    E.save_store(args.store, store)
    print(f"\nstore -> {args.store}  "
          f"({len(store['files'])} documents, {len(store['episodes'])} episodes, "
          f"{len(live_candidates(store))} live candidates)")


if __name__ == "__main__":
    main()
