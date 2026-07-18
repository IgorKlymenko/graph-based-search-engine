"""Make pipeline/mentions.py survive a failed chunk, like the other stages.

resolution.py:239 and relations.py:416 both catch API failures per unit of work
to "keep pipeline alive"; mentions.py does not, so one timed-out chunk aborts a
multi-hour run. This applies the same convention, and records what was dropped
in a `failed_chunks` key (mirroring relations.py's cache) so the loss is visible
rather than silent.

Run from the repo root once file access is restored:
    python /private/tmp/.../scratchpad/apply_mentions_resilience.py
It verifies each target matches exactly and refuses to write otherwise.
"""

import pathlib
import sys

TARGET = pathlib.Path("pipeline/mentions.py")

EDITS = [
    # 1. process_file: guard each chunk, collect failures, return them.
    (
        '''def process_file(
    path: Path,
    text: str,
    *,
    model: str,
    enable_gleaning: bool,
    episode_workers: int,
) -> list[dict]:
    """Extract mentions from every chunk of one file, in episode order."""
    chunks = chunk_doc(text, CHUNK_MAX_CHARS)

    def _one(item: tuple[int, Chunk]) -> tuple[int, list[dict]]:
        idx, chunk = item
        first = extract_mentions(chunk.text, model=model)
        mentions = [_tag(m, idx, chunk, str(path), gleaned=False) for m in first]
        if enable_gleaning:
            gleaned = glean_chunk(chunk.text, first, model=model)
            mentions += [_tag(m, idx, chunk, str(path), gleaned=True) for m in gleaned]
        return idx, mentions
''',
        '''def process_file(
    path: Path,
    text: str,
    *,
    model: str,
    enable_gleaning: bool,
    episode_workers: int,
) -> tuple[list[dict], list[dict]]:
    """Extract mentions from every chunk of one file, in episode order.

    Returns the mentions plus a record of any chunk whose API call failed. A
    failed chunk is skipped instead of aborting the run; the record keeps the
    loss visible, since the file is marked processed and never retried.
    """
    chunks = chunk_doc(text, CHUNK_MAX_CHARS)
    failures: list[dict] = []

    def _one(item: tuple[int, Chunk]) -> tuple[int, list[dict]]:
        idx, chunk = item
        try:
            first = extract_mentions(chunk.text, model=model)
        except Exception as exc:  # noqa: BLE001 — API failure per chunk
            failures.append({"source_file": str(path), "episode_idx": idx,
                             "stage": "extract", "error": repr(exc)})
            return idx, []
        mentions = [_tag(m, idx, chunk, str(path), gleaned=False) for m in first]
        if enable_gleaning:
            try:
                gleaned = glean_chunk(chunk.text, first, model=model)
            except Exception as exc:  # noqa: BLE001 — keep the first-pass result
                failures.append({"source_file": str(path), "episode_idx": idx,
                                 "stage": "glean", "error": repr(exc)})
                gleaned = []
            mentions += [_tag(m, idx, chunk, str(path), gleaned=True) for m in gleaned]
        return idx, mentions
''',
    ),
    # 2. process_file return.
    (
        '''    ordered: list[dict] = []
    for idx in sorted(results):
        ordered.extend(results[idx])
    return ordered
''',
        '''    ordered: list[dict] = []
    for idx in sorted(results):
        ordered.extend(results[idx])
    return ordered, failures
''',
    ),
    # 3. cache carries failed_chunks; older caches predate the key.
    (
        '''def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"processed_files": [], "mentions": []}
''',
        '''def load_cache(path: Path) -> dict:
    if path.exists():
        cache = json.loads(path.read_text())
        cache.setdefault("failed_chunks", [])  # caches written before this key
        return cache
    return {"processed_files": [], "mentions": [], "failed_chunks": []}
''',
    ),
    # 4. run(): accumulate failures and surface them.
    (
        '''        for future in as_completed(futures):
            path = futures[future]
            mentions = future.result()
            with lock:
                cache["mentions"].extend(mentions)
                cache["processed_files"].append(str(path))
                _save_cache(cache_path, cache)
            print(f"  {path.name}: {len(mentions)} mentions")

    print(f"  total: {len(cache['mentions'])} mentions "
          f"from {len(cache['processed_files'])} files")
    return cache
''',
        '''        for future in as_completed(futures):
            path = futures[future]
            mentions, failures = future.result()
            with lock:
                cache["mentions"].extend(mentions)
                cache["failed_chunks"].extend(failures)
                cache["processed_files"].append(str(path))
                _save_cache(cache_path, cache)
            failed = f" ({len(failures)} chunks failed)" if failures else ""
            print(f"  {path.name}: {len(mentions)} mentions{failed}")

    print(f"  total: {len(cache['mentions'])} mentions "
          f"from {len(cache['processed_files'])} files")
    if cache["failed_chunks"]:
        print(f"  WARNING: {len(cache['failed_chunks'])} chunks failed and were "
              f"skipped — see 'failed_chunks' in {cache_path}")
    return cache
''',
    ),
]


def main() -> None:
    if not TARGET.exists():
        sys.exit(f"run me from the repo root: {TARGET} not found")
    source = TARGET.read_text()
    if "failed_chunks" in source:
        sys.exit("already applied — mentions.py mentions failed_chunks")

    for old, new in EDITS:
        if source.count(old) != 1:
            sys.exit(f"no unique match, file has changed:\n{old[:90]}...")
        source = source.replace(old, new)

    TARGET.write_text(source)
    print(f"patched {TARGET}: failed chunks are now recorded, not fatal")


if __name__ == "__main__":
    main()
