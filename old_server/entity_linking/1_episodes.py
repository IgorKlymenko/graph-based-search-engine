"""Splitting documents into episodes, and the store that tracks them across edits.

Two layers:

*Splitting* — greedy paragraph merging into ~``max_chars`` slices. Every episode
is an exact slice of the source text, so ``doc.find(text)`` recovers its position
and no offsets need storing.

*Identity* — a document is identified by where it came from, an episode by its
content:

    file_id    = uuid5(NS, "file:<tenant_id>|<source>|<external_id>")
    episode_id = uuid5(NS, "episode:<file_id>|<sha1(text)>")

A document is a container that gets revised, so it keeps its identity across
edits. An episode is a unit of content, so it keeps its identity when an edit
above it moves it — keying episodes on offsets would renumber every episode
below any insertion, even ones whose text never changed.

``tenant_id`` and ``source`` are part of file identity because external ids are
only unique within a workspace (the same note path, Slack channel id or HubSpot
object id can exist in two tenants), and because one tenant may connect several
systems that number their objects independently.

Byte-identical episodes within a document collapse to one record.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# Same namespace as graph_upload.stable_uuid — two namespaces drifting apart
# would silently split every id, so there must only ever be one.
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# A connector renders its source objects to plain text; that rendering feeds
# full_text_hash and every episode hash, so changing it silently would re-extract
# the corpus with no record of why. Bump the entry for the connector you changed
# — per-source, so a Slack rendering change does not invalidate fs or HubSpot.
RENDER_VERSIONS: dict[str, int] = {
    "upload": 1,    # read the file as text; near-identity
    "slack": 1,     # thread -> text: which of author/ts/reactions/joins appear
    "gdrive": 1,
    "notion": 1,
}
DEFAULT_RENDER_VERSION = 1


def render_version_for(source: str) -> int:
    return RENDER_VERSIONS.get(source, DEFAULT_RENDER_VERSION)


def stable_uuid(key: str) -> str:
    """Deterministic UUID from a string key — idempotent across runs."""
    return str(uuid.uuid5(_UUID_NS, key))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def rel_path(path: Path, root: Path) -> str:
    """Path relative to the corpus root, forward-slashed — the fs connector's
    ``external_id``.

    Ids must not depend on how the CLI was invoked: the same corpus addressed by
    an absolute vs relative root has to produce the same file_id, so
    normalisation happens here and nowhere else.
    """
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_id_for(tenant_id: str, source: str, external_id: str) -> str:
    """Identity of one source document, scoped to its tenant and system.

    ``external_id`` is whatever the source system calls the object — a relative
    path for files, ``<channel>/<thread_ts>`` for a Slack thread, an object id
    for HubSpot or Granola. Prefer the system's own id over anything synthesised:
    it is stable across renames, which a path is not.
    """
    return stable_uuid(f"file:{tenant_id}|{source}|{external_id}")


# ── splitting ─────────────────────────────────────────────────────────────────

class Episode(NamedTuple):
    """A ~``max_chars`` slice of a document with its source offsets."""

    text: str
    char_start: int
    char_end: int


def _paragraph_spans(doc: str) -> list[tuple[int, int]]:
    """Offsets ``(start, end)`` of each non-blank paragraph in ``doc``.

    Splits on blank lines (``"\\n\\n"``) while tracking absolute offsets.
    Leading/trailing whitespace is trimmed so ``doc[start:end]`` begins and ends
    on real content.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in doc.split("\n\n"):
        seg_start = cursor
        cursor = seg_start + len(part) + 2  # +2 for the removed "\n\n"
        if not part.strip():
            continue
        lead = len(part) - len(part.lstrip())
        trail = len(part) - len(part.rstrip())
        spans.append((seg_start + lead, seg_start + len(part) - trail))
    return spans


def split_episodes(doc: str, max_chars: int = 1000) -> list[Episode]:
    """Split ``doc`` into ~``max_chars`` episodes, greedily merging paragraphs.

    A single paragraph longer than ``max_chars`` becomes its own episode — the
    limit is a merge threshold, not a hard cap. Each returned episode's ``text``
    equals ``doc[char_start:char_end]`` exactly.
    """
    episodes: list[Episode] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end in _paragraph_spans(doc):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= max_chars:
            cur_end = end
        else:
            episodes.append(Episode(doc[cur_start:cur_end], cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        episodes.append(Episode(doc[cur_start:cur_end], cur_start, cur_end))
    return episodes


# ── episode records ───────────────────────────────────────────────────────────

def build_records(file_id: str, doc: str, max_chars: int = 1000) -> list[dict]:
    """Split ``doc`` and attach identity to each episode.

    The record carries no offsets: an episode is a verbatim slice, so
    ``doc.find(text)`` recovers its position exactly. Offsets would describe a
    *version* of the document while ``episode_id`` and ``text`` describe content.

    Records come back in reading order, and ``files.episode_ids`` preserves that
    order, so a consumer holding only an episode id can still find its neighbours
    by position in that list. Nothing on the record encodes order — a stored
    position would be version-dependent while the rest of the record is not.

    Repeated identical text yields one record, at its first occurrence.
    """
    seen: set[str] = set()
    records: list[dict] = []
    for ep in split_episodes(doc, max_chars):
        episode_id = stable_uuid(f"episode:{file_id}|{sha1_text(ep.text)}")
        if episode_id in seen:
            continue
        seen.add(episode_id)
        records.append({
            "episode_id": episode_id,
            "file_id": file_id,
            "text": ep.text,
        })
    return records


# ── store ─────────────────────────────────────────────────────────────────────

class FileDiff(NamedTuple):
    """Outcome of syncing one source document against the store."""

    file_id: str
    source: str
    external_id: str
    status: str  # "unchanged" | "new" | "updated"
    unchanged: list[str]
    added: list[str]
    removed: list[str]

    @property
    def needs_work(self) -> list[str]:
        """Episode ids that have no extraction yet."""
        return self.added


def empty_store() -> dict:
    return {"files": {}, "episodes": {}, "candidates": {}, "chains": {}}


def load_store(path: Path) -> dict:
    if not path.exists():
        return empty_store()
    store = json.loads(path.read_text())
    store.setdefault("files", {})
    store.setdefault("episodes", {})
    store.setdefault("candidates", {})
    store.setdefault("chains", {})
    return store


def save_store(path: Path, store: dict) -> None:
    path.write_text(json.dumps(store, ensure_ascii=False, indent=1))


def sync_file(
    store: dict,
    doc: str,
    *,
    tenant_id: str,
    source: str,
    external_id: str,
    max_chars: int = 1000,
) -> FileDiff:
    """Reconcile one source document's episodes into ``store``; return the diff.

    ``doc`` is the document already rendered to text — the connector owns that
    rendering, and ``RENDER_VERSIONS[source]`` records which rendering produced
    the hashes, since changing it invalidates every one of them.

    Cheap path first: an unchanged full_text_hash means the split is skipped.
    Otherwise episodes are re-derived and diffed by id, so an edit only marks the
    episodes whose *text* changed — not everything below the edit.

    Removed episodes are kept with a ``removed_at`` stamp rather than deleted,
    so whatever consumes this store can retire their downstream products.
    """
    fid = file_id_for(tenant_id, source, external_id)
    digest = sha1_text(doc)
    known = store["files"].get(fid)

    if (
        known
        and known.get("full_text_hash") == digest
        and known.get("render_version") == render_version_for(source)
    ):
        return FileDiff(
            fid, source, external_id, "unchanged",
            list(known["episode_ids"]), [], [],
        )

    records = build_records(fid, doc, max_chars)
    new_ids = [r["episode_id"] for r in records]
    prev_ids = list(known["episode_ids"]) if known else []
    prev = set(prev_ids)
    new = set(new_ids)

    for record in records:
        existing = store["episodes"].get(record["episode_id"])
        if existing is not None:
            # same text, possibly moved: position changes, identity does not, so
            # the episode is only un-retired and never re-extracted
            existing["removed_at"] = None
        else:
            store["episodes"][record["episode_id"]] = {
                **record, "extracted_at": None, "removed_at": None,
            }

    now = datetime.now(timezone.utc).isoformat()
    removed = [eid for eid in prev_ids if eid not in new]
    for eid in removed:
        episode = store["episodes"].get(eid)
        if episode is not None:
            episode["removed_at"] = now

    store["files"][fid] = {
        "file_id": fid,
        "tenant_id": tenant_id,
        "source": source,
        "external_id": external_id,
        "full_text_hash": digest,
        "render_version": render_version_for(source),
        "episode_ids": new_ids,
        "synced_at": now,
        "removed_at": None,
    }
    return FileDiff(
        file_id=fid,
        source=source,
        external_id=external_id,
        status="new" if known is None else "updated",
        unchanged=[eid for eid in new_ids if eid in prev],
        added=[eid for eid in new_ids if eid not in prev],
        removed=removed,
    )


UPLOAD_SOURCE = "upload"


def sync_corpus(
    store: dict,
    notes_dir: Path,
    *,
    tenant_id: str,
    source: str = UPLOAD_SOURCE,
    max_chars: int = 1000,
) -> list[FileDiff]:
    """Upload connector: sync every ``*.md`` under ``notes_dir``.

    Deletion is inferred from absence, which only works for sources that can be
    enumerated cheaply. Push-based connectors (Slack, HubSpot webhooks) call
    ``sync_file`` per delta and must retire documents from explicit events
    instead.
    """
    diffs: list[FileDiff] = []
    seen_files: set[str] = set()
    for path in sorted(notes_dir.rglob("*.md")):
        try:
            doc = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not doc.strip():
            continue
        external_id = rel_path(path, notes_dir)
        seen_files.add(file_id_for(tenant_id, source, external_id))
        diffs.append(sync_file(
            store, doc, tenant_id=tenant_id, source=source,
            external_id=external_id, max_chars=max_chars,
        ))

    now = datetime.now(timezone.utc).isoformat()
    for fid, meta in store["files"].items():
        if meta.get("tenant_id") != tenant_id or meta.get("source") != source:
            continue  # another tenant or connector owns this document
        if fid in seen_files or meta.get("removed_at"):
            continue
        meta["removed_at"] = now
        gone = list(meta["episode_ids"])
        for eid in gone:
            episode = store["episodes"].get(eid)
            if episode is not None:
                episode["removed_at"] = now
        diffs.append(FileDiff(
            fid, meta["source"], meta["external_id"], "updated", [], [], gone,
        ))
    return diffs
