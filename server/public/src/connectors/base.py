"""The connector contract, and the one door from a connector into the store.

A connector yields ``Document``s: text already rendered the way
``episodes.sync_file`` expects -- one blank-line-separated paragraph per
appendable unit, every bit of markup resolved *before* the text is hashed --
keyed by the source's own stable id. ``sync`` feeds them through ``sync_file``
and retires what the connector says is gone. Nothing else touches the store.

Structured sources know more than prose does. Code, for one, has already
resolved its references: ``sync_file`` in ``main.py`` denotes one function, no
model needs to guess. A ``Document`` therefore carries optional ``mentions``:
offset-addressed occurrences grouped by ``key``, where one key means one thing.
``sync`` cuts them to episodes and stores them as ``seeds`` on each episode;
stage 2 (``chains.merge_seeded``) folds them into the model's chains, so the
symbol becomes a chain -- and later an entity -- that Slack mentions of the same
name can link to through ordinary alias evidence. A mention is never text the
model must re-find; it is offsets the connector already knows.

Adding a source, per ``ADRs/supabase/security.md`` entry 4, is a row in
``connector_installations`` plus: the name in the three ``CHECK (source IN …)``
constraints, an entry in ``episodes.RENDER_VERSIONS``, and a class satisfying
``Connector``. No role, no policy.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import NamedTuple, Protocol

from src.inject import episodes as EP


class Mention(NamedTuple):
    """One occurrence of a name in a document, with what it refers to."""

    start: int
    end: int
    text: str
    key: str  # mentions sharing a key corefer; a qualified name for code


class Document(NamedTuple):
    """One source object, rendered. ``text[m.start:m.end] == m.text`` for every mention."""

    external_id: str
    text: str
    title: str | None = None
    url: str | None = None
    mentions: tuple[Mention, ...] = ()


class Connector(Protocol):
    """What ``sync`` needs from a source. Order matters: ``removed`` after ``documents``."""

    source: str

    def documents(self) -> Iterator[Document]:
        """Every document that is new or changed since the connector's cursor."""
        ...

    def removed(self) -> Iterator[str]:
        """External ids gone since the cursor. Consumed after ``documents``.

        Enumerable sources compute this from absence in their listing; push
        sources take it from explicit delete events. Retiring from absence when
        a listing was partial would drop live documents, so a connector must
        only report what it has positively seen disappear.
        """
        ...


class SyncCounts(NamedTuple):
    documents: int
    changed: int
    episodes_added: int
    episodes_retired: int
    seeded: int


def sync(store: dict, connector: Connector, *, tenant_id: str) -> SyncCounts:
    """Bring ``store`` in line with ``connector``: new and changed in, gone out."""
    counts = {"documents": 0, "changed": 0, "episodes_added": 0,
              "episodes_retired": 0, "seeded": 0}
    for document in connector.documents():
        diff = sync_document(store, document, tenant_id=tenant_id,
                             source=connector.source)
        counts["documents"] += 1
        counts["changed"] += diff.status != "unchanged"
        counts["episodes_added"] += len(diff.added)
        counts["episodes_retired"] += len(diff.removed)
        counts["seeded"] += bool(document.mentions)
    for external_id in connector.removed():
        gone = EP.retire_file(store, EP.file_id_for(
            tenant_id, connector.source, external_id))
        counts["episodes_retired"] += len(gone)
    return SyncCounts(**counts)


def sync_document(store: dict, document: Document, *, tenant_id: str,
                  source: str) -> EP.FileDiff:
    """``sync_file`` plus the seeds, cut to the episodes the split produced."""
    diff = EP.sync_file(
        store, document.text, tenant_id=tenant_id, source=source,
        external_id=document.external_id, title=document.title,
        url=document.url,
    )
    if document.mentions:
        attach_seeds(store, diff.file_id, document)
    return diff


def attach_seeds(store: dict, file_id: str, document: Document) -> int:
    """Store each episode's share of the document's mentions as ``seeds``.

    Re-splits the text with the same function ``sync_file`` used, so offsets
    are cut at exactly the episode boundaries the store holds. An episode whose
    seeds changed while its text did not -- a better parser, say -- is already
    extracted and would keep stale chains forever, so its ``extracted_at`` is
    cleared and stage 2 redoes it. That is the right cost: one call per episode
    the parser improved on.

    Returns the number of episodes whose seeds changed.
    """
    by_episode: dict[str, list[dict]] = {}
    for episode in EP.split_episodes(document.text):
        episode_id = EP.episode_id_for(file_id, episode.text)
        inside = [m for m in document.mentions
                  if episode.char_start <= m.start and m.end <= episode.char_end]
        by_episode.setdefault(episode_id, seeds_for(inside, episode.char_start))

    changed = 0
    for episode_id, seeds in by_episode.items():
        record = store["episodes"].get(episode_id)
        if record is None or record.get("seeds") == seeds:
            continue
        record["seeds"] = seeds
        record["extracted_at"] = None
        changed += 1
    name_seeds(store, document.mentions)
    return changed


# A surface that could be a word in prose: no dot, no underscore, no capital.
_PLAIN_WORD = re.compile(r"^[a-z0-9]+$")


def name_seeds(store: dict, mentions: tuple[Mention, ...]) -> int:
    """Record that a connector's mentions are names, so the judge never asks.

    The name judge (ADR tables 30) reads prose contexts and answers whether
    an author would look something up by that string. Shown
    ``function candidate_pairs (pairing.candidate_pairs)`` it says no to
    seven in ten identifiers -- measured on this repository against the
    author's vault: 421 of 605 judged seed surfaces rejected, 714 seeded
    chains withheld with no name left. A connector resolved the mention; it is
    a name by construction, and a qualified name is nothing else.

    A bare lower-case word (``work``, ``run``) keeps a verdict the prose side
    already gave: in a docstring it is a common noun, and the chain still has
    its qualified name to be known by. Returns the number of verdicts written.
    """
    names = store.setdefault("names", {})
    written = 0
    for mention in mentions:
        text = mention.text
        if _PLAIN_WORD.match(text):
            if text not in names:
                names[text] = True
                written += 1
        elif names.get(text) is not True:
            names[text] = True
            written += 1
    return written


def seeds_for(mentions: list[Mention], offset: int) -> list[dict]:
    """Mentions grouped by key, offsets made episode-relative, deterministic order."""
    grouped: dict[str, list[list]] = {}
    for m in sorted(mentions):
        span = [m.start - offset, m.end - offset, m.text]
        if span not in grouped.setdefault(m.key, []):
            grouped[m.key].append(span)
    return [{"key": key, "spans": grouped[key]} for key in sorted(grouped)]
