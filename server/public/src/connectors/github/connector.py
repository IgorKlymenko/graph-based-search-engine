"""Repositories -> documents: code through the parsers, markdown through strip_markup.

What a GitHub document is:

    external_id   repo/{repo_id}/{path}      the numeric repo id survives renames
    title         {owner}/{name}/{path}
    url           {html_url}/blob/{ref}/{path}

Deletion is inferred from absence -- GitHub is enumerable, one tree call lists
the whole repo -- but only within a repo whose tree was fetched whole. A repo
that errors is skipped entirely, so nothing of it is retired.

The cursor per repo, kept in the store under ``connectors.github``, is the head
commit and every path's blob sha at the last sync, plus the render version
that produced them. Same head, same version: the repo costs one request. New
head: one tree call, then blobs only for paths whose sha changed. A render
version bump refetches everything, because every hash it produced is stale.

Code is rendered in two phases because one file's rendering depends on the
whole repo: a short name defined twice renders qualified (``Style.ambiguous``,
ADR connectors 11). The cursor therefore also keeps each path's names
(``names_of``), so the ambiguous set is recomputed from the map, and a file
whose content did not change but whose names flipped ambiguity is fetched and
rendered again. The synthetic index document (``index_document``, off by
default) lists every colliding name's definitions so the linker sees them kept
apart; measured, it adds review-band pairs without removing false merges that
the rendering rule has not already removed.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Protocol

from src.connectors import base
from src.connectors.code import (
    Parsed,
    Style,
    ambiguous_from,
    index_document,
    names_of,
    parser_for,
    render,
)
from src.inject import episodes as EP

SOURCE = "github"
INDEX_PATH = ".kg-index"  # no real file starts with .kg-
DEFAULT_REF = "HEAD"
MAX_BLOB_BYTES = 200_000  # generated and vendored files are the big ones
PROSE_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt"})
# Directories that hold other people's code or build output. Indexing them
# grows the graph with symbols nobody in the company wrote or will ask about.
SKIP_DIRS = frozenset({
    "node_modules", "vendor", "dist", "build", ".venv", "venv", "site-packages",
    "__pycache__", ".git", ".github", "third_party", "generated",
})


class Client(Protocol):
    """The slice of ``client.GitHub`` the connector needs; tests fake it."""

    def repo(self, full_name: str) -> dict: ...
    def head(self, full_name: str, ref: str) -> str: ...
    def tree(self, full_name: str, sha: str) -> list[dict]: ...
    def blob(self, full_name: str, sha: str) -> bytes: ...


def external_id_for(repo_id: int, path: str) -> str:
    return f"repo/{repo_id}/{path}"


def wanted(path: str, size: int | None) -> bool:
    """Is this path something we render? Parsers and prose, outside skip dirs."""
    parts = PurePosixPath(path).parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return False
    if size is not None and size > MAX_BLOB_BYTES:
        return False
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in PROSE_SUFFIXES or parser_for(path) is not None


def decode(raw: bytes) -> str | None:
    """Text, or None for a binary blob."""
    if b"\x00" in raw[:8000]:
        return None
    return raw.decode("utf-8", errors="replace")


def parse_code(path: str, text: str, *, tree: frozenset[str]) -> Parsed | None:
    """Symbols, or None when the language's parser rejects the file."""
    parser = parser_for(path)
    if parser is None:
        return None
    try:
        return parser.parse(path, text, tree=tree)
    except (SyntaxError, ValueError, RecursionError):
        return None


def render_file(path: str, raw: bytes, *, tree: frozenset[str] = frozenset(),
                ) -> tuple[str, tuple[base.Mention, ...]] | None:
    """Text and mentions for one file on its own, or None when binary or unparseable.

    The connector renders code repo-wide (ambiguity needs every file); this is
    the single-file form for prose, tests and quick looks.
    """
    text = decode(raw)
    if text is None:
        return None
    if parser_for(path) is None:
        return EP.strip_markup(text), ()
    parsed = parse_code(path, text, tree=tree)
    return None if parsed is None else render(parsed, path=path)


class GitHubConnector:
    """Yields a ``Document`` per new or changed file across ``repos``."""

    source = SOURCE

    def __init__(self, client: Client, repos: list[str], *, cursors: dict,
                 ref: str = DEFAULT_REF, include: tuple[str, ...] = (),
                 style: Style | None = None, index: bool = False) -> None:
        self._client = client
        self._repos = repos
        self._cursors = cursors  # store["connectors"]["github"], keyed by repo id
        self._ref = ref
        self._style = style or Style()
        self._index = index
        # Path prefixes to index; empty means the whole repo. A monorepo's
        # legacy half is not the company's current knowledge. Paths outside the
        # prefixes are treated exactly like unwanted files: never rendered,
        # never remembered, so narrowing later retires them by absence.
        self._include = tuple(p.strip("/") + "/" for p in include if p.strip("/"))
        self._removed: list[str] = []
        self.counts = {"repos": 0, "unchanged_repos": 0, "files": 0,
                       "skipped": 0, "unparsed": 0}

    def documents(self) -> Iterator[base.Document]:
        self._removed = []
        for full_name in self._repos:
            yield from self._repo_documents(full_name)

    def removed(self) -> Iterator[str]:
        yield from self._removed

    def _repo_documents(self, full_name: str) -> Iterator[base.Document]:
        repo = self._client.repo(full_name)
        repo_id, html_url = int(repo["id"]), repo["html_url"]
        ref = repo["default_branch"] if self._ref == DEFAULT_REF else self._ref
        head = self._client.head(full_name, ref)
        version = EP.render_version_for(SOURCE)
        cursor = self._cursors.get(str(repo_id), {})
        self.counts["repos"] += 1
        if (cursor.get("head") == head and cursor.get("render_version") == version
                and cursor.get("include", []) == list(self._include)
                and cursor.get("html_url", html_url) == html_url):
            self.counts["unchanged_repos"] += 1
            return

        same_version = cursor.get("render_version") == version
        # Deletion is judged against every path the last sync rendered, whatever
        # version rendered it; `before` only decides what can be skipped. A new
        # citation base (the remote changed) touches every file's url, which
        # only a re-emit carries, so nothing is skipped then either.
        previous = set(cursor.get("blobs", {}))
        same_base = cursor.get("html_url", html_url) == html_url
        before = cursor.get("blobs", {}) if same_version and same_base else {}
        symbols: dict[str, dict] = dict(cursor.get("symbols", {})) if same_version else {}
        was_ambiguous = frozenset(cursor.get("ambiguous", [])) if same_version else frozenset()

        blobs: dict[str, str] = {}
        parsed: dict[str, Parsed] = {}
        entries = self._client.tree(full_name, head)
        paths = frozenset(e["path"] for e in entries)
        for entry in entries:
            path = entry["path"]
            if entry.get("type") != "blob" or not wanted(path, entry.get("size")):
                continue
            if self._include and not self._included(path):
                continue
            blobs[path] = entry["sha"]
            if before.get(path) == entry["sha"]:
                continue
            text = decode(self._client.blob(full_name, entry["sha"]))
            if text is None:
                self.counts["unparsed"] += 1
                del blobs[path]  # not rendered, so absence next time is not a change
                continue
            if parser_for(path) is None:
                yield from self._emit(repo_id, full_name, html_url, ref, path,
                                      EP.strip_markup(text), ())
                continue
            code = parse_code(path, text, tree=paths)
            if code is None:
                self.counts["unparsed"] += 1
                del blobs[path]
                continue
            parsed[path] = code
            symbols[path] = names_of(code, package=PurePosixPath(path).stem == "__init__")

        for path in [p for p in symbols if p not in blobs]:
            del symbols[path]
        ambiguous = ambiguous_from(symbols.values())
        flipped = ambiguous ^ was_ambiguous
        # Content unchanged, rendering not: a name it shows became (or stopped
        # being) ambiguous. Fetched again by its known sha.
        for path, names in symbols.items():
            if path in parsed or not flipped & _names_shown(names):
                continue
            text = decode(self._client.blob(full_name, blobs[path]))
            code = parse_code(path, text or "", tree=paths) if text else None
            if code is not None:
                parsed[path] = code
        style = self._style._replace(ambiguous=ambiguous)
        for path, code in parsed.items():
            text, mentions = render(code, path=path, style=style)
            yield from self._emit(repo_id, full_name, html_url, ref, path, text, mentions)

        index_before = cursor.get("index", False)
        index_now = False
        if self._index:
            text, mentions = index_document(symbols.values(), ambiguous)
            index_now = bool(text)
            if index_now:
                yield base.Document(external_id_for(repo_id, INDEX_PATH), text,
                                    title=f"{full_name}: names defined more than once",
                                    mentions=mentions)

        self._removed.extend(external_id_for(repo_id, path)
                             for path in sorted(previous) if path not in blobs)
        if index_before and not index_now:
            self._removed.append(external_id_for(repo_id, INDEX_PATH))
        self._cursors[str(repo_id)] = {
            "full_name": full_name, "head": head, "render_version": version,
            "blobs": blobs, "symbols": symbols, "ambiguous": sorted(ambiguous),
            "index": index_now, "include": list(self._include), "html_url": html_url,
        }

    def _included(self, path: str) -> bool:
        """Under one of the prefixes, or the prefix names this very file (``CLAUDE.md``)."""
        return any(path.startswith(p) or path == p[:-1] for p in self._include)

    def _emit(self, repo_id: int, full_name: str, html_url: str, ref: str,
              path: str, text: str, mentions: tuple[base.Mention, ...]):
        if not text.strip():
            return
        self.counts["files"] += 1
        yield base.Document(
            external_id=external_id_for(repo_id, path), text=text,
            title=f"{full_name}/{path}", url=f"{html_url}/blob/{ref}/{path}",
            mentions=mentions,
        )


def _names_shown(names: dict) -> set[str]:
    return set(names["defs"]) | set(names["refs"])


def cursors_in(store: dict) -> dict:
    """The connector's own corner of the store."""
    return store.setdefault("connectors", {}).setdefault(SOURCE, {})


def _local_client(root: Path) -> Client:
    from src.connectors.github.local import LocalRepo

    client = LocalRepo(root)
    print(f"local checkout {client.root}")
    return client


def _api_client() -> Client:
    from src.connectors.github.client import GitHub, GitHubError

    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        print("GITHUB_TOKEN is not set: public repositories only, "
              "unauthenticated rate limit")
    client = GitHub(token)
    try:
        who = client.whoami()
    except GitHubError as error:
        raise SystemExit(f"{error}\n  a 401 means the token is wrong or expired; "
                         "a fine-grained token needs Contents: read on the repos") \
            from None
    print(f"authenticated as {who.get('login')}")
    return client


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repos", nargs="+", required=True,
                        help="owner/name, one or more")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help="branch, tag or sha; default: the default branch")
    parser.add_argument("--include", nargs="*", default=(),
                        help="path prefixes to index; default: the whole repo")
    parser.add_argument("--local", type=Path, default=None,
                        help="a git checkout to read instead of the API "
                             "(uncommitted edits included); --repos names it")
    args = parser.parse_args()

    client: Client = _local_client(args.local) if args.local else _api_client()

    store = EP.load_store(args.store)
    connector = GitHubConnector(client, args.repos, cursors=cursors_in(store),
                                ref=args.ref, include=tuple(args.include))
    counts = base.sync(store, connector, tenant_id=args.tenant)
    EP.save_store(args.store, store)
    print("  ".join(f"{k}={v}" for k, v in connector.counts.items()))
    print("  ".join(f"{k}={v}" for k, v in counts._asdict().items()))
    print(f"store -> {args.store}")


if __name__ == "__main__":
    main()
