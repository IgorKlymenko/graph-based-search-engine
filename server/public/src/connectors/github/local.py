"""A git checkout standing in for the GitHub API: same four calls, no network.

    from src.connectors.github.local import LocalRepo
    connector = GitHubConnector(LocalRepo("/path/to/checkout"), ["local/name"], ...)

Blob shas are git's own (``git hash-object``), so a cursor built here and one
built against the API agree about what changed. The repo id is a hash of the
root path -- stable across runs on one machine, not the same as GitHub's.
Citations (``files.url``) point at the ``origin`` remote when it is a GitHub
URL, so a Slack answer links to github.com rather than to a path on one
laptop; with no such remote they fall back to ``file://``.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

_GITHUB_REMOTE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:)([^/]+/[^/]+?)(?:\.git)?/?$"
)


class LocalRepo:
    """The ``connector.Client`` protocol over ``git ls-files`` in ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def repo(self, full_name: str) -> dict:
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:8]
        return {
            "id": int(digest, 16),
            "html_url": self._html_url(),
            "default_branch": self._git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        }

    def _html_url(self) -> str:
        try:
            origin = self._git("remote", "get-url", "origin").strip()
        except subprocess.CalledProcessError:
            return f"file://{self.root}"
        match = _GITHUB_REMOTE.match(origin)
        return (
            f"https://github.com/{match.group(1)}" if match else f"file://{self.root}"
        )

    def head(self, full_name: str, ref: str) -> str:
        """A content hash of the working tree, not the commit.

        The commit sha does not move when a tracked file is edited, so keying
        the cursor on it made every uncommitted change invisible to a re-sync
        (measured: a function added the night before was not in the graph).
        Hashing ``(sha, path)`` over the listing is what a git tree object is.
        """
        listing = "\n".join(f"{e['sha']} {e['path']}" for e in self.tree(full_name, ""))
        return hashlib.sha1(listing.encode()).hexdigest()

    def tree(self, full_name: str, sha: str) -> list[dict]:
        """Files in the working tree, with their blob shas.

        Tracked files plus untracked ones git does not ignore, so uncommitted
        edits and not-yet-added modules are rendered too.
        """
        out = []
        for path in self._git("ls-files", "-co", "--exclude-standard").splitlines():
            file = self.root / path
            if file.is_file():
                out.append(
                    {
                        "path": path,
                        "type": "blob",
                        "sha": self._blob_sha(file),
                        "size": file.stat().st_size,
                    }
                )
        return out

    def blob(self, full_name: str, sha: str) -> bytes:
        for entry in self.tree(full_name, ""):
            if entry["sha"] == sha:
                return (self.root / entry["path"]).read_bytes()
        raise KeyError(sha)

    @staticmethod
    def _blob_sha(file: Path) -> str:
        data = file.read_bytes()
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
