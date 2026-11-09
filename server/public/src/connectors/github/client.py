"""The smallest GitHub REST client that syncs a repository, with the limits handled.

Four endpoints: the repository, a commit for a ref, the recursive git tree at
that commit, and a blob by sha. Trees and blobs are content-addressed on
GitHub's side, which is what makes the connector's diff cheap: one tree call
tells us every path's blob sha, and only changed shas are fetched.

Rate limits are honoured rather than raced. A primary limit answers 403 or 429
with ``x-ratelimit-remaining: 0`` and a reset epoch; a secondary limit sends
``retry-after``. Both are slept through and the request retried, because the
alternative -- failing the sync -- costs a full restart next run. Numbers here
are GitHub policy and *unverified* against current docs; see the ADR.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

API = "https://api.github.com"
API_VERSION = "2022-11-28"
PAGE = 100
MAX_SLEEP = 3600.0  # a primary-limit reset is at most an hour away
RETRIES_5XX = 3


class GitHubError(RuntimeError):
    """GitHub answered with an error the client will not retry."""


class GitHub:
    """Bearer-token client. ``transport`` and ``sleep`` are injectable for tests.

    An empty token sends no ``Authorization`` header: public repositories only,
    at GitHub's unauthenticated rate *(unverified: 60 requests an hour)*. That
    is enough to try the connector on a small public repo and nothing else.
    """

    def __init__(self, token: str = "", *,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        headers = {"Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": API_VERSION}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=API, timeout=30.0,
                                    transport=transport, headers=headers)
        self._sleep = sleep

    def get(self, path: str, **params: Any) -> httpx.Response:
        """One GET, retried through rate limits and transient server errors."""
        failures = 0
        while True:
            response = self._client.get(path, params=params or None)
            wait = _rate_limit_wait(response)
            if wait is not None:
                print(f"  github: rate limited, sleeping {wait:.0f}s")
                self._sleep(min(wait, MAX_SLEEP))
                continue
            if response.status_code >= 500 and failures < RETRIES_5XX:
                failures += 1
                self._sleep(2.0 ** failures)
                continue
            if response.status_code >= 400:
                raise GitHubError(f"{response.status_code} {path}: "
                                  f"{_message(response)}")
            return response

    def json(self, path: str, **params: Any) -> Any:
        return self.get(path, **params).json()

    def paged(self, path: str, **params: Any) -> Iterator[dict]:
        """Follow ``Link: <…>; rel="next"`` until it is gone."""
        response = self.get(path, per_page=PAGE, **params)
        while True:
            yield from response.json()
            nxt = response.links.get("next", {}).get("url")
            if not nxt:
                return
            response = self.get(nxt)

    def whoami(self) -> dict:
        """Preflight: fails here with a name attached, not mid-sync.

        ``/user`` needs a token; without one the rate-limit resource stands in,
        so a bad network or a blocked host still fails before the sync starts.
        """
        if "Authorization" not in self._client.headers:
            return {"login": "anonymous", **self.json("/rate_limit")}
        return self.json("/user")

    def repo(self, full_name: str) -> dict:
        return self.json(f"/repos/{full_name}")

    def head(self, full_name: str, ref: str) -> str:
        """The commit sha a branch, tag or sha resolves to right now."""
        return self.json(f"/repos/{full_name}/commits/{ref}")["sha"]

    def tree(self, full_name: str, sha: str) -> list[dict]:
        """Every entry under the commit, flat. Refuses a truncated listing.

        GitHub truncates recursive trees past its size limit *(unverified:
        100,000 entries or 7 MB)* and says so with ``truncated: true``. A
        partial tree would read as mass deletion to the connector's diff, so it
        is an error, not a warning.
        """
        payload = self.json(f"/repos/{full_name}/git/trees/{sha}", recursive="1")
        if payload.get("truncated"):
            raise GitHubError(f"tree for {full_name}@{sha[:8]} is truncated; "
                              "sync by subtree is not implemented")
        return payload.get("tree", [])

    def blob(self, full_name: str, sha: str) -> bytes:
        payload = self.json(f"/repos/{full_name}/git/blobs/{sha}")
        if payload.get("encoding") != "base64":
            raise GitHubError(f"blob {sha[:8]}: encoding {payload.get('encoding')!r}")
        return base64.b64decode(payload.get("content", ""))


def _rate_limit_wait(response: httpx.Response) -> float | None:
    """Seconds to wait if this response is a rate limit, else None."""
    if response.status_code not in (403, 429):
        return None
    retry_after = response.headers.get("retry-after")
    if retry_after:
        return max(1.0, float(retry_after))
    if response.headers.get("x-ratelimit-remaining") == "0":
        reset = float(response.headers.get("x-ratelimit-reset", "0"))
        return max(1.0, reset - time.time() + 1.0)
    return None


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("message", response.text[:200]))
    except ValueError:
        return response.text[:200]
