"""Rate limits are slept through, pages are followed, a truncated tree refuses."""

import base64
import json

import httpx
import pytest

from src.connectors.github.client import GitHub, GitHubError


def make(responses):
    """A client whose transport pops canned responses; records sleeps."""
    slept: list[float] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        status, body, headers = queue.pop(0)
        return httpx.Response(status, json=body, headers=headers)

    client = GitHub("tok", transport=httpx.MockTransport(handler),
                    sleep=slept.append)
    return client, slept


def test_a_secondary_limit_waits_retry_after_then_retries() -> None:
    client, slept = make([(403, {"message": "slow down"}, {"retry-after": "7"}),
                          (200, {"login": "igor"}, {})])
    assert client.whoami() == {"login": "igor"}
    assert slept == [7.0]


def test_a_primary_limit_waits_until_reset() -> None:
    import time

    client, slept = make([(429, {}, {"x-ratelimit-remaining": "0",
                                    "x-ratelimit-reset": str(int(time.time()) + 30)}),
                          (200, {"sha": "abc"}, {})])
    assert client.head("a/b", "main") == "abc"
    assert 25 <= slept[0] <= 32


def test_a_plain_403_is_an_error_not_a_wait() -> None:
    client, slept = make([(403, {"message": "Resource not accessible"}, {})])
    with pytest.raises(GitHubError, match="403 /user: Resource not accessible"):
        client.whoami()
    assert slept == []


def test_server_errors_are_retried_with_backoff_then_raised() -> None:
    client, slept = make([(502, {}, {})] * 4)
    with pytest.raises(GitHubError, match="502"):
        client.repo("a/b")
    assert slept == [2.0, 4.0, 8.0]


def test_pages_follow_the_link_header() -> None:
    client, _ = make([
        (200, [1, 2], {"link": '<https://api.github.com/x?page=2>; rel="next"'}),
        (200, [3], {}),
    ])
    assert list(client.paged("/x")) == [1, 2, 3]


def test_a_truncated_tree_is_refused() -> None:
    """Half a listing would read as mass deletion to the diff."""
    client, _ = make([(200, {"tree": [], "truncated": True}, {})])
    with pytest.raises(GitHubError, match="truncated"):
        client.tree("a/b", "sha")


def test_blobs_are_decoded_from_base64() -> None:
    content = base64.b64encode(b"print(1)\n").decode()
    client, _ = make([(200, {"encoding": "base64", "content": content}, {})])
    assert client.blob("a/b", "sha") == b"print(1)\n"


def test_the_request_carries_the_token_and_api_version() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={})

    GitHub("tok", transport=httpx.MockTransport(handler)).whoami()
    assert seen["authorization"] == "Bearer tok"
    assert seen["x-github-api-version"]
    assert json.loads("{}") == {}
