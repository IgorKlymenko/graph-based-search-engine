"""Slack request signing, shared by the ingress and its tests."""

import hashlib
import hmac
import time


def slack_sign(
    body: bytes, signing_secret: bytes, ts: str | None = None
) -> tuple[str, str]:
    """Return ``(timestamp, signature)`` for ``body`` the way Slack computes it.

    The signature covers ``v0:{timestamp}:{raw body}``; the body must be the
    exact bytes on the wire, never a re-serialised JSON document.
    """
    if not ts:
        ts = str(int(time.time()))
    basestring = b"v0:" + ts.encode() + b":" + body
    sig = "v0=" + hmac.new(signing_secret, basestring, hashlib.sha256).hexdigest()
    return ts, sig
