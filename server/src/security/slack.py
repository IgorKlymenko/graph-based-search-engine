import time
import hashlib
import hmac
from typing import Optional

def slack_sign(body: bytes, signing_secret: bytes, ts: Optional[str] = None) -> tuple[str, str]:
    if not ts:
        ts = str(int(time.time()))
    basestring = b"v0:" + ts.encode() + b":" + body
    sig = "v0=" + hmac.new(signing_secret, basestring, hashlib.sha256).hexdigest()
    return ts, sig
