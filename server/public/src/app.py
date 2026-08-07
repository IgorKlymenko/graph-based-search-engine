"""
Minimal Slack events ingress — the shape your production ingress will keep.

Run:
    pip install fastapi uvicorn slack_sdk python-dotenv
    uvicorn slack_events:app --port 8000 --reload

.env:
    SLACK_SIGNING_SECRET=...      # Basic Information -> App Credentials
    SLACK_BOT_TOKEN=xoxb-...      # OAuth & Permissions -> after Install to Workspace
    VERIFY_TIMESTAMP=true         # set false only while replaying via ngrok
"""

from src.security.slack import slack_sign
import json
import os
import time
import hmac
from datetime import datetime

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from slack_sdk import WebClient

load_dotenv()

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
VERIFY_TIMESTAMP = os.environ.get("VERIFY_TIMESTAMP", "true").lower() == "true"

app = FastAPI()
slack = WebClient(token=BOT_TOKEN)

# Stand-in for the Postgres `insert event_id on conflict do nothing`.
# In-memory means it resets on reload — fine for dev, wrong for production.
_seen_event_ids: set[str] = set()


def verify_signature(headers, raw: bytes) -> None:
    """Slack signs the RAW bytes. Never re-serialize the JSON before hashing."""
    signing_secret = os.environ["SLACK_SIGNING_SECRET"].encode()
    timestamp = headers.get("X-Slack-Request-Timestamp", "")
    signature = headers.get("X-Slack-Signature", "")

    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="missing signature headers")

    # Replay guard. ngrok's Replay button resends the ORIGINAL timestamp,
    # so this will reject replayed requests — hence the env flag.
    try:
        timestamp_age = abs(time.time() - int(timestamp))
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid timestamp")

    if VERIFY_TIMESTAMP and timestamp_age > 60 * 5:
        raise HTTPException(status_code=401, detail="stale timestamp")

    _, expected = slack_sign(raw, signing_secret, timestamp)

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="bad signature")


@app.post("/slack/event")
async def slack_event(request: Request, background: BackgroundTasks):
    raw = await request.body()
    verify_signature(request.headers, raw)          # security boundary, first
    payload = json.loads(raw)

    # 1. The handshake. Must return a BODY, so it branches before the ACK path.
    if payload.get("type") == "url_verification":
        return PlainTextResponse(payload["challenge"])

    # 2. Slack retried a delivery we already have. ACK, do nothing.
    event_id = payload.get("event_id")
    if event_id is not None:
        if event_id in _seen_event_ids:
            return JSONResponse({})
        _seen_event_ids.add(event_id)

    event = payload.get("event", {})

    # 3. Never react to ourselves — otherwise DMs become an infinite loop.
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return JSONResponse({})

    # 4. Hand off, then ACK inside the 3s budget.
    #    BackgroundTasks stands in for the Cloud Tasks enqueue. Same boundary:
    #    everything slow lives on the far side of the ACK.
    if event.get("type") in ("app_mention", "message"):
        background.add_task(handle_event, event)

    return JSONResponse({})


def handle_event(event: dict) -> None:
    """Where tenant lookup -> CSR -> PPR -> OpenAI will go. Echoes for now."""
    text = event.get("text", "")
    slack.chat_postMessage(
        channel=event["channel"],
        thread_ts=event.get("thread_ts") or event.get("ts"),  # always thread
        text=f"услишал, иди нахй",
    )


@app.get("/health")
def health():
    """Cheap and unauthenticated. Never hang graph loading off this."""
    return {"ok": True}