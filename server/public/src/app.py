"""
Slack events ingress: verify, dedupe, ACK; answer in the thread afterwards.

Run (from server/public):
    uv run uvicorn src.app:app --port 8000 --reload

.env:
    SLACK_SIGNING_SECRET=...        # Basic Information -> App Credentials
    SLACK_BOT_TOKEN=xoxb-...        # OAuth & Permissions -> after Install to Workspace
    CONNECTOR_DATABASE_URL=...      # app_connector_login; falls back to DATABASE_URL
    ANSWER_MODEL=gpt-4.1            # optional
    VERIFY_TIMESTAMP=true           # set false only while replaying via ngrok

An @mention is answered from the workspace's indexed channels by the query path
(src/query): retrieval runs as app_connector for the event's team_id, so the
database resolves the tenant, and the reply may say the question is the wrong
one to ask (ADR slack 16). Message text is never logged.
"""

import hmac
import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from slack_sdk import WebClient

from src.query import answer as AN
from src.query import thread as TH
from src.query.retrieve import answer_question
from src.security.slack import slack_sign

load_dotenv()

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
DATABASE_URL = os.environ.get("CONNECTOR_DATABASE_URL") or os.environ.get(
    "DATABASE_URL", ""
)
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", AN.ANSWER_MODEL)
BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID", "")
log = logging.getLogger("ingress")
# uvicorn wires handlers for its own loggers only; without this the answer
# trail below never reaches stdout.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s"
    )
log.setLevel(logging.INFO)
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
    verify_signature(request.headers, raw)  # security boundary, first
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
    #    everything slow lives on the far side of the ACK. Mentions anywhere
    #    (ADR slack 13) and replies in threads Nazar is in (ADR slack 17);
    #    every other subscribed message is ACKed and dropped here.
    team_id = payload.get("team_id", "")
    if event.get("type") == "app_mention":
        background.add_task(handle_event, event, team_id)
    elif (
        event.get("type") == "message"
        and event.get("thread_ts")
        and not event.get("subtype")
        and event.get("channel_type") == "channel"
    ):
        # A reply in a thread: answered only if Nazar is already in it (ADR slack 17).
        background.add_task(handle_thread_reply, event, team_id)
    else:
        log.info(
            "ingress: dropped type=%s subtype=%s thread=%s",
            event.get("type"),
            event.get("subtype"),
            bool(event.get("thread_ts")),
        )

    return JSONResponse({})


NOTHING_FOUND = (
    "I found nothing in the indexed notes and code about that. "
    "Try naming the file, function, person or project you mean."
)
NOT_A_QUESTION = (
    "Ask me something and I'll look through the notes and code connected to this "
    "workspace, with sources."
)


def bot_user_id() -> str:
    """The bot's own user id, from the env or one auth.test call, cached."""
    global BOT_USER_ID
    if not BOT_USER_ID:
        try:
            BOT_USER_ID = slack.auth_test()["user_id"]
        except Exception:
            log.exception("auth.test failed; bot user id unknown")
    return BOT_USER_ID


def question_from(text: str) -> str:
    """The question in a message: markup resolved, every mention gone."""
    return TH.plain_text(TH._MENTION.sub(" ", text))


def handle_thread_reply(event: dict, team_id: str) -> None:
    """A reply in a thread: join the conversation only if Nazar is already in it."""
    turns = TH.thread_context(
        slack,
        event["channel"],
        event["thread_ts"],
        bot_user_id=bot_user_id(),
        before_ts=event.get("ts"),
    )
    if not TH.bot_in_thread(turns):
        return
    handle_event(event, team_id, turns)


def handle_event(event: dict, team_id: str, turns: list | None = None) -> None:
    """Post-ACK work: retrieve as this workspace, decide a stance, reply in thread.

    Runs as app_connector for ``team_id``, so an unknown or uninstalled
    workspace resolves to no tenant and simply finds nothing. Inside a thread
    the earlier turns are read back and the message is condensed into a
    standalone question first (src/query/thread.py). Failures post a neutral
    fallback; the exception is logged without the message text, which is
    customer content.
    """
    question = question_from(event.get("text", ""))
    verdict = None
    blocks = None
    if not question:
        text = NOT_A_QUESTION
    else:
        try:
            if turns is None and event.get("thread_ts"):
                turns = TH.thread_context(
                    slack,
                    event["channel"],
                    event["thread_ts"],
                    bot_user_id=bot_user_id(),
                    before_ts=event.get("ts"),
                )
            condensed = TH.condense(question, turns or [])
            verdict = answer_question(
                condensed["question"],
                dsn=DATABASE_URL,
                connector="slack",
                workspace=team_id,
                model=ANSWER_MODEL,
                conversation=turns or None,
                latest=question,
                needs_lookup=condensed["needs_lookup"],
            )
        except Exception:
            log.exception("answer failed for workspace %s", team_id)
            verdict = None
        if verdict:
            text, blocks = AN.render_slack_blocks(
                verdict, links=verdict.get("retrieval", {}).get("links")
            )
        else:
            text, blocks = NOTHING_FOUND, None
    posted = slack.chat_postMessage(
        channel=event["channel"],
        thread_ts=event.get("thread_ts") or event.get("ts"),  # always thread
        text=text,  # notification text and the fallback for clients without blocks
        blocks=blocks,
        unfurl_links=False,
        unfurl_media=False,
    )
    # Operational trail without customer content: which workspace and channel,
    # what stance, and the message ts so the reply can be found in Slack.
    stance = verdict["stance"] if verdict else ("nothing" if question else "empty")
    if turns:
        stance += f" turns={len(turns)} lookup={(verdict or {}).get('retrieval', {}).get('lookup', '-')}"
    log.info(
        "answered workspace=%s channel=%s stance=%s ts=%s",
        team_id,
        event.get("channel"),
        stance,
        posted.get("ts") if hasattr(posted, "get") else "",
    )


@app.get("/health")
def health():
    """Cheap and unauthenticated. Never hang graph loading off this."""
    return {"ok": True}
