"""Threads are conversations: read the thread, condense the follow-up, keep the context.

A thread Nazar has answered in is a conversation the person started, so a
reply there is answered whether or not it mentions him (ADR slack 17). The
earlier turns are read back from Slack, capped, and handed to the answer step;
a cheap model first turns the latest message into a question that stands on
its own ("and the pool model?" -> "How does the Pool model handle tenant
isolation?") and says whether it needs a lookup at all -- "thanks", "shorter
please" and "what did you mean by that" are answered from the thread.

Nothing here posts; ``app.py`` owns the reply.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

from src.inject._stages import load

CH = load("2_chains", "src.inject.chains")

CONDENSE_MODEL = "gpt-4.1-mini"
MAX_TURNS = 12
MAX_TURN_CHARS = 600
ASSISTANT = "Nazar"
_SKIP_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "bot_message",
    "message_changed",
    "message_deleted",
}

_MENTION = re.compile(r"<@[A-Z0-9]+(?:\|[^>]*)?>")
_LINK = re.compile(r"<(?:https?|mailto):[^|>]*\|([^>]*)>")
_BARE_LINK = re.compile(r"<((?:https?|mailto):[^|>]*)>")
_CHANNEL = re.compile(r"<#[A-Z0-9]+\|([^>]*)>")
_BROADCAST = re.compile(r"<!(here|channel|everyone)(?:\|[^>]*)?>")


class Turn(NamedTuple):
    role: str  # "user" | "assistant"
    name: str
    text: str


def plain_text(text: str, *, bot_user_id: str | None = None) -> str:
    """Slack markup -> plain text. The bot's own mention disappears, others become @user."""
    if bot_user_id:
        text = text.replace(f"<@{bot_user_id}>", " ")
    text = _MENTION.sub("@user", text)
    text = _BROADCAST.sub(lambda m: "@" + m.group(1), text)
    text = _CHANNEL.sub(lambda m: "#" + m.group(1), text)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _BARE_LINK.sub(lambda m: m.group(1), text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(text.split())


def thread_context(
    client,
    channel: str,
    thread_ts: str,
    *,
    bot_user_id: str,
    before_ts: str | None = None,
    limit: int = MAX_TURNS,
) -> list[Turn]:
    """The thread's earlier turns, oldest first, without the message being answered.

    ``client`` is a slack_sdk WebClient (or anything with ``conversations_replies``).
    Needs ``channels:history``. The bot's own replies carry its user id and are
    marked as the assistant; other bots read as users named "bot".
    """
    messages = client.conversations_replies(channel=channel, ts=thread_ts, limit=200)[
        "messages"
    ]
    turns: list[Turn] = []
    for m in messages:
        if before_ts and float(m.get("ts", 0)) >= float(before_ts):
            continue
        if m.get("subtype") in _SKIP_SUBTYPES:
            continue
        text = plain_text(m.get("text", ""), bot_user_id=bot_user_id)
        if not text:
            continue
        if m.get("user") == bot_user_id:
            turns.append(Turn("assistant", ASSISTANT, text[:MAX_TURN_CHARS]))
        else:
            profile = m.get("user_profile") or {}
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or ("bot" if m.get("bot_id") else "someone")
            )
            turns.append(Turn("user", name, text[:MAX_TURN_CHARS]))
    return turns[-limit:]


def bot_in_thread(turns: list[Turn]) -> bool:
    return any(t.role == "assistant" for t in turns)


def format_turns(turns: list[Turn]) -> str:
    return "\n".join(f"{t.name}: {t.text}" for t in turns)


CONDENSE_PROMPT = """You read a Slack thread between people and Nazar, an assistant that answers from the team's indexed notes and channels, and you rewrite the LATEST message as a question that stands on its own.

- Resolve references from the conversation: "it", "that", "the other option", "why?", "and the pool model?" become explicit, in the person's own words where possible.
- If the latest message is already self-contained, return it unchanged.
- needs_lookup is false when the message can be answered from the conversation itself: thanks and acknowledgements, requests to rephrase, shorten, expand or summarise what Nazar just said, "what did you mean by X", or a question the thread already answers. It is true when new information from the notes and channels is needed.
- Never answer the question. Output only the standalone question and the flag."""

CONDENSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"question": {"type": "string"}, "needs_lookup": {"type": "boolean"}},
    "required": ["question", "needs_lookup"],
}

Completion = Callable[..., dict]


def condense(
    latest: str,
    turns: list[Turn],
    *,
    model: str = CONDENSE_MODEL,
    complete: Completion | None = None,
) -> dict:
    """``{"question": standalone, "needs_lookup": bool}``; a passthrough with no turns."""
    if not turns:
        return {"question": latest, "needs_lookup": True}
    complete = CH._complete_json if complete is None else complete
    user = f"Conversation so far:\n{format_turns(turns)}\n\nLatest message: {latest}"
    raw = complete(
        system=CONDENSE_PROMPT,
        user=user,
        model=model,
        schema=CONDENSE_SCHEMA,
        schema_name="standalone_question",
    )
    question = (raw.get("question") or "").strip() or latest
    return {"question": question, "needs_lookup": bool(raw.get("needs_lookup", True))}
