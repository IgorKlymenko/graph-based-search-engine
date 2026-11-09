"""Threads: read the earlier turns, know when Nazar is in one, condense the follow-up."""

from src.query import thread as TH


class StubSlack:
    def __init__(self, messages):
        self.messages = messages
        self.calls = []

    def conversations_replies(self, **kw):
        self.calls.append(kw)
        return {"messages": self.messages}


THREAD = [
    {
        "ts": "1.0",
        "user": "U1",
        "user_profile": {"display_name": "igor"},
        "text": "<@UBOT> how should I scale neo4j?",
    },
    {
        "ts": "2.0",
        "user": "UBOT",
        "bot_id": "B1",
        "text": "*Heads-up:* Neo4j was dropped.\n\nUse Postgres.",
    },
    {"ts": "2.5", "user": "U9", "subtype": "channel_join", "text": "joined"},
    {
        "ts": "3.0",
        "user": "U1",
        "user_profile": {"display_name": "igor"},
        "text": "and the pool model?",
    },
]


def test_plain_text_resolves_markup_and_drops_the_bots_mention() -> None:
    text = "<@UBOT> see <https://x.y|the doc> in <#C1|eng> with <@U2> &amp; <!here>"
    assert (
        TH.plain_text(text, bot_user_id="UBOT")
        == "see the doc in #eng with @user & @here"
    )


def test_thread_context_marks_nazar_and_excludes_the_message_being_answered() -> None:
    slack = StubSlack(THREAD)
    turns = TH.thread_context(slack, "C1", "1.0", bot_user_id="UBOT", before_ts="3.0")
    assert slack.calls[0]["ts"] == "1.0"
    assert [(t.role, t.name) for t in turns] == [
        ("user", "igor"),
        ("assistant", "Nazar"),
    ]
    assert turns[0].text == "how should I scale neo4j?"
    assert TH.bot_in_thread(turns)


def test_a_thread_without_nazar_is_not_his_conversation() -> None:
    slack = StubSlack([m for m in THREAD if m.get("user") != "UBOT"])
    assert not TH.bot_in_thread(
        TH.thread_context(slack, "C1", "1.0", bot_user_id="UBOT")
    )


def test_condense_passes_through_without_turns_and_uses_the_model_with_them() -> None:
    assert TH.condense("hello?", []) == {"question": "hello?", "needs_lookup": True}
    seen = []

    def complete(*, system, user, model, schema, schema_name):
        seen.append(user)
        return {
            "question": "How does the Pool model isolate tenants?",
            "needs_lookup": True,
        }

    turns = [
        TH.Turn("user", "igor", "how should I scale neo4j?"),
        TH.Turn("assistant", "Nazar", "Use Postgres."),
    ]
    out = TH.condense("and the pool model?", turns, complete=complete)
    assert out == {
        "question": "How does the Pool model isolate tenants?",
        "needs_lookup": True,
    }
    assert (
        "Nazar: Use Postgres." in seen[0]
        and "Latest message: and the pool model?" in seen[0]
    )


def test_condense_falls_back_to_the_message_when_the_model_returns_nothing() -> None:
    turns = [TH.Turn("assistant", "Nazar", "x")]
    out = TH.condense(
        "thanks!", turns, complete=lambda **kw: {"question": "", "needs_lookup": False}
    )
    assert out == {"question": "thanks!", "needs_lookup": False}
