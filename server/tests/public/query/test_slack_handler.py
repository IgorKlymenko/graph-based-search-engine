"""An @mention is answered in its thread from the workspace's graph, and never crashes the reply."""

import json

from src import app as ingress
from src.security.slack import slack_sign

REFRAME = {
    "stance": "reframe",
    "answer": "Use Postgres.",
    "reframe": {
        "why": "Neo4j was dropped.",
        "question": "How do I migrate?",
        "evidence": ["Day 12"],
    },
    "clarify": None,
    "sources": ["Day 12"],
}


def mention(event_id: str, text: str) -> bytes:
    return json.dumps(
        {
            "type": "event_callback",
            "team_id": "T_ACME",
            "event_id": event_id,
            "event": {
                "type": "app_mention",
                "user": "U1",
                "channel": "C1",
                "ts": "1700000000.000100",
                "text": text,
            },
        }
    ).encode()


def post(client, body: bytes):
    ts, sig = slack_sign(body, b"mysecret")
    return client.post(
        "/slack/event",
        content=body,
        headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig},
    )


def test_a_mention_is_answered_in_thread_as_its_workspace(client, monkeypatch) -> None:
    posted, asked = [], []
    monkeypatch.setattr(
        ingress.slack, "chat_postMessage", lambda **kw: posted.append(kw)
    )

    def fake_answer(question, **kw):
        asked.append((question, kw))
        return REFRAME

    monkeypatch.setattr(ingress, "answer_question", fake_answer)
    assert (
        post(
            client, mention("Ev_1", "<@UBOT> why is <#C9|eng> scaling neo4j?")
        ).status_code
        == 200
    )
    assert asked[0][0] == "why is #eng scaling neo4j?"
    assert asked[0][1]["connector"] == "slack" and asked[0][1]["workspace"] == "T_ACME"
    assert (
        posted[0]["channel"] == "C1" and posted[0]["thread_ts"] == "1700000000.000100"
    )
    assert posted[0]["text"].startswith("*Heads-up:* Neo4j was dropped.")


def test_nothing_found_posts_the_fallback(client, monkeypatch) -> None:
    posted = []
    monkeypatch.setattr(
        ingress.slack, "chat_postMessage", lambda **kw: posted.append(kw)
    )
    monkeypatch.setattr(ingress, "answer_question", lambda q, **kw: None)
    post(client, mention("Ev_2", "<@UBOT> anything about zzz?"))
    assert posted[0]["text"] == ingress.NOTHING_FOUND and posted[0]["blocks"] is None


def test_a_failing_retrieval_still_replies_and_logs_no_text(
    client, monkeypatch, caplog
) -> None:
    posted = []
    monkeypatch.setattr(
        ingress.slack, "chat_postMessage", lambda **kw: posted.append(kw)
    )

    def boom(question, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(ingress, "answer_question", boom)
    with caplog.at_level("ERROR", logger="ingress"):
        post(client, mention("Ev_3", "<@UBOT> secret question text"))
    assert posted[0]["text"] == ingress.NOTHING_FOUND
    assert "secret question text" not in caplog.text and "T_ACME" in caplog.text


def test_an_empty_mention_gets_a_prompt_not_a_query(client, monkeypatch) -> None:
    posted, asked = [], []
    monkeypatch.setattr(
        ingress.slack, "chat_postMessage", lambda **kw: posted.append(kw)
    )
    monkeypatch.setattr(ingress, "answer_question", lambda q, **kw: asked.append(q))
    post(client, mention("Ev_4", "<@UBOT>"))
    assert posted[0]["text"] == ingress.NOT_A_QUESTION and asked == []


def test_question_from_resolves_slack_markup() -> None:
    assert ingress.question_from(
        "<@U1|bot> see <https://x.y|the doc> in <#C1|eng> &amp; <https://z.q>"
    ) == ("see the doc in #eng & https://z.q")


def test_each_answer_is_logged_by_workspace_and_stance_without_the_text(
    client, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        ingress.slack,
        "chat_postMessage",
        lambda **kw: {"ok": True, "ts": "1700000000.000200"},
    )
    monkeypatch.setattr(ingress, "answer_question", lambda q, **kw: REFRAME)
    with caplog.at_level("INFO", logger="ingress"):
        post(client, mention("Ev_5", "<@UBOT> a private question"))
    assert "workspace=T_ACME" in caplog.text and "stance=reframe" in caplog.text
    assert "ts=1700000000.000200" in caplog.text
    assert "private question" not in caplog.text


THREAD_MESSAGES = [
    {
        "ts": "1700000000.000100",
        "user": "U1",
        "user_profile": {"display_name": "igor"},
        "text": "<@UBOT> how should I scale neo4j?",
    },
    {
        "ts": "1700000000.000200",
        "user": "UBOT",
        "bot_id": "B1",
        "text": "*Heads-up:* dropped.\n\nUse Postgres.",
    },
]


def thread_reply(event_id: str, text: str, with_bot: bool = True) -> bytes:
    return json.dumps(
        {
            "type": "event_callback",
            "team_id": "T_ACME",
            "event_id": event_id,
            "event": {
                "type": "message",
                "channel_type": "channel",
                "user": "U1",
                "channel": "C1",
                "ts": "1700000000.000300",
                "thread_ts": "1700000000.000100",
                "text": text,
            },
        }
    ).encode()


def _wire(monkeypatch, messages):
    posted, asked, condensed = [], [], []
    monkeypatch.setattr(ingress, "BOT_USER_ID", "UBOT")
    monkeypatch.setattr(
        ingress.slack, "conversations_replies", lambda **kw: {"messages": messages}
    )
    monkeypatch.setattr(
        ingress.slack,
        "chat_postMessage",
        lambda **kw: posted.append(kw) or {"ok": True, "ts": "1700000000.000400"},
    )

    def fake_condense(latest, turns, **kw):
        condensed.append((latest, [t.role for t in turns]))
        return {
            "question": "How does the Pool model isolate tenants?",
            "needs_lookup": True,
        }

    monkeypatch.setattr(ingress.TH, "condense", fake_condense)

    def fake_answer(question, **kw):
        asked.append((question, kw))
        return {
            "stance": "answer",
            "answer": "Row-level security per tenant.",
            "reframe": None,
            "clarify": None,
            "sources": ["Day 38"],
            "retrieval": {"lookup": True},
        }

    monkeypatch.setattr(ingress, "answer_question", fake_answer)
    return posted, asked, condensed


def test_a_reply_in_nazars_thread_is_answered_with_the_thread_as_context(
    client, monkeypatch
) -> None:
    posted, asked, condensed = _wire(monkeypatch, THREAD_MESSAGES)
    assert post(client, thread_reply("Ev_t1", "and the pool model?")).status_code == 200
    assert condensed == [("and the pool model?", ["user", "assistant"])]
    question, kw = asked[0]
    assert question == "How does the Pool model isolate tenants?"
    assert kw["latest"] == "and the pool model?" and kw["needs_lookup"] is True
    assert [t.role for t in kw["conversation"]] == ["user", "assistant"]
    assert posted[0]["thread_ts"] == "1700000000.000100"
    assert posted[0]["text"].startswith("Row-level security per tenant.")


def test_a_reply_in_a_thread_without_nazar_is_ignored(client, monkeypatch) -> None:
    posted, asked, _ = _wire(
        monkeypatch, [m for m in THREAD_MESSAGES if m["user"] != "UBOT"]
    )
    post(client, thread_reply("Ev_t2", "anyone?"))
    assert posted == [] and asked == []


def test_a_top_level_channel_message_is_ignored(client, monkeypatch) -> None:
    posted, asked, _ = _wire(monkeypatch, THREAD_MESSAGES)
    body = json.dumps(
        {
            "type": "event_callback",
            "team_id": "T_ACME",
            "event_id": "Ev_t3",
            "event": {
                "type": "message",
                "channel_type": "channel",
                "user": "U1",
                "channel": "C1",
                "ts": "1700000000.000500",
                "text": "morning all",
            },
        }
    ).encode()
    post(client, body)
    assert posted == [] and asked == []


def test_a_mention_inside_a_thread_reads_the_thread_first(client, monkeypatch) -> None:
    posted, asked, condensed = _wire(monkeypatch, THREAD_MESSAGES)
    body = json.dumps(
        {
            "type": "event_callback",
            "team_id": "T_ACME",
            "event_id": "Ev_t4",
            "event": {
                "type": "app_mention",
                "user": "U1",
                "channel": "C1",
                "ts": "1700000000.000300",
                "thread_ts": "1700000000.000100",
                "text": "<@UBOT> why?",
            },
        }
    ).encode()
    post(client, body)
    assert condensed == [("why?", ["user", "assistant"])]
    assert asked[0][1]["conversation"] is not None and posted
