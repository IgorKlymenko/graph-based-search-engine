"""Grounded answers: reframe with evidence, fall back to answer, render for terminal and Slack."""

from src.query import answer as AN

PASSAGES = [
    ("Day 12", "Neo4j is unscalable due to linear pricing; migrate to supabase."),
    ("Day 14", "Decide on RLS vs Bridge Model + migrate the DB from Neo4j"),
]


def stub(response: dict):
    calls: list[dict] = []

    def complete(*, system, user, model, schema, schema_name):
        calls.append({"system": system, "user": user, "model": model})
        return response

    return complete, calls


def test_a_reframe_is_returned_with_its_grounds() -> None:
    complete, calls = stub(
        {
            "stance": "reframe",
            "answer": "Move to Postgres.",
            "reframe": {
                "why": "Neo4j was dropped.",
                "question": "How do I migrate to Postgres?",
                "evidence": ["Day 12"],
            },
            "clarify": None,
            "sources": ["Day 12", "Day 14"],
        }
    )
    verdict = AN.answer(
        "How do I scale Neo4j?", PASSAGES, complete=complete, model="stub"
    )
    assert verdict["stance"] == "reframe" and verdict["reframe"]["evidence"] == [
        "Day 12"
    ]
    assert "### Day 12" in calls[0]["user"]
    text = AN.render(verdict)
    assert text.startswith("Wrong question, per the notes: Neo4j was dropped.")
    assert "Better question: How do I migrate to Postgres?" in text


def test_no_reframe_pins_the_stance_and_tells_the_model() -> None:
    complete, calls = stub(
        {
            "stance": "reframe",
            "answer": "x",
            "reframe": {"why": "w", "question": "q", "evidence": []},
            "clarify": None,
            "sources": [],
        }
    )
    verdict = AN.answer(
        "q?", PASSAGES, complete=complete, model="stub", allow_reframe=False
    )
    assert verdict["stance"] == "answer" and verdict["reframe"] is None
    assert AN.ANSWER_ONLY.strip() in calls[0]["system"]


def test_a_reframe_without_grounds_becomes_a_plain_answer() -> None:
    complete, _ = stub(
        {
            "stance": "reframe",
            "answer": "x",
            "reframe": None,
            "clarify": None,
            "sources": [],
        }
    )
    assert (
        AN.answer("q?", PASSAGES, complete=complete, model="stub")["stance"] == "answer"
    )


def test_slack_rendering_of_a_reframe_is_a_short_preface() -> None:
    verdict = {
        "stance": "reframe",
        "answer": "Use Postgres & RLS.",
        "reframe": {
            "why": "Neo4j <was> dropped.",
            "question": "How do I migrate?",
            "evidence": ["Day 12"],
        },
        "clarify": None,
        "sources": ["earlier/r87/days/Day 12.md"],
    }
    text, blocks = AN.render_slack_blocks(
        verdict, links={"Day 12": "https://x.y/day12"}
    )
    assert text.splitlines()[0] == "*Heads-up:* Neo4j &lt;was&gt; dropped."
    assert "*Better question:* How do I migrate?" in text
    assert "Use Postgres &amp; RLS." in text
    assert [b["type"] for b in blocks] == ["section", "section", "divider", "context"]
    assert blocks[3]["elements"][0]["text"] == "*Sources* · Day 12 (note)"
    assert AN.render_slack(verdict) == text


def test_sources_render_as_readable_titles_and_link_when_a_url_is_known() -> None:
    verdict = {
        "stance": "answer",
        "answer": "x",
        "reframe": None,
        "clarify": None,
        "sources": ["thread/C1/1.2", "earlier/Data Model.md", "earlier/Data Model.md"],
    }
    _, blocks = AN.render_slack_blocks(
        verdict, links={"thread/C1/1.2": "https://acme.slack.com/archives/C1/p12"}
    )
    line = blocks[-1]["elements"][0]["text"]
    assert line == (
        "*Sources* · <https://acme.slack.com/archives/C1/p12|thread/C1/1.2> (thread)"
        " · Data Model (note)"
    )


def test_code_sources_show_their_path_tail_and_are_labelled() -> None:
    verdict = {"stance": "answer", "answer": "x", "reframe": None, "clarify": None,
               "sources": ["acme/core/server/public/src/inject/src/4_link.py"]}
    url = "https://github.com/acme/core/blob/main/server/public/src/inject/src/4_link.py"
    _, blocks = AN.render_slack_blocks(
        verdict, links={"acme/core/server/public/src/inject/src/4_link.py": url})
    assert blocks[-1]["elements"][0]["text"] == f"*Sources* · <{url}|inject/src/4_link.py> (code)"


def test_two_notes_with_the_same_stem_keep_their_folder() -> None:
    names = AN.display_titles(["a/Day 3.md", "b/Day 3.md", "c/Day 4.md"])
    assert names == {
        "a/Day 3.md": "a/Day 3",
        "b/Day 3.md": "b/Day 3",
        "c/Day 4.md": "Day 4",
    }


def test_markdown_the_model_emits_becomes_mrkdwn() -> None:
    md = "## Plan\n- use **RLS** for [tenants](https://x.y/t)\n```sql\nselect 1\n```"
    assert (
        AN.to_mrkdwn(md)
        == "*Plan*\n• use *RLS* for <https://x.y/t|tenants>\n```\nselect 1\n```"
    )


def test_slack_rendering_of_a_clarification_uses_bullets() -> None:
    verdict = {
        "stance": "clarify",
        "answer": "",
        "reframe": None,
        "clarify": {
            "question": "Which S. Chen?",
            "options": ["Sarah Chen", "Steven Chen"],
        },
        "sources": [],
    }
    text, blocks = AN.render_slack_blocks(verdict)
    assert text == "Which S. Chen?\n• Sarah Chen\n• Steven Chen" and len(blocks) == 1


def test_slack_sections_are_clipped_to_the_block_limit() -> None:
    verdict = {
        "stance": "answer",
        "answer": "x" * 20_000,
        "reframe": None,
        "clarify": None,
        "sources": [],
    }
    text, blocks = AN.render_slack_blocks(verdict)
    assert len(blocks[0]["text"]["text"]) == AN.SECTION_LIMIT
    assert len(text) <= AN.SLACK_TEXT_LIMIT


def test_a_reframed_answer_does_not_restate_its_own_preface() -> None:
    verdict = {
        "stance": "reframe",
        "answer": (
            "HDBSCAN was not settled as the final approach. The better question is: "
            "What are the pros and cons of HDBSCAN? HDBSCAN gives denser clusters "
            "but is unstable with low member counts."
        ),
        "reframe": {
            "why": "HDBSCAN is not settled as the final approach.",
            "question": "What are the pros and cons of HDBSCAN?",
            "evidence": ["Day 23"],
        },
        "clarify": None,
        "sources": ["Day 23"],
    }
    _, blocks = AN.render_slack_blocks(verdict)
    assert blocks[1]["text"]["text"] == (
        "HDBSCAN gives denser clusters but is unstable with low member counts."
    )
    plain = {
        "stance": "answer",
        "answer": "The better question is moot here.",
        "reframe": None,
        "clarify": None,
        "sources": [],
    }
    assert (
        AN.render_slack_blocks(plain)[1][0]["text"]["text"]
        == "The better question is moot here."
    )


def test_the_conversation_and_the_message_as_written_reach_the_model() -> None:
    from src.query import thread as TH

    complete, calls = stub(
        {
            "stance": "answer",
            "answer": "RLS per tenant.",
            "reframe": None,
            "clarify": None,
            "sources": [],
        }
    )
    turns = [
        TH.Turn("user", "igor", "how should I scale neo4j?"),
        TH.Turn("assistant", "Nazar", "Use Postgres."),
    ]
    AN.answer(
        "How does the Pool model isolate tenants?",
        [],
        complete=complete,
        model="stub",
        conversation=turns,
        latest="and the pool model?",
    )
    user = calls[0]["user"]
    assert "Passages: none retrieved for this turn." in user
    assert (
        "Conversation so far (you are Nazar):\nigor: how should I scale neo4j?\nNazar: Use Postgres."
        in user
    )
    assert "Latest message as written: and the pool model?" in user
    assert user.endswith("Question: How does the Pool model isolate tenants?")


def test_sources_are_limited_to_the_passages_the_model_was_given() -> None:
    complete, _ = stub(
        {
            "stance": "answer",
            "answer": "Costs rise per tenant.",
            "reframe": None,
            "clarify": None,
            "sources": ["scalability", "made-up.md"],
        }
    )
    with_passages = AN.answer(
        "why?",
        [("scalability", "Aura bills per tenant.")],
        complete=complete,
        model="stub",
    )
    assert with_passages["sources"] == ["scalability"]
    from_thread_only = AN.answer("shorter", [], complete=complete, model="stub")
    assert from_thread_only["sources"] == []
