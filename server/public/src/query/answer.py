"""Answer from passages -- and say when the question is the wrong one to ask.

The retrieval path collects what the notes say about the names a question
uses. Sometimes that material shows the question itself is off: it assumes
something the notes contradict (a tool that was dropped, a plan that was
superseded), it calls a provisional choice final, it asks how to do X where the
notes record that X was a route to a goal now reached another way, or it is
narrower or wider than what the notes can answer. Answering such a question
literally is the classic failure of a retrieval system: confident, cited, and
beside the point.

So the model decides a *stance* first:

    answer    the question is fine; answer it from the passages
    reframe   the passages show a better question; say why, ask it, answer it
    clarify   a name in the question means several things; ask which

A reframe must name the passages that justify it. Passages that merely lack
the answer are never grounds for one -- that is "the notes do not say", not
"wrong question". Measured on the author's vault (ADR slack 16): fine
questions left alone 7/7, bad premises caught 3/5, no stance flipped across
three runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.inject._stages import load

CH = load("2_chains", "src.inject.chains")

ANSWER_MODEL = "gpt-4.1"
STANCES = ("answer", "reframe", "clarify")
ASSISTANT = "Nazar"
SLACK_TEXT_LIMIT = (
    12_000  # well under Slack's 40k; a thread reply should not be an essay
)

ANSWER_PROMPT = """You answer questions from a team's indexed conversations and notes. You are given passages retrieved for the question, each under its source title, and -- when there are any -- the names in the question that the corpus uses for more than one thing.

FIRST decide the stance.

reframe -- the passages show the question is the wrong one to ask. Grounds, each of which must be visible in the passages you were given:
  - a premise the passages contradict: it asks about something recorded as dropped, replaced, or decided the other way
  - a superseded plan: the passages record a later decision that makes the question moot
  - means for goal: it asks how to do X where the passages show X was one route to a goal that is reached another way
  - scope: the passages answer a sharper or broader question that is what the person actually needs
  - a provisional choice presented as settled: the question calls something final, chosen or the plan ("why did we pick", "the final approach", "before the demo") while the passages call it a first pass, unstable, picked "by vibes", a poor option, or record a next move away from it -- then the better question is what to do next, not why the choice was made
  When you reframe: say in one sentence why the question as asked is off, state the better question, list the source titles that show it, then answer the better question from the passages.

clarify -- a name in the question denotes several different things in the corpus (listed below), and the passages for each would answer differently. Ask which one is meant, listing the options as the corpus names them. Do not answer.

answer -- otherwise. Answer only from the passages. If they do not contain the answer, say so plainly and say what they do cover instead. A missing answer is never grounds for a reframe.

Rules:
- Use only the passages. Quote the source titles you relied on in `sources`.
- Passages are rough working text: fragments, arrows, abbreviations, chat. Read through the shorthand rather than complaining about it.
- Be concise and concrete. Default to `answer`; reframe only on evidence you can point to.

CONVERSATION -- when earlier turns are given
- You are continuing that thread: the question may refer to earlier turns ("why?", "and the other option?"); answer the current turn in that light and address the person directly.
- Do not repeat what you already said unless asked; do not reframe a question you already reframed in this thread; build on your earlier answer.
- If no passages are given, answer from the conversation, and say plainly when it does not contain enough.

FORMAT -- the reply is posted in a Slack thread
- Lead with the answer in one or two sentences, then at most four short bullets if they add something. Keep `answer` under 120 words.
- Plain sentences. No headings, no tables, no markdown links. Bold only a key term, with *single asterisks*. Bullets start with "- ".
- Refer to sources by their titles exactly as given; put the ones you used in `sources` (titles only).
- When reframing, `why` is one sentence under 25 words, and `answer` answers the better question only -- do not repeat why the question was off or restate the better question."""

ANSWER_ONLY = "\n\nFor this request the stance must be `answer`."

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "stance": {"type": "string", "enum": list(STANCES)},
        "answer": {"type": "string"},
        "reframe": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "why": {"type": "string"},
                "question": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["why", "question", "evidence"],
        },
        "clarify": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "options"],
        },
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["stance", "answer", "reframe", "clarify", "sources"],
}

Completion = Callable[..., dict]


def build_user(
    question: str,
    passages: list[tuple[str, str]],
    contested: dict[str, list[str]] | None = None,
    conversation: list | None = None,
    latest: str | None = None,
) -> str:
    """The user turn: passages, contested names, the thread so far, the question."""
    parts = []
    if passages:
        context = "\n\n".join(f"### {title}\n{text}" for title, text in passages)
        parts.append(f"Passages:\n\n{context}")
    else:
        parts.append("Passages: none retrieved for this turn.")
    if contested:
        names = "\n".join(
            f"- {name}: {', '.join(options)}" for name, options in contested.items()
        )
        parts.append(f"Names the corpus uses for more than one thing:\n{names}")
    if conversation:
        turns = "\n".join(f"{t.name}: {t.text}" for t in conversation)
        parts.append(f"Conversation so far (you are {ASSISTANT}):\n{turns}")
        if latest and latest.strip() != question.strip():
            parts.append(f"Latest message as written: {latest}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def answer(
    question: str,
    passages: list[tuple[str, str]],
    *,
    conversation: list | None = None,
    latest: str | None = None,
    contested: dict[str, list[str]] | None = None,
    model: str = ANSWER_MODEL,
    allow_reframe: bool = True,
    complete: Completion | None = None,
) -> dict:
    """One structured completion: stance, answer, and the reframe or clarification.

    ``allow_reframe=False`` pins the stance to ``answer``. The result is
    normalised so callers can rely on the shape even if the model leaves a
    branch empty.
    """
    complete = CH._complete_json if complete is None else complete
    system = ANSWER_PROMPT + ("" if allow_reframe else ANSWER_ONLY)
    raw = complete(
        system=system,
        user=build_user(question, passages, contested, conversation, latest),
        model=model,
        schema=ANSWER_SCHEMA,
        schema_name="grounded_answer",
    )
    stance = raw.get("stance") if raw.get("stance") in STANCES else "answer"
    if not allow_reframe or (stance == "reframe" and not raw.get("reframe")):
        stance = "answer"
    if stance == "clarify" and not raw.get("clarify"):
        stance = "answer"
    return {
        "stance": stance,
        "answer": (raw.get("answer") or "").strip(),
        "reframe": raw.get("reframe") if stance == "reframe" else None,
        "clarify": raw.get("clarify") if stance == "clarify" else None,
        "sources": _cited(raw.get("sources"), passages),
    }


def _cited(sources: object, passages: list[tuple[str, str]]) -> list[str]:
    """Only titles the model was actually given; nothing when no lookup ran."""
    given = {title for title, _ in passages}
    if not isinstance(sources, list):
        return []
    return [s for s in sources if isinstance(s, str) and s in given]


def display_title(title: str) -> str:
    """A citation people can read: a note's stem instead of its path.

    Only file-like titles (ending in ``.md``) are shortened; a Slack thread's
    external id (``thread/C1/1.2``) or a real title is left alone.
    """
    if not title.endswith(".md"):
        return title
    return title.rsplit("/", 1)[-1][:-3]


def display_titles(titles: list[str]) -> dict[str, str]:
    """title -> readable name; two notes sharing a stem keep their parent folder."""
    stems: dict[str, list[str]] = {}
    for title in dict.fromkeys(titles):
        stems.setdefault(display_title(title), []).append(title)
    out: dict[str, str] = {}
    for stem, group in stems.items():
        for title in group:
            if len(group) == 1 or "/" not in title:
                out[title] = stem
            else:
                out[title] = f"{title.rsplit('/', 2)[-2]}/{stem}"
    return out


def render(verdict: dict) -> str:
    """Plain text for a terminal."""
    if verdict["stance"] == "clarify":
        options = "".join(f"\n  - {o}" for o in verdict["clarify"]["options"])
        return f"{verdict['clarify']['question']}{options}"
    lines: list[str] = []
    if verdict["stance"] == "reframe":
        r = verdict["reframe"]
        shown = "; ".join(r["evidence"]) if r["evidence"] else "the passages"
        lines += [
            f"Wrong question, per the notes: {r['why']}",
            f"Better question: {r['question']}",
            f"(from: {shown})",
            "",
        ]
    lines.append(verdict["answer"] or "The notes do not say.")
    if verdict["sources"]:
        lines += ["", "sources: " + "; ".join(verdict["sources"])]
    return "\n".join(lines)


_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_BULLET = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_MD_FENCE_LANG = re.compile(r"```[A-Za-z0-9_+-]+\n")
SECTION_LIMIT = 2_900  # Slack caps a section block's text at 3,000 characters


def _mrkdwn(text: str) -> str:
    """Escape the three characters Slack's mrkdwn treats as control."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_mrkdwn(text: str) -> str:
    """Markdown the model may still emit -> Slack mrkdwn.

    Headings become a bold line, **bold** becomes *bold*, "- " bullets become
    "• ", [text](url) becomes <url|text>, code fences lose their language tag
    (Slack renders bare fences). Control characters are escaped first, so a
    link's angle brackets are the only raw ones left.
    """
    text = _mrkdwn(text.strip())
    text = _MD_FENCE_LANG.sub("```\n", text)
    text = _MD_HEADING.sub(lambda m: f"*{m.group(1)}*", text)
    text = _MD_BOLD.sub(r"*\1*", text)
    text = _MD_BULLET.sub(r"\1• ", text)
    text = _MD_LINK.sub(r"<\2|\1>", text)
    return text


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_RESTATES = re.compile(
    r"^(the (better|right) question( to ask)? (is|would be|should be)|the question(,)? as asked|"
    r"the question is off|this question|instead of asking)",
    re.IGNORECASE,
)


def _without_preface(answer: str, reframe: dict | None) -> str:
    """Drop sentences that restate the reframe's preface inside the answer.

    The prompt forbids it and the model still does it about half the time
    (measured on the live path: "HDBSCAN was not settled... The better question
    is: ..." above an answer that then answers it). The preface already carries
    the why and the better question; a sentence that quotes the better
    question, or opens by explaining that the question is off, is redundant.
    """
    if not reframe:
        return answer
    question = reframe.get("question", "").strip().rstrip("?").lower()
    why_words = _content_words(reframe.get("why", ""))
    kept = []
    for sentence in _SENTENCE.split(answer.strip()):
        plain = sentence.strip()
        if not plain:
            continue
        if question and question in plain.lower():
            continue
        if _RESTATES.match(plain):
            continue
        words = _content_words(plain)
        if why_words and words and len(words & why_words) / len(words) >= 0.6:
            continue  # mostly the preface's own words again
        kept.append(plain)
    return " ".join(kept) if kept else answer.strip()


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4}


def _clip(text: str, limit: int = SECTION_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _source_kind(title: str, url: str | None) -> str:
    """What a reader is about to open: code, a thread, or a note."""
    if url and "github.com" in url:
        return "code"
    if title.startswith(("thread/", "day/")) or (url and "slack.com" in url):
        return "thread"
    return "note"


def _sources_line(
    sources: list[str], links: dict[str, str] | None, limit: int = 4
) -> str:
    """``*Sources* · <url|title> (code) · title (note) · +2 more``.

    Every source is labelled by kind so the reader knows what they are being
    pointed at before clicking, and the bold label marks the line as the
    part to check rather than decoration. Code titles are ``owner/repo/path``;
    they show as the path's last three segments, which is what a colleague
    would say ("inject/src/4_link.py").
    """
    names = display_titles(sources)
    shown = []
    for title in dict.fromkeys(sources):
        url = (links or {}).get(title)
        kind = _source_kind(title, url)
        name = names[title]
        if kind == "code":
            name = "/".join(title.split("/")[-3:])
        name = _mrkdwn(name)
        shown.append((f"<{url}|{name}>" if url else name) + f" ({kind})")
        if len(shown) == limit:
            break
    more = len(dict.fromkeys(sources)) - len(shown)
    return "*Sources* · " + " · ".join(shown) + (f" · +{more} more" if more > 0 else "")


def render_slack(verdict: dict, links: dict[str, str] | None = None) -> str:
    """Slack mrkdwn as one text: the fallback for clients without blocks."""
    text, _ = render_slack_blocks(verdict, links)
    return text


def render_slack_blocks(
    verdict: dict,
    links: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """``(fallback text, Block Kit blocks)`` for a thread reply.

    A reframe is a two-line preface in its own section, the answer is a
    section, sources are a muted context line with readable titles linked
    when a URL is known. Nothing here is a lecture: sections are clipped to
    Slack's limit and the prompt already keeps answers short.
    """

    def section(text: str) -> dict:
        return {"type": "section", "text": {"type": "mrkdwn", "text": _clip(text)}}

    if verdict["stance"] == "clarify":
        c = verdict["clarify"]
        body = _mrkdwn(c["question"]) + "".join(
            f"\n• {_mrkdwn(o)}" for o in c["options"]
        )
        return body, [section(body)]

    blocks: list[dict] = []
    parts: list[str] = []
    if verdict["stance"] == "reframe":
        r = verdict["reframe"]
        preface = (
            f"*Heads-up:* {_mrkdwn(r['why'])}\n"
            f"*Better question:* {_mrkdwn(r['question'])}"
        )
        blocks.append(section(preface))
        parts.append(preface)
    answer = (
        to_mrkdwn(_without_preface(verdict["answer"], verdict.get("reframe")))
        or "I found nothing in the indexed notes and code that answers this."
    )
    blocks.append(section(answer))
    parts.append(answer)
    if verdict["sources"]:
        line = _sources_line(verdict["sources"], links)
        blocks.append({"type": "divider"})
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": _clip(line)}]}
        )
        parts.append(line)
    text = "\n\n".join(parts)
    return (
        text if len(text) <= SLACK_TEXT_LIMIT else text[: SLACK_TEXT_LIMIT - 1] + "…"
    ), blocks
