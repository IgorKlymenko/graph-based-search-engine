"""Questions the harness author did not write: sampled passages, asked by a model.

Two sets beside the hand-written ``agent_questions.json``:

    generated    ``agent_questions_generated.json``: random live passages from
                 the indexed tenant (code and notes), each turned by gpt-4.1
                 into the question a colleague would ask that the passage
                 answers, phrased the way people talk, plus a two-sentence
                 answer. A second call drops questions the passage does not
                 support or that general knowledge would answer.
    paraphrased  ``agent_questions_paraphrased.json``: the hand-written
                 questions rewritten in plain words with no identifiers, so the
                 text channel does not get its best case for free.

    uv run python scripts/make_agent_questions.py [--code 30 --notes 30 --seed 7]

Run it after an ingest has landed, not during one: an episode sampled before
the upload may be expired by it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import psycopg
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
TENANT = "dddddddd-0000-4000-8000-00000000000d"
DSN = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
MODEL = "gpt-4.1"
MIN_CHARS = 300
GITHUB_TITLE_PREFIX = "IgorKlymenko/corev1/"

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "skip": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["skip", "question", "answer"],
    "additionalProperties": False,
}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"keep": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["keep", "why"],
    "additionalProperties": False,
}
PARA_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def sample_passages(n_code: int, n_notes: int, seed: int) -> list[dict]:
    """Live episodes long enough to hold a fact, spread over both sources."""
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT e.id::text, f.source, coalesce(f.title, f.external_id), e.text"
            "  FROM public.episodes e JOIN public.files f ON f.id = e.file_id"
            " WHERE e.tenant_id = %s AND e.expired_at IS NULL AND f.removed_at IS NULL"
            "   AND length(e.text) > %s",
            (TENANT, MIN_CHARS),
        ).fetchall()
    rng = random.Random(seed)
    code = [r for r in rows if r[1] == "github"]
    notes = [r for r in rows if r[1] == "upload"]
    picked = rng.sample(code, min(n_code, len(code))) + rng.sample(
        notes, min(n_notes, len(notes))
    )
    return [
        {"episode": eid, "source": src, "title": title, "text": text}
        for eid, src, title, text in picked
    ]


def ask(passage: dict, complete) -> dict:
    kind = (
        "a source file's rendered symbols"
        if passage["source"] == "github"
        else "a note"
    )
    return complete(
        system=(
            "You know this software project well: a Slack assistant (Nazar) that "
            "builds a knowledge graph from notes and code. You are given "
            f"{kind} from it. Write ONE question a colleague would plausibly ask "
            "in Slack or while working in the code, which this passage answers. "
            "Phrase it the way people talk: no copying of function names, "
            "identifiers or exact phrases unless that name is what anyone would "
            "say; not a yes/no question; specific to this project, not general "
            "knowledge. Then answer it in at most two sentences from the passage "
            "only. If the passage is boilerplate, a bare list of names, or a "
            "fragment nobody would ask about, set skip to true."
        ),
        user=f"Source: {passage['title']}\n\nPassage:\n{passage['text']}",
        model=MODEL,
        schema=ASK_SCHEMA,
        schema_name="question_from_passage",
    )


def check(passage: dict, q: dict, complete) -> dict:
    return complete(
        system=(
            "Decide whether to keep a benchmark question. Keep it only if (a) the "
            "passage supports the given answer, (b) the question is about this "
            "project rather than answerable from general knowledge, and (c) the "
            "question is natural: someone would actually ask it."
        ),
        user=f"Passage:\n{passage['text']}\n\nQuestion: {q['question']}\n"
        f"Answer: {q['answer']}",
        model=MODEL,
        schema=CHECK_SCHEMA,
        schema_name="keep_question",
    )


def paraphrase(question: str, gold_answer: str, complete) -> str:
    """The same question in plain words: identifiers out, the project's vocabulary kept."""
    return complete(
        system=(
            "You rewrite questions about one software project: a Slack assistant "
            "that builds a knowledge graph from a founder's notes and from source "
            "code (episodes, chains, surfaces, entities, tenants, connectors, the "
            "linker, the ingress are its ordinary vocabulary and must stay). "
            "Rewrite the question in plain words with the same meaning and the "
            "same referent: remove code identifiers, file names, quoted terms and "
            "numbers, describing the thing instead of naming it. Do not reinterpret "
            "domain words as everyday words. One sentence. The reference answer "
            "tells you what the question is really about."
        ),
        user=f"Question: {question}\nReference answer: {gold_answer}",
        model=MODEL,
        schema=PARA_SCHEMA,
        schema_name="paraphrase",
    )["question"]


def gold_file_for(passage: dict) -> str:
    title = passage["title"]
    if title.startswith(GITHUB_TITLE_PREFIX):
        return title[len(GITHUB_TITLE_PREFIX) :]
    return f"vault:{title}"


def main() -> None:
    import sys

    sys.path.insert(0, str(HERE.parent))
    from src.inject import chains as CH

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--code", type=int, default=30)
    parser.add_argument("--notes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", choices=["generated", "paraphrased"], default=None)
    args = parser.parse_args()

    complete = CH._complete_json
    if args.only != "paraphrased":
        write_generated(args, complete)
    if args.only != "generated":
        write_paraphrased(complete)


def write_generated(args: argparse.Namespace, complete) -> None:
    generated = []
    dropped = 0
    for passage in sample_passages(args.code, args.notes, args.seed):
        q = ask(passage, complete)
        if q["skip"] or not q["question"].strip():
            dropped += 1
            continue
        verdict = check(passage, q, complete)
        if not verdict["keep"]:
            dropped += 1
            continue
        generated.append(
            {
                "id": f"G{len(generated) + 1}",
                "category": "code" if passage["source"] == "github" else "vault",
                "question": q["question"].strip(),
                "gold": [],
                "gold_answer": q["answer"].strip(),
                "gold_files": [gold_file_for(passage)],
                "episode": passage["episode"],
            }
        )
    out = HERE / "agent_questions_generated.json"
    out.write_text(
        json.dumps(
            {
                "_doc": "Model-written questions from sampled passages; see "
                "make_agent_questions.py. gold is empty on purpose: graded by "
                "answer only.",
                "questions": generated,
            },
            indent=1,
        )
    )
    print(f"generated {len(generated)} questions ({dropped} dropped) -> {out}")


def write_paraphrased(complete) -> None:
    hand = json.loads((HERE / "agent_questions.json").read_text())["questions"]
    para = [
        {
            **q,
            "id": q["id"] + "p",
            "question": paraphrase(q["question"], q["gold_answer"], complete),
        }
        for q in hand
    ]
    out = HERE / "agent_questions_paraphrased.json"
    out.write_text(
        json.dumps(
            {
                "_doc": "The hand-written questions with identifiers removed.",
                "questions": para,
            },
            indent=1,
        )
    )
    print(f"paraphrased {len(para)} -> {out}")


if __name__ == "__main__":
    main()
