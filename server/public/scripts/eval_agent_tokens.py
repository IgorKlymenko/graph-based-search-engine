"""Does letting an agent search the graph save tokens over reading files?

An agent that needs a fact about this project has a few ways to get it:

    read_gold    open exactly the right files (an oracle; the floor for reading)
    grep_read    grep the corpus for the question's key terms, open the top
                 files whole (what a coding agent does with no search tool)
    wiki         read wiki/README.md, let a small model route to a page, read it
    search_k     call the graph search once and read the top k passages
    search_open  search_3, then open the file behind the top passage
    hybrid_k     the same search with the BM25 text channel fused in
    hybrid_open  hybrid_5, then open the file behind the top passage

For each question in ``agent_questions.json`` the harness builds the context
each strategy would put in front of the agent, counts its tokens (o200k), and
checks whether every gold string is in it (*sufficient*: the agent could answer
without another round trip). Every tool call an agent would make adds
``CALL_OVERHEAD`` tokens. With ``--grade`` a small model answers from each
context and a stronger one grades the answer against ``gold_answer``, which is
the honest measure when a passage paraphrases the file.

    uv run python scripts/eval_agent_tokens.py [--grade] [--show] [--category code]

Needs the local database with the vault-plus-code tenant (``TENANT``) and an
OpenAI key for the question embedding. Corpus roots mirror what that tenant
indexes; ``stale`` questions point at files the index has not been re-synced
with, which is the cost of an out-of-date index, not of the search. When the
GitHub connector lands, run the code questions again after a sync: nothing
here is specific to how the code got in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent
ROOT = PUBLIC.parent.parent
VAULT = ROOT / "old_server" / "artifacts" / "vault_notes"
QUESTIONS = HERE / "agent_questions.json"
CACHE = HERE / "eval_agent_tokens.cache.json"
RESULTS = HERE / "eval_agent_tokens.results.json"
load_dotenv(PUBLIC / ".env")

TENANT = "dddddddd-0000-4000-8000-00000000000d"
DSN = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
# What the vault-plus-code tenant indexes (the code cursor's --include), so the
# grep baseline walks the same material the search can reach.
CORPUS_ROOTS = ("server/public", "ADRs", "CLAUDE.md")  # legacy copy retired 2026-09-19
TEXT_CANDIDATES = 10  # the text channel, for the hybrid conditions
CORPUS_SUFFIXES = frozenset({".py", ".md"})
SKIP_PARTS = frozenset({"artifacts", ".venv", "__pycache__", "node_modules", ".git"})
GITHUB_TITLE_PREFIX = "IgorKlymenko/corev1/"
CALL_OVERHEAD = 40  # tokens an agent spends on one tool call and its framing
GREP_LINES = 40  # lines of grep output an agent would look at
GREP_OPEN = 3  # files it then opens whole
ROUTER_MODEL = "gpt-4.1-mini"
ANSWER_MODEL = "gpt-4.1-mini"
GRADER_MODEL = "gpt-4.1"
STOPWORDS = frozenset(
    [
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "with",
        "by",
        "from",
        "into",
        "than",
        "that",
        "this",
        "these",
        "those",
        "its",
        "it",
        "their",
        "there",
        "here",
        "between",
        "rather",
        "over",
        "each",
        "one",
        "two",
        "three",
        "both",
        "any",
        "all",
        "not",
        "our",
        "we",
        "us",
        "them",
        "they",
        "them",
        "should",
        "would",
        "could",
        "kept",
        "out",
        "get",
        "gets",
        "step",
        "steps",
    ]
)


# ── tokens and sufficiency ────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """o200k tokens, the tokenizer the answer models share."""
    import tiktoken

    return len(tiktoken.get_encoding("o200k_base").encode(text))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def sufficient(context: str, gold: list[str]) -> bool:
    """Every gold string is in the context: the agent could answer from it."""
    haystack = norm(context)
    return all(norm(g) in haystack for g in gold)


# ── the corpus on disk, as a naive agent sees it ───────────────────────────────


def corpus_files(root: Path = ROOT, vault: Path = VAULT) -> list[Path]:
    """The files the index covers, as paths a grep would walk."""
    out: list[Path] = []
    for rel in CORPUS_ROOTS:
        base = root / rel
        if base.is_file():
            out.append(base)
            continue
        for path in base.rglob("*"):
            if path.suffix in CORPUS_SUFFIXES and not SKIP_PARTS & set(path.parts):
                out.append(path)
    out.extend(p for p in vault.rglob("*.md") if ".obsidian" not in p.parts)
    return sorted(set(out))


def gold_path(spec: str, root: Path = ROOT, vault: Path = VAULT) -> Path:
    """``vault:<note>`` or a repo-relative path."""
    return vault / spec[6:] if spec.startswith("vault:") else root / spec


def keywords(question: str, limit: int = 3) -> list[str]:
    """What an agent would grep for: identifiers first, then the longest words."""
    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_]|[0-9][0-9./]*[0-9]", question
    )
    seen: list[str] = []
    for w in words:
        if w.lower() in STOPWORDS or w in seen:
            continue
        if len(w) >= 4 or "_" in w or "." in w or any(c.isdigit() for c in w):
            seen.append(w)

    def specific(w: str) -> tuple[int, int]:
        ident = (
            "_" in w or "." in w or (w[:1].islower() and any(c.isupper() for c in w))
        )
        return (int(ident) + int(any(c.isdigit() for c in w)), len(w))

    return sorted(seen, key=specific, reverse=True)[:limit]


class GrepHit(NamedTuple):
    path: Path
    distinct: int
    matches: int
    lines: list[str]


def grep_rank(files: list[Path], terms: list[str]) -> list[GrepHit]:
    """Files by how many of the terms they hold, then how often; grep-style lines."""
    hits: list[GrepHit] = []
    lowered = [t.lower() for t in terms]
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        low = text.lower()
        counts = [low.count(t) for t in lowered]
        if not any(counts):
            continue
        lines = [
            f"{path}:{i}:{line.strip()[:160]}"
            for i, line in enumerate(text.splitlines(), 1)
            if any(t in line.lower() for t in lowered)
        ]
        hits.append(GrepHit(path, sum(1 for c in counts if c), sum(counts), lines))
    return sorted(hits, key=lambda h: (-h.distinct, -h.matches, h.path.stat().st_size))


def grep_context(
    files: list[Path], question: str, *, open_n: int = GREP_OPEN
) -> tuple[str, int]:
    """Grep output the agent reads, plus the top files opened whole; (text, calls)."""
    hits = grep_rank(files, keywords(question))
    shown = [line for hit in hits for line in hit.lines][:GREP_LINES]
    opened = [h.path.read_text(errors="replace") for h in hits[:open_n]]
    return "\n".join(shown) + "\n\n" + "\n\n".join(opened), 1 + len(opened)


# ── the wiki route ────────────────────────────────────────────────────────────

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"pages": {"type": "array", "items": {"type": "string"}}},
    "required": ["pages"],
    "additionalProperties": False,
}


def wiki_context(question: str, complete) -> tuple[str, int]:
    """README plus the one or two pages a small model routes to; (text, calls)."""
    readme = (ROOT / "wiki" / "README.md").read_text()
    raw = complete(
        system="You route questions to pages of a project wiki. Answer with the "
        "file names (e.g. retrieval.md) of the one or two pages most likely to "
        "answer the question, best first.",
        user=f"{readme}\n\nQuestion: {question}",
        model=ROUTER_MODEL,
        schema=ROUTE_SCHEMA,
        schema_name="wiki_route",
    )
    pages = [p for p in raw.get("pages", [])[:2] if (ROOT / "wiki" / p).is_file()]
    texts = [(ROOT / "wiki" / p).read_text() for p in pages]
    return "\n\n".join([readme, *texts]), 1 + len(texts)


# ── the graph search ──────────────────────────────────────────────────────────


def tool_result(found) -> str:
    """What the search tool hands the agent (``retrieve.format_passages``)."""
    from src.query import retrieve as R

    return R.format_passages(found)


def passage_path(title: str, root: Path = ROOT, vault: Path = VAULT) -> Path | None:
    """The file behind a passage title, for the agent that opens it next."""
    if title.startswith(GITHUB_TITLE_PREFIX):
        return root / title[len(GITHUB_TITLE_PREFIX) :]
    candidate = vault / title
    return candidate if candidate.is_file() else None


def search(vector: list[float], question: str, *, passages: int, text: int = 0):
    """One walled retrieval, as the Slack path does it; ``text`` adds BM25 hits."""
    import psycopg

    from src.query import retrieve as R

    with psycopg.connect(DSN) as conn, conn.transaction():
        cur = conn.cursor()
        R.enter(cur, tenant=TENANT)
        return R.retrieve(cur, vector, question, passages=passages, text=text)


# ── grading ───────────────────────────────────────────────────────────────────

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]}
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


def graded(question: str, context: str, gold_answer: str, complete) -> str:
    """Answer from the context alone, then grade against the gold answer."""
    answered = complete(
        system="Answer the question using only the material provided. If it is "
        "not there, say exactly: not in the material. Two sentences at most.",
        user=f"Material:\n{context[:60000]}\n\nQuestion: {question}",
        model=ANSWER_MODEL,
        schema=ANSWER_SCHEMA,
        schema_name="agent_answer",
    )
    verdict = complete(
        system="Grade an answer against a reference. correct: states the "
        "reference's key facts; partial: some of them, nothing false; wrong: "
        "missing, contradicts, or says the material lacks it.",
        user=f"Question: {question}\nReference: {gold_answer}\n"
        f"Answer: {answered.get('answer', '')}",
        model=GRADER_MODEL,
        schema=GRADE_SCHEMA,
        schema_name="grade",
    )
    return verdict.get("verdict", "wrong")


def cached_complete(cache: dict):
    """The stage-2 JSON completion, remembered by prompt so re-runs are free."""
    from src.inject import chains as CH

    def complete(**kw) -> dict:
        key = hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()
        if key not in cache:
            cache[key] = CH._complete_json(**kw)
        return cache[key]

    return complete


def cached_embedding(cache: dict, question: str) -> list[float]:
    from src.query import retrieve as R

    key = "embed:" + hashlib.sha1(question.encode()).hexdigest()
    if key not in cache:
        cache[key] = R.embed_texts([question])[0]
    return cache[key]


# ── the run ───────────────────────────────────────────────────────────────────


class Row(NamedTuple):
    id: str
    category: str
    condition: str
    tokens: int
    sufficient: bool
    seconds: float
    grade: str | None


def measure(text: str, calls: int, gold: list[str]) -> tuple[int, bool]:
    return count_tokens(text) + calls * CALL_OVERHEAD, sufficient(text, gold)


def contexts_for(
    q: dict, files: list[Path], cache: dict
) -> dict[str, tuple[str, int, float]]:
    """What each strategy would put in front of the agent: (text, tool calls, seconds)."""
    complete = cached_complete(cache)
    contexts: dict[str, tuple[str, int, float]] = {}

    gold_texts = [gold_path(f).read_text(errors="replace") for f in q["gold_files"]]
    contexts["read_gold"] = ("\n\n".join(gold_texts), len(gold_texts), 0.0)
    text, calls = grep_context(files, q["question"])
    contexts["grep_read"] = (text, calls, 0.0)
    text, calls = wiki_context(q["question"], complete)
    contexts["wiki"] = (text, calls, 0.0)

    vector = cached_embedding(cache, q["question"])
    for name, k, text in (
        ("search_3", 3, 0),
        ("search_5", 5, 0),
        ("search_8", 8, 0),
        ("hybrid_5", 5, TEXT_CANDIDATES),
        ("hybrid_8", 8, TEXT_CANDIDATES),
    ):
        t0 = time.perf_counter()
        found = search(vector, q["question"], passages=k, text=text)
        contexts[name] = (tool_result(found), 1, time.perf_counter() - t0)
        if name in ("search_3", "hybrid_5"):
            opened = passage_path(found.passages[0][0]) if found.passages else None
            extra = (
                opened.read_text(errors="replace")
                if opened and opened.is_file()
                else ""
            )
            open_name = "search_open" if name == "search_3" else "hybrid_open"
            contexts[open_name] = (contexts[name][0] + "\n\n" + extra, 2, 0.0)

    return contexts


def score(q: dict, contexts: dict, complete, *, grade: bool) -> list[Row]:
    """Tokens and sufficiency per strategy; with ``grade``, the graded answer too."""
    rows = []
    for condition, (text, calls, seconds) in contexts.items():
        tokens, ok = measure(text, calls, q["gold"])
        verdict = (
            graded(q["question"], text, q["gold_answer"], complete) if grade else None
        )
        rows.append(
            Row(q["id"], q["category"], condition, tokens, ok, seconds, verdict)
        )
    return rows


def summarise(rows: list[Row], *, grade: bool) -> str:
    conditions = list(dict.fromkeys(r.condition for r in rows))
    lines = [
        f"{'condition':12} {'n':>3} {'mean tok':>9} {'median':>7} {'p90':>7} "
        f"{'sufficient':>10}"
        + (f" {'correct':>8} {'partial':>8}" if grade else "")
        + f" {'tok/sufficient':>14}"
    ]
    for c in conditions:
        sub = [r for r in rows if r.condition == c]
        toks = sorted(r.tokens for r in sub)
        ok = sum(r.sufficient for r in sub)
        p90 = toks[min(len(toks) - 1, int(0.9 * len(toks)))]
        line = (
            f"{c:12} {len(sub):3} {statistics.mean(toks):9.0f} "
            f"{statistics.median(toks):7.0f} {p90:7} {ok:4}/{len(sub):<5}"
        )
        if grade:
            correct = sum(r.grade == "correct" for r in sub)
            partial = sum(r.grade == "partial" for r in sub)
            line += f" {correct:8} {partial:8}"
        line += f" {sum(toks) / max(ok, 1):14.0f}"
        lines.append(line)
    return "\n".join(lines)


def by_category(rows: list[Row], conditions: tuple[str, ...]) -> str:
    cats = list(dict.fromkeys(r.category for r in rows))
    lines = [f"{'category':10}" + "".join(f" {c:>22}" for c in conditions)]
    for cat in cats:
        line = f"{cat:10}"
        for c in conditions:
            sub = [r for r in rows if r.category == cat and r.condition == c]
            ok = sum(r.sufficient for r in sub)
            line += f" {statistics.mean(r.tokens for r in sub):9.0f} tok {ok:2}/{len(sub):<2} ok"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grade", action="store_true", help="LLM-graded answers too")
    parser.add_argument("--show", action="store_true", help="one line per question")
    parser.add_argument("--category", help="only questions of this category")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()

    sys.path.insert(0, str(PUBLIC))
    questions = json.loads(args.questions.read_text())["questions"]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    files = corpus_files()
    print(
        f"{len(questions)} questions, {len(files)} files a grep would walk", flush=True
    )

    # Retrieval first for every question, then grading: the index must not move
    # between the first and the last question, and grading is the slow part.
    gathered = []
    for q in questions:
        gathered.append((q, contexts_for(q, files, cache)))
        CACHE.write_text(json.dumps(cache))
    complete = cached_complete(cache)
    rows: list[Row] = []
    for q, contexts in gathered:
        rows.extend(score(q, contexts, complete, grade=args.grade))
        CACHE.write_text(json.dumps(cache))
        if args.show:
            for r in rows[-10:]:
                mark = "ok " if r.sufficient else "-- "
                grade = f" {r.grade}" if r.grade else ""
                print(f"  {r.id:4} {r.condition:12} {r.tokens:7} tok {mark}{grade}")
    print()
    print(summarise(rows, grade=args.grade))
    print()
    print(
        by_category(rows, ("grep_read", "wiki", "search_5", "hybrid_5", "hybrid_open"))
    )
    print()
    for c in ("search_5", "hybrid_5"):
        seconds = [r.seconds for r in rows if r.condition == c]
        print(f"{c} retrieval: mean {statistics.mean(seconds):.2f}s per question")
    RESULTS.write_text(json.dumps([r._asdict() for r in rows], indent=1))
    print(f"rows -> {RESULTS}")


if __name__ == "__main__":
    main()
