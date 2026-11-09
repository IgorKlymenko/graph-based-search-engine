"""A real agent, three tool kits, the same questions: what does an answer cost?

The static harness (``eval_agent_tokens.py``) measures what each strategy
would put in front of an agent. This one lets a model decide: gpt-4.1-mini
gets a question and a tool kit, calls tools until it answers, and every token
it is billed for is counted -- prompt tokens of every turn (the context is
re-sent each call), completion tokens, and the tool results inside them.

Kits:
    files    glob, grep, read                 what a coding agent has today
    both     glob, grep, read, search         the graph search as one more tool
    search   search, read                     search first; read only to confirm
    trust    search, read + a line saying the passages are sufficient evidence

Sets (``--sets``): hand (``agent_questions.json``), para (the same questions
with identifiers removed), gen (model-written from sampled passages). Answers
are graded by gpt-4.1 against each question's gold answer. Trajectories are
cached by their inputs, so a re-run with the same index is free.

    uv run python scripts/eval_agent_loop.py [--kits files,both,search] [--sets hand,para,gen]
                                             [--workers 6] [--limit N] [--show]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent
ROOT = PUBLIC.parent.parent
VAULT = ROOT / "old_server" / "artifacts" / "vault_notes"
CACHE = HERE / "eval_agent_loop.cache.json"
RESULTS = HERE / "eval_agent_loop.results.json"
load_dotenv(PUBLIC / ".env")
sys.path.insert(0, str(PUBLIC))

TENANT = "dddddddd-0000-4000-8000-00000000000d"
DSN = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
CORPUS_ROOTS = ("server/public", "ADRs", "CLAUDE.md")
CORPUS_SUFFIXES = frozenset({".py", ".md"})
SKIP_PARTS = frozenset({".venv", "__pycache__", "node_modules", ".git", ".obsidian"})
AGENT_MODEL = "gpt-4.1-mini"
GRADER_MODEL = "gpt-4.1"
MAX_STEPS = 8
GREP_LINES = 50
READ_LINES = 400
GLOB_LIMIT = 60
SETS = {
    "hand": HERE / "agent_questions.json",
    "para": HERE / "agent_questions_paraphrased.json",
    "gen": HERE / "agent_questions_generated.json",
}

SYSTEM = """You answer questions about one software project using tools. The project is a Slack assistant (Nazar) that builds a knowledge graph from a founder's notes and from source code; Python; Postgres with row-level security.

Layout of what you can read:
  server/public/src/      the live code (inject/ = ingest pipeline, query/ = retrieval and answers, connectors/ = code and GitHub connectors, app.py = Slack ingress)
  server/public/scripts/  evaluation scripts
  ADRs/                   decision records;  CLAUDE.md  project rules
  vault/                  the founder's notes (markdown)

Find evidence with the tools; do not guess. Prefer few, targeted calls. When you have the answer, reply in at most two sentences and name the file or note it came from. If a reasonable search finds nothing, reply exactly: not found."""

TOOLS_ALL = {
    "glob": {
        "description": "List files whose path matches a glob, e.g. server/public/src/**/*.py or vault/**/Day*.md.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    "grep": {
        "description": "Search file contents with a case-insensitive regular expression. Returns path:line:text, at most 50 lines. Optional path_glob narrows the files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path_glob": {"type": ["string", "null"]},
            },
            "required": ["pattern", "path_glob"],
            "additionalProperties": False,
        },
    },
    "read": {
        "description": "Read a file with line numbers. Whole file by default (long files are cut at 400 lines); give start and end lines to read a range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": ["integer", "null"]},
                "end": {"type": ["integer", "null"]},
            },
            "required": ["path", "start", "end"],
            "additionalProperties": False,
        },
    },
    "search": {
        "description": "Search the project's knowledge graph (notes and code) with a natural-language question. Returns the most relevant passages with their sources.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}
KITS = {
    "files": ("glob", "grep", "read"),
    "both": ("glob", "grep", "read", "search"),
    "search": ("search", "read"),
    "trust": ("search", "read"),  # same tools; told the passages are evidence
}
TRUST = """

Passages returned by search are verbatim from the notes and code and are sufficient evidence: answer from them and cite the passage's source. Open a file only when the passages leave the question unanswered."""


# ── the corpus as files ───────────────────────────────────────────────────────


def corpus_files(root: Path = ROOT, vault: Path = VAULT) -> dict[str, Path]:
    """Relative name -> path: repo files under the indexed roots, notes under vault/."""
    out: dict[str, Path] = {}
    for rel in CORPUS_ROOTS:
        base = root / rel
        if base.is_file():
            out[rel] = base
            continue
        for p in base.rglob("*"):
            if p.suffix in CORPUS_SUFFIXES and not SKIP_PARTS & set(p.parts):
                out[p.relative_to(root).as_posix()] = p
    for p in vault.rglob("*.md"):
        if not SKIP_PARTS & set(p.parts):
            out["vault/" + p.relative_to(vault).as_posix()] = p
    return dict(sorted(out.items()))


class Tools:
    """The tool kit over one corpus; ``search`` goes to the graph."""

    def __init__(self, files: dict[str, Path], searcher=None) -> None:
        self.files = files
        self.searcher = searcher

    def glob(self, pattern: str) -> str:
        names = [n for n in self.files if fnmatch.fnmatch(n, pattern)]
        if not names:
            return "no files match"
        shown = names[:GLOB_LIMIT]
        more = (
            f"\n... {len(names) - len(shown)} more" if len(names) > len(shown) else ""
        )
        return "\n".join(shown) + more

    def grep(self, pattern: str, path_glob: str | None = None) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            return f"bad pattern: {error}"
        lines: list[str] = []
        total = 0
        for name, path in self.files.items():
            if path_glob and not fnmatch.fnmatch(name, path_glob):
                continue
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(lines) < GREP_LINES:
                        lines.append(f"{name}:{i}:{line.strip()[:200]}")
        if not lines:
            return "no matches"
        more = f"\n... {total - len(lines)} more matches" if total > len(lines) else ""
        return "\n".join(lines) + more

    def read(self, path: str, start: int | None = None, end: int | None = None) -> str:
        target = self.files.get(path.strip("/"))
        if target is None:
            return f"no such file: {path}"
        lines = target.read_text(errors="replace").splitlines()
        lo = max(1, start or 1)
        hi = min(len(lines), end or len(lines))
        if end is None and hi - lo + 1 > READ_LINES:
            hi = lo + READ_LINES - 1
            note = f"\n... {len(lines) - hi} more lines; read a range to see them"
        else:
            note = ""
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1)) + note

    def search(self, question: str) -> str:
        return self.searcher(question) if self.searcher else "search is not available"

    def call(self, name: str, arguments: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None:
            return f"unknown tool {name}"
        try:
            return fn(**arguments)
        except TypeError as error:
            return f"bad arguments: {error}"


def graph_search(question: str) -> str:
    from src.query import retrieve as R

    found = R.search_passages(question, dsn=DSN, tenant=TENANT, passages=8)
    return R.format_passages(found)


# ── the loop ──────────────────────────────────────────────────────────────────


class Run(NamedTuple):
    set: str
    id: str
    category: str
    kit: str
    answer: str
    steps: int
    calls: list[str]
    prompt_tokens: int
    completion_tokens: int
    tool_tokens: int
    seconds: float
    grade: str | None


def tool_specs(kit: str) -> list[dict]:
    return [
        {"type": "function", "function": {"name": n, "strict": True, **TOOLS_ALL[n]}}
        for n in KITS[kit]
    ]


def count_tokens(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("o200k_base").encode(text))


def run_agent(client, question: str, kit: str, tools: Tools) -> dict:
    """One question, one kit: the answer and everything it cost."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM + (TRUST if kit == "trust" else "")},
        {"role": "user", "content": question},
    ]
    specs = tool_specs(kit)
    prompt = completion = tool_tokens = 0
    calls: list[str] = []
    t0 = time.perf_counter()
    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=specs,
            tool_choice="none" if last else "auto",
            temperature=0,
        )
        prompt += response.usage.prompt_tokens
        completion += response.usage.completion_tokens
        message = response.choices[0].message
        if not message.tool_calls:
            return {
                "answer": (message.content or "").strip(),
                "steps": step,
                "calls": calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "tool_tokens": tool_tokens,
                "seconds": time.perf_counter() - t0,
            }
        messages.append(message)
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = tools.call(tc.function.name, arguments)
            tool_tokens += count_tokens(result)
            calls.append(f"{tc.function.name}({json.dumps(arguments)[:80]})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    raise AssertionError("unreachable")


GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]}
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


def grade(question: str, answer: str, gold_answer: str, complete) -> str:
    return complete(
        system="Grade an answer against a reference. correct: states the reference's "
        "key facts; partial: some of them, nothing false; wrong: missing, contradicts, "
        "or says it was not found.",
        user=f"Question: {question}\nReference: {gold_answer}\nAnswer: {answer}",
        model=GRADER_MODEL,
        schema=GRADE_SCHEMA,
        schema_name="grade",
    ).get("verdict", "wrong")


def cached(cache: dict, key: str, produce):
    if key not in cache:
        cache[key] = produce()
    return cache[key]


def key_for(*parts: object) -> str:
    return hashlib.sha1(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()


# ── reporting ─────────────────────────────────────────────────────────────────


def billed(r: Run) -> int:
    return r.prompt_tokens + r.completion_tokens


def bootstrap_mean(
    values: list[float], n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarise(runs: list[Run]) -> str:
    kits = list(dict.fromkeys(r.kit for r in runs))
    sets = list(dict.fromkeys(r.set for r in runs))
    lines = [
        (
            f"{'set':5} {'kit':7} {'n':>3} {'billed tok':>10} {'95% CI':>17} {'median':>7} "
            f"{'tool tok':>8} {'calls':>5} {'correct':>7} {'partial':>7} {'wrong':>5} {'tok/correct':>11}"
        )
    ]
    for s in [*sets, "all"]:
        for k in kits:
            sub = [r for r in runs if r.kit == k and (s == "all" or r.set == s)]
            if not sub:
                continue
            tok = [billed(r) for r in sub]
            lo, hi = bootstrap_mean(tok)
            correct = sum(r.grade == "correct" for r in sub)
            partial = sum(r.grade == "partial" for r in sub)
            lines.append(
                f"{s:5} {k:7} {len(sub):3} {statistics.mean(tok):10.0f} "
                f"{lo:8.0f}–{hi:<8.0f} {statistics.median(tok):7.0f} "
                f"{statistics.mean(r.tool_tokens for r in sub):8.0f} "
                f"{statistics.mean(r.steps for r in sub):5.1f} {correct:7} {partial:7} "
                f"{len(sub) - correct - partial:5} {sum(tok) / max(correct, 1):11.0f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kits", default="files,both,search")
    parser.add_argument("--sets", default="hand,para,gen")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="questions per set")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from src.inject import chains as CH

    client = CH._get_client()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    files = corpus_files()
    tools = Tools(files, graph_search)
    jobs = []
    for s in args.sets.split(","):
        questions = json.loads(SETS[s].read_text())["questions"][: args.limit]
        jobs.extend((s, q, k) for q in questions for k in args.kits.split(","))
    print(f"{len(jobs)} runs over {len(files)} files", flush=True)

    def one(job) -> Run:
        s, q, k = job
        traj = cached(
            cache,
            key_for("traj", AGENT_MODEL, k, q["question"], MAX_STEPS),
            lambda: run_agent(client, q["question"], k, tools),
        )
        verdict = cached(
            cache,
            key_for(
                "grade", GRADER_MODEL, q["question"], traj["answer"], q["gold_answer"]
            ),
            lambda: grade(
                q["question"], traj["answer"], q["gold_answer"], CH._complete_json
            ),
        )
        return Run(
            s,
            q["id"],
            q["category"],
            k,
            traj["answer"],
            traj["steps"],
            traj["calls"],
            traj["prompt_tokens"],
            traj["completion_tokens"],
            traj["tool_tokens"],
            traj["seconds"],
            verdict,
        )

    runs: list[Run] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, run in enumerate(pool.map(one, jobs), 1):
            runs.append(run)
            if args.show:
                print(
                    f"  {run.set:4} {run.id:5} {run.kit:7} {billed(run):7} tok {run.steps} steps "
                    f"{run.grade:8} {run.answer[:70]!r}",
                    flush=True,
                )
            if i % 10 == 0:
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))
    print()
    print(summarise(runs))
    RESULTS.write_text(json.dumps([r._asdict() for r in runs], indent=1))
    print(f"runs -> {RESULTS}")


if __name__ == "__main__":
    main()
