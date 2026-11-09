"""Stance accuracy and reply length of the live answer module on the vault store.

    uv run python scripts/eval_reframe.py [--repeat 3] [--show]

The twelve questions and expected stances are the same as
old_server/entity_linking/demos/eval_reframe.py (which scores the superseded
copy); passages come from the extracted vault store by keyword, so this
measures the stance decision and the format, not retrieval. Reports the
majority stance per question, flips across runs, and the answer length in
words -- the Slack format rules ask for under 120.
"""


from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.query import answer as AN

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# The extracted vault store is a gitignored artifact; override with --store.
STORE = ROOT.parent.parent / "old_server" / "artifacts" / "vault" / "store_nojudge.json"

CASES = [
    (
        "How should I scale the Neo4j Aura instance for the graph?",
        ["Neo4j", "supabase", "unscalable"],
        "reframe",
    ),
    (
        "How do I finish the facts extraction stage before the demo?",
        ["facts extraction", "Fuck facts"],
        "reframe",
    ),
    (
        "How do I run the MCP server locally for the demo?",
        ["MCP", "Railway", "locally"],
        "reframe",
    ),
    (
        "Why did we pick HDBSCAN clustering as the final approach?",
        ["HDBSCAN", "cluster"],
        "reframe",
    ),
    (
        "How do I keep the database always up so the cache stays warm?",
        ["always up", "warm", "cache"],
        "reframe",
    ),
    (
        "What are the four ACID guarantees Postgres gives?",
        ["ACID", "Atomicity"],
        "answer",
    ),
    (
        "What does the sloppy policy failure mean in RLS?",
        ["Sloppy", "policy widening"],
        "answer",
    ),
    (
        "Which accelerators in Toronto did I list?",
        ["Antler", "Techstars", "MaRS"],
        "answer",
    ),
    ("What is the CEC lookback window?", ["CEC", "lookback"], "answer"),
    ("What does the DIKW pyramid stand for?", ["DIKW", "Ackoff"], "answer"),
    ("How do I make a stateless JWT revocable?", ["JWT", "revocable"], "answer"),
    ("What did Tess say about procurement?", ["Tess", "procurement"], "answer"),
]


def passages_for(store: dict, keys: list[str], limit: int = 6) -> list[tuple[str, str]]:
    files = store["files"]
    scored = []
    for ep in store["episodes"].values():
        if ep.get("removed_at"):
            continue
        hits = sum(1 for k in keys if k.lower() in ep["text"].lower())
        if hits:
            scored.append((hits, -len(ep["text"]), ep))
    scored.sort(key=lambda t: (-t[0], t[1]))
    titles = AN.display_titles(
        [
            files.get(ep["file_id"], {}).get("external_id", "untitled")
            for _, _, ep in scored[:limit]
        ]
    )
    return [
        (
            titles[files.get(ep["file_id"], {}).get("external_id", "untitled")],
            ep["text"],
        )
        for _, _, ep in scored[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--model", default=AN.ANSWER_MODEL)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.store.is_file():
        parser.error(f"store not found: {args.store} (pass --store <path>)")
    store = json.loads(args.store.read_text())
    right = flips = 0
    words: list[int] = []
    rows = []
    for question, keys, expected in CASES:
        passages = passages_for(store, keys)
        verdicts = [
            AN.answer(question, passages, model=args.model) for _ in range(args.repeat)
        ]
        stances = Counter(v["stance"] for v in verdicts)
        got = stances.most_common(1)[0][0]
        flips += len(stances) > 1
        ok = got == expected
        right += ok
        words += [len(v["answer"].split()) for v in verdicts]
        rows.append(
            (
                question,
                expected,
                got,
                ok,
                dict(stances),
                len(verdicts[0]["answer"].split()),
            )
        )
        if args.show or not ok:
            text, _ = AN.render_slack_blocks(verdicts[0])
            print(
                f"--- {question}\n    expected {expected}, got {got} {dict(stances)}\n    "
                + text.replace("\n", "\n    ")[:700]
            )
    print(f"\n{'question':<62}{'expected':>9}{'got':>9}{'words':>7}")
    for q, e, g, ok, st, n in rows:
        print(f"{q[:60]:<62}{e:>9}{g:>9}{n:>7} {'' if ok else '<-- miss'}")
    reframes = [r for r in rows if r[1] == "reframe"]
    controls = [r for r in rows if r[1] == "answer"]
    print(
        f"\nstance accuracy {right}/{len(rows)}; reframed when it should {sum(1 for r in reframes if r[3])}/{len(reframes)}; "
        f"left a fine question alone {sum(1 for r in controls if r[3])}/{len(controls)}"
    )
    print(
        f"answer length: mean {sum(words) / len(words):.0f} words, max {max(words)}, over 120: {sum(1 for w in words if w > 120)}/{len(words)}"
    )
    if args.repeat > 1:
        print(f"stance flipped across {args.repeat} runs: {flips}/{len(rows)}")


if __name__ == "__main__":
    main()
