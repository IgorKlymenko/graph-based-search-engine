"""Walk one episode through the span pipeline, dumping every intermediate step.

    uv run python entity_linking/demo_chains.py                    # stubbed LLM
    uv run python entity_linking/demo_chains.py --live             # real call
    uv run python entity_linking/demo_chains.py --dump /tmp/steps.json

Without ``--live`` the model response is a canned fixture, so the parser,
marker injection, id validation and merge are all exercised offline. The dump
holds one entry per step so each stage can be inspected on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import spans as SP  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage1", Path(__file__).parent / "1_get_candidates.py"
)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

# Replace this with any text you want to inspect; the stub below adapts to it.
EPISODE = ("Supabase is a combination of open source tools. We’re building the features of Firebase using enterprise-grade, open source products. If the tools and communities exist, with an MIT, Apache 2, or equivalent open license, we will use and support that tool. If the tool doesn't exist, we build and open source it ourselves. Supabase is not a 1-to-1 mapping of Firebase. Our aim is to give developers a Firebase-like developer experience using open source tools.")


def build_stub(episode: str) -> dict:
    """A deliberately imperfect response, derived from whatever EPISODE holds.

    Exercises every branch of validation without a network call: a verbatim
    span, one the model reflowed, one it typo'd, one it invented, one that is
    ambiguous, and a chain naming an id that was never issued.
    """
    words = re.findall(r"[A-Za-z][\w-]*", episode)
    phrases = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
    unique = next((p for p in phrases if episode.count(p) == 1), words[0])
    repeated = next((p for p in phrases if episode.count(p) > 1), None)

    new_spans = [
        {"id": "n0", "text": unique, "candidate_type": "proper_name"},
        {"id": "n1", "text": unique.replace(" ", "  ", 1),
         "candidate_type": "proper_name"},                    # reflowed
        {"id": "n2", "text": unique[:-1] + "x" + unique[-1:],
         "candidate_type": "proper_name"},                    # typo
        {"id": "n3", "text": "Zendesk Enterprise Portal",
         "candidate_type": "proper_name"},                    # invented
    ]
    if repeated:
        new_spans.append({"id": "n4", "text": repeated,
                          "candidate_type": "nominal"})       # ambiguous
    return {
        "new_spans": new_spans,
        "chains": [
            {"head": "c0", "spans": ["c0", "n0"]},
            {"head": "n1", "spans": ["n1"]},
            {"head": "c999", "spans": ["c999"]},              # never issued
        ],
    }


def dump_step(steps: list[dict], name: str, note: str, payload) -> None:
    steps.append({"step": name, "note": note, "payload": payload})
    print(f"\n=== {name} — {note} ===")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--live", action="store_true", help="Call the real model")
    p.add_argument("--dump", type=Path, default=None, help="Write steps to JSON")
    p.add_argument("--model", default=S.EXTRACTION_MODEL)
    args = p.parse_args()

    steps: list[dict] = []

    dump_step(steps, "0_episode", "clean text — the only thing offsets refer to",
              EPISODE)
    print(EPISODE)

    parser_spans = SP.parse_spans(EPISODE)
    dump_step(steps, "1_parser_spans",
              f"{len(parser_spans)} spans found deterministically",
              [s._asdict() for s in parser_spans])
    for s in parser_spans:
        print(f"  {s.sid:>4} [{s.start:>3}:{s.end:<3}] "
              f"{s.candidate_type:<21} {s.text!r}")

    marked = SP.mark_text(EPISODE, parser_spans)
    dump_step(steps, "2_marked_copy", "what the model sees; original untouched",
              marked)
    print(marked)
    print(f"\n  offsets still valid against clean text: "
          f"{all(EPISODE[s.start:s.end] == s.text for s in parser_spans)}")

    if args.live:
        raw = S._complete_json(
            system=S.CHAIN_SYSTEM_PROMPT, user=marked, model=args.model,
            schema=S.CHAIN_SCHEMA, schema_name="coreference_chains",
        )
    else:
        raw = build_stub(EPISODE)
    dump_step(steps, "3_model_response",
              "live" if args.live else "stubbed, with deliberate faults", raw)
    print(json.dumps(raw, indent=2))

    issued = {s.sid for s in parser_spans}
    new_spans, dropped = SP.locate_new_spans(EPISODE, raw.get("new_spans", []),
                                             issued)
    dump_step(steps, "4_locate_new_spans",
              f"{len(new_spans)} located, {len(dropped)} dropped",
              {"kept": [s._asdict() for s in new_spans], "dropped": dropped})
    for s in new_spans:
        print(f"  KEPT    {s.sid} [{s.start}:{s.end}] {s.text!r}")
    for d in dropped:
        print(f"  DROPPED {d.get('id')} {d['text']!r} — {d['reason']}")

    declared = issued | {s.sid for s in new_spans}
    unverified = {d["id"] for d in dropped if d.get("id")}
    unknown = sorted({
        sid for chain in raw.get("chains", [])
        for sid in [chain.get("head", "")] + list(chain.get("spans", []))
        if sid and sid not in declared and sid not in unverified
    })
    dump_step(steps, "5_id_validation",
              "undeclared ids are a protocol error: retry once, then fail",
              {"issued": sorted(issued), "declared_new": sorted(
                  s.sid for s in new_spans), "unverified": sorted(unverified),
               "unknown": unknown})
    print(f"  issued by parser : {len(issued)}")
    print(f"  declared new     : {sorted(s.sid for s in new_spans)}")
    print(f"  unverified (drop): {sorted(unverified)}")
    print(f"  UNKNOWN (fail)   : {unknown or 'none'}")

    table, collapsed = SP.merge_spans(parser_spans, new_spans)
    dump_step(steps, "6_merge", f"{len(collapsed)} overlaps collapsed (longest wins)",
              {"collapsed": collapsed,
               "table": {k: v._asdict() for k, v in sorted(table.items())}})
    for c in collapsed:
        print(f"  {c['dropped']} -> {c['kept']}   {c['text']!r}")

    annotated = {"table": table, "raw_chains": [
        ch for ch in raw.get("chains", [])
        if not any(sid not in declared and sid not in unverified
                   for sid in [ch.get("head", "")] + list(ch.get("spans", [])))
    ]}
    chains = S.resolve_chains(annotated)
    dump_step(steps, "7_chains", f"{len(chains)} chains, heads resolved",
              [{"head": c[0].sid, "spans": [s.sid for s in c]} for c in chains])
    for chain in chains:
        head = chain[0]
        print(f"  head {head.sid} {head.text!r} ({head.candidate_type})")
        for s in chain[1:]:
            print(f"       {s.sid} {s.text!r}")

    episode_id = "6680d019-8ad1-571a-ba1c-b0bac7fc3559"
    records = S.build_candidates(episode_id, chains)
    dump_step(steps, "8_candidate_records", f"{len(records)} records", records)
    print(json.dumps(records[:2], indent=2))
    print(f"\n  every record slices back to the clean text: "
          f"{all(EPISODE[r['start']:r['end']] == r['text'] for r in records)}")

    if args.dump:
        args.dump.write_text(json.dumps(steps, indent=2, ensure_ascii=False))
        print(f"\nwrote {len(steps)} steps -> {args.dump} "
              f"({args.dump.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
