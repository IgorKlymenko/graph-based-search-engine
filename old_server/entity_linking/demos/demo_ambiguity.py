"""Show, step by step, how one contested name is kept from fusing two people.

    uv run python entity_linking/demos/demo_ambiguity.py

Two people share the short form "S. Chen". Nothing in the name can separate them,
and three different mechanisms have to cooperate to get it right:

  1. a witness episode naming both in *different* chains marks the surface contested
  2. the contested surface stops carrying evidence AND stops matching itself
  3. what is left over is handed to the model with the graph built so far

Every step prints what it saw and what it decided. Real LLM calls throughout.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from entity_linking import chains as CH  # noqa: E402
from entity_linking import episodes as EP  # noqa: E402
from entity_linking import pairing as PR  # noqa: E402
from entity_linking import pipeline as PL  # noqa: E402

TENANT_ID = "00000000-0000-0000-0000-000000000001"

DOC = """Sarah Chen leads the identity migration at Workvivo. S. Chen signed \
off the rollback plan before the freeze.

Steven Chen owns the billing rollout. S. Chen escalated the pricing defect \
to the vendor.

Sarah Chen and Steven Chen both joined the quarterly review meeting.

S. Chen reviewed the Okta configuration for the identity project and \
approved the change.

S. Chen published the pricing note for the billing team."""

RULE = "─" * 78


def heading(step: str, title: str) -> None:
    print(f"\n{RULE}\n{step}  {title}\n{RULE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    args = parser.parse_args()

    store = EP.empty_store()
    diff = EP.sync_file(store, DOC, tenant_id=TENANT_ID,
                        source=EP.UPLOAD_SOURCE, external_id="chens.md",
                        max_chars=150)
    order = store["files"][diff.file_id]["episode_ids"]
    label = {eid: f"ep{i + 1}" for i, eid in enumerate(order)}

    heading("1.", "the corpus")
    for eid in order:
        print(f"  {label[eid]}  {store['episodes'][eid]['text']}")

    CH.run(store, model=args.model, workers=8)
    records = PL.encode(store)
    by_id = {r.chain_id: r for r in records}

    heading("2.", "chains per episode  (what one LLM call saw in one passage)")
    for eid in order:
        here = [r for r in records if r.episode_id == eid]
        names = "  ".join("{" + ", ".join(sorted(r.surfaces)) + "}"
                          for r in here if any("Chen" in s for s in r.surfaces))
        print(f"  {label[eid]}  {names or '(no Chen mention)'}")

    evidence = PR.build_evidence(records)
    heading("3.", "evidence  (why 'S. Chen' cannot be trusted)")
    witness = [p for p in evidence.split_in_episode if p == ("Sarah Chen",
                                                             "Steven Chen")]
    print("  kept apart in one episode :", witness or "NONE")
    print("  contested surfaces        :", sorted(evidence.ambiguous_surfaces))
    print("  surviving alias edges     :",
          sorted(p for p in evidence.same_chain if "Chen" in str(p)) or "none")
    print("\n  Without the ep3 witness nothing marks the conflict and the two")
    print("  people fuse silently. The safeguard needs a passage naming both.")

    heading("4.", "scoring  (the contested surface no longer matches itself)")
    scorer = PR.PairScorer()
    thresholds = PR.Thresholds(low=0.30, high=0.50)
    scored = []
    for a, b in itertools.combinations(records, 2):
        if a.episode_id == b.episode_id:
            continue
        feats = PR.features(a, b, evidence)
        scored.append((a, b, feats, scorer.score(feats)))

    for a, b, feats, probability in scored:
        if not (any("Chen" in s for s in a.surfaces)
                and any("Chen" in s for s in b.surfaces)):
            continue
        raw = PR._string_similarity(a, b)
        print(f"  {sorted(a.surfaces)} ~ {sorted(b.surfaces)}")
        print(f"     string_similarity {raw:.2f} raw -> "
              f"{feats['string_similarity']:.2f} with contested surfaces removed")
        print(f"     p={probability:.2f} -> {PR.decide(probability, thresholds)}")

    merges = [(a.chain_id, b.chain_id, p) for a, b, _f, p in scored
              if PR.decide(p, thresholds) == PR.MERGE]
    assignment = PR.cluster([r.chain_id for r in records], merges)
    pending = PR.unresolved(records, assignment, evidence)

    heading("5.", "what is left over  (contested name, nothing to anchor it)")
    for chain_id, clusters in pending.items():
        record = by_id[chain_id]
        print(f"  {label[record.episode_id]}  {sorted(record.surfaces)} "
              f"-> no node created; candidates {clusters}")
    if not pending:
        print("  none — every chain had a name that identifies on its own")

    passages = {eid: ep["text"] for eid, ep in store["episodes"].items()}
    decided = PL.resolve_against_graph(records, assignment, pending, passages,
                                       model=args.model) if pending else {}

    heading("6.", "the model decides, using the graph so far")
    for chain_id, result in decided.items():
        record = by_id[chain_id]
        print(f"  mention  {sorted(record.surfaces)} in "
              f"{label[record.episode_id]}")
        print(f"     '{passages[record.episode_id]}'")
        for cluster in result["candidates"]:
            members = [r for r in records if assignment[r.chain_id] == cluster]
            print(f"     [{cluster}] {PL._profile(members, passages)[:180]}")
        answer = result["answer"]
        chosen = result["cluster"]
        print(f"     -> {'entity ' + str(chosen) if chosen is not None else 'DECLINED'}"
              f"  (confidence {answer.get('confidence', 0):.2f})")
        print(f"     reason: {answer.get('reason', '')}")

    for chain_id, result in decided.items():
        if result["cluster"] is not None:
            assignment[chain_id] = result["cluster"]
    still_pending = {k: v for k, v in pending.items()
                     if decided.get(k, {}).get("cluster") is None}

    heading("7.", "entities")
    grouped: dict[int, list[PR.ChainRecord]] = {}
    for record in records:
        if record.chain_id in still_pending:
            continue
        grouped.setdefault(assignment[record.chain_id], []).append(record)
    for cluster, members in sorted(grouped.items()):
        surfaces = sorted({s for m in members for s in m.surfaces})
        if not any("Chen" in s for s in surfaces):
            continue
        print(f"  entity {cluster}: {surfaces}")
        for member in members:
            print(f"     {label[member.episode_id]}  "
                  f"{passages[member.episode_id][:64]}…")
    for chain_id in still_pending:
        record = by_id[chain_id]
        print(f"  unresolved: {sorted(record.surfaces)} in "
              f"{label[record.episode_id]} (no node, kept for later evidence)")


if __name__ == "__main__":
    main()
