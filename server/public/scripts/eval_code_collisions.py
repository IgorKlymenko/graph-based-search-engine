"""Do short-name collisions fuse code symbols, and which rendering stops it?

Ground truth is free here: every seeded chain carries the qualified name it
came from, so a merge between two chains with different keys is a false merge
and a split between two with the same key is a miss. The rig renders a repo
under several variants, builds seed-only chains (no model call), encodes the
heads with nomic, and runs the linker's blocking, evidence and prior scorer
exactly as stage 4 does. Pairs in the review band are counted, not sent.

    uv run python scripts/eval_code_collisions.py src ../../old_server/entity_linking \\
        .venv/lib/python3.13/site-packages/httpx

Variants:
    baseline    the shipped renderer
    qualify     references to ambiguous short names render qualified (C17)
    always      every reference renders qualified
    defs        qualify, and an ambiguous *definition* shows its short name as
                plain text rather than a mention
    classdot    methods render as Class.method everywhere (PEP 8 capitalised owner)
    index       baseline plus one synthetic document that lists each colliding
                name's definitions as separate chains in one episode, so the
                short form is contested by evidence rather than by rendering
    a+b         combinations; ``all`` is defs+classdot+index
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.connectors import base
from src.connectors.code import ambiguous_names, parser_for, render
from src.connectors.code import render as RENDER
from src.connectors.code.render import Style, short
from src.inject import chains as CH
from src.inject import episodes as EP
from src.inject import main as MAIN
from src.inject import pairing as PR

TENANT = "eeeeeeee-0000-4000-8000-00000000000e"
THRESHOLDS = PR.Thresholds(low=0.30, high=0.50)
CACHE = Path(__file__).with_suffix(".vectors.npz")


def parse_repo(root: Path):
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    tree = frozenset(p.relative_to(root).as_posix() for p in files)
    out = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            out.append((rel, parser_for(rel).parse(rel, p.read_text(errors="replace"),
                                                    tree=tree)))
        except (SyntaxError, ValueError, RecursionError):
            continue
    return out


def index_document(parsed_files, ambiguous) -> base.Document:
    text, mentions = RENDER.index_document(
        [RENDER.names_of(p) for _, p in parsed_files], ambiguous)
    return base.Document("repo/index", text, mentions=mentions)


def documents(parsed_files, variant: str):
    ambiguous = ambiguous_names([p for _, p in parsed_files])
    parts = set(variant.split("+"))
    use = ambiguous if parts & {"qualify", "defs", "all"} else frozenset()
    if "always" in parts:
        use = frozenset(short(r) for _, p in parsed_files
                        for s in p.symbols for r in s.references) | ambiguous
    style = Style(ambiguous=use,
                  short_ambiguous_definitions=not (parts & {"defs", "all"}),
                  class_dot_methods=bool(parts & {"classdot", "all"}))
    for rel, parsed in parsed_files:
        text, mentions = render(parsed, path=rel, style=style)
        yield base.Document(f"repo/{rel}", text, mentions=mentions)
    if parts & {"index", "all"}:
        yield index_document(parsed_files, ambiguous)


def build_store(docs) -> tuple[dict, dict[str, str]]:
    """Seed-only chains: what stage 2 would keep if the model added nothing."""
    store = EP.empty_store()
    for doc in docs:
        base.sync_document(store, doc, tenant_id=TENANT, source="github")
    key_of: dict[str, str] = {}
    for eid, episode in store["episodes"].items():
        seeded = CH.seeded_chains(episode)
        keys = [seed["key"] for seed in episode.get("seeds", [])]
        resolved = CH.resolve_chains({"table": {}, "raw_chains": []}, seeded)
        for chain in CH.build_chains(eid, resolved):
            store["chains"][chain["chain_id"]] = chain
            surfaces = {s.text for c in resolved for s in c
                        if CH.chain_id_for(eid, CH.content_key(c) or "") == chain["chain_id"]}
            # a chain's key: the seed whose spans it holds (union may join two
            # seeds sharing an identical proper name -- then it is ambiguous)
            owners = {k for k, seed in zip(keys, episode.get("seeds", []))
                      if any(sp[2] in surfaces for sp in seed["spans"])}
            key_of[chain["chain_id"]] = min(owners) if len(owners) == 1 else "|".join(sorted(owners))
        for cand in CH.build_candidates(eid, resolved):
            store["candidates"][cand["candidate_id"]] = cand
        episode["extracted_at"] = "rig"
    return store, key_of


def evaluate(records, key_of):
    evidence = PR.build_evidence(records)
    scorer = PR.PairScorer()
    counts = Counter()
    merges = []
    for a, b in PR.candidate_pairs(records):
        p = scorer.score(PR.features(a, b, evidence))
        same = key_of[a.chain_id] == key_of[b.chain_id]
        verdict = PR.decide(p, THRESHOLDS)
        counts[(verdict, "same" if same else "diff")] += 1
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, p))
    assignment = PR.cluster([r.chain_id for r in records], merges)
    clusters: dict[int, set[str]] = defaultdict(set)
    for cid, c in assignment.items():
        clusters[c].add(key_of[cid])
    impure = sum(1 for keys in clusters.values() if len(keys) > 1)
    by_key: dict[str, set[int]] = defaultdict(set)
    for cid, c in assignment.items():
        by_key[key_of[cid]].add(c)
    fragmented = sum(1 for cs in by_key.values() if len(cs) > 1)
    return counts, len(clusters), impure, fragmented, len(evidence.ambiguous_surfaces)


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]]
    variants = ["baseline", "qualify", "always", "defs", "classdot", "index",
                "qualify+index", "defs+classdot", "all"]
    cache = MAIN.load_vector_cache(CACHE)
    print(f"{'repo':10} {'variant':14} {'chains':>6} {'pairs':>7} {'merge ok/bad':>13} "
          f"{'review ok/bad':>14} {'split ok/bad':>13} {'ents':>5} {'impure':>6} "
          f"{'frag':>5} {'contested':>9}")
    for root in roots:
        parsed_files = parse_repo(root)
        for variant in variants:
            store, key_of = build_store(documents(parsed_files, variant))
            records = MAIN.encode(store, context=False, cache=cache)
            MAIN.save_vector_cache(CACHE, cache)
            counts, ents, impure, frag, contested = evaluate(records, key_of)
            pairs = sum(counts.values())
            row = (f"{root.name:10} {variant:14} {len(records):6} {pairs:7} "
                   f"{counts[('merge','same')]:6}/{counts[('merge','diff')]:<6} "
                   f"{counts[('review','same')]:6}/{counts[('review','diff')]:<7} "
                   f"{counts[('split','same')]:6}/{counts[('split','diff')]:<6} "
                   f"{ents:5} {impure:6} {frag:5} {contested:9}")
            print(row, flush=True)


if __name__ == "__main__":
    main()
