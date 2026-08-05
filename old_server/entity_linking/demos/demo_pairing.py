"""Compare three ways of deciding whether two chains are the same entity.

    uv run python entity_linking/demo_pairing.py            # stubbed coref
    uv run python entity_linking/demo_pairing.py --live     # real coref calls
    uv run python entity_linking/demo_pairing.py --dump /tmp/pairs.json

Three scorers over the same labelled cross-episode pairs:

    context cosine   the level-2 chain vector (span-pooled, episode-centred)
    name cosine      the head string encoded on its own
    combined         logistic regression over six features (pairing.py)

The question is not which scores highest — it is whether *any* threshold
separates the true pairs from the false ones. A method where the worst positive
outranks the best negative is usable; one where they interleave is not, no
matter how good its average looks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_linking import encoding as EN  # noqa: E402
from entity_linking import pairing as P  # noqa: E402
from entity_linking import spans as SP  # noqa: E402

EPISODES = {
    "ep0": "The Supabase Plugin for AI Coding Agents gives your AI coding agent everything it needs to work with Supabase. It bundles the Supabase MCP server and Supabase agent skills so your agent can query your database, manage migrations, deploy Edge Functions, and follow Supabase and Postgres best practices — without manual configuration. ",
    "ep1": "The Supabase MCP server connects your AI coding agent directly to your Supabase projects. Once authenticated, your agent can query your database, manage migrations, deploy Edge Functions, and more — see the full list of available tools.",
    "ep2": "Every Supabase project gets a full Postgres database, not a Postgres abstraction. This database is the foundation that Auth, Storage, Realtime, and Edge Functions are built on, and Supabase manages daily database backups and offers point-in-time recovery on paid plans.",
}

# What an LLM coref pass produces. Stubbed so the demo is deterministic and
# offline; --live replaces it with real calls.
STUB_CHAINS = {
    "ep0": [["Sarah Chen", "S. Chen"], ["the migration"], ["the rollback plan"]],
    "ep1": [["S. Chen"], ["the schema"], ["the analytics team"]],
    "ep2": [["Sarah Chen"], ["Sarah Kim"], ["the review"]],
    "ep3": [["Sarah Kim"], ["the pricing note"], ["the board"]],
    "ep4": [["Okta"], ["SAML"], ["every tenant"]],  # parses standalone
    "ep5": [["Okta"], ["SCIM"], ["a file import"]],
}

# (episode, surface) pairs and whether they denote the same entity.
LABELLED = [
    (("ep0", "Sarah Chen"), ("ep1", "S. Chen"), 1),   # alias, low string overlap
    (("ep0", "Sarah Chen"), ("ep2", "Sarah Chen"), 1),
    (("ep2", "Sarah Kim"), ("ep3", "Sarah Kim"), 1),
    (("ep4", "Okta"), ("ep5", "Okta"), 1),
    (("ep0", "Sarah Chen"), ("ep3", "Sarah Kim"), 0),  # the hard negative
    (("ep1", "S. Chen"), ("ep3", "Sarah Kim"), 0),
    (("ep2", "Sarah Chen"), ("ep3", "Sarah Kim"), 0),
    (("ep4", "Okta"), ("ep3", "Sarah Kim"), 0),
]


def name_vectors(strings: list[str]) -> dict[str, np.ndarray]:
    """Identity embedding: the head string encoded with no episode context."""
    import torch

    tokenizer, model, device = EN._load()
    encoded = tokenizer([EN.PREFIX + s for s in strings], return_tensors="pt",
                        padding=True, truncation=True, max_length=64)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state.cpu().numpy()
    mask = encoded["attention_mask"].cpu().numpy()[..., None]
    vectors = (hidden * mask).sum(1) / mask.sum(1)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return dict(zip(strings, vectors))


def build_records(live: bool, model: str) -> list[P.ChainRecord]:
    """One ChainRecord per chain, carrying both vector kinds."""
    stage1 = None
    if live:
        spec = importlib.util.spec_from_file_location(
            "stage1", Path(__file__).parent / "1_get_candidates.py")
        stage1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage1)

    records: list[P.ChainRecord] = []
    for episode_id, text in EPISODES.items():
        encoded = EN.encode_episode(text)
        if live:
            chains = stage1.resolve_chains(stage1.extract_chains(text, model=model))
        else:
            spans = {s.text: s for s in SP.parse_spans(text)}
            chains = [[spans[t] for t in group if t in spans]
                      for group in STUB_CHAINS[episode_id]]
            chains = [c for c in chains if c]
        heads = [c[0].text for c in chains]
        vectors = name_vectors(heads)
        for i, chain in enumerate(chains):
            positions = EN.chain_positions(encoded.offsets, encoded.valid,
                                           chain, cap=12)
            context = EN.pool(encoded.hidden, positions,
                              EN.episode_center(encoded))
            records.append(P.ChainRecord(
                chain_id=f"{episode_id}-{i}", episode_id=episode_id,
                head=chain[0].text,
                surfaces=frozenset(s.text for s in chain),
                candidate_type=chain[0].candidate_type,
                name_vector=vectors[chain[0].text], context_vector=context,
            ))
    return records


def separability(scores: list[tuple[float, int]]) -> tuple[float, float, bool]:
    positives = [s for s, y in scores if y == 1]
    negatives = [s for s, y in scores if y == 0]
    worst_positive, best_negative = min(positives), max(negatives)
    return worst_positive, best_negative, worst_positive > best_negative


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Real coref calls")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    records = build_records(args.live, args.model)
    evidence = P.build_evidence(records)
    scorer = P.PairScorer()

    def find(episode_id: str, surface: str) -> P.ChainRecord | None:
        return next((r for r in records
                     if r.episode_id == episode_id and surface in r.surfaces), None)

    print(f"\n=== corpus: {len(EPISODES)} episodes, {len(records)} chains "
          f"({'live coref' if args.live else 'stubbed coref'}) ===")
    for episode_id in EPISODES:
        groups = [sorted(r.surfaces) for r in records if r.episode_id == episode_id]
        print(f"  {episode_id}: " + " | ".join("{" + ", ".join(g) + "}"
                                               for g in groups))
    print(f"\nevidence harvested: {len(evidence.same_chain)} alias pairs, "
          f"{len(evidence.split_in_episode)} kept-apart pairs")

    rows, missing = [], 0
    for (ea, sa), (eb, sb), label in LABELLED:
        a, b = find(ea, sa), find(eb, sb)
        if a is None or b is None:
            missing += 1
            continue
        feats = P.features(a, b, evidence)
        rows.append({
            "pair": f"{sa} ({ea}) ~ {sb} ({eb})", "label": label,
            "context_cosine": feats["context_cosine"],
            "name_cosine": feats["name_cosine"],
            "combined": scorer.score(feats), "features": feats,
        })
    if missing:
        print(f"  ({missing} labelled pairs had no matching chain)")

    print(f"\n{'pair':<38} {'gold':>5} {'context':>8} {'name':>7} {'combined':>9}")
    print("-" * 72)
    for row in sorted(rows, key=lambda r: -r["label"]):
        print(f"{row['pair']:<38} {row['label']:>5} {row['context_cosine']:>8.3f} "
              f"{row['name_cosine']:>7.3f} {row['combined']:>9.3f}")

    print(f"\n{'method':<20} {'worst pos':>10} {'best neg':>10} {'gap':>8}  separable")
    print("-" * 62)
    for method, key in (("context cosine", "context_cosine"),
                        ("name cosine", "name_cosine"),
                        ("combined probability", "combined")):
        scores = [(r[key], r["label"]) for r in rows]
        worst, best, ok = separability(scores)
        print(f"{method:<20} {worst:>10.3f} {best:>10.3f} {worst - best:>+8.3f}"
              f"  {'YES' if ok else 'no'}")

    if args.dump:
        args.dump.write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {len(rows)} scored pairs -> {args.dump}")


if __name__ == "__main__":
    main()
