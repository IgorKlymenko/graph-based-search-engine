"""The blackbox: a directory of documents in, graph rows in Postgres out.

Called from the API service:

    from src.inject.main import ingest
    counts = ingest(notes_dir=Path("/tmp/corpus"),
                    tenant_id=tenant, dsn=settings.database_url)

Or from a shell, same code path:

    uv run python -m src.inject.main --store ./store.json \\
        --notes-dir ./notes --tenant-id <uuid> --dsn postgresql://...

Every stage is resumable through the JSON store, so a re-run after a document
changes redoes only what that document invalidated, and a crash mid-corpus
resumes rather than restarting.

Three model steps: one coref call per *new* episode, one name-judge call per
forty *new* surfaces, and one adjudication call per chain pair the scorer
cannot settle. Everything else runs locally.

NOT safe to call concurrently for the same store path -- it is a file, not a
database. One ingest at a time per tenant, which is what a task queue gives you.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.inject import chains as CH
from src.inject import encoding as EN
from src.inject import episodes as EP
from src.inject import pairing as PR
from src.inject import store as ST
from src.inject.utils import judge as JG
from src.inject.utils import spans as SP
from src.inject.utils.prompts import (
    ENTITY_CHOICE_PROMPT,
    ENTITY_CHOICE_SCHEMA,
    MERGE_REVIEW_PROMPT,
    MERGE_REVIEW_SCHEMA,
)

REVIEW_WORKERS = 8
# Save the store this often during the paid stages, so a kill mid-way keeps
# what was already answered. A full-vault link is ~10,000 review calls over
# half an hour; the first such run lost all of them to a crash at the end.
CHECKPOINT_EVERY = 200


# ── vector cache ──────────────────────────────────────────────────────────────

def empty_vector_cache() -> dict:
    return {"model": EN.MODEL_NAME, "names": {}, "contexts": {}}


def load_vector_cache(path: Path | None) -> dict:
    """The sidecar of vectors from earlier runs, or an empty one.

    Name vectors are keyed by string, context vectors by chain and span set.
    Measured: a forward pass costs 63 ms per episode plus a 14.5 s model load,
    against 11 ms to reload the whole file, so a re-link over an unchanged
    corpus skips every pass. A cache from another model is ignored, not merged:
    the vectors would not be comparable.
    """
    cache = empty_vector_cache()
    if path is None or not Path(path).exists():
        return cache
    try:
        with np.load(path) as z:
            if str(z["model"]) != EN.MODEL_NAME:
                return cache
            cache["names"] = dict(zip(z["name_keys"].tolist(), z["name_vectors"]))
            cache["contexts"] = dict(zip(z["context_keys"].tolist(),
                                         z["context_vectors"]))
    except (OSError, ValueError, KeyError):
        return empty_vector_cache()
    return cache


def save_vector_cache(path: Path, cache: dict) -> None:
    def pack(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        keys = sorted(vectors)
        matrix = (np.vstack([vectors[k] for k in keys]) if keys
                  else np.zeros((0, 0)))
        return np.array(keys, dtype=str), matrix

    name_keys, name_vectors = pack(cache["names"])
    context_keys, context_vectors = pack(cache["contexts"])
    np.savez(path, model=np.array(cache["model"]), name_keys=name_keys,
             name_vectors=name_vectors, context_keys=context_keys,
             context_vectors=context_vectors)


def cached(keys: list[str], cache: dict[str, np.ndarray], compute) -> dict:
    """Vectors for ``keys``, computing only the ones ``cache`` lacks."""
    missing = [k for k in keys if k not in cache]
    if missing:
        cache.update(compute(missing))
    return {k: cache[k] for k in keys if k in cache}


def sync(store: dict, notes_dir: Path, *, tenant_id: str, source: str) -> dict:
    """Stage 1 — bring the store in line with the corpus."""
    diffs = EP.sync_corpus(store, notes_dir, tenant_id=tenant_id, source=source)
    added = sum(len(d.added) for d in diffs)
    removed = sum(len(d.removed) for d in diffs)
    changed = sum(1 for d in diffs if d.status != "unchanged")
    print(f"[1/4] episodes: {len(diffs)} documents, {changed} changed "
          f"-> +{added} episodes, {removed} retired")
    return {"documents": len(diffs), "added": added, "removed": removed}


def extract(store: dict, *, model: str, workers: int, limit: int | None,
            store_path: Path | None) -> dict:
    """Stage 2 — coref chains for every episode that has none yet."""
    result = CH.run(store, model=model, workers=workers, limit=limit,
                    store_path=store_path)
    print(f"[2/4] chains: {result['episodes']} episodes extracted, "
          f"{result['candidates']} candidates in store")
    return result


def judge(store: dict, *, models: tuple[str, ...], workers: int,
          checkpoint=None) -> dict:
    """Stage 2b — decide which new surfaces are names; verdicts live in store["names"].

    Only surfaces without a verdict are sent, so an unchanged corpus costs
    nothing. Delete ``store["names"]`` to re-judge after changing the prompt.
    """
    names = store.setdefault("names", {})
    surfaces = {c["text"] for c in CH.live_candidates(store)
                if SP.is_identifying(c["text"])}
    todo = surfaces - set(names)
    if todo:
        names.update(JG.judge_surfaces(
            JG.contexts_for(store, todo), models=models, workers=workers,
            on_progress=lambda partial: (names.update(partial), checkpoint())
            if checkpoint else None))
    junk = sum(1 for s in surfaces if not names.get(s, True))
    print(f"[2b/5] names: {len(todo)} surfaces judged, "
          f"{junk} of {len(surfaces)} live surfaces are not names")
    return {"judged": len(todo), "surfaces": len(surfaces), "junk": junk}


def _name_vectors(strings: list[str]) -> dict[str, np.ndarray]:
    """Identity embedding: the head string alone, no episode context.

    Measured at AUC 0.936 for cross-episode identity on ECB+ against 0.745 for
    the contextual chain vector, so this is what the linker keys on. Context
    enters only through ``context_gated``, on pairs this cannot separate.
    """
    import torch

    tokenizer, model, device = EN._load()
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(strings), 64):
        batch = strings[i:i + 64]
        encoded = tokenizer([EN.PREFIX + s for s in batch], return_tensors="pt",
                            padding=True, truncation=True, max_length=64)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.cpu().numpy()
        mask = encoded["attention_mask"].cpu().numpy()[..., None]
        vectors = (hidden * mask).sum(1) / mask.sum(1)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        out.update(dict(zip(batch, vectors)))
    return out


def _context_key(chain_id: str, group: list[dict]) -> str:
    """Cache key: the chain plus the exact spans pooled into its vector."""
    spans = ",".join(f"{c['start']}-{c['end']}"
                     for c in sorted(group, key=lambda c: c["start"]))
    return f"{chain_id}|{EP.sha1_text(spans)}"


def _context_vectors(store: dict, by_chain: dict[str, list[dict]],
                     cache: dict | None = None) -> dict:
    """One forward pass per episode -> a contextual vector per chain.

    Only consumed by ``context_gated``, which fires on pairs whose surfaces are
    the same string and where the name features carry no information. Uncentered:
    episode-centering measured at chance for identity, uncentered at 0.745.

    Only episodes with at least one chain missing from ``cache`` are encoded.
    """
    cache = {} if cache is None else cache
    keys = {chain_id: _context_key(chain_id, group)
            for chain_id, group in by_chain.items()}
    by_episode: dict[str, list[str]] = {}
    for chain_id, group in by_chain.items():
        if keys[chain_id] not in cache:
            by_episode.setdefault(group[0]["episode_id"], []).append(chain_id)

    for episode_id, chain_ids in by_episode.items():
        episode = store["episodes"].get(episode_id)
        if episode is None:
            continue
        encoded = EN.encode_episode(episode["text"])
        chains = [[EN.Span(sid=c["candidate_id"], start=c["start"], end=c["end"],
                           text=c["text"], candidate_type=c["candidate_type"],
                           origin=c["origin"])
                   for c in sorted(by_chain[cid], key=lambda c: c["start"])]
                  for cid in chain_ids]
        vectors = EN.encode_chains(episode["text"], chains, encoded=encoded)
        # encode_chains drops a chain whose mentions fell outside the token
        # window, so match on the head span rather than by position
        by_head = {(v.head.start, v.head.end): v.vector for v in vectors}
        for chain_id, chain in zip(chain_ids, chains):
            vector = by_head.get((chain[0].start, chain[0].end))
            if vector is not None:
                cache[keys[chain_id]] = vector
    return {chain_id: cache[key] for chain_id, key in keys.items()
            if key in cache}


def _seed_key(store: dict, group: list[dict]) -> str | None:
    """The connector key of a chain that holds a seeded span, else None.

    Looked up by offsets in the episode's ``seeds``; a chain that absorbed
    spans of two seeds (``merge_seeded`` on an identical name) takes the
    smallest key, which keeps the record deterministic.
    """
    seeded = [c for c in group if c.get("origin") == CH.SEED_ORIGIN]
    if not seeded:
        return None
    episode = store["episodes"].get(seeded[0]["episode_id"], {})
    at = {(s, e): seed["key"] for seed in episode.get("seeds") or []
          for s, e, _ in seed["spans"]}
    keys = {at[(c["start"], c["end"])] for c in seeded if (c["start"], c["end"]) in at}
    return min(keys) if keys else None


def _head_of(group: list[dict]) -> dict:
    """The chain's head, re-picked when the model's choice was filtered out.

    Stage 2 may have made a pronoun the head; once it is gone the chain still
    needs a name, and the longest remaining surface is the same fallback
    ``spans.pick_head`` uses.
    """
    return next((c for c in group if c["is_head"]),
                max(group, key=lambda c: len(c["text"])))


def encode(store: dict, *, context: bool = True,
           cache: dict | None = None,
           names: dict[str, bool] | None = None) -> list[PR.ChainRecord]:
    """Stage 3 — one ChainRecord per live chain, carrying both vector kinds.

    ``cache`` is a vector cache (see ``load_vector_cache``); vectors it already
    holds are not recomputed, and the ones computed here are added to it.

    ``names`` is the judge's verdict per surface (``store["names"]``). A surface
    judged not to be a name is dropped from the chain's identity exactly like a
    pronoun, and a chain with no name left is withheld: it becomes no record,
    so no entity, and its ``entity_id`` stays NULL.
    """
    cache = empty_vector_cache() if cache is None else cache
    live = [c for c in store["candidates"].values()
            if store["episodes"].get(c["episode_id"], {}).get("removed_at") is None]
    by_chain: dict[str, list[dict]] = {}
    for candidate in live:
        by_chain.setdefault(candidate["chain_id"], []).append(candidate)

    # A pronoun is a legitimate chain member -- it is how the model linked the
    # mentions -- but it is not a name, and letting it into `surfaces` makes two
    # chains that both say "it" score string_similarity 1.0 and lets
    # ("Sarah Chen", "she") count as an alias. Candidates keep them; identity
    # does not. `is_identifying` also drops "the", "three" and friends.
    names = {} if names is None else names
    identifying: dict[str, list[dict]] = {}
    dropped_chains = dropped_surfaces = 0
    for chain_id, group in by_chain.items():
        kept = [c for c in group if SP.is_identifying(c["text"])
                and names.get(c["text"], True)]
        dropped_surfaces += len(group) - len(kept)
        if kept:
            identifying[chain_id] = kept
        else:
            dropped_chains += 1  # nothing left that could name anything

    heads = sorted({_head_of(group)["text"] for group in identifying.values()})
    vectors = cached(heads, cache["names"], _name_vectors) if heads else {}
    contexts = (_context_vectors(store, by_chain, cache["contexts"])
                if context else {})

    records: list[PR.ChainRecord] = []
    for chain_id, group in identifying.items():
        head = _head_of(group)
        records.append(PR.ChainRecord(
            chain_id=chain_id, episode_id=head["episode_id"], head=head["text"],
            surfaces=frozenset(c["text"] for c in group),
            candidate_type=head["candidate_type"],
            name_vector=vectors.get(head["text"]),
            context_vector=contexts.get(chain_id),
            key=_seed_key(store, by_chain[chain_id])))
    print(f"[3/4] encode: {len(records)} chains, {len(heads)} distinct names, "
          f"{len(contexts)} context vectors "
          f"({dropped_surfaces} non-identifying or non-name surfaces dropped, "
          f"{dropped_chains} chains withheld with none left)")
    return records


def _ask_cached(cache: dict, prompt: str, call) -> dict:
    """One judgement per distinct question, remembered across runs.

    The review calls are the pipeline's remaining source of non-determinism: at
    temperature 0 the same prompt still comes back differently often enough that
    a passage moved between two people across runs on unchanged text, which
    reshuffles clusters and therefore entity ids. Caching on the prompt makes the
    answer a fact about the corpus rather than about when you asked -- and a
    re-run of an unchanged corpus costs nothing.

    Keyed on the prompt itself, so any change to what the reviewer is shown --
    new passages, a contested-surface note appearing -- is a new question rather
    than a stale answer. Delete `store["reviews"]` to force a re-review.
    """
    key = EP.sha1_text(prompt)
    if key not in cache:
        cache[key] = call()
    return cache[key]


def link(records: list[PR.ChainRecord], *, model: str, k: int,
         thresholds: PR.Thresholds, workers: int,
         passages: dict[str, str] | None = None,
         cache: dict | None = None,
         previous: dict | None = None,
         adjudicate=None,
         checkpoint=None) -> dict:
    """Stage 4 — block, score, review the uncertain band, cluster.

    ``previous`` is the last run's linking state, ``{"chains": [...],
    "merges": [[a, b, p], ...]}``. With it, only pairs that could have changed
    are scored: those touching a chain that was not linked last time, and old
    pairs whose chains share a surface with one -- their evidence may have
    moved. Every other accepted merge is carried into the clustering as it was.
    Measured on the synthetic corpus: 10 new episodes out of 92 scored 23% of
    the pairs and reproduced the full re-link's partition exactly. Without
    ``previous`` every pair is scored, which is what ``--full-relink`` forces.

    ``adjudicate`` is the review call; tests inject a stub. ``checkpoint`` is
    called every ``CHECKPOINT_EVERY`` answers so the cache they landed in can
    be saved.
    """
    adjudicate = _adjudicate if adjudicate is None else adjudicate
    cache = {} if cache is None else cache
    evidence = PR.build_evidence(records)
    scorer = PR.PairScorer()
    candidates = (PR.ann_candidate_pairs(records, k=k)
                  if len(records) > PR.ANN_MIN_FACTOR * k
                  else PR.candidate_pairs(records))

    carried: list[tuple[str, str, float]] = []
    if previous:
        live = {r.chain_id for r in records}
        known = set(previous.get("chains", ())) & live
        touched = {s for r in records if r.chain_id not in known
                   for s in r.surfaces}

        def stale(a: PR.ChainRecord, b: PR.ChainRecord) -> bool:
            return (a.chain_id not in known or b.chain_id not in known
                    or bool(a.surfaces & touched) or bool(b.surfaces & touched))

        candidates = [(a, b) for a, b in candidates if stale(a, b)]
        rescored = {frozenset((a.chain_id, b.chain_id)) for a, b in candidates}
        carried = [(a, b, p) for a, b, p in previous.get("merges", ())
                   if a in known and b in known
                   and frozenset((a, b)) not in rescored]

    scored = [(a, b, scorer.score(PR.features(a, b, evidence)))
              for a, b in candidates]

    to_review = [(a, b) for a, b, p in scored
                 if PR.decide(p, thresholds) == PR.REVIEW
                 and not PR.keys_settle(a, b)]
    answers: dict[tuple[str, str], dict] = {}
    for (a, b), answer in CH.completed(
            lambda ab: adjudicate(*ab, model, evidence, passages, cache),
            to_review, workers=workers, label="reviews"):
        answers[(a.chain_id, b.chain_id)] = answer
        if checkpoint and len(answers) % CHECKPOINT_EVERY == 0:
            checkpoint()
    abandoned = len(to_review) - len(answers)

    merges: list[tuple[str, str, float]] = list(carried)
    for a, b, probability in scored:
        verdict = PR.decide(probability, thresholds)
        if verdict == PR.REVIEW and PR.keys_settle(a, b):
            verdict = PR.SPLIT  # two connector keys, or a key beside prose
        elif verdict == PR.REVIEW:
            # an abandoned review is a split for this run; the caller drops the
            # linking state so the pair is scored again next time
            answer = answers.get((a.chain_id, b.chain_id))
            verdict = PR.MERGE if answer and answer["same_entity"] else PR.SPLIT
        if verdict == PR.MERGE:
            merges.append((a.chain_id, b.chain_id, probability))

    assignment = PR.cluster([r.chain_id for r in records], merges)
    print(f"[4/4] link: {len(candidates)} pairs scored, {len(to_review)} reviewed, "
          f"{len(merges) - len(carried)} merged, {len(carried)} carried "
          f"-> {len(set(assignment.values()))} entities")
    return {"assignment": assignment, "reviewed": len(to_review),
            "merges": merges, "pairs": len(candidates), "carried": len(carried),
            "abandoned": abandoned, "evidence": evidence}


def _profile(members: list[PR.ChainRecord], passages: dict[str, str],
             max_passages: int = 2) -> str:
    """What the graph already knows about one entity, for the reviewer to use."""
    surfaces = sorted({s for m in members for s in m.surfaces})
    episodes = sorted({m.episode_id for m in members})[:max_passages]
    seen = " | ".join(passages.get(e, "")[:300] for e in episodes)
    return f"names: {surfaces}\n  appears in: {seen}"


def resolve_against_graph(
    records: list[PR.ChainRecord], assignment: dict[str, int],
    pending: dict[str, list[int]], passages: dict[str, str], *, model: str,
    workers: int = REVIEW_WORKERS, cache: dict | None = None,
) -> dict[str, dict]:
    """Ask the model which known entity an ambiguous mention belongs to.

    The pairwise reviewer sees two passages and nothing else, which is why it
    scores 81% on the pairs the scorer could not settle. Here it sees the mention
    against a *profile* of each candidate entity assembled from the rest of the
    corpus — every name that entity goes by and the passages it appears in. The
    name is known to be uninformative, so the decision has to come from whether
    the mention's activity matches one candidate's established record.

    Returns ``chain_id -> {"cluster": int | None, "answer": {...}}``. A null
    cluster means it declined, and the mention stays unresolved: a wrong
    attachment asserts false facts about a real entity, an unplaced mention only
    stays unplaced. The raw answer is kept so callers can show the reasoning.
    """
    if not pending:
        return {}
    by_cluster: dict[int, list[PR.ChainRecord]] = {}
    by_id: dict[str, PR.ChainRecord] = {}
    for record in records:
        by_cluster.setdefault(assignment[record.chain_id], []).append(record)
        by_id[record.chain_id] = record

    def ask(item: tuple[str, list[int]]) -> tuple[str, dict]:
        chain_id, clusters = item
        record = by_id[chain_id]
        profiles = "\n\n".join(
            f"[{cluster}] {_profile(by_cluster[cluster], passages)}"
            for cluster in clusters)
        user = (f"Mention: {sorted(record.surfaces)}\n"
                f"Passage: {passages.get(record.episode_id, '')}\n\n"
                f"Candidate entities:\n\n{profiles}")
        answer = _ask_cached(
            {} if cache is None else cache, user,
            lambda: CH._complete_json(
                system=ENTITY_CHOICE_PROMPT, user=user, model=model,
                schema=ENTITY_CHOICE_SCHEMA, schema_name="entity_choice"))
        try:
            picked = int(answer.get("entity_id"))
        except (TypeError, ValueError):
            picked = None
        if picked not in clusters:
            picked = None
        return chain_id, {"cluster": picked, "answer": answer,
                          "candidates": clusters, "prompt": user}

    return dict(result for _, result in CH.completed(
        ask, list(pending.items()), workers=workers, label="entity choices"))


def _adjudicate(a: PR.ChainRecord, b: PR.ChainRecord, model: str,
                evidence: PR.Evidence, passages: dict[str, str] | None,
                cache: dict | None = None) -> dict:
    contested = sorted((a.surfaces | b.surfaces) & evidence.ambiguous_surfaces)
    note = (f"\n\nNOTE: {contested} is used for more than one entity elsewhere."
            if contested else "")
    text = passages or {}
    user = (f"A names: {sorted(a.surfaces)}\n"
            f"A passage: {text.get(a.episode_id, '')}\n\n"
            f"B names: {sorted(b.surfaces)}\n"
            f"B passage: {text.get(b.episode_id, '')}{note}")
    return _ask_cached(
        {} if cache is None else cache, user,
        lambda: CH._complete_json(system=MERGE_REVIEW_PROMPT, user=user,
                                  model=model, schema=MERGE_REVIEW_SCHEMA,
                                  schema_name="merge_review"))


def unresolved_report(records: list[PR.ChainRecord], pending: dict[str, list[int]],
                      entity_of_cluster: dict[int, str]) -> list[dict]:
    """The withheld mentions, each with the entities it could plausibly belong to.

    ``pending`` was computed before the graph resolver ran, and placing a chain
    can empty the cluster it came from. A candidate cluster that no longer
    exists is not a candidate: it is skipped rather than raised on. The first
    full-vault run died here on ``KeyError`` after 10,274 review calls.
    """
    by_id = {r.chain_id: r for r in records}
    return [
        {"chain_id": chain_id, "episode_id": by_id[chain_id].episode_id,
         "surfaces": sorted(by_id[chain_id].surfaces),
         "candidates": sorted(entity_of_cluster[c] for c in clusters
                              if c in entity_of_cluster),
         "reason": "contested_surface"}
        for chain_id, clusters in sorted(pending.items())
    ]


def ingest(
    *,
    store_path: Path,
    notes_dir: Path | None = None,
    tenant_id: str | None = None,
    dsn: str | None = None,
    source: str | None = None,
    model: str | None = None,
    workers: int | None = None,
    limit: int | None = None,
    ann_k: int = 50,
    merge_at: float = 0.50,
    split_at: float = 0.30,
    graph_resolve: bool = True,
    context: bool = True,
    dry_run: bool = False,
    vector_cache: bool = True,
    full_relink: bool = False,
    full_upload: bool = False,
    judge_names: bool = True,
    judge_models: tuple[str, ...] | None = None,
) -> dict:
    """Documents in, graph rows out. Returns per-table counts when `dsn` is set.

    Defaults for `source`, `model` and `workers` are resolved here rather than in
    the signature so the stage modules stay the single place they are defined.

    `merge_at` / `split_at`: the prior weights top out at 0.50 for an exact name
    match, so a higher cut point never fires. Do NOT replace these with
    choose_thresholds() output -- on ECB+ its precision target scored 0.634 CoNLL
    against 0.775 for the cut chosen to maximise CoNLL directly.
    """
    source = EP.UPLOAD_SOURCE if source is None else source
    model = CH.EXTRACTION_MODEL if model is None else model
    workers = CH.EPISODE_WORKERS if workers is None else workers

    started = time.perf_counter()
    data = EP.load_store(store_path)

    if notes_dir is not None:
        if not tenant_id:
            raise ValueError("tenant_id is required with notes_dir")
        if not Path(notes_dir).is_dir():
            raise ValueError(f"not a directory: {notes_dir}")
        sync(data, Path(notes_dir), tenant_id=tenant_id, source=source)

    if dry_run:
        pending = CH.pending_episodes(data)
        print(f"[2/5] dry-run: {len(pending)} episodes pending extraction")
        EP.save_store(store_path, data)
        return {"pending": len(pending)}

    extract(data, model=model, workers=workers, limit=limit,
            store_path=store_path)
    if judge_names:
        judge(data, models=judge_models or JG.JUDGE_MODELS, workers=workers,
              checkpoint=lambda: EP.save_store(store_path, data))
        EP.save_store(store_path, data)  # paid verdicts; bank them before anything can fail
    cache_path = store_path.with_suffix(".vectors.npz") if vector_cache else None
    vectors = load_vector_cache(cache_path)
    records = encode(data, context=context, cache=vectors,
                     names=data.get("names") if judge_names else None)
    if cache_path is not None:
        save_vector_cache(cache_path, vectors)
    if len(records) < 2:
        print("  fewer than two chains; nothing to link")
        EP.save_store(store_path, data)
        return {"chains": len(records)}

    passages = {eid: ep["text"] for eid, ep in data["episodes"].items()}
    reviews = data.setdefault("reviews", {})
    result = link(records, model=model, k=ann_k, workers=REVIEW_WORKERS,
                  thresholds=PR.Thresholds(low=split_at, high=merge_at),
                  passages=passages, cache=reviews,
                  previous=None if full_relink else data.get("linking"),
                  checkpoint=lambda: EP.save_store(store_path, data))
    if result["abandoned"]:
        # some pairs never got their verdict: forget the incremental state so
        # the next run scores everything rather than carrying a silent split
        data.pop("linking", None)
    else:
        data["linking"] = {
            "chains": sorted(r.chain_id for r in records),
            "merges": [[a, b, round(p, 6)] for a, b, p in result["merges"]],
        }

    assignment = result["assignment"]
    pending = PR.unresolved(records, assignment, result["evidence"])
    placed: dict[str, int] = {}
    if pending and graph_resolve:
        decided = resolve_against_graph(records, assignment, pending, passages,
                                        model=model, cache=reviews)
        placed = {cid: r["cluster"] for cid, r in decided.items()
                  if r["cluster"] is not None}
        for chain_id, cluster in placed.items():
            assignment[chain_id] = cluster
        pending = {k: v for k, v in pending.items() if k not in placed}
    if pending or placed:
        print(f"  ambiguous mentions: {len(placed)} placed by the graph, "
              f"{len(pending)} left unresolved (no node created)")

    # Bank the review answers the moment they are all in, before any of the
    # shaping below can fail. Every call above is paid for; a crash between here
    # and the save at the end of the run throws the whole spend away, which is
    # exactly what happened on the first full-vault run (10,274 reviews lost to
    # a KeyError in the unresolved report).
    EP.save_store(store_path, data)

    # Name clusters by their smallest chain id rather than by position, so an
    # entity whose membership did not change keeps its id across re-runs.
    entity_id_of = PR.stable_entity_ids(assignment)
    entity_of_cluster = {assignment[c]: e for c, e in entity_id_of.items()}
    entities: dict[str, dict] = {}
    for record in records:
        if record.chain_id in pending:
            continue  # a contested name with nothing to anchor it is not an entity
        entity = entities.setdefault(entity_id_of[record.chain_id],
                                     {"surfaces": set(), "chain_ids": [],
                                      "episode_ids": set()})
        entity["surfaces"] |= record.surfaces
        entity["chain_ids"].append(record.chain_id)
        entity["episode_ids"].add(record.episode_id)
    data["entities"] = {
        entity_id: {"surfaces": sorted(entity["surfaces"]),
                    "chain_ids": sorted(entity["chain_ids"]),
                    "episode_ids": sorted(entity["episode_ids"])}
        for entity_id, entity in sorted(entities.items())
    }
    data["unresolved"] = unresolved_report(records, pending, entity_of_cluster)
    # Stamp the clustering onto the chain records so the DB write has it. A
    # withheld chain keeps entity_id NULL -- the mention happened, we simply
    # refuse to invent an entity for a name the corpus uses for two things.
    ST.assign_entities(data, entity_id_of, set(pending))
    EP.save_store(store_path, data)

    counts: dict = {}
    if dsn:
        if not tenant_id:
            raise ValueError("tenant_id is required with dsn")
        counts = ST.upload(data, dsn=dsn, tenant_id=tenant_id, full=full_upload)
        EP.save_store(store_path, data)  # the upload baseline lives in the store
        print("[5/5] postgres: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\nstore -> {store_path}  "
          f"({len(data['files'])} documents, {len(data['episodes'])} episodes, "
          f"{len(data['candidates'])} candidates, {len(data['entities'])} entities)")
    print(f"wall clock {time.perf_counter() - started:.1f}s")
    return counts or {"entities": len(data["entities"])}


def main() -> None:
    """CLI wrapper. The API calls ``ingest`` directly."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--source", default=EP.UPLOAD_SOURCE)
    parser.add_argument("--model", default=CH.EXTRACTION_MODEL)
    parser.add_argument("--workers", type=int, default=CH.EPISODE_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract at most N episodes this run")
    parser.add_argument("--ann-k", type=int, default=50)
    # The prior weights top out at 0.50 for an exact name match, so a higher
    # merge cut point never fires. Do NOT replace these with choose_thresholds()
    # output: on ECB+ its precision target scored 0.634 CoNLL against 0.775 for
    # the cut chosen to maximise CoNLL directly. See DATAFLOW.md.
    parser.add_argument("--merge-at", type=float, default=0.50,
                        help="Score at or above which pairs merge unreviewed")
    parser.add_argument("--split-at", type=float, default=0.30,
                        help="Score below which pairs split unreviewed")
    parser.add_argument("--no-graph-resolve", action="store_true",
                        help="Skip the graph-context call for ambiguous mentions")
    parser.add_argument("--no-context", action="store_true",
                        help="Skip the per-episode forward pass; disables "
                             "context_gated, which only fires on same-string pairs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stages 1 and 3 only; no model calls")
    parser.add_argument("--no-vector-cache", action="store_true",
                        help="Recompute every nomic vector instead of reading "
                             "<store>.vectors.npz")
    parser.add_argument("--full-relink", action="store_true",
                        help="Score every pair; ignore the previous run's merges")
    parser.add_argument("--full-upload", action="store_true",
                        help="Re-send every row instead of the delta since the "
                             "last upload (after a database reset)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the name judge; every identifying surface "
                             "counts as a name (the pre-2026-09-17 behaviour)")
    parser.add_argument("--judge-models", default=",".join(JG.JUDGE_MODELS),
                        help="Comma-separated cascade for the name judge, cheapest "
                             "first. Measured against hand labels: mini alone F1 "
                             "0.53, gpt-4.1 alone 0.80, mini then gpt-4.1 0.82 at "
                             "~60%% of the cost")
    parser.add_argument("--entities-out", type=Path, default=None)
    parser.add_argument("--dsn", default=None,
                        help="Postgres DSN. With it, the run also writes "
                             "files/episodes/surfaces/chains/entities.")
    args = parser.parse_args()
    ingest(store_path=args.store, notes_dir=args.notes_dir,
           tenant_id=args.tenant_id, dsn=args.dsn, source=args.source,
           model=args.model, workers=args.workers, limit=args.limit,
           ann_k=args.ann_k, merge_at=args.merge_at, split_at=args.split_at,
           graph_resolve=not args.no_graph_resolve,
           context=not args.no_context, dry_run=args.dry_run,
           vector_cache=not args.no_vector_cache,
           full_relink=args.full_relink, full_upload=args.full_upload,
           judge_names=not args.no_judge,
           judge_models=tuple(m.strip() for m in args.judge_models.split(",") if m.strip()))
    if args.entities_out:
        args.entities_out.write_text(
            json.dumps(EP.load_store(args.store)["entities"], indent=2))
        print(f"entities -> {args.entities_out}")


if __name__ == "__main__":
    main()
