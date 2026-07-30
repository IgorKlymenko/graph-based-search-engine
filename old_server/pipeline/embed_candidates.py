"""Stage 2 — contextual span embeddings, one vector per resolution unit.

Absorbs ``embed_cache.py``. A unit is a ``(concept, context, episode)`` triple
(``models.unit_key``), so one concept appearing in five different sentences is
five units with five different vectors.

Only ONE averaging step remains: within a unit, the mean of the
``mxbai-embed-large-v1`` token vectors overlapping the candidate's character span,
which reassembles a name the tokenizer split into subwords.

Averaging *across* occurrences of a concept is switched off — see the note in
``run``. It collapsed every context into one centroid, which both blinded the
vector to context and made it structurally impossible for one concept to
resolve to two entities.

The notebook script's exploratory ``tasks[: len(tasks)//4]`` sampling is dropped
— every candidate is embedded. Resumable via ``vector_cache.npz``.

The model is loaded lazily so importing this module (and running the unit tests)
never downloads or instantiates a transformer.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from pipeline.config import (
    CHARS_PER_TOKEN,
    EMBED_BATCH_SIZE,
    EMBED_MAX_TOKENS,
    EMBED_SAVE_EVERY,
    ENTITY_CACHE,
    CANDIDATE_EMBED_MODEL,
    VECTOR_CACHE,
)
from pipeline.models import unit_key

_tokenizer = None
_model = None
_device = None


def _load():
    """Load the mxbai tokenizer/model onto the best available device, once."""
    global _tokenizer, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(CANDIDATE_EMBED_MODEL)
        _model = AutoModel.from_pretrained(CANDIDATE_EMBED_MODEL)
        _model.eval()
        _device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        _model.to(_device)
    return _tokenizer, _model, _device


def apply_window(
    episode_text: str, span_start: int, span_end: int
) -> tuple[str, int, int]:
    """Clip the episode to a window around the span so it fits ``EMBED_MAX_TOKENS``.

    Runs before the tokenizer's own truncation, which cuts from the left and
    would otherwise drop a span sitting late in a long episode.
    """
    if len(episode_text) > EMBED_MAX_TOKENS * CHARS_PER_TOKEN:
        half = (EMBED_MAX_TOKENS * CHARS_PER_TOKEN) // 2
        window_start = max(0, span_start - half)
        window_end = min(len(episode_text), span_end + half)
        episode_text = episode_text[window_start:window_end]
        span_start -= window_start
        span_end -= window_start
    return episode_text, span_start, span_end


def embed_spans_batch(items: list[tuple[str, int, int]]) -> list[np.ndarray]:
    """One forward pass for a batch of ``(episode_text, span_start, span_end)``.

    The whole episode is encoded so self-attention contextualizes every token;
    only the tokens overlapping ``[span_start, span_end)`` are then mean-pooled,
    giving one vector per candidate occurrence rather than per episode.
    """
    import torch

    tokenizer, model, device = _load()
    windowed = [apply_window(ep, s, e) for ep, s, e in items]

    encoding = tokenizer(
        [w[0] for w in windowed],
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=EMBED_MAX_TOKENS,
    )
    offset_mappings = encoding.pop("offset_mapping")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        hidden = model(**encoding).last_hidden_state.cpu()  # (B, L, D)

    results: list[np.ndarray] = []
    for i, (_, span_start, span_end) in enumerate(windowed):
        mask = torch.tensor(
            [s < span_end and e > span_start for s, e in offset_mappings[i].tolist()]
        )
        tokens = hidden[i][mask]
        if tokens.shape[0] == 0:
            tokens = hidden[i]
        results.append(tokens.mean(dim=0).numpy())
    return results


def _locate_span(
    episode_text: str, concept: str, context: str
) -> tuple[int, bool] | None:
    """``(offset, exact)`` of THIS occurrence of ``concept``, or None if absent.

    A concept repeated inside one episode has several valid offsets, so the
    candidate's own ``context`` sentence is located first and the concept found
    within it. ``episode_text.index(concept)`` alone would pool every occurrence
    around the first one, discarding the contextual difference that is the whole
    point of span pooling.

    ``exact`` is False when the offset came from the episode-wide fallback,
    which happens only if the extractor did not copy ``context`` verbatim.
    """
    ctx_start = episode_text.find(context)
    if ctx_start >= 0:
        within = context.find(concept)
        if within >= 0:
            return ctx_start + within, True
    first = episode_text.find(concept)
    return (first, False) if first >= 0 else None


class Unit(NamedTuple):
    """One embedded resolution unit: its vector plus the concept it belongs to."""

    vector: np.ndarray
    concept: str
    type: str


def _load_existing(vector_cache: Path) -> dict[str, Unit]:
    """Return already-embedded {unit_key: Unit} if the model matches."""
    if not vector_cache.exists():
        return {}
    existing = np.load(vector_cache, allow_pickle=False)
    cached_model = str(existing["model"]) if "model" in existing else ""
    if cached_model != CANDIDATE_EMBED_MODEL:
        print(f"  model changed ({cached_model!r} -> {CANDIDATE_EMBED_MODEL!r}) — rebuilding")
        return {}
    if "keys" not in existing:
        print("  pre-unit vector cache (one row per concept) — rebuilding")
        return {}
    return {
        str(k): Unit(vector=v, concept=str(c), type=str(t))
        for k, v, c, t in zip(
            existing["keys"], existing["matrix"],
            existing["concepts"], existing["types"],
        )
    }


def _save(vector_cache: Path, done: dict[str, Unit]) -> None:
    keys = sorted(done)
    np.savez(
        vector_cache,
        keys=np.array(keys),
        matrix=np.stack([done[k].vector for k in keys]).astype(np.float32),
        concepts=np.array([done[k].concept for k in keys]),
        types=np.array([done[k].type for k in keys]),
        model=np.array(CANDIDATE_EMBED_MODEL),
    )


def run(
    *,
    cache_path: Path = ENTITY_CACHE,
    vector_cache: Path = VECTOR_CACHE,
    batch_size: int = EMBED_BATCH_SIZE,
    save_every: int = EMBED_SAVE_EVERY,
) -> dict[str, Unit]:
    """Embed one span per resolution unit and write ``vector_cache.npz``.

    Units are ``(concept, context, episode)`` — see ``models.unit_key``. Repeat
    candidates of one concept in one sentence collapse to a single unit; the same
    concept in a different sentence stays a separate unit with its own vector.
    """
    import json

    cache = json.loads(cache_path.read_text())
    candidates: list[dict] = cache["candidates"]
    done = _load_existing(vector_cache)

    tasks: list[tuple[str, str, str, int, int, str]] = []
    seen: set[str] = set()
    skipped = approximated = merged = 0
    for m in candidates:
        key = unit_key(m)
        if key in done:
            continue
        if key in seen:
            merged += 1  # same concept, same sentence, same episode
            continue
        concept, mtype = m["concept"], m["type"]
        located = _locate_span(m["episode"], concept, m["context"])
        if located is None:
            skipped += 1
            continue
        start, exact = located
        if not exact:
            approximated += 1
        seen.add(key)
        tasks.append(
            (key, concept, m["episode"], start, start + len(concept), mtype)
        )

    print(f"[2/6] embed: {len(tasks)} units to embed "
          f"({len(done)} cached, {merged} merged on (concept, context), "
          f"{skipped} skipped — span not verbatim in episode)")
    if approximated:
        print(f"  {approximated} spans fell back to the first episode-wide match "
              f"— context not verbatim")
    if not tasks:
        if done:
            _save(vector_cache, done)
        return done

    batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

    for batch_idx, batch in enumerate(batches):
        vecs = embed_spans_batch([(ep, s, e) for _, _, ep, s, e, _ in batch])
        for (key, concept, _, _, _, mtype), vec in zip(batch, vecs):
            done[key] = Unit(vector=vec, concept=concept, type=mtype)
        # Averaging across occurrences of one concept is deliberately OFF:
        # it produced a context-blind centroid and made it impossible for one
        # concept to resolve to two entities. Units now reach stage 4 intact
        # and the LLM decides. To restore, accumulate per-concept here and
        # collapse with np.mean once a concept's occurrences are exhausted.
        if (batch_idx + 1) % save_every == 0 and done:
            _save(vector_cache, done)

    _save(vector_cache, done)
    dim = next(iter(done.values())).vector.shape[0]
    n_cand = len({u.concept for u in done.values()})
    print(f"  saved {len(done)} unit vectors ({n_cand} distinct concepts) "
          f"-> {vector_cache}  ({len(done)} x {dim})")
    return done
