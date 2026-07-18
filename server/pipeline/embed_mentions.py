"""Stage 2 — contextual span embeddings for entity mentions.

Absorbs ``embed_cache.py``: each mention is embedded by mean-pooling the
``mxbai-embed-large-v1`` token vectors that overlap its character span within
the episode, then averaged across all occurrences of the same surface form.
The notebook script's exploratory ``tasks[: len(tasks)//4]`` sampling is dropped
— every mention is embedded. Resumable via ``vector_cache.npz``.

The model is loaded lazily so importing this module (and running the unit tests)
never downloads or instantiates a transformer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.config import (
    CHARS_PER_TOKEN,
    EMBED_BATCH_SIZE,
    EMBED_MAX_TOKENS,
    EMBED_SAVE_EVERY,
    ENTITY_CACHE,
    MENTION_EMBED_MODEL,
    VECTOR_CACHE,
)

_tokenizer = None
_model = None
_device = None


def _load():
    """Load the mxbai tokenizer/model onto the best available device, once."""
    global _tokenizer, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MENTION_EMBED_MODEL)
        _model = AutoModel.from_pretrained(MENTION_EMBED_MODEL)
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


def apply_window(text: str, span_start: int, span_end: int) -> tuple[str, int, int]:
    """Clip ``text`` to a window around the span so it fits in ``EMBED_MAX_TOKENS``."""
    if len(text) > EMBED_MAX_TOKENS * CHARS_PER_TOKEN:
        window = (EMBED_MAX_TOKENS * CHARS_PER_TOKEN) // 2
        chunk_start = max(0, span_start - window)
        chunk_end = min(len(text), span_end + window)
        text = text[chunk_start:chunk_end]
        span_start -= chunk_start
        span_end -= chunk_start
    return text, span_start, span_end


def embed_spans_batch(items: list[tuple[str, int, int]]) -> list[np.ndarray]:
    """One forward pass for a batch of ``(context, span_start, span_end)`` tuples."""
    import torch

    tokenizer, model, device = _load()
    windowed = [apply_window(ctx, s, e) for ctx, s, e in items]

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


def _load_existing(vector_cache: Path) -> dict[str, tuple[np.ndarray, str]]:
    """Return already-embedded {surface: (vec, type)} if the model matches."""
    if not vector_cache.exists():
        return {}
    existing = np.load(vector_cache, allow_pickle=False)
    cached_model = str(existing["model"]) if "model" in existing else ""
    if cached_model != MENTION_EMBED_MODEL:
        print(f"  model changed ({cached_model!r} -> {MENTION_EMBED_MODEL!r}) — rebuilding")
        return {}
    return {
        str(e): (v, str(t))
        for e, v, t in zip(existing["entities"], existing["matrix"], existing["types"])
    }


def _save(vector_cache: Path, done: dict[str, tuple[np.ndarray, str]]) -> None:
    entities = sorted(done)
    np.savez(
        vector_cache,
        entities=np.array(entities),
        matrix=np.stack([done[e][0] for e in entities]).astype(np.float32),
        types=np.array([done[e][1] for e in entities]),
        model=np.array(MENTION_EMBED_MODEL),
    )


def run(
    *,
    cache_path: Path = ENTITY_CACHE,
    vector_cache: Path = VECTOR_CACHE,
    batch_size: int = EMBED_BATCH_SIZE,
    save_every: int = EMBED_SAVE_EVERY,
) -> dict[str, tuple[np.ndarray, str]]:
    """Embed every cached mention span and write averaged vectors per surface."""
    import json

    cache = json.loads(cache_path.read_text())
    mentions: list[dict] = cache["mentions"]
    done = _load_existing(vector_cache)

    tasks: list[tuple[str, str, int, int, str]] = []
    skipped = 0
    for m in mentions:
        text, mtype = m["text"], m["type"]
        if text in done:
            continue
        episode = m.get("episode", m["context"])
        try:
            start = episode.index(text)
        except ValueError:
            skipped += 1
            continue
        tasks.append((text, episode, start, start + len(text), mtype))

    print(f"[2/6] embed: {len(tasks)} spans to embed "
          f"({len(done)} cached, {skipped} skipped — span not verbatim in episode)")
    if not tasks:
        if done:
            _save(vector_cache, done)
        return done

    remaining: Counter[str] = Counter(text for text, *_ in tasks)
    raw_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

    for batch_idx, batch in enumerate(batches):
        vecs = embed_spans_batch([(ep, s, e) for _, ep, s, e, _ in batch])
        for (text, _, _, _, mtype), vec in zip(batch, vecs):
            raw_vecs[text].append(vec)
            remaining[text] -= 1
            if remaining[text] == 0:
                done[text] = (np.mean(raw_vecs.pop(text), axis=0), mtype)
        if (batch_idx + 1) % save_every == 0 and done:
            _save(vector_cache, done)

    _save(vector_cache, done)
    dim = next(iter(done.values()))[0].shape[0]
    print(f"  saved {len(done)} entity vectors -> {vector_cache}  ({len(done)} x {dim})")
    return done
