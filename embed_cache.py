"""Embed all cached entity mentions and write vectors to vector_cache.npz.

Run with:  uv run python embed_cache.py
Output:    vector_cache.npz  (load in notebook with np.load)

Resumable: already-embedded entities are skipped on re-run if the model is the same.
If MODEL changes, the cache is wiped and rebuilt from scratch.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── config ────────────────────────────────────────────────────────────────────
MODEL           = "mixedbread-ai/mxbai-embed-large-v1"
ENTITY_CACHE    = Path("entity_cache.json")
VECTOR_CACHE    = Path("vector_cache.npz")
BATCH_SIZE      = 16
MAX_TOKENS      = 512
CHARS_PER_TOKEN = 4
SAVE_EVERY      = 10   # checkpoint every N batches

# ── model ─────────────────────────────────────────────────────────────────────
print(f"loading {MODEL} …", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModel.from_pretrained(MODEL)
model.eval()

device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)
model.to(device)
print(f"device: {device}", flush=True)

# ── span embedding ────────────────────────────────────────────────────────────
def _apply_window(text: str, span_start: int, span_end: int) -> tuple[str, int, int]:
    """Clip text to a window around the span so it fits in MAX_TOKENS."""
    if len(text) > MAX_TOKENS * CHARS_PER_TOKEN:
        window      = (MAX_TOKENS * CHARS_PER_TOKEN) // 2
        chunk_start = max(0, span_start - window)
        chunk_end   = min(len(text), span_end + window)
        text        = text[chunk_start:chunk_end]
        span_start -= chunk_start
        span_end   -= chunk_start
    return text, span_start, span_end


def embed_spans_batch(items: list[tuple[str, int, int]]) -> list[np.ndarray]:
    """One forward pass for a batch of (context, span_start, span_end) tuples."""
    windowed = [_apply_window(ctx, s, e) for ctx, s, e in items]

    encoding = tokenizer(
        [w[0] for w in windowed],
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=MAX_TOKENS,
    )
    offset_mappings = encoding.pop("offset_mapping")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        hidden = model(**encoding).last_hidden_state.cpu()  # (B, L, D)

    results = []
    for i, (_, span_start, span_end) in enumerate(windowed):
        mask = torch.tensor([
            s < span_end and e > span_start
            for s, e in offset_mappings[i].tolist()
        ])
        tokens = hidden[i][mask]
        if tokens.shape[0] == 0:
            tokens = hidden[i]
        results.append(tokens.mean(dim=0).numpy())
    return results


# ── load existing vector cache ────────────────────────────────────────────────
done: dict[str, tuple[np.ndarray, str]] = {}  # entity → (vec, type)

if VECTOR_CACHE.exists():
    existing = np.load(VECTOR_CACHE, allow_pickle=False)
    cached_model = str(existing["model"]) if "model" in existing else ""
    if cached_model == MODEL:
        for entity, vec, mtype in zip(existing["entities"], existing["matrix"], existing["types"]):
            done[str(entity)] = (vec, str(mtype))
        print(f"resuming: {len(done)} entities already cached")
    else:
        print(f"model changed ({cached_model!r} → {MODEL!r}) — rebuilding cache")


def save_checkpoint() -> None:
    entities = sorted(done.keys())
    np.savez(
        VECTOR_CACHE,
        entities=np.array(entities),
        matrix=np.stack([done[e][0] for e in entities]).astype(np.float32),
        types=np.array([done[e][1] for e in entities]),
        model=np.array(MODEL),
    )


# ── load mention cache ────────────────────────────────────────────────────────
if not ENTITY_CACHE.exists():
    sys.exit(f"entity cache not found: {ENTITY_CACHE}")

with ENTITY_CACHE.open() as f:
    cache = json.load(f)

bulk_mentions: list[dict] = cache["mentions"]
print(f"loaded {len(bulk_mentions)} mentions from {len(cache['processed_files'])} files")

# ── build task list ───────────────────────────────────────────────────────────
Task = tuple[str, str, int, int, str]  # (entity_text, episode, start, end, mtype)
tasks: list[Task] = []
skipped = 0

for m in bulk_mentions:
    text, ctx, mtype = m["text"], m["context"], m["type"]
    if text in done:
        continue  # already embedded with this model
    episode = m.get("episode", ctx)
    try:
        start = episode.index(text)
    except ValueError:
        skipped += 1
        continue
    tasks.append((text, episode, start, start + len(text), mtype))

print(f"{len(tasks)} tasks to embed  ({skipped} skipped — span not verbatim in episode)")
tasks = tasks[: len(tasks) // 4]
print(f"sampling 1/4 → {len(tasks)} tasks")

if not tasks:
    print("nothing to do")
    sys.exit(0)

# ── embed ─────────────────────────────────────────────────────────────────────
# Count how many mentions remain per entity so we know when to average and flush.
entity_remaining: Counter[str] = Counter(text for text, *_ in tasks)
raw_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
batches = [tasks[i : i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]

with tqdm(total=len(tasks), unit="mention", desc="embedding") as bar:
    for batch_idx, batch in enumerate(batches):
        spans = [(ep, s, e) for _, ep, s, e, _ in batch]
        vecs  = embed_spans_batch(spans)

        for (entity_text, _, _, _, mtype), vec in zip(batch, vecs):
            raw_vecs[entity_text].append(vec)
            entity_remaining[entity_text] -= 1
            if entity_remaining[entity_text] == 0:
                # all mentions for this entity are in — average and free memory
                done[entity_text] = (np.mean(raw_vecs.pop(entity_text), axis=0), mtype)

        bar.update(len(batch))

        if (batch_idx + 1) % SAVE_EVERY == 0 and done:
            save_checkpoint()
            bar.set_postfix(entities=len(done))

# ── final save ────────────────────────────────────────────────────────────────
save_checkpoint()
dim = next(iter(done.values()))[0].shape[0]
print(f"saved {len(done)} entity vectors → {VECTOR_CACHE}  ({len(done)} × {dim})")
