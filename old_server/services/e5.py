"""Local intfloat/e5-base-v2 embeddings: asymmetric, prefix required on every input.

"query: " for questions, "passage: " for indexed text (entity names, facts).
Unlike some asymmetric models, e5 needs a prefix on BOTH sides — a missing
"passage: " prefix is the most common silent bug: embeddings still compute,
they're just off-distribution and retrieval quality quietly degrades.
"""
from __future__ import annotations

import asyncio

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL          = "intfloat/e5-base-v2"
DIMS           = 768
QUERY_PREFIX   = "query: "
PASSAGE_PREFIX = "passage: "
MAX_TOKENS     = 512

torch.set_num_threads(1)
_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None


def _load() -> tuple[AutoTokenizer, AutoModel]:
    """Load from the local HF cache if present; only hit the network on first-ever use."""
    global _tokenizer, _model
    if _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            _model = AutoModel.from_pretrained(MODEL, local_files_only=True)
        except OSError:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL)
            _model = AutoModel.from_pretrained(MODEL)
        _model.eval()
    return _tokenizer, _model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over real (non-padding) tokens."""
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_e5_sync(
    texts: list[str], is_query: bool = False, batch_size: int = 32
) -> list[list[float]]:
    """Embed with e5-base-v2's asymmetric convention. Output vectors are L2-normalized."""
    if not texts:
        return []
    tokenizer, model = _load()
    prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
    prefixed = [prefix + t for t in texts]

    vectors: list[list[float]] = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        encoding = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_TOKENS,
        )
        with torch.no_grad():
            output = model(**encoding)
        pooled = _mean_pool(output.last_hidden_state, encoding["attention_mask"])
        normed = F.normalize(pooled, p=2, dim=1)
        vectors.extend(normed.numpy().tolist())
    return vectors


class E5Embedder:
    """Embedder protocol implementation backed by local e5-base-v2.

    Always embeds with the "query: " prefix — within retrieval/, Embedder.embed()
    is only ever called on query-role text (the raw query, expansion terms, HyDE
    answers, rerank queries); passages (entity names, facts) are embedded once at
    upload time with "passage: " via embed_e5_sync(..., is_query=False).
    """

    async def embed(self, text: str) -> list[float]:
        result = await asyncio.to_thread(embed_e5_sync, [text], True)
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(embed_e5_sync, texts, True)
