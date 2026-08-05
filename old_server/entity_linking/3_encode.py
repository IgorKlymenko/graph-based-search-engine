"""Level 2 — one vector per coreference chain, from one forward pass per episode.

The episode is encoded once, so every token vector is contextualised over the
whole episode. A chain's vector is then a mean over a *set* of token positions
gathered from all of its mentions:

    proper name           span + window
    definite description  span + narrow window
    pronoun               span only, no window

The set is the point. Two mentions of one chain sitting a few words apart have
overlapping windows, and a list would let the shared tokens vote twice; a set
counts each token once. The result is one vector per chain — not per mention.

Windows are asymmetric by mention type on purpose. A proper name is a rigid
designator whose neighbourhood describes it ("Okta, the identity provider"), so
context earns its place. A pronoun carries no descriptive content of its own,
and its neighbourhood belongs to whatever the sentence is saying *about* the
referent — pooling it in would drown the chain in unrelated prose.

nomic-embed-text-v1.5 is mean-pooled natively, so pooling token states is the
model's own idiom rather than an off-label use of a CLS-trained space. It needs
the ``search_document:`` prefix and ``trust_remote_code=True``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from entity_linking.spans import (
    DEFINITE_DESCRIPTION,
    NOMINAL,
    PRONOUN,
    PROPER_NAME,
    Span,
)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
PREFIX = "search_document: "
MAX_TOKENS = 2048

# Window radius in *tokens* on each side of the span.
WINDOW_TOKENS: dict[str, int] = {
    PROPER_NAME: 8,
    DEFINITE_DESCRIPTION: 3,
    NOMINAL: 3,
    PRONOUN: 0,
}

_TOKENIZER = None
_MODEL = None
_DEVICE = None


def _load():
    """Load nomic once; importing this module must not fetch a model."""
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _MODEL.eval()
        _DEVICE = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        _MODEL.to(_DEVICE)
    return _TOKENIZER, _MODEL, _DEVICE


class Encoded(NamedTuple):
    """One episode's token states, addressed in the episode's own offsets."""

    hidden: np.ndarray  # (n_tokens, dim)
    offsets: list[tuple[int, int]]  # char span per token, prefix already removed
    valid: list[int]  # token indices belonging to the episode text


def encode_episode(text: str) -> Encoded:
    """One forward pass. Offsets come back in the episode's own coordinates.

    The prefix shifts every character position, so it is subtracted here rather
    than left for each caller to remember — a span would otherwise silently
    address the wrong tokens.
    """
    import torch

    tokenizer, model, device = _load()
    encoding = tokenizer(
        PREFIX + text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    raw_offsets = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        hidden = model(**encoding).last_hidden_state[0].cpu().numpy()

    shift = len(PREFIX)
    offsets: list[tuple[int, int]] = []
    valid: list[int] = []
    for i, (start, end) in enumerate(raw_offsets):
        # special tokens carry (0, 0); prefix tokens end inside the prefix
        if end > shift:
            offsets.append((start - shift, end - shift))
            valid.append(i)
        else:
            offsets.append((-1, -1))
    return Encoded(hidden=hidden, offsets=offsets, valid=valid)


def span_token_indices(
    offsets: list[tuple[int, int]], valid: list[int], start: int, end: int
) -> list[int]:
    """Token indices overlapping ``[start, end)`` in episode coordinates."""
    valid_set = set(valid)
    return [
        i for i in range(len(offsets))
        if i in valid_set
        and offsets[i][0] < end
        and offsets[i][1] > start
    ]


def chain_positions(
    offsets: list[tuple[int, int]],
    valid: list[int],
    chain: list[Span],
    windows: dict[str, int] | None = None,
    cap: int | None = None,
) -> set[int]:
    """Deduped token positions contributing to one chain's vector.

    Returned as a set so overlapping windows from nearby mentions cannot make a
    shared token count more than once.

    ``cap`` bounds the total instead of the radius. A radius sized for a
    document swallows half a short episode, and since every token is already
    contextualised over the whole episode, the extra tokens add shared signal
    rather than distinguishing signal. Span tokens are never dropped; the budget
    that remains goes to context tokens nearest a mention.
    """
    radii = WINDOW_TOKENS if windows is None else windows
    valid_set = set(valid)
    core: set[int] = set()
    ring: dict[int, int] = {}
    for span in chain:
        tokens = span_token_indices(offsets, valid, span.start, span.end)
        if not tokens:
            continue
        core.update(tokens)
        radius = radii.get(span.candidate_type, 0)
        for i in range(min(tokens) - radius, max(tokens) + radius + 1):
            if i in valid_set:
                distance = min(abs(i - t) for t in tokens)
                ring[i] = min(ring.get(i, radius + 1), distance)
    ring = {i: d for i, d in ring.items() if i not in core}
    if cap is None:
        return core | set(ring)
    budget = max(0, cap - len(core))
    nearest = sorted(ring.items(), key=lambda kv: (kv[1], kv[0]))[:budget]
    return core | {i for i, _ in nearest}


def pool(
    hidden: np.ndarray, positions: set[int], center: np.ndarray | None = None
) -> np.ndarray | None:
    """Mean over the selected token vectors, L2 normalised for cosine use.

    ``center`` subtracts the episode's own mean first. Every token is
    contextualised over the whole episode, so all chain vectors share a large
    common component that inflates every cosine; removing it leaves what
    actually distinguishes one chain from another.
    """
    if not positions:
        return None
    vector = hidden[sorted(positions)].mean(axis=0)
    if center is not None:
        vector = vector - center
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def episode_center(encoded: Encoded) -> np.ndarray:
    """Mean of every real-text token — the episode's shared component."""
    return encoded.hidden[encoded.valid].mean(axis=0)


class ChainVector(NamedTuple):
    """One chain, one vector, plus what went into it."""

    head: Span
    vector: np.ndarray
    n_tokens: int
    n_mentions: int


def encode_chains(
    text: str,
    chains: list[list[Span]],
    *,
    encoded: Encoded | None = None,
    windows: dict[str, int] | None = None,
    cap: int | None = None,
    center: bool = False,
) -> list[ChainVector]:
    """Vector per chain from a single forward pass over the episode.

    ``encoded`` can be passed in when several callers share one episode, so the
    forward pass is never repeated.
    """
    if encoded is None:
        encoded = encode_episode(text)
    origin = episode_center(encoded) if center else None
    out: list[ChainVector] = []
    for chain in chains:
        positions = chain_positions(
            encoded.offsets, encoded.valid, chain, windows, cap
        )
        vector = pool(encoded.hidden, positions, origin)
        if vector is None:  # every mention fell outside the token window
            continue
        out.append(ChainVector(
            head=chain[0], vector=vector,
            n_tokens=len(positions), n_mentions=len(chain),
        ))
    return out
