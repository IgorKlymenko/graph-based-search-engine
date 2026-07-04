"""OpenAI-backed LLMClient and Embedder implementations."""
from __future__ import annotations

import json
import logging
from typing import Any

import openai

log = logging.getLogger(__name__)


class OpenAILLMClient:
    """LLMClient backed by the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI()

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def complete_json(
        self,
        prompt: str,
        schema: dict,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON: %s", content[:200])
            return {}


class OpenAIEmbedder:
    """Embedder backed by the OpenAI Embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._client = openai.AsyncOpenAI()
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=text, model=self._model
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts, model=self._model
        )
        # API returns embeddings in the same order as input
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
