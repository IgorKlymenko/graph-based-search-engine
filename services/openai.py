"""OpenAI-backed LLMClient and Embedder implementations."""
from __future__ import annotations

import json
import logging
from typing import Any

import openai

log = logging.getLogger(__name__)

_sync_client: openai.OpenAI | None = None


def get_sync_client() -> openai.OpenAI:
    """Process-wide synchronous OpenAI client (reads OPENAI_API_KEY from env)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = openai.OpenAI()
    return _sync_client


def complete_json_schema(
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 90.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Chat completion constrained to a strict JSON schema; returns parsed JSON.

    Used by the extraction pipeline stages, which run many of these calls
    concurrently from worker threads. The shared client is thread-safe.
    """
    client = get_sync_client().with_options(
        timeout=timeout, max_retries=max_retries
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content or "{}")


def complete_json_schema_image(
    *,
    system: str,
    user: str,
    image_url: str,
    schema: dict[str, Any],
    schema_name: str,
    model: str,
    temperature: float = 0.0,
    timeout: float = 90.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Vision variant of :func:`complete_json_schema`; returns parsed JSON.

    ``image_url`` is fetched by the OpenAI API, so it must be publicly
    reachable. Thread-safe, like the shared client it uses.
    """
    client = get_sync_client().with_options(timeout=timeout, max_retries=max_retries)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content or "{}")


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
