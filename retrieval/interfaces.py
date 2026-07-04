from typing import Any, Protocol


class GraphDriver(Protocol):
    async def execute_query(
        self,
        query: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized database query and return rows as dicts."""
        ...


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Return a unit-norm embedding vector for the given text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call where the backend supports it."""
        ...


class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Return the model's text completion for the given prompt."""
        ...

    async def complete_json(
        self,
        prompt: str,
        schema: dict,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> dict:
        """Return a JSON-structured completion validated against `schema`."""
        ...
