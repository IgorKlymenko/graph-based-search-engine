"""Neo4j-backed GraphDriver implementation."""
from __future__ import annotations

import os
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.time import DateTime as _NeoDateTime
from neo4j.time import Date as _NeoDate
from neo4j.time import Time as _NeoTime


def _to_python(value: Any) -> Any:
    """Convert neo4j temporal types to Python native equivalents."""
    if isinstance(value, (_NeoDateTime, _NeoDate, _NeoTime)):
        return value.to_native()
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


class Neo4jGraphDriver:
    """GraphDriver backed by a Neo4j database via the async bolt driver."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(username, password)
        )
        self._database = database

    @classmethod
    def from_env(cls) -> "Neo4jGraphDriver":
        """Construct from environment variables.

        Required: NEO4J_PASSWORD
        Optional: NEO4J_URI (default bolt://localhost:7687),
                  NEO4J_USERNAME (default neo4j),
                  NEO4J_DATABASE (default neo4j)
        """
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            raise RuntimeError(
                "NEO4J_PASSWORD environment variable is not set. "
                "Add it to your .env file."
            )
        database = os.environ.get("NEO4J_DATABASE", "neo4j")
        return cls(uri, username, password, database)

    async def execute_query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, dict(params))
            records = await result.data()
            return [{k: _to_python(v) for k, v in row.items()} for row in records]

    async def close(self) -> None:
        await self._driver.close()
