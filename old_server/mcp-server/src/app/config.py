"""Environment-driven settings. All secrets come from the environment
(Railway service variables in production, .env locally — never committed)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Railway injects PORT; MCP_PORT is the local-dev fallback.
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("MCP_PORT", "8848")))

# Bearer token required on every request (except /health). The server
# refuses to start without it — see main.main().
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DEFAULT_GROUP_ID = os.getenv("DEFAULT_GROUP_ID", "testing-1")

# Must match graph_upload.py's ENTITY_EMBED_MODEL — queries and the stored
# Entity.name_embedding vectors must come from the same model.
QUERY_EMBED_MODEL = os.getenv("QUERY_EMBED_MODEL", "text-embedding-3-large")

# Cap per-file note content returned by search_notes.
MAX_CONTENT_CHARS = int(os.getenv("MAX_CONTENT_CHARS", "6000"))
