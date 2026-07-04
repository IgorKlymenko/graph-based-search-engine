import re as _re


class RetrievalError(Exception):
    """Base for all retrieval errors."""


class GraphLoadError(RetrievalError):
    """Database query failed during graph loading."""


class GraphTooLargeError(RetrievalError):
    """Group graph exceeds configured node or edge limits."""

    def __init__(self, num_nodes: int, num_edges: int, limits: dict[str, int]) -> None:
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.limits = limits
        super().__init__(
            f"Graph too large: {num_nodes} nodes, {num_edges} edges "
            f"(limits: {limits})"
        )


class LLMCallError(RetrievalError):
    """LLM API call failed."""


class EmbeddingError(RetrievalError):
    """Embedding API call failed."""


class SearchTimeoutError(RetrievalError):
    """Search exceeded its wall-clock deadline."""


class InvalidInputError(RetrievalError):
    """Caller-supplied input failed validation."""


_GROUP_ID_RE = _re.compile(r"^[A-Za-z0-9_\-]+$")
_MAX_QUERY_LEN = 2000


def validate_inputs(query: str, group_id: str) -> None:
    """Raise InvalidInputError if query or group_id violates API contract."""
    if len(query) > _MAX_QUERY_LEN:
        raise InvalidInputError(f"query exceeds {_MAX_QUERY_LEN} characters")
    if not _GROUP_ID_RE.match(group_id):
        raise InvalidInputError("group_id must match ^[A-Za-z0-9_-]+$")
