from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from retrieval.exceptions import InvalidInputError

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifiers(
    values: list[str] | None, field_name: str
) -> list[str] | None:
    if values is None:
        return None
    for v in values:
        if not _SAFE_IDENTIFIER.match(v):
            raise InvalidInputError(
                f"{field_name} must match [A-Za-z_][A-Za-z0-9_]*, got: {v!r}"
            )
    return values


class RerankMode(str, Enum):
    embedding = "embedding"
    cross_encoder = "cross_encoder"
    none = "none"


class TemporalMode(str, Enum):
    all = "all"
    currently_true = "currently_true"
    as_of = "as_of"
    changed_since = "changed_since"


class SearchFilters(BaseModel):
    node_labels: list[str] | None = None
    edge_types: list[str] | None = None
    valid_at_after: datetime | None = None
    valid_at_before: datetime | None = None
    created_at_after: datetime | None = None
    created_at_before: datetime | None = None
    edge_uuids: list[str] | None = None

    @field_validator("node_labels")
    @classmethod
    def validate_node_labels(cls, v: list[str] | None) -> list[str] | None:
        return _validate_identifiers(v, "node_labels")

    @field_validator("edge_types")
    @classmethod
    def validate_edge_types(cls, v: list[str] | None) -> list[str] | None:
        return _validate_identifiers(v, "edge_types")


class PPRSearchConfig(BaseModel):
    num_seeds: int = Field(default=10, ge=1, le=100)
    alpha: float = Field(default=0.15, gt=0.0, lt=1.0)
    max_iter: int = Field(default=100, ge=10, le=500)
    tol: float = Field(default=1e-6, gt=0.0)
    top_nodes: int = Field(default=10, ge=1, le=200)
    top_facts: int = Field(default=15, ge=1, le=500)
    candidate_facts: int = Field(default=50, ge=1, le=1000)
    rerank: RerankMode = RerankMode.embedding
    min_relevance: float | None = None
    expand_seeds: bool = True
    expansion_terms: int = Field(default=6, ge=0, le=20)
    expansion_model: str = "gpt-4o-mini"
    max_nodes: int = Field(default=100_000, ge=1)
    max_edges: int = Field(default=500_000, ge=1)
    ppr_timeout_seconds: float = Field(default=10.0, gt=0.0)


class TemporalPPRSearchConfig(BaseModel):
    mode: TemporalMode = TemporalMode.currently_true
    as_of: datetime | None = None
    max_facts: int = Field(default=15, ge=1, le=500)
    hydrate: bool = True
    max_passages: int = Field(default=5, ge=0, le=50)
    ppr: PPRSearchConfig = Field(default_factory=PPRSearchConfig)


class MultihopSearchConfig(BaseModel):
    dynamic: bool = True
    max_hops: int = Field(default=4, ge=1, le=20)
    fixed_hops: int = Field(default=2, ge=1, le=20)
    min_new_facts: int = Field(default=1, ge=0)
    bridge_model: str = "gpt-4o-mini"
    bridge_timeout_seconds: float = Field(default=15.0, gt=0.0)
    total_timeout_seconds: float = Field(default=120.0, gt=0.0)
    temporal: TemporalPPRSearchConfig = Field(default_factory=TemporalPPRSearchConfig)
