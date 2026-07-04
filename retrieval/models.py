from __future__ import annotations

from datetime import datetime

import re as _re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_UUID_RE = _re.compile(r"^[0-9a-fA-F-]{36}$")


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    fact: str
    source_uuid: str
    target_uuid: str
    score: float
    rerank_score: float | None = None
    created_at: datetime
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    episodes: list[str] = Field(default_factory=list)

    @field_validator("uuid", mode="before")
    @classmethod
    def validate_uuid(cls, v: object) -> object:
        if isinstance(v, str) and not _UUID_RE.match(v):
            raise ValueError(f"invalid UUID format: {v!r}")
        return v

    @field_validator("score", "rerank_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    name: str
    content: str
    valid_at: datetime | None = None
    source_description: str = ""
    supports_fact_uuids: list[str] = Field(default_factory=list)
    best_relevance: float | None = None


class RankedNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    uuid: str
    name: str
    summary: str
    score: float
    episodes: list[str] = Field(default_factory=list)


class PPRResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: list[RankedNode]
    facts: list[Fact]
    seed_uuids: list[str]
    expansion_terms: list[str] = Field(default_factory=list)
    hyde_answers: list[str] = Field(default_factory=list)
    num_nodes: int
    num_edges: int


class TemporalPPRResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    facts: list[Fact]
    mode: str
    as_of: datetime
    relevant_count: int
    dropped_by_temporal: int = 0
    num_seeds: int
    num_nodes: int
    num_edges: int
    passages: list[Passage] = Field(default_factory=list)
    seed_uuids: list[str] = Field(default_factory=list)
    expansion_terms: list[str] = Field(default_factory=list)
    hyde_answers: list[str] = Field(default_factory=list)


class MultihopHop(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    temporal: TemporalPPRResults
    new_uuids: frozenset[str]


class MultihopSearchResults(BaseModel):
    model_config = ConfigDict(frozen=True)

    hops: list[MultihopHop]
    merged: list[Fact]
    stop_reason: str
