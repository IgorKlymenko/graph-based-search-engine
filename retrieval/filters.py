from __future__ import annotations

from datetime import datetime, timezone

from retrieval.config import TemporalMode
from retrieval.models import Fact


def make_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def temporal_match(fact: Fact, mode: TemporalMode, as_of: datetime) -> bool:
    """Return True if `fact` satisfies the temporal predicate for `mode` at `as_of`."""
    v = make_aware(fact.valid_at)
    i = make_aware(fact.invalid_at)
    t = make_aware(as_of)

    match mode:
        case TemporalMode.all:
            return True
        case TemporalMode.currently_true:
            return i is None
        case TemporalMode.as_of:
            return (v is None or v <= t) and (i is None or i > t)
        case TemporalMode.changed_since:
            return i is not None and i > t
