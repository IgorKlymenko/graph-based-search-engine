"""The vector sidecar must skip what it already has and ignore another model's vectors."""

import numpy as np

from src.inject import encoding as EN
from src.inject.main import (
    cached,
    empty_vector_cache,
    load_vector_cache,
    save_vector_cache,
)


def test_cached_computes_only_the_missing_keys() -> None:
    calls: list[list[str]] = []

    def compute(keys):
        calls.append(list(keys))
        return {k: np.full(3, len(k)) for k in keys}

    cache = {"a": np.zeros(3)}
    out = cached(["a", "bb", "ccc"], cache, compute)
    assert calls == [["bb", "ccc"]]
    assert set(out) == {"a", "bb", "ccc"} == set(cache)
    cached(["a", "bb"], cache, compute)
    assert len(calls) == 1


def test_round_trip_through_npz(tmp_path) -> None:
    cache = empty_vector_cache()
    cache["names"] = {"Sarah Chen": np.arange(4.0), "Okta": np.ones(4)}
    cache["contexts"] = {"c1|deadbeef": np.full(4, 0.5)}
    path = tmp_path / "store.vectors.npz"
    save_vector_cache(path, cache)

    back = load_vector_cache(path)
    assert back["model"] == EN.MODEL_NAME
    np.testing.assert_array_equal(back["names"]["Sarah Chen"], np.arange(4.0))
    assert set(back["names"]) == {"Sarah Chen", "Okta"}
    assert set(back["contexts"]) == {"c1|deadbeef"}


def test_an_empty_cache_round_trips(tmp_path) -> None:
    path = tmp_path / "empty.npz"
    save_vector_cache(path, empty_vector_cache())
    back = load_vector_cache(path)
    assert back["names"] == {} and back["contexts"] == {}


def test_another_models_vectors_are_ignored(tmp_path) -> None:
    cache = empty_vector_cache()
    cache["model"] = "someone-else/embedder"
    cache["names"] = {"Okta": np.ones(2)}
    path = tmp_path / "other.npz"
    save_vector_cache(path, cache)
    assert load_vector_cache(path)["names"] == {}


def test_a_missing_file_is_an_empty_cache(tmp_path) -> None:
    assert load_vector_cache(tmp_path / "nope.npz")["names"] == {}
    assert load_vector_cache(None)["contexts"] == {}
