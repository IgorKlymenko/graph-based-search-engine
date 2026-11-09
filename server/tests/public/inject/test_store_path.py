"""`--store out/store.json` must work before `out/` exists; the first save creates it."""

from src.inject import episodes as EP


def test_save_store_creates_the_parent_directory(tmp_path) -> None:
    path = tmp_path / "out" / "nested" / "store.json"
    EP.save_store(path, {"files": {}})
    assert EP.load_store(path)["files"] == {}


def test_save_store_overwrites_in_place(tmp_path) -> None:
    path = tmp_path / "store.json"
    EP.save_store(path, {"files": {"a": 1}})
    EP.save_store(path, {"files": {}})
    assert EP.load_store(path)["files"] == {}
