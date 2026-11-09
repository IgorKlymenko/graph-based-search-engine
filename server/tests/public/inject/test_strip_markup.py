"""Markup must not survive into surfaces on the upload path.

Ported from the old copy's suite (`old_server/tests/test_entity_linking_markup.py`),
which is where the full set of cases lives; this pins the port and the wiring.
"""

from src.inject import episodes as EP

strip = EP.strip_markup


def test_highlights_lose_their_delimiters() -> None:
    assert strip("- CTO ==best at seed==") == "- CTO best at seed"


def test_wiki_link_targets_survive_as_text() -> None:
    assert strip("based on [[Potential Markets]] clearly") == (
        "based on Potential Markets clearly"
    )


def test_code_fences_go_but_the_code_stays() -> None:
    assert strip("intro\n```SQL\nSELECT member FROM t\n```\nafter") == (
        "intro\nSELECT member FROM t\nafter"
    )


def test_indented_fences_go_too() -> None:
    assert strip("\t  ```\nx = 1\n\t  ```") == "x = 1\n"


def test_sync_corpus_strips_before_hashing(tmp_path) -> None:
    (tmp_path / "a.md").write_text("==x== is [[Okta]]\n\n```sql\nselect 1\n```\n")
    store = EP.empty_store()
    EP.sync_corpus(store, tmp_path, tenant_id="t")
    texts = [e["text"] for e in store["episodes"].values()]
    assert texts
    assert all("==" not in t and "[[" not in t and "```" not in t for t in texts)
    assert "Okta" in "".join(texts)


def test_the_upload_render_version_records_the_change() -> None:
    assert EP.RENDER_VERSIONS["upload"] == 2
