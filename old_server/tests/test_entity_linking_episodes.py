"""Episode identity, splitting, and change detection across document edits."""

from __future__ import annotations

import pytest

from entity_linking import episodes as E

TENANT_A = "aaaaaaaa-0000-0000-0000-000000000001"
TENANT_B = "bbbbbbbb-0000-0000-0000-000000000002"
DOC_A = dict(tenant_id=TENANT_A, source="upload", external_id="notes/doc.md")


def para(label: str, words: int = 40) -> str:
    return f"{label}. " + "word " * words


@pytest.fixture
def store() -> dict:
    return E.empty_store()


# ── identity ──────────────────────────────────────────────────────────────────

def test_stable_uuid_is_deterministic() -> None:
    assert E.stable_uuid("file:x") == E.stable_uuid("file:x")
    assert E.stable_uuid("file:x") != E.stable_uuid("file:y")


def test_file_id_is_scoped_by_tenant_source_and_external_id() -> None:
    base = E.file_id_for(TENANT_A, "upload", "notes/doc.md")
    assert base != E.file_id_for(TENANT_B, "upload", "notes/doc.md")
    assert base != E.file_id_for(TENANT_A, "slack", "notes/doc.md")
    assert base != E.file_id_for(TENANT_A, "upload", "notes/other.md")
    assert base == E.file_id_for(TENANT_A, "upload", "notes/doc.md")


def test_rel_path_is_invariant_to_how_the_root_was_given(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    doc = tmp_path / "sub" / "a.md"
    doc.write_text("x")
    from_absolute = E.rel_path(doc, tmp_path)
    from_indirect = E.rel_path(doc, tmp_path / "sub" / "..")
    assert from_absolute == "sub/a.md"
    assert from_absolute == from_indirect


def test_render_version_falls_back_for_unknown_sources() -> None:
    assert E.render_version_for("upload") == E.RENDER_VERSIONS["upload"]
    assert E.render_version_for("not-a-connector") == E.DEFAULT_RENDER_VERSION


# ── splitting ─────────────────────────────────────────────────────────────────

def test_episodes_are_exact_slices() -> None:
    doc = "First.\n\nSecond para is here.\n\n\nThird after blank runs."
    eps = E.split_episodes(doc, max_chars=1000)
    assert eps
    for ep in eps:
        assert doc[ep.char_start : ep.char_end] == ep.text


def test_paragraphs_merge_up_to_max_chars() -> None:
    doc = "\n\n".join([para("P", 60)] * 5)
    eps = E.split_episodes(doc, max_chars=1000)
    assert len(eps) >= 2
    for ep in eps:
        assert doc[ep.char_start : ep.char_end] == ep.text


def test_paragraph_longer_than_max_chars_is_not_split() -> None:
    """``max_chars`` is a merge threshold, not a hard cap."""
    big = "y" * 5000
    eps = E.split_episodes(big, max_chars=1000)
    assert len(eps) == 1
    assert eps[0].text == big


def test_blank_documents_yield_no_episodes() -> None:
    assert E.split_episodes("") == []
    assert E.split_episodes("\n\n   \n\n") == []


def test_offsets_are_monotonic_and_nonoverlapping() -> None:
    doc = "\n\n".join(f"paragraph number {i}" for i in range(10))
    prev_end = -1
    for ep in E.split_episodes(doc, max_chars=60):
        assert ep.char_start > prev_end
        prev_end = ep.char_end


# ── records ───────────────────────────────────────────────────────────────────

def test_record_shape() -> None:
    record = E.build_records("file-1", "Only paragraph.")[0]
    assert set(record) == {"episode_id", "file_id", "text"}


def test_reading_order_is_recoverable_from_episode_ids(store) -> None:
    """Order lives in ``files.episode_ids``, not on the episode.

    A consumer holding only an episode id (an MCP tool asking for neighbouring
    episodes) finds its position there, so nothing needs an ordering column.
    """
    paras = [para(f"P{i}", 160) for i in range(4)]
    diff = E.sync_file(store, "\n\n".join(paras), max_chars=1000, **DOC_A)
    before = store["files"][diff.file_id]["episode_ids"]
    assert len(before) == len(paras)

    E.sync_file(store, "\n\n".join([para("Inserted", 160)] + paras),
                max_chars=1000, **DOC_A)
    after = store["files"][diff.file_id]["episode_ids"]
    assert after[1:] == before  # every episode shifted by one, order intact


def test_insertion_moves_episodes_without_re_extracting(store) -> None:
    paras = [para(f"P{i}", 160) for i in range(3)]
    E.sync_file(store, "\n\n".join(paras), max_chars=1000, **DOC_A)
    for episode in store["episodes"].values():
        episode["extracted_at"] = "2026-07-30T00:00:00+00:00"

    E.sync_file(store, "\n\n".join([para("Inserted", 160)] + paras),
                max_chars=1000, **DOC_A)
    live = [e for e in store["episodes"].values() if e["removed_at"] is None]
    assert len(live) == len(paras) + 1
    kept = [e for e in live if e["extracted_at"]]
    assert len(kept) == len(paras)  # positions moved, extraction survived


def test_episode_id_depends_on_text_and_file() -> None:
    doc = "Only paragraph."
    same = E.build_records("file-1", doc)[0]["episode_id"]
    assert E.build_records("file-1", doc)[0]["episode_id"] == same
    assert E.build_records("file-2", doc)[0]["episode_id"] != same
    assert E.build_records("file-1", "Different.")[0]["episode_id"] != same


def test_identical_episodes_collapse_to_one_record() -> None:
    block = para("Same block", 80)
    records = E.build_records("file-1", block + "\n\n" + block, max_chars=1000)
    assert len(records) == 1


def test_position_is_recoverable_from_text() -> None:
    doc = "\n\n".join(para(f"P{i}", 80) for i in range(4))
    records = E.build_records("file-1", doc, max_chars=1000)
    pos = 0
    for record in records:
        found = doc.find(record["text"], pos)
        assert found >= 0
        assert doc[found : found + len(record["text"])] == record["text"]
        pos = found + len(record["text"])


# ── change detection ──────────────────────────────────────────────────────────

def test_first_sync_reports_every_episode_as_added(store) -> None:
    doc = "\n\n".join(para(f"P{i}", 80) for i in range(3))
    diff = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    assert diff.status == "new"
    assert len(diff.added) == len(store["files"][diff.file_id]["episode_ids"])
    assert diff.unchanged == [] and diff.removed == []
    assert diff.needs_work == diff.added


def test_resync_of_untouched_document_short_circuits(store) -> None:
    doc = "\n\n".join(para(f"P{i}", 80) for i in range(3))
    first = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    again = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    assert again.status == "unchanged"
    assert again.unchanged == first.added
    assert again.added == [] and again.removed == []


def test_append_keeps_existing_episodes(store) -> None:
    doc = "\n\n".join(para(f"P{i}", 160) for i in range(3))
    first = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    diff = E.sync_file(store, doc + "\n\n" + para("Appended", 160),
                       max_chars=1000, **DOC_A)
    assert diff.status == "updated"
    assert set(diff.unchanged) == set(first.added)
    assert len(diff.added) == 1
    assert diff.removed == []


def test_editing_one_paragraph_touches_only_its_episode(store) -> None:
    paras = [para(f"P{i}", 160) for i in range(4)]
    E.sync_file(store, "\n\n".join(paras), max_chars=1000, **DOC_A)
    paras[2] = para("P2 REWRITTEN", 160)
    diff = E.sync_file(store, "\n\n".join(paras), max_chars=1000, **DOC_A)
    assert len(diff.added) == 1
    assert len(diff.removed) == 1
    assert len(diff.unchanged) == 3


def test_removed_episodes_are_retired_not_deleted(store) -> None:
    paras = [para(f"P{i}", 160) for i in range(3)]
    E.sync_file(store, "\n\n".join(paras), max_chars=1000, **DOC_A)
    diff = E.sync_file(store, "\n\n".join(paras[:2]), max_chars=1000, **DOC_A)
    assert len(diff.removed) == 1
    retired = store["episodes"][diff.removed[0]]
    assert retired["removed_at"] is not None
    assert retired["text"]  # record kept so downstream products can be retired


def test_restored_text_clears_the_retirement_stamp(store) -> None:
    paras = [para(f"P{i}", 160) for i in range(3)]
    full = "\n\n".join(paras)
    E.sync_file(store, full, max_chars=1000, **DOC_A)
    dropped = E.sync_file(store, "\n\n".join(paras[:2]), max_chars=1000, **DOC_A)
    gone = dropped.removed[0]
    assert store["episodes"][gone]["removed_at"] is not None

    restored = E.sync_file(store, full, max_chars=1000, **DOC_A)
    assert gone in restored.added
    assert store["episodes"][gone]["removed_at"] is None


def test_same_external_id_in_two_tenants_does_not_collide(store) -> None:
    doc = para("Shared", 80)
    a = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    b = E.sync_file(store, doc, max_chars=1000, tenant_id=TENANT_B,
                    source="upload", external_id="notes/doc.md")
    assert a.file_id != b.file_id
    assert len(store["files"]) == 2
    assert set(a.added).isdisjoint(b.added)


def test_render_version_bump_forces_a_resplit(store, monkeypatch) -> None:
    doc = para("Body", 80)
    E.sync_file(store, doc, max_chars=1000, **DOC_A)
    assert E.sync_file(store, doc, max_chars=1000, **DOC_A).status == "unchanged"

    monkeypatch.setitem(E.RENDER_VERSIONS, "upload", E.RENDER_VERSIONS["upload"] + 1)
    bumped = E.sync_file(store, doc, max_chars=1000, **DOC_A)
    assert bumped.status == "updated"
    expected = E.RENDER_VERSIONS["upload"]
    assert store["files"][bumped.file_id]["render_version"] == expected


def test_render_version_bump_is_scoped_to_one_source(store, monkeypatch) -> None:
    doc = para("Body", 80)
    E.sync_file(store, doc, max_chars=1000, **DOC_A)
    E.sync_file(store, doc, max_chars=1000, tenant_id=TENANT_A,
                source="slack", external_id="C1/1.0")

    monkeypatch.setitem(E.RENDER_VERSIONS, "slack", E.RENDER_VERSIONS["slack"] + 1)
    assert E.sync_file(store, doc, max_chars=1000, **DOC_A).status == "unchanged"
    slack = E.sync_file(store, doc, max_chars=1000, tenant_id=TENANT_A,
                        source="slack", external_id="C1/1.0")
    assert slack.status == "updated"


# ── corpus sync ───────────────────────────────────────────────────────────────

def _corpus(root) -> None:
    (root / "sub").mkdir()
    (root / "a.md").write_text(para("Alpha", 80))
    (root / "sub" / "b.md").write_text(para("Beta", 80))
    (root / "empty.md").write_text("   \n\n  ")
    (root / "ignored.txt").write_text(para("Not markdown", 80))


def test_sync_corpus_walks_markdown_recursively(store, tmp_path) -> None:
    _corpus(tmp_path)
    diffs = E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    assert sorted(d.external_id for d in diffs) == ["a.md", "sub/b.md"]
    assert all(d.status == "new" for d in diffs)


def test_second_corpus_pass_is_unchanged(store, tmp_path) -> None:
    _corpus(tmp_path)
    E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    diffs = E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    assert all(d.status == "unchanged" for d in diffs)
    assert all(not d.added for d in diffs)


def test_deleted_file_retires_its_episodes(store, tmp_path) -> None:
    _corpus(tmp_path)
    E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    (tmp_path / "sub" / "b.md").unlink()

    diffs = E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    retired = [d for d in diffs if d.external_id == "sub/b.md"]
    assert len(retired) == 1 and retired[0].removed
    meta = store["files"][retired[0].file_id]
    assert meta["removed_at"] is not None
    assert all(store["episodes"][e]["removed_at"] for e in meta["episode_ids"])


def test_corpus_sync_does_not_retire_another_tenants_documents(store, tmp_path):
    _corpus(tmp_path)
    E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    E.sync_corpus(store, tmp_path, tenant_id=TENANT_B)
    (tmp_path / "sub" / "b.md").unlink()

    E.sync_corpus(store, tmp_path, tenant_id=TENANT_A)
    live_b = [m for m in store["files"].values()
              if m["tenant_id"] == TENANT_B and not m["removed_at"]]
    assert len(live_b) == 2


# ── persistence ───────────────────────────────────────────────────────────────

def test_store_survives_a_save_load_round_trip(store, tmp_path) -> None:
    E.sync_file(store, para("Body", 80), max_chars=1000, **DOC_A)
    path = tmp_path / "store.json"
    E.save_store(path, store)
    assert E.load_store(path) == store


def test_load_store_returns_an_empty_store_when_absent(tmp_path) -> None:
    assert E.load_store(tmp_path / "missing.json") == E.empty_store()
