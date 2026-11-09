"""``base.sync`` is the one door: episodes diffed, seeds cut per episode, gone retired."""

from src.connectors import base
from src.inject import episodes as EP

TENANT = "aaaaaaaa-0000-4000-8000-00000000000a"


class Fake:
    source = "github"

    def __init__(self, docs, removed=()):
        self._docs, self._removed = docs, list(removed)

    def documents(self):
        yield from self._docs

    def removed(self):
        yield from self._removed


def doc(text: str, mentions=(), external_id: str = "repo/1/a.py") -> base.Document:
    return base.Document(external_id, text, title="t", url="u", mentions=mentions)


def test_seeds_land_on_the_episode_that_holds_them_with_local_offsets() -> None:
    p1 = "function a (m.a)"
    p2 = "function b (m.b)\ncalls: a"
    text = f"{p1}\n\n{p2}"
    mentions = (
        base.Mention(9, 10, "a", "m.a"),
        base.Mention(12, 15, "m.a", "m.a"),
        base.Mention(len(p1) + 2 + 9, len(p1) + 2 + 10, "b", "m.b"),
        base.Mention(len(text) - 1, len(text), "a", "m.a"),
    )
    store = EP.empty_store()
    base.sync(store, Fake([doc(text, mentions)]), tenant_id=TENANT)
    fid = EP.file_id_for(TENANT, "github", "repo/1/a.py")
    # max_chars=1000 packs both paragraphs into one episode
    (episode,) = store["episodes"].values()
    assert episode["file_id"] == fid
    assert episode["seeds"] == [
        {"key": "m.a", "spans": [[9, 10, "a"], [12, 15, "m.a"],
                                 [len(text) - 1, len(text), "a"]]},
        {"key": "m.b", "spans": [[len(p1) + 2 + 9, len(p1) + 2 + 10, "b"]]},
    ]
    for seed in episode["seeds"]:
        for start, end, surface in seed["spans"]:
            assert episode["text"][start:end] == surface


def test_a_document_without_mentions_gets_no_seeds() -> None:
    store = EP.empty_store()
    base.sync(store, Fake([doc("plain prose")]), tenant_id=TENANT)
    (episode,) = store["episodes"].values()
    assert "seeds" not in episode


def test_changed_seeds_on_unchanged_text_reset_extraction() -> None:
    """A better parser must re-seed, and the only way is another extraction."""
    text = "function a (m.a)"
    store = EP.empty_store()
    base.sync(store, Fake([doc(text, (base.Mention(9, 10, "a", "m.a"),))]),
              tenant_id=TENANT)
    (episode,) = store["episodes"].values()
    episode["extracted_at"] = "2026-09-18T00:00:00"
    base.sync(store, Fake([doc(text, (base.Mention(9, 10, "a", "m.a"),))]),
              tenant_id=TENANT)
    assert episode["extracted_at"] is not None  # same seeds: nothing to redo
    base.sync(store, Fake([doc(text, (base.Mention(9, 10, "a", "m.a"),
                                      base.Mention(12, 15, "m.a", "m.a")))]),
              tenant_id=TENANT)
    assert episode["extracted_at"] is None


def test_removed_ids_retire_the_file_and_its_episodes() -> None:
    store = EP.empty_store()
    base.sync(store, Fake([doc("some text")]), tenant_id=TENANT)
    counts = base.sync(store, Fake([], removed=["repo/1/a.py"]), tenant_id=TENANT)
    fid = EP.file_id_for(TENANT, "github", "repo/1/a.py")
    assert store["files"][fid]["removed_at"]
    assert all(e["removed_at"] for e in store["episodes"].values())
    assert counts.episodes_retired == 1


def test_retiring_twice_or_an_unknown_file_is_a_no_op() -> None:
    store = EP.empty_store()
    assert EP.retire_file(store, "nope") == []
    base.sync(store, Fake([doc("some text")]), tenant_id=TENANT)
    fid = EP.file_id_for(TENANT, "github", "repo/1/a.py")
    assert len(EP.retire_file(store, fid)) == 1
    assert EP.retire_file(store, fid) == []


def test_title_and_url_ride_on_the_file_record() -> None:
    store = EP.empty_store()
    base.sync(store, Fake([doc("some text")]), tenant_id=TENANT)
    (meta,) = store["files"].values()
    assert (meta["title"], meta["url"]) == ("t", "u")
    base.sync(store, Fake([base.Document("repo/1/a.py", "some text", "t2", "u2")]),
              tenant_id=TENANT)
    assert (meta["title"], meta["url"]) == ("t2", "u2")  # updated on the cheap path


def test_counts_describe_the_run() -> None:
    store = EP.empty_store()
    counts = base.sync(store, Fake([doc("x", (base.Mention(0, 1, "x", "k"),)),
                                    doc("y", external_id="repo/1/b.py")]),
                       tenant_id=TENANT)
    assert counts == base.SyncCounts(documents=2, changed=2, episodes_added=2,
                                     episodes_retired=0, seeded=1)


def test_seeded_surfaces_are_names_without_asking_the_judge() -> None:
    """A qualified name is a name by construction; a bare word keeps the prose verdict."""
    store = EP.empty_store()
    store["names"] = {"work": False, "Scorer": False}
    text = "function work (m.jobs.work)\nclass Scorer (m.Scorer)"
    mentions = (base.Mention(9, 13, "work", "m.jobs.work"),
                base.Mention(15, 26, "m.jobs.work", "m.jobs.work"),
                base.Mention(34, 40, "Scorer", "m.Scorer"),
                base.Mention(42, 50, "m.Scorer", "m.Scorer"))
    assert all(text[m.start:m.end] == m.text for m in mentions)
    base.sync(store, Fake([doc(text, mentions)]), tenant_id=TENANT)
    assert store["names"]["m.jobs.work"] is True
    assert store["names"]["work"] is False          # prose already said no
    assert store["names"]["Scorer"] is True         # identifier-shaped: overridden
    assert store["names"]["m.Scorer"] is True
    store["names"].pop("work")
    base.sync(store, Fake([doc(text + " ", mentions)]), tenant_id=TENANT)
    assert store["names"]["work"] is True           # no prose verdict: a name
