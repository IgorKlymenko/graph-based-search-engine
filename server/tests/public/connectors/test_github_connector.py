"""Only changed blobs are fetched; absence within a whole tree retires; junk is skipped."""

from src.connectors import base
from src.connectors.github import connector as GH
from src.inject import episodes as EP

TENANT = "aaaaaaaa-0000-4000-8000-00000000000a"
PY = b'"""Doc."""\n\ndef f():\n    return 1\n'


class FakeClient:
    def __init__(self, tree: dict[str, bytes], head: str = "h1") -> None:
        self.tree_by_path = tree
        self.head_sha = head
        self.fetched: list[str] = []

    def repo(self, full_name):
        return {
            "id": 42,
            "html_url": f"https://github.com/{full_name}",
            "default_branch": "main",
        }

    def head(self, full_name, ref):
        return self.head_sha

    def tree(self, full_name, sha):
        return [
            {"path": p, "type": "blob", "sha": f"sha-{hash(b)}", "size": len(b)}
            for p, b in self.tree_by_path.items()
        ]

    def blob(self, full_name, sha):
        self.fetched.append(sha)
        return next(b for b in self.tree_by_path.values() if f"sha-{hash(b)}" == sha)


def run(client, store, ref="HEAD"):
    connector = GH.GitHubConnector(
        client, ["acme/core"], cursors=GH.cursors_in(store), ref=ref
    )
    counts = base.sync(store, connector, tenant_id=TENANT)
    return connector, counts


def test_documents_are_keyed_by_repo_id_and_path_with_a_citation_url() -> None:
    store = EP.empty_store()
    run(FakeClient({"src/a.py": PY}), store)  # no src/__init__.py: src is layout
    (meta,) = store["files"].values()
    assert meta["external_id"] == "repo/42/src/a.py"
    assert meta["title"] == "acme/core/src/a.py"
    assert meta["url"] == "https://github.com/acme/core/blob/main/src/a.py"
    (episode,) = store["episodes"].values()
    assert episode["text"].startswith("module a (a), file a.py at src/a.py")
    assert episode["seeds"]


def test_an_unchanged_head_costs_no_tree_or_blob_calls() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY})
    run(client, store)
    fetched = len(client.fetched)
    connector, counts = run(client, store)
    assert len(client.fetched) == fetched
    assert connector.counts["unchanged_repos"] == 1
    assert counts.documents == 0


def test_a_new_head_fetches_only_blobs_whose_sha_changed() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY, "src/b.py": PY.replace(b"f()", b"g()")})
    run(client, store)
    client.head_sha = "h2"
    client.tree_by_path["src/b.py"] = PY.replace(b"f()", b"h()")
    client.fetched.clear()
    _, counts = run(client, store)
    assert len(client.fetched) == 1
    assert counts.documents == 1


def test_a_path_gone_from_the_tree_is_retired() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY, "src/b.py": PY.replace(b"f()", b"g()")})
    run(client, store)
    client.head_sha = "h2"
    del client.tree_by_path["src/b.py"]
    run(client, store)
    gone = store["files"][EP.file_id_for(TENANT, "github", "repo/42/src/b.py")]
    kept = store["files"][EP.file_id_for(TENANT, "github", "repo/42/src/a.py")]
    assert gone["removed_at"] and not kept["removed_at"]


def test_a_render_version_bump_refetches_everything_and_still_retires() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY, "src/b.py": PY.replace(b"f()", b"g()")})
    run(client, store)
    GH.cursors_in(store)["42"]["render_version"] = (
        0  # as if rendered by an old renderer
    )
    client.head_sha = "h2"
    del client.tree_by_path["src/b.py"]
    client.fetched.clear()
    run(client, store)
    assert len(client.fetched) == 1
    gone = store["files"][EP.file_id_for(TENANT, "github", "repo/42/src/b.py")]
    assert gone["removed_at"]


def test_vendored_binary_large_and_unknown_files_are_skipped() -> None:
    assert GH.wanted("src/a.py", 10)
    assert GH.wanted("README.md", 10)
    assert not GH.wanted("node_modules/x/index.js", 10)
    assert not GH.wanted("src/big.py", GH.MAX_BLOB_BYTES + 1)
    assert not GH.wanted("assets/logo.png", 10)
    assert GH.render_file("a.py", b"\x00\x01binary") is None


def test_a_file_the_parser_rejects_is_counted_not_fatal_and_not_remembered() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/old.py": b"print 'python 2'\n", "src/a.py": PY})
    connector, counts = run(client, store)
    assert connector.counts["unparsed"] == 1
    assert counts.documents == 1
    assert "src/old.py" not in GH.cursors_in(store)["42"]["blobs"]


def test_markdown_goes_through_strip_markup_without_seeds() -> None:
    store = EP.empty_store()
    run(FakeClient({"README.md": b"# Title\n\nSee **ingest** and [[Okta]].\n"}), store)
    (episode,) = store["episodes"].values()
    assert episode["text"] == "Title\n\nSee ingest and Okta."
    assert "seeds" not in episode


def test_an_explicit_ref_is_used_in_the_url() -> None:
    store = EP.empty_store()
    run(FakeClient({"src/a.py": PY}), store, ref="v1.2")
    (meta,) = store["files"].values()
    assert meta["url"].endswith("/blob/v1.2/src/a.py")


def test_include_prefixes_narrow_the_repo_and_retire_what_falls_outside() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY, "legacy/b.py": PY.replace(b"f()", b"g()")})
    run(client, store)
    assert len(store["files"]) == 2
    # same head: narrowing the scope is itself a change
    connector = GH.GitHubConnector(
        client, ["acme/core"], cursors=GH.cursors_in(store), include=("src",)
    )
    base.sync(store, connector, tenant_id=TENANT)
    gone = store["files"][EP.file_id_for(TENANT, "github", "repo/42/legacy/b.py")]
    kept = store["files"][EP.file_id_for(TENANT, "github", "repo/42/src/a.py")]
    assert gone["removed_at"] and not kept["removed_at"]


def test_an_include_that_names_a_single_file_keeps_that_file() -> None:
    store = EP.empty_store()
    client = FakeClient({"CLAUDE.md": b"# Rules\n", "src/a.py": PY})
    connector = GH.GitHubConnector(
        client, ["acme/core"], cursors=GH.cursors_in(store), include=("CLAUDE.md",)
    )
    base.sync(store, connector, tenant_id=TENANT)
    assert [f["external_id"] for f in store["files"].values()] == ["repo/42/CLAUDE.md"]


A_RUN = b"def run():\n    return 1\n"
B_RUN = b"def run():\n    return 2\n"
C_CALLS = b"from pkg.a import run\n\ndef go():\n    return run()\n"


def run_indexed(client, store):
    connector = GH.GitHubConnector(
        client, ["acme/core"], cursors=GH.cursors_in(store), index=True
    )
    return connector, base.sync(store, connector, tenant_id=TENANT)


def test_a_colliding_name_renders_qualified_and_lands_in_the_index() -> None:
    store = EP.empty_store()
    run_indexed(
        FakeClient(
            {
                "pkg/__init__.py": b"",
                "pkg/a.py": A_RUN,
                "pkg/b.py": B_RUN,
                "pkg/c.py": C_CALLS,
            }
        ),
        store,
    )
    texts = {
        m["external_id"]: "\n".join(
            store["episodes"][e]["text"] for e in m["episode_ids"]
        )
        for m in store["files"].values()
    }
    assert "calls: pkg.a.run" in texts["repo/42/pkg/c.py"]
    assert "function run (pkg.a.run)" in texts["repo/42/pkg/a.py"]
    a = store["files"][EP.file_id_for(TENANT, "github", "repo/42/pkg/a.py")]
    seeds = store["episodes"][a["episode_ids"][0]]["seeds"]
    spans = [sp[2] for seed in seeds for sp in seed["spans"]]
    assert "pkg.a.run" in spans and "run" not in spans  # bare name is plain text
    index = texts["repo/42/.kg-index"]
    assert "run is defined more than once: pkg.a.run, pkg.b.run" in index
    cursor = GH.cursors_in(store)["42"]
    assert cursor["ambiguous"] == ["run"] and cursor["index"] is True
    assert set(cursor["symbols"]) == {
        "pkg/__init__.py",
        "pkg/a.py",
        "pkg/b.py",
        "pkg/c.py",
    }


def test_a_name_that_stops_colliding_re_renders_the_unchanged_caller() -> None:
    """c.py's content did not change, but what its call means did."""
    store = EP.empty_store()
    client = FakeClient(
        {
            "pkg/__init__.py": b"",
            "pkg/a.py": A_RUN,
            "pkg/b.py": B_RUN,
            "pkg/c.py": C_CALLS,
        }
    )
    run_indexed(client, store)
    client.head_sha = "h2"
    del client.tree_by_path["pkg/b.py"]
    client.fetched.clear()
    run_indexed(client, store)
    c = store["files"][EP.file_id_for(TENANT, "github", "repo/42/pkg/c.py")]
    text = "\n".join(store["episodes"][e]["text"] for e in c["episode_ids"])
    assert "calls: run" in text and "pkg.a.run" not in text.split("calls:")[1]
    # c.py (its call) and a.py (its module's defines: line) by their known shas;
    # __init__.py shows no name that flipped and is not fetched
    assert len(client.fetched) == 2
    index = store["files"][EP.file_id_for(TENANT, "github", "repo/42/.kg-index")]
    assert index["removed_at"]  # no collisions left: the index is retired
    assert GH.cursors_in(store)["42"]["ambiguous"] == []


def test_the_index_is_off_unless_asked_for() -> None:
    store = EP.empty_store()
    run(FakeClient({"pkg/a.py": A_RUN, "pkg/b.py": B_RUN}), store)
    assert not any(
        m["external_id"].endswith(".kg-index") for m in store["files"].values()
    )


def test_a_new_citation_base_re_emits_unchanged_files() -> None:
    store = EP.empty_store()
    client = FakeClient({"src/a.py": PY})
    run(client, store)
    client.repo = lambda full_name: {
        "id": 42,
        "html_url": "https://github.com/acme/moved",
        "default_branch": "main",
    }
    client.fetched.clear()
    _, counts = run(client, store)
    assert counts.documents == 1 and len(client.fetched) == 1
    (meta,) = store["files"].values()
    assert meta["url"] == "https://github.com/acme/moved/blob/main/src/a.py"
    assert not any(e.get("removed_at") for e in store["episodes"].values())
