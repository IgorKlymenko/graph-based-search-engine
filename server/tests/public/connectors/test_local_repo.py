"""A checkout answers the four client calls the connector needs, with git's shas."""

import subprocess

from src.connectors import base
from src.connectors.github import connector as GH
from src.connectors.github.local import LocalRepo
from src.inject import episodes as EP

TENANT = "aaaaaaaa-0000-4000-8000-00000000000a"


def checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text('"""A."""\n\ndef f():\n    return 1\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    return LocalRepo(tmp_path)


def test_blob_shas_match_git(tmp_path) -> None:
    repo = checkout(tmp_path)
    (entry,) = [e for e in repo.tree("x", "") if e["path"] == "pkg/a.py"]
    expected = subprocess.run(
        ["git", "-C", str(tmp_path), "hash-object", "pkg/a.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert entry["sha"] == expected
    assert repo.blob("x", entry["sha"]).startswith(b'"""A."""')


def test_the_connector_syncs_a_checkout_like_a_repo(tmp_path) -> None:
    repo = checkout(tmp_path)
    store = EP.empty_store()
    connector = GH.GitHubConnector(repo, ["local/x"], cursors=GH.cursors_in(store))
    counts = base.sync(store, connector, tenant_id=TENANT)
    assert counts.documents == 2  # __init__.py renders its (empty) module paragraph
    texts = [e["text"] for e in store["episodes"].values()]
    assert any(t.startswith("module a (pkg.a), file a.py at pkg/a.py") for t in texts)
    again = base.sync(
        store,
        GH.GitHubConnector(repo, ["local/x"], cursors=GH.cursors_in(store)),
        tenant_id=TENANT,
    )
    assert again.documents == 0


def test_citations_point_at_the_github_remote_when_there_is_one(tmp_path) -> None:
    repo = checkout(tmp_path)
    assert repo.repo("x")["html_url"].startswith("file://")
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "git@github.com:acme/core.git",
        ],
        check=True,
    )
    assert repo.repo("x")["html_url"] == "https://github.com/acme/core"


def test_an_uncommitted_edit_changes_the_head(tmp_path) -> None:
    repo = checkout(tmp_path)
    before = repo.head("x", "HEAD")
    (tmp_path / "pkg" / "a.py").write_text('"""A."""\n\ndef f():\n    return 2\n')
    assert repo.head("x", "HEAD") != before


def test_untracked_files_are_in_the_tree_and_ignored_ones_are_not(tmp_path) -> None:
    repo = checkout(tmp_path)
    (tmp_path / "pkg" / "new.py").write_text("def g():\n    return 2\n")  # never added
    (tmp_path / ".gitignore").write_text("scratch/\n")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "junk.py").write_text("x = 1\n")
    paths = {e["path"] for e in repo.tree("x", "")}
    assert "pkg/new.py" in paths and "scratch/junk.py" not in paths
