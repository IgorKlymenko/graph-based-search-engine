"""The agent's tool kit over a small corpus, and the billing it reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3] / "public" / "scripts" / "eval_agent_loop.py"
)


def load():
    spec = importlib.util.spec_from_file_location("eval_agent_loop", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E = load()


def corpus(tmp_path):
    (tmp_path / "server/public/src").mkdir(parents=True)
    (tmp_path / "ADRs").mkdir()
    (tmp_path / "vault/earlier").mkdir(parents=True)
    (tmp_path / "server/public/src/app.py").write_text(
        "def verify():\n    pass\n\ndef ack():\n    pass\n"
    )
    (tmp_path / "ADRs/x.md").write_text("# Roles\napp_ack inserts only\n")
    (tmp_path / "CLAUDE.md").write_text("rules\n")
    (tmp_path / "vault/earlier/scalability.md").write_text("Neo4j is unscalable\n")
    return E.corpus_files(tmp_path, tmp_path / "vault")


def test_corpus_names_repo_files_by_relative_path_and_notes_under_vault(
    tmp_path,
) -> None:
    files = corpus(tmp_path)
    assert set(files) == {
        "server/public/src/app.py",
        "ADRs/x.md",
        "CLAUDE.md",
        "vault/earlier/scalability.md",
    }


def test_glob_grep_and_read_behave_like_an_agents_tools(tmp_path) -> None:
    tools = E.Tools(corpus(tmp_path))
    assert tools.glob("server/**/*.py") == "server/public/src/app.py"
    assert tools.glob("nothing/*") == "no files match"
    assert tools.grep("APP_ACK") == "ADRs/x.md:2:app_ack inserts only"
    assert (
        tools.grep("def ", "server/*")
        == "server/public/src/app.py:1:def verify():\nserver/public/src/app.py:4:def ack():"
    )
    assert tools.grep("(").startswith("bad pattern")
    assert tools.read("ADRs/x.md") == "1: # Roles\n2: app_ack inserts only"
    assert tools.read("server/public/src/app.py", 4, 5) == "4: def ack():\n5:     pass"
    assert tools.read("missing.md").startswith("no such file")
    assert tools.call("search", {"question": "x"}) == "search is not available"
    assert tools.call("read", {"nope": 1}).startswith("bad arguments")


def test_long_reads_are_cut_and_say_so(tmp_path) -> None:
    files = corpus(tmp_path)
    (tmp_path / "ADRs/long.md").write_text(
        "\n".join(f"line {i}" for i in range(1, 1001))
    )
    tools = E.Tools(E.corpus_files(tmp_path, tmp_path / "vault"))
    out = tools.read("ADRs/long.md")
    assert out.splitlines()[0] == "1: line 1" and "600 more lines" in out
    assert len(files) == 4


def test_kits_expose_the_right_tools_with_strict_schemas() -> None:
    assert [t["function"]["name"] for t in E.tool_specs("files")] == [
        "glob",
        "grep",
        "read",
    ]
    assert [t["function"]["name"] for t in E.tool_specs("search")] == ["search", "read"]
    assert all(t["function"]["strict"] for t in E.tool_specs("both"))


def test_billing_sums_prompt_and_completion_tokens_and_bootstrap_brackets_the_mean() -> (
    None
):
    run = E.Run(
        "hand", "C1", "code", "files", "a", 2, [], 1000, 50, 800, 1.0, "correct"
    )
    assert E.billed(run) == 1050
    lo, hi = E.bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert lo <= 2.5 <= hi
    assert "tok/correct" in E.summarise([run])
