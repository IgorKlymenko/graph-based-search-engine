"""The pure parts of the agent-token harness, and the question file it reads.

The harness measures what an agent reads under each strategy; these tests pin
the parts that need no database: the grep the naive agent runs, the tool
result the search hands back, and the mapping from a passage to the file
behind it -- including a document rendered by the code connector, so a
connector change that alters titles or paths shows up here first.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.connectors.github import connector as GH

SCRIPT = (
    Path(__file__).resolve().parents[3] / "public" / "scripts" / "eval_agent_tokens.py"
)


def load():
    spec = importlib.util.spec_from_file_location("eval_agent_tokens", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E = load()


def test_keywords_prefer_identifiers_then_long_words_and_drop_stopwords() -> None:
    kws = E.keywords(
        "What do connector keys settle for a candidate pair (keys_settle)?"
    )
    assert kws[0] == "keys_settle" and "What" not in kws and len(kws) == 3


def test_grep_ranks_by_distinct_terms_then_matches(tmp_path) -> None:
    (tmp_path / "a.py").write_text("stall stall stall\n")
    (tmp_path / "b.py").write_text("stall abandoned\nwarning\n")
    (tmp_path / "c.py").write_text("nothing here\n")
    hits = E.grep_rank(sorted(tmp_path.glob("*.py")), ["stall", "abandoned"])
    assert [h.path.name for h in hits] == ["b.py", "a.py"]
    assert hits[0].distinct == 2 and hits[0].lines[0].endswith(":1:stall abandoned")


def test_sufficient_ignores_case_and_whitespace() -> None:
    assert E.sufficient("The Pool\n  model is shared", ["pool model"])
    assert not E.sufficient("The Pool model", ["pool model", "Supabase"])


def test_measure_charges_one_overhead_per_tool_call() -> None:
    tokens, ok = E.measure("hello world", 3, ["hello"])
    assert tokens == E.count_tokens("hello world") + 3 * E.CALL_OVERHEAD and ok


def test_tool_result_shows_readable_titles_and_contested_names() -> None:
    found = SimpleNamespace(
        passages=[
            ("earlier/scalability.md", "Neo4j is unscalable.", 0.5, None),
            ("IgorKlymenko/corev1/server/public/src/app.py", "module app", 0.25, "u"),
        ],
        contested={"run": ["pkg.jobs.run", "pkg.cli.run"]},
    )
    out = E.tool_result(found)
    assert out.startswith("[1] scalability (score 0.50)\nNeo4j is unscalable.")
    assert "[2] " in out and "run: pkg.jobs.run, pkg.cli.run" in out


def test_passage_paths_map_code_titles_to_the_checkout_and_notes_to_the_vault(
    tmp_path,
) -> None:
    root, vault = tmp_path / "repo", tmp_path / "vault"
    (vault / "earlier").mkdir(parents=True)
    (vault / "earlier" / "scalability.md").write_text("x")
    assert (
        E.passage_path(E.GITHUB_TITLE_PREFIX + "src/app.py", root, vault)
        == root / "src/app.py"
    )
    assert (
        E.passage_path("earlier/scalability.md", root, vault)
        == vault / "earlier/scalability.md"
    )
    assert E.passage_path("thread/C1/1.2", root, vault) is None


def test_a_document_rendered_by_the_code_connector_round_trips_to_its_file(
    tmp_path,
) -> None:
    """The title the connector gives a file is what the search cites, and the
    harness must turn it back into the path an agent would open."""
    (tmp_path / "pkg").mkdir()
    source = (
        b'"""Ingest."""\n\ndef ingest():\n    """Documents in, graph rows out."""\n'
    )
    (tmp_path / "pkg" / "main.py").write_bytes(source)
    text, _mentions = GH.render_file("pkg/main.py", source)
    title = f"{E.GITHUB_TITLE_PREFIX}pkg/main.py"
    found = SimpleNamespace(passages=[(title, text, 0.4, None)], contested={})
    assert "function ingest (pkg.main.ingest)" in E.tool_result(found)
    assert (
        E.passage_path(title, tmp_path, tmp_path / "none") == tmp_path / "pkg/main.py"
    )
    assert E.sufficient(E.tool_result(found), ["graph rows out"])


def test_question_file_is_well_formed_and_points_at_real_files() -> None:
    data = json.loads(E.QUESTIONS.read_text())
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))
    for q in data["questions"]:
        assert q["category"] in {"code", "vault", "cross", "stale"}
        assert q["gold"] and q["gold_answer"] and q["gold_files"]
        for spec in q["gold_files"]:
            if spec.startswith("vault:") and not E.VAULT.is_dir():
                pytest.skip("vault notes are a local artifact")
            assert E.gold_path(spec).is_file(), (q["id"], spec)
