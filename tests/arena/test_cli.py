"""Tests for arena/cli.py using Click's CliRunner."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from arena.cli import cli
from arena.tracker import CostBreakdown, MethodResult, StageTimings, TokenUsage
from tests.arena.conftest import make_method_result


def _runner() -> CliRunner:
    return CliRunner()


def _stub_results(method_id: str = "ppr") -> list[MethodResult]:
    return [make_method_result(method_id)]


def _patch_execute(results: list[MethodResult] | None = None):
    """Patch execute_methods to return stub results without real API calls."""
    effective = results if results is not None else _stub_results()

    async def _fake_execute(*args, **kwargs):
        return effective

    return patch("arena.cli.execute_methods", side_effect=_fake_execute)


def _patch_run_fn(method_id: str = "ppr", result: MethodResult | None = None):
    """Patch METHOD_REGISTRY runner for a specific method."""
    r = result or make_method_result(method_id)

    async def _fake_run(*args, **kwargs):
        return r

    return patch(f"arena.runner.METHOD_REGISTRY", {method_id: _fake_run})


def test_run_command_valid_invocation(stub_env) -> None:
    runner = _runner()
    with _patch_execute():
        result = runner.invoke(
            cli,
            ["run", "--query", "Who led Apollo 11?", "--group-id", "testgraph"],
        )
    assert result.exit_code == 0, result.output


def test_run_command_invalid_method(stub_env) -> None:
    runner = _runner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--query",
            "test",
            "--group-id",
            "testgraph",
            "--methods",
            "unknown-method",
        ],
    )
    assert result.exit_code != 0
    assert "unknown-method" in (result.output + (result.stderr or "")).lower() or \
           "Unknown" in (result.output + (result.stderr or ""))


def test_run_command_query_too_long(stub_env) -> None:
    runner = _runner()
    long_query = "x" * 2001
    result = runner.invoke(
        cli, ["run", "--query", long_query, "--group-id", "testgraph"]
    )
    assert result.exit_code != 0


def test_run_command_invalid_group_id(stub_env) -> None:
    runner = _runner()
    result = runner.invoke(
        cli, ["run", "--query", "test", "--group-id", "x;DROP TABLE"]
    )
    assert result.exit_code != 0


def test_run_command_output_json(stub_env) -> None:
    runner = _runner()
    with _patch_execute():
        result = runner.invoke(
            cli,
            [
                "run",
                "--query",
                "test query",
                "--group-id",
                "mygraph",
                "--output",
                "json",
                "--methods",
                "ppr",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["arena_version"] == "1.0"
    assert data["query"] == "test query"


def test_run_command_out_file_created(stub_env, tmp_path: Path) -> None:
    out_file = tmp_path / "results.json"
    runner = _runner()
    with _patch_execute(), patch("arena.cli._validate_out_file", return_value=out_file):
        result = runner.invoke(
            cli,
            [
                "run",
                "--query",
                "test",
                "--group-id",
                "g1",
                "--output",
                "json",
                "--out-file",
                str(out_file),
                "--allow-absolute-path",
            ],
        )
    assert result.exit_code == 0, result.output
    assert out_file.exists()


def test_run_command_out_file_path_traversal_rejected(stub_env) -> None:
    runner = _runner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--query",
            "test",
            "--group-id",
            "g1",
            "--out-file",
            "../../etc/evil.json",
        ],
    )
    assert result.exit_code != 0


def test_run_command_out_file_bad_extension(stub_env) -> None:
    runner = _runner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--query",
            "test",
            "--group-id",
            "g1",
            "--out-file",
            "result.exe",
            "--allow-absolute-path",
        ],
    )
    assert result.exit_code != 0


def test_duel_command_requires_two_methods(stub_env) -> None:
    runner = _runner()
    # Missing --method-b
    result = runner.invoke(
        cli,
        ["duel", "--query", "test", "--group-id", "g1", "--method-a", "ppr"],
    )
    assert result.exit_code != 0


def test_compare_command_query_file_parsed(stub_env, tmp_path: Path) -> None:
    qfile = tmp_path / "queries.jsonl"
    qfile.write_text(
        json.dumps({"query": "Q1", "group_id": "g"}) + "\n"
        + json.dumps({"query": "Q2", "group_id": "g"}) + "\n"
        + json.dumps({"query": "Q3", "group_id": "g"}) + "\n"
    )
    with _patch_execute(_stub_results()):
        runner = _runner()
        result = runner.invoke(
            cli,
            [
                "compare",
                "--query-file",
                str(qfile),
                "--methods",
                "ppr",
            ],
        )
    assert result.exit_code == 0, result.output


def test_compare_command_max_queries_enforced(stub_env, tmp_path: Path) -> None:
    qfile = tmp_path / "big.jsonl"
    lines = [json.dumps({"query": f"Q{i}", "group_id": "g"}) for i in range(1001)]
    qfile.write_text("\n".join(lines) + "\n")
    runner = _runner()
    result = runner.invoke(
        cli, ["compare", "--query-file", str(qfile), "--methods", "ppr"]
    )
    assert result.exit_code != 0


def test_compare_command_parallel_cap(stub_env, tmp_path: Path) -> None:
    """--parallel 100 is silently capped at 16; no error."""
    qfile = tmp_path / "queries.jsonl"
    qfile.write_text(json.dumps({"query": "Q1", "group_id": "g"}) + "\n")
    with _patch_execute():
        runner = _runner()
        result = runner.invoke(
            cli,
            ["compare", "--query-file", str(qfile), "--methods", "ppr", "--parallel", "100"],
        )
    assert result.exit_code == 0, result.output


def test_dry_run_no_api_calls(stub_env) -> None:
    """--dry-run prints estimate and makes zero LLM/embed calls."""
    call_count = {"n": 0}
    original_execute = __import__("arena.runner", fromlist=["execute_methods"]).execute_methods

    async def _counting_execute(*args, **kwargs):
        call_count["n"] += 1
        return await original_execute(*args, **kwargs)

    runner = _runner()
    with patch("arena.cli.execute_methods", side_effect=_counting_execute):
        result = runner.invoke(
            cli,
            [
                "run",
                "--query",
                "test",
                "--group-id",
                "g1",
                "--dry-run",
                "--methods",
                "ppr",
            ],
        )
    assert result.exit_code == 0
    assert call_count["n"] == 0  # no API calls made
    assert "dry run" in result.output.lower() or "DRY RUN" in result.output


def test_max_cost_cap_aborts_run(stub_env) -> None:
    """--max-cost-usd 0.0 aborts after first expensive method with a warning."""
    expensive = make_method_result("ppr", llm_cost=1.0, embed_cost=0.5)
    call_count = {"n": 0}

    async def _fake_run(*args, **kwargs):
        call_count["n"] += 1
        return expensive

    runner = _runner()
    with patch.dict("arena.runner.METHOD_REGISTRY", {"ppr": _fake_run, "dag": _fake_run}):
        with patch("arena.cli._create_clients", return_value=(
            __import__("arena.runner", fromlist=["StubGraphDriver"]).StubGraphDriver(),
            __import__("arena.runner", fromlist=["StubEmbedder"]).StubEmbedder(),
            __import__("arena.runner", fromlist=["StubLLMClient"]).StubLLMClient(),
        )):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "--query",
                    "test",
                    "--group-id",
                    "g1",
                    "--methods",
                    "ppr,dag",
                    "--max-cost-usd",
                    "0.5",
                ],
            )
    # Should exit 0 (cost cap is a warning, not an error)
    assert result.exit_code == 0
    # Only one method should have run (cap exceeded after first)
    assert call_count["n"] == 1


def test_as_of_invalid_format_rejected(stub_env) -> None:
    runner = _runner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--query",
            "test",
            "--group-id",
            "g1",
            "--as-of",
            "not-a-date",
            "--temporal-mode",
            "as_of",
        ],
    )
    assert result.exit_code != 0
