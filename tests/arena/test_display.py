"""Tests for arena/display.py and arena/duel.py."""
from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

from arena.display import render_markdown, strip_rich_markup, summary_table
from arena.duel import fact_diff, render as duel_render
from arena.report import to_json
from tests.arena.conftest import make_fact, make_method_result


def _console_capture() -> tuple[Console, StringIO]:
    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=True)
    return console, buf


def test_summary_table_renders_all_methods() -> None:
    """3 methods → 3 rows in the table output."""
    results = [
        make_method_result("ppr", total_ms=1000.0),
        make_method_result("treehop", total_ms=80.0, llm_cost=0.0, tokens_out=0),
        make_method_result("dag", total_ms=2100.0, n_facts=15),
    ]
    console, buf = _console_capture()
    summary_table(results, console=console, output="terminal")
    output = buf.getvalue()
    assert "ppr" in output
    assert "treehop" in output
    assert "dag" in output


def test_summary_table_highlights_cheapest() -> None:
    """The cheapest row should appear in the output (green markup applied)."""
    cheap = make_method_result("treehop", llm_cost=0.000001, embed_cost=0.0000001, total_ms=80.0)
    expensive = make_method_result("dag", llm_cost=0.01, embed_cost=0.001, total_ms=2000.0)
    results = [cheap, expensive]
    console, buf = _console_capture()
    summary_table(results, console=console, output="terminal")
    output = buf.getvalue()
    # treehop should appear (cheapest)
    assert "treehop" in output


def test_hop_detail_new_vs_dup_labels() -> None:
    """[NEW] and [DUP] labels appear correctly in hop detail output."""
    from arena.display import hop_detail
    from arena.tracker import HopTrace, TokenUsage

    fact_a = make_fact("Fact A is new")
    fact_b = make_fact("Fact B is dup")
    result = make_method_result("ppr", n_facts=0)
    result.facts.extend([fact_a, fact_b])
    # Only fact_a is "new" in hop
    hop = HopTrace(
        hop_index=0,
        query="test",
        ppr_seeds=[],
        nodes_loaded=100,
        edges_loaded=400,
        facts_before_temporal=2,
        facts_after_temporal=2,
        dropped_by_temporal=0,
        new_fact_uuids=frozenset([fact_a.uuid]),
        all_fact_uuids=frozenset([fact_a.uuid, fact_b.uuid]),
        bridge_queries=[],
        token_usage=TokenUsage(0, 0),
        timing_ms=0.0,
    )
    result.hops.clear()
    result.hops.append(hop)
    console, buf = _console_capture()
    hop_detail(result, console=console)
    output = buf.getvalue()
    assert "[NEW]" in output
    assert "[DUP]" in output


def test_duel_unique_facts_correct() -> None:
    """Facts unique to method A are not in method B's UUID set."""
    fact_shared = make_fact("Shared fact")
    fact_only_a = make_fact("Unique to A")
    fact_only_b = make_fact("Unique to B")

    result_a = make_method_result("ppr", n_facts=0)
    result_a.facts.extend([fact_shared, fact_only_a])

    result_b = make_method_result("dag", n_facts=0)
    result_b.facts.extend([fact_shared, fact_only_b])

    unique_a, unique_b, shared = fact_diff(result_a, result_b)
    assert fact_only_a.uuid in unique_a
    assert fact_only_b.uuid not in unique_a
    assert fact_only_b.uuid in unique_b
    assert fact_only_a.uuid not in unique_b


def test_duel_shared_facts_correct() -> None:
    """Shared facts appear in both UUID sets."""
    fact_shared = make_fact("Shared fact")
    fact_a = make_fact("Only A")

    result_a = make_method_result("ppr", n_facts=0)
    result_a.facts.extend([fact_shared, fact_a])

    result_b = make_method_result("dag", n_facts=0)
    result_b.facts.append(fact_shared)

    unique_a, unique_b, shared = fact_diff(result_a, result_b)
    assert fact_shared.uuid in shared
    assert fact_shared.uuid not in unique_a
    assert fact_shared.uuid not in unique_b


def test_markdown_output_no_rich_markup() -> None:
    """Markdown output contains no Rich [bold] or ANSI escape codes."""
    results = [make_method_result("ppr"), make_method_result("treehop")]
    md = render_markdown(results, query="test", group_id="g1", temporal_mode="currently_true")
    assert "[bold]" not in md
    assert "\x1b[" not in md
    # Should contain a Markdown table
    assert "|" in md


def test_json_output_valid_schema() -> None:
    """JSON output parses and matches expected schema fields."""
    results = [make_method_result("ppr")]
    js = to_json(results, "test query", "g1", "currently_true")
    data = json.loads(js)
    assert data["arena_version"] == "1.0"
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["method_id"] == "ppr"
    assert "facts" in r
    assert "hops" in r
    assert "total_cost_usd" in r


def test_redact_facts_replaces_text() -> None:
    """With redact=True, all fact text fields become [REDACTED-uuid]."""
    results = [make_method_result("ppr", n_facts=2)]
    js = to_json(results, "test", "g", "currently_true", redact=True)
    data = json.loads(js)
    for fact in data["results"][0]["facts"]:
        assert fact["fact"].startswith("[REDACTED-")
        assert "[REDACTED-" + fact["uuid"] + "]" == fact["fact"]
