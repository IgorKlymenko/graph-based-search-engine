"""Rich-based terminal and Markdown rendering for arena results."""
from __future__ import annotations

import re
from io import StringIO
from typing import Sequence

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arena.tracker import HopTrace, MethodResult


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def _fmt_cost(usd: float) -> str:
    return f"${usd:.6f}".rstrip("0").rstrip("$") if usd else "$0.000000"


def _cheapest_idx(results: Sequence[MethodResult]) -> int:
    return min(range(len(results)), key=lambda i: results[i].cost.total_usd)


def _fastest_idx(results: Sequence[MethodResult]) -> int:
    return min(range(len(results)), key=lambda i: results[i].timing.total_ms)


def _most_facts_idx(results: Sequence[MethodResult]) -> int:
    return max(range(len(results)), key=lambda i: len(results[i].facts))


def summary_table(
    results: Sequence[MethodResult],
    query: str = "",
    group_id: str = "",
    temporal_mode: str = "",
    console: Console | None = None,
    output: str = "terminal",
) -> str | None:
    """Render a summary table. Returns str for markdown/json; prints to console for terminal."""
    if output == "markdown":
        return _summary_markdown(results, query, group_id, temporal_mode)
    if output == "json":
        return None  # json handled by report.py

    _console = console or Console()

    if query:
        header = f'Arena Run  ·  Query: "{query[:60]}{"…" if len(query) > 60 else ""}"'
        sub = f"Group: {group_id}  ·  Temporal: {temporal_mode}"
        _console.print(Panel(f"[bold]{header}[/bold]\n{sub}"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Method", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Tokens In", justify="right")
    table.add_column("Tokens Out", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Facts", justify="right")
    table.add_column("Hops", justify="right")
    table.add_column("Stop")

    cheap_i = _cheapest_idx(results)
    fast_i = _fastest_idx(results)
    facts_i = _most_facts_idx(results)

    for i, r in enumerate(results):
        style = ""
        if i == cheap_i:
            style = "green"
        elif i == fast_i:
            style = "blue"
        elif i == facts_i:
            style = "yellow"

        table.add_row(
            r.method_id,
            _fmt_ms(r.timing.total_ms),
            f"{r.token_usage.tokens_in:,}",
            f"{r.token_usage.tokens_out:,}",
            f"${r.cost.total_usd:.6f}",
            str(len(r.facts)),
            str(len(r.hops)),
            r.stop_reason,
            style=style,
        )

    _console.print(table)

    if results:
        _console.print(
            f"[green]Cheapest:[/green]   {results[cheap_i].method_id}"
            f"  (${results[cheap_i].cost.total_usd:.6f})\n"
            f"[blue]Fastest:[/blue]    {results[fast_i].method_id}"
            f"  ({_fmt_ms(results[fast_i].timing.total_ms)})\n"
            f"[yellow]Most facts:[/yellow] {results[facts_i].method_id}"
            f"  ({len(results[facts_i].facts)})"
        )
    return None


def _summary_markdown(
    results: Sequence[MethodResult],
    query: str,
    group_id: str,
    temporal_mode: str,
) -> str:
    lines: list[str] = []
    if query:
        lines.append("## Arena Run")
        lines.append(f"**Query:** {query}")
        lines.append(f"**Group:** {group_id}  **Temporal:** {temporal_mode}")
        lines.append("")
    lines.append(
        "| Method | Time | Tokens In | Tokens Out | Cost | Facts | Hops | Stop |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.method_id} | {_fmt_ms(r.timing.total_ms)} | "
            f"{r.token_usage.tokens_in:,} | {r.token_usage.tokens_out:,} | "
            f"${r.cost.total_usd:.6f} | {len(r.facts)} | {len(r.hops)} | {r.stop_reason} |"
        )
    answered = [r for r in results if r.generated_answer is not None]
    if answered:
        lines.append("")
        lines.append("## Generated Answers")
        for r in answered:
            lines.append(f"\n### {r.method_id} ({len(r.facts)} facts)")
            lines.append(r.generated_answer or "")
    return "\n".join(lines)


def hop_detail(
    result: MethodResult,
    console: Console | None = None,
    all_prev_uuids: frozenset[str] | None = None,
) -> None:
    """Print verbose hop-by-hop detail for a single MethodResult."""
    _console = console or Console()
    seen: frozenset[str] = all_prev_uuids or frozenset()

    for hop in result.hops:
        header = (
            f"HOP {hop.hop_index + 1}  ·  {hop.query!r}"
            f"  ·  {_fmt_ms(hop.timing_ms)}"
            f"  ·  {hop.token_usage.tokens_in}in/{hop.token_usage.tokens_out}out tokens"
        )
        _console.print(f"\n[bold]{header}[/bold]")
        _console.print(
            f"  Nodes/Edges: {hop.nodes_loaded:,} / {hop.edges_loaded:,}"
        )
        _console.print(
            f"  Temporal:    {hop.facts_before_temporal} → {hop.facts_after_temporal}"
            f" (dropped {hop.dropped_by_temporal})"
        )
        _console.print(
            f"  NEW facts: {len(hop.new_fact_uuids)}  "
            f"DUP facts: {hop.facts_after_temporal - len(hop.new_fact_uuids)}"
        )
        if hop.ppr_seeds:
            sample = ", ".join(s[:8] for s in hop.ppr_seeds[:5])
            tail = f" +{len(hop.ppr_seeds) - 5} more" if len(hop.ppr_seeds) > 5 else ""
            _console.print(f"  Seeds ({len(hop.ppr_seeds)}): {sample}{tail}")
        if hop.expansion_terms:
            _console.print(f"  Expansion: {', '.join(hop.expansion_terms)}")
        if hop.hyde_answers:
            for h in hop.hyde_answers:
                _console.print(f"  HyDE: {h[:100]}")
        if hop.bridge_queries:
            for bq in hop.bridge_queries:
                _console.print(f"  Bridge: {bq!r}")

        for fact in _facts_for_hop(result, hop):
            tag = "[NEW]" if fact.uuid in hop.new_fact_uuids else "[DUP]"
            score = fact.rerank_score if fact.rerank_score is not None else fact.score
            _console.print(f"  {tag} [{score:.2f}] {fact.fact[:120]}")
        seen = seen | hop.new_fact_uuids


def _facts_for_hop(result: MethodResult, hop: HopTrace) -> list:
    """Return the facts retrieved in this hop, ordered by score."""
    hop_uuids = hop.all_fact_uuids
    facts = [f for f in result.facts if f.uuid in hop_uuids]
    return sorted(
        facts,
        key=lambda f: f.rerank_score if f.rerank_score is not None else f.score,
        reverse=True,
    )


def render_generated_answers(
    results: Sequence[MethodResult],
    console: Console | None = None,
) -> None:
    """Print generated answers as labelled panels, one per method."""
    _console = console or Console()
    answered = [r for r in results if r.generated_answer is not None]
    if not answered:
        return
    _console.print()
    for r in answered:
        label = f"[bold cyan]{r.method_id}[/bold cyan]  [dim]({len(r.facts)} facts)[/dim]"
        _console.print(Panel(r.generated_answer or "", title=label, title_align="left"))


def render_terminal(
    results: Sequence[MethodResult],
    query: str = "",
    group_id: str = "",
    temporal_mode: str = "",
    verbose: bool = False,
    redact: bool = False,
    console: Console | None = None,
) -> None:
    """Print full terminal output for a run."""
    _console = console or Console()
    effective = [_maybe_redact(r, redact) for r in results]
    summary_table(effective, query, group_id, temporal_mode, _console, output="terminal")
    render_generated_answers(effective, _console)
    if verbose:
        for r in effective:
            _console.print(f"\n[bold cyan]── {r.method_id} ──[/bold cyan]")
            hop_detail(r, _console)


def render_markdown(
    results: Sequence[MethodResult],
    query: str = "",
    group_id: str = "",
    temporal_mode: str = "",
    redact: bool = False,
) -> str:
    """Return Markdown string without any Rich markup or ANSI escapes."""
    effective = [_maybe_redact(r, redact) for r in results]
    return _summary_markdown(effective, query, group_id, temporal_mode)


def _maybe_redact(result: MethodResult, redact: bool) -> MethodResult:
    if not redact:
        return result
    from dataclasses import replace

    redacted_facts = [
        f.model_copy(update={"fact": f"[REDACTED-{f.uuid}]"}) for f in result.facts
    ]
    return MethodResult(
        method_id=result.method_id,
        query=result.query,
        group_id=result.group_id,
        facts=redacted_facts,
        hops=result.hops,
        stop_reason=result.stop_reason,
        token_usage=result.token_usage,
        cost=result.cost,
        timing=result.timing,
        error=result.error,
        generated_answer=result.generated_answer,
    )


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags and ANSI escape codes from a string."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    text = ansi_escape.sub("", text)
    rich_tag = re.compile(r"\[/?[a-zA-Z0-9_ #]+\]")
    return rich_tag.sub("", text)
