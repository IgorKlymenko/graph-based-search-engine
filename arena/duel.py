"""Side-by-side two-method comparison with fact diff."""
from __future__ import annotations

from io import StringIO

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arena.display import hop_detail
from arena.tracker import MethodResult


def fact_diff(
    a: MethodResult, b: MethodResult
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return (unique_to_a, unique_to_b, shared) UUID sets."""
    uuids_a = a.fact_uuids
    uuids_b = b.fact_uuids
    shared = uuids_a & uuids_b
    unique_a = uuids_a - shared
    unique_b = uuids_b - shared
    return unique_a, unique_b, shared


def render(
    a: MethodResult,
    b: MethodResult,
    output: str = "terminal",
    console: Console | None = None,
    verbose: bool = False,
) -> str | None:
    """Render duel view. Returns str for markdown; prints to console for terminal."""
    if output == "markdown":
        return _duel_markdown(a, b)

    _console = console or Console()
    unique_a, unique_b, shared = fact_diff(a, b)

    def _panel(r: MethodResult, unique: frozenset[str]) -> str:
        buf: list[str] = []
        buf.append(
            f"[bold]{r.method_id}[/bold]  ·  {len(r.hops)} hops  ·  "
            f"{r.timing.total_ms / 1000:.2f}s\n"
            f"{len(r.facts)} facts  ·  ${r.cost.total_usd:.6f}"
        )
        buf.append("")
        for i, hop in enumerate(r.hops):
            buf.append(f"[bold]HOP {i + 1}:[/bold] {hop.query[:60]}")
            buf.append(f"  Nodes: {hop.nodes_loaded:,}  Edges: {hop.edges_loaded:,}")
            buf.append(f"  {hop.facts_after_temporal} facts ({len(hop.new_fact_uuids)} new)")
        buf.append("")
        buf.append("[bold]UNIQUE:[/bold]")
        fact_by_uuid = {f.uuid: f for f in r.facts}
        for uuid in sorted(unique)[:10]:
            f = fact_by_uuid.get(uuid)
            if f:
                score = f.rerank_score if f.rerank_score is not None else f.score
                buf.append(f"  [{score:.2f}] {f.fact[:60]}")
        return "\n".join(buf)

    panel_a = Panel(_panel(a, unique_a), title=a.method_id, border_style="blue")
    panel_b = Panel(_panel(b, unique_b), title=b.method_id, border_style="green")
    _console.print(Columns([panel_a, panel_b]))

    if shared:
        _console.print(f"\n[bold]SHARED FACTS ({len(shared)}):[/bold]")
        fact_by_uuid = {f.uuid: f for f in a.facts + b.facts}
        for uuid in sorted(shared)[:5]:
            f = fact_by_uuid.get(uuid)
            if f:
                _console.print(f"  · {f.fact[:80]}")

    _token_breakdown(_console, a, b)
    _verdict_row(_console, a, b)

    if verbose:
        for r in (a, b):
            _console.print(f"\n[bold cyan]── {r.method_id} (verbose) ──[/bold cyan]")
            hop_detail(r, _console)

    return None


def _token_breakdown(console: Console, a: MethodResult, b: MethodResult) -> None:
    table = Table(title="Token Breakdown", show_header=True)
    table.add_column("Method")
    table.add_column("Tokens In", justify="right")
    table.add_column("Tokens Out", justify="right")
    table.add_column("LLM Cost", justify="right")
    table.add_column("Embed Cost", justify="right")
    for r in (a, b):
        table.add_row(
            r.method_id,
            f"{r.token_usage.tokens_in:,}",
            f"{r.token_usage.tokens_out:,}",
            f"${r.cost.llm_cost_usd:.6f}",
            f"${r.cost.embed_cost_usd:.6f}",
        )
    console.print(table)


def _verdict_row(console: Console, a: MethodResult, b: MethodResult) -> None:
    faster = a.method_id if a.timing.total_ms <= b.timing.total_ms else b.method_id
    cheaper = a.method_id if a.cost.total_usd <= b.cost.total_usd else b.method_id
    more_facts = a.method_id if len(a.facts) >= len(b.facts) else b.method_id
    console.print(
        f"\n[bold]Verdict:[/bold] "
        f"Faster={faster}  Cheaper={cheaper}  MoreFacts={more_facts}"
    )


def _duel_markdown(a: MethodResult, b: MethodResult) -> str:
    unique_a, unique_b, shared = fact_diff(a, b)
    lines: list[str] = [
        f"## Duel: {a.method_id} vs {b.method_id}",
        "",
        f"| | {a.method_id} | {b.method_id} |",
        "|---|---|---|",
        f"| Hops | {len(a.hops)} | {len(b.hops)} |",
        f"| Facts | {len(a.facts)} | {len(b.facts)} |",
        f"| Time | {a.timing.total_ms:.1f}ms | {b.timing.total_ms:.1f}ms |",
        f"| Cost | ${a.cost.total_usd:.6f} | ${b.cost.total_usd:.6f} |",
        f"| Unique facts | {len(unique_a)} | {len(unique_b)} |",
        f"| Shared facts | {len(shared)} | {len(shared)} |",
    ]
    return "\n".join(lines)
