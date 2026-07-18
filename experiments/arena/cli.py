"""Arena CLI — compare retrieval methods side-by-side."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from arena import report as _report
from arena import runner as _runner
from arena.display import render_markdown, render_terminal, strip_rich_markup
from arena.duel import fact_diff, render as duel_render
from arena.pricing import load_pricing
from arena.runner import (
    ArenaArgs,
    METHOD_REGISTRY,
    StubEmbedder,
    StubGraphDriver,
    StubLLMClient,
    execute_methods,
    generate_answer,
)
from arena.tracker import MethodResult, TokenUsage

log = logging.getLogger("arena")

_GROUP_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_MAX_QUERY_LEN = 2000
_MAX_QUERIES_PER_FILE = 1000
_MAX_PARALLEL = 16
_DEFAULT_MAX_COST_USD = 1.0
_DEFAULT_MAX_TOTAL_COST_USD = 10.0


def _validate_query(query: str) -> None:
    if len(query) > _MAX_QUERY_LEN:
        raise click.UsageError(f"query exceeds {_MAX_QUERY_LEN} characters")


def _validate_group_id(group_id: str) -> None:
    if not _GROUP_ID_RE.match(group_id):
        raise click.UsageError(
            "group-id must match ^[A-Za-z0-9_-]+$, got: " + repr(group_id)
        )


def _validate_methods(method_strs: str) -> list[str]:
    if method_strs.lower() == "all":
        return list(METHOD_REGISTRY.keys())
    methods = [m.strip() for m in method_strs.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        raise click.UsageError(
            f"Unknown method(s): {unknown}. Choices: {list(METHOD_REGISTRY.keys())}"
        )
    return methods


def _parse_as_of(as_of_str: str | None) -> datetime | None:
    if as_of_str is None:
        return None
    try:
        dt = datetime.fromisoformat(as_of_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except ValueError:
        raise click.UsageError(
            f"--as-of must be ISO 8601 datetime, got: {as_of_str!r}"
        )


def _validate_out_file(
    out_file: str | None, allow_absolute: bool
) -> Path | None:
    if out_file is None:
        return None
    p = Path(out_file).resolve()
    if not allow_absolute:
        try:
            p.relative_to(Path.cwd())
        except ValueError:
            raise click.UsageError(
                f"--out-file resolves outside cwd ({p}). "
                "Use --allow-absolute-path to override."
            )
    if p.suffix not in {".json", ".jsonl", ".md"}:
        raise click.UsageError(
            f"--out-file must have .json, .jsonl, or .md extension, got: {p.suffix!r}"
        )
    # Reject paths inside .git or hidden directories
    for part in p.parts:
        if part.startswith(".git"):
            raise click.UsageError("--out-file may not be inside a .git directory")
    return p


def _write_output_file(path: Path, content: str) -> None:
    log.warning("Output file will contain graph fact content: %s", path)
    # Atomic write via temp file
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".arena_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _create_clients(embed_model: str = "text-embedding-3-small"):
    """Create retrieval clients. Uses stubs when ARENA_USE_STUBS=1."""
    if os.environ.get("ARENA_USE_STUBS"):
        return StubGraphDriver(), StubEmbedder(), StubLLMClient()
    from services.neo4j import Neo4jGraphDriver
    from services.openai import OpenAIEmbedder, OpenAILLMClient

    try:
        driver = Neo4jGraphDriver.from_env()
    except RuntimeError as exc:
        raise click.UsageError(str(exc)) from exc
    return driver, OpenAIEmbedder(embed_model), OpenAILLMClient()


def _dry_run_estimate(
    query: str, methods: list[str], max_hops: int, llm_model: str
) -> None:
    """Print a cost estimate without making any API calls."""
    words = len(query.split())
    est_tokens_per_hop = int(words * 1.5 + 100)
    console = Console()
    console.print("[bold yellow]DRY RUN — No API calls will be made[/bold yellow]")
    console.print(f"Query: {query[:80]}")
    console.print(f"Methods: {', '.join(methods)}")
    console.print(f"Max hops: {max_hops}")
    console.print("")
    from arena.pricing import compute_cost, DEFAULT_PRICING
    from arena.tracker import TokenUsage

    pricing = load_pricing()
    for method in methods:
        hops = max_hops
        has_llm = method not in ("treehop",)
        tokens_in = est_tokens_per_hop * hops * (3 if has_llm else 1)
        tokens_out = int(tokens_in * 0.2) if has_llm else 0
        usage = TokenUsage(tokens_in=tokens_in, tokens_out=tokens_out)
        cost = compute_cost(usage, llm_model, pricing)
        console.print(
            f"  {method:<20} ~{tokens_in:,} in / {tokens_out:,} out  ~${cost:.6f}"
        )


def _check_cost_cap(
    results: list[MethodResult],
    max_cost: float,
    console: Console,
) -> bool:
    """Returns True if cost cap exceeded."""
    total = sum(r.cost.total_usd for r in results)
    if total >= max_cost:
        console.print(
            f"[yellow]WARNING: cumulative cost ${total:.6f} reached cap ${max_cost:.6f}. "
            "Aborting remaining methods.[/yellow]"
        )
        return True
    return False


@click.group()
@click.option("--verbose", is_flag=True, default=False, help="Enable DEBUG logging")
def cli(verbose: bool) -> None:
    """Arena — compare knowledge-graph retrieval methods."""
    _configure_logging(verbose)


@cli.command("run")
@click.option("--query", required=True, help="Search query")
@click.option("--group-id", required=True, help="Graph group identifier")
@click.option("--methods", default="all", show_default=True, help="Comma-separated method IDs or 'all'")
@click.option(
    "--output",
    type=click.Choice(["terminal", "json", "markdown"]),
    default="terminal",
    show_default=True,
)
@click.option("--out-file", default=None, help="Write output to file instead of stdout")
@click.option("--allow-absolute-path", is_flag=True, default=False, hidden=True)
@click.option("--temporal-mode", default="currently_true", show_default=True)
@click.option("--as-of", default=None, help="ISO 8601 datetime for as_of temporal mode")
@click.option("--max-hops", default=4, show_default=True, type=int)
@click.option("--beam-width", default=3, show_default=True, type=int)
@click.option("--llm-model", default="gpt-4o-mini", show_default=True)
@click.option("--embed-model", default="text-embedding-3-small", show_default=True)
@click.option("--no-hydrate", is_flag=True, default=False)
@click.option("--redact-facts", is_flag=True, default=False)
@click.option(
    "--fast-ppr-push-threshold",
    default=1e-5,
    show_default=True,
    type=float,
    help="Epsilon for fast-ppr forward push (lower = more accurate, slower)",
)
@click.option("--no-generate", is_flag=True, default=False, help="Skip answer generation")
@click.option("--dry-run", is_flag=True, default=False, help="Estimate cost without API calls")
@click.option("--max-cost-usd", default=_DEFAULT_MAX_COST_USD, show_default=True, type=float)
@click.option("--verbose", is_flag=True, default=False)
def run_cmd(
    query: str,
    group_id: str,
    methods: str,
    output: str,
    out_file: Optional[str],
    allow_absolute_path: bool,
    temporal_mode: str,
    as_of: Optional[str],
    max_hops: int,
    beam_width: int,
    llm_model: str,
    embed_model: str,
    no_hydrate: bool,
    redact_facts: bool,
    fast_ppr_push_threshold: float,
    no_generate: bool,
    dry_run: bool,
    max_cost_usd: float,
    verbose: bool,
) -> None:
    """Run one or more methods on a single query."""
    # Validate all inputs before any network call
    _validate_query(query)
    _validate_group_id(group_id)
    method_ids = _validate_methods(methods)
    as_of_dt = _parse_as_of(as_of)
    out_path = _validate_out_file(out_file, allow_absolute_path)

    if dry_run:
        _dry_run_estimate(query, method_ids, max_hops, llm_model)
        return

    args = ArenaArgs(
        temporal_mode=temporal_mode,
        as_of=as_of_dt,
        max_hops=max_hops,
        beam_width=beam_width,
        llm_model=llm_model,
        embed_model=embed_model,
        hydrate=not no_hydrate,
        redact_facts=redact_facts,
        verbose=verbose,
        generate=not no_generate,
        fast_ppr_push_threshold=fast_ppr_push_threshold,
    )

    driver, embedder, llm_client = _create_clients(embed_model)
    console = Console()

    results: list[MethodResult] = []
    accumulated_cost = 0.0

    async def _run_all() -> None:
        nonlocal accumulated_cost
        for method_id in method_ids:
            from arena.tracker import TokenTracker

            tracker = TokenTracker()
            run_fn = METHOD_REGISTRY[method_id]
            result = await run_fn(driver, embedder, llm_client, query, group_id, args, tracker)
            if args.generate and not result.error:
                from arena.runner import _add_generation_cost

                result.generated_answer = await generate_answer(
                    query, result.facts, llm_client, tracker, args.llm_model, result.timing
                )
                _add_generation_cost(result, tracker, args)
            results.append(result)
            accumulated_cost += result.cost.total_usd
            if _check_cost_cap(results, max_cost_usd, console):
                break

    asyncio.run(_run_all())

    _emit_output(
        results, output, out_path, query, group_id, temporal_mode, as_of_dt, redact_facts, verbose, console
    )


@cli.command("duel")
@click.option("--query", required=True)
@click.option("--group-id", required=True)
@click.option("--method-a", required=True)
@click.option("--method-b", required=True)
@click.option("--output", type=click.Choice(["terminal", "json", "markdown"]), default="terminal")
@click.option("--out-file", default=None)
@click.option("--allow-absolute-path", is_flag=True, default=False, hidden=True)
@click.option("--temporal-mode", default="currently_true")
@click.option("--as-of", default=None)
@click.option("--max-hops", default=4, type=int)
@click.option("--beam-width", default=3, type=int)
@click.option("--llm-model", default="gpt-4o-mini")
@click.option("--embed-model", default="text-embedding-3-small")
@click.option("--no-hydrate", is_flag=True, default=False)
@click.option("--redact-facts", is_flag=True, default=False)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--max-cost-usd", default=_DEFAULT_MAX_COST_USD, type=float)
@click.option("--verbose", is_flag=True, default=False)
def duel_cmd(
    query: str,
    group_id: str,
    method_a: str,
    method_b: str,
    output: str,
    out_file: Optional[str],
    allow_absolute_path: bool,
    temporal_mode: str,
    as_of: Optional[str],
    max_hops: int,
    beam_width: int,
    llm_model: str,
    embed_model: str,
    no_hydrate: bool,
    redact_facts: bool,
    dry_run: bool,
    max_cost_usd: float,
    verbose: bool,
) -> None:
    """Run exactly two methods and compare results side-by-side."""
    _validate_query(query)
    _validate_group_id(group_id)
    _validate_methods(method_a)
    _validate_methods(method_b)
    as_of_dt = _parse_as_of(as_of)
    out_path = _validate_out_file(out_file, allow_absolute_path)

    if dry_run:
        _dry_run_estimate(query, [method_a, method_b], max_hops, llm_model)
        return

    args = ArenaArgs(
        temporal_mode=temporal_mode,
        as_of=as_of_dt,
        max_hops=max_hops,
        beam_width=beam_width,
        llm_model=llm_model,
        embed_model=embed_model,
        hydrate=not no_hydrate,
        redact_facts=redact_facts,
    )

    driver, embedder, llm_client = _create_clients(embed_model)
    console = Console()

    async def _run() -> list[MethodResult]:
        return await execute_methods(
            [method_a, method_b], driver, embedder, llm_client, query, group_id, args
        )

    results = asyncio.run(_run())
    result_a, result_b = results[0], results[1]

    if output == "json":
        content = _report.to_json(results, query, group_id, temporal_mode, as_of_dt, redact_facts)
        _emit_content(content, out_path)
    elif output == "markdown":
        content = duel_render(result_a, result_b, output="markdown") or ""
        _emit_content(content, out_path)
    else:
        duel_render(result_a, result_b, output="terminal", console=console, verbose=verbose)


@cli.command("compare")
@click.option("--query-file", required=True, help="JSONL file: one {query, group_id} per line")
@click.option("--methods", default="all")
@click.option("--out-file", default=None)
@click.option("--allow-absolute-path", is_flag=True, default=False, hidden=True)
@click.option("--parallel", default=1, type=int, show_default=True)
@click.option("--temporal-mode", default="currently_true")
@click.option("--as-of", default=None)
@click.option("--max-hops", default=4, type=int)
@click.option("--beam-width", default=3, type=int)
@click.option("--llm-model", default="gpt-4o-mini")
@click.option("--embed-model", default="text-embedding-3-small")
@click.option("--no-hydrate", is_flag=True, default=False)
@click.option("--redact-facts", is_flag=True, default=False)
@click.option("--max-total-cost-usd", default=_DEFAULT_MAX_TOTAL_COST_USD, type=float)
@click.option("--dry-run", is_flag=True, default=False)
def compare_cmd(
    query_file: str,
    methods: str,
    out_file: Optional[str],
    allow_absolute_path: bool,
    parallel: int,
    temporal_mode: str,
    as_of: Optional[str],
    max_hops: int,
    beam_width: int,
    llm_model: str,
    embed_model: str,
    no_hydrate: bool,
    redact_facts: bool,
    max_total_cost_usd: float,
    dry_run: bool,
) -> None:
    """Batch comparison from a JSONL query file."""
    method_ids = _validate_methods(methods)
    as_of_dt = _parse_as_of(as_of)
    out_path = _validate_out_file(out_file, allow_absolute_path)
    parallel_capped = min(parallel, _MAX_PARALLEL)

    # Parse query file
    qfile = Path(query_file)
    if not qfile.exists():
        raise click.UsageError(f"query-file not found: {query_file}")

    queries: list[tuple[str, str]] = []
    for lineno, line in enumerate(qfile.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            q = str(obj["query"])
            g = str(obj["group_id"])
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Line %d skipped: %s", lineno, exc)
            continue
        try:
            _validate_query(q)
            _validate_group_id(g)
        except click.UsageError as exc:
            log.warning("Line %d skipped: %s", lineno, exc)
            continue
        queries.append((q, g))
        if len(queries) >= _MAX_QUERIES_PER_FILE:
            break

    # Check if file had more lines than the cap
    total_lines = sum(1 for ln in qfile.read_text().splitlines() if ln.strip())
    if total_lines > _MAX_QUERIES_PER_FILE:
        raise click.UsageError(
            f"query-file has {total_lines} queries; cap is {_MAX_QUERIES_PER_FILE}"
        )

    if not queries:
        raise click.UsageError("No valid queries found in query-file")

    if dry_run:
        click.echo(f"Would run {len(queries)} queries × {len(method_ids)} methods")
        return

    args = ArenaArgs(
        temporal_mode=temporal_mode,
        as_of=as_of_dt,
        max_hops=max_hops,
        beam_width=beam_width,
        llm_model=llm_model,
        embed_model=embed_model,
        hydrate=not no_hydrate,
        redact_facts=redact_facts,
    )

    driver, embedder, llm_client = _create_clients(embed_model)
    all_results: list[dict] = []
    total_cost = 0.0
    console = Console()
    sem = asyncio.Semaphore(parallel_capped)

    async def _run_query(q: str, g: str) -> list[MethodResult]:
        async with sem:
            return await execute_methods(method_ids, driver, embedder, llm_client, q, g, args)

    async def _run_all() -> None:
        nonlocal total_cost
        tasks = [_run_query(q, g) for q, g in queries]
        for coro in asyncio.as_completed(tasks):
            results = await coro
            for r in results:
                total_cost += r.cost.total_usd
            all_results.append(results)
            if total_cost >= max_total_cost_usd:
                console.print(
                    f"[yellow]WARNING: total cost ${total_cost:.6f} reached cap "
                    f"${max_total_cost_usd:.6f}. Aborting remaining queries.[/yellow]"
                )
                break

    asyncio.run(_run_all())

    # Output
    header = _report.to_jsonl_header(method_ids, len(queries))
    lines = [header]
    for query_results in all_results:
        if query_results:
            q = query_results[0].query
            g = query_results[0].group_id
            lines.append(
                _report.to_jsonl_line(query_results, q, g, temporal_mode, as_of_dt, redact_facts)
            )

    content = "\n".join(lines) + "\n"
    if out_path:
        _write_output_file(out_path, content)
    else:
        click.echo(content, nl=False)


def _emit_output(
    results: list[MethodResult],
    output: str,
    out_path: Path | None,
    query: str,
    group_id: str,
    temporal_mode: str,
    as_of_dt: datetime | None,
    redact: bool,
    verbose: bool,
    console: Console,
) -> None:
    if output == "json":
        content = _report.to_json(results, query, group_id, temporal_mode, as_of_dt, redact)
        _emit_content(content, out_path)
    elif output == "markdown":
        content = render_markdown(results, query, group_id, temporal_mode, redact)
        _emit_content(content, out_path)
    else:
        render_terminal(results, query, group_id, temporal_mode, verbose, redact, console)
        if out_path:
            md = render_markdown(results, query, group_id, temporal_mode, redact)
            _write_output_file(out_path, md)


def _emit_content(content: str, out_path: Path | None) -> None:
    if out_path:
        _write_output_file(out_path, content)
    else:
        click.echo(content)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level)
    logging.getLogger("arena").setLevel(level)
    # Neo4j warns about missing properties (e.g. `embedding`) when they're absent on
    # some edges — this is expected when embeddings aren't stored in the graph.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
