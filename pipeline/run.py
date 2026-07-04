"""Pipeline orchestrator.

Runs the six extraction stages in order and finishes by invoking
``graph_upload.py`` (always with ``--skip-facts``). Each stage's artifact is
reused when it is newer than its inputs; ``--force`` reruns from ``--from``.

    uv run python -m pipeline.run --notes-dir /Applications/reink
    uv run python -m pipeline.run --notes-dir ./notes --from relations --force
    uv run python -m pipeline.run --notes-dir ./notes --skip-upload
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from pipeline import config

STAGES = ["mentions", "embed", "resolve", "entity-vectors", "relations", "upload"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--notes-dir", type=Path, required=True,
                   help="Directory scanned recursively for .md source files")
    p.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0],
                   help="Run from this stage onward (default: mentions)")
    p.add_argument("--only", choices=STAGES, default=None,
                   help="Run exactly one stage")
    p.add_argument("--force", action="store_true",
                   help="Rerun stages from --from even if their artifacts are fresh")
    p.add_argument("--extraction-model", default=config.EXTRACTION_MODEL)
    p.add_argument("--no-gleaning", action="store_true",
                   help="Disable the second-pass gleaning call in mention extraction")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Cap input tokens of pending files processed this run")
    p.add_argument("--skip-upload", action="store_true",
                   help="Stop after relations; do not invoke graph_upload.py")
    p.add_argument("--dry-run", action="store_true",
                   help="Forward --dry-run to graph_upload.py (no Neo4j writes)")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Forward --skip-embeddings to graph_upload.py")
    p.add_argument("--saga", default=None)
    p.add_argument("--group-id", default=None)
    p.add_argument("--min-confidence", type=float, default=None)
    return p.parse_args()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _is_fresh(output: Path, inputs: list[Path]) -> bool:
    """True when ``output`` exists and is at least as new as every input."""
    if not output.exists():
        return False
    return _mtime(output) >= max((_mtime(p) for p in inputs), default=0.0)


def _load_surface_vectors(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        sys.exit(f"missing vector cache: {path} — run the 'embed' stage first")
    npz = np.load(path, allow_pickle=False)
    return {str(e): v for e, v in zip(npz["entities"], npz["matrix"])}


def _selected_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [args.only]
    return STAGES[STAGES.index(args.from_stage):]


def _run_upload(args: argparse.Namespace) -> None:
    cmd = ["uv", "run", "python", "graph_upload.py", "--skip-facts"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_embeddings:
        cmd.append("--skip-embeddings")
    if args.saga:
        cmd += ["--saga", args.saga]
    if args.group_id:
        cmd += ["--group-id", args.group_id]
    if args.min_confidence is not None:
        cmd += ["--min-confidence", str(args.min_confidence)]
    print(f"[upload] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.notes_dir.is_dir():
        sys.exit(f"--notes-dir is not a directory: {args.notes_dir}")

    stages = _selected_stages(args)
    forced = args.force or args.only is not None

    for name in stages:
        start = time.monotonic()

        if name == "mentions":
            # Incremental + resumable: safe to always run (no-op if all cached).
            from pipeline import mentions

            mentions.run(
                args.notes_dir,
                model=args.extraction_model,
                enable_gleaning=not args.no_gleaning,
                token_budget=args.token_budget,
                file_workers=config.FILE_WORKERS,
                episode_workers=config.EPISODE_WORKERS,
            )

        elif name == "embed":
            if not forced and _is_fresh(config.VECTOR_CACHE, [config.ENTITY_CACHE]):
                print("[2/6] embed: fresh — skipped")
            else:
                from pipeline import embed_mentions

                embed_mentions.run()

        elif name == "resolve":
            inputs = [config.VECTOR_CACHE, config.ENTITY_CACHE]
            if not forced and _is_fresh(config.RESOLVED_CACHE, inputs):
                print("[4/6] resolve: fresh — skipped")
            else:
                from pipeline import resolution

                vectors = _load_surface_vectors(config.VECTOR_CACHE)
                mentions_data = json.loads(config.ENTITY_CACHE.read_text())["mentions"]
                resolution.run(
                    vectors,  # type: ignore[arg-type]
                    mentions_data,
                    resolve_model=config.RESOLVE_MODEL,
                    escalate_model=config.ESCALATE_MODEL,
                    workers=config.RESOLVE_WORKERS,
                )

        elif name == "entity-vectors":
            inputs = [config.RESOLVED_CACHE, config.VECTOR_CACHE]
            if not forced and _is_fresh(config.ENTITY_VECTOR_CACHE, inputs):
                print("[5/6] entity vectors: fresh — skipped")
            else:
                from pipeline import entity_vectors

                resolved = json.loads(config.RESOLVED_CACHE.read_text())
                surface_vectors = _load_surface_vectors(config.VECTOR_CACHE)
                entity_vectors.run(resolved["entities"], surface_vectors)

        elif name == "relations":
            inputs = [config.ENTITY_CACHE, config.RESOLVED_CACHE]
            if not forced and _is_fresh(config.RELATIONS_CACHE, inputs):
                print("[6/6] relations: fresh — skipped")
            else:
                from pipeline import relations

                relations.run(model=config.RELATION_MODEL)

        elif name == "upload":
            if args.skip_upload:
                print("[upload] skipped (--skip-upload)")
            else:
                _run_upload(args)

        print(f"  ({name}: {time.monotonic() - start:.1f}s)")


if __name__ == "__main__":
    main()
