"""Pipeline orchestrator.

Runs the six extraction stages in order and finishes by invoking
``graph_upload.py`` (always with ``--skip-facts``). Each stage's artifact is
reused when it is newer than its inputs; ``--force`` reruns from ``--from``.

    uv run python -m pipeline.run --notes-dir /Applications/reink
    uv run python -m pipeline.run --notes-dir ./notes --from relations --force
    uv run python -m pipeline.run --notes-dir ./notes --skip-upload
    uv run python -m pipeline.run --notes-dir ./other --group-id my-new-group

With ``--group-id`` every stage artifact lives in ``artifacts/<group_id>/`` and
the id is stamped on all uploaded nodes/edges — the group is fully isolated
from other groups' caches and search results. Without it, artifacts stay at
the repo root (the original layout).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from dotenv import load_dotenv

from pipeline import config

STAGES = ["candidates", "embed", "resolve", "entity-vectors", "relations", "upload"]


class ArtifactPaths(NamedTuple):
    """Locations of the stage artifacts for one document group."""

    base: Path
    entity: Path
    vector: Path
    resolved: Path
    entity_vector: Path
    relations: Path


def artifact_paths(group_id: str | None) -> ArtifactPaths:
    """Caches at the repo root by default; ``artifacts/<group_id>/`` when set."""
    base = Path("artifacts") / group_id if group_id else Path(".")
    return ArtifactPaths(
        base=base,
        entity=base / config.ENTITY_CACHE.name,
        vector=base / config.VECTOR_CACHE.name,
        resolved=base / config.RESOLVED_CACHE.name,
        entity_vector=base / config.ENTITY_VECTOR_CACHE.name,
        relations=base / config.RELATIONS_CACHE.name,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--notes-dir", type=Path, required=True,
                   help="Directory scanned recursively for .md source files")
    p.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0],
                   help="Run from this stage onward (default: candidates)")
    p.add_argument("--only", choices=STAGES, default=None,
                   help="Run exactly one stage")
    p.add_argument("--force", action="store_true",
                   help="Rerun stages from --from even if their artifacts are fresh")
    p.add_argument("--extraction-model", default=config.EXTRACTION_MODEL)
    p.add_argument("--no-gleaning", action="store_true",
                   help="Disable the second-pass gleaning call in candidate extraction")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Cap input tokens of pending files processed this run")
    p.add_argument("--skip-upload", action="store_true",
                   help="Stop after relations; do not invoke graph_upload.py")
    p.add_argument("--dry-run", action="store_true",
                   help="Forward --dry-run to graph_upload.py (no Neo4j writes)")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Forward --skip-embeddings to graph_upload.py")
    p.add_argument("--saga", default=None,
                   help="Saga container node name (default: the group id)")
    p.add_argument("--group-id", default=None,
                   help="Isolated document group: caches live in "
                        "artifacts/<group_id>/ and the id is stamped on every "
                        "uploaded node/edge (default: repo-root caches)")
    p.add_argument("--min-confidence", type=float, default=None)
    return p.parse_args()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _is_fresh(output: Path, inputs: list[Path]) -> bool:
    """True when ``output`` exists and is at least as new as every input."""
    if not output.exists():
        return False
    return _mtime(output) >= max((_mtime(p) for p in inputs), default=0.0)


def _load_unit_vectors(path: Path) -> dict[str, np.ndarray]:
    """``{unit_key: vector}`` — one row per (concept, context, episode)."""
    if not path.exists():
        sys.exit(f"missing vector cache: {path} — run the 'embed' stage first")
    npz = np.load(path, allow_pickle=False)
    if "keys" not in npz:
        sys.exit(f"{path} predates unit-level vectors — rerun the 'embed' stage "
                 f"with --force")
    return {str(k): v for k, v in zip(npz["keys"], npz["matrix"])}


def _selected_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [args.only]
    return STAGES[STAGES.index(args.from_stage):]


def _run_upload(args: argparse.Namespace, artifacts_dir: Path) -> None:
    cmd = ["uv", "run", "python", "graph_upload.py", "--skip-facts",
           "--artifacts-dir", str(artifacts_dir)]
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

    paths = artifact_paths(args.group_id)
    if args.group_id:
        paths.base.mkdir(parents=True, exist_ok=True)
        print(f"group {args.group_id!r}: artifacts in {paths.base}/")

    stages = _selected_stages(args)
    forced = args.force or args.only is not None

    for name in stages:
        start = time.monotonic()

        if name == "candidates":
            # Incremental + resumable: safe to always run (no-op if all cached).
            from pipeline import candidates

            candidates.run(
                args.notes_dir,
                model=args.extraction_model,
                enable_gleaning=not args.no_gleaning,
                token_budget=args.token_budget,
                file_workers=config.FILE_WORKERS,
                episode_workers=config.EPISODE_WORKERS,
                cache_path=paths.entity,
            )

        elif name == "embed":
            if not forced and _is_fresh(paths.vector, [paths.entity]):
                print("[2/6] embed: fresh — skipped")
            else:
                from pipeline import embed_candidates

                embed_candidates.run(
                    cache_path=paths.entity, vector_cache=paths.vector
                )

        elif name == "resolve":
            inputs = [paths.vector, paths.entity]
            if not forced and _is_fresh(paths.resolved, inputs):
                print("[4/6] resolve: fresh — skipped")
            else:
                from pipeline import resolution

                unit_vectors = _load_unit_vectors(paths.vector)
                candidates_data = json.loads(paths.entity.read_text())["candidates"]
                resolution.run(
                    unit_vectors,  # type: ignore[arg-type]
                    candidates_data,
                    resolve_model=config.RESOLVE_MODEL,
                    escalate_model=config.ESCALATE_MODEL,
                    workers=config.RESOLVE_WORKERS,
                    cache_path=paths.resolved,
                )

        elif name == "entity-vectors":
            inputs = [paths.resolved, paths.vector]
            if not forced and _is_fresh(paths.entity_vector, inputs):
                print("[5/6] entity vectors: fresh — skipped")
            else:
                from pipeline import entity_vectors

                resolved = json.loads(paths.resolved.read_text())
                unit_vectors = _load_unit_vectors(paths.vector)
                entity_vectors.run(
                    resolved["entities"],
                    unit_vectors,
                    cache_path=paths.entity_vector,
                )

        elif name == "relations":
            inputs = [paths.entity, paths.resolved]
            if not forced and _is_fresh(paths.relations, inputs):
                print("[6/6] relations: fresh — skipped")
            else:
                from pipeline import relations

                relations.run(
                    model=config.RELATION_MODEL,
                    entity_cache=paths.entity,
                    resolved_cache=paths.resolved,
                    cache_path=paths.relations,
                )

        elif name == "upload":
            if args.skip_upload:
                print("[upload] skipped (--skip-upload)")
            else:
                _run_upload(args, paths.base)

        print(f"  ({name}: {time.monotonic() - start:.1f}s)")


if __name__ == "__main__":
    main()
