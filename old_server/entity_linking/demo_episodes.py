"""Dry-run the episode store: one sample document, split, then edited.

    uv run python entity_linking/demo_episodes.py
    uv run python entity_linking/demo_episodes.py --dump-store /tmp/store.json

Prints the sample text, the episodes it splits into, and the store before and
after a one-paragraph edit. No LLM, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import episodes as E  # noqa: E402  (needs the path shim above)

TENANT = "aaaaaaaa-0000-0000-0000-000000000001"
DOC = dict(tenant_id=TENANT, source="upload", external_id="notes/provisioning.md")
MAX_CHARS = 200

BEFORE = """\
Workvivo supports user provisioning from a file import or from SCIM.

Email addresses must stay the same across a provisioning switch.

Team names are linked to shoutouts and to auto-enrollment for spaces.

Reach out to Workvivo Support before changing a provisioning source.
"""

AFTER = BEFORE + "Also, we need to work on pipipupu and PAPAPAPPTPTPR"


def show_split(doc: str) -> None:
    for i, record in enumerate(E.build_records("demo", doc, MAX_CHARS), 1):
        text = record["text"].replace("\n\n", " ⏎ ")
        print(f"  [{i}] {record['episode_id'][:8]}  "
              f"{len(record['text']):>3}c  {text}")


def show_store(store: dict, title: str) -> None:
    """Print the store with episode text shortened for the console."""
    view = {
        "files": store["files"],
        "episodes": {
            eid: {**ep, "text": ep["text"][:48] + f"… ({len(ep['text'])}c)"}
            for eid, ep in store["episodes"].items()
        },
    }
    print(f"\n=== STORE {title} ===")
    print(json.dumps(view, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump-store", type=Path, default=None,
                   help="Write the full file + episode records to this JSON file")
    args = p.parse_args()

    print(f"=== SAMPLE DOCUMENT ===\n{BEFORE}")
    print(f"\n=== SPLIT (max_chars={MAX_CHARS}, ⏎ = merged paragraph break) ===")
    show_split(BEFORE)

    store = E.empty_store()
    first = E.sync_file(store, BEFORE, max_chars=MAX_CHARS, **DOC)
    print(f"\nsync 1: {first.status}  added={len(first.added)} "
          f"removed={len(first.removed)}")
    show_store(store, "BEFORE CHANGE")

    print("\n=== EDIT: paragraph 2 rewritten ===")
    print("  - Email addresses must stay the same across a provisioning switch.")
    print("  + Email addresses MUST match exactly or a new account is created.")

    second = E.sync_file(store, AFTER, max_chars=MAX_CHARS, **DOC)
    print(f"\nsync 2: {second.status}  unchanged={len(second.unchanged)} "
          f"added={len(second.added)} removed={len(second.removed)}")
    for eid in second.unchanged:
        print(f"  kept    {eid[:8]}")
    for eid in second.added:
        print(f"  added   {eid[:8]}")
    for eid in second.removed:
        print(f"  retired {eid[:8]}")
    show_store(store, "AFTER CHANGE")

    if args.dump_store:
        E.save_store(args.dump_store, store)
        print(f"\nwrote full records -> {args.dump_store} "
              f"({args.dump_store.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
