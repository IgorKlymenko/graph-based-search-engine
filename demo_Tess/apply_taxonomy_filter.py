"""Add --categories / --sections filters to demo_Tess/collect_workvivo_docs.py.

The existing --match filters on article *title*, which is the wrong axis for
"ingest only these parts of the help centre" and misfires (a title regex for
"Teams" pulls in "Teams and the Teams Directory", which is about org teams, not
Microsoft Teams). These filter on the Zendesk taxonomy instead.

The two are OR'd, so a single section can be lifted out of an otherwise
excluded category (e.g. Admin Hub out of Features & Capabilities). Unknown
names are rejected loudly rather than silently ingesting nothing.

Run from the repo root once file access is restored:
    python /private/tmp/.../scratchpad/apply_taxonomy_filter.py
It verifies each target matches exactly and refuses to write otherwise.
"""

import pathlib
import sys

TARGET = pathlib.Path("demo_Tess/collect_workvivo_docs.py")

EDITS = [
    # 1. signature
    (
        '''                     limit: int | None = None, match: str | None = None,
                     ) -> tuple[list[Article], int]:''',
        '''                     limit: int | None = None, match: str | None = None,
                     only_categories: set[str] | None = None,
                     only_sections: set[str] | None = None,
                     ) -> tuple[list[Article], int]:''',
    ),
    # 2. resolve taxonomy before filtering; single selection predicate
    (
        '''        if match and not re.search(match, raw["title"], re.IGNORECASE):
            skipped += 1
            continue
        markdown, figures = to_markdown(
            raw["body"], h2t, describe_images=describe_images
        )
        if not markdown and not figures:  # body was nothing but video embeds
            skipped += 1
            continue
        section = sections.get(raw.get("section_id"), {})
        section_name = section.get("name", "Unknown")
        prepared.append(Article(
            raw=raw,
            path=_unique_path(out, section_name, raw, seen),
            markdown=markdown,
            figures=figures,
            category=categories.get(section.get("category_id"), "Uncategorized"),
            section=section_name,
        ))
    return prepared, skipped
''',
        '''        section = sections.get(raw.get("section_id"), {})
        section_name = section.get("name", "Unknown")
        category = categories.get(section.get("category_id"), "Uncategorized")
        if not _selected(category, section_name, raw["title"], match=match,
                         only_categories=only_categories,
                         only_sections=only_sections):
            skipped += 1
            continue
        markdown, figures = to_markdown(
            raw["body"], h2t, describe_images=describe_images
        )
        if not markdown and not figures:  # body was nothing but video embeds
            skipped += 1
            continue
        prepared.append(Article(
            raw=raw,
            path=_unique_path(out, section_name, raw, seen),
            markdown=markdown,
            figures=figures,
            category=category,
            section=section_name,
        ))
    return prepared, skipped


def _selected(category: str, section: str, title: str, *, match: str | None,
              only_categories: set[str] | None,
              only_sections: set[str] | None) -> bool:
    """True when an article passes every filter that was supplied.

    ``only_categories`` and ``only_sections`` are OR'd: naming a section pulls
    its articles in even when their category was not named, so one section can
    be lifted out of an otherwise excluded category.
    """
    if match and not re.search(match, title, re.IGNORECASE):
        return False
    if only_categories is None and only_sections is None:
        return True
    return (category in (only_categories or set())
            or section in (only_sections or set()))
''',
    ),
    # 3. CLI flags
    (
        '''    p.add_argument("--match", default=None,
                   help="Only articles whose title matches this regex "
                        "(small themed demo runs)")''',
        '''    p.add_argument("--match", default=None,
                   help="Only articles whose title matches this regex "
                        "(small themed demo runs)")
    p.add_argument("--categories", default=None,
                   help="Comma-separated category names to ingest; OR'd with "
                        "--sections. Omit both to ingest everything.")
    p.add_argument("--sections", default=None,
                   help="Comma-separated section names to ingest; OR'd with "
                        "--categories.")''',
    ),
    # 4. helpers
    (
        '''def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(''',
        '''def _name_set(value: str | None) -> set[str] | None:
    """Parse a comma-separated CLI list of taxonomy names."""
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _reject_unknown(requested: set[str] | None, known: set[str], what: str) -> None:
    """Exit on a misspelled name rather than silently ingesting nothing."""
    unknown = (requested or set()) - known
    if unknown:
        sys.exit(f"unknown {what} name(s): {sorted(unknown)}\\n"
                 f"known {what} names:\\n  " + "\\n  ".join(sorted(known)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(''',
    ),
    # 5. wire into main()
    (
        '''    articles, skipped = prepare_articles(
        raw_articles, sections, categories, args.out,
        describe_images=describe_images, limit=args.limit, match=args.match,
    )''',
        '''    only_categories = _name_set(args.categories)
    only_sections = _name_set(args.sections)
    _reject_unknown(only_categories, set(categories.values()), "category")
    _reject_unknown(only_sections, {s["name"] for s in sections.values()}, "section")

    articles, skipped = prepare_articles(
        raw_articles, sections, categories, args.out,
        describe_images=describe_images, limit=args.limit, match=args.match,
        only_categories=only_categories, only_sections=only_sections,
    )''',
    ),
]


def main() -> None:
    if not TARGET.exists():
        sys.exit(f"run me from the repo root: {TARGET} not found")
    source = TARGET.read_text()
    if "only_categories" in source:
        sys.exit("already applied — collect_workvivo_docs.py has only_categories")

    for old, new in EDITS:
        if source.count(old) != 1:
            sys.exit(f"no unique match, file has changed:\n{old[:90]}...")
        source = source.replace(old, new)

    TARGET.write_text(source)
    print(f"patched {TARGET}: --categories / --sections now available")


if __name__ == "__main__":
    main()
