"""Collect Workvivo help-centre articles into a clean markdown corpus.

Fetches every published article from the Zendesk Help Center API, strips the
WYSIWYG markup down to prose, replaces each screenshot with a vision-model
description, and writes one ``.md`` file per article for the standard pipeline:

    uv run python demo_Tess/collect_workvivo_docs.py
    uv run python -m pipeline.run --notes-dir demo_Tess/workvivo_corpus \\
        --group-id workvivo

Output:
    workvivo_corpus/<section>/<article>.md   prose only — no front matter
    workvivo_corpus/manifest.json            url/section/figures per file
    .image_cache.json                        image url -> description (reusable)

The ``.md`` bodies are deliberately front-matter free: ``pipeline.mentions``
globs ``*.md`` and feeds the raw file text to the entity extractor, so any YAML
or JSON header would be chunked and extracted as if it were prose. Article
metadata lives in ``manifest.json`` instead, keyed by the path recorded on every
mention as ``source_file`` — map ``updated_at`` onto ``valid_at`` from there.

Screenshots carry facts that appear nowhere in the prose (admin menu trees,
field limits, option names), and their alt text is worthless ("mceclip0.png").
Each is described by ``IMAGE_MODEL`` and inlined as "Figure N: ...", with
``manifest.json`` mapping every label back to its ``image_url`` and article
``url`` — so a graph answer citing a source file can point at the page and the
original picture. Descriptions are cached by image url, so reruns are free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import html2text
import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from services.openai import complete_json_schema_image  # noqa: E402

BASE = "https://support.workvivo.com/api/v2/help_center"
DEFAULT_LOCALE = "en-gb"
DEFAULT_OUT = pathlib.Path(__file__).parent / "workvivo_corpus"
# Beside the script, not in --out: descriptions are keyed by image url alone, so
# a small --match demo run pre-warms the full run instead of being re-paid for.
DEFAULT_CACHE = pathlib.Path(__file__).parent / ".image_cache.json"

IMAGE_MODEL = "gpt-4.1-mini"  # vision-capable; ~5x cheaper than gpt-4.1
IMAGE_WORKERS = 8
IMAGE_CONTEXT_CHARS = 400
ICON_MAX_PX = 100  # images declaring a smaller side are icons/emoji, not content
# A caption of a dense table can run past pipeline.config.CHUNK_MAX_CHARS (1000).
# chunk_doc splits on blank lines and never splits a paragraph, so a one-paragraph
# caption becomes one oversized, fact-dense chunk that pushes mention extraction
# past LLM_TIMEOUT. Emit long captions as several paragraphs instead.
FIGURE_PARAGRAPH_CHARS = 700
CACHE_SAVE_EVERY = 25

# Carry no text: video embeds (Zoom Clips/Loom) and the ~1.5k <hr> separators.
DROP_TAGS = (
    "script", "style", "noscript", "iframe", "video",
    "audio", "source", "picture", "svg", "button", "colgroup",
)
# Pure styling wrappers — unwrap to keep the text, drop the tag. Nearly every
# sentence is wrapped in <span class="wysiwyg-font-size-large">.
UNWRAP_TAGS = (
    "span", "font", "div", "figure", "figcaption", "section",
    "zd-summary-block", "u", "ins",
)
# Meaningful only when they hold text.
PRUNE_IF_EMPTY = ("p", "li", "td", "th", "tr", "ul", "ol", "table", "blockquote")

REQUEST_TIMEOUT = 30
PAGE_SIZE = 100
POLITE_DELAY = 0.3
MAX_RETRIES = 4

IMAGE_SYSTEM_PROMPT = """You caption screenshots from the Workvivo help centre so that their content survives as text in a knowledge graph. The picture itself is discarded; your caption is all a reader will ever see of it.

Describe the screen's main content area. Transcribe verbatim every piece of text in it: tab and section headings, field labels, button text, option names, default values, limits, and help text. Name the product area and the screen it shows. Prefer full sentences that state what the screen offers ("The Authentication Mode section offers four options: ...") over a bare list of words, because the sentences are what a downstream extractor reads.

For navigation chrome — left sidebars, top navigation, breadcrumbs — record only which item is selected and the path to it ("reached under Provisioning > Authentication and System Settings"). Do not transcribe the whole menu. The same sidebar appears in dozens of pictures, and listing every item turns each menu label into a significant-looking entity in every article that shows it.

Keep every sentence under about 30 words. When the picture shows a table, a set of field mappings, or any other repeated row-and-value structure, do not enumerate it in one long run-on sentence. Introduce it with a short sentence and then render one row per line as a markdown list, like:

The Attribute Mappings screen maps Microsoft Entra ID attributes to Workvivo fields:
- userName maps to userPrincipalName (the Email field in Workvivo)
- title maps to jobTitle (the Job Title field in Workvivo)

A downstream extractor quotes the whole sentence containing each name it finds, so one sentence listing thirty rows makes it quote all thirty rows thirty times.

Never speculate about anything not visible, never invent feature behaviour, and never describe colours, spacing, or visual styling. Do not mention that this is a screenshot or an image; describe the interface directly.

If the picture is purely decorative and carries no information a reader would need — an icon, logo, spacer, divider, arrow, emoji, or a generic avatar or stock photo — set "decorative" to true and "description" to an empty string."""

IMAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decorative": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["decorative", "description"],
}


class Figure(NamedTuple):
    """One image in an article, awaiting or holding its description."""

    url: str
    alt: str


class Article(NamedTuple):
    """An article converted to markdown, before figures are substituted in."""

    raw: dict
    path: pathlib.Path
    markdown: str  # still contains {{FIG:i}} placeholders
    figures: list[Figure]
    category: str
    section: str


# ── fetching ──────────────────────────────────────────────────────────────────

def fetch_all(session: requests.Session, locale: str, endpoint: str) -> list[dict]:
    """Walk Zendesk pagination and return all records for ``endpoint``."""
    url = f"{BASE}/{locale}/{endpoint}.json?per_page={PAGE_SIZE}"
    records: list[dict] = []
    while url:
        data = _get_json(session, url)
        records.extend(data[endpoint])
        url = data.get("next_page")
        time.sleep(POLITE_DELAY)
    return records


def _get_json(session: requests.Session, url: str) -> dict[str, Any]:
    """GET ``url``, retrying on rate limits and transient server errors."""
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("Retry-After", 2**attempt))
            print(f"  {resp.status_code} on {url} — retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts: {url}")


# ── html -> markdown ──────────────────────────────────────────────────────────

def _markdown_converter() -> html2text.HTML2Text:
    """html2text tuned for prose: no link URLs, no images, no wrapping."""
    h2t = html2text.HTML2Text()
    h2t.ignore_images = True
    h2t.ignore_links = True  # keep anchor text, drop the 572 inline URLs
    h2t.body_width = 0  # no hard wrapping -> cleaner chunks
    return h2t


def clean_html(body: str, *, describe_images: bool) -> tuple[str, list[Figure]]:
    """Strip Zendesk WYSIWYG markup, leaving only content-bearing elements.

    Returns the cleaned HTML and the figures found. When ``describe_images`` is
    set each ``<img>`` becomes a ``{{FIG:i}}`` placeholder indexing the returned
    list; otherwise images are dropped outright.
    """
    soup = BeautifulSoup(body, "lxml")
    # Drop first: an <img> inside a dropped container (<picture>, <video>) must
    # never become a figure whose placeholder then vanishes with its parent.
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    if describe_images:
        figures = _extract_figures(soup)
    else:
        figures = []
        for img in soup.find_all("img"):
            img.decompose()

    for tag in soup.find_all(UNWRAP_TAGS):
        tag.unwrap()

    # <a> anchors wrapping only an image, and in-page "back to top" links,
    # would otherwise leave empty brackets behind.
    for anchor in soup.find_all("a"):
        if not anchor.get_text(strip=True):
            anchor.decompose()

    _prune_empty(soup)
    return str(soup), figures


def _extract_figures(soup: BeautifulSoup) -> list[Figure]:
    """Swap every ``<img>`` for a ``{{FIG:i}}`` placeholder, in document order."""
    figures: list[Figure] = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src.startswith(("http://", "https://")) or _is_icon(img):
            img.decompose()  # unfetchable, or too small to hold information
            continue
        figures.append(Figure(url=src, alt=(img.get("alt") or "").strip()))
        img.replace_with(f" {{{{FIG:{len(figures) - 1}}}}} ")
    return figures


def _is_icon(img: Tag) -> bool:
    """True for inline icons and emoji — too small to caption, so never sent."""
    for dimension in ("width", "height"):
        value = str(img.get(dimension) or "").strip().rstrip("px")
        if value.isdigit() and int(value) < ICON_MAX_PX:
            return True
    return False


def _prune_empty(soup: BeautifulSoup) -> None:
    """Drop elements left textless by DROP_TAGS (e.g. <p> holding only a video).

    Runs innermost-first and repeats, so a list whose every item was dropped
    collapses entirely rather than leaving an empty <ul> behind.
    """
    while True:
        removed = False
        for tag in soup.find_all(PRUNE_IF_EMPTY):
            if isinstance(tag, Tag) and not tag.get_text(strip=True):
                tag.decompose()
                removed = True
        if not removed:
            return


def to_markdown(body: str, h2t: html2text.HTML2Text, *, describe_images: bool
                ) -> tuple[str, list[Figure]]:
    """Convert one article's HTML body to clean markdown prose."""
    html, figures = clean_html(body, describe_images=describe_images)
    md = h2t.handle(html)
    md = md.replace("\xa0", " ").replace("​", "")
    # html2text escapes "1." as "1\." so it can't start an ordered list; no
    # article body contains a literal "\.", so unescaping can't corrupt the
    # JSON samples (which use \") that some articles document.
    md = md.replace("\\.", ".")
    md = re.sub(r"^\s*\*\s\*\s\*\s*$", "", md, flags=re.MULTILINE)  # <hr> rules
    md = re.sub(r"^[ \t]*[*+-][ \t]*$", "", md, flags=re.MULTILINE)  # empty bullets
    md = re.sub(r"^(#+)\s*$", "", md, flags=re.MULTILINE)  # emptied headings
    md = _strip_emphasis_around_figures(md)
    md = re.sub(r"[ \t]+\n", "\n", md)  # trailing spaces / <br> markers
    md = re.sub(r"[ \t]{2,}", " ", md)  # runs left by removed inline images
    md = re.sub(r"\n{3,}", "\n\n", md)  # blank runs -> one paragraph break
    return md.strip(), figures


def _strip_emphasis_around_figures(md: str) -> str:
    """Unwrap emphasis that spans only figure placeholders.

    Images are often wrapped in ``<em>``/``<strong>``, giving ``_{{FIG:4}}_``.
    A caption is substituted in as its own block, which would strand those
    markers on their own line.
    """
    return re.sub(
        r"(\*\*|__|\*|_)(\s*(?:\{\{FIG:\d+\}\}\s*)+)\1",
        lambda m: m.group(2),
        md,
    )


# ── image description ─────────────────────────────────────────────────────────

def describe_image(url: str, *, title: str, alt: str, context: str, model: str
                   ) -> dict[str, Any]:
    """Caption one screenshot. Returns ``{"decorative": bool, "description": str}``."""
    user = f"This picture appears in the Workvivo help-centre article {title!r}."
    if alt and not re.fullmatch(r"[\w.-]+\.(png|jpe?g|gif|webp)", alt, re.I):
        user += f'\nIts alt text is "{alt}".'
    if context:
        user += f"\n\nSurrounding article text:\n{context}"
    user += "\n\nCaption this picture."
    return complete_json_schema_image(
        system=IMAGE_SYSTEM_PROMPT,
        user=user,
        image_url=url,
        schema=IMAGE_SCHEMA,
        schema_name="image_description",
        model=model,
    )


def _context_around(markdown: str, index: int) -> str:
    """The prose surrounding a ``{{FIG:index}}`` placeholder, for grounding."""
    at = markdown.find(f"{{{{FIG:{index}}}}}")
    if at == -1:
        return ""
    before = markdown[max(0, at - IMAGE_CONTEXT_CHARS):at]
    after = markdown[at:at + IMAGE_CONTEXT_CHARS]
    text = f"{before}{after}"
    return re.sub(r"\{\{FIG:\d+\}\}", " ", text).strip()


def describe_all(articles: list[Article], cache: dict[str, dict],
                 cache_path: pathlib.Path, *, model: str) -> None:
    """Describe every uncached figure across all articles, in parallel.

    Mutates ``cache`` in place and checkpoints it, so an interrupted run
    resumes without re-paying for images already described.
    """
    jobs = {
        fig.url: (article, fig, index)
        for article in articles
        for index, fig in enumerate(article.figures)
        if fig.url not in cache
    }
    total = sum(len(a.figures) for a in articles)
    print(f"figures: {total} ({len(jobs)} to describe, {total - len(jobs)} cached)")
    if not jobs:
        return

    lock = threading.Lock()
    done = 0

    def _one(item: tuple[str, tuple[Article, Figure, int]]) -> None:
        nonlocal done
        url, (article, fig, index) = item
        try:
            result = describe_image(
                url,
                title=article.raw["title"],
                alt=fig.alt,
                context=_context_around(article.markdown, index),
                model=model,
            )
        except Exception as exc:  # keep the corpus building; drop this figure
            print(f"  failed {url}: {exc}")
            result = {"decorative": True, "description": ""}
        with lock:
            cache[url] = result
            done += 1
            if done % CACHE_SAVE_EVERY == 0:
                _save_cache(cache_path, cache)
                print(f"  described {done}/{len(jobs)}")

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as executor:
        list(executor.map(_one, jobs.items()))
    _save_cache(cache_path, cache)
    print(f"  described {done}/{len(jobs)}")


def _prompt_fingerprint() -> str:
    """Identify the prompt a cached caption was written by."""
    return hashlib.sha256(IMAGE_SYSTEM_PROMPT.encode()).hexdigest()[:12]


def _load_cache(path: pathlib.Path) -> dict[str, dict]:
    """Cached captions, dropped wholesale when IMAGE_SYSTEM_PROMPT has changed."""
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("prompt") != _prompt_fingerprint():
        print(f"  caption prompt changed — ignoring {len(stored.get('images', {}))} "
              f"cached descriptions")
        return {}
    return stored.get("images", {})


def _save_cache(path: pathlib.Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"prompt": _prompt_fingerprint(), "images": cache},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _caption_paragraphs(label: str, description: str) -> str:
    """Lay a caption out as paragraphs of at most ``FIGURE_PARAGRAPH_CHARS``.

    Splits on sentence boundaries and keeps the label on every paragraph, so a
    caption that spans chunks stays attributable to its figure. A caption that
    renders a table as one row per line is left as-is: its line structure is
    what keeps each row's extracted context down to a single row.
    """
    description = description.strip()
    lead, newline, rows = description.partition("\n")
    if newline and rows.strip():
        return f"{label}: {lead.strip()}\n{rows.rstrip()}"

    sentences = re.split(r"(?<=[.!?])\s+", description)
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > FIGURE_PARAGRAPH_CHARS:
            paragraphs.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        paragraphs.append(current)
    return "\n\n".join(
        f"{label}: {text}" if i == 0 else f"{label} (continued): {text}"
        for i, text in enumerate(paragraphs)
    )


def substitute_figures(article: Article, cache: dict[str, dict]
                       ) -> tuple[str, list[dict]]:
    """Replace ``{{FIG:i}}`` placeholders with "Figure N: ..." captions.

    Decorative and failed figures are dropped. Numbering runs over the kept
    figures in document order; the returned records map each label back to its
    image url for the manifest.
    """
    markdown = article.markdown
    kept: list[dict] = []
    for index, fig in enumerate(article.figures):
        described = cache.get(fig.url) or {}
        description = (described.get("description") or "").strip()
        placeholder = f"{{{{FIG:{index}}}}}"
        if described.get("decorative") or not description:
            markdown = markdown.replace(placeholder, " ")
            continue
        label = f"Figure {len(kept) + 1}"
        caption = _caption_paragraphs(label, description)
        markdown = markdown.replace(placeholder, f"\n\n{caption}\n\n")
        kept.append({
            "label": label,
            "image_url": fig.url,
            "description": description,
        })
    # Emphasis that opened before a caption and closed after it strands markers.
    markdown = re.sub(r"^[ \t]*[_*]{1,3}[ \t]*$", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"[ \t]{2,}", " ", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip(), kept


# ── assembly ──────────────────────────────────────────────────────────────────

def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "untitled"


def build_document(title: str, category: str, section: str, body_md: str) -> str:
    """Assemble the file text: H1, a prose locator line, then the body.

    The locator is a sentence rather than a metadata block because the extractor
    reads it as ordinary prose; it grounds the article in the help-centre
    taxonomy so the first chunk yields the section and category as entities.
    """
    locator = (
        f'This article belongs to the "{section}" section of the Workvivo '
        f'help centre, under the "{category}" category.'
    )
    return f"# {title}\n\n{locator}\n\n{body_md}\n"


def _unique_path(out: pathlib.Path, section: str, article: dict,
                 seen: set[pathlib.Path]) -> pathlib.Path:
    """Path for an article, suffixed with its id when the slug already exists."""
    directory = out / slugify(section)
    path = directory / f"{slugify(article['title'])}.md"
    if path in seen:
        path = directory / f"{slugify(article['title'])}-{article['id']}.md"
    seen.add(path)
    return path


def prepare_articles(raw_articles: list[dict], sections: dict, categories: dict,
                     out: pathlib.Path, *, describe_images: bool,
                     limit: int | None = None, match: str | None = None,
                     only_categories: set[str] | None = None,
                     only_sections: set[str] | None = None,
                     ) -> tuple[list[Article], int]:
    """Convert raw API records to markdown-with-placeholders, in document order."""
    h2t = _markdown_converter()
    prepared: list[Article] = []
    seen: set[pathlib.Path] = set()
    skipped = 0

    for raw in raw_articles:
        if limit is not None and len(prepared) >= limit:
            break
        if raw.get("draft") or not raw.get("body"):
            skipped += 1
            continue
        section = sections.get(raw.get("section_id"), {})
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


def _name_set(value: str | None) -> set[str] | None:
    """Parse a comma-separated CLI list of taxonomy names."""
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _reject_unknown(requested: set[str] | None, known: set[str], what: str) -> None:
    """Exit on a misspelled name rather than silently ingesting nothing."""
    unknown = (requested or set()) - known
    if unknown:
        sys.exit(f"unknown {what} name(s): {sorted(unknown)}\n"
                 f"known {what} names:\n  " + "\n  ".join(sorted(known)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                   help=f"Corpus output directory (default: {DEFAULT_OUT})")
    p.add_argument("--locale", default=DEFAULT_LOCALE,
                   help=f"Help-centre locale (default: {DEFAULT_LOCALE})")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N articles (smoke tests)")
    p.add_argument("--match", default=None,
                   help="Only articles whose title matches this regex "
                        "(small themed demo runs)")
    p.add_argument("--categories", default=None,
                   help="Comma-separated category names to ingest; OR'd with "
                        "--sections. Omit both to ingest everything.")
    p.add_argument("--sections", default=None,
                   help="Comma-separated section names to ingest; OR'd with "
                        "--categories.")
    p.add_argument("--image-model", default=IMAGE_MODEL,
                   help=f"Vision model for figure captions (default: {IMAGE_MODEL})")
    p.add_argument("--image-cache", type=pathlib.Path, default=DEFAULT_CACHE,
                   help=f"Figure description cache (default: {DEFAULT_CACHE})")
    p.add_argument("--no-describe-images", action="store_true",
                   help="Drop screenshots instead of captioning them (no LLM calls)")
    return p.parse_args()


def main() -> None:
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
    args = parse_args()
    describe_images = not args.no_describe_images

    session = requests.Session()
    session.headers["User-Agent"] = "workvivo-corpus-collector/1.0"
    categories = {
        c["id"]: c["name"] for c in fetch_all(session, args.locale, "categories")
    }
    sections = {s["id"]: s for s in fetch_all(session, args.locale, "sections")}
    raw_articles = fetch_all(session, args.locale, "articles")
    print(f"{len(categories)} categories, {len(sections)} sections, "
          f"{len(raw_articles)} articles")

    only_categories = _name_set(args.categories)
    only_sections = _name_set(args.sections)
    _reject_unknown(only_categories, set(categories.values()), "category")
    _reject_unknown(only_sections, {s["name"] for s in sections.values()}, "section")

    articles, skipped = prepare_articles(
        raw_articles, sections, categories, args.out,
        describe_images=describe_images, limit=args.limit, match=args.match,
        only_categories=only_categories, only_sections=only_sections,
    )

    cache = _load_cache(args.image_cache)
    if describe_images:
        describe_all(articles, cache, args.image_cache, model=args.image_model)

    manifest: dict[str, dict] = {}
    for article in articles:
        body_md, figures = substitute_figures(article, cache)
        article.path.parent.mkdir(parents=True, exist_ok=True)
        article.path.write_text(
            build_document(
                article.raw["title"], article.category, article.section, body_md
            ),
            encoding="utf-8",
        )
        manifest[str(article.path.relative_to(args.out))] = {
            "title": article.raw["title"],
            "url": article.raw["html_url"],
            "category": article.category,
            "section": article.section,
            "labels": article.raw.get("label_names", []),
            "created_at": article.raw["created_at"],
            "updated_at": article.raw["updated_at"],
            "figures": figures,
        }

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    chars = sum(len(p.read_text(encoding="utf-8")) for p in args.out.rglob("*.md"))
    n_figures = sum(len(m["figures"]) for m in manifest.values())
    print(f"wrote {len(manifest)} articles, {n_figures} figures, {chars:,} chars "
          f"(skipped {skipped})")
    print(f"Done -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
