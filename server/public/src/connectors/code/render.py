"""Symbols -> text with one paragraph per symbol, and the mentions inside it.

The shape is chosen for the splitter and the hashes, not for a reader:

* One blank-line-separated paragraph per symbol, so ``split_episodes`` packs
  symbols and an edit inside one function re-hashes that function's episode
  and the ones packed after it, never the whole file (the same reasoning as
  one Slack message per paragraph, ADR slack 8).
* No line numbers, no offsets. They move on every edit above them and would
  re-extract every symbol below an inserted blank line.
* Lower-case kind words (``function``, ``class``) so the parser does not read
  them as proper names.
* Every identifier the text shows is a ``Mention`` keyed by its qualified name.
  Prose (the docstring) is left to the model; it is what the model is for.

The renderer is versioned by ``episodes.RENDER_VERSIONS["github"]``; change
the output here and bump it there in the same commit.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import NamedTuple

from src.connectors.base import Mention
from src.connectors.code.model import CLASS, MODULE, Parsed, Symbol

_REFERENCE_LABEL = {MODULE: "imports", CLASS: "uses"}


class _Builder:
    """Appends text while recording where each mention landed."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.mentions: list[Mention] = []

    def text(self) -> str:
        return "".join(self.parts)

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def mention(self, text: str, key: str) -> None:
        self.mentions.append(Mention(self.length, self.length + len(text), text, key))
        self.add(text)

    def listing(self, label: str, items: list[tuple[str, str]]) -> None:
        self.add(f"\n{label}: ")
        for i, (text, key) in enumerate(items):
            if i:
                self.add(", ")
            self.mention(text, key)


class Style(NamedTuple):
    """How names are shown. The defaults are what ships (ADR connectors 11);
    every combination is measured by ``scripts/eval_code_collisions.py``.

    With the defaults, a short name that means more than one thing in the repo
    is never a mention on its own: references to it render qualified, and its
    definitions show the bare name as plain text next to the qualified one.
    Measured: false merges 303 -> 1 (this repo's live source) and 2039 -> 2
    (``old_server/entity_linking``) with no same-key pair missed; qualifying
    references alone left 68 and 1206.
    """

    ambiguous: frozenset[str] = frozenset()  # short names meaning >1 thing
    short_ambiguous_definitions: bool = False  # emit the bare name on those defs
    class_dot_methods: bool = False  # show methods as Class.method everywhere


def render(parsed: Parsed, *, path: str, ambiguous: frozenset[str] = frozenset(),
           style: Style | None = None) -> tuple[str, tuple[Mention, ...]]:
    """The document for one file: text and the mentions addressed into it.

    ``ambiguous`` is the set of short names that denote more than one thing in
    the repository (``ambiguous_names``). A reference to one of them renders
    by its qualified name, so the surface the chain carries is the one that
    tells the two apart. Definitions show both forms unless ``style`` says
    otherwise.
    """
    style = style or Style(ambiguous=ambiguous)
    builder = _Builder()
    for i, symbol in enumerate(parsed.symbols):
        if i:
            builder.add("\n\n")
        _paragraph(builder, symbol, path, style)
    return builder.text(), tuple(builder.mentions)


def names_of(parsed: Parsed, *, package: bool = False) -> dict:
    """Short name -> qualified names, split into what the file defines and refers to.

    Small enough to keep per path in a connector's cursor, which is what lets
    ambiguity be recomputed on an incremental sync without re-fetching files
    whose content did not change. ``module`` is the file's own importable
    name and ``package`` says whether it is an ``__init__``, so the set of
    modules a repo actually has can be rebuilt from the maps
    (``ambiguous_from`` needs both).
    """
    defs: dict[str, set[str]] = {}
    refs: dict[str, set[str]] = {}
    for symbol in parsed.symbols:
        defs.setdefault(symbol.name, set()).add(symbol.qualname)
        for reference in symbol.references:
            refs.setdefault(short(reference), set()).add(reference)
    return {"module": parsed.module, "package": package,
            "defs": {k: sorted(v) for k, v in defs.items()},
            "refs": {k: sorted(v) for k, v in refs.items()}}


def ambiguous_from(symbol_maps) -> frozenset[str]:
    """Short names that name more than one qualified thing across ``names_of`` maps.

    Counts definitions and references alike, so ``load`` is ambiguous when the
    repo defines one ``load`` and also calls ``numpy.load``: rendered short,
    the two would share a surface and alias evidence would fuse them.

    A reference whose module is *inside* the repo's own package tree but is
    not a file there is a re-export the parser cannot follow -- here,
    ``src.inject.pairing`` is the loader's alias for ``4_link.py`` -- and it
    may well name the very definition it seems to compete with. Such a
    reference is dropped from the count rather than read as a second thing:
    measured, it made every stage function in this repository "ambiguous",
    stripped the bare name from their definitions, and the judge then marked
    ``keys_settle`` a non-name. External references (root not in the repo)
    still count.
    """
    maps = list(symbol_maps)
    files = {m["module"] for m in maps if m.get("module") and not m.get("package")}
    packages = {m["module"] for m in maps if m.get("module") and m.get("package")}
    roots = {m.partition(".")[0] for m in files | packages}
    seen: dict[str, set[str]] = {}
    for names in maps:
        for name, qualnames in names["defs"].items():
            seen.setdefault(name, set()).update(qualnames)
        for name, qualnames in names["refs"].items():
            seen.setdefault(name, set()).update(
                q for q in qualnames if not _dangling(q, files, packages, roots))
    return frozenset(name for name, qualnames in seen.items() if len(qualnames) > 1)


def _dangling(qualname: str, files: set[str], packages: set[str],
              roots: set[str]) -> bool:
    """In the repo's own tree, but nothing there can define it.

    Walk the dotted prefix up. Reaching a module *file* means the rest is a
    member of something the parser saw (``pkg.mod.Class.method``); reaching
    a *package* first means the next segment should have been a file or
    subpackage and is not -- a re-export the parser cannot follow.
    """
    prefix = qualname.rpartition(".")[0]
    if prefix.partition(".")[0] not in roots:
        return False  # external: numpy.load is a real, different thing
    while prefix:
        if prefix in files:
            return False
        if prefix in packages:
            return True
        prefix = prefix.rpartition(".")[0]
    return True


def ambiguous_names(parsed_files: list[Parsed]) -> frozenset[str]:
    """``ambiguous_from`` over freshly parsed files."""
    return ambiguous_from(names_of(p) for p in parsed_files)


def index_document(symbol_maps, ambiguous: frozenset[str],
                   ) -> tuple[str, tuple[Mention, ...]]:
    """One paragraph per colliding name, each definition its own mention key.

    The paragraph puts the qualified names of one short name into a single
    episode as separate chains, which is the evidence ``pairing`` needs to mark
    the short name contested: two things some episode kept apart. Nothing else
    in the corpus says that about two functions called ``run``. Empty text when
    no name is defined more than once.
    """
    defs: dict[str, set[str]] = {}
    for names in symbol_maps:
        for name, qualnames in names["defs"].items():
            if name in ambiguous:
                defs.setdefault(name, set()).update(qualnames)
    b = _Builder()
    for i, name in enumerate(sorted(n for n in defs if len(defs[n]) > 1)):
        if i:
            b.add("\n\n")
        b.add(f"{name} is defined more than once: ")
        for j, qualname in enumerate(sorted(defs[name])):
            if j:
                b.add(", ")
            b.mention(qualname, qualname)
    return b.text(), tuple(b.mentions)


def _paragraph(b: _Builder, symbol: Symbol, path: str, style: Style) -> None:
    b.add(f"{symbol.kind} ")
    shown = _display(symbol.qualname, Style(class_dot_methods=style.class_dot_methods),
                     kind=symbol.kind)
    if symbol.name in style.ambiguous and not style.short_ambiguous_definitions:
        b.add(shown)  # plain text: two things share this name
    else:
        b.mention(shown, symbol.qualname)
    b.add(" (")
    b.mention(symbol.qualname, symbol.qualname)
    b.add(")")
    if symbol.kind == MODULE:
        filename = PurePosixPath(path).name
        if filename.startswith("__init__"):
            b.add(f", package at {path}")  # every package has one; it names none
        else:
            b.add(", file ")
            b.mention(filename, symbol.qualname)
            b.add(f" at {path}")
    if symbol.signature:
        b.add("\n" + symbol.signature)
    if symbol.children:
        b.listing("defines", [(_display(c, style), c) for c in symbol.children])
    if symbol.references:
        label = _REFERENCE_LABEL.get(symbol.kind, "calls")
        full = symbol.kind == MODULE  # a module is named by its dotted path
        b.listing(label, [(r if full else _display(r, style), r)
                          for r in symbol.references])
    if symbol.doc.strip():
        b.add("\n" + symbol.doc.strip())


def _display(qualname: str, style: Style, *, kind: str = "") -> str:
    """How a reference or definition shows a qualified name."""
    name = short(qualname)
    if name in style.ambiguous:
        return qualname
    if style.class_dot_methods and kind != MODULE:
        owner = qualname.rpartition(".")[0].rpartition(".")[2]
        if owner[:1].isupper():  # PEP 8: a capitalised owner is a class
            return f"{owner}.{name}"
    return name


def short(qualname: str) -> str:
    """The name people say: the last segment of a qualified name."""
    return qualname.rpartition(".")[2]
