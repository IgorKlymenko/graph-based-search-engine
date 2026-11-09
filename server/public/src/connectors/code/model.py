"""What every language parser produces, and how a file finds its parser.

A ``Symbol`` is the unit the graph will hold: a module, class, function or
method with a qualified name, a signature, its docstring and the qualified
names it refers to. That is deliberately all -- no offsets, no line numbers,
no bodies. Anything positional changes on every edit above it and would churn
episode hashes for symbols whose meaning never moved; anything larger than a
docstring is code, which the coreference model cannot read and the renderer
would only have to throw away.

References are only what the file can resolve: an imported name to the module
that defines it, ``self.m`` to a method the class defines, a bare name to a
definition in the same module. A call on a local variable is dropped, not kept
as written -- ``cur.execute`` in every database file would be one surface
named "execute" with no referent, and a hub in the graph. That precision costs
recall on duck-typed calls; type inference is the only way to get it back.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import NamedTuple, Protocol, runtime_checkable

MODULE = "module"
CLASS = "class"
FUNCTION = "function"
METHOD = "method"


class Symbol(NamedTuple):
    """One named thing in a source file. ``qualname`` is unique within a repo."""

    kind: str          # MODULE | CLASS | FUNCTION | METHOD
    name: str          # the short name people say: "ingest"
    qualname: str      # module-qualified: "src.inject.main.ingest"
    signature: str     # "def ingest(*, store_path: Path) -> dict"; "" for modules
    doc: str           # the docstring, cleaned; "" when there is none
    references: tuple[str, ...]  # qualified where resolvable: calls, bases, imports
    children: tuple[str, ...]    # qualnames defined directly inside this symbol


class Parsed(NamedTuple):
    """One file's symbols, module first, in source order."""

    module: str
    symbols: tuple[Symbol, ...]


@runtime_checkable
class Parser(Protocol):
    """A language. ``parse`` may raise ``SyntaxError``; the caller skips the file."""

    extensions: frozenset[str]

    def parse(self, path: str, text: str, *,
              tree: frozenset[str] = frozenset()) -> Parsed:
        """``path`` is repo-relative; ``tree`` is every path in the repo, so the
        parser can tell where the importable name of ``path`` starts."""
        ...


def _registry() -> dict[str, Parser]:
    from src.connectors.code.python import PythonParser

    parsers: dict[str, Parser] = {}
    for parser in (PythonParser(),):
        for extension in parser.extensions:
            parsers[extension] = parser
    return parsers


PARSERS: dict[str, Parser] = _registry()


def parser_for(path: str) -> Parser | None:
    """The parser for a file by extension, or None when the language is unknown."""
    return PARSERS.get(PurePosixPath(path).suffix.lower())
