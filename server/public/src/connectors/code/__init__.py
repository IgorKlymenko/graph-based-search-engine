"""Source code into text the ingest can read, with its references pre-resolved.

    model.py    Symbol / Parsed records, the Parser protocol, the registry
    python.py   the Python parser (stdlib ``ast``; no third-party grammar)
    render.py   symbols -> one paragraph each, plus offset-addressed mentions

The package knows nothing about where code comes from. GitHub, a local
checkout or a zip upload all end here, and a new language is a class that
satisfies ``Parser`` registered by extension in ``model.PARSERS``.
"""

from src.connectors.code.model import PARSERS, Parsed, Parser, Symbol, parser_for
from src.connectors.code.render import (
    Style,
    ambiguous_from,
    ambiguous_names,
    index_document,
    names_of,
    render,
)

__all__ = ["PARSERS", "Parsed", "Parser", "Style", "Symbol", "ambiguous_from",
           "ambiguous_names", "index_document", "names_of", "parser_for", "render"]
