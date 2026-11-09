"""Symbols are qualified, references resolve through imports, noise is dropped."""

from src.connectors.code import parser_for
from src.connectors.code.model import CLASS, FUNCTION, METHOD, MODULE

SRC = '''"""Top doc."""
import json
import numpy as np
from src.inject import episodes as EP
from .utils import spans

def helper(x):
    """Helps."""
    return json.dumps(x)

class Scorer(EP.Base):
    """Scores."""

    def fit(self, pairs):
        self.score(pairs)
        self.unknown_attr.thing()
        return np.mean(pairs)

    def score(self, pairs):
        return helper(pairs) + spans.parse_spans("x") + len(pairs)

async def run():
    cur = object()
    cur.execute("x")
    return helper(1)
'''


TREE = frozenset({"src/__init__.py", "src/inject/__init__.py", "src/inject/link.py"})


def parse(path: str = "src/inject/link.py"):
    return parser_for(path).parse(path, SRC, tree=TREE)


def test_the_module_comes_first_with_its_definitions_as_children() -> None:
    parsed = parse()
    module = parsed.symbols[0]
    assert module.kind == MODULE
    assert module.qualname == "src.inject.link"
    assert module.name == "link"
    assert module.children == ("src.inject.link.helper", "src.inject.link.Scorer",
                               "src.inject.link.run")
    assert module.doc == "Top doc."


def test_stdlib_imports_are_not_references_but_the_rest_are() -> None:
    module = parse().symbols[0]
    assert "json" not in module.references
    assert module.references == ("numpy", "src.inject", "src.inject.utils")


def test_relative_imports_resolve_against_the_package() -> None:
    method = next(s for s in parse().symbols if s.name == "score")
    assert "src.inject.utils.spans.parse_spans" in method.references


def test_the_import_root_is_the_first_directory_without_an_init() -> None:
    """``src/`` is a layout directory in one repo and a package in another."""
    name = parser_for("x.py").module_name
    layout = frozenset({"src/pkg/__init__.py", "src/pkg/mod.py", "setup.py"})
    assert name("src/pkg/mod.py", layout) == "pkg.mod"
    package = frozenset({"src/__init__.py", "src/pkg/__init__.py", "src/pkg/mod.py"})
    assert name("src/pkg/mod.py", package) == "src.pkg.mod"
    assert name("src/pkg/__init__.py", package) == "src.pkg"
    assert name("scripts/tool.py", layout) == "tool"
    assert name("a/b/c.py") == "a.b.c"  # no tree: every directory is a package


def test_an_init_file_is_the_package_itself() -> None:
    parsed = parser_for("pkg/__init__.py").parse(
        "pkg/__init__.py", "from .sub import thing\n")
    assert parsed.module == "pkg"
    assert parsed.symbols[0].references == ("pkg.sub",)


def test_methods_are_typed_and_qualified_by_their_class() -> None:
    by_name = {s.name: s for s in parse().symbols}
    assert by_name["Scorer"].kind == CLASS
    assert by_name["fit"].kind == METHOD
    assert by_name["fit"].qualname == "src.inject.link.Scorer.fit"
    assert by_name["helper"].kind == FUNCTION
    assert by_name["run"].kind == FUNCTION
    assert by_name["Scorer"].children == ("src.inject.link.Scorer.fit",
                                          "src.inject.link.Scorer.score")


def test_self_calls_resolve_only_to_methods_the_class_defines() -> None:
    fit = next(s for s in parse().symbols if s.name == "fit")
    assert "src.inject.link.Scorer.score" in fit.references
    assert not any("unknown_attr" in r for r in fit.references)


def test_calls_on_locals_and_builtins_are_dropped() -> None:
    """``cur.execute`` in every database file would be one meaningless hub."""
    run = next(s for s in parse().symbols if s.name == "run")
    assert run.references == ("src.inject.link.helper",)
    score = next(s for s in parse().symbols if s.name == "score")
    assert not any(r == "len" for r in score.references)


def test_aliased_third_party_calls_are_qualified() -> None:
    fit = next(s for s in parse().symbols if s.name == "fit")
    assert "numpy.mean" in fit.references


def test_bases_are_references_of_the_class() -> None:
    cls = next(s for s in parse().symbols if s.name == "Scorer")
    assert cls.references == ("src.inject.episodes.Base",)
    assert cls.signature == "class Scorer(EP.Base)"


def test_signatures_keep_annotations_and_async() -> None:
    by_name = {s.name: s for s in parse().symbols}
    assert by_name["run"].signature == "async def run()"
    assert by_name["helper"].signature == "def helper(x)"


def test_a_syntax_error_is_the_callers_problem() -> None:
    import pytest

    with pytest.raises(SyntaxError):
        parser_for("a.py").parse("a.py", "def (:\n")


def test_unknown_extensions_have_no_parser() -> None:
    assert parser_for("README.md") is None
    assert parser_for("x.PY") is not None
