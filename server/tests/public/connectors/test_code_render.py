"""One paragraph per symbol, mentions point at their text, nothing positional."""

from src.connectors.code import parser_for, render
from src.inject import episodes as EP

SRC = '''"""Module doc.

Second paragraph of the module doc."""
from src.inject import episodes as EP

def a():
    """Does a."""
    return EP.sync_file()

class K:
    def m(self):
        return a()
'''


def rendered(src: str = SRC, path: str = "pkg/mod.py"):
    return render(parser_for(path).parse(path, src), path=path)


def test_every_mention_is_findable_at_its_offsets() -> None:
    text, mentions = rendered()
    assert mentions
    for m in mentions:
        assert text[m.start:m.end] == m.text


def test_one_paragraph_per_symbol_plus_docstring_paragraphs() -> None:
    text, _ = rendered()
    heads = [p.split("\n")[0] for p in text.split("\n\n")]
    assert heads[0] == "module mod (pkg.mod), file mod.py at pkg/mod.py"
    assert "function a (pkg.mod.a)" in heads
    assert "class K (pkg.mod.K)" in heads
    assert "method m (pkg.mod.K.m)" in heads


def test_the_text_carries_no_line_numbers() -> None:
    """A line number changes on every edit above it and would churn the hashes."""
    text, _ = rendered()
    assert "line" not in text.lower()


def test_short_and_qualified_names_share_one_key() -> None:
    _, mentions = rendered()
    keys = {m.text: m.key for m in mentions if m.text in ("a", "pkg.mod.a")}
    assert keys == {"a": "pkg.mod.a", "pkg.mod.a": "pkg.mod.a"}


def test_references_render_short_but_keep_their_qualified_key() -> None:
    text, mentions = rendered()
    call = next(m for m in mentions if m.key == "src.inject.episodes.sync_file")
    assert call.text == "sync_file"
    assert "calls: sync_file" in text


def test_module_imports_render_by_dotted_path() -> None:
    text, mentions = rendered()
    assert "imports: src.inject" in text
    assert any(m.text == "src.inject" and m.key == "src.inject" for m in mentions)


def test_appending_a_function_changes_the_index_and_the_tail_only() -> None:
    """The module paragraph lists what the file defines, so it re-extracts; the
    symbols between it and the new one are byte-identical and do not."""
    before, _ = rendered()
    after, _ = rendered(SRC + "\ndef z():\n    return 1\n")
    old = [e.text for e in EP.split_episodes(before, max_chars=80)]
    new = [e.text for e in EP.split_episodes(after, max_chars=80)]
    assert old[0] != new[0] and "z" in new[0]
    assert new[1:len(old) - 1] == old[1:-1]
    assert len(old) > 3


def test_rendering_is_deterministic() -> None:
    assert rendered() == rendered()


def test_ambiguous_references_render_qualified_but_definitions_keep_both() -> None:
    from src.connectors.code import ambiguous_names

    other = parser_for("pkg/b.py").parse("pkg/b.py", "def a():\n    return 2\n")
    mine = parser_for("pkg/mod.py").parse("pkg/mod.py", SRC)
    ambiguous = ambiguous_names([mine, other])
    assert ambiguous == frozenset({"a"})
    text, mentions = render(mine, path="pkg/mod.py", ambiguous=ambiguous)
    assert "calls: pkg.mod.a" in text                     # the reference, qualified
    assert "function a (pkg.mod.a)" in text               # the definition, both forms
    assert all(text[m.start:m.end] == m.text for m in mentions)
    assert not any(m.text == "a" for m in mentions)       # bare name is plain text
    from src.connectors.code import Style

    text, mentions = render(mine, path="pkg/mod.py",
                            style=Style(ambiguous=ambiguous,
                                        short_ambiguous_definitions=True))
    assert any(m.text == "a" and m.key == "pkg.mod.a" for m in mentions)


def test_init_modules_do_not_mention_their_filename() -> None:
    """Every package has an __init__.py; as a surface it would alias them all."""
    parsed = parser_for("pkg/__init__.py").parse("pkg/__init__.py", "")
    text, mentions = render(parsed, path="pkg/__init__.py")
    assert text == "module pkg (pkg), package at pkg/__init__.py"
    assert all(m.text != "__init__.py" for m in mentions)


def test_external_calls_that_clash_with_a_repo_name_are_ambiguous() -> None:
    from src.connectors.code import ambiguous_names

    src = "import numpy as np\n\ndef load():\n    return np.load('x')\n"
    parsed = parser_for("pkg/m.py").parse("pkg/m.py", src)
    assert ambiguous_names([parsed]) == frozenset({"load"})


def test_a_reference_through_an_unresolvable_re_export_is_not_a_second_thing() -> None:
    """``PR.keys_settle`` via the loader alias must not make ``keys_settle`` ambiguous."""
    from src.connectors.code import ambiguous_from, names_of

    tree = frozenset({"src/__init__.py", "src/inject/__init__.py",
                      "src/inject/main.py", "src/inject/src/4_link.py"})
    stage = parser_for("src/inject/src/4_link.py").parse(
        "src/inject/src/4_link.py", "def keys_settle(a, b):\n    return a != b\n", tree=tree)
    caller = parser_for("src/inject/main.py").parse(
        "src/inject/main.py",
        "from src.inject import pairing as PR\n\ndef link():\n    return PR.keys_settle(1, 2)\n",
        tree=tree)
    init = parser_for("src/inject/__init__.py").parse("src/inject/__init__.py", "", tree=tree)
    maps = [names_of(stage), names_of(caller), names_of(init, package=True)]
    assert stage.symbols[1].qualname == "4_link.keys_settle"
    assert "src.inject.pairing.keys_settle" in caller.symbols[1].references
    assert ambiguous_from(maps) == frozenset()
    # a member of a module file is resolvable, so it still counts
    method_caller = parser_for("src/inject/m.py").parse(
        "src/inject/m.py",
        "from src.inject import main\n\ndef keys_settle():\n    return main.Thing.keys_settle()\n",
        tree=tree)
    assert "keys_settle" in ambiguous_from(maps + [names_of(method_caller)])
    # but a real second definition, or an external call, still counts
    other = parser_for("src/inject/other.py").parse(
        "src/inject/other.py", "def keys_settle():\n    pass\n", tree=tree | {"src/inject/other.py"})
    assert "keys_settle" in ambiguous_from([names_of(stage), names_of(other)])
    ext = parser_for("src/inject/e.py").parse(
        "src/inject/e.py", "import numpy as np\n\ndef load():\n    return np.load(1)\n", tree=tree)
    assert "load" in ambiguous_from([names_of(ext)])
