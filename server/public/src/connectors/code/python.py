"""Python source -> symbols, on the standard library's ``ast``.

No grammar to install and no version to pin: the interpreter that runs the
ingest is the parser. The cost is that a file the running Python cannot parse
(Python 2, a newer syntax) raises ``SyntaxError`` and is skipped; the
connector counts those rather than failing the repo.

Bodies are read only for the names they call, and only calls the file can
resolve are kept: an import, a definition in the same module, or ``self.m``
where ``m`` is defined on the class. ``cur.execute`` on a local variable would
need type inference to mean anything, and kept as written it is a surface
named "execute" in every file that touches a database. Builtins and the
standard library are dropped for the same reason -- ``len`` and ``json`` would
be the biggest hubs in the graph and say nothing about the codebase.
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import PurePosixPath

from src.connectors.code.model import (
    CLASS,
    FUNCTION,
    METHOD,
    MODULE,
    Parsed,
    Symbol,
)

MAX_SIGNATURE = 300
_BUILTINS = frozenset(dir(builtins))
_STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class PythonParser:
    extensions = frozenset({".py"})

    def module_name(self, path: str, tree: frozenset[str] = frozenset()) -> str:
        """The importable name: ``src/pkg/mod.py`` -> ``pkg.mod`` when ``src`` is
        a layout directory, ``src.pkg.mod`` when ``src/__init__.py`` exists.

        The import root is found by walking up from the file while the
        directory has an ``__init__.py`` in ``tree``. A wrong prefix would make
        every qualified name a string nobody types, and the qualified name is
        what keeps two ``run``s apart. With no tree to consult, every directory
        is taken to be a package -- the reading that keeps names distinct.
        """
        parts = list(PurePosixPath(path).with_suffix("").parts)
        dirs, stem = parts[:-1], parts[-1]
        depth = len(dirs) if tree else 0
        while depth > 0 and "/".join([*dirs[:depth], "__init__.py"]) in tree:
            depth -= 1
        package = dirs[depth:]
        return ".".join(package if stem == "__init__" else [*package, stem])

    def parse(self, path: str, text: str, *,
              tree: frozenset[str] = frozenset()) -> Parsed:
        module = self.module_name(path, tree)
        is_package = PurePosixPath(path).stem == "__init__"
        package = module if is_package else module.rpartition(".")[0]
        tree = ast.parse(text)
        top = {node.name: f"{module}.{node.name}"
               for node in tree.body if isinstance(node, _DEFS)}
        scope = {**_imports(tree, package), **top}
        symbols = [Symbol(
            kind=MODULE,
            name=module.rpartition(".")[2] or module,
            qualname=module,
            signature="",
            doc=ast.get_docstring(tree) or "",
            references=_unique([m for m in _imported_modules(tree, package)
                                if not _is_stdlib(m)]),
            children=tuple(top.values()),
        )]
        symbols.extend(_walk(tree.body, module, scope, in_class=False))
        return Parsed(module=module, symbols=tuple(symbols))


def _walk(body: list[ast.stmt], owner: str, scope: dict[str, str], *,
          in_class: bool, members: dict[str, str] | None = None) -> list[Symbol]:
    """Definitions in ``body``, depth first, in source order; ``owner`` qualifies them.

    ``members`` are the owner's own definitions when it is a class, so that
    ``self.m`` resolves to a method that exists rather than to any attribute.
    """
    symbols: list[Symbol] = []
    for node in body:
        if not isinstance(node, _DEFS):
            continue
        qualname = f"{owner}.{node.name}"
        inner = {n.name: f"{qualname}.{n.name}" for n in node.body
                 if isinstance(n, _DEFS)}
        local = {**scope, **inner}
        decorators = [_dotted(d) for d in node.decorator_list]
        if isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                kind=CLASS, name=node.name, qualname=qualname,
                signature=_class_signature(node),
                doc=ast.get_docstring(node) or "",
                references=_resolve(
                    [_dotted(b) for b in node.bases] + decorators, local,
                    cls=None, members=inner),
                children=tuple(inner.values()),
            ))
            symbols.extend(_walk(node.body, qualname, local, in_class=True,
                                 members=inner))
            continue
        symbols.append(Symbol(
            kind=METHOD if in_class else FUNCTION,
            name=node.name, qualname=qualname,
            signature=_def_signature(node),
            doc=ast.get_docstring(node) or "",
            references=_resolve(decorators + _calls(node), local,
                                cls=owner if in_class else None,
                                members=members if in_class else {}),
            children=tuple(inner.values()),
        ))
        symbols.extend(_walk(node.body, qualname, local, in_class=False))
    return symbols


def _calls(node: ast.AST) -> list[str]:
    """Dotted names called in ``node``'s body, not descending into nested defs."""
    found: list[str] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop(0)
        if isinstance(child, _DEFS):
            continue
        if isinstance(child, ast.Call):
            found.append(_dotted(child.func))
        stack.extend(ast.iter_child_nodes(child))
    return [f for f in found if f]


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for a Name/Attribute chain; "" for anything else (``f()()``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _resolve(names: list[str], scope: dict[str, str], *, cls: str | None,
             members: dict[str, str]) -> tuple[str, ...]:
    """Qualify what the file can; drop what it cannot, and the standard library."""
    out: list[str] = []
    for name in names:
        head, dot, rest = name.partition(".")
        if not head or head in _BUILTINS:
            continue
        if head in ("self", "cls"):
            if cls and rest in members:
                out.append(members[rest])
            continue
        if head in scope and not _is_stdlib(scope[head]):
            out.append(scope[head] + dot + rest)
    return _unique(out)


def _is_stdlib(qualname: str) -> bool:
    return qualname.partition(".")[0] in _STDLIB


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _imports(tree: ast.Module, package: str) -> dict[str, str]:
    """Local alias -> qualified name, for every import at module level."""
    scope: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    scope[alias.asname] = alias.name
                else:  # `import a.b.c` binds `a`
                    scope[alias.name.partition(".")[0]] = alias.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = _relative(package, node.level, node.module)
            for alias in node.names:
                if alias.name != "*":
                    scope[alias.asname or alias.name] = f"{base}.{alias.name}"
    return scope


def _imported_modules(tree: ast.Module, package: str) -> list[str]:
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(_relative(package, node.level, node.module))
    return [m for m in modules if m]


def _relative(package: str, level: int, module: str | None) -> str:
    """``from ..x import y`` inside package ``a.b.c`` -> ``a.b.x``."""
    if level == 0:
        return module or ""
    parts = package.split(".") if package else []
    base = parts[:len(parts) - (level - 1)] if level > 1 else parts
    return ".".join(p for p in [*base, module or ""] if p)


def _def_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return _cap(f"{keyword} {node.name}({ast.unparse(node.args)}){returns}")


def _class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    return _cap(f"class {node.name}({bases})" if bases else f"class {node.name}")


def _cap(text: str) -> str:
    return text if len(text) <= MAX_SIGNATURE else text[:MAX_SIGNATURE - 1] + "…"
