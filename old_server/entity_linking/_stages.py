"""Import numbered stage modules, whose filenames are not valid identifiers.

The pipeline stages are named ``1_episodes.py``, ``2_chains.py`` and so on so
that reading the directory tells you the order things run. Python cannot import
a module whose name starts with a digit, so they are loaded by path here and
re-exported from ``__init__`` under ordinary names — ``entity_linking.episodes``
and friends keep working, and nothing outside this file deals with the numbers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HERE = Path(__file__).resolve().parent


def load(filename: str, alias: str) -> ModuleType:
    """Load ``<filename>.py`` from this package and register it as ``alias``."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _HERE / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module          # registered before exec, so the stages
    spec.loader.exec_module(module)      # can import one another by alias
    return module
