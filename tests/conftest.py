"""Shared test setup.

`pennsieve_llm` is an internal SDK that isn't installed in every dev env, and
importing `scout.llm` (directly or via `scout.pipeline`) requires it. Stub it
here so it's in place before any test module is imported, rather than relying
on whichever module happens to be collected first.
"""
from __future__ import annotations

import sys
import types

if "pennsieve_llm" not in sys.modules:  # pragma: no cover - env dependent
    _stub = types.ModuleType("pennsieve_llm")
    _stub.Governor = object
    sys.modules["pennsieve_llm"] = _stub
