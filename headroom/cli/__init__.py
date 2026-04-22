"""Headroom CLI - Command-line interface for memory and proxy management."""

import sys
from importlib import import_module

from .main import main

_LAZY_SUBMODULES = {
    "evals",
    "init",
    "install",
    "learn",
    "mcp",
    "memory",
    "perf",
    "proxy",
    "tools",
    "wrap",
}

__all__ = ["main"]

for _name in _LAZY_SUBMODULES:
    _module = sys.modules.get(f"{__name__}.{_name}")
    if _module is not None:
        globals()[_name] = _module


def __getattr__(name: str) -> object:
    if name == "__path__":
        raise AttributeError(name)

    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)
