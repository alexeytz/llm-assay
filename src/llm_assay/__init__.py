"""
llm-assay - llama-bench style benchmarking tool for all backends

This package provides a benchmarking tool for OpenAI-compatible LLM endpoints,
generating statistics similar to `llama-bench`.
"""

import os
import subprocess

__all__ = ["__version__"]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_revision() -> str:
    """Short commit id of the checkout, plus '.dirty' for uncommitted changes.

    Benchmark results record the tool version, so a run stays traceable to the
    exact code that produced it. Returns "" when git is unavailable or this is
    not a checkout (e.g. an extracted archive).
    """
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", _ROOT, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()

    try:
        revision = _git("rev-parse", "--short", "HEAD")
    except Exception:
        return ""
    try:
        if _git("status", "--porcelain", "--untracked-files=no"):
            revision += ".dirty"
    except Exception:
        pass
    return revision


def _read_version() -> str:
    base = "0.0.0"
    try:
        with open(os.path.join(_ROOT, "VERSION")) as handle:
            base = handle.read().strip() or base
    except OSError:
        pass
    revision = _git_revision()
    return f"{base}+g{revision}" if revision else base


__version__ = _read_version()
