#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tokenizers",
#     "transformers>=5.2.0",
#     "tabulate",
#     "numpy",
#     "requests",
#     "aiohttp",
#     "pydantic>=2.12.5",
# ]
# ///
#
# The dependency list above is PEP 723 inline script metadata: it lets
# `uv run llm-assay.py` resolve and cache an environment with no setup step.
# It must stay in sync with requirements.txt (for pip users); a test enforces
# that they do not drift.
"""Entry point for llm-assay.

This project is run from a checkout; there is no packaging or install step.
This script puts ``src/`` on the import path and hands off to the package.

    pip install -r requirements.txt
    python llm-assay.py --base-url http://localhost:8000/v1
    python llm-assay.py tune http://localhost:8000/v1
    python llm-assay.py compare baseline.json candidate.json
    python llm-assay.py probe http://localhost:8000/v1 --model my-model

``tune``, ``compare`` and ``probe`` are exposed as subcommands so the checkout needs exactly
one script. Any other first argument is passed through to the benchmark
unchanged, so every flag documented in the README works here.
"""

from __future__ import annotations

import os
import sys

MIN_PYTHON = (3, 10)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")


def _fail(message: str, hint: str = "") -> "None":
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    raise SystemExit(1)


def _check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        _fail(
            f"llm-assay needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
            f"but this is {sys.version.split()[0]}."
        )


def _check_layout() -> None:
    if not os.path.isdir(os.path.join(SRC, "llm_assay")):
        _fail(
            f"could not find the package at {SRC}/llm_assay",
            "Run this script from inside the llm-assay checkout.",
        )


def _import_or_explain(name: str):
    try:
        return __import__(f"llm_assay.{name}", fromlist=["main"])
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        _fail(
            f"a required dependency is missing ({missing}).",
            "Install the runtime dependencies first:\n"
            "    pip install -r requirements.txt",
        )


def main() -> int:
    _check_python()
    _check_layout()
    # Prepend so this checkout always wins over any stray copy on the path.
    sys.path.insert(0, SRC)

    if len(sys.argv) > 1 and sys.argv[1] in ("compare", "tune", "probe"):
        name = sys.argv[1]
        module = _import_or_explain(name)
        sys.argv = [f"{sys.argv[0]} {name}", *sys.argv[2:]]
        return module.main()

    module = _import_or_explain("__main__")
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
