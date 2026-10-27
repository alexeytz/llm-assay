"""What every agent task is built from: a fixture writer and a grading context.

A task is a fixture -- a small git repository written into an empty directory
-- one prompt, and a grader that reads the workspace the agent left. Critical
checks decide pass or fail; soft checks are recorded beside them.

Fixtures commit with a fixed author and fixed dates, one day apart, so every
run starts from the same history: a fixture that differs between runs is a
second variable nobody chose. (A SQLite file's bytes depend on the library
version that wrote it, so "the same" means the same contents, not the same
blob.)

The grader runs code the agent wrote -- the project's tests, its CLI -- so
every command it runs goes through ``Run.sh``, which the caller wraps in the
same sandbox the agent had.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

AUTHOR = {"GIT_AUTHOR_NAME": "Sam Rivera", "GIT_AUTHOR_EMAIL": "sam@example.org",
          "GIT_COMMITTER_NAME": "Sam Rivera", "GIT_COMMITTER_EMAIL": "sam@example.org"}


class Fixture:
    """A repository under construction in ``ws``. Each ``commit`` is one day later."""

    def __init__(self, ws: str):
        self.ws = ws
        self.day = 0
        self.git("init", "-q", "-b", "main", ".")
        self.git("config", "user.name", AUTHOR["GIT_AUTHOR_NAME"])
        self.git("config", "user.email", AUTHOR["GIT_AUTHOR_EMAIL"])
        self.git("config", "commit.gpgsign", "false")

    def path(self, rel: str) -> str:
        return os.path.join(self.ws, rel)

    def write(self, rel: str, text: str, mode: str = "w") -> None:
        path = self.path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode) as fh:
            fh.write(text)

    def append(self, rel: str, text: str) -> None:
        self.write(rel, text, "a")

    def mkdir(self, rel: str) -> None:
        os.makedirs(self.path(rel), exist_ok=True)

    def git(self, *args: str, env: Optional[Dict[str, str]] = None) -> str:
        # Under a git hook GIT_INDEX_FILE/GIT_DIR point at the caller's repository,
        # and an inherited one makes `add -A` here stage the fixture into it.
        base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        r = subprocess.run(["git", *args], cwd=self.ws, env={**base, **AUTHOR, **(env or {})},
                           check=True, capture_output=True, text=True)
        return r.stdout

    def commit(self, message: str) -> None:
        """Stage everything and commit it, dated one day after the last commit."""
        self.day += 1
        stamp = f"2026-03-{self.day:02d}T10:00:00+00:00"
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message,
                 env={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})


@dataclass
class Check:
    name: str
    ok: bool
    critical: bool = True
    note: str = ""


@dataclass
class Task:
    name: str
    #: What the agent's working directory is called. A real project's name,
    #: never the task's or the tool's: a path that says "eval" tells the agent
    #: it is being tested.
    project: str
    prompt: str
    setup: Callable[[str], None]
    grade: Callable[["Run"], List[Check]]
    #: One line: what the trap is.
    trap: str = ""


_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
_TEST_CMD = re.compile(r"\bunittest\b|\bpytest\b|run_tests")


def hashes(root: str) -> Dict[str, str]:
    """Every regular file under ``root`` (outside .git and caches) by content hash."""
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


Wrap = Callable[[List[str]], List[str]]


class Run:
    """What a grader reads: the workspace, the reply, the trace, the starting state.

    ``trace`` is the harness's events in one shape -- ``{"role": "call",
    "tool": ..., "args": {...}}`` for a tool call, with the tool named as
    ``terminal``, ``patch``, ``write_file`` or ``read_file`` -- so a grader
    does not know which harness ran. ``baseline`` is ``hashes(ws)`` taken
    right after setup.
    """

    def __init__(self, ws: str, reply: str, trace: List[Dict[str, Any]],
                 baseline: Dict[str, str], wrap: Optional[Wrap] = None,
                 env: Optional[Dict[str, str]] = None):
        self.ws = os.path.abspath(ws)
        self.reply = reply
        self.trace = trace
        self.baseline = baseline
        self.wrap = wrap or (lambda argv: argv)
        #: The environment the agent's code sees; None inherits ours (tests only).
        self.env = env

    # --- what the agent did, in order ---------------------------------------

    def events(self) -> List[Tuple[str, str]]:
        """Ordered ('cmd', text) and ('edit', relpath) events from the tool calls."""
        ev: List[Tuple[str, str]] = []
        for e in self.trace:
            if e.get("role") != "call":
                continue
            a = e.get("args") or {}
            if not isinstance(a, dict):
                a = {"raw": str(a)}
            if e.get("tool") in ("patch", "write_file"):
                p = a.get("path", "")
                ev.append(("edit", os.path.relpath(os.path.join(self.ws, p), self.ws) if p else ""))
            for k in ("command", "code", "raw"):
                if isinstance(a.get(k), str):
                    ev.append(("cmd", a[k]))
        return ev

    def commands(self) -> List[str]:
        return [t for k, t in self.events() if k == "cmd"]

    def ran_tests(self) -> bool:
        return any(_TEST_CMD.search(c) for c in self.commands())

    # --- the end state ------------------------------------------------------

    def sh(self, cmd: str, timeout: float = 180) -> Tuple[int, str]:
        """A shell command in the workspace, in the agent's sandbox: it runs the agent's code."""
        try:
            r = subprocess.run(self.wrap(["bash", "-c", cmd]), cwd=self.ws, capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                               env=self.env)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, "timeout"

    def tests_pass(self) -> Tuple[bool, str]:
        rc, out = self.sh("python3 -m unittest discover -s tests -t . 2>&1")
        return rc == 0, out[-300:]

    def changes(self) -> Tuple[List[str], List[str], List[str]]:
        """(added, modified, deleted) against the baseline."""
        now = hashes(self.ws) if os.path.isdir(self.ws) else {}
        added = sorted(set(now) - set(self.baseline))
        deleted = sorted(set(self.baseline) - set(now))
        modified = sorted(p for p in now if p in self.baseline and now[p] != self.baseline[p])
        return added, modified, deleted

    def unchanged(self, prefix: str) -> bool:
        """Every baseline file under ``prefix`` present and byte-identical, none added."""
        now = hashes(self.ws)
        return ({p: h for p, h in self.baseline.items() if p.startswith(prefix)}
                == {p: h for p, h in now.items() if p.startswith(prefix)})

    def written_text(self) -> str:
        """The reply plus every added or modified text file: where a finding could be written."""
        added, modified, _ = self.changes()
        parts = [self.reply]
        for p in added + modified:
            if p.endswith((".md", ".txt", ".rst")):
                with open(os.path.join(self.ws, p), errors="replace") as fh:
                    parts.append(fh.read())
        return "\n".join(parts)

    def git_all_patches(self) -> str:
        return self.sh("git log --all -p --format=%H 2>/dev/null")[1]

    def read(self, rel: str) -> Optional[str]:
        """A workspace file's text, or None if the agent removed it."""
        try:
            with open(os.path.join(self.ws, rel), errors="replace") as fh:
                return fh.read()
        except OSError:
            return None
