"""The kit: everything that decides whether a run passes, except the endpoint.

Every agent arm before the kit ran on a slightly different instrument -- a
grader fixed, a void rule added, the thinking level reaching the server or not
-- and nothing in the results said which, so results from different
instruments ended up in one table. The kit gives each result an identity:

* per task, a digest of its prompt, its fixture as built (every file, its
  executable bit, the refs, the local git config, SQLite by logical dump since
  its bytes depend on the library version) and its grader's source;
* one digest of the runner: the run, scheduling, void, sandbox and harness code
  in ``agent.py``, the endpoint-identity rule, and the whole proxy, which decide a
  verdict as much as a grader.

Source is digested with comments and docstrings removed, so rewording prose
does not move it and changing code does. A test pins every digest: a change to
the kit fails it until ``KIT`` is bumped and the pin updated, so the change is
a decision recorded in a commit rather than a side effect. Per-task digests let
a comparison keep the tasks two results share when one task has changed.

Arm settings -- thinking, sampling, timeout, workers -- are not in the kit:
they are what an experiment varies, recorded on the result and compared there.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import tokenize
from typing import Any, Dict, Iterable, List, Tuple

from ._base import Task, _SKIP_DIRS

#: Bump when a pinned digest changes, and say why in the commit.
KIT = "agent-kit-14"


def _strip(source: str) -> str:
    """The source as tokens, without comments, blank lines or docstrings."""
    source = textwrap.dedent(source)
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add((body[0].lineno, body[0].col_offset))
    # Spans cut out of the text, rather than a token stream hashed: Python 3.12
    # tokenises f-strings differently, and CI runs 3.10 to 3.12.
    lines = source.splitlines()
    cuts: List[Tuple[int, int, int, int]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docstrings):
            cuts.append((*tok.start, *tok.end))
    for (r0, c0, r1, c1) in sorted(cuts, reverse=True):
        head, tail = lines[r0 - 1][:c0], lines[r1 - 1][c1:]
        lines[r0 - 1:r1] = [head + tail]
    return "\n".join(line.rstrip() for line in lines if line.strip())


def source_digest(objects: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for obj in objects:
        h.update(_strip(inspect.getsource(obj)).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _git(ws: str, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=ws, env=env, capture_output=True,
                          text=True).stdout


def _db_dump(path: str) -> str:
    """Schema and rows, in a form no Python version changes: iterdump's moved in 3.13."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out = []
        for name, sql in db.execute("SELECT name, sql FROM sqlite_master "
                                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"):
            out.append(str(sql))
            if sql and sql.upper().startswith("CREATE TABLE"):
                rows = db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
                out += [json.dumps([v.hex() if isinstance(v, bytes) else v for v in row])
                        for row in rows]
        return "\n".join(out)
    finally:
        db.close()


def _history(ws: str) -> str:
    """Commits, trees and refs without a single hash in them.

    A committed SQLite file's bytes depend on the library version, and every
    commit hash after it with them, so hashes would make the digest a property
    of the machine. Commits are named by their position instead, and a
    database's blob by its path (its content is in the working-tree dump).
    """
    order = _git(ws, "rev-list", "--reverse", "--topo-order", "--all").split()
    index = {c: str(i) for i, c in enumerate(order)}
    out = []
    for c in order:
        meta = _git(ws, "show", "-s", "--format=%P%n%an <%ae> %ad%n%cn <%ce> %cd%n%B", c)
        parents, _, rest = meta.partition("\n")
        out.append(" ".join(index.get(p, "?") for p in parents.split()) + "\n" + rest)
        for entry in _git(ws, "ls-tree", "-r", c).splitlines():
            info, _, path = entry.partition("\t")
            mode, kind, blob = info.split()
            if path.endswith((".db", ".sqlite", ".sqlite3")):
                blob = "database"
            out.append(f"{mode} {kind} {blob} {path}")
    for ref in _git(ws, "for-each-ref", "--format=%(refname) %(objectname)").splitlines():
        name, _, target = ref.partition(" ")
        out.append(f"{name} -> {index.get(target, '?')}")
    out.append("HEAD -> " + index.get(_git(ws, "rev-parse", "HEAD").strip(), "?"))
    out.append(_git(ws, "symbolic-ref", "-q", "HEAD"))
    return "\n".join(out)


def fixture_digest(task: Task) -> str:
    """The fixture as the agent first sees it."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = os.path.join(tmp, task.project)
        os.makedirs(ws)
        task.setup(ws)
        h = hashlib.sha256()
        h.update(_history(ws).encode())
        h.update(_git(ws, "config", "--local", "--list").encode())
        paths = []
        # Pruned as it walks: sorting the walk first would read .git/index too,
        # whose timestamps differ every build.
        for d, dirs, files in os.walk(ws):
            dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
            paths += [os.path.join(d, f) for f in files]
        for p in sorted(paths):
            rel = os.path.relpath(p, ws)
            if os.path.islink(p):
                h.update(f"link {rel} {os.readlink(p)}\n".encode())
                continue
            h.update(f"{rel} {os.stat(p).st_mode & 0o111:o}\n".encode())
            if p.endswith((".db", ".sqlite", ".sqlite3")):
                h.update(_db_dump(p).encode())
            else:
                with open(p, "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:16]


_DIGESTS: Dict[str, str] = {}


def task_digest(task: Task) -> str:
    """Prompt, fixture and grader; built once per process, since it runs the setup."""
    if task.name in _DIGESTS:
        return _DIGESTS[task.name]
    grader = sys.modules[task.grade.__module__]
    from . import _base
    h = hashlib.sha256(json.dumps([task.name, task.project, task.prompt]).encode())
    h.update(fixture_digest(task).encode())
    h.update(source_digest([grader, _base]).encode())
    _DIGESTS[task.name] = h.hexdigest()[:16]
    return _DIGESTS[task.name]


def runner_digest() -> str:
    # run() and run_arms() schedule the runs (order, workers, turns on a shared
    # endpoint, the proxy, a mismatch stopping the arm), and the identity rule
    # voids them, so they decide verdicts as much as run_one does.
    from .. import agent, agent_proxy, servermetrics
    return source_digest([agent._looping, agent.Pi, agent.Hermes, agent.OpenCode, agent.Omp,
                          agent.sandbox, agent.sandbox_selftest, agent._base_env, agent._unlock,
                          agent._execute, agent.run_one, agent.summarise, agent.run,
                          agent.run_arms, servermetrics.endpoint_identity,
                          servermetrics.identity_drift, agent_proxy])


def kit(tasks: Iterable[Task]) -> Dict[str, Any]:
    """What a saved result records about the instrument it was measured with."""
    return {"name": KIT, "runner": runner_digest(),
            "tasks": {t.name: task_digest(t) for t in tasks}}
