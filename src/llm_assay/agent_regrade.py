"""``agent regrade``: grade a saved result's kept workspaces again, under today's graders.

A grader change moves its task's digest, and ``compare`` then refuses every
result measured before it -- rightly, since a verdict is the grader's as much as
the model's. When the workspaces were kept, the older runs can be graded again
instead of run again. The kept tree is the one the original grader left, and
graders run the agent's code, so a regrade proves itself before it rewrites
anything: every run of a task whose digest has *not* changed must reproduce
its stored verdict, check by critical check. One disagreement and nothing is
written.

What is rewritten is only what grading decides -- a changed task's checks and
pass -- and the task's digest in the result's kit, so ``compare`` pairs it
again. Voids, timeouts and the runner digest stay as they were measured, and
the result says what was regraded, from what, and how well the rest agreed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .agent_tasks import TASKS, Run, Task, hashes
from .agent_tasks.kit import KIT, task_digest


#: Digests a task's grader alone moved from: prompt and fixture byte-identical, so a
#: kept workspace is the same task's and only the verdict can differ. A digest that
#: moved for any other reason -- red-test's kit-2 fixture, say -- is a different task,
#: and no regrade makes it comparable. Add a row with each grader-only bump.
GRADER_ONLY = {
    # agent-kit-12: the lead test must survive (flake); every live SKU, not a count (sync).
    "flake": {"45025abe1e6d45ea"},
    "sync": {"26a1982af266da06"},
    "sync-quiet": {"465b1af74dd5fe21"},
}


def _arms(result: Dict[str, Any]) -> Dict[Optional[str], Dict[str, Any]]:
    """A models-file result's arms by name; a single-model result as one arm."""
    models = result.get("models")
    if isinstance(models, dict):
        return dict(models)
    return {None: result}


def locate(root: str) -> Dict[str, str]:
    """Each kept run directory under ``root``, by its name (the run's id)."""
    found: Dict[str, str] = {}
    for d, dirs, files in os.walk(root):
        if "trace.jsonl" in files:
            found[os.path.basename(d)] = d
            dirs[:] = []
    return found


def _baseline(task: Task, scratch: str) -> Dict[str, str]:
    """The untouched fixture's hashes, as ``run_one`` took them before the agent ran."""
    ws = os.path.join(tempfile.mkdtemp(dir=scratch), task.project)
    os.makedirs(ws)
    task.setup(ws)
    return hashes(ws)


def _seen(rundir: str, harness: Optional[str], stored_reply: str
          ) -> Tuple[List[Dict[str, Any]], str]:
    """The trace and reply the original grader saw.

    Parsed again from the harness's own events, as ``run_one`` did: the result
    keeps only a reply's last 3,000 characters, and review's grader reads the
    whole reply -- two Flash-Next review runs lost a finding to the cut. The
    kept trace and the stored reply are the fallback when no events were kept.
    """
    from .agent import HARNESSES
    events = os.path.join(rundir, "events.jsonl")
    # No name: a result from before the second harness, when every arm was Pi's.
    name = harness.split()[0] if harness else "pi"
    if name in HARNESSES and os.path.isfile(events):
        with open(events, errors="replace") as fh:
            parse: Any = getattr(HARNESSES[name], "parse")
            trace, reply, _ = parse(fh.read())
        return trace, reply
    with open(os.path.join(rundir, "trace.jsonl")) as fh:
        return [json.loads(line) for line in fh if line.strip()], stored_reply


def grade_again(task: Task, rundir: str, harness: Optional[str], stored_reply: str,
                baseline: Dict[str, str], sandboxed: bool,
                memory: Optional[str]) -> List[Dict[str, Any]]:
    """The task's checks on a kept run, in the run's own sandbox, as ``run_one`` grades."""
    from .agent import _base_env, sandbox
    trace, reply = _seen(rundir, harness, stored_reply)

    def wrap(cmd: List[str]) -> List[str]:
        return sandbox(cmd, rw=[rundir], ro=[], memory=memory) if sandboxed else cmd
    checks = task.grade(Run(os.path.join(rundir, task.project), reply, trace, baseline, wrap,
                            _base_env(os.path.join(rundir, "home"))))
    return [c.__dict__ for c in checks]


def _critical(checks: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {c["name"]: bool(c["ok"]) for c in checks if c.get("critical")}


def regrade(result: Dict[str, Any], root: str, scratch: str, sandboxed: bool = True,
            memory: Optional[str] = "1500M", workers: int = 4,
            only: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """(new result, disagreements, per-arm account). Nothing changes when there are disagreements."""
    out = copy.deepcopy(result)
    kept = locate(root)
    disagree: List[str] = []
    account: Dict[str, Any] = {}
    baselines: Dict[str, Dict[str, str]] = {}
    for name, arm in _arms(out).items():
        # A copy: the kit's own mapping is rewritten below, and "from" must keep the old.
        old_digests = dict((arm.get("kit") or {}).get("tasks") or {})
        moved_digest = sorted(t for t, d in old_digests.items()
                              if t in TASKS and task_digest(TASKS[t]) != d)
        changed = [t for t in moved_digest
                   if old_digests[t] in GRADER_ONLY.get(t, ()) and (only is None or t in only)]
        # Moved for more than its grader: left as measured, and compare keeps refusing it.
        skipped = [t for t in moved_digest if t not in changed]
        jobs = []
        for i, r in enumerate(arm["runs"]):
            if r.get("void") or r["task"] not in TASKS or r["task"] in skipped:
                continue
            rundir = kept.get(os.path.basename(r.get("kept") or ""))
            if rundir is None:
                disagree.append(f"{name}: run {i} ({r['task']}) has no kept workspace under {root}")
                continue
            if r["task"] not in baselines:
                baselines[r["task"]] = _baseline(TASKS[r["task"]], scratch)
            jobs.append((i, r, rundir))

        def one(job: Tuple[int, Dict[str, Any], str]) -> Tuple[int, List[Dict[str, Any]]]:
            i, r, rundir = job
            return i, grade_again(TASKS[r["task"]], rundir, arm.get("harness"),
                                  r.get("reply") or "", baselines[r["task"]], sandboxed, memory)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            graded = dict(pool.map(one, jobs))

        reproduced = soft_differs = 0
        moved: List[str] = []
        for i, r, _ in jobs:
            checks = graded[i]
            if r["task"] in changed:
                passed = all(c["ok"] for c in checks if c.get("critical")) and not r["timed_out"]
                if passed != r["pass"]:
                    moved.append(f"{r['task']} run {i}: {r['pass']} -> {passed}")
                r["checks"], r["pass"] = checks, passed
                continue
            if _critical(checks) != _critical(r["checks"]):
                disagree.append(f"{name}: {r['task']} run {i} (unchanged grader) graded "
                                f"{_critical(checks)} against the stored {_critical(r['checks'])}")
            else:
                reproduced += 1
                soft_differs += {c["name"]: c["ok"] for c in checks} != \
                    {c["name"]: c["ok"] for c in r["checks"]}
        from .agent import summarise
        arm["summary"] = summarise(arm["runs"])
        for t in changed:
            arm["kit"]["tasks"][t] = task_digest(TASKS[t])
        account[name or ""] = {
            "kit": KIT, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tasks": changed, "from": {t: old_digests[t] for t in changed},
            "not_regradable": skipped,
            "verdicts_moved": moved,
            "unchanged_reproduced": f"{reproduced}/{sum(1 for _, r, _ in jobs if r['task'] not in changed)}",
            "unchanged_soft_differs": soft_differs,
        }
        arm["regraded"] = account[name or ""]
    return out, disagree, account


def _workspaces(path: str, scratch: str) -> str:
    """A directory of kept runs: the path itself, or an archive extracted into scratch."""
    if os.path.isdir(path):
        return path
    dest = tempfile.mkdtemp(dir=scratch)
    zstd = ["-I", "zstd"] if path.endswith((".zst", ".zstd")) else []
    subprocess.run(["tar", "-C", dest, *zstd, "-xf", path], check=True)
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py agent regrade",
        description="Grade a saved agent result's kept workspaces again under the current "
                    "graders, so tasks whose grader changed compare again. Every run of an "
                    "unchanged task must reproduce its verdict first, or nothing is written.")
    ap.add_argument("result", help="a saved agent result (JSON)")
    ap.add_argument("--workspaces", required=True,
                    help="the kept runs: the --keep directory, or its .tar.zst archive")
    ap.add_argument("--save", required=True, help="where to write the regraded result")
    ap.add_argument("--tasks", nargs="+", choices=sorted(TASKS),
                    help="regrade only these (default: every task whose grader changed)")
    ap.add_argument("--workers", type=int, default=4, help="runs graded at once (default 4)")
    ap.add_argument("--memory", default="1500M", help="memory cap per graded run")
    ap.add_argument("--unsandboxed", action="store_true",
                    help="grade on the host, without bubblewrap (the grader runs the agent's code)")
    args = ap.parse_args(argv)
    with open(args.result) as fh:
        result = json.load(fh)
    scratch = tempfile.mkdtemp(prefix="regrade-")
    try:
        root = _workspaces(args.workspaces, scratch)
        new, disagree, account = regrade(result, root, scratch, not args.unsandboxed,
                                         args.memory, args.workers, args.tasks)
    finally:
        subprocess.run(["chmod", "-R", "u+rwX", scratch], check=False)
        shutil.rmtree(scratch, ignore_errors=True)
    for name, a in account.items():
        label = name or "result"
        print(f"{label}: regraded {', '.join(a['tasks']) or 'nothing (no grader changed)'}; "
              f"unchanged tasks reproduced {a['unchanged_reproduced']}"
              + (f" ({a['unchanged_soft_differs']} differ in a soft check only)"
                 if a["unchanged_soft_differs"] else "")
              + (f"; verdicts moved: {'; '.join(a['verdicts_moved'])}"
                 if a["verdicts_moved"] else "; no verdict moved")
              + (f"; not regradable (changed beyond the grader): {', '.join(a['not_regradable'])}"
                 if a["not_regradable"] else ""))
    if disagree:
        print(f"error: {len(disagree)} run(s) did not reproduce; nothing written:", file=sys.stderr)
        for line in disagree[:20]:
            print("  " + line, file=sys.stderr)
        return 1
    with open(args.save, "w") as fh:
        json.dump(new, fh, indent=2)
    print(f"Saved to {args.save}")
    return 0
