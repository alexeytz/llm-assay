"""``compare`` for agent results: per task, then across tasks.

A pass rate is a proportion, so a task is compared by Fisher's exact test on
(passed, failed) in each arm, and the arm as a whole by a Cochran-Mantel-
Haenszel test stratified by task. Pooling every run into one 2x2 table would
let the task mix decide the answer: tasks differ wildly in base rate (Flash-Next
passes `sync` 20/20 and `feature` 13/20), and an arm with more runs of an easy
task would look better for that alone.

Two results are compared only on what they measured with the same instrument:
a task whose kit digest differs between them is left out and named. A result
saved before the kit existed has no digest, which is unknown rather than
different -- it is compared, with a warning, because "unknown" is the honest
reading and refusing it would make every older result unreadable.

What an experiment varies (thinking, sampling, the endpoint's deployment) is
reported when it differs, not refused: varying it is usually the point.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

from .probe import fisher_exact
from .agent_regrade import GRADER_ONLY


def is_agent(path: str) -> bool:
    try:
        with open(path.split("#", 1)[0]) as fh:
            return str(json.load(fh).get("kind", "")).startswith("agent")
    except (OSError, ValueError, AttributeError):
        return False


def load(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """``FILE`` or ``FILE#ARM``: the report(s) it names, each with a label."""
    file, _, arm = path.partition("#")
    with open(file) as fh:
        doc = json.load(fh)
    stem = file.rsplit("/", 1)[-1].removesuffix(".json")
    if doc.get("kind") == "agent-models":
        models = doc.get("models") or {}
        if arm:
            if arm not in models:
                raise ValueError(f"{file} has no arm {arm!r}; it has {', '.join(models)}")
            return [(f"{stem}#{arm}", models[arm])]
        return [(f"{stem}#{name}", report) for name, report in models.items()]
    if doc.get("kind") != "agent":
        raise ValueError(f"{file} is not an agent result")
    if arm:
        raise ValueError(f"{file} holds one arm; drop #{arm}")
    return [(stem, doc)]


def cmh(strata: List[Tuple[int, int, int, int]]) -> Optional[float]:
    """Two-sided p of the Mantel-Haenszel test, continuity-corrected, over 2x2 strata.

    Each stratum is (a passed, a failed, b passed, b failed). A stratum whose
    margins leave nothing to vary (both arms all pass, say) adds no variance and
    no evidence, which is what it should add.
    """
    num, var = 0.0, 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        row1, row2, col1, col2 = a + b, c + d, a + c, b + d
        num += a - row1 * col1 / n
        var += row1 * row2 * col1 * col2 / (n * n * (n - 1))
    if var <= 0:
        return None
    chi2 = max(0.0, abs(num) - 0.5) ** 2 / var
    return math.erfc(math.sqrt(chi2 / 2))


def mh_difference(strata: List[Tuple[int, int, int, int]]) -> Optional[float]:
    """The Mantel-Haenszel risk difference, b's pass rate less a's, weighted by task.

    The verdict's direction must come from the same stratification as its p:
    the pooled difference can point the other way when the task mix differs
    (Simpson's paradox), and calling a candidate that is better on every task
    WORSE is the error stratifying exists to prevent. Its sign is the sign of
    the CMH statistic's own numerator, and when every task has the same run
    count on both sides it equals the pooled difference.
    """
    num, den = 0.0, 0.0
    for a, b, c, d in strata:
        na, nc = a + b, c + d
        if na == 0 or nc == 0:
            continue
        w = na * nc / (na + nc)
        num += w * (c / nc - a / na)
        den += w
    return num / den if den else None


#: Arm settings worth naming when they differ; an experiment usually varies one.
_SETTINGS = (("thinking", "thinking"), ("sampling", "sampling"), ("timeout_s", "timeout"),
             ("max_tokens", "max tokens"), ("context_window", "context window"),
             ("workers", "workers"), ("harness_version", "harness"),
             ("served_model", "model served"), ("served_build", "server build"),
             ("engine_version", "engine"), ("deployment", "deployment"),
             ("sandboxed", "sandbox"), ("memory", "memory cap"),
             ("append_system_prompt", "appended system prompt"),
             ("system_prompt", "replacement system prompt"))


def _canonical(value: Any) -> str:
    """A setting as text, with 1200 and 1200.0 the same: one file wrote an int."""
    def norm(v: Any) -> Any:
        if isinstance(v, bool) or not isinstance(v, (int, float, dict, list)):
            return v
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return float(v)
    return json.dumps(norm(value), sort_keys=True)


def analyse(paths: List[str], alpha: float = 0.05) -> Dict[str, Any]:
    reports = [r for p in paths for r in load(p)]
    if len(reports) < 2:
        raise ValueError("need at least two agent results (FILE, or FILE#ARM for a models file)")
    (base_label, base), others = reports[0], reports[1:]
    warnings: List[str] = []
    for label, r in reports:
        if not r.get("kit"):
            warnings.append(f"{label} predates the kit: which instrument measured it is unknown")
        warnings += [f"{label}: {w}" for w in r.get("warnings") or []]
        # A regrade rewrote verdicts the arm did not originally have: say so, with
        # how well the tasks it did not rewrite reproduced.
        regraded = r.get("regraded") or {}
        if regraded.get("tasks"):
            warnings.append(f"{label}: {', '.join(regraded['tasks'])} regraded under "
                            f"{regraded.get('kit')} from its kept workspaces (unchanged tasks "
                            f"reproduced {regraded.get('unchanged_reproduced')})")
    for field, name in _SETTINGS:
        values = [(label, r.get(field)) for label, r in reports]
        # Unknown is not different, but two known values that disagree still do.
        known = {_canonical(v) for _, v in values if v is not None}
        if len(known) > 1:
            warnings.append(f"{name} differs: " + ", ".join(f"{l}={v!r}" for l, v in values))
    for label, r in reports:
        server = (r.get("server_settings") or {}).get("cache_dtype")
        if server and any((o.get("server_settings") or {}).get("cache_dtype") not in (None, server)
                          for _, o in reports):
            warnings.append("KV-cache dtype differs: " + ", ".join(
                f"{l}={(o.get('server_settings') or {}).get('cache_dtype')!r}" for l, o in reports))
            break

    # The code that runs, sandboxes and voids a run is in no task's digest, so a
    # change there would otherwise pass unnoticed with every task "the same".
    runners = [(label, (r.get("kit") or {}).get("runner")) for label, r in reports]
    if all(v for _, v in runners) and len({v for _, v in runners}) > 1:
        warnings.append("the runner differs (the code that runs and voids a run): "
                        + ", ".join(f"{l}={v}" for l, v in runners))

    # A void run is left out of the pass rate, so a side that lost many of its runs
    # is compared on the survivors, which need not be a random subset.
    for label, r in reports:
        void = {t: s.get("void", 0) for t, s in (r.get("summary") or {}).items() if s.get("void")}
        if void:
            warnings.append(f"{label}: {sum(void.values())} void run(s) left out of the "
                            "pass rates (" + ", ".join(f"{t} {n}" for t, n in void.items()) + ")")

    comparisons = []
    for label, cand in others:
        rows, skipped, strata = [], [], []
        base_kit = (base.get("kit") or {}).get("tasks") or {}
        cand_kit = (cand.get("kit") or {}).get("tasks") or {}
        for task, bs in base["summary"].items():
            cs = cand["summary"].get(task)
            if cs is None:
                skipped.append((task, f"not in {label}"))
                continue
            if task in base_kit and task in cand_kit and base_kit[task] != cand_kit[task]:
                older = GRADER_ONLY.get(task, set())
                if base_kit[task] in older or cand_kit[task] in older:
                    skipped.append((task, "graded by an older grader on one side; `agent regrade` "
                                          "that side's kept workspaces to compare it"))
                else:
                    skipped.append((task, "measured with a different kit (the task's digest "
                                          "differs)"))
                continue
            if not bs["runs"] or not cs["runs"]:
                skipped.append((task, "no runs that were not void"))
                continue
            a, b = bs["passed"], bs["runs"] - bs["passed"]
            c, d = cs["passed"], cs["runs"] - cs["passed"]
            strata.append((a, b, c, d))
            p = fisher_exact(a, b, c, d)
            diff = c / cs["runs"] - a / bs["runs"]
            rows.append({"task": task, "baseline": [a, bs["runs"]], "candidate": [c, cs["runs"]],
                         "diff": diff, "p": p, "verdict": _verdict(diff, p, alpha)})
        skipped += [(t, f"not in {base_label}") for t in cand["summary"] if t not in base["summary"]]
        overall = None
        if strata:
            pa, na = sum(s[0] for s in strata), sum(s[0] + s[1] for s in strata)
            pc, nc = sum(s[2] for s in strata), sum(s[2] + s[3] for s in strata)
            p_all = cmh(strata)
            diff = mh_difference(strata)
            overall = {"baseline": [pa, na], "candidate": [pc, nc], "diff": diff, "p": p_all,
                       "verdict": "no difference" if p_all is None or diff is None
                       else _verdict(diff, p_all, alpha)}
        comparisons.append({"candidate": label, "rows": rows, "skipped": skipped,
                            "overall": overall})
    return {"baseline": base_label, "alpha": alpha, "warnings": warnings,
            "comparisons": comparisons}


def _verdict(diff: float, p: float, alpha: float) -> str:
    if p >= alpha:
        return "no difference"
    return "BETTER" if diff > 0 else "WORSE"


def render(result: Dict[str, Any]) -> str:
    lines = []
    for w in result["warnings"]:
        lines.append(f"warning: {w}")
    for comp in result["comparisons"]:
        lines += ["", f"agent tasks | baseline: {result['baseline']} | candidate: "
                  f"{comp['candidate']} | alpha {result['alpha']}", "",
                  "| task | baseline | candidate | diff | p | verdict |",
                  "|:-----|---------:|----------:|-----:|--:|:--------|"]
        for r in comp["rows"]:
            lines.append(f"| {r['task']} | {r['baseline'][0]}/{r['baseline'][1]} | "
                         f"{r['candidate'][0]}/{r['candidate'][1]} | {100 * r['diff']:+.0f} pts | "
                         f"{r['p']:.3f} | {r['verdict']} |")
        o = comp["overall"]
        if o:
            p = "-" if o["p"] is None else f"{o['p']:.4f}"
            diff = "-" if o["diff"] is None else f"{100 * o['diff']:+.1f} pts"
            lines.append(f"| **all, stratified by task** | {o['baseline'][0]}/{o['baseline'][1]} | "
                         f"{o['candidate'][0]}/{o['candidate'][1]} | {diff} | "
                         f"{p} | **{o['verdict']}** |")
        lines.append("")
        lines.append("Per task: Fisher's exact test. All: Cochran-Mantel-Haenszel, stratified "
                     "by task, so the task mix cannot decide it; its diff is the Mantel-Haenszel "
                     "difference, weighted by task, not the pooled one.")
        for task, why in comp["skipped"]:
            lines.append(f"not compared: {task} -- {why}")
    return "\n".join(lines)


def main(paths: List[str], alpha: float, as_json: bool, fail_on_regression: bool) -> int:
    try:
        result = analyse(paths, alpha)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2) if as_json else render(result))
    if not any(c["rows"] for c in result["comparisons"]):
        print("error: no task was compared: none is in both results with the same kit",
              file=sys.stderr)
        return 1
    if fail_on_regression and any((c["overall"] or {}).get("verdict") == "WORSE"
                                  for c in result["comparisons"]):
        print(f"error: the candidate is significantly worse across tasks at alpha={alpha}",
              file=sys.stderr)
        return 1
    return 0
