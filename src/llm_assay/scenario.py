"""Scenario runners: one question about an endpoint, answered end to end.

The other subcommands each take one measurement. The mistakes that cost the
most were never inside a measurement; they were in what surrounded it -- the
"before" side of an A/B nobody took because the change was already made, the
second side served from the cache the first one filled, a reasoning parser
missing for a whole batch of trials. A scenario is the measurements plus those
controls, run in order, with a verdict at the end.

``qualify`` asks whether a deployment is served right before anything is
measured on it: `tune`, a canary reply at the deployment's default thinking
and, where an off switch changes the prompt, with thinking off, the smoke
benchmark, a short probe and, optionally, a few agent runs. ``ab-deploy`` asks
whether a change to the server helped, in two invocations around the change:
``before`` records the endpoint and measures it at a seed it keeps, ``after``
reads the endpoint again, says what changed, measures at the same seed, and
compares.

Each step is the same command a user would type, run through this checkout's
entry point, so every result in a scenario directory is an ordinary result,
and ``VERDICT.txt`` lists the commands that made them.

Usage::

    python llm-assay.py scenario qualify http://localhost:8000/v1
    python llm-assay.py scenario ab-deploy before http://localhost:8000/v1 --dir results/ab-kv
    # ... change the server ...
    python llm-assay.py scenario ab-deploy after --dir results/ab-kv
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import resolve_api_key

#: How a step came out. FAIL means the scenario's question has no answer yet;
#: WARN means it has one, with something the reader must know beside it.
STATUSES = ("PASS", "WARN", "FAIL", "SKIP")

#: The agent tasks `qualify` runs: one per trap family that finishes quickly.
QUALIFY_TASKS = ("sync", "dedupe", "secret-commit")

#: A prefix-cache hit rate above this on a shape the scenario expects to be
#: cold means it was not: the same threshold as the benchmark's own warning.
_STALE_HIT_RATE = 0.10

_CANARY = "Reply with exactly one word: OK"


def _entry_point() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "llm-assay.py")


class Step:
    def __init__(self, name: str, status: str, detail: str,
                 command: Optional[List[str]] = None) -> None:
        assert status in STATUSES
        self.name, self.status, self.detail, self.command = name, status, detail, command

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail,
                "command": self.command}


class Scenario:
    """A directory, the steps taken so far, and the commands behind them."""

    def __init__(self, kind: str, directory: str, stem: str = "") -> None:
        #: ``stem`` names the verdict files: the before side of an A/B writes
        #: VERDICT-before.txt, so the after side's VERDICT.txt does not erase it.
        self.kind, self.dir, self.stem = kind, directory, stem
        self.steps: List[Step] = []
        os.makedirs(directory, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def run(self, label: str, args: List[str]) -> Tuple[int, str]:
        """Run ``llm-assay.py ARGS``, teeing its output to ``<label>.log``."""
        cmd = [sys.executable, _entry_point(), *args]
        shown = " ".join(["llm-assay.py", *(_shell(a) for a in args)])
        print(f"\n==> {label}: {shown}", flush=True)
        lines: List[str] = []
        with open(self.path(f"{label}.log"), "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, stdin=subprocess.DEVNULL)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                lines.append(line)
            code = proc.wait()
        return code, "".join(lines)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        print(f"[{step.status}] {step.name}: {step.detail}", flush=True)
        return step

    def verdict(self, extra: Optional[List[str]] = None) -> int:
        worst = "PASS"
        for step in self.steps:
            if step.status == "FAIL":
                worst = "FAIL"
            elif step.status == "WARN" and worst == "PASS":
                worst = "WARN"
        lines = [f"llm-assay scenario {self.kind}: {worst}", ""]
        for step in self.steps:
            lines.append(f"[{step.status}] {step.name}: {step.detail}")
        if extra:
            lines += ["", *extra]
        commands = [s for s in self.steps if s.command]
        if commands:
            lines += ["", "Commands:"]
            lines += ["  " + " ".join(["llm-assay.py", *(_shell(a) for a in s.command or [])])
                      for s in commands]
        text = "\n".join(lines) + "\n"
        suffix = f"-{self.stem}" if self.stem else ""
        with open(self.path(f"VERDICT{suffix}.txt"), "w") as handle:
            handle.write(text)
        with open(self.path(f"verdict{suffix}.json"), "w") as handle:
            json.dump({"scenario": self.kind, "verdict": worst,
                       "steps": [s.as_dict() for s in self.steps]}, handle, indent=2)
        print("\n" + text + f"Written to {self.path(f'VERDICT{suffix}.txt')}")
        return 1 if worst == "FAIL" else 0


def _shell(arg: str) -> str:
    return arg if re.fullmatch(r"[A-Za-z0-9_./:=@%+,-]+", arg) else "'" + arg.replace("'", "'\\''") + "'"


def _load(path: str) -> Optional[Any]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _tune(sc: Scenario, base_url: str, model: Optional[str]) -> Tuple[Step, Optional[Dict[str, Any]]]:
    args = ["tune", base_url, "--json"]
    if model:
        args += ["--model", model]
    cmd = [sys.executable, _entry_point(), *args]
    print(f"\n==> tune: llm-assay.py {' '.join(args)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    with open(sc.path("tune.log"), "w") as log:
        log.write(proc.stderr)
    found: Optional[Dict[str, Any]] = None
    if proc.returncode == 0:
        try:
            found = json.loads(proc.stdout)
        except ValueError:
            found = None
    if not found:
        return Step("detect", "FAIL", "tune could not read the endpoint (see tune.log)", args), None
    with open(sc.path("tune.json"), "w") as handle:
        json.dump(found, handle, indent=2)
    d = found["detected"]
    bits = [f"{d.get('hf_model') or d.get('served_model')}",
            f"served as {d.get('served_model')}" if d.get("served_model") else "",
            f"max ctx {d.get('max_model_len')}" if d.get("max_model_len") else "",
            f"{d.get('server_version')}" if d.get("server_version") else ""]
    return Step("detect", "PASS", ", ".join(b for b in bits if b), args), found


def _target(found: Dict[str, Any]) -> List[str]:
    """The flags naming the endpoint, as tune put them on its smoke preset."""
    args: List[str] = next(p["args"] for p in found["presets"] if p["key"] == "smoke")
    return args[:args.index("--pp")]


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key not in ("", "EMPTY") else {}


def canary(base_url: str, model: str, api_key: str, extra: Dict[str, Any],
           label: str) -> Step:
    """One short reply: does the answer arrive as an answer?

    The checks are the silent failures of a new deployment: a missing or wrong
    reasoning parser leaves the think block in the content, or the whole reply
    in reasoning with empty content; a model that will not stop runs to the
    budget. Each is invisible in a throughput table.
    """
    body = {"model": model, "messages": [{"role": "user", "content": _CANARY}],
            "max_tokens": 4096, "temperature": 0, **extra}
    try:
        resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=body,
                             headers=_headers(api_key), timeout=300)
    except requests.RequestException as exc:
        return Step(label, "FAIL", f"request failed: {exc}")
    if resp.status_code != 200:
        return Step(label, "FAIL", f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        choice = resp.json()["choices"][0]
    except (ValueError, KeyError, IndexError):
        return Step(label, "FAIL", "the reply is not a chat completion")
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish = choice.get("finish_reason")
    said = f"content {content[:40]!r}, {len(reasoning)} reasoning chars, finish {finish}"
    if re.search(r"</?think>", content):
        return Step(label, "FAIL", f"the think block is in the content -- the server's "
                                   f"reasoning parser is missing or wrong ({said})")
    if not content:
        return Step(label, "FAIL", f"empty content: the answer is in reasoning or nowhere, "
                                   f"and the probe would score nothing ({said})")
    if finish != "stop":
        return Step(label, "WARN", f"the reply did not stop by itself ({said})")
    if len(content) > 200 or "ok" not in content.lower():
        return Step(label, "WARN", f"the reply is not the one word asked for ({said})")
    return Step(label, "PASS", said)


def _thinking_step(found: Dict[str, Any]) -> Step:
    d = found["detected"]
    controls = d.get("thinking_controls")
    if not controls:
        return Step("thinking", "SKIP", "no thinking switch could be compared on this endpoint")
    live = [k for k, v in controls.items() if v == "live"]
    rejected = [k for k, v in controls.items() if v == "rejected"]
    levels = d.get("thinking_levels") or []
    if not live:
        return Step("thinking", "WARN", "no thinking switch changes the prompt: "
                                        "--thinking off does nothing on this deployment")
    detail = (f"levels that make different prompts: {', '.join(levels)}" if levels
              else "only on/off changes the prompt; every graded level renders what sending nothing renders")
    if rejected:
        detail += f"; rejected: {', '.join(k.split('=', 1)[-1] for k in rejected)}"
    return Step("thinking", "PASS", detail)


def _warm(result: Optional[Dict[str, Any]]) -> bool:
    return any(((r.get("server_metrics") or {}).get("prefix_cache_hit_rate") or 0) > _STALE_HIT_RATE
               for r in (result or {}).get("benchmarks") or [])


def _benchmark_checks(result: Optional[Dict[str, Any]], expect_cold: bool) -> List[Tuple[str, str]]:
    """(status, sentence) for what a saved benchmark says about its own conditions."""
    if not result:
        return [("FAIL", "no result was saved")]
    out: List[Tuple[str, str]] = []
    runs = result.get("benchmarks") or []
    busy = [r for r in runs if (r.get("server_metrics") or {}).get("requests_running_at_start")]
    if busy:
        out.append(("WARN", f"{len(busy)} shape(s) started with other requests on the server: "
                            "another client shares it, and its load is in these numbers"))
    if expect_cold:
        warm = [r for r in runs if _warm({"benchmarks": [r]})]
        if warm:
            out.append(("WARN", f"{len(warm)} shape(s) were partly served from the prefix cache: "
                                "this side read text the server had already seen, so its "
                                "prefill is not comparable"))
    if result.get("identity_drift"):
        out.append(("FAIL", "the endpoint changed during the run: " + "; ".join(result["identity_drift"])))
    if result.get("tokenizer_fallback"):
        # The field names the tokenizer that failed to load; gpt2 stood in.
        out.append(("WARN", f"{result['tokenizer_fallback']} did not load, so shapes were "
                            "counted with gpt2 and their labels are approximate: pass "
                            "--tokenizer <base model repo>"))
    return out


def _agent_args(base_url: str, served: str, harness: str, tasks: List[str], runs: int,
                workers: int, thinking: str, harness_cmd: Optional[str],
                deployment: List[str], result: str) -> List[str]:
    args = ["agent", base_url, "--model", served, "--harness", harness, "--tasks", *tasks,
            "--runs", str(runs), "--workers", str(workers), "--thinking", thinking,
            "--save-result", result]
    if harness_cmd:
        args += ["--harness-cmd", harness_cmd]
    if deployment:
        args += ["--deployment", *deployment]
    return args


def _agent_summary(result: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not result:
        return "FAIL", "no result was saved (see the log)"
    # One arm saves its runs at the top level; a models file nests them by arm.
    arms = list(result["models"].values()) if result.get("models") else [result]
    runs = [r for arm in arms for r in arm.get("runs", [])]
    void = [r for r in runs if r.get("void")]
    graded = [r for r in runs if not r.get("void")]
    passed = sum(1 for r in graded if r.get("pass"))
    failed = sorted({r["task"] for r in graded if not r.get("pass")})
    detail = f"{passed}/{len(graded)} passed"
    if failed:
        detail += f" (failed: {', '.join(failed)})"
    if void:
        return "FAIL", detail + f"; {len(void)} void -- the harness or endpoint, not the model"
    return ("PASS" if graded else "FAIL"), detail


def _pick_harness(found: Dict[str, Any], wanted: str) -> Optional[str]:
    have = found.get("harnesses") or []
    if wanted == "auto":
        return have[0] if have else None
    return wanted


def _agent_thinking(found: Dict[str, Any]) -> str:
    for preset in found.get("agent") or []:
        args: List[str] = preset["args"]
        if "--thinking" in args:
            return str(args[args.index("--thinking") + 1])
    controls = found["detected"].get("thinking_controls") or {}
    return "off" if controls.get("reasoning_effort=medium") == "rejected" else "medium"


# --- qualify ------------------------------------------------------------------

def qualify(args: argparse.Namespace) -> int:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sc = Scenario("qualify", args.dir or os.path.join("results", f"qualify-{stamp}"))
    step, found = _tune(sc, args.base_url, args.model)
    sc.add(step)
    if not found:
        return sc.verdict()
    served = found["detected"].get("served_model")
    if not served:
        sc.add(Step("canary", "FAIL", "the endpoint names no model to send requests to"))
        return sc.verdict()
    api_key = resolve_api_key(None)

    sc.add(canary(found["endpoint"], served, api_key, {}, "canary (thinking default)"))
    controls = found["detected"].get("thinking_controls") or {}
    if any(v == "live" for k, v in controls.items() if k in ("reasoning_effort=none",
                                                            "enable_thinking=false")):
        off = canary(found["endpoint"], served, api_key,
                     {"reasoning_effort": "none",
                      "chat_template_kwargs": {"enable_thinking": False}},
                     "canary (thinking off)")
        sc.add(off)
    sc.add(_thinking_step(found))

    target = _target(found)
    smoke = sc.path("smoke.json")
    smoke_args = [*target, "--pp", "512", "--tg", "64", "--exact-tg", "--depth", "0",
                  "--runs", "3", "--latency-mode", "generation",
                  "--seed", str(random.randrange(1, 2**31)), "--save-result", smoke]
    code, _ = sc.run("smoke", smoke_args)
    checks = _benchmark_checks(_load(smoke), expect_cold=False)
    status = "FAIL" if code else "PASS"
    status = next((s for s, _ in checks if s == "FAIL"), None) or (
        "WARN" if status == "PASS" and checks else status)
    sc.add(Step("smoke benchmark", status,
                "; ".join(t for _, t in checks) or ("measured" if not code else f"exit {code}"),
                smoke_args))

    if args.no_probe:
        sc.add(Step("probe", "SKIP", "--no-probe"))
    else:
        probe = sc.path("probe.json")
        probe_args = ["probe", found["endpoint"], *target[target.index("--model"):],
                      "--depths", str(args.probe_depth), "--rungs", "outlier", "clean",
                      "--trials", str(args.probe_trials), "--seed",
                      str(random.randrange(1, 2**31)), "--save-result", probe]
        code, _ = sc.run("probe", probe_args)
        sc.add(_probe_step(code, _load(probe), probe_args))

    harness = _pick_harness(found, args.agent) if args.agent else None
    if not args.agent:
        sc.add(Step("agent", "SKIP", "not asked for (--agent HARNESS or --agent auto)"))
    elif not harness:
        sc.add(Step("agent", "SKIP", "--agent auto, and no harness is on PATH"))
    else:
        result = sc.path("agent.json")
        agent_args = _agent_args(found["endpoint"], served, harness, list(QUALIFY_TASKS),
                                 1, 1, _agent_thinking(found), args.harness_cmd, [], result)
        code, _ = sc.run("agent", agent_args)
        status, detail = _agent_summary(_load(result))
        if code and status == "PASS":
            status, detail = "FAIL", detail + f"; exit {code}"
        elif status == "PASS" and "failed" in detail:
            status = "WARN"
        sc.add(Step(f"agent ({harness})", status, detail, agent_args))

    return sc.verdict(["A WARN is something to read before measuring; a FAIL is something to",
                       "fix first. Each step's output is in its .log beside this file."])


def _probe_step(code: int, result: Optional[Dict[str, Any]], args: List[str]) -> Step:
    if not result:
        return Step("probe", "FAIL", f"no result was saved (exit {code}; see probe.log)", args)
    if result.get("void"):
        return Step("probe", "FAIL", "void: " + "; ".join(map(str, result["void"])), args)
    cells = {c["rung"]: c for c in result.get("results") or []}
    parts, status = [], "PASS"
    for rung in ("outlier", "clean"):
        cell = cells.get(rung)
        if not cell:
            continue
        parts.append(f"{rung} {cell.get('ok', 0)}/{cell.get('trials', 0)} right, "
                     f"{cell.get('answered', 0)} answered")
        if cell.get("trials") and not cell.get("answered"):
            status = "FAIL"
    detail = "; ".join(parts) or "no cells"
    if status == "FAIL":
        detail += " -- no reply could be scored: check the reasoning parser and --max-tokens"
    elif "outlier" in cells and not cells["outlier"].get("ok"):
        status = "WARN"
        detail += " -- the easy rung failed: the model may not be reading the context"
    if result.get("tokenizer_fallback") and status != "FAIL":
        status = "WARN"
        detail += (f" -- {result['tokenizer_fallback']} did not load, so the depth was "
                   "counted with gpt2: pass --tokenizer")
    return Step("probe", status, detail, args)


# --- ab-deploy ----------------------------------------------------------------

def _read_endpoint(base_url: str, model: Optional[str], api_key: str) -> Dict[str, Any]:
    import aiohttp
    from .servermetrics import endpoint_identity, server_settings

    async def go() -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            ident = await endpoint_identity(session, base_url, api_key, model)
            ident["server_settings"] = await server_settings(session, base_url, api_key)
            return ident
    return asyncio.run(go())


def endpoint_changes(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """What the endpoint says is different, field by field. Unknown on either
    side is skipped -- the repo-wide rule -- so a server that publishes nothing
    reports no change rather than a false one."""
    out: List[str] = []
    for key in ("served_model", "engine_version"):
        if before.get(key) is not None and after.get(key) is not None and before[key] != after[key]:
            out.append(f"{key}: {before[key]} -> {after[key]}")
    b_build, a_build = before.get("served_build") or {}, after.get("served_build") or {}
    for key in sorted(set(b_build) & set(a_build)):
        if b_build[key] != a_build[key]:
            out.append(f"served_build.{key}: {b_build[key]} -> {a_build[key]}")
    b_set, a_set = before.get("server_settings") or {}, after.get("server_settings") or {}
    for key in sorted(set(b_set) | set(a_set)):
        if key in b_set and key in a_set and b_set[key] != a_set[key]:
            out.append(f"{key}: {b_set[key]} -> {a_set[key]}")
    return out


def _ab_shapes(found: Dict[str, Any], runs: int) -> List[str]:
    ladder = found.get("depth_ladder") or [0]
    depths = ["0"] + ([str(ladder[1])] if len(ladder) > 1 else [])
    return ["--pp", "512", "2048", "8192", "--tg", "64", "--exact-tg", "--depth", *depths,
            "--runs", str(runs), "--latency-mode", "generation"]


def ab_before(args: argparse.Namespace) -> int:
    sc = Scenario("ab-deploy (before)", args.dir, stem="before")
    if os.path.exists(sc.path("state.json")) and not args.force:
        print(f"error: {sc.path('state.json')} exists: this directory already has a before "
              "side. Use another --dir, or --force to start again.", file=sys.stderr)
        return 2
    step, found = _tune(sc, args.base_url, args.model)
    sc.add(step)
    if not found:
        return sc.verdict()
    api_key = resolve_api_key(None)
    served = found["detected"].get("served_model")
    ident = _read_endpoint(found["endpoint"], served, api_key)
    seed = args.seed or random.randrange(1, 2**31)
    bench = [*_target(found), *_ab_shapes(found, args.runs), "--seed", str(seed)]
    harness = _pick_harness(found, args.agent) if args.agent else None
    state = {
        "base_url": found["endpoint"], "model": args.model, "seed": seed, "bench": bench,
        "before": {"endpoint": ident, "note": args.note, "tune": "tune.json"},
        "agent": None if not harness else {
            "harness": harness, "tasks": args.agent_tasks, "runs": args.agent_runs,
            "workers": args.agent_workers, "thinking": _agent_thinking(found),
            "harness_cmd": args.harness_cmd, "served": served},
    }
    with open(sc.path("state.json"), "w") as handle:
        json.dump(state, handle, indent=2)
    if args.agent and not harness:
        sc.add(Step("agent", "SKIP", "--agent auto, and no harness is on PATH"))
    _ab_side(sc, state, "before", args.note)
    return sc.verdict([
        f"Seed {seed} is kept in state.json; the after side reuses it, so the two read",
        "the same text and pair. Change the server now -- restarting it empties its",
        "prefix cache, which the after side needs -- then run:",
        f"  llm-assay.py scenario ab-deploy after --dir {args.dir} --note '...'",
        "(or, from a runner tune wrote, the ab-after preset with the same AB).",
    ])


def _ab_side(sc: Scenario, state: Dict[str, Any], side: str, note: Optional[str]) -> None:
    result = sc.path(f"{side}.json")
    bench = [*state["bench"], "--save-result", result]
    code, _ = sc.run(side, bench)
    checks = _benchmark_checks(_load(result), expect_cold=True)
    status = "FAIL" if code else "PASS"
    if any(s == "FAIL" for s, _ in checks):
        status = "FAIL"
    elif checks and status == "PASS":
        status = "WARN"
    sc.add(Step(f"benchmark ({side})", status,
                "; ".join(t for _, t in checks) or ("measured" if not code else f"exit {code}"),
                bench))
    agent = state.get("agent")
    if agent:
        deployment = [f"scenario_side={side}"] + ([f"note={note}"] if note else [])
        agent_args = _agent_args(state["base_url"], agent["served"], agent["harness"],
                                 agent["tasks"], agent["runs"], agent["workers"],
                                 agent["thinking"], agent["harness_cmd"], deployment,
                                 sc.path(f"{side}-agent.json"))
        code, _ = sc.run(f"{side}-agent", agent_args)
        status, detail = _agent_summary(_load(sc.path(f"{side}-agent.json")))
        if code and status == "PASS":
            status, detail = "FAIL", detail + f"; exit {code}"
        sc.add(Step(f"agent ({side}, {agent['harness']})", status, detail, agent_args))


def _compare_summary(text: str) -> str:
    """compare's own summary line, which says how many rows differed."""
    for line in text.splitlines():
        if "statistically significant difference" in line or "all, stratified by task" in line:
            return line.strip().strip("|").strip()
    return ""


def ab_after(args: argparse.Namespace) -> int:
    sc = Scenario("ab-deploy", args.dir)
    state = _load(sc.path("state.json"))
    if not state:
        print(f"error: no before side in {args.dir} (state.json): run "
              "`scenario ab-deploy before` first", file=sys.stderr)
        return 2
    if os.path.exists(sc.path("after.json")) and not args.force:
        print(f"error: {sc.path('after.json')} exists: this A/B already has an after side. "
              "--force measures it again.", file=sys.stderr)
        return 2
    # The before side's steps open this verdict, so one file holds the whole A/B
    # and every command behind it.
    earlier = _load(sc.path("verdict-before.json")) or {}
    for step in earlier.get("steps", []):
        name = step["name"] if "(before" in step["name"] else f"{step['name']} (before)"
        sc.steps.append(Step(name, step["status"], step["detail"],
                             step.get("command")))
    api_key = resolve_api_key(None)
    before = state["before"]["endpoint"]
    after = _read_endpoint(state["base_url"], before.get("served_model"), api_key)
    if after.get("served_model") is None:
        sc.add(Step("endpoint", "FAIL", "the endpoint does not answer /v1/models: is it up?"))
        return sc.verdict()
    changes = endpoint_changes(before, after)
    note = args.note
    if changes:
        sc.add(Step("endpoint", "PASS", "changed: " + "; ".join(changes)))
    else:
        sc.add(Step("endpoint", "WARN" if not note else "PASS",
                    "nothing the endpoint reports changed"
                    + (f"; the change is recorded as: {note}" if note else
                       " -- fine for a change it cannot see (a GPU setting, a driver), "
                       "but say what it was with --note")))
    state["after"] = {"endpoint": after, "note": note, "changes": changes}
    with open(sc.path("state.json"), "w") as handle:
        json.dump(state, handle, indent=2)

    _ab_side(sc, state, "after", note)

    extra: List[str] = []
    #: Prefill is what a deployment change usually moves, decode what it
    #: usually should not; an agent A/B compares pass rates per task instead.
    for label, a, b, metric in (("compare-prefill", "before.json", "after.json", "pp_throughput"),
                                ("compare-decode", "before.json", "after.json", "tg_throughput"),
                                ("compare-agent", "before-agent.json", "after-agent.json", None)):
        if not (os.path.exists(sc.path(a)) and os.path.exists(sc.path(b))):
            continue
        cmd = ["compare", sc.path(a), sc.path(b), *(["--metric", metric] if metric else [])]
        code, text = sc.run(label, cmd)
        status = "PASS" if code == 0 else "WARN"
        detail = (_compare_summary(text) or f"see {label}.log") + ("" if code == 0 else f" (exit {code})")
        if metric == "pp_throughput" and _warm(_load(sc.path(b))):
            # Measured live: an unrestarted server read +429% at pp8192 here.
            status = "WARN"
            detail = "not a result: the after side read from the prefix cache -- " + detail
        sc.add(Step(label, status, detail, cmd))
        extra += [f"--- {label} ---", text.rstrip()]
    if state["before"].get("note") or note:
        extra.insert(0, f"before: {state['before'].get('note') or '-'}   after: {note or '-'}")
    return sc.verdict(extra)


# --- command line ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py scenario",
        description="Answer one question about an endpoint end to end: qualify a "
                    "deployment, or A/B a change to it.")
    sub = ap.add_subparsers(dest="scenario", required=True)

    q = sub.add_parser("qualify", help="is this deployment served right, before measuring it?")
    q.add_argument("base_url", help="OpenAI-compatible endpoint URL, including /v1")
    q.add_argument("--model", help="pick this model when the endpoint serves several")
    q.add_argument("--dir", help="where to write it all (default results/qualify-<time>)")
    q.add_argument("--no-probe", action="store_true", help="skip the short probe")
    q.add_argument("--probe-depth", type=int, default=4096)
    q.add_argument("--probe-trials", type=int, default=2)
    q.add_argument("--agent", metavar="HARNESS",
                   help="also run the agent on three quick tasks, once each, through "
                        "HARNESS (pi, hermes, opencode, omp) or 'auto' for the first "
                        "one on PATH")
    q.add_argument("--harness-cmd", help="the harness executable, if not on PATH")

    ab = sub.add_parser("ab-deploy", help="did a change to the server help? "
                                          "`before`, change it, then `after`")
    ab_sub = ab.add_subparsers(dest="side", required=True)
    b = ab_sub.add_parser("before", help="record the endpoint and measure it")
    b.add_argument("base_url", help="OpenAI-compatible endpoint URL, including /v1")
    b.add_argument("--dir", required=True, help="the A/B's directory; `after` reads it")
    b.add_argument("--model", help="pick this model when the endpoint serves several")
    b.add_argument("--runs", type=int, default=5, help="runs per shape (default 5)")
    b.add_argument("--seed", type=int, help="the seed both sides use (default: a fresh one)")
    b.add_argument("--note", help="what this side is, e.g. 'kv fp8'")
    b.add_argument("--agent", metavar="HARNESS",
                   help="also run an agent arm on each side, through HARNESS or 'auto'")
    b.add_argument("--agent-tasks", nargs="+", default=["hard"])
    b.add_argument("--agent-runs", type=int, default=5)
    b.add_argument("--agent-workers", type=int, default=1)
    b.add_argument("--harness-cmd", help="the harness executable, if not on PATH")
    b.add_argument("--force", action="store_true", help="overwrite an existing before side")
    a = ab_sub.add_parser("after", help="read the endpoint again, measure, compare")
    a.add_argument("--dir", required=True, help="the directory `before` wrote")
    a.add_argument("--note", help="what changed, e.g. 'kv auto' -- needed when the "
                                  "endpoint cannot show it")
    a.add_argument("--force", action="store_true", help="measure the after side again")

    args = ap.parse_args()
    if args.scenario == "qualify":
        return qualify(args)
    return ab_before(args) if args.side == "before" else ab_after(args)


if __name__ == "__main__":
    raise SystemExit(main())
