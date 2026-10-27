"""Next-token divergence between two servings of one model.

The agent suite says whether a serving change -- KV-cache dtype, weight
quantisation, TP size, engine version -- changes outcomes, and 240 pass/fail
runs see only a large effect: fp8 against BF16 KV on the 27B came out 210
against 215, "no difference", where a drop to ~190 would have been needed to
reach p < 0.05. This measures the distribution instead, on the prefixes an
agent arm actually sent:

``capture``
    samples prefixes from a kept arm's ``requests.jsonl``, renders each to token
    ids once (``/tokenize``, with the harness's own tool schemas), lets the
    reference continue it greedily, and scores the reference at chosen positions
    of that continuation;
``score``
    scores the same token ids on another serving: one single-token request per
    position, top-20 logprobs, so both sides see exactly the same input and the
    same code path generation uses;
``compare``
    divergence from the reference, per prefix: top-1 disagreement and KL. The
    verdict is a candidate against a *floor* -- the reference scored again after
    a restart -- because two scorings of one config are never identical, and a
    nonzero KL without its floor means nothing.

Single-token requests rather than ``prompt_logprobs``, which materialise logits
over the whole vocabulary for every prompt position: on a server pinned near
its memory limit that is an out-of-memory crash, measured on the 27B. A
position is then scored the way generation reads it, from the KV cache.

The KL is over the reference's top-20 tokens plus one bucket for the rest, with
a candidate token outside its own top-20 taken at its 20th probability: an
estimate of the true divergence, and the standard one for top-k logprobs.

Divergence is not harm. A serving can diverge and pass as often; this is the
sensitive early signal, to be read beside the suite, never instead of it.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from . import __version__
from .servermetrics import endpoint_identity, identity_drift, server_settings

TOP_K = 20

#: Tokenised on both sides before anything is scored: a different vocabulary
#: makes every comparison meaningless, and ids that differ here say so.
PROBE = [{"role": "user", "content": "The quick brown fox, 12345, Tokyo 東京, "
                                     "def f(x): return x**2  # ok?"}]


def _root(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "EMPTY" else {}


# --- the statistics ---------------------------------------------------------

def kl(p: Dict[int, float], q: Dict[int, float]) -> float:
    """KL(p || q) from two top-k logprob maps, over p's tokens plus the rest.

    A token in p's top-k but not in q's is at most q's k-th probability, which
    is what it is taken at, so a candidate that drops a token is charged for it.
    """
    pp = {t: math.exp(v) for t, v in p.items()}
    qq = {t: math.exp(v) for t, v in q.items()}
    q_rest_all = max(1.0 - sum(qq.values()), 1e-12)
    floor = min(min(qq.values()) if qq else 1e-12, q_rest_all)
    total, q_used = 0.0, 0.0
    for t, pv in pp.items():
        qv = qq.get(t, floor)
        total += pv * math.log(pv / qv)
        q_used += qv
    p_rest = max(1.0 - sum(pp.values()), 0.0)
    if p_rest > 1e-12:
        total += p_rest * math.log(p_rest / max(1.0 - q_used, 1e-12))
    return max(total, 0.0)


def top1(p: Dict[int, float]) -> Optional[int]:
    return max(p, key=lambda t: p[t]) if p else None


def wilcoxon(diffs: List[float]) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p, normal approximation with tie correction.

    Zero differences are dropped (Wilcoxon's method). None when nothing differs.
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i, ties = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return None
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


# --- the prefixes -----------------------------------------------------------

def _arm(result: str) -> Tuple[str, Dict[str, Any]]:
    from .agent_compare import load
    arms = load(result)
    if len(arms) != 1:
        raise ValueError(f"{result} holds {len(arms)} arms; name one as FILE#ARM")
    return arms[0]


def select(result: str, runs_dir: str, per_task: int, seed: int,
           max_len: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """Seeded, per task: which recorded requests to replay.

    A request is a candidate when it went to a chat endpoint and its recorded
    prompt plus the continuation fits the reference's window.
    """
    label, report = _arm(result)
    task_of: Dict[str, str] = {}
    versions = set()
    for run in report.get("runs") or []:
        if run.get("kept") and not run.get("void"):
            task_of[os.path.basename(run["kept"].rstrip("/"))] = run["task"]
            versions.add(run.get("harness_version"))
    pool: Dict[str, List[Dict[str, Any]]] = {}
    tools: set = set()
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "requests.jsonl"), recursive=True)):
        task = task_of.get(os.path.basename(os.path.dirname(path)))
        if task is None:
            continue
        with open(path) as fh:
            for n, line in enumerate(fh):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                prompt = (r.get("usage") or {}).get("prompt_tokens")
                if not str(r.get("path", "")).endswith("chat/completions") or not prompt:
                    continue
                if prompt > max_len:
                    continue
                tools.update(r.get("tools") or [])
                pool.setdefault(task, []).append({"task": task, "file": path, "line": n,
                                                  "prompt_tokens": prompt})
    chosen = []
    for task in sorted(pool):
        rng = random.Random(f"{seed}:{task}")
        chosen += rng.sample(pool[task], min(per_task, len(pool[task])))
    source = {"result": result, "arm": label, "thinking": report.get("thinking"),
              "harness_versions": sorted(v for v in versions if v), "kit": report.get("kit"),
              "served_model": next((r.get("served_model") for r in report.get("runs") or []
                                    if r.get("served_model")), None)}
    return source, chosen, sorted(tools)


def _request(entry: Dict[str, Any]) -> Dict[str, Any]:
    with open(entry["file"]) as fh:
        for n, line in enumerate(fh):
            if n == entry["line"]:
                req: Dict[str, Any] = json.loads(line)
                return req
    raise ValueError(f"{entry['file']} has no line {entry['line']}")


# --- the harness's tools ----------------------------------------------------

async def harness_tools(harness: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """The tool schemas the installed harness sends, read off its first request.

    ``requests.jsonl`` records which tools a request offered, by name; the
    schemas are part of the prompt, so the replay needs them. A stub endpoint
    answers the harness's first request with "OK" and keeps its body.
    """
    from .agent import _base_env
    got: List[Dict[str, Any]] = []

    async def chat(req: web.Request) -> web.StreamResponse:
        got.append(await req.json())
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for delta in ({"role": "assistant", "content": "OK"}, {}):
            chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                     "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": None if delta else "stop"}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    scratch = tempfile.mkdtemp()
    try:
        home = os.path.join(scratch, ".agent")
        harness.configure(home, f"http://127.0.0.1:{port}/v1", "m", False, 131072, 1024)
        ws = os.path.join(scratch, "ws")
        os.makedirs(ws)
        proc = await asyncio.create_subprocess_exec(
            *harness.argv("medium", "Reply OK."), cwd=ws, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env={**_base_env(os.path.join(scratch, "home")), **harness.env(home)})
        try:
            await asyncio.wait_for(proc.wait(), 120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    finally:
        await runner.cleanup()
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    if not got:
        raise RuntimeError("the harness sent no request to the stub endpoint")
    return await asyncio.to_thread(harness.version), list(got[0].get("tools") or [])


# --- talking to the endpoint ------------------------------------------------

class Endpoint:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, model: str, api_key: str):
        self.session, self.base_url, self.model, self.api_key = session, base_url, model, api_key

    async def _post(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        last = ""
        for _ in range(2):
            try:
                async with self.session.post(url, json=body, headers=_headers(self.api_key)) as r:
                    data = await r.json(content_type=None)
                    if r.status == 200 and isinstance(data, dict):
                        return data
                    last = f"HTTP {r.status}: {str(data)[:300]}"
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                last = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(last)

    async def tokenize_chat(self, messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[int], Optional[int]]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages,
                                "add_generation_prompt": True}
        if tools:
            body["tools"] = tools
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        data = await self._post(_root(self.base_url) + "/tokenize", body)
        return [int(t) for t in data["tokens"]], data.get("max_model_len")

    async def token_id(self, text: str) -> Optional[int]:
        data = await self._post(_root(self.base_url) + "/tokenize",
                                {"model": self.model, "prompt": text, "add_special_tokens": False})
        ids = data.get("tokens") or []
        return int(ids[0]) if len(ids) == 1 else None

    async def _complete(self, ids: List[int], max_tokens: int, logprobs: int) -> Dict[str, Any]:
        data = await self._post(self.base_url.rstrip("/") + "/completions", {
            "model": self.model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
            "logprobs": logprobs, "return_tokens_as_token_ids": True})
        logprobs_: Dict[str, Any] = data["choices"][0]["logprobs"]
        return logprobs_

    async def greedy(self, ids: List[int], length: int) -> List[int]:
        lp = await self._complete(ids, length, 1)
        return [_tid(t) for t in lp["tokens"]]

    async def next_token(self, ids: List[int]) -> Dict[int, float]:
        lp = await self._complete(ids, 1, TOP_K)
        return {_tid(t): float(v) for t, v in (lp["top_logprobs"][0] or {}).items()}


def _tid(token: str) -> int:
    return int(str(token).rsplit(":", 1)[-1])


def _kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """The template arguments a recorded request reached the template with."""
    out = dict(params.get("chat_template_kwargs") or {})
    if "reasoning_effort" in params:
        out.setdefault("reasoning_effort", params["reasoning_effort"])
    return out


async def _describe(session: aiohttp.ClientSession, base_url: str, api_key: str,
                    model: str) -> Dict[str, Any]:
    return {"base_url": base_url, "model": model,
            **await endpoint_identity(session, base_url, api_key, model),
            "server_settings": await server_settings(session, base_url, api_key)}


async def _score_all(ep: Endpoint, prefixes: List[Dict[str, Any]], concurrency: int,
                     max_len: Optional[int]) -> Tuple[Dict[str, Any], List[str]]:
    """Every chosen position of every prefix: top-20 next-token logprobs."""
    sem = asyncio.Semaphore(max(1, concurrency))
    out: Dict[str, Any] = {}
    errors: List[str] = []
    done = 0

    async def one(p: Dict[str, Any]) -> None:
        nonlocal done
        if max_len and len(p["ids"]) + len(p["continuation"]) > max_len:
            errors.append(f"{p['id']}: longer than this endpoint's window")
            return
        async with sem:
            scores: List[Optional[Dict[str, float]]] = []
            for i in p["positions"]:
                try:
                    scores.append({str(k): v for k, v in
                                   (await ep.next_token(p["ids"] + p["continuation"][:i])).items()})
                except RuntimeError as exc:
                    errors.append(f"{p['id']}@{i}: {str(exc)[:200]}")
                    scores.append(None)
            out[p["id"]] = scores
        done += 1
        print(f"  scored {done}/{len(prefixes)}", file=sys.stderr, flush=True)

    await asyncio.gather(*(one(p) for p in prefixes))
    return out, errors


# --- the subcommands --------------------------------------------------------

async def capture(args: argparse.Namespace) -> int:
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        ep = Endpoint(session, args.base_url, args.model, args.api_key)
        before = await _describe(session, args.base_url, args.api_key, args.model)
        probe_ids, max_len = await ep.tokenize_chat(PROBE)
        window = max_len or 131072
        source, chosen, offered = select(args.result, args.runs_dir, args.per_task, args.seed,
                                         window - args.length)
        if not chosen:
            print("error: no recorded request to replay (is RUNS_DIR the arm's --keep "
                  "directory?)", file=sys.stderr)
            return 1
        if args.tools:
            with open(args.tools) as fh:
                tools, tools_from = json.load(fh), args.tools
        else:
            from .agent import Pi
            version, tools = await harness_tools(Pi(args.harness_cmd))
            tools_from = version or "pi"
            if source["harness_versions"] and version not in source["harness_versions"]:
                print(f"error: the arm ran {', '.join(source['harness_versions'])}, the "
                      f"installed harness is {version}: its tool schemas may differ; pass "
                      f"--tools", file=sys.stderr)
                return 1
        names = sorted(t.get("function", {}).get("name", "") for t in tools)
        if offered and names != offered:
            print(f"error: the arm offered tools {offered}, the schemas name {names}",
                  file=sys.stderr)
            return 1
        open_id, close_id = await ep.token_id("<think>"), await ep.token_id("</think>")

        prefixes: List[Dict[str, Any]] = []
        skipped: List[str] = []
        for n, entry in enumerate(chosen):
            req = _request(entry)
            pid = f"{entry['task']}/{os.path.basename(os.path.dirname(entry['file']))}/{entry['line']}"
            ids, _ = await ep.tokenize_chat(req["messages"], tools, _kwargs(req.get("params") or {}))
            # The recorded usage is the server's own count of what the harness
            # sent; a rendering that disagrees is not the prefix the arm saw.
            if len(ids) != entry["prompt_tokens"]:
                skipped.append(f"{pid}: renders to {len(ids)} tokens, the arm sent "
                               f"{entry['prompt_tokens']}")
                continue
            cont = await ep.greedy(ids, args.length)
            if not cont:
                skipped.append(f"{pid}: the reference generated nothing")
                continue
            # Position 0 always: the first token is where a prefix's own state decides most.
            rng = random.Random(f"{args.seed}:{pid}")
            rest = rng.sample(range(1, len(cont)), min(args.positions - 1, len(cont) - 1))
            positions = sorted([0, *rest])
            tail = ids[-8:]
            opened = (open_id is not None and open_id in tail and
                      (close_id not in tail or
                       max(i for i, t in enumerate(tail) if t == open_id) >
                       max(i for i, t in enumerate(tail) if t == close_id)))
            closes = cont.index(close_id) if close_id in cont else len(cont)
            regions = ["reasoning" if opened and i <= closes else "output" for i in positions]
            prefixes.append({"id": pid, "task": entry["task"], "ids": ids, "continuation": cont,
                             "positions": positions, "regions": regions})
            print(f"  captured {n + 1}/{len(chosen)}", file=sys.stderr, flush=True)

        scores, errors = await _score_all(ep, prefixes, args.concurrency, max_len)
        after = await _describe(session, args.base_url, args.api_key, args.model)
    drift = identity_drift(before, after)
    agree = [s is not None and top1({int(k): v for k, v in s.items()}) == p["continuation"][i]
             for p in prefixes for s, i in zip(scores.get(p["id"], []), p["positions"])]
    doc = {"kind": "divergence-reference", "version": __version__, "source": source,
           "endpoint": before, "probe_ids": probe_ids, "tools_from": tools_from,
           "length": args.length, "seed": args.seed, "prefixes": prefixes, "scores": scores,
           "skipped": skipped, "errors": errors, "void": drift}
    with open(args.save, "w") as fh:
        json.dump(doc, fh)
    print(f"{len(prefixes)} prefixes captured ({len(skipped)} skipped), "
          f"{sum(len(p['positions']) for p in prefixes)} positions scored; the reference's own "
          f"top-1 matches its greedy token at {sum(agree)}/{len(agree)}")
    for line in skipped[:5] + errors[:5]:
        print(f"  note: {line}")
    print(f"Saved to {args.save}")
    if drift:
        print("void: " + "; ".join(drift), file=sys.stderr)
        return 4
    return 0 if prefixes else 1


async def score(args: argparse.Namespace) -> int:
    with open(args.reference) as fh:
        ref = json.load(fh)
    if ref.get("kind") != "divergence-reference":
        print(f"error: {args.reference} is not a divergence reference", file=sys.stderr)
        return 1
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        ep = Endpoint(session, args.base_url, args.model, args.api_key)
        before = await _describe(session, args.base_url, args.api_key, args.model)
        probe_ids, max_len = await ep.tokenize_chat(PROBE)
        if probe_ids != ref["probe_ids"]:
            print("error: this endpoint tokenises the probe differently from the reference: "
                  "another vocabulary or chat template, and KL across them is meaningless",
                  file=sys.stderr)
            return 1
        scores, errors = await _score_all(ep, ref["prefixes"], args.concurrency, max_len)
        after = await _describe(session, args.base_url, args.api_key, args.model)
    drift = identity_drift(before, after)
    doc = {"kind": "divergence-score", "version": __version__, "reference": args.reference,
           "reference_endpoint": ref.get("endpoint"), "endpoint": before, "scores": scores,
           "errors": errors, "void": drift}
    with open(args.save, "w") as fh:
        json.dump(doc, fh)
    n = sum(1 for s in scores.values() for x in s if x is not None)
    print(f"{n} positions scored, {len(errors)} errors")
    for line in errors[:5]:
        print(f"  note: {line}")
    print(f"Saved to {args.save}")
    if drift:
        print("void: " + "; ".join(drift), file=sys.stderr)
        return 4
    return 0 if n else 1


def _per_prefix(ref: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each prefix's divergence from the reference: mean KL, top-1 disagreement, by region."""
    out = {}
    for p in ref["prefixes"]:
        a, b = ref["scores"].get(p["id"]), scores.get(p["id"])
        if not a or not b:
            continue
        rows = []
        for x, y, region in zip(a, b, p["regions"]):
            if x is None or y is None:
                continue
            px = {int(k): v for k, v in x.items()}
            qy = {int(k): v for k, v in y.items()}
            rows.append((kl(px, qy), top1(px) != top1(qy), region))
        if rows:
            out[p["id"]] = {"task": p["task"], "rows": rows}
    return out


def _summary(per: Dict[str, Dict[str, Any]], ids: List[str],
             region: Optional[str] = None) -> Dict[str, Any]:
    rows = [r for i in ids for r in per[i]["rows"] if region is None or r[2] == region]
    if not rows:
        return {"positions": 0}
    kls = sorted(r[0] for r in rows)
    return {"positions": len(rows), "mean_kl": sum(kls) / len(kls),
            "p99_kl": kls[min(len(kls) - 1, int(0.99 * len(kls)))],
            "top1_disagree": sum(r[1] for r in rows) / len(rows)}


def analyse(ref: Dict[str, Any], floor: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    pf, pc = _per_prefix(ref, floor["scores"]), _per_prefix(ref, cand["scores"])
    ids = sorted(set(pf) & set(pc))

    def mean(per: Dict[str, Any], i: str, field: int) -> float:
        rows = per[i]["rows"]
        return sum(float(r[field]) for r in rows) / len(rows)

    tasks = sorted({pf[i]["task"] for i in ids})
    table = []
    for label, sel, region in ([(t, [i for i in ids if pf[i]["task"] == t], None) for t in tasks]
                               + [("all", ids, None), ("all: reasoning", ids, "reasoning"),
                                  ("all: output", ids, "output")]):
        sel = [i for i in sel if region is None or any(r[2] == region for r in pf[i]["rows"])]
        table.append({"label": label, "prefixes": len(sel), "floor": _summary(pf, sel, region),
                      "candidate": _summary(pc, sel, region)})
    return {"prefixes": len(ids), "table": table,
            "p_kl": wilcoxon([mean(pc, i, 0) - mean(pf, i, 0) for i in ids]),
            "p_top1": wilcoxon([mean(pc, i, 1) - mean(pf, i, 1) for i in ids])}


def _fmt(s: Dict[str, Any], key: str, pct: bool = False) -> str:
    if not s.get("positions"):
        return "-"
    return f"{100 * s[key]:.2f}%" if pct else f"{s[key]:.4f}"


def render(result: Dict[str, Any], names: Tuple[str, str, str], alpha: float) -> str:
    lines = [f"divergence | reference: {names[0]} | floor: {names[1]} | candidate: {names[2]}",
             "", "| task | prefixes | top-1 disagree: floor | candidate | mean KL: floor | "
                 "candidate | p99 KL: floor | candidate |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for row in result["table"]:
        f, c = row["floor"], row["candidate"]
        label = f"**{row['label']}**" if row["label"].startswith("all") else row["label"]
        lines.append(f"| {label} | {row['prefixes']} | {_fmt(f, 'top1_disagree', True)} | "
                     f"{_fmt(c, 'top1_disagree', True)} | {_fmt(f, 'mean_kl')} | "
                     f"{_fmt(c, 'mean_kl')} | {_fmt(f, 'p99_kl')} | {_fmt(c, 'p99_kl')} |")

    def verdict(p: Optional[float], what: str) -> str:
        if p is None:
            return f"{what}: no prefix differs from the floor"
        word = "DIFFERS from the floor" if p < alpha else "within the floor"
        return f"{what}: p={p:.4f}, {word}"
    lines += ["", verdict(result["p_kl"], "mean KL"), verdict(result["p_top1"], "top-1"),
              "", "Paired by prefix, Wilcoxon signed-rank: the candidate's divergence from the "
                  "reference against the floor's.",
              "Divergence is not harm: read it beside the agent suite, not instead of it."]
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> int:
    docs = []
    for path, kind in ((args.reference, "divergence-reference"), (args.floor, "divergence-score"),
                       (args.candidate, "divergence-score")):
        with open(path) as fh:
            doc = json.load(fh)
        if doc.get("kind") != kind:
            print(f"error: {path} is not a {kind}", file=sys.stderr)
            return 1
        if doc.get("void"):
            print(f"warning: {path} is void: {'; '.join(doc['void'])}", file=sys.stderr)
        docs.append(doc)
    ref, floor, cand = docs
    result = analyse(ref, floor, cand)
    if not result["prefixes"]:
        print("error: no prefix was scored on both sides", file=sys.stderr)
        return 1
    settings = [(os.path.basename(p), (d.get("endpoint") or {}).get("server_settings") or {})
                for p, d in ((args.reference, ref), (args.floor, floor), (args.candidate, cand))]
    for key in sorted({k for _, s in settings for k in s}):
        values = {name: s.get(key) for name, s in settings}
        if len({json.dumps(v) for v in values.values()}) > 1:
            print(f"setting {key}: " + ", ".join(f"{n}={v}" for n, v in values.items()))
    names = (os.path.basename(args.reference), os.path.basename(args.floor),
             os.path.basename(args.candidate))
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(render(result, names, args.alpha))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Next-token divergence between two servings of one model, on the "
                    "prefixes an agent arm sent.")
    sub = ap.add_subparsers(dest="step", required=True)

    def endpoint(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (vLLM)")
        p.add_argument("--model", required=True, help="the name the endpoint serves under")
        p.add_argument("--api-key", help="prefer the LLM_ASSAY_API_KEY environment variable: "
                                         "a key on the command line is visible in ps")
        p.add_argument("--concurrency", type=int, default=1,
                       help="prefixes scored at once (default 1: batching moves logits)")
        p.add_argument("--save", required=True, help="where to write the result")

    c = sub.add_parser("capture", help="sample prefixes, continue them on the reference, score it")
    c.add_argument("result", help="the arm's saved agent result (FILE#ARM for one arm of many)")
    c.add_argument("runs_dir", help="the arm's --keep directory (its requests.jsonl files)")
    endpoint(c)
    c.add_argument("--per-task", type=int, default=10, help="prefixes per task (default 10)")
    c.add_argument("--length", type=int, default=256,
                   help="greedy continuation tokens (default 256)")
    c.add_argument("--positions", type=int, default=64,
                   help="continuation positions scored per prefix (default 64)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--tools", help="the harness's tool schemas as JSON, instead of reading "
                                   "them from the installed harness")
    c.add_argument("--harness-cmd", help="the harness executable, if not on PATH")

    s = sub.add_parser("score", help="score a reference's positions on another serving")
    s.add_argument("reference")
    endpoint(s)

    m = sub.add_parser("compare", help="a candidate against the floor, both from one reference")
    m.add_argument("reference")
    m.add_argument("floor", help="the reference config scored again, after a restart")
    m.add_argument("candidate")
    m.add_argument("--alpha", type=float, default=0.05)
    m.add_argument("--json", dest="as_json", action="store_true")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.step != "compare":
        from .config import resolve_api_key
        args.api_key = resolve_api_key(args.api_key)
    if args.step == "compare":
        return compare(args)
    return asyncio.run(capture(args) if args.step == "capture" else score(args))


if __name__ == "__main__":
    raise SystemExit(main())
