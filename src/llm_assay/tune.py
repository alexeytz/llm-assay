"""Detect what an endpoint is, then propose runs that fit it.

The alternative this replaces was a hardcoded table of hosts: model names,
context sizes and depth ladders written down by hand for two boxes on one
private network. That table was wrong the moment a server was restarted with a
different ``--max-model-len``, and useless to anyone else.

Everything it encoded is available from the endpoint itself:

* ``/v1/models`` gives the served name, the HuggingFace root (which selects the
  tokenizer) and ``max_model_len`` -- the ceiling every depth must fit under.
* ``/metrics`` carries ``vllm:cache_config_info``, whose labels state outright
  whether prefix caching is enabled, how many tokens of KV cache exist, and how
  many max-length sequences fit at once. Those three numbers are what makes a
  depth ladder and a concurrency ladder *safe* rather than aspirational.
* The speculative-decode counters say whether drafting is active, which decides
  how many repetitions a decode measurement needs to mean anything.

Detection is best-effort throughout. A server that exposes none of this still
gets suggestions, built from conservative defaults and labelled as such -- the
subcommand degrades, it does not fail.

Usage::

    python llm-assay.py tune http://localhost:8000/v1
    python llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import __version__
# The engine-name families live there, and are imported rather than restated:
# servermetrics had already been taught llama.cpp when tune had not, and two
# copies of a list like that is exactly how the stale one goes unnoticed.
from .servermetrics import _BUILD_FIELDS, _build_summary
from .servermetrics import _COUNTERS as _SM_COUNTERS

# Room left for the prompt, the generated tokens and the chat template on top of
# any depth we propose. Nothing is gained by benchmarking a shape the server will
# reject, and max_model_len is a hard ceiling.
_HEADROOM_TOKENS = 4096

# Depth ladders are powers of two; this is where one starts being interesting.
_MIN_LADDER_DEPTH = 4096

# Concurrency is capped here regardless of KV capacity. Past this the numbers say
# more about the client and the network than about the server.
_MAX_CONCURRENCY = 16


class Detected:
    """What could be learned about an endpoint, plus what could not."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.served_model: Optional[str] = None
        self.hf_model: Optional[str] = None
        self.max_model_len: Optional[int] = None
        self.server_version: Optional[str] = None
        self.prefix_caching: Optional[bool] = None
        self.kv_cache_tokens: Optional[int] = None
        self.kv_max_concurrency: Optional[float] = None
        self.block_size: Optional[int] = None
        self.sliding_window: Optional[int] = None
        self.served_build: Optional[Dict[str, str]] = None
        self.spec_decode: Optional[bool] = None
        self.spec_acceptance_length: Optional[float] = None
        self.thinking_controls: Optional[Dict[str, bool]] = None
        self.notes: List[str] = []

    @property
    def usable_context(self) -> int:
        """Largest depth worth proposing, leaving room for prompt + generation."""
        if self.max_model_len is None:
            return 32768  # conservative: fits essentially every served model
        return max(0, self.max_model_len - _HEADROOM_TOKENS)


def _root(base_url: str) -> str:
    """Strip the trailing /v1 so /metrics and /version can be reached."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "EMPTY" else {}


def _get(url: str, api_key: str, timeout: float = 8.0) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException:
        return None
    return resp if resp.status_code == 200 else None


def _read_props(root: str, api_key: str) -> Dict[str, Any]:
    """llama.cpp's ``/props``, or an empty dict. Read once, used twice."""
    props = _get(f"{root}/props", api_key, timeout=4.0)
    if props is None:
        return {}
    try:
        body = props.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _llamacpp_context(
    props: Dict[str, Any], entry: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Context length for an engine that publishes no ``max_model_len``.

    ``/props`` gives the *slot* context, which is the number that binds a single
    request: a box launched with ``-c`` split across ``-np`` slots serves less
    per request than its total. ``meta.n_ctx_train`` on ``/v1/models`` is the
    weaker fallback -- it is what the model was trained for, not what this
    server was launched with, so it can overstate a deliberately smaller run.
    """
    slot_ctx = _as_int((props.get("default_generation_settings") or {}).get("n_ctx"))
    if slot_ctx:
        return slot_ctx
    if entry:
        return _as_int((entry.get("meta") or {}).get("n_ctx_train"))
    return None


def _build_from_props(props: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """What the server says it loaded, when it says anything.

    ``/v1/models`` answers with the *alias*, and pinning that alias to something
    stable is sensible operations -- llama.cpp's default is the literal string
    ``llama.cpp`` -- so `tune` was naming a box by a string that identifies
    nothing, while the benchmark and the probe had both been taught to read
    this. A tuning report is exactly where the reader decides whether a saved
    result is still comparable, so it needs the same answer.
    """
    build = {
        field: str(props[field]) for field in _BUILD_FIELDS
        if isinstance(props.get(field), (str, int, float)) and str(props[field])
    }
    return build or None


def _any_counter(metrics_text: str, logical: str) -> Optional[float]:
    """First counter present for a logical value, across every engine spelling.

    The names live in ``servermetrics`` and are read from there rather than
    restated: they had already been taught llama.cpp when this had not, so
    `tune` reported "spec decode not active" against a box running MTP at 95%
    acceptance. A wrong fact is worse than an unknown, and two lists of engine
    names is how the first one gets stale.
    """
    for name in _SM_COUNTERS.get(logical, ()):
        value = _counter(metrics_text, name)
        if value is not None:
            return value
    return None


def _prefix_caching_from_evidence(
    metrics_text: str, found: "Detected",
) -> Optional[bool]:
    """Whether prefix caching is on, for an engine that states no flag.

    Only vLLM's ``cache_config_info`` asserts it. llama.cpp publishes the
    *evidence* instead -- ``prompt_tokens_cached_total`` counts prompt tokens it
    served from cache -- so a counter above zero proves caching is both enabled
    and working, which is the thing a user actually needs to know before
    trusting `--measure-cached-followup`.

    Zero is **unknown, not off**: the counter is cumulative, so a freshly
    restarted server reads zero for a cache that is enabled and simply has not
    been hit yet. Same rule as the spec-decode counters above, and for the same
    reason -- reporting "off" would talk someone out of a mode that works.
    """
    served = _any_counter(metrics_text, "prefix_hits")
    if served is None:
        found.notes.append(
            "this engine publishes no prefix-caching flag and no cached-token "
            "counter, so the status is unknown; send one prompt twice and diff "
            "the counters if it matters"
        )
        return None
    if served > 0:
        return True
    found.notes.append(
        "prefix caching is instrumented but has served no cached tokens yet, so "
        "whether it is enabled is unknown rather than off"
    )
    return None


def _parse_cache_config(metrics_text: str) -> Dict[str, str]:
    """Pull the label set off vllm:cache_config_info.

    The engine publishes its whole CacheConfig as Prometheus labels on a gauge.
    It is the only place that states prefix-caching status as a fact rather than
    something to be inferred from hit rates after the fact.
    """
    for line in metrics_text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return {}


def _counter(metrics_text: str, name: str) -> Optional[float]:
    total = None
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        match = re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line.strip())
        if match:
            try:
                total = (total or 0.0) + float(match.group(1))
            except ValueError:
                pass
    return total


def _as_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def detect(base_url: str, api_key: str = "EMPTY", model: Optional[str] = None) -> Detected:
    """Interrogate an endpoint. Never raises for a server that answers partially."""
    found = Detected(base_url)
    root = _root(base_url)
    # One read, two answers: the slot context and the build. Both are things
    # only this endpoint can state, and neither is on /v1/models.
    props = _read_props(root, api_key)
    found.served_build = _build_from_props(props)

    resp = _get(f"{base_url.rstrip('/')}/models", api_key)
    if resp is None:
        found.notes.append(
            f"could not read {base_url.rstrip('/')}/models -- is the endpoint up, and does "
            "the URL include /v1?"
        )
    else:
        entries = resp.json().get("data") or []
        chosen = None
        if model:
            chosen = next(
                (e for e in entries if model in (e.get("id"), e.get("root"))), None
            )
            if chosen is None:
                found.notes.append(f"--model {model} is not served here; using the first model")
        chosen = chosen or (entries[0] if entries else None)
        if chosen:
            found.served_model = chosen.get("id")
            found.hf_model = chosen.get("root") or chosen.get("id")
            found.max_model_len = _as_int(chosen.get("max_model_len"))
            if len(entries) > 1:
                others = ", ".join(str(e.get("id")) for e in entries[1:])
                found.notes.append(f"endpoint serves more than one model; also available: {others}")
        if found.max_model_len is None:
            # vLLM states it on /v1/models; llama.cpp does not, and answering
            # "unknown" there cost an 8x underestimate -- a 262,144-token box
            # was handed a 32,768 ceiling and a "long" preset labelled the
            # deepest context this server accepts. Two fallbacks, in order of
            # how binding they are: the slot's own n_ctx is what a single
            # request may actually use, and the model's trained length is the
            # weaker claim to fall back on.
            found.max_model_len = _llamacpp_context(props, chosen)
        if found.max_model_len is None:
            found.notes.append(
                "endpoint did not report max_model_len; depths assume a conservative 32768"
            )

    if found.served_model:
        found.thinking_controls = probe_thinking_controls(
            base_url, found.served_model, api_key
        )

    version = _get(f"{root}/version", api_key, timeout=4.0)
    if version is not None:
        try:
            found.server_version = version.json().get("version")
        except ValueError:
            pass

    metrics = _get(f"{root}/metrics", api_key)
    if metrics is None:
        found.notes.append(
            f"no readable {root}/metrics -- prefix-caching status and KV capacity are unknown, "
            "so the suggestions below are conservative"
        )
        return found

    labels = _parse_cache_config(metrics.text)
    if not labels:
        # Readable metrics that simply do not carry vLLM's config block. Saying
        # "/metrics unreadable" there blames the wrong thing and sends the
        # reader to check a URL that is answering fine -- and llama.cpp does
        # publish the prefix-cache evidence, just not the flag.
        found.prefix_caching = _prefix_caching_from_evidence(metrics.text, found)
    if labels:
        flag = labels.get("enable_prefix_caching")
        if flag is not None:
            found.prefix_caching = flag.strip().lower() == "true"
        found.kv_cache_tokens = _as_int(labels.get("kv_cache_size_tokens"))
        found.block_size = _as_int(labels.get("block_size"))
        window = labels.get("sliding_window")
        if window and window != "None":
            found.sliding_window = _as_int(window)
        try:
            found.kv_max_concurrency = float(labels.get("kv_cache_max_concurrency"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    # These are cumulative counters, so a freshly restarted server reports zero
    # for a configured draft head that simply has not served a request yet.
    # Absent means not configured; present-but-zero means unknown, not "off" --
    # reporting "off" there would talk a user out of the adaptive sampling their
    # decode numbers are about to need.
    drafts = _any_counter(metrics.text, "spec_drafts")
    accepted = _any_counter(metrics.text, "spec_accepted")
    if drafts is None:
        found.spec_decode = False
    elif drafts > 0:
        found.spec_decode = True
        if accepted is not None:
            found.spec_acceptance_length = accepted / drafts + 1.0
    else:
        found.spec_decode = None
        found.notes.append(
            "speculative decoding is configured but has served no requests yet, so "
            "acceptance is unknown; if it is active expect wide decode spread"
        )

    return found


# Candidate ways to ask a server to stop thinking. Which one works is a property
# of the deployment's chat template, not of the model card: on an unsloth
# conversion of Qwen3.8 the graded levels the official card documents render a
# byte-identical prompt, while reasoning_effort="none" and enable_thinking=false
# both prefill an empty <think></think> block.
_THINKING_PROBES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("reasoning_effort=medium", {"reasoning_effort": "medium"}),
    ("reasoning_effort=xhigh", {"reasoning_effort": "xhigh"}),
)


def probe_thinking_controls(
    base_url: str, model: str, api_key: str = "EMPTY"
) -> Optional[Dict[str, bool]]:
    """Which thinking switches actually change the prompt this server builds.

    Uses /v1/chat/completions/render, so it costs no generation: render the same
    message with and without each switch and compare the token ids. If the
    rendered prompt is unchanged the model cannot possibly respond to the
    setting -- there is no channel for it -- which is a far more reliable check
    than generating twice and squinting at the token counts.

    Returns None when the endpoint has no render route.
    """
    url = _root(base_url) + "/v1/chat/completions/render"
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}

    def render(extra: Optional[Dict[str, Any]] = None) -> Optional[list]:
        try:
            resp = requests.post(
                url, json={**body, **(extra or {})}, headers=_headers(api_key), timeout=10
            )
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        try:
            ids = resp.json().get("token_ids")
        except ValueError:
            return None
        return list(ids) if isinstance(ids, list) else None

    baseline = render()
    if baseline is None:
        return None

    effective: Dict[str, bool] = {}
    for label, extra in _THINKING_PROBES:
        rendered = render(extra)
        if rendered is not None:
            effective[label] = rendered != baseline
    return effective or None


def depth_ladder(found: Detected, steps: int = 4) -> List[int]:
    """Powers of two that fit under max_model_len, plus a depth-0 baseline.

    Depth 0 is always included: it is the only row that isolates prefill cost
    from context, and it is the cheapest thing in any suite.
    """
    ceiling = found.usable_context
    ladder: List[int] = []
    depth = _MIN_LADDER_DEPTH
    while depth <= ceiling:
        ladder.append(depth)
        depth *= 4
    if not ladder:
        return [0]
    # Keep the extremes and thin the middle, so a 262144-token server does not
    # produce a ten-row suite nobody will wait for.
    if len(ladder) > steps:
        keep = {0, len(ladder) - 1}
        while len(keep) < steps:
            keep.add(len(ladder) * len(keep) // steps)
        ladder = [ladder[i] for i in sorted(keep)]
    return [0] + ladder


def max_depth_for_long_run(found: Detected) -> Optional[int]:
    """The deepest round depth this server can actually take."""
    ceiling = found.usable_context
    if ceiling < _MIN_LADDER_DEPTH:
        return None
    # Round down to a multiple of 1024 so the number reads as deliberate.
    return (ceiling // 1024) * 1024


def concurrency_ladder(found: Detected, depth: int, pp: int, tg: int) -> List[int]:
    """Concurrency levels whose combined KV footprint the server can hold.

    Oversubscribing does not error -- vLLM queues instead -- but a queued request
    measures scheduling, not throughput, and quietly ruins the row.
    """
    if not found.kv_cache_tokens:
        return [1, 4]
    per_request = max(1, depth + pp + tg)
    capacity = int(found.kv_cache_tokens // per_request)
    levels = [1]
    level = 4
    while level <= min(capacity, _MAX_CONCURRENCY):
        levels.append(level)
        level *= 4
    return levels


def _fmt_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "unknown"


def report(found: Detected) -> str:
    """The detection summary, printed before any suggestion."""
    lines = ["", "Detected"]

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<14}{value}")

    endpoint = found.base_url
    if found.server_version:
        endpoint += f"  (vLLM {found.server_version})"
    row("endpoint", endpoint)

    if found.hf_model:
        model = found.hf_model
        if found.served_model and found.served_model != found.hf_model:
            model += f"  (served as {found.served_model})"
        row("model", model)
    else:
        row("model", "unknown")

    if found.served_build:
        # Beside the model, never instead of it: one is what the endpoint
        # answers to, the other is what it actually loaded.
        row("build", _build_summary(found.served_build))

    row("max ctx", f"{_fmt_int(found.max_model_len)} tokens")

    if found.prefix_caching is None:
        # Not always an unreadable endpoint any more: an engine can publish
        # readable metrics that simply carry no flag and no cached-token
        # counter. Every path that leaves this None appends a note saying
        # which, so the row points there instead of naming one cause.
        row("prefix cache", "unknown -- see note below")
    elif found.prefix_caching:
        row("prefix cache", "ENABLED -- cached-follow-up measurement is meaningful here")
    else:
        row("prefix cache", "DISABLED -- --measure-cached-followup would measure a re-prefill")

    if found.spec_decode:
        detail = "ACTIVE"
        if found.spec_acceptance_length:
            detail += f" (accept len {found.spec_acceptance_length:.2f})"
        detail += " -- expect wide decode spread; sample adaptively"
        row("spec decode", detail)
    elif found.spec_decode is False:
        row("spec decode", "not active")
    else:
        row("spec decode", "configured, no traffic yet -- acceptance unknown")

    if found.kv_cache_tokens:
        detail = f"{_fmt_int(found.kv_cache_tokens)} tokens"
        if found.kv_max_concurrency:
            detail += f" (~{found.kv_max_concurrency:.1f} seqs at max ctx)"
        row("kv cache", detail)

    if found.thinking_controls:
        live = [k for k, v in found.thinking_controls.items() if v]
        inert = [k for k, v in found.thinking_controls.items() if not v]
        row("thinking", ", ".join(live) if live else "no switch changes the prompt")
        if inert:
            lines.append(f"  {'':<14}inert here: {', '.join(inert)}")

    if found.sliding_window:
        row("sliding win", f"{_fmt_int(found.sliding_window)} tokens")

    for note in found.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def cache_guidance(found: Detected) -> str:
    """What this endpoint's cache state means for the runs suggested below.

    The detection block states the fact and the commands embody the
    consequence, but nothing connected them: a reader had to know that
    `--seed $RANDOM` is there because caching is on, and that the absence of a
    cached-follow-up preset is a decision rather than an oversight. Prefix
    caching is the single easiest way to publish a confident wrong throughput
    number, so the consequence is spelled out where the commands are.
    """
    if found.prefix_caching:
        return (
            "  prefix cache is ENABLED here, and two things below follow from it.\n"
            "  Every preset uses --seed $RANDOM: a fixed seed sends byte-identical\n"
            "  prompts, so a repeat invocation is served from the cache the previous\n"
            "  one filled. Measured that way, prefill read 7.9k t/s cold and 26k t/s\n"
            "  warm, and `compare` reported +110% between a configuration and itself.\n"
            "  And --measure-cached-followup is included where it belongs, because\n"
            "  this server can actually honour it. Only prefill and TTFT move;\n"
            "  decode numbers are unaffected either way."
        )
    if found.prefix_caching is False:
        return (
            "  prefix cache is DISABLED here, so every prefill below is genuinely\n"
            "  cold and those numbers need no defending. --measure-cached-followup is\n"
            "  deliberately absent: with caching off its second request re-prefills\n"
            "  the whole context and would be reported as a cached follow-up -- about\n"
            "  a 10x misread at depth, with plausible-looking timings either way.\n"
            "  Enable caching at the server first if that is the measurement you want."
        )
    return (
        "  prefix cache status is UNKNOWN here, so the runs below assume nothing:\n"
        "  --seed $RANDOM everywhere, and no cached-follow-up preset. To settle it,\n"
        "  run the smoke preset twice with a FIXED seed and compare pp -- a large\n"
        "  jump on the second run is the cache, not the server. llm-assay also warns\n"
        "  on its own when a run it did not expect to be cached was served from one."
    )


def _target_args(found: Detected) -> List[str]:
    """The flags that identify the endpoint, shared by every suggested run."""
    args = ["--base-url", found.base_url]
    if found.hf_model:
        args += ["--model", found.hf_model]
    if found.served_model and found.served_model != found.hf_model:
        args += ["--served-model-name", found.served_model]
    return args


def build_presets(found: Detected) -> List[Tuple[str, str, List[str]]]:
    """(key, description, args) for each suggested run, sized to this endpoint."""
    ladder = depth_ladder(found)
    sweep_depths = [str(d) for d in ladder]
    deep = ladder[-1] if ladder[-1] else 0
    conc = concurrency_ladder(found, deep, 2048, 256)
    cached = ["--measure-cached-followup"] if found.prefix_caching else []
    presets: List[Tuple[str, str, List[str]]] = []

    presets.append((
        "smoke",
        "is the endpoint alive and sane -- run this first",
        ["--pp", "512", "--tg", "64", "--exact-tg", "--depth", "0", "--runs", "3"],
    ))

    presets.append((
        "sweep",
        f"how cost grows with depth, across {len(ladder)} depths this server can hold",
        ["--pp", "2048", "--tg", "256", "--exact-tg", "--depth", *sweep_depths,
         "--runs", "3", *cached],
    ))

    presets.append((
        "decode",
        "decode only, sampled until the CI is tight enough to A/B on",
        ["--pp", "2048", "--tg", "256", "--exact-tg",
         "--depth", "0", str(ladder[1]) if len(ladder) > 1 else "0",
         "--runs", "5", "--target-ci", "0.03", "--max-runs", "40", "--stats", "ci", *cached],
    ))

    if len(conc) > 1:
        presets.append((
            "concurrency",
            f"throughput under load; {conc[-1]} concurrent requests fit this KV cache at d{deep}",
            ["--pp", "2048", "--tg", "256", "--exact-tg", "--depth", str(deep),
             "--runs", "3", "--concurrency", *[str(c) for c in conc], *cached],
        ))

    longest = max_depth_for_long_run(found)
    if longest and longest > ladder[-1]:
        presets.append((
            "long",
            f"the deepest context this server accepts ({_fmt_int(longest)} tokens)",
            ["--pp", "2048", "--tg", "256", "--exact-tg", "--depth", str(longest),
             "--runs", "3", *cached],
        ))

    return presets


def _quote(arg: str) -> str:
    return arg if re.fullmatch(r"[A-Za-z0-9_.:/=@,+-]+", arg) else "'" + arg.replace("'", "'\\''") + "'"


def _wrap(tokens: List[str], indent: str, width: int = 84) -> List[str]:
    """Break a long command before flags, so it stays readable and pasteable."""
    lines: List[str] = []
    current = indent
    for token in tokens:
        candidate = token if current.strip() == "" else f"{current} {token}"
        if len(candidate) > width and token.startswith("--") and current.strip():
            lines.append(current + " \\")
            current = indent + "    " + token
        else:
            current = candidate if current.strip() else current + token
    lines.append(current)
    return lines


def _command(found: Detected, args: List[str], runner: str = "uv run llm-assay.py") -> List[str]:
    # The runner is a command line, not one argument -- quoting it whole would
    # make the shell look for a program named "uv run llm-assay.py".
    tokens = runner.split() + [_quote(a) for a in [*_target_args(found), *args]]
    tokens += ["--latency-mode", "generation", "--seed", "$RANDOM"]
    return _wrap(tokens, "  ")


def probe_preset(found: Detected) -> Tuple[str, str, List[str]]:
    """A quality probe sized to this endpoint.

    Separate from ``build_presets`` because it is a different subcommand with a
    different flag surface -- no --runs, no --latency-mode, no --format -- and
    folding it into the benchmark presets would put flags on it that it does not
    accept.

    Depth 0 is dropped: there is no context to hide an anomaly in. The ladder is
    trimmed because probe trials are serial and a deep one is slow, and the point
    is to bracket where comprehension falls off, not to measure every rung.

    The budget is deliberately large. Truncated trials are not missing at random
    -- a trial the model reasons through quickly finishes, a hard one runs long
    and hits the ceiling -- so a small budget quietly raises the reported score
    by dropping the difficult cases.
    """
    ladder = [d for d in depth_ladder(found) if d > 0][:3]
    if not ladder:
        ladder = [4096]
    # All three rungs. `outlier` is the retrieval floor: without it the probe
    # warns that every clean and log rate is unqualified, because a model that
    # is not reading the log scores perfectly on `clean`. An earlier version
    # suggested `log clean` only, citing the deleted prose rungs -- which
    # measured artifacts -- but `outlier` was never one of those.
    return (
        "probe",
        "does the model still USE the context, at the depths above",
        ["--depths", *[str(d) for d in ladder], "--rungs", "clean", "outlier",
         "log", "--trials", "10", "--thinking", "on",
         "--max-tokens", str(probe_budget(found, max(ladder)))],
    )


#: Share of the room left at the deepest probe depth that the suggested budget
#: takes. Below the probe's own 80% warning, so the suggestion runs clean.
_PROBE_BUDGET_SHARE = 0.75
#: The budget when the window is unknown. Too small for a reasoning model at
#: depth -- traces of 80,000-225,000 tokens were measured -- which the output
#: says, rather than inventing a window.
_PROBE_BUDGET_UNKNOWN = 40000


def _probe_budget_note(found: Detected, args: List[str]) -> str:
    """What the suggested probe budget means for how long it takes."""
    budget = args[args.index("--max-tokens") + 1]
    if found.max_model_len is None:
        return ("  --max-tokens " + budget + " because the window is unknown. A reasoning\n"
                "  model at depth can need far more; trials that hit it finish `length`,\n"
                "  and the probe notes them. Raise it once you know the window.")
    return ("  --max-tokens " + budget + " is three quarters of what the window leaves at\n"
            "  the deepest depth. Truncated trials are the hard ones, so the budget is\n"
            "  kept large; a reasoning model can spend tens of minutes on one deep trial.")


def probe_budget(found: Detected, deepest: int) -> int:
    """The probe's --max-tokens: as large as the window allows, with margin.

    Truncated trials are the hard ones, so a small budget quietly reports the
    easy subset. A fixed 40,000 did exactly that on a reasoning model whose
    traces at depth ran to 80,000-225,000 tokens. Sized from the detected
    window instead: three quarters of what is left at the deepest depth after
    the prompt overhead, rounded down to a thousand.
    """
    if found.max_model_len is None:
        return _PROBE_BUDGET_UNKNOWN
    from .probe import PROMPT_OVERHEAD
    room = found.max_model_len - deepest - PROMPT_OVERHEAD
    budget = int(room * _PROBE_BUDGET_SHARE) // 1000 * 1000
    return max(1000, budget)


def _probe_command(found: Detected, args: List[str],
                   runner: str = "uv run llm-assay.py") -> List[str]:
    target = ["probe", found.base_url]
    if found.hf_model:
        target += ["--model", found.hf_model]
    if found.served_model and found.served_model != found.hf_model:
        target += ["--served-model-name", found.served_model]
    tokens = runner.split() + [_quote(a) for a in [*target, *args]]
    tokens += ["--seed", "$RANDOM"]
    return _wrap(tokens, "  ")


def suggestions(found: Detected) -> str:
    lines = ["", "Suggested runs", "", cache_guidance(found)]
    for key, description, args in build_presets(found):
        lines.append("")
        lines.append(f"  # {key} -- {description}")
        lines.extend(_command(found, args))
    key, description, args = probe_preset(found)
    lines.append("")
    lines.append(f"  # {key} -- {description}")
    lines.extend(_probe_command(found, args))
    lines.append("")
    lines.append(
        "  The probe answers the half the benchmark cannot: throughput at depth says\n"
        "  nothing about whether the model can still use that depth, and the settings\n"
        "  that make long context cheap are the ones that can cost comprehension."
    )
    lines.append("")
    lines.append(_probe_budget_note(found, args))
    lines.append("")
    lines.append(
        "  --seed $RANDOM is deliberate: a fixed seed sends byte-identical prompts, so a\n"
        "  repeat invocation is served from the prefix cache and prefill numbers inflate.\n"
        "  For a paired A/B, use the SAME fixed seed on both sides and prime once first."
    )
    return "\n".join(lines)


def _entry_point() -> str:
    """Absolute path to llm-assay.py in this checkout."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "llm-assay.py")


def as_dict(found: Detected) -> Dict[str, Any]:
    """The detection and the suggestions as data.

    Same shape of contract as ``compare --json``: something a script can act on
    without scraping a table. The presets carry both the argv list (to exec
    directly) and the rendered command (to log or paste), because a caller
    usually wants one and a human reading the log wants the other.
    """
    presets = []
    for key, description, args in build_presets(found):
        argv = [*_target_args(found), *args, "--latency-mode", "generation"]
        presets.append({
            "key": key,
            "description": description,
            "args": argv,
            "command": " ".join(["uv", "run", "llm-assay.py", *(_quote(a) for a in argv)]),
        })

    key, description, args = probe_preset(found)
    probe_argv = ["probe", found.base_url]
    if found.hf_model:
        probe_argv += ["--model", found.hf_model]
    if found.served_model and found.served_model != found.hf_model:
        probe_argv += ["--served-model-name", found.served_model]
    probe_argv += args
    probe = {
        "key": key,
        "description": description,
        "args": probe_argv,
        "command": " ".join(["uv", "run", "llm-assay.py",
                             *(_quote(a) for a in probe_argv)]),
    }

    return {
        "endpoint": found.base_url,
        "probe": probe,
        "detected": {
            "served_model": found.served_model,
            "hf_model": found.hf_model,
            "max_model_len": found.max_model_len,
            "usable_context": found.usable_context,
            "server_version": found.server_version,
            "prefix_caching": found.prefix_caching,
            "spec_decode": found.spec_decode,
            "spec_acceptance_length": found.spec_acceptance_length,
            "kv_cache_tokens": found.kv_cache_tokens,
            "kv_max_concurrency": found.kv_max_concurrency,
            "block_size": found.block_size,
            "sliding_window": found.sliding_window,
            "thinking_controls": found.thinking_controls,
        },
        "depth_ladder": depth_ladder(found),
        "notes": list(found.notes),
        "presets": presets,
    }


def runner_keys(found: Detected) -> List[str]:
    """Every preset the written runner offers, in its order: the benchmark
    presets, then `probe`, which is assembled separately. The confirmation
    message counts from here; counting `build_presets` alone told users a
    six-preset runner had five."""
    return [key for key, _, _ in build_presets(found)] + [probe_preset(found)[0]]


def _script_name(found: Detected) -> str:
    name = found.served_model or found.hf_model or "endpoint"
    return "run-" + re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() + ".sh"


def render_script(found: Detected) -> str:
    """A standalone runner for this endpoint, with everything already resolved."""
    presets = build_presets(found)
    keys = " ".join(key for key, _, _ in presets)
    detected = report(found).replace("\n", "\n# ")

    out = [
        "#!/usr/bin/env bash",
        "# Generated by `llm-assay.py tune` -- do not expect it to fit another endpoint.",
        f"# llm-assay {__version__}",
        "#" + detected,
        "#",
        "# What that means for the runs below:",
        "#" + ("\n" + cache_guidance(found)).replace("\n", "\n# "),
        "#",
        "# Regenerate after the server is restarted with different settings:",
        f"#   uv run llm-assay.py tune {found.base_url} --write $0",
        "set -euo pipefail",
        "",
        # An absolute path to the entry point, so this script works from any
        # directory. Results land relative to wherever you run it.
        f"BENCHY=${{BENCHY:-uv run {_entry_point()}}}",
        "SEED=${SEED:-$RANDOM}",
        "OUTDIR=${OUTDIR:-results}",
        "RUNS=${RUNS:-}",
        "EXTRA=${EXTRA:-}",
        "",
        "usage() {",
        '  cat <<EOF',
        f"usage: $0 <preset>",
        "",
        "presets:",
    ]
    for key, description, _ in presets:
        out.append(f"  {key:<12} {description}")
    pkey, pdesc, _ = probe_preset(found)
    out.append(f"  {pkey:<12} {pdesc}")
    out += [
        "",
        "env overrides: BENCHY SEED OUTDIR RUNS EXTRA TAG DRY_RUN",
        "",
        "A fixed SEED makes two runs directly comparable (paired test), but also lets the",
        "second one hit the prefix cache the first populated -- prefill inflates while decode",
        "stays honest. Prime once before an A/B, or vary SEED when prefill is what you want.",
        "EOF",
        "}",
        "",
        'preset="${1:-}"',
        '[[ -z "$preset" || "$preset" == "-h" || "$preset" == "--help" ]] && { usage; exit 0; }',
        "",
        'TAG="${TAG:-${preset}-$(date +%Y%m%d-%H%M%S)}"',
        'mkdir -p "$OUTDIR"',
        'RESULT="$OUTDIR/${TAG}.json"',
        "",
        'case "$preset" in',
    ]

    for key, _, args in presets:
        tokens = [_quote(a) for a in [*_target_args(found), *args]]
        tokens += ["--latency-mode", "generation"]
        body = _wrap(tokens, "          ")
        body[0] = "    args=(" + body[0].lstrip()
        body[-1] = body[-1] + ")"
        out += [f"  {key})", *body, "    ;;"]

    pkey, _, pargs = probe_preset(found)
    ptarget = ["probe", found.base_url]
    if found.hf_model:
        ptarget += ["--model", found.hf_model]
    if found.served_model and found.served_model != found.hf_model:
        ptarget += ["--served-model-name", found.served_model]
    pbody = _wrap([_quote(a) for a in [*ptarget, *pargs]], "          ")
    pbody[0] = "    args=(" + pbody[0].lstrip()
    pbody[-1] = pbody[-1] + ")"
    # IS_PROBE because the tail below adds --runs/--format/--latency-mode, none
    # of which the probe subcommand accepts.
    out += [f"  {pkey})", *pbody, "    IS_PROBE=1", "    ;;"]

    out += [
        "  *)",
        '    echo "error: unknown preset \'$preset\'" >&2; usage >&2; exit 2 ;;',
        "esac",
        "",
        'if [[ -n "${IS_PROBE:-}" ]]; then',
        # Transcripts too: the fabrication evidence -- which lines a model
        # cited, and whether they exist -- lives there, not in the result.
        '  args+=(--seed "$SEED" --save-result "$RESULT" --save-transcripts "${RESULT%.json}-transcripts.json")',
        "else",
        '[[ -n "$RUNS" ]] && args+=(--runs "$RUNS")',
        "# shellcheck disable=SC2206",
        '[[ -n "$EXTRA" ]] && args+=($EXTRA)',
        'args+=(--seed "$SEED" --format json --save-result "$RESULT")',
        "fi",
        "",
        'echo "==> preset=$preset seed=$SEED"',
        'echo "    result: $RESULT"',
        "printf '   '; printf ' %q' $BENCHY \"${args[@]}\"; echo",
        "",
        '[[ -n "${DRY_RUN:-}" ]] && exit 0',
        "",
        '$BENCHY "${args[@]}"',
        "",
        "cat <<EOF",
        "",
        "Saved: $RESULT",
        "",
        "To A/B against another run, use the SAME seed on both sides:",
        "    SEED=$SEED TAG=other $0 $preset",
        "    $BENCHY compare $RESULT $OUTDIR/other.json",
        "EOF",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py tune",
        description="Detect what an endpoint is and suggest benchmark runs that fit it.",
    )
    ap.add_argument("base_url", help="OpenAI-compatible endpoint URL, including /v1")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the endpoint")
    ap.add_argument(
        "--model",
        default=None,
        help="Pick this model when the endpoint serves several (default: the first)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the detection and the suggested runs as JSON instead of prose",
    )
    ap.add_argument(
        "--write",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Write an executable runner script (default name: run-<model>.sh)",
    )
    args = ap.parse_args()

    found = detect(args.base_url, args.api_key, args.model)
    nothing_found = found.served_model is None and found.max_model_len is None

    if args.as_json:
        # stdout stays parseable: the human report and any error go to stderr.
        if nothing_found:
            print(report(found), file=sys.stderr)
            print(
                "\nNothing could be read from this endpoint. Check the URL includes /v1 "
                "and that the server is reachable.",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(as_dict(found), indent=2))
        if args.write is None:
            return 0
    else:
        print(report(found))

        if nothing_found:
            sys.stdout.flush()  # keep the report above the error, not interleaved
            print(
                "\nNothing could be read from this endpoint. Check the URL includes /v1 "
                "and that the server is reachable.",
                file=sys.stderr,
            )
            return 1

        print(suggestions(found))

    if args.write is None:
        print(f"\n  Pass --write to save these as a runner script.")
        return 0

    path = args.write or _script_name(found)
    with open(path, "w") as handle:
        handle.write(render_script(found))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    where = sys.stderr if args.as_json else sys.stdout
    print(f"\nWrote {path} ({len(runner_keys(found))} presets). Run it with:", file=where)
    print(f"    {path if path.startswith('/') else './' + path.lstrip('./')} smoke", file=where)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
