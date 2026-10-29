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

The agent harnesses are the one thing read off this machine rather than the
endpoint: each one found on PATH gets an ``agent-<harness>`` run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import re
import shutil
import stat
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import __version__
# The engine-name families live there, and are imported rather than restated:
# servermetrics had already been taught llama.cpp when tune had not, and two
# copies of a list like that is exactly how the stale one goes unnoticed.
from .servermetrics import _BUILD_FIELDS, _build_summary, strata_as_prometheus
from .servermetrics import _GAUGES as _SM_GAUGES
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
        #: True when detection needed an API key. The suggested commands never
        #: carry it; they read it from the environment, and the output says so.
        self.uses_key: bool = False
        #: A router's declared output cap for its default provider, which binds
        #: the probe budget before the context window does (GLM 5.3's: 131,072).
        self.completion_cap: Optional[int] = None
        #: Per-token prices in dollars, when the endpoint publishes them.
        self.prompt_price: Optional[float] = None
        self.completion_price: Optional[float] = None
        #: Whether the model's tokenizer could be loaded. False on most hosted
        #: models, whose vocabularies are not public: depths are then counted
        #: in a stand-in vocabulary, and only --calibrate-depth fixes them.
        self.tokenizer_ok: Optional[bool] = None
        # A base model whose tokenizer loaded when the served name's did not:
        # derived from the GGUF file name, and used only once it has loaded.
        self.tokenizer: Optional[str] = None
        #: Decode rate from one short timed request, for the time estimate.
        self.decode_tps: Optional[float] = None
        self.server_version: Optional[str] = None
        self.prefix_caching: Optional[bool] = None
        self.kv_cache_tokens: Optional[int] = None
        self.kv_max_concurrency: Optional[float] = None
        self.block_size: Optional[int] = None
        self.sliding_window: Optional[int] = None
        self.served_build: Optional[Dict[str, str]] = None
        self.spec_decode: Optional[bool] = None
        self.spec_acceptance_length: Optional[float] = None
        self.thinking: Optional[ThinkingProbe] = None
        #: Agent harnesses installed on this machine (``installed_harnesses``),
        #: each of which gets an agent run. Set by ``main``, not ``detect``:
        #: it is a fact about the caller's PATH, not the endpoint.
        self.harnesses: List[str] = []
        #: Whether bwrap is on PATH: the agent sandboxes every run with it.
        self.sandbox: bool = True
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


def _recognised(metrics_text: str) -> bool:
    """Whether the text comes from an engine family this tool knows.

    By family, not by metric: a vLLM without spec-decode counters really has
    none configured, which is the "absent means not configured" rule below. It
    is a format we cannot read at all whose silence proves nothing.
    """
    known = {name for names in _SM_COUNTERS.values() for name in names}
    known |= {name for names in _SM_GAUGES.values() for name in names}
    families = {name.split(":", 1)[0] + ":" for name in known}
    return any(line.strip().startswith(tuple(families)) for line in metrics_text.splitlines())


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


def _as_price(value: Any) -> Optional[float]:
    """A router's per-token price string ("0.0000003") as dollars, or None."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _tokenizer_loads(name: str) -> bool:
    """Whether the model's own tokenizer can be loaded, without the gpt2 fallback."""
    try:
        from .corpus import LightweightTokenizer
        LightweightTokenizer.from_pretrained(name)
        return True
    except Exception:
        return False


_QUANT_SUFFIX = re.compile(
    r"[-.](?:UD-)?(?:I?Q\d[\w]*|F16|BF16|F32|MXFP4\w*|NVFP4\w*)$", re.IGNORECASE)
_SHARD = re.compile(r"-\d{5}-of-\d{5}$")


def _base_from_gguf(model_path: str) -> Optional[str]:
    """The base model a GGUF file name spells, as an org/name guess.

    `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` -> `Qwen/Qwen3.8-Flash-Next`.
    The org is the name's leading letters, which is right for Qwen and wrong for
    plenty of others -- so this is only a candidate, and the caller keeps it only
    if its tokenizer actually loads.
    """
    name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.lower().endswith(".gguf"):
        return None
    stem = _SHARD.sub("", name[:-5])
    stem = _QUANT_SUFFIX.sub("", stem)
    org = re.match(r"[A-Za-z]+", stem)
    return f"{org.group(0)}/{stem}" if org and stem else None


def _substitute_tokenizer(build: Optional[Dict[str, str]]) -> Optional[str]:
    """A loadable base-model tokenizer for a served name that is not a repo."""
    path = (build or {}).get("model_path")
    candidate = _base_from_gguf(path) if path else None
    if candidate and _tokenizer_loads(candidate):
        return candidate
    return None


def _measure_decode(base_url: str, model: str, api_key: str) -> Optional[float]:
    """Tokens per second from one short request, or None if it gave no count.

    The time estimate needs the endpoint's real decode rate, not a guess. One
    64-token request is enough to order-of-magnitude it; prefill of a
    one-line prompt is negligible beside it.
    """
    body = {"model": model, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Count from one to fifty."}]}
    try:
        start = time.monotonic()
        resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=body,
                             headers=_headers(api_key), timeout=60)
        elapsed = time.monotonic() - start
        tokens = (resp.json().get("usage") or {}).get("completion_tokens")
    except Exception:
        return None
    if not isinstance(tokens, int) or tokens < 8 or elapsed <= 0:
        return None
    return tokens / elapsed


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
            found.max_model_len = (_as_int(chosen.get("max_model_len"))
                                   or _as_int(chosen.get("context_length")))
            top = chosen.get("top_provider") if isinstance(chosen.get("top_provider"), dict) else {}
            found.completion_cap = _as_int(top.get("max_completion_tokens"))
            pricing = chosen.get("pricing") if isinstance(chosen.get("pricing"), dict) else {}
            found.prompt_price = _as_price(pricing.get("prompt"))
            found.completion_price = _as_price(pricing.get("completion"))
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
        found.thinking = probe_thinking_controls(base_url, found.served_model, api_key)
        found.decode_tps = _measure_decode(base_url, found.served_model, api_key)
    if found.hf_model:
        found.tokenizer_ok = _tokenizer_loads(found.hf_model)
        if found.tokenizer_ok is False:
            found.tokenizer = _substitute_tokenizer(found.served_build)
            if found.tokenizer:
                found.tokenizer_ok = True

    version = _get(f"{root}/version", api_key, timeout=4.0)
    if version is not None:
        try:
            found.server_version = version.json().get("version")
        except ValueError:
            pass

    metrics = _get(f"{root}/metrics", api_key)
    if metrics is None:
        found.notes.append(
            f"no readable {root}/metrics -- prefix-caching status, speculative decoding and KV "
            "capacity are unknown, so the suggestions below are conservative"
        )
        return found

    text = metrics.text
    strata = strata_as_prometheus(text)
    if strata is not None:
        text = strata
    if not _recognised(text):
        # Readable, and in a shape this tool does not read. Every conclusion
        # below is drawn from the absence of a counter, and absence in a format
        # we cannot parse proves nothing: Strata's JSON once read as "spec
        # decode not active" on a box accepting 21 of 25 drafts.
        found.notes.append(
            f"{root}/metrics answers, but in a format this tool does not read, so "
            "prefix caching and speculative decoding are unknown rather than off"
        )
        return found

    labels = _parse_cache_config(text)
    if not labels:
        # Readable metrics that simply do not carry vLLM's config block. Saying
        # "/metrics unreadable" there blames the wrong thing and sends the
        # reader to check a URL that is answering fine -- and llama.cpp does
        # publish the prefix-cache evidence, just not the flag.
        found.prefix_caching = _prefix_caching_from_evidence(text, found)
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
    drafts = _any_counter(text, "spec_drafts")
    accepted = _any_counter(text, "spec_accepted")
    # Strata counts draft *tokens* and no rounds, so it proves drafting is
    # live without giving an acceptance length.
    draft_tokens = _any_counter(text, "spec_draft_tokens")
    if drafts is None and draft_tokens:
        found.spec_decode = True
    elif drafts is None and draft_tokens is not None:
        found.spec_decode = None
        found.notes.append(
            "speculative decoding is instrumented but no recent request drafted, "
            "so whether it is active is unknown"
        )
    elif drafts is None:
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


# Candidate ways to ask a server to stop thinking, and every graded level the
# OpenAI API names. Which ones work is a property of the deployment's chat
# template, not of the model card: on an unsloth conversion of Qwen3.8 the
# graded levels the official card documents render a byte-identical prompt,
# while reasoning_effort="none" and enable_thinking=false both prefill an empty
# <think></think> block. Levels a template does not know are tried too, because
# a template that raises on them is the deployment stating its own list:
# Flash-Next's answers "high" with "Supported types are xhigh (default),
# medium, and low."
_THINKING_PROBES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
    *((f"reasoning_effort={level}", {"reasoning_effort": level})
      for level in ("minimal", "low", "medium", "high", "xhigh", "max")),
)

#: How a switch fared: it changed the prompt, it rendered what sending nothing
#: renders (the default, or ignored), or the server refused the request.
THINKING_STATES = ("live", "same", "rejected")


class ThinkingProbe:
    """Which thinking switches change the prompt, and how that was found out."""

    def __init__(self, route: str, controls: Dict[str, str],
                 rejection: Optional[str] = None, render_refusal: Optional[str] = None) -> None:
        #: "render" (exact token ids) or "prompt_tokens" (a count, from a
        #: one-token generation through the request path itself).
        self.route = route
        self.controls = controls
        #: The server's own words on the first switch it refused.
        self.rejection = rejection
        #: Why /render could not be used, when it answered with a refusal.
        self.render_refusal = render_refusal
        #: The reasoning_effort values that make different prompts, one per
        #: distinct rendering, in the API's order -- what a level sweep should
        #: run. Empty when no graded level changes the prompt: then the only
        #: sweep there is --thinking on off.
        self.levels: List[str] = []


class _Refused:
    """A non-200 answer: its status and the server's own words."""

    def __init__(self, resp: Any) -> None:
        self.status = resp.status_code
        self.message = _error_message(resp)


def _error_message(resp: Any) -> str:
    try:
        err = resp.json().get("error")
        msg = err.get("message") if isinstance(err, dict) else err
    except (ValueError, AttributeError):
        msg = None
    return str(msg or resp.text or f"HTTP {resp.status_code}").strip()[:200]


def probe_thinking_controls(
    base_url: str, model: str, api_key: str = "EMPTY"
) -> Optional[ThinkingProbe]:
    """Which thinking switches actually change the prompt this server builds.

    Prefers /v1/chat/completions/render, which costs no generation: render the
    same message with and without each switch and compare the token ids. If
    the rendered prompt is unchanged the model cannot possibly respond to the
    setting -- there is no channel for it -- which is a far more reliable check
    than generating twice and squinting at the completions.

    vLLM 0.32 put /render behind --enable-scale-out and answers 403 without it,
    which used to make this whole report vanish. The fallback is a one-token
    generation per switch, compared on usage.prompt_tokens: a count rather than
    the ids, but exact, and taken through the request path a benchmark uses.
    /tokenize is not a substitute: it ignores a top-level reasoning_effort, and
    passed as a template variable "none" is refused by a template the chat
    route would have handed enable_thinking=false instead.

    A switch the server refuses with a 400 is "rejected". Returns None when
    neither route answers a request that carries no switch at all.
    """
    rendered = _render_settings(base_url, model, api_key, list(_THINKING_PROBES))
    if rendered is None:
        return None
    controls: Dict[str, str] = {}
    rejection: Optional[str] = None
    for label, got in rendered.results.items():
        if isinstance(got, _Refused):
            if got.status == 400:
                controls[label] = "rejected"
                rejection = rejection or got.message
        elif got is not None:
            controls[label] = "live" if got != rendered.baseline else "same"
    if not controls:
        return None
    probe = ThinkingProbe(rendered.route, controls, rejection, rendered.render_refusal)
    efforts = [(label, extra["reasoning_effort"]) for label, extra in _THINKING_PROBES
               if set(extra) == {"reasoning_effort"} and controls.get(label) in ("live", "same")]
    if any(controls[label] == "live" for label, level in efforts if level != "none"):
        seen: List[Any] = []
        for label, level in efforts:
            if rendered.results[label] not in seen:
                seen.append(rendered.results[label])
                probe.levels.append(level)
    return probe


class _Rendered:
    def __init__(self, route: str, baseline: Any, results: Dict[str, Any],
                 render_refusal: Optional[str]) -> None:
        self.route, self.baseline, self.results = route, baseline, results
        self.render_refusal = render_refusal


def _render_settings(base_url: str, model: str, api_key: str,
                     settings: List[Tuple[str, Dict[str, Any]]]) -> Optional[_Rendered]:
    """Each setting's rendering beside the rendering of a request carrying none.

    A rendering is the token ids (render) or the prompt_tokens count (the
    fallback), a ``_Refused`` for a non-200, or None when nothing answered.
    None overall when even the bare request gets no rendering on either route.
    """
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    root = _root(base_url)
    render_refusal: Optional[str] = None

    def post(url: str, extra: Optional[Dict[str, Any]]) -> Optional[requests.Response]:
        try:
            return requests.post(url, json={**body, **(extra or {})},
                                 headers=_headers(api_key), timeout=30)
        except requests.RequestException:
            return None

    def by_render(extra: Optional[Dict[str, Any]]) -> Any:
        resp = post(root + "/v1/chat/completions/render", extra)
        if resp is None:
            return None
        if resp.status_code != 200:
            return _Refused(resp)
        try:
            ids = resp.json().get("token_ids")
        except ValueError:
            return None
        return list(ids) if isinstance(ids, list) else None

    def by_count(extra: Optional[Dict[str, Any]]) -> Any:
        resp = post(base_url.rstrip("/") + "/chat/completions", {"max_tokens": 1, **(extra or {})})
        if resp is None:
            return None
        if resp.status_code != 200:
            return _Refused(resp)
        try:
            tokens = (resp.json().get("usage") or {}).get("prompt_tokens")
        except ValueError:
            return None
        return tokens if isinstance(tokens, int) else None

    route, ask = "render", by_render
    baseline = ask(None)
    if isinstance(baseline, _Refused) and baseline.status in (401, 403):
        render_refusal = baseline.message
    if baseline is None or isinstance(baseline, _Refused):
        route, ask = "prompt_tokens", by_count
        baseline = ask(None)
        if baseline is None or isinstance(baseline, _Refused):
            return None
    return _Rendered(route, baseline, {label: ask(extra) for label, extra in settings},
                     render_refusal)


#: The request fields that carry a thinking setting, which is all the guard sends.
THINKING_FIELDS = ("reasoning_effort", "chat_template_kwargs")


class ThinkingCheck:
    """What the guard found about the thinking settings a run is about to send."""

    def __init__(self, route: str) -> None:
        self.route = route
        #: arm label -> the server's refusal. Every request of such an arm would fail.
        self.rejected: Dict[str, str] = {}
        #: groups of two or more arms whose prompts render identically.
        self.identical: List[List[str]] = []
        #: arms that set a graded level on a deployment where no graded level
        #: changes the prompt.
        self.inert: List[str] = []

    def errors(self) -> List[str]:
        return [f"{label}: the server rejects this thinking setting, so every request would "
                f"fail: {msg}" for label, msg in self.rejected.items()]

    def warnings(self) -> List[str]:
        basis = ("render the same prompt" if self.route == "render"
                 else "render the same number of prompt tokens")
        out = [f"{' and '.join(f'[{g}]' for g in group)} {basis}: one configuration under "
               f"{len(group)} labels" for group in self.identical]
        if self.inert:
            out.append(f"{', '.join(self.inert)}: no reasoning_effort level changes the prompt "
                       f"on this deployment (its chat template ignores the level), so the label "
                       f"describes nothing the model sees")
        if out and self.route == "prompt_tokens":
            out.append("compared by prompt_tokens: an effort a service applies outside the "
                       "prompt (a hosted API, a reasoning budget) cannot show here")
        return out


def describe_setting(fields: Dict[str, Any]) -> str:
    """``reasoning_effort=low``, ``enable_thinking=false, reasoning_effort=none``,
    or ``no thinking setting`` -- the request's own words, for a message."""
    parts = [f"{k}={json.dumps(v) if not isinstance(v, str) else v}"
             for k, v in sorted((fields.get("chat_template_kwargs") or {}).items())]
    if "reasoning_effort" in fields:
        parts.append(f"reasoning_effort={fields['reasoning_effort']}")
    return ", ".join(parts) or "no thinking setting"


def check_thinking(base_url: str, model: str, arms: Dict[str, Dict[str, Any]],
                   api_key: str = "EMPTY") -> Optional[ThinkingCheck]:
    """The guard every subcommand runs before it measures anything.

    ``arms`` maps a label to the thinking fields that arm's requests carry
    (see ``THINKING_FIELDS``). A rejected setting is an error -- every request
    would fail -- and arms that render identically, or a graded level on a
    template that ignores every level, are warnings: the result would label as
    different what the model saw as the same. That is the trap the 27B's
    banked arms were in for weeks, found by a tune check rather than by any
    of them. None when the endpoint renders nothing, so the run goes ahead.
    """
    arms = {label: {k: v for k, v in fields.items() if k in THINKING_FIELDS}
            for label, fields in arms.items()}
    if not any(arms.values()):
        return None
    graded = [(label, extra) for label, extra in _THINKING_PROBES
              if set(extra) == {"reasoning_effort"} and extra["reasoning_effort"] != "none"]
    rendered = _render_settings(base_url, model, api_key,
                                [(f"arm:{k}", v) for k, v in arms.items()] + graded)
    if rendered is None:
        return None
    check = ThinkingCheck(rendered.route)
    got = {label: rendered.results[f"arm:{label}"] for label in arms}
    for label, value in got.items():
        if isinstance(value, _Refused) and value.status == 400:
            check.rejected[label] = value.message
    # Grouped by setting, not by arm: two arms that send the same thinking
    # fields differ in something else on purpose (a system prompt, a model).
    seen: List[Tuple[Any, List[str]]] = []
    for label, value in got.items():
        if value is None or isinstance(value, _Refused):
            continue
        setting = describe_setting(arms[label])
        for rendering, group in seen:
            if rendering == value:
                if setting not in group:
                    group.append(setting)
                break
        else:
            seen.append((value, [setting]))
    check.identical = [group for _, group in seen if len(group) > 1]
    levels = [rendered.results[label] for label, _ in graded]
    answered = [v for v in levels if v is not None and not isinstance(v, _Refused)]
    if answered and all(v == rendered.baseline for v in answered):
        check.inert = [label for label, fields in arms.items()
                       if fields.get("reasoning_effort") not in (None, "none")
                       and label not in check.rejected]
    return check


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


def _switches(labels: List[str]) -> str:
    """Switch labels with the graded levels after one prefix: seven levels in
    full made a 200-character line."""
    prefix = "reasoning_effort="
    levels = [k[len(prefix):] for k in labels if k.startswith(prefix)]
    others = [k for k in labels if not k.startswith(prefix)]
    if levels:
        others.append(prefix + ", ".join(levels) if len(levels) > 1 else prefix + levels[0])
    return ", ".join(others)


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
        # None is also an endpoint whose /metrics did not answer, not only a draft
        # head with no traffic yet; each path that knows which leaves a note.
        row("spec decode", "unknown -- see note below")

    if found.kv_cache_tokens:
        detail = f"{_fmt_int(found.kv_cache_tokens)} tokens"
        if found.kv_max_concurrency:
            detail += f" (~{found.kv_max_concurrency:.1f} seqs at max ctx)"
        row("kv cache", detail)

    if found.thinking:
        by = {state: _switches([k for k, v in found.thinking.controls.items() if v == state])
              for state in THINKING_STATES}
        row("thinking", by["live"] or "no switch changes the prompt")
        if by["same"]:
            lines.append(f"  {'':<14}same as sending nothing: {by['same']}")
        if by["rejected"]:
            lines.append(f"  {'':<14}rejected: {by['rejected']}")
            lines.append(f"  {'':<14}  server: {found.thinking.rejection}")
        if found.thinking.route == "prompt_tokens":
            why = (f" (/render refused: {found.thinking.render_refusal})"
                   if found.thinking.render_refusal else " (no /render route)")
            lines.append(f"  {'':<14}compared by prompt_tokens{why}")

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


def _model_args(found: Detected) -> List[str]:
    """--model, the alias, and a substitute tokenizer when one was verified."""
    args: List[str] = []
    if found.hf_model:
        args += ["--model", found.hf_model]
    if found.served_model and found.served_model != found.hf_model:
        args += ["--served-model-name", found.served_model]
    if found.tokenizer:
        args += ["--tokenizer", found.tokenizer]
    return args


def _target_args(found: Detected) -> List[str]:
    """The flags that identify the endpoint, shared by every suggested run."""
    return ["--base-url", found.base_url, *_model_args(found)]


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
            (f"throughput under load; {conc[-1]} concurrent requests fit this KV cache at d{deep}"
             if found.kv_cache_tokens else
             f"throughput under load at d{deep}; the KV cache size is unknown, so {conc[-1]} "
             "is a guess -- watch for queued requests"),
            ["--pp", "2048", "--tg", "256", "--exact-tg", "--depth", str(deep),
             "--runs", "3", "--concurrency", *[str(c) for c in conc], *cached],
        ))

    levels = found.thinking.levels if found.thinking else []
    if len(levels) > 1:
        presets.append((
            "levels",
            f"one shape at each thinking level that makes a different prompt here "
            f"({', '.join(levels)})",
            ["--pp", "512", "--tg", "256", "--exact-tg", "--depth", "0", "--runs", "5",
             "--stats", "ci", "--reasoning-effort", *levels],
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
         "--max-tokens", str(probe_budget(found, max(ladder))),
         # A model whose vocabulary cannot be loaded is sized in a stand-in
         # one, so every depth would be nominal; calibration measures the
         # endpoint's own count instead.
         *(["--calibrate-depth"] if found.tokenizer_ok is False else []),
         *(["--max-spend", str(probe_spend_cap(found, ladder))]
           if found.prompt_price else [])],
    )


#: Trials the probe preset runs per (depth, rung), and its rungs.
_PROBE_TRIALS = 10
_PROBE_RUNGS = 3


def probe_prompt_cost(found: Detected, ladder: List[int]) -> float:
    """Dollars the preset's prompt tokens cost alone, from the published price."""
    from .probe import PROMPT_OVERHEAD
    tokens = sum(d + PROMPT_OVERHEAD for d in ladder) * _PROBE_TRIALS * _PROBE_RUNGS
    return tokens * (found.prompt_price or 0.0)


def probe_spend_cap(found: Detected, ladder: List[int]) -> int:
    """A suggested --max-spend: three times the prompt cost, in whole dollars.

    Not an estimate of the bill. The prompt side is knowable; the reasoning a
    model produces is not -- 856 to 179,847 tokens were measured on identical
    input -- so the cap is a stop between trials, sized from the part that is
    known, and the output says so.
    """
    return max(1, math.ceil(3 * probe_prompt_cost(found, ladder)))


#: Share of the room left at the deepest probe depth that the suggested budget
#: takes. Below the probe's own 80% warning, so the suggestion runs clean.
_PROBE_BUDGET_SHARE = 0.75
#: The budget when the window is unknown. Too small for a reasoning model at
#: depth -- traces of 80,000-225,000 tokens were measured -- which the output
#: says, rather than inventing a window.
_PROBE_BUDGET_UNKNOWN = 40000


def _probe_notes(found: Detected, args: List[str]) -> List[str]:
    """Calibration, spend and time notes for the suggested probe, when they apply."""
    notes = []
    ladder = [int(d) for d in args[args.index("--depths") + 1: args.index("--rungs")]]
    trials = len(ladder) * _PROBE_TRIALS * _PROBE_RUNGS
    if "--calibrate-depth" in args:
        notes.append(
            f"  --calibrate-depth because {found.hf_model}'s tokenizer could not be\n"
            "  loaded: without it every depth would be counted in a stand-in vocabulary\n"
            "  and the endpoint would see a different amount of text than the label says.")
    if "--max-spend" in args:
        cap = args[args.index("--max-spend") + 1]
        notes.append(
            f"  --max-spend {cap}: the prompt tokens alone cost about "
            f"${probe_prompt_cost(found, ladder):.2f} at the\n"
            "  published price. Reasoning output is the unknown part, and usually the\n"
            "  larger; the cap stops the run between trials. Set your own.")
    if found.decode_tps:
        per_10k = trials * 10000 / found.decode_tps / 3600
        notes.append(
            f"  Time: decode measured at {found.decode_tps:.0f} t/s, so every 10,000 tokens a\n"
            f"  trial reasons adds about {per_10k:.1f} h across the preset's {trials} trials.\n"
            "  Reasoning models used 10,000-200,000 per trial at depth in this tool's\n"
            "  own studies; a non-reasoning model answers in a few hundred.")
    return notes


def _probe_budget_note(found: Detected, args: List[str]) -> str:
    """What the suggested probe budget means for how long it takes."""
    budget = args[args.index("--max-tokens") + 1]
    if found.max_model_len is None and not found.completion_cap:
        return ("  --max-tokens " + budget + " because the window is unknown. A reasoning\n"
                "  model at depth can need far more; trials that hit it finish `length`,\n"
                "  and the probe notes them. Raise it once you know the window.")
    deepest = max(int(d) for d in args[args.index("--depths") + 1: args.index("--rungs")])
    from .probe import PROMPT_OVERHEAD
    room = (found.max_model_len - deepest - PROMPT_OVERHEAD
            if found.max_model_len is not None else None)
    if found.completion_cap and (room is None or found.completion_cap <= room):
        basis = f"the router's declared {found.completion_cap:,}-token output cap"
    else:
        basis = "what the window leaves at the deepest depth"
    return ("  --max-tokens " + budget + " is three quarters of\n  " + basis + ".\n"
            "  Truncated trials are the hard ones, so the budget is kept large; a\n"
            "  reasoning model can spend tens of minutes on one deep trial.")


def probe_budget(found: Detected, deepest: int) -> int:
    """The probe's --max-tokens: as large as the window allows, with margin.

    Truncated trials are the hard ones, so a small budget quietly reports the
    easy subset. A fixed 40,000 did exactly that on a reasoning model whose
    traces at depth ran to 80,000-225,000 tokens. Sized from the detected
    window instead: three quarters of what is left at the deepest depth after
    the prompt overhead, rounded down to a thousand.
    """
    from .probe import PROMPT_OVERHEAD
    ceilings = []
    if found.max_model_len is not None:
        ceilings.append(found.max_model_len - deepest - PROMPT_OVERHEAD)
    # A router's declared output cap binds too, and on hosted models usually
    # first: the probe refuses a budget above it and warns at 80% of it.
    if found.completion_cap:
        ceilings.append(found.completion_cap)
    if not ceilings:
        return _PROBE_BUDGET_UNKNOWN
    budget = int(min(ceilings) * _PROBE_BUDGET_SHARE) // 1000 * 1000
    return max(1000, budget)


def _probe_command(found: Detected, args: List[str],
                   runner: str = "uv run llm-assay.py") -> List[str]:
    target = ["probe", found.base_url, *_model_args(found)]
    tokens = runner.split() + [_quote(a) for a in [*target, *args]]
    tokens += ["--seed", "$RANDOM"]
    return _wrap(tokens, "  ")


#: The harnesses ``agent`` drives, by the executable each one installs.
AGENT_HARNESSES = ("pi", "hermes", "opencode", "omp")

#: A worker per this many tokens of KV cache: an agent run's context grows to
#: tens of thousands of tokens, and runs that do not fit are preempted, which
#: changed pass rates before. Capped, because the endpoint is shared by the runs.
_AGENT_TOKENS_PER_WORKER = 65536
_AGENT_MAX_WORKERS = 4


def installed_harnesses() -> List[str]:
    """The agent harnesses on PATH, in ``AGENT_HARNESSES`` order."""
    return [name for name in AGENT_HARNESSES if shutil.which(name)]


def agent_workers(found: Detected) -> int:
    if not found.kv_cache_tokens:
        return 1
    return max(1, min(_AGENT_MAX_WORKERS, found.kv_cache_tokens // _AGENT_TOKENS_PER_WORKER))


def agent_thinking(found: Detected) -> str:
    """The agent's level: its default, medium, unless this server rejects it.

    The agent refuses to start on a rejected setting, so suggesting one would
    be a run that cannot begin. Then the first level that does change the
    prompt, and failing that off.
    """
    controls = found.thinking.controls if found.thinking else {}
    if controls.get("reasoning_effort=medium") != "rejected":
        return "medium"
    graded = [level for level in (found.thinking.levels if found.thinking else [])
              if level != "none"]
    return graded[0] if graded else "off"


def agent_presets(found: Detected) -> List[Tuple[str, str, List[str]]]:
    """(key, description, argv) for one agent run per installed harness.

    The `hard` tasks, the agent's default, at 5 runs: enough to see a harness
    work end to end and to spot a task it fails outright, not to tell two
    configurations apart -- the README's A/Bs use 20. Sampling stays unpinned,
    which the agent warns about: the model card's values are not on the
    endpoint, so pinning them is the operator's call (``--sampling``).
    """
    if not found.served_model:
        return []
    workers, thinking = agent_workers(found), agent_thinking(found)
    out = []
    for harness in found.harnesses:
        out.append((
            f"agent-{harness}",
            f"the agent suite through {harness}: the hard tasks, 5 runs each, "
            f"{workers} worker{'s' if workers > 1 else ''}, thinking {thinking}",
            ["agent", found.base_url, "--model", found.served_model, "--harness", harness,
             "--tasks", "hard", "--runs", "5", "--workers", str(workers),
             "--thinking", thinking],
        ))
    return out


def _agent_command(args: List[str], runner: str = "uv run llm-assay.py") -> List[str]:
    return _wrap(runner.split() + [_quote(a) for a in args], "  ")


def _agent_notes(found: Detected) -> List[str]:
    if not found.harnesses:
        return [
            "  Agent suite: no harness found on PATH (" + ", ".join(AGENT_HARNESSES) + ").\n"
            "  Install one and re-run tune to get an agent-<harness> run, which hands\n"
            "  the model a repository through that harness and grades what it did."]
    notes = [
        "  Agent suite: the agent runs above grade the workspace a harness leaves,\n"
        "  not the reply. Each costs minutes per run; an A/B wants --runs 20 on both\n"
        "  sides and the same --workers, then `compare` on the two results."]
    if found.served_model and agent_thinking(found) != "medium":
        notes.append(
            f"  --thinking {agent_thinking(found)} because this server rejects medium, the\n"
            "  agent's default.")
    if not found.sandbox:
        notes.append(
            "  bwrap is not on PATH: the agent runs every harness inside it and will\n"
            "  refuse to start. Install bubblewrap (on Ubuntu, see the README's note on\n"
            "  AppArmor) rather than passing --unsandboxed.")
    return notes


def scenario_presets(found: Detected) -> List[Tuple[str, str, List[str]]]:
    """(key, description, argv) for the scenario runs: qualify the deployment,
    and the two halves of an A/B around a change to it. The A/B's directory is
    the runner's ``$AB`` (default ``deploy``) under ``$OUTDIR``, so ``ab-after``
    finds what ``ab-before`` kept there."""
    harness = ["--agent", "auto"] if found.harnesses else []
    return [
        ("qualify", "is this deployment served right? canary, thinking, smoke, a short probe"
                    + (", 3 agent runs" if harness else ""),
         ["scenario", "qualify", found.base_url, *harness]),
        ("ab-before", "A/B a server change, part 1: record the endpoint and measure it",
         ["scenario", "ab-deploy", "before", found.base_url]),
        ("ab-after", "A/B a server change, part 2, after changing it: measure and compare",
         ["scenario", "ab-deploy", "after"]),
    ]


def level_arms(found: Detected) -> List[str]:
    """A models file for ``agent --models``: one arm per level a sweep should run.

    The agent spells "none" as its harness level "off". Sampling is left for the
    operator to pin from the model card, and the task set to choose: those are
    decisions about the experiment, not facts about the endpoint.
    """
    levels = found.thinking.levels if found.thinking else []
    if len(levels) < 2 or not found.served_model:
        return []
    out = ["# One agent arm per thinking level this endpoint renders differently. Pin",
           "# sampling (the model card's values); the tasks start at the hard set.",
           f"defaults: {{runs: 20, harness: {(found.harnesses or ['pi'])[0]}, tasks: [hard]}}",
           "models:"]
    for level in levels:
        thinking = "off" if level == "none" else level
        out.append(f"  - {{name: {found.served_model}-{thinking}, base_url: {found.base_url}, "
                   f"model: {found.served_model}, thinking: {json.dumps(thinking)}}}")
    return out


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
    for note in _probe_notes(found, args):
        lines.append(note)
        lines.append("")
    for key, description, argv in agent_presets(found):
        lines.append(f"  # {key} -- {description}")
        lines.extend(_agent_command(argv))
        lines.append("")
    for note in _agent_notes(found):
        lines.append(note)
        lines.append("")
    levels = found.thinking.levels if found.thinking else []
    if len(levels) > 1:
        lines.append(
            f"  Thinking levels: {', '.join(levels)} each make a different prompt here, so\n"
            "  each is its own configuration. The `levels` preset above measures speed at\n"
            "  each; for quality, run the probe once per level with --reasoning-effort L\n"
            "  (and --save-result per level), and for agent behaviour, one arm per level:")
        lines.append("")
        lines.extend(f"    {line}" for line in level_arms(found))
        lines.append("")
    elif found.thinking:
        lines.append(
            "  No reasoning_effort level changes the prompt here, so there is no level\n"
            "  sweep to run: only --thinking on off makes two configurations.")
        lines.append("")
    if found.tokenizer:
        lines.append(
            f"  --tokenizer {found.tokenizer} because the served name is not a model\n"
            "  repository. It was read off the GGUF file name and kept because its\n"
            "  tokenizer loads; check it is the base model. Without it every shape is\n"
            "  counted with gpt2, and a `pp512` row sent 451-499 tokens on one engine.")
        lines.append("")
    elif found.tokenizer_ok is False:
        lines.append(
            f"  {found.hf_model} is not a model repository with a tokenizer, so the\n"
            "  benchmark will count shapes with gpt2 (a `pp512` row sent 451-499 tokens\n"
            "  on one engine). Add --tokenizer <base model repo> to the runs above.")
        lines.append("")
    if found.uses_key:
        lines.append(
            "  These commands carry no API key: they read it from LLM_ASSAY_API_KEY,\n"
            "  so export it in the shell that runs them. Do not add --api-key -\n"
            "  a key on the command line is visible in ps to every user on the machine."
        )
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
    probe_argv = ["probe", found.base_url, *_model_args(found)]
    probe_argv += args
    probe = {
        "key": key,
        "description": description,
        "args": probe_argv,
        "command": " ".join(["uv", "run", "llm-assay.py",
                             *(_quote(a) for a in probe_argv)]),
    }

    agent = [{"key": key, "description": description, "args": argv,
              "command": " ".join(["uv", "run", "llm-assay.py", *(_quote(a) for a in argv)])}
             for key, description, argv in agent_presets(found)]

    return {
        "endpoint": found.base_url,
        "probe": probe,
        "agent": agent,
        "harnesses": list(found.harnesses),
        "scenarios": [{"key": key, "description": description, "args": argv,
                       "command": " ".join(["uv", "run", "llm-assay.py",
                                            *(_quote(a) for a in argv)])}
                      for key, description, argv in scenario_presets(found)],
        # The models file the text output prints for a level sweep, one line per
        # entry; empty when fewer than two levels make different prompts.
        "level_arms": level_arms(found),
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
            "thinking_controls": found.thinking.controls if found.thinking else None,
            "thinking_route": found.thinking.route if found.thinking else None,
            "thinking_rejection": found.thinking.rejection if found.thinking else None,
            "thinking_levels": found.thinking.levels if found.thinking else None,
        },
        "depth_ladder": depth_ladder(found),
        "notes": list(found.notes),
        "presets": presets,
    }


def runner_keys(found: Detected) -> List[str]:
    """Every preset the written runner offers, in its order: the benchmark
    presets, then `probe`, which is assembled separately, then one agent run
    per installed harness. The confirmation
    message counts from here; counting `build_presets` alone told users a
    six-preset runner had five."""
    return ([key for key, _, _ in build_presets(found)] + [probe_preset(found)[0]]
            + [key for key, _, _ in agent_presets(found)]
            + [key for key, _, _ in scenario_presets(found)])


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
    agents = agent_presets(found)
    for key, description, _ in agents:
        out.append(f"  {key:<12} {description}")
    scenarios = scenario_presets(found)
    for key, description, _ in scenarios:
        out.append(f"  {key:<12} {description}")
    out += [
        "",
        "env overrides: BENCHY SEED OUTDIR RUNS EXTRA TAG DRY_RUN AB NOTE",
        "",
        "ab-before and ab-after share \\$OUTDIR/ab-\\$AB (AB defaults to 'deploy'): change the",
        "server between them. NOTE='what this side is' records a change the endpoint cannot",
        "show; EXTRA='--agent auto' on ab-before adds an agent arm to both sides.",
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
        *(['# This endpoint needed an API key. It is read from the environment, never',
           '# written here or passed on a command line.',
           'if [[ -z "${LLM_ASSAY_API_KEY:-}" ]]; then',
           '  echo "warning: LLM_ASSAY_API_KEY is not set; this endpoint needed a key" >&2',
           'fi'] if found.uses_key else []),
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
    ptarget = ["probe", found.base_url, *_model_args(found)]
    pbody = _wrap([_quote(a) for a in [*ptarget, *pargs]], "          ")
    pbody[0] = "    args=(" + pbody[0].lstrip()
    pbody[-1] = pbody[-1] + ")"
    # IS_PROBE because the tail below adds --runs/--format/--latency-mode, none
    # of which the probe subcommand accepts.
    out += [f"  {pkey})", *pbody, "    IS_PROBE=1", "    ;;"]

    for key, _, argv in agents:
        abody = _wrap([_quote(a) for a in argv], "          ")
        abody[0] = "    args=(" + abody[0].lstrip()
        abody[-1] = abody[-1] + ")"
        # IS_AGENT: no --seed, --format or --latency-mode there either, and a
        # run's result is compared with another agent result, not a benchmark.
        out += [f"  {key})", *abody, "    IS_AGENT=1", "    ;;"]

    for key, _, argv in scenarios:
        sbody = _wrap([_quote(a) for a in argv], "          ")
        sbody[0] = "    args=(" + sbody[0].lstrip()
        sbody[-1] = sbody[-1] + ")"
        where = ('"$OUTDIR/ab-${AB:-deploy}"' if key.startswith("ab-")
                 else '"$OUTDIR/$TAG"')
        out += [f"  {key})", *sbody, f"    SCENARIO_DIR={where}", "    ;;"]

    out += [
        "  *)",
        '    echo "error: unknown preset \'$preset\'" >&2; usage >&2; exit 2 ;;',
        "esac",
        "",
        'if [[ -n "${SCENARIO_DIR:-}" ]]; then',
        '  args+=(--dir "$SCENARIO_DIR")',
        '  [[ -n "${NOTE:-}" && "$preset" == ab-* ]] && args+=(--note "$NOTE")',
        # Only ab-before measures with a --runs of its own; ab-after repeats
        # exactly what before ran, and qualify's steps are fixed-size checks.
        '  [[ -n "$RUNS" && "$preset" == ab-before ]] && args+=(--runs "$RUNS")',
        # SEED too, so an A/B meant to pair with an earlier run can: ab-after
        # reuses whatever seed before kept, and qualify picks its own.
        '  [[ "$preset" == ab-before ]] && args+=(--seed "$SEED")',
        "  # shellcheck disable=SC2206",
        '  [[ -n "$EXTRA" ]] && args+=($EXTRA)',
        '  RESULT="$SCENARIO_DIR/VERDICT.txt"',
        '  [[ "$preset" == ab-before ]] && RESULT="$SCENARIO_DIR/VERDICT-before.txt"',
        'elif [[ -n "${IS_PROBE:-}" ]]; then',
        # Transcripts too: the fabrication evidence -- which lines a model
        # cited, and whether they exist -- lives there, not in the result.
        # EXTRA but not RUNS: the probe counts --trials, which EXTRA can set.
        "  # shellcheck disable=SC2206",
        '  [[ -n "$EXTRA" ]] && args+=($EXTRA)',
        '  args+=(--seed "$SEED" --save-result "$RESULT" --save-transcripts "${RESULT%.json}-transcripts.json")',
        'elif [[ -n "${IS_AGENT:-}" ]]; then',
        '[[ -n "$RUNS" ]] && args+=(--runs "$RUNS")',
        "# shellcheck disable=SC2206",
        '[[ -n "$EXTRA" ]] && args+=($EXTRA)',
        '  args+=(--save-result "$RESULT")',
        "else",
        '[[ -n "$RUNS" ]] && args+=(--runs "$RUNS")',
        "# shellcheck disable=SC2206",
        '[[ -n "$EXTRA" ]] && args+=($EXTRA)',
        'args+=(--seed "$SEED" --format json --save-result "$RESULT")',
        "fi",
        "",
        'if [[ ( -n "${SCENARIO_DIR:-}" && "$preset" != ab-before ) || -n "${IS_AGENT:-}" ]]; then',
        '  echo "==> preset=$preset"',
        "else",
        '  echo "==> preset=$preset seed=$SEED"',
        "fi",
        'echo "    result: $RESULT"',
        "printf '   '; printf ' %q' $BENCHY \"${args[@]}\"; echo",
        "",
        '[[ -n "${DRY_RUN:-}" ]] && exit 0',
        "",
        'if [[ -n "${SCENARIO_DIR:-}" ]]; then',
        '  exec $BENCHY "${args[@]}"',
        "fi",
        '$BENCHY "${args[@]}"',
        "",
        'if [[ -n "${IS_AGENT:-}" ]]; then',
        "cat <<EOF",
        "",
        "Saved: $RESULT",
        "",
        "To A/B against another configuration, run it the same way (RUNS=20 on both",
        "sides for a comparison worth reading), then compare by task:",
        "    TAG=other $0 $preset",
        "    $BENCHY compare $RESULT $OUTDIR/other.json",
        "EOF",
        "else",
        "cat <<EOF",
        "",
        "Saved: $RESULT",
        "",
        "To A/B against another run, use the SAME seed on both sides:",
        "    SEED=$SEED TAG=other $0 $preset",
        "    $BENCHY compare $RESULT $OUTDIR/other.json",
        "EOF",
        "fi",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py tune",
        description="Detect what an endpoint is and suggest benchmark runs that fit it.",
    )
    ap.add_argument("base_url", help="OpenAI-compatible endpoint URL, including /v1")
    ap.add_argument("--api-key", default=None,
                    help="API key for the endpoint. Prefer the LLM_ASSAY_API_KEY "
                         "environment variable, which the suggested commands also "
                         "read: a key given here is visible in ps")
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

    from .config import API_KEY_ENV, resolve_api_key
    args.api_key = resolve_api_key(args.api_key)
    found = detect(args.base_url, args.api_key, args.model)
    found.harnesses = installed_harnesses()
    found.sandbox = shutil.which("bwrap") is not None
    # The suggestions never carry the key; they read it from the environment,
    # as every subcommand does. Saying so is what keeps someone from pasting it
    # back onto the command line when the first suggested run fails with a 401.
    found.uses_key = args.api_key not in ("", "EMPTY")
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
