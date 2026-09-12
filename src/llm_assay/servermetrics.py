"""Optional scraping of a vLLM-style Prometheus ``/metrics`` endpoint.

Two things the client cannot see from timings alone, but which explain most of
what makes a benchmark hard to read:

* **Prefix cache hit rate.** In ``--measure-cached-followup`` mode the second
  phase re-sends the same context and is *reported* as a follow-up turn. If the
  server has prefix caching disabled that phase silently measures a full
  re-prefill instead, and nothing in the timings says so. Scraping the hit rate
  turns that from an invisible misreading into a loud warning. The same counter
  catches the mirror-image failure -- an ordinary run whose prompt was already
  resident in the server's cache, so its prefill numbers measure a lookup
  rather than the ingest they claim to.
* **Speculative-decode acceptance.** Acceptance is content-dependent and is the
  dominant source of decode variance on MTP/Eagle setups. Recording it per run
  lets you explain the spread instead of just absorbing it.

Everything here is best-effort: a missing or unparseable endpoint disables the
feature rather than failing the benchmark.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field

# Counter names differ across engines and versions; try each in order. These are
# monotonic counters, so they are read twice and diffed to get per-shape numbers.
# Consecutive failed scrapes before the probe gives up for the rest of the run.
_MAX_TRANSIENT_FAILURES = 3

_COUNTERS = {
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    "prefix_hits": (
        "vllm:prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits_total",
        "llamacpp:prompt_tokens_cached_total",
    ),
    "spec_accepted": (
        "vllm:spec_decode_num_accepted_tokens_total",
        "llamacpp:spec_decode_num_accepted_tokens_total",
    ),
    "spec_drafts": (
        "vllm:spec_decode_num_drafts_total",
        "llamacpp:spec_decode_num_drafts_total",
    ),
    "spec_draft_tokens": (
        "vllm:spec_decode_num_draft_tokens_total",
        "llamacpp:spec_decode_num_draft_tokens_total",
    ),
    # Not a metric anyone asked for -- it is how we find out whether the numbers
    # above describe this run at all. Every counter here is server-global, so a
    # second client on the endpoint lands inside our window silently.
    # llama.cpp publishes no completed-request counter, so this stays None
    # there and the concurrent-traffic check falls back to its gauge tell alone.
    "requests_finished": ("vllm:request_success_total",),
}

# Logical values an engine publishes in pieces rather than whole. llama.cpp
# counts prompt tokens it *processed* and prompt tokens it served from cache in
# two separate counters, so the number queried is their sum -- measured on a
# 3081-token prompt sent twice, the cold pass moved
# (prompt_tokens_total=3081, cached=0) and the repeat moved (4, 3070). Mapping
# prefix_queries straight onto prompt_tokens_total would have divided 3070 by 4
# and published a 76,750% hit rate. Every component must be present before a sum
# is used: a partially published family would otherwise yield a denominator that
# is too small, which is the same failure wearing a plausible number.
_SUMMED = {
    "prefix_queries": (
        ("llamacpp:prompt_tokens_total", "llamacpp:prompt_tokens_cached_total"),
    ),
}

# Some engines publish the same facts as *gauges* rather than counters -- SGLang
# exposes a cumulative cache hit rate and mean accepted length directly. Those
# cannot be diffed into a per-shape figure, so they are read as-is and labelled
# with a wider scope rather than being passed off as per-shape.
#
# Names here are best-effort: they were not verified against a live SGLang, and
# an unrecognised endpoint now says so out loud instead of silently disabling,
# which is what makes a wrong guess self-correcting.
_GAUGES = {
    "prefix_hit_rate": ("sglang:cache_hit_rate",),
    "spec_accept_len": ("sglang:spec_accept_length",),
    # Not part of the sglang fallback -- diff() only ever consults the two
    # logical names above. This one answers "was anyone else mid-request when we
    # started?", which the completed-request counter cannot: a stream that spans
    # our whole window never increments it.
    "requests_running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
}

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eEnaN]+)$")


class ServerCounters(BaseModel):
    """A point-in-time read of the counters we care about."""

    raw: Dict[str, float] = Field(default_factory=dict)
    gauges: Dict[str, float] = Field(default_factory=dict)
    # Metric-name prefixes seen in the response, whether or not recognised. Used
    # only to tell the user what the endpoint *does* publish when nothing matched.
    prefixes: List[str] = Field(default_factory=list)

    def get(self, logical: str) -> Optional[float]:
        for name in _COUNTERS.get(logical, ()):
            if name in self.raw:
                return self.raw[name]
        for names in _SUMMED.get(logical, ()):
            if all(name in self.raw for name in names):
                return sum(self.raw[name] for name in names)
        return None

    def gauge(self, logical: str) -> Optional[float]:
        for name in _GAUGES.get(logical, ()):
            if name in self.gauges:
                return self.gauges[name]
        return None

    def has_anything(self) -> bool:
        return bool(self.raw or self.gauges)


class PhaseCounterAccumulator:
    """Sums raw counter movement per benchmark phase across a shape's runs.

    Under ``--measure-cached-followup`` the two phases interleave -- ctx, inf,
    ctx, inf -- so one before/after pair per shape cannot separate them. Summing
    each phase's own movement can, and feeding the totals back through
    :meth:`ServerMetricsProbe.diff` reuses the rate arithmetic, the counter-reset
    guard and the gauge fallback rather than restating any of it here.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.last_gauges: Dict[str, float] = {}
        self.samples = 0

    def add(self, before: Optional[ServerCounters], after: Optional[ServerCounters]) -> None:
        if before is None or after is None:
            return
        for name, end in after.raw.items():
            start = before.raw.get(name)
            if start is None:
                continue
            moved = end - start
            # A counter that went backwards means the server restarted mid-shape;
            # the window is unusable, so drop the sample rather than add a
            # negative that would quietly deflate the phase total.
            if moved < 0:
                continue
            self.totals[name] = self.totals.get(name, 0.0) + moved
        self.last_gauges = dict(after.gauges)
        self.samples += 1

    def as_pair(self) -> Tuple[Optional[ServerCounters], Optional[ServerCounters]]:
        """The accumulated totals shaped as a (before, after) pair for diff()."""
        if not self.samples:
            return None, None
        before = ServerCounters(raw={name: 0.0 for name in self.totals})
        after = ServerCounters(raw=dict(self.totals), gauges=dict(self.last_gauges))
        return before, after


class ServerMetricsDelta(BaseModel):
    """Counter movement across one measured run."""

    prefix_cache_queries: Optional[float] = Field(None, description="Prefix cache lookups during the run")
    prefix_cache_hits: Optional[float] = Field(None, description="Prefix cache hit tokens/blocks during the run")
    prefix_cache_hit_rate: Optional[float] = Field(None, description="hits/queries during the run (0..1)")
    spec_acceptance_rate: Optional[float] = Field(None, description="accepted/drafted speculative tokens (0..1)")
    spec_acceptance_length: Optional[float] = Field(None, description="Mean accepted tokens per draft, +1 for the bonus token")
    requests_running_at_start: Optional[float] = Field(
        None,
        description=(
            "Requests the server had in flight when this shape's window opened, "
            "when llm-assay had none of its own. Anything above zero is "
            "another client, including a long stream that never completes inside "
            "our window and so never moves the completed-request counter."
        ),
    )
    requests_finished: Optional[float] = Field(
        None,
        description=(
            "Requests the server completed during this window, from its own "
            "counter. Compared against the number llm-assay sent: more means "
            "another client shared the endpoint, and every figure here describes "
            "the box rather than this run."
        ),
    )
    scope: str = Field(
        "shape",
        description="'shape' when diffed from counters around this shape; 'shape (both phases)' under --measure-cached-followup, where one delta spans the context load and the follow-up together; 'server-lifetime' when the engine only publishes gauges, which cannot be attributed to one shape",
    )

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "prefix_cache_queries",
                "prefix_cache_hit_rate",
                "spec_acceptance_rate",
            )
        )


def _metrics_url(base_url: str) -> str:
    """Derive the /metrics URL from an OpenAI base URL (…/v1 -> …/metrics)."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/metrics"


def _parse(text: str) -> ServerCounters:
    wanted = {name for names in _COUNTERS.values() for name in names}
    wanted |= {name for groups in _SUMMED.values() for names in groups for name in names}
    gauge_names = {name for names in _GAUGES.values() for name in names}
    raw: Dict[str, float] = {}
    gauges: Dict[str, float] = {}
    prefixes: Dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name = match.group("name")
        prefix = name.split(":", 1)[0] if ":" in name else name.split("_", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if name in wanted:
            # Counters may be reported per-engine; sum the label sets.
            raw[name] = raw.get(name, 0.0) + value
        elif name in gauge_names:
            # A gauge is a level, not a total: take the last one seen rather
            # than summing across label sets.
            gauges[name] = value
    ranked = sorted(prefixes.items(), key=lambda kv: -kv[1])
    return ServerCounters(
        raw=raw, gauges=gauges, prefixes=[p for p, _ in ranked[:5]]
    )


class ServerMetricsProbe:
    """Fetches and diffs server counters around a measured run."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", enabled: bool = True):
        self.url = _metrics_url(base_url)
        self.enabled = enabled
        self.headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "EMPTY" else {}
        )
        self.available: Optional[bool] = None
        # A transient read must not disable the run. Only a definitive
        # negative (a non-200) proves the endpoint is absent; a timeout
        # is plausible mid-prefill and says nothing about the next shape.
        self._transient_failures = 0

    async def snapshot(self, session: aiohttp.ClientSession) -> Optional[ServerCounters]:
        if not self.enabled or self.available is False:
            return None
        try:
            async with session.get(self.url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    self.available = False
                    return None
                counters = _parse(await resp.text())
                self._transient_failures = 0
        except Exception:
            # Give up only after this keeps happening: the two warnings that
            # catch silent misreadings live behind this probe, so trading them
            # away for one slow scrape is the wrong bargain. Retrying costs a
            # request per shape against an endpoint that is already answering
            # the benchmark itself.
            self._transient_failures += 1
            if self._transient_failures >= _MAX_TRANSIENT_FAILURES:
                self.available = False
            return None

        if not counters.has_anything():
            # Readable, parseable, and none of it recognised. Saying so beats
            # disabling the feature in silence: the prefixes name the engine, so
            # a missing family is reported rather than guessed at.
            if counters.prefixes and self.available is None:
                print(
                    f"[WARNING] {self.url} is readable but publishes no metrics this "
                    "tool recognises.\n"
                    f"          Found: {', '.join(counters.prefixes)}. Prefix-cache and "
                    "speculative-decode\n"
                    "          reporting are disabled for this run; the benchmark itself "
                    "is unaffected."
                )
            self.available = False
            return None
        self.available = True
        return counters

    @staticmethod
    def diff(before: Optional[ServerCounters], after: Optional[ServerCounters]) -> Optional[ServerMetricsDelta]:
        if before is None or after is None:
            return None

        def delta(logical: str) -> Optional[float]:
            a, b = before.get(logical), after.get(logical)
            if a is None or b is None:
                return None
            moved = b - a
            # Counter reset (server restart mid-suite) -- treat as unusable.
            return moved if moved >= 0 else None

        requests_finished = delta("requests_finished")
        running_at_start = before.gauge("requests_running")
        queries = delta("prefix_queries")
        hits = delta("prefix_hits")
        accepted = delta("spec_accepted")
        drafts = delta("spec_drafts")
        draft_tokens = delta("spec_draft_tokens")

        hit_rate = (hits / queries) if (queries and hits is not None and queries > 0) else None
        acc_rate = (
            (accepted / draft_tokens)
            if (draft_tokens and accepted is not None and draft_tokens > 0)
            else None
        )
        # +1: every draft round also emits the verified bonus token.
        acc_len = (
            (accepted / drafts + 1.0)
            if (drafts and accepted is not None and drafts > 0)
            else None
        )

        # Engines that publish only gauges (SGLang's cumulative cache_hit_rate and
        # spec_accept_length) cannot be diffed into a per-shape figure. Use the
        # value as it stands and say so, rather than dropping the signal or
        # presenting a lifetime average as if it described this shape.
        scope = "shape"
        if hit_rate is None:
            gauge_rate = after.gauge("prefix_hit_rate")
            if gauge_rate is not None:
                # Some engines report a percentage, others a fraction.
                hit_rate = gauge_rate / 100.0 if gauge_rate > 1.0 else gauge_rate
                scope = "server-lifetime"
        if acc_len is None:
            gauge_len = after.gauge("spec_accept_len")
            if gauge_len is not None:
                acc_len = gauge_len
                scope = "server-lifetime"

        result = ServerMetricsDelta(
            scope=scope,
            requests_finished=requests_finished,
            requests_running_at_start=running_at_start,
            prefix_cache_queries=queries,
            prefix_cache_hits=hits,
            prefix_cache_hit_rate=hit_rate,
            spec_acceptance_rate=acc_rate,
            spec_acceptance_length=acc_len,
        )
        return None if result.is_empty() else result


def concurrent_traffic_warning(
    delta: Optional[ServerMetricsDelta], requests_sent: int
) -> Optional[str]:
    """Warn when someone else was using the endpoint during this shape.

    Every counter in ``_COUNTERS`` is server-global. llm-assay diffs them
    across a shape and reports the movement as if it belonged to this run, which
    holds only while nothing else is talking to the server. A second client's
    requests land inside the same window, so its cache hits and draft
    acceptances are silently blended into ours -- and the two warnings that
    exist to catch a misread prefix cache inherit the error.

    The server's own completed-request counter settles it: we know exactly how
    many requests we sent, so anything above that came from somewhere else. The
    comparison is deliberately one-sided. Fewer completions than we sent is
    normal -- a request still streaming when the window closes has not been
    counted yet -- and only an excess proves company.
    """
    if delta is None or requests_sent <= 0:
        return None

    # Two independent tells, because either alone has a blind spot. The counter
    # misses a stream that spans the whole window without completing; the gauge
    # misses a client that starts after our snapshot.
    running = delta.requests_running_at_start
    if running is not None and running >= 1:
        return (
            f"[WARNING] the server already had {running:.0f} request(s) in flight when "
            "this shape started,\n"
            "          and llm-assay had none of its own. Another client is using the "
            "endpoint,\n"
            "          so the prefix cache and speculative decode figures above are "
            "diffed from\n"
            "          counters the whole box shares, and the GPU was contended while we "
            "timed it.\n"
            "          Re-run against an idle endpoint before trusting these numbers."
        )

    if delta.requests_finished is None:
        return None
    extra = delta.requests_finished - requests_sent
    if extra < 1:
        return None
    return (
        f"[WARNING] the server completed {delta.requests_finished:.0f} requests during "
        f"this shape but llm-assay sent {requests_sent}.\n"
        f"          Another client was using the endpoint, so the prefix cache and "
        "speculative\n"
        "          decode figures above describe the server, not this run -- they are "
        "diffed from\n"
        "          counters the whole box shares. Timings are affected too: a shared GPU "
        "is a\n"
        "          slower one. Re-run against an idle endpoint before trusting these "
        "numbers."
    )


def cached_followup_warning(
    delta: Optional[ServerMetricsDelta],
    measure_cached_followup: bool,
    depth: int,
    threshold: float = 0.01,
) -> Optional[str]:
    """Warn when a cached-follow-up measurement got no cache hits.

    This is the failure that silently corrupts interpretation: the two-phase mode
    labels phase 2 as ``pp/tg @ depth``, but if the server never reused the
    context that row is a full re-prefill. The timings look plausible either way,
    so only the server's own counters can tell the difference.
    """
    if not measure_cached_followup or depth <= 0 or delta is None:
        return None
    rate = delta.prefix_cache_hit_rate
    if rate is None or rate >= threshold:
        return None
    if delta.scope == "shape (follow-up)":
        # Measured between the two phases, so this is the follow-up's own rate --
        # no longer diluted by the always-cold context load.
        measured = "on the follow-up turn itself"
    elif delta.scope == "shape (context load)":
        measured = "during the context load"
    elif delta.scope == "shape":
        measured = "during this shape"
    elif delta.scope.startswith("shape"):
        measured = "across the context load and the follow-up together"
    else:
        measured = "over its lifetime"
    return (
        "[WARNING] --measure-cached-followup is on but the server reported "
        f"{rate:.1%} prefix cache hits {measured}.\n"
        "          The 'pp/tg @ depth' rows are therefore a FULL RE-PREFILL of the "
        "context, not a\n"
        "          cached follow-up turn. Enable prefix caching on the server "
        "(vLLM needs an explicit\n"
        "          --enable-prefix-caching for hybrid models) to measure what this "
        "mode is meant to measure."
    )


def stale_prefix_cache_warning(
    delta: Optional[ServerMetricsDelta],
    measure_cached_followup: bool,
    threshold: float = 0.10,
) -> Optional[str]:
    """Warn when a plain run had much of its prompt served from the prefix cache.

    The mirror image of :func:`cached_followup_warning`. Outside
    ``--measure-cached-followup`` every request is meant to be a full prefill, so
    ``t/s``, ``ttfr``, ``est_ppt`` and ``e2e_ttft`` are read as the cost of
    ingesting the whole prompt. When the server already holds the prefix it skips
    that work, and those metrics measure a cache lookup instead -- inflated
    severalfold, with nothing in the timings to say so.

    The usual cause is a previous invocation rather than a server setting:
    ``--seed`` makes corpus sampling deterministic, so re-running the same
    command sends byte-identical prompts and the second run hits the cache the
    first one populated. ``--no-cache`` does not prevent it -- its cache-buster
    is derived from the seed, so it is identical across invocations too.

    The threshold sits well above the handful of chat-template tokens every
    request shares, so a genuinely cold run reports 0% and stays silent.
    """
    if measure_cached_followup or delta is None:
        return None
    if not delta.scope.startswith("shape"):
        # A cumulative gauge cannot say whether *this* run was served warm, and
        # a server that has been busy all day would trip this on every run.
        return None
    rate = delta.prefix_cache_hit_rate
    if rate is None or rate < threshold:
        return None
    return (
        f"[WARNING] the server served {rate:.1%} of this run's prompt tokens from its "
        "prefix cache.\n"
        "          Prefill metrics (t/s, ttfr, est_ppt, e2e_ttft) therefore measure a "
        "PARTLY CACHED\n"
        "          prefill and overstate prefill throughput. A previous run of the same "
        "shape with\n"
        "          the same --seed is the usual cause; --no-cache does not help, since its\n"
        "          cache-buster is seed-derived. Vary --seed per invocation for a cold "
        "prefill, or\n"
        "          compare only decode (tg) across invocations."
    )
