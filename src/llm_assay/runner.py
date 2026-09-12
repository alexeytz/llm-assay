import hashlib
import asyncio
import os
import subprocess
import time
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import urllib.request
import aiohttp

from . import __version__
from .config import BenchmarkConfig
from .client import CONTEXT_LOAD_USER_MESSAGE, LLMClient
from .corpus import tokenizer_fallback
from .prompts import PromptGenerator
from .results import BenchmarkResults, BenchmarkMetadata
from .servermetrics import (
    PhaseCounterAccumulator,
    ServerMetricsProbe,
    cached_followup_warning,
    concurrent_traffic_warning,
    stale_prefix_cache_warning,
)

def _duration(seconds: float) -> str:
    """Sub-second waits read better in ms; anything longer in seconds."""
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _run_summary(results: Sequence[Any]) -> str:
    """What a run actually did, for the line that announced it.

    Raw observables only -- token counts, time to the first content token, and
    wall clock. A throughput printed here would have to either duplicate
    results.py's latency subtraction or quietly disagree with the table printed
    underneath it, and a progress line that contradicts the report is worse than
    one that says less. `est_ppt` and the t/s columns stay the report's job.

    At concurrency > 1 the ttft is the slowest request's, because that is when
    the batch actually had all its first tokens.
    """
    if not results:
        return "  no result"
    failed = [r for r in results if r.error]
    if len(failed) == len(results):
        return f"  FAILED: {(failed[0].error or 'unknown error')[:90]}"

    ok = [r for r in results if not r.error]
    bits = [f"{max(r.prompt_tokens for r in ok):,} tok prompt"]
    ttfts = [r.first_token_ts - r.start_ts for r in ok if r.first_token_ts]
    if ttfts:
        bits.append(f"ttft {_duration(max(ttfts))}")
    wall = max(r.end_ts for r in ok) - min(r.start_ts for r in ok)
    bits.append(f"{sum(r.total_tokens for r in ok):,} gen in {_duration(wall)}")
    if failed:
        bits.append(f"{len(failed)}/{len(results)} requests FAILED")
    return "  " + ", ".join(bits)


class BenchmarkFailure(Exception):
    pass

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: PromptGenerator, progress=None):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results = BenchmarkResults()
        self.progress = progress
        self._next_request_id = 0
        self.metrics_probe = ServerMetricsProbe(
            config.base_url, config.api_key, enabled=config.server_metrics
        )
        # Per-shape server counter deltas, keyed by (depth, pp, tg, concurrency, effort).
        self.server_deltas: Dict[Tuple[Any, ...], Any] = {}

        # We need to track deltas from warmup to adapt prompts
        self.run_salt: Optional[str] = None
        self.delta_user = 0
        self.delta_context = 0

    def _new_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    @staticmethod
    def _target_label(effort: Optional[str], thinking: Optional[str]) -> str:
        """Name the swept dimensions this request belongs to.

        A consumer of the progress stream watching a `--thinking on off` sweep
        otherwise cannot tell which mode a request came from -- every other
        output surface carries it, and this one did not.
        """
        parts = []
        if thinking:
            parts.append(f"think={thinking}")
        if effort:
            parts.append(f"re={effort}")
        return " ".join(parts)

    @staticmethod
    def corpus_key(
        depth: int,
        pp: int,
        tg: int,
        concurrency: int,
        run_index: int,
        effort: Optional[str],
        thinking: Optional[str],
        is_warmup: bool = False,
    ) -> Tuple[Any, ...]:
        """Which slice of the corpus a shape reads.

        Every swept dimension belongs in here, or two arms of an A/B send
        byte-identical prompts and the second one's prefill is served from the
        cache the first built -- inside a single invocation, with nothing the
        user did wrong, and with --no-cache powerless because its buster is
        appended after the prefix.

        ``thinking`` is appended only when set, so the tuple keeps its old shape
        whenever --thinking is not sweeping. ``repr()`` of the key is what the
        offset hashes, so an unconditional element would move every slice and
        stop new results pairing with anything saved earlier -- the same rule the
        --cold salt follows.

        A method rather than an inline literal so the corpus-key test pins
        *this* construction. A test that rebuilds the tuple itself passes
        happily while the runner drops a dimension, which is how every previous
        dimension bug shipped.
        """
        key: Tuple[Any, ...] = (
            "warmup" if is_warmup else "run",
            depth, pp, tg, concurrency, run_index, effort,
        )
        if thinking is not None:
            key = key + (thinking,)
        return key

    def _emit_request_start(self, request_id: int, pp: int, tg: int, depth: int, concurrency: int, run_index: int, effort=None, thinking=None, phase: str = "standard") -> None:
        if self.progress is None:
            return
        try:
            self.progress.request_start(
                request_id=request_id,
                model=self.config.model,
                base_url=self.config.base_url,
                prompt_size=pp,
                response_size=tg,
                context_size=depth,
                concurrency=concurrency,
                run_index=run_index,
                target_label=self._target_label(effort, thinking),
                phase=phase,
            )
        except Exception:
            pass

    @staticmethod
    def _batch_decode_rate(batch) -> float:
        """Decode tokens/s for one batch, mirroring the reporting definition.

        Counts only tokens observed strictly after the first token timestamp, so
        the measured interval is one the server actually spent generating.

        The denominator must be the batch-wide span -- last token of any request
        minus first token of any request -- because that is what the reported
        tg total divides by. Taking the longest single request's span instead
        gives the same numerator a smaller denominator whenever arrivals are
        staggered, so the gate would score a rate the report never publishes,
        and --target-ci (a *relative* CI) would stop sampling early on a mean
        inflated above the one the user is shown.
        """
        total_tokens = 0
        first_stamps: List[float] = []
        last_stamps: List[float] = []
        for r in batch:
            ts = getattr(r, "token_timestamps", None) or []
            # A request whose stamps carry no interval contributes no tokens, but
            # it still widens the window the report divides by -- results.py takes
            # its first_token_ts and its last stamp (or end_ts) into the batch
            # span. Skipping it here too would leave the gate scoring a shorter
            # window than the report -- the same divergence as taking the longest
            # single request's span instead of the batch's, just smaller.
            if len(ts) >= 2:
                total_tokens += sum(1 for t in ts if t > ts[0])
            first_ts = getattr(r, "first_token_ts", None) or (ts[0] if ts else None)
            if first_ts is None:
                continue
            first_stamps.append(first_ts)
            last_stamps.append(ts[-1] if ts else (getattr(r, "end_ts", None) or first_ts))
        span = (max(last_stamps) - min(first_stamps)) if first_stamps else 0.0
        return (total_tokens / span) if span > 0 else 0.0

    def _decode_ci_reached(self, run_results) -> bool:
        """True when the observed decode samples satisfy --target-ci."""
        target = self.config.target_ci
        if not target:
            return True
        rates = [r for r in (self._batch_decode_rate(b) for b in run_results) if r > 0]
        if len(rates) < max(2, self.config.num_runs):
            return False
        metric = self.results._calculate_metric(rates)
        return bool(metric and metric.rel_ci95 <= target)

    def _tokenizer_fallback(self) -> Optional[str]:
        """The name gpt2 stood in for, "" if nothing fell back, None if unknown.

        The three states are distinct and comparability depends on it: None is
        reserved for "not recorded", which it skips, so "nothing fell back"
        has to be "" or a run counted with gpt2 compares silently against one
        counted correctly. With no corpus to ask -- which no real run reaches,
        the shapes are sliced from it -- unknown is the honest answer.
        """
        corpus = getattr(self.prompt_gen, "corpus", None)
        if corpus is None:
            return None
        return tokenizer_fallback(corpus.get_tokenizer()) or ""

    def _metadata(self, latency: float, max_concurrency: int) -> BenchmarkMetadata:
        """Everything needed to reproduce this run, minus the API key.

        Built in one place because the interrupted path saves results too, and a
        partial result that describes itself differently from a complete one is
        worse than useless.
        """
        cfg = self.config
        return BenchmarkMetadata(
            version=__version__,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            latency_mode=cfg.latency_mode,
            latency_ms=latency * 1000,
            model=cfg.model,
            prefix_caching_enabled=cfg.enable_prefix_caching,
            seed=cfg.seed,
            reasoning_efforts=cfg.reasoning_efforts,
            thinking_modes=cfg.thinking_modes,
            max_concurrency=max_concurrency,
            base_url=cfg.base_url,
            served_model_name=cfg.served_model_name,
            tokenizer=cfg.tokenizer,
            tokenizer_fallback=self._tokenizer_fallback(),
            book_url=cfg.book_url,
            endpoint=cfg.endpoint,
            exact_tg=cfg.exact_tg,
            no_cache=cfg.no_cache,
            cold=cfg.cold or None,
            cold_salt=self.run_salt,
            adapt_prompt=cfg.adapt_prompt,
            skip_coherence=cfg.skip_coherence,
            num_runs=cfg.num_runs,
            warmup_runs=cfg.warmup_runs,
            target_ci=cfg.target_ci,
            max_runs=cfg.max_runs,
            # {} rather than None when unset: None is reserved for "written by a
            # version that did not record this", which comparability treats as
            # unknown and skips. Collapsing "no extra body" into None would make
            # a run with --extra-body compare silently against one without.
            extra_body=cfg.extra_body,
        )

    async def run_suite(self):
        # Initialize session
        # No total cap. A 256k-token prefill can legitimately take many minutes,
        # and a hard ceiling silently kills the measurement mid-run -- the very
        # shapes people most want to measure are the ones it would cut off.
        # A stalled-but-open stream is the real failure, so bound the gap between
        # chunks instead: a slow server is fine, a dead one is caught in minutes.
        stall = self.config.stall_timeout or None
        timeout = aiohttp.ClientTimeout(
            total=None, connect=None, sock_connect=30, sock_read=stall
        )
        max_concurrency = max(self.config.concurrency_levels)
        connector = aiohttp.TCPConnector(limit=max_concurrency + 5, force_close=False, keepalive_timeout=600)
        latency = 0.0  # default in case of early interrupt

        # trust_env=True is kept, because on some networks the proxy is the only
        # route to the endpoint. But a proxy sits in the middle of every
        # timing-sensitive request, so it must not be silent: every number in
        # this run would be measuring the proxy as much as the server.
        proxies = {
            var: os.environ[var]
            for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY")
            if os.environ.get(var)
        }
        # ...unless NO_PROXY already exempts this endpoint. aiohttp honours it
        # too, so warning anyway would state the opposite of what happens, and
        # advise the escape hatch the user had already taken.
        bypassed = False
        host = urlparse(self.config.base_url).hostname
        if host:
            # Not in typeshed, but present on every platform: it is the
            # env-only check, which is the one aiohttp's trust_env applies.
            # proxy_bypass() would consult macOS/Windows system config instead.
            bypass = getattr(urllib.request, "proxy_bypass_environment", None)
            try:
                bypassed = bool(bypass(host)) if bypass else False
            except Exception:
                bypassed = False
        if proxies and not bypassed:
            named = ", ".join(f"{k}={v}" for k, v in sorted(proxies.items()))
            print(
                "[WARNING] a proxy is set in the environment and aiohttp will honour it:\n"
                f"          {named}\n"
                "          Every request is routed through it, so latency, TTFT and "
                "throughput measure\n"
                "          the proxy as much as the server. Unset it, or add the endpoint "
                "to NO_PROXY."
            )
        elif proxies and bypassed:
            print(
                f"[INFO] a proxy is set in the environment but NO_PROXY exempts {host}, "
                "so requests go direct."
            )

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
                # Warmup
                should_warmup = not self.config.no_warmup
                if self.config.adapt_prompt:
                    should_warmup = True

                tokenizer = self.prompt_gen.corpus.get_tokenizer()

                if should_warmup:
                    self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

                # Coherence test after warmup (by default, unless skipped)
                if not self.config.skip_coherence:
                    if not await self.client.run_coherence_test(
                        session,
                        prompt=self.config.coherence_prompt,
                        expect=self.config.coherence_expect,
                    ):
                        print("\nBenchmark failed due to coherence test failure.")
                        raise SystemExit(1)
                else:
                    print("\nSkipping coherence test (--skip-coherence specified)")

                # Measure latency
                warmup_runs = 0 if self.config.no_warmup else self.config.warmup_runs
                latency = await self.client.measure_latency(
                    session,
                    self.config.latency_mode,
                    warmup_runs=warmup_runs,
                )
                if self.progress is not None:
                    try:
                        self.progress.latency_measured(
                            latency_s=latency, mode=self.config.latency_mode
                        )
                    except Exception:
                        pass

                # Main Loop
                # A single None means "run once without sending reasoning_effort".
                efforts: List[Optional[str]] = (
                    list(self.config.reasoning_efforts)
                    if self.config.reasoning_efforts
                    else [None]
                )
                modes: List[Optional[str]] = (
                    list(self.config.thinking_modes)
                    if self.config.thinking_modes
                    else [None]
                )
                # One flat product rather than another level of nesting: the loop
                # in here is already six deep and adding a seventh for a
                # two-valued dimension is not worth the indentation.
                for thinking, effort in [(m, e) for m in modes for e in efforts]:
                    if thinking is not None:
                        print(f"\n=== thinking={thinking} ===")
                    if effort is not None:
                        print(f"\n=== reasoning_effort={effort} ===")
                    for depth in self.config.depths:
                        for pp in self.config.pp_counts:
                            for tg in self.config.tg_counts:
                                for concurrency in self.config.concurrency_levels:
                                    print(f"Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={concurrency}")

                                    run_std_results: List[List[Any]] = []
                                    run_ctx_results: List[List[Any]] = []
                                    expected_pp = pp
                                    expected_ctx = depth

                                    total_runs = self.config.num_runs + warmup_runs
                                    hard_cap = warmup_runs + (
                                        self.config.max_runs
                                        if self.config.max_runs
                                        else self.config.num_runs * 5
                                    )
                                    # Carries every swept dimension, like
                                    # compare.shape_key and _shape_label. Store
                                    # and read happen in one iteration today, so
                                    # a collision could not serve a stale delta --
                                    # but that is a property of the loop, not of
                                    # the key, and moving the read would break it
                                    # silently.
                                    shape_key = (depth, pp, tg, concurrency, effort, thinking)
                                    counters_before = None
                                    # Two-phase mode only: the phases interleave across
                                    # runs, so each needs its own running total rather
                                    # than a single pair spanning the shape.
                                    two_phase = self.config.enable_prefix_caching and depth > 0
                                    ctx_counters = PhaseCounterAccumulator()
                                    inf_counters = PhaseCounterAccumulator()
                                    # Digest of the text this shape actually sent, so
                                    # "did these two runs read the same slice?" is
                                    # answerable from the saved file instead of by
                                    # convention. The convention -- same seed, same key,
                                    # therefore same text -- has been quietly wrong twice.
                                    corpus_digest = hashlib.blake2b(digest_size=8)
                                    # Requests we send inside the counter window,
                                    # so an excess on the server's own tally shows
                                    # another client shared it.
                                    shape_requests = 0
                                    run = -1
                                    while True:
                                        run += 1
                                        if run >= total_runs:
                                            # --target-ci: keep sampling until the decode CI is
                                            # tight enough, or the cap is hit. Without it this
                                            # exits immediately at --runs, as before.
                                            if run >= hard_cap or self._decode_ci_reached(run_std_results):
                                                break
                                        is_warmup = run < warmup_runs
                                        measured_run_index = run - warmup_runs
                                        if not is_warmup and counters_before is None:
                                            counters_before = await self.metrics_probe.snapshot(session)
                                        if is_warmup:
                                            run_label = f"Warmup {run + 1}/{warmup_runs}"
                                        elif self.config.target_ci:
                                            run_label = (
                                                f"Run {measured_run_index + 1} "
                                                f"(adaptive, cap {hard_cap - warmup_runs})"
                                            )
                                        else:
                                            run_label = f"Run {measured_run_index + 1}/{self.config.num_runs}"

                                        # Adapt prompt tokens
                                        current_pp = pp
                                        current_depth = depth
                                        if self.config.adapt_prompt:
                                            if depth == 0:
                                                current_pp = max(1, pp - self.delta_user)
                                            else:
                                                current_depth = max(1, depth - self.delta_context)

                                        expected_pp = current_pp
                                        expected_ctx = current_depth

                                        corpus_key = self.corpus_key(
                                            depth, pp, tg, concurrency,
                                            measured_run_index, effort, thinking,
                                            is_warmup=is_warmup,
                                        )

                                        prompt_batch = self.prompt_gen.generate_batch(
                                            concurrency,
                                            current_pp,
                                            current_depth,
                                            self.config.no_cache,
                                            key=corpus_key,
                                        )
                                        if not is_warmup:
                                            # Warmups are not measured, so they must not
                                            # move the digest -- otherwise a run with a
                                            # different --warmup-runs would look like it
                                            # read different text.
                                            for _ctx, _prompt in prompt_batch:
                                                corpus_digest.update(_ctx.encode())
                                                corpus_digest.update(b"\x00")
                                                corpus_digest.update(_prompt.encode())
                                                corpus_digest.update(b"\x00")

                                        if self.config.enable_prefix_caching and depth > 0:
                                            # Phase 1: Context Load
                                            phase_start = (
                                                await self.metrics_probe.snapshot(session)
                                                if not is_warmup else None
                                            )
                                            # Held open and completed after the
                                            # gather below, so the line reports
                                            # what the run did instead of
                                            # trailing off. A `_warn_once` from
                                            # token counting can land mid-line
                                            # on the first run of a server that
                                            # reports no usage; it is cosmetic
                                            # and fires at most once.
                                            print(f"  {run_label} (Context Load, batch size {concurrency})",
                                                  end="", flush=True)
                                            load_tasks = []
                                            for i in range(concurrency):
                                                context, _ = prompt_batch[i]
                                                if not is_warmup:
                                                    rid = self._new_request_id()
                                                    self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index, effort, thinking, "ctx")
                                                load_tasks.append(self.client.run_generation(
                                                    session,
                                                    context_text=context,
                                                    prompt_text=CONTEXT_LOAD_USER_MESSAGE,
                                                    max_tokens=tg,
                                                    no_cache=self.config.no_cache,
                                                    tokenizer=tokenizer,
                                                    progress=None if is_warmup else self.progress,
                                                    request_id=None if is_warmup else rid,
                                                    reasoning_effort=effort,
                                                    thinking=thinking,
                                                ))

                                            load_results = await asyncio.gather(*load_tasks)
                                            print(_run_summary(load_results), flush=True)
                                            if not is_warmup:
                                                shape_requests += len(load_tasks)
                                                run_ctx_results.append(load_results)

                                            if self.config.exit_on_first_fail and any(r.error for r in load_results):
                                                first_error = next(r.error for r in load_results if r.error)
                                                print(f"\n[Error] Stopping due to error in context load: {first_error}")
                                                raise BenchmarkFailure()

                                            # Between the phases, not inside either: this is
                                            # what makes the follow-up's hit rate its own
                                            # number instead of one diluted by the always-cold
                                            # context load. It costs a scrape per run and a
                                            # slightly longer gap before the follow-up.
                                            phase_mid = (
                                                await self.metrics_probe.snapshot(session)
                                                if not is_warmup else None
                                            )
                                            ctx_counters.add(phase_start, phase_mid)

                                            # Phase 2: Inference
                                            print(f"  {run_label} (Inference, batch size {concurrency})",
                                                  end="", flush=True)
                                            inf_tasks = []
                                            for i in range(concurrency):
                                                context, prompt = prompt_batch[i]
                                                if not is_warmup:
                                                    rid = self._new_request_id()
                                                    self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index, effort, thinking, "inf")
                                                inf_tasks.append(self.client.run_generation(
                                                    session,
                                                    context_text=context,
                                                    prompt_text=prompt,
                                                    max_tokens=tg,
                                                    no_cache=self.config.no_cache,
                                                    tokenizer=tokenizer,
                                                    progress=None if is_warmup else self.progress,
                                                    request_id=None if is_warmup else rid,
                                                    reasoning_effort=effort,
                                                    thinking=thinking,
                                                ))

                                            batch_results = await asyncio.gather(*inf_tasks)
                                            print(_run_summary(batch_results), flush=True)
                                            if not is_warmup:
                                                shape_requests += len(inf_tasks)
                                                run_std_results.append(batch_results)
                                                inf_counters.add(
                                                    phase_mid,
                                                    await self.metrics_probe.snapshot(session),
                                                )

                                            if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                                first_error = next(r.error for r in batch_results if r.error)
                                                print(f"\n[Error] Stopping due to error in inference: {first_error}")
                                                raise BenchmarkFailure()

                                        else:
                                            # Standard Run
                                            print(f"  {run_label} (batch size {concurrency})",
                                                  end="", flush=True)
                                            batch_tasks = []
                                            for i in range(concurrency):
                                                context, prompt = prompt_batch[i]
                                                if not is_warmup:
                                                    rid = self._new_request_id()
                                                    self._emit_request_start(rid, pp, tg, depth, concurrency, measured_run_index, effort, thinking, "standard")
                                                batch_tasks.append(self.client.run_generation(
                                                    session,
                                                    context_text=context,
                                                    prompt_text=prompt,
                                                    max_tokens=tg,
                                                    no_cache=self.config.no_cache,
                                                    tokenizer=tokenizer,
                                                    progress=None if is_warmup else self.progress,
                                                    request_id=None if is_warmup else rid,
                                                    reasoning_effort=effort,
                                                    thinking=thinking,
                                                ))

                                            batch_results = await asyncio.gather(*batch_tasks)
                                            print(_run_summary(batch_results), flush=True)
                                            if not is_warmup:
                                                shape_requests += len(batch_tasks)
                                                run_std_results.append(batch_results)

                                            if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                                first_error = next(r.error for r in batch_results if r.error)
                                                print(f"\n[Error] Stopping due to error in standard run: {first_error}")
                                                raise BenchmarkFailure()


                                        # Post Run Command
                                        if self.config.post_run_cmd:
                                            try:
                                                subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                            except subprocess.CalledProcessError as e:
                                                print(f"Post-run command failed: {e}")

                                    counters_after = await self.metrics_probe.snapshot(session)
                                    delta = self.metrics_probe.diff(counters_before, counters_after)
                                    # One snapshot pair spans the whole shape, so
                                    # under the two-phase mode the delta covers the
                                    # context load and the follow-up together. The
                                    # context load is always cold, so it dilutes the
                                    # rate downward: a follow-up that truly hit 70%
                                    # reports ~39%. The <1% warning stays sound
                                    # (dilution cannot manufacture hits), but the
                                    # number is not the follow-up's own rate, and
                                    # saying so beats implying otherwise.
                                    # Only a diffed counter describes this shape. A
                                    # gauge-only engine reports a server-lifetime
                                    # level, and relabelling that as a shape window
                                    # would save a false provenance and let
                                    # cached_followup_warning claim a window the
                                    # number does not cover -- the same trap
                                    # stale_prefix_cache_warning avoids by skipping
                                    # lifetime scopes outright.
                                    if (delta is not None and self.config.enable_prefix_caching
                                            and depth > 0 and delta.scope.startswith("shape")):
                                        delta = delta.model_copy(update={"scope": "shape (both phases)"})
                                    # Per-phase deltas, when the phases were measured
                                    # separately. Each scope still starts with "shape",
                                    # which is what stale_prefix_cache_warning tests, so
                                    # a lifetime gauge stays excluded while these count.
                                    if two_phase:
                                        for phase, acc, scope in (
                                            ("ctx", ctx_counters, "shape (context load)"),
                                            ("inf", inf_counters, "shape (follow-up)"),
                                        ):
                                            phase_delta = self.metrics_probe.diff(*acc.as_pair())
                                            if phase_delta is None:
                                                continue
                                            if phase_delta.scope.startswith("shape"):
                                                phase_delta = phase_delta.model_copy(
                                                    update={"scope": scope}
                                                )
                                            self.server_deltas[shape_key + (phase,)] = phase_delta

                                    if delta is not None:
                                        self.server_deltas[shape_key] = delta
                                        # Prefer the follow-up's own rate when the phases
                                        # were measured apart: the blended figure is the
                                        # one that misleads, since the cold context load
                                        # sits in its denominator.
                                        rate_delta = self.server_deltas.get(
                                            shape_key + ("inf",), delta
                                        )
                                        if delta.spec_acceptance_length is not None:
                                            print(
                                                f"  server: spec acceptance length "
                                                f"{delta.spec_acceptance_length:.2f}"
                                                + (
                                                    f", prefix cache hit rate "
                                                    f"{rate_delta.prefix_cache_hit_rate:.1%}"
                                            + {"shape": "",
                                               "shape (context load)": " (context load)",
                                               "shape (follow-up)": " (follow-up turn)",
                                               "shape (both phases)": " (ctx load + follow-up)"}.get(
                                                   rate_delta.scope, " (server lifetime)")
                                                    if rate_delta.prefix_cache_hit_rate is not None
                                                    else ""
                                                )
                                            )
                                        warning = cached_followup_warning(
                                            delta, self.config.enable_prefix_caching, depth
                                        )
                                        if warning:
                                            print("  " + warning)
                                        stale = stale_prefix_cache_warning(
                                            delta, self.config.enable_prefix_caching
                                        )
                                        if stale:
                                            print("  " + stale)
                                        # Printed last of the three: it says the other
                                        # two may be describing someone else's traffic.
                                        shared = concurrent_traffic_warning(
                                            delta, shape_requests
                                        )
                                        if shared:
                                            print("  " + shared)

                                    # Aggregate and Record
                                    if self.config.enable_prefix_caching and depth > 0:
                                        self.results.add(self.config.model, pp, tg, depth, concurrency, run_ctx_results, latency, expected_ctx, is_context_phase=True, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries, reasoning_effort=effort, thinking=thinking, server_metrics=self.server_deltas.get(shape_key + ("ctx",), self.server_deltas.get(shape_key)), corpus_fingerprint=corpus_digest.hexdigest())
                                        self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries, reasoning_effort=effort, thinking=thinking, server_metrics=self.server_deltas.get(shape_key + ("inf",), self.server_deltas.get(shape_key)), corpus_fingerprint=corpus_digest.hexdigest())
                                    else:
                                        # Single-phase run: the prompt carries the
                                        # context, so the expected token count is
                                        # the sum of both.
                                        self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp + expected_ctx, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries, reasoning_effort=effort, thinking=thinking, server_metrics=self.server_deltas.get(shape_key), corpus_fingerprint=corpus_digest.hexdigest())

                self.results.metadata = self._metadata(
                    latency,
                    max(self.config.concurrency_levels) if self.config.concurrency_levels else 1,
                )

            self.results.save_report(self.config.save_result, self.config.result_format, max(self.config.concurrency_levels) if self.config.concurrency_levels else 1, self.config.stats_display, self.config.legend)

        except (asyncio.CancelledError, KeyboardInterrupt, BenchmarkFailure) as e:
            if self.results.runs:
                should_save = True
                if isinstance(e, BenchmarkFailure) and self.config.no_results_on_fail:
                    should_save = False
                    print("\n[Failed] Results discarded per --no-results-on-fail.")

                if should_save:
                    print("\n[Interrupted/Failed] Saving partial results...")
                    if self.results.metadata is None:
                        self.results.metadata = self._metadata(latency, max_concurrency)
                    self.results.save_report(self.config.save_result, self.config.result_format, max_concurrency, self.config.stats_display, self.config.legend)
            
            if isinstance(e, BenchmarkFailure):
                sys.exit(1)
            raise
