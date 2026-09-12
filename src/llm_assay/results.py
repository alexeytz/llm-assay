import math
import re
import textwrap
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json
import csv
import sys

from .client import RequestResult
from .servermetrics import ServerMetricsDelta

# Type alias for a time series: List of [timestamp, value] pairs
TimeSeries = List[List[float]]

# Two-sided 95% Student-t critical values by degrees of freedom. At n=3 (df=2)
# this is 4.30, not 1.96 -- using the normal value would understate the interval
# by more than 2x exactly where runs are most underpowered.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040, 32: 2.037,
    33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026, 38: 2.024,
    39: 2.023, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t95_for(df: int) -> float:
    """Two-sided 95% t critical value, falling back to the nearest *smaller* df.

    A plain dict lookup with a 1.96 default is not safe here: the table used to
    skip df 31-39, so n=33 got 1.960 while n=31 got 2.042 and n=41 got 2.021 --
    a CI that shrank as the sample grew, understating the interval exactly in
    the range --target-ci sampling lands in. Rounding down to a tabulated df is
    always the conservative direction, since t decreases as df grows.
    """
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k < df]
    return _T95[max(smaller)] if smaller else _T95[1]

class BenchmarkMetric(BaseModel):
    mean: float = Field(..., description="Mean value")
    std: float = Field(..., description="Sample standard deviation (ddof=1)")
    values: List[float] = Field(..., description="Raw values")
    n: int = Field(0, description="Number of samples")
    sem: float = Field(0.0, description="Standard error of the mean")
    ci95: float = Field(0.0, description="Half-width of the 95% confidence interval of the mean")
    median: float = Field(0.0, description="Middle value; unlike the mean it is not moved by one outlier")
    iqr: float = Field(0.0, description="Inter-quartile range (p75-p25); spread that ignores the tails")

    @property
    def rel_ci95(self) -> float:
        """95% CI half-width as a fraction of the mean (0.0 when mean is 0)."""
        return abs(self.ci95 / self.mean) if self.mean else 0.0

    def runs_needed_for(self, target_rel_ci: float) -> Optional[int]:
        """Sample count needed to reach ``target_rel_ci``, given observed spread.

        n scales as 1/sqrt(n), so n_needed = n * (current_rel / target_rel)^2.
        Returns None if the target is already met or cannot be estimated.
        """
        if self.n < 2 or not self.mean or target_rel_ci <= 0:
            return None
        current = self.rel_ci95
        if current <= target_rel_ci:
            return None
        return int(math.ceil(self.n * (current / target_rel_ci) ** 2))

class BenchmarkMetadata(BaseModel):
    version: str = Field(..., description="Benchmark tool version")
    timestamp: str = Field(..., description="Run timestamp")
    latency_mode: str = Field(..., description="Latency measurement mode used")
    latency_ms: float = Field(..., description="Measured or assumed latency in ms")
    model: str = Field(..., description="Model name")
    prefix_caching_enabled: bool = Field(..., description="Whether prefix caching was enabled")
    max_concurrency: int = Field(..., description="Maximum concurrency level used in the suite")
    seed: Optional[int] = Field(None, description="Corpus sampling seed; equal seeds make two runs pairwise comparable")
    reasoning_efforts: Optional[List[str]] = Field(None, description="Reasoning effort levels swept, if any")
    thinking_modes: Optional[List[str]] = Field(None, description="Thinking modes swept ('on'/'off'), if any")

    # The invocation, not just the code. `version` traces a result back to the
    # commit that produced it; without these it cannot be traced back to the
    # command. --exact-tg is the pointed example: it inflates measured
    # speculative acceptance, and nothing in a saved file used to say whether it
    # was on. Every field here changes the numbers above it.
    #
    # Optional with defaults throughout, so results written by older versions
    # still load. --api-key is deliberately absent and must stay that way.
    base_url: Optional[str] = Field(None, description="Endpoint the run was made against")
    served_model_name: Optional[str] = Field(None, description="Model name sent in API calls")
    tokenizer: Optional[str] = Field(None, description="Tokenizer override, if any")
    # "" rather than None when nothing fell back, for the same reason
    # extra_body uses {}: None is reserved for "written by a version that did
    # not record this", which comparability skips as unknown. Collapsing
    # "nothing fell back" into None would make a run whose depths were
    # counted with gpt2 compare silently against one counted correctly.
    tokenizer_fallback: Optional[str] = Field(
        None,
        description=(
            "Tokenizer name that failed to load, so gpt2 counted the tokens "
            'instead; "" if nothing fell back, null if unrecorded'
        ),
    )
    book_url: Optional[str] = Field(None, description="Corpus the prompts were sliced from")
    endpoint: Optional[str] = Field(None, description="Route benchmarked: chat or completions")
    exact_tg: Optional[bool] = Field(None, description="Output length forced with min_tokens + ignore_eos; inflates measured speculative acceptance")
    no_cache: Optional[bool] = Field(None, description="Requests were made unique to defeat prefix caching")
    cold: Optional[bool] = Field(None, description="Corpus offsets were salted per invocation, so the prefix cache was cold")
    cold_salt: Optional[str] = Field(None, description="The per-invocation salt; two runs with different salts saw different text and cannot be paired")
    adapt_prompt: Optional[bool] = Field(None, description="Prompt size corrected for chat-template overhead")
    skip_coherence: Optional[bool] = Field(None, description="Coherence check was skipped")
    num_runs: Optional[int] = Field(None, description="Measured runs per shape requested")
    warmup_runs: Optional[int] = Field(None, description="Discarded warmup runs per shape")
    target_ci: Optional[float] = Field(None, description="Adaptive sampling target, if used")
    max_runs: Optional[int] = Field(None, description="Cap on adaptive sampling, if used")
    extra_body: Optional[Dict[str, Any]] = Field(None, description="Extra JSON fields merged into each request")

class BenchmarkRun(BaseModel):
    concurrency: int = Field(..., description="Concurrency level for this run")
    context_size: int = Field(..., description="Context size (prefix tokens)")
    prompt_size: int = Field(..., description="Prompt size (tokens)")
    response_size: int = Field(..., description="Response size (tokens)")
    is_context_prefill_phase: bool = Field(..., description="Whether this was a context prefill phase run")
    reasoning_effort: Optional[str] = Field(None, description="reasoning_effort level used for this run, if any")
    thinking: Optional[str] = Field(None, description="Thinking mode used for this run ('on'/'off'), if any")
    server_metrics: Optional["ServerMetricsDelta"] = Field(None, description="Server-side counter movement during this shape")
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Distinct request errors for this shape, with counts; the reason a shape came out short or empty",
    )
    
    # Metrics (using BenchmarkMetric)
    pp_throughput: Optional[BenchmarkMetric] = Field(None, description="Prefill tokens per second (total)")
    pp_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Prefill tokens per second (per request)")
    tg_throughput: Optional[BenchmarkMetric] = Field(None, description="Generation tokens per second (total)")
    tg_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Generation tokens per second (per request)")
    corpus_fingerprint: Optional[str] = Field(
        None,
        description=(
            "Digest of the prompt text this shape actually sent. Two runs that "
            "share it read identical text, so a paired comparison between them is "
            "sound; two that differ did not, whatever their seeds say. Absent on "
            "results saved before this existed, which means unknown, not different."
        ),
    )
    peak_throughput: Optional[BenchmarkMetric] = Field(None, description="Peak generation tokens per second (total)")
    peak_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Peak generation tokens per second (per request)")
    ttfr: Optional[BenchmarkMetric] = Field(None, description="Time to First Response (ms)")
    est_ppt: Optional[BenchmarkMetric] = Field(None, description="Estimated pure processing time (ms)")
    e2e_ttft: Optional[BenchmarkMetric] = Field(None, description="End-to-end Time to First Token (ms)")
    
    # List of time series, one per run (aggregated across all requests in that run)
    throughput_over_time: Optional[List[TimeSeries]] = Field(
        None, 
        description="A collection of time series data capturing the aggregated throughput over time. Each item in the list represents a sequence of [timestamp, value] pairs for a specific execution batch or iteration."
    )
    
    # List of lists of time series, one list per run, containing one time series per request
    requests_throughput_over_time: Optional[List[List[TimeSeries]]] = Field(
        None, 
        description="A collection of time series data for individual requests. Organized as a list of lists, where the outer list represents batches and the inner list contains the throughput time series for each request in that batch."
    )

class BenchmarkReport(BenchmarkMetadata):
    benchmarks: List[BenchmarkRun] = Field(..., description="List of benchmark run results")

def _approx_words(tokens: str) -> str:
    """A parenthetical word estimate for the legend, or "" when it would mislead.

    Non-technical readers have no feel for what 16,384 tokens is, and "roughly
    12,000 words" gives them one. The ratio is a rule of thumb for English
    (~0.75 words per token) and nothing more: a different tokenizer, or any
    other language, moves it a long way. So the figure is rounded hard enough
    that it cannot read as a measurement, and the legend says "roughly".

    Suppressed below 100 tokens, where rounding leaves nothing informative and
    the reader gains no intuition they did not already have.
    """
    try:
        n = int(tokens)
    except ValueError:
        return ""
    if n < 100:
        return ""
    words = n * 0.75
    if words >= 10000:
        rounded = round(words / 1000) * 1000
    elif words >= 1000:
        rounded = round(words / 100) * 100
    else:
        rounded = round(words / 10) * 10
    return f" (roughly {rounded:,.0f} words of English)"


def _re_suffix(run) -> str:
    """Label rows with the swept dimensions, so two rows are never confusable."""
    parts = []
    thinking = getattr(run, "thinking", None)
    if thinking:
        parts.append(f"think={thinking}")
    effort = getattr(run, "reasoning_effort", None)
    if effort:
        parts.append(f"re={effort}")
    return "".join(f" ({p})" for p in parts)


class BenchmarkResults:
    def __init__(self):
        self.runs: List[BenchmarkRun] = []
        self.metadata: Optional[BenchmarkMetadata] = None
        self.model_name: Optional[str] = None
        # Request-level failures. A suite can lose individual requests -- or whole
        # shapes -- and still print a table, so the count has to be kept rather
        # than inferred from what survived.
        self.total_requests: int = 0
        self.failed_requests: int = 0

    def _count_tokens_after_first_timestamp(self, timestamps: List[float]) -> int:
        if len(timestamps) < 2:
            return 0

        first_ts = timestamps[0]
        return sum(1 for ts in timestamps if ts > first_ts)

    def _calculate_metric(self, values: List[float], multiplier: float = 1.0) -> Optional[BenchmarkMetric]:
        if not values:
            return None
        scaled_values = [v * multiplier for v in values]
        n = len(scaled_values)
        mean = float(np.mean(scaled_values))
        # Decode throughput under speculative decoding is content-skewed, so in a
        # 3-6 run sample one slow run drags the mean and inflates the t-interval.
        # The median and IQR cost nothing to compute and say what the typical run
        # did, which is often the number worth acting on.
        q25, med, q75 = (float(x) for x in np.percentile(scaled_values, [25, 50, 75]))
        # ddof=1: the runs are a *sample*, not the population. ddof=0 understates
        # spread badly at small n (at n=3 it divides by 3 instead of 2), which
        # makes an underpowered run look precise.
        std = float(np.std(scaled_values, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        ci95 = _t95_for(n - 1) * sem if n > 1 else 0.0
        return BenchmarkMetric(
            mean=mean,
            std=std,
            values=scaled_values,
            n=n,
            sem=sem,
            ci95=ci95,
            median=med,
            iqr=float(q75 - q25),
        )

    def _calculate_peak_throughput(self, all_timestamps: List[float], window: float = 1.0, return_series: bool = False) -> Any:
        """Highest token count observed in any `window`-second span.

        Returns 0.0 when the tokens carry no observable interval at all -- a
        backend that emits the whole response in one chunk stamps every token
        with the same instant, and "64 tokens at one instant" is a protocol
        artifact, not a throughput. README promises decode throughput is left
        blank in that case; the mean honoured it and this did not, so the row
        showed a blank t/s beside a confident-looking peak.
        """
        if not all_timestamps:
            return (0.0, []) if return_series else 0.0

        # Sort a copy: the caller passes ``res.token_timestamps`` directly and
        # sorting in place would mutate a request's record from inside what
        # reads as a pure calculation. Harmless while the client emits
        # monotonic stamps, which is not a property this function controls.
        all_timestamps = sorted(all_timestamps)
        total_duration = all_timestamps[-1] - all_timestamps[0]
        start_time = all_timestamps[0]

        if total_duration <= 0:
            return (0.0, []) if return_series else 0.0

        # Shorter than the window: rate over the interval actually observed,
        # otherwise dividing by a window the burst never filled understates it
        # to below the mean.
        short_burst = total_duration < window
        adjusted_peak = len(all_timestamps) / total_duration if short_burst else 0.0

        max_tokens = 0
        series: List[List[float]] = []
        start_idx = 0
        for end_idx, end_time in enumerate(all_timestamps):
            while start_idx < end_idx and all_timestamps[start_idx] <= end_time - window:
                start_idx += 1
            current_tokens = end_idx - start_idx + 1
            max_tokens = max(max_tokens, current_tokens)
            if return_series:
                series.append([end_time - start_time, float(current_tokens) / window])

        final_peak = adjusted_peak if short_burst else float(max_tokens) / window

        if return_series:
            # The sliding window divides by a window the burst never filled, so
            # every point would sit below the reported peak and the saved series
            # would contradict the headline figure in the same file. Rescale the
            # series onto the same basis as the number it accompanies.
            if short_burst and series:
                scale = window / total_duration
                series = [[offset, value * scale] for offset, value in series]
            return final_peak, series
        return final_peak

    def add(self, 
            model: str, 
            pp: int, 
            tg: int, 
            depth: int, 
            concurrency: int, 
            run_results: List[List[RequestResult]], # List of batches (one batch per run)
            latency: float, 
            expected_pp_tokens: int,
            is_context_phase: bool = False,
            save_total_throughput_timeseries: bool = False,
            save_all_throughput_timeseries: bool = False,
            reasoning_effort: Optional[str] = None,
            thinking: Optional[str] = None,
            server_metrics: Optional[ServerMetricsDelta] = None,
            corpus_fingerprint: Optional[str] = None):
        
        if self.model_name is None:
            self.model_name = model

        # Keep the messages, not just the count. A run that dies at depth with an
        # OOM otherwise reports "1 shape produced no measurement" and puts the
        # server's reason nowhere in the artifact -- so the one file you still
        # have a week later cannot tell you what went wrong.
        error_counts: Dict[str, int] = {}
        for batch in run_results:
            self.total_requests += len(batch)
            for res in batch:
                message = getattr(res, "error", "") or ""
                if message:
                    self.failed_requests += 1
                    trimmed = message if len(message) <= 500 else message[:497] + "..."
                    error_counts[trimmed] = error_counts.get(trimmed, 0) + 1
        recorded_errors = [
            {"message": m, "count": c}
            for m, c in sorted(error_counts.items(), key=lambda kv: -kv[1])[:5]
        ] or None

        # Aggregators
        agg_pp_speeds: List[float] = []
        agg_tg_speeds: List[float] = []
        agg_ttfr_values: List[float] = []
        agg_est_ppt_values: List[float] = []
        agg_e2e_ttft_values: List[float] = []
        
        agg_batch_pp_throughputs: List[float] = []
        agg_batch_tg_throughputs: List[float] = []
        agg_peak_throughputs: List[float] = []
        agg_peak_req_throughputs: List[float] = []
        
        agg_throughput_series: List[TimeSeries] = []
        agg_req_throughput_series: List[List[TimeSeries]] = []

        for batch in run_results:
            self._process_batch(
                batch, 
                expected_pp_tokens, 
                latency, 
                agg_pp_speeds, 
                agg_tg_speeds, 
                agg_ttfr_values, 
                agg_est_ppt_values, 
                agg_e2e_ttft_values, 
                agg_batch_pp_throughputs, 
                agg_batch_tg_throughputs,
                agg_peak_throughputs,
                agg_peak_req_throughputs,
                save_total_throughput_timeseries=save_total_throughput_timeseries,
                save_all_throughput_timeseries=save_all_throughput_timeseries,
                agg_throughput_series=agg_throughput_series,
                agg_req_throughput_series=agg_req_throughput_series
            )

        # Calculate metrics for BenchmarkRun
        run_metric_pp_throughput = self._calculate_metric(agg_batch_pp_throughputs if concurrency > 1 else agg_pp_speeds)
        run_metric_pp_req_throughput = run_metric_pp_throughput if concurrency == 1 else self._calculate_metric(agg_pp_speeds)
        
        run_metric_tg_throughput = self._calculate_metric(agg_batch_tg_throughputs if concurrency > 1 else agg_tg_speeds)
        run_metric_tg_req_throughput = run_metric_tg_throughput if concurrency == 1 else self._calculate_metric(agg_tg_speeds)

        run_metric_peak_throughput = self._calculate_metric(agg_peak_throughputs)
        run_metric_peak_req_throughput = self._calculate_metric(agg_peak_req_throughputs)

        run_metric_ttfr = self._calculate_metric(agg_ttfr_values, 1000)
        run_metric_est_ppt = self._calculate_metric(agg_est_ppt_values, 1000)
        run_metric_e2e_ttft = self._calculate_metric(agg_e2e_ttft_values, 1000)

        self.runs.append(BenchmarkRun(
            concurrency=concurrency,
            context_size=depth,
            prompt_size=pp, # Configured prompt size
            response_size=tg,
            is_context_prefill_phase=is_context_phase,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            server_metrics=server_metrics,
            corpus_fingerprint=corpus_fingerprint,
            errors=recorded_errors,
            pp_throughput=run_metric_pp_throughput,
            pp_req_throughput=run_metric_pp_req_throughput,
            tg_throughput=run_metric_tg_throughput,
            tg_req_throughput=run_metric_tg_req_throughput,
            peak_throughput=run_metric_peak_throughput,
            peak_req_throughput=run_metric_peak_req_throughput,
            ttfr=run_metric_ttfr,
            est_ppt=run_metric_est_ppt,
            e2e_ttft=run_metric_e2e_ttft,
            throughput_over_time=agg_throughput_series if save_total_throughput_timeseries else None,
            requests_throughput_over_time=agg_req_throughput_series if save_all_throughput_timeseries else None
        ))

    def _process_batch(self, 
                       results: List[RequestResult], 
                       expected_pp_tokens: int, 
                       latency: float,
                       agg_pp_speeds: List[float],
                       agg_tg_speeds: List[float],
                       agg_ttfr_values: List[float],
                       agg_est_ppt_values: List[float],
                       agg_e2e_ttft_values: List[float],
                       agg_batch_pp_throughputs: List[float],
                       agg_batch_tg_throughputs: List[float],
                       agg_peak_throughputs: List[float],
                       agg_peak_req_throughputs: List[float],
                       save_total_throughput_timeseries: bool = False,
                       save_all_throughput_timeseries: bool = False,
                       agg_throughput_series: Optional[List[TimeSeries]] = None,
                       agg_req_throughput_series: Optional[List[List[TimeSeries]]] = None):
        
        valid_results = [r for r in results if r and not r.error]
        if not valid_results:
            return

        batch_prompt_tokens = 0
        batch_gen_tokens = 0
        
        start_times = []
        end_times = []
        first_token_times = []
        last_token_times = []
        
        # Collect all token timestamps for peak calculation
        all_token_timestamps = []
        
        batch_req_series = []

        for res in valid_results:
            start_times.append(res.start_ts)
            end_times.append(res.end_ts)
            all_token_timestamps.extend(res.token_timestamps)
            
            if save_all_throughput_timeseries:
                if res.token_timestamps:
                    # Calculate per-request throughput series
                    peak, series = self._calculate_peak_throughput(res.token_timestamps, return_series=True)
                    batch_req_series.append(series)
                    if peak > 0:
                        agg_peak_req_throughputs.append(peak)
                else:
                    batch_req_series.append([])
            elif res.token_timestamps:
                 peak = self._calculate_peak_throughput(res.token_timestamps, return_series=False)
                 if peak > 0:
                     agg_peak_req_throughputs.append(peak)

            if res.token_timestamps:
                last_token_times.append(res.token_timestamps[-1])
            elif res.end_ts:
                # Fallback if no timestamps recorded but request finished
                last_token_times.append(res.end_ts)
            
            # Use reported usage if available and reasonable, else expected
            prompt_tokens = expected_pp_tokens
            if res.prompt_tokens > 0:
                diff = abs(res.prompt_tokens - expected_pp_tokens)
                if diff < expected_pp_tokens * 0.2:
                    prompt_tokens = res.prompt_tokens
            
            batch_prompt_tokens += prompt_tokens
            batch_gen_tokens += res.total_tokens

            # Metrics Calculation
            e2e_ttft = 0.0
            ttfr = 0.0
            est_ppt = 0.0
            
            if res.first_response_ts:
                ttfr = res.first_response_ts - res.start_ts
                agg_ttfr_values.append(ttfr)
            
            if res.first_token_ts:
                first_token_times.append(res.first_token_ts)
                e2e_ttft = res.first_token_ts - res.start_ts
                est_ppt = max(0, ttfr - latency)

                agg_e2e_ttft_values.append(e2e_ttft)
                agg_est_ppt_values.append(est_ppt)

            # Individual Speeds
            if est_ppt > 0:
                pp_speed = prompt_tokens / est_ppt
                agg_pp_speeds.append(pp_speed)
            
            if res.total_tokens > 1 and len(res.token_timestamps) > 1:
                decode_time = res.token_timestamps[-1] - res.token_timestamps[0]
                decode_tokens = self._count_tokens_after_first_timestamp(res.token_timestamps)
                if decode_time > 0 and decode_tokens > 0:
                    tg_speed = decode_tokens / decode_time
                    agg_tg_speeds.append(tg_speed)
        
        if save_all_throughput_timeseries and agg_req_throughput_series is not None:
             agg_req_throughput_series.append(batch_req_series)

        # Batch-Level Throughput
        if start_times and end_times and first_token_times:
            min_start = min(start_times)
            max_end = max(end_times)
            
            # Wall clock from the first request starting to the last one having
            # its prompt ingested. That is deliberately the whole span: this is
            # the *aggregate* figure, so a straggler lengthening it is the system
            # genuinely taking longer to ingest the batch, not an artifact. The
            # per-request view is reported separately as t/s (req).
            max_first_token = max(first_token_times)
            pp_duration = max_first_token - min_start
            
            if pp_duration > 0:
                batch_pp_throughput = batch_prompt_tokens / pp_duration
                agg_batch_pp_throughputs.append(batch_pp_throughput)
            
            min_first_token = min(first_token_times)
            
            # Use max(last_token_times) instead of max(end_times) to remove protocol overhead (headers, [DONE], etc)
            # This makes the throughput metric purely about token generation speed.
            max_last_token = max(last_token_times) if last_token_times else max_end
            tg_duration = max_last_token - min_first_token
            
            if tg_duration > 0:
                observed_decode_tokens = sum(
                    self._count_tokens_after_first_timestamp(r.token_timestamps)
                    for r in valid_results
                )
                if observed_decode_tokens > 0:
                    batch_tg_throughput = observed_decode_tokens / tg_duration
                    agg_batch_tg_throughputs.append(batch_tg_throughput)

        if all_token_timestamps:
            res = self._calculate_peak_throughput(all_token_timestamps, return_series=save_total_throughput_timeseries)
            if save_total_throughput_timeseries:
                peak, series = res
                # 0 means the tokens carried no observable interval; appending it
                # would publish a zero that reads as a measurement. The series is
                # dropped with it: throughput_over_time claims one entry per run,
                # so keeping an empty series for a run that contributed no peak
                # shifts every later series against peak_throughput.values, and a
                # consumer pairing them by index labels one run with another's
                # number.
                if peak > 0:
                    agg_peak_throughputs.append(peak)
                    if agg_throughput_series is not None:
                        agg_throughput_series.append(series)
            elif res > 0:
                agg_peak_throughputs.append(res)


    def empty_shapes(self) -> List[str]:
        """Shapes that were attempted but produced no measurement at all.

        Distinct from a shape that merely lost a run: that one still yields a
        number, just from fewer samples. A shape with nothing in it is a hole in
        the data, and a table printed around the hole looks complete.
        """
        return [
            self._shape_label(run)
            for run in self.runs
            if not (run.pp_throughput or run.tg_throughput or run.peak_throughput)
        ]

    @staticmethod
    def _shape_label(run: "BenchmarkRun") -> str:
        label = f"pp{run.prompt_size} tg{run.response_size} @ d{run.context_size}"
        # Under --measure-cached-followup a shape contributes two runs, the
        # context load and the follow-up. Without the phase they get identical
        # labels, so a failure summary naming both cannot say which one fell out.
        if run.is_context_prefill_phase:
            label += " ctx"
        if run.concurrency > 1:
            label += f" (c{run.concurrency})"
        # Same rule as compare.py's shape_key: every swept dimension, or two
        # failed shapes get labels you cannot tell apart.
        if getattr(run, "thinking", None):
            label += f" (think={run.thinking})"
        if run.reasoning_effort:
            label += f" (re={run.reasoning_effort})"
        return label

    def failure_summary(self) -> Optional[str]:
        """One line describing what did not survive, or None if all was well."""
        if not self.failed_requests and not self.empty_shapes():
            return None
        parts = []
        if self.failed_requests:
            parts.append(f"{self.failed_requests} of {self.total_requests} requests failed")
        empty = self.empty_shapes()
        if empty:
            shown = ", ".join(empty[:3]) + (" ..." if len(empty) > 3 else "")
            parts.append(f"{len(empty)} shape(s) produced no measurement: {shown}")
        return "Failures: " + "; ".join(parts) + "."

    def has_measurements(self) -> bool:
        """Whether anything was actually measured.

        Not the same as ``self.runs`` being non-empty: a shape whose every
        request failed still appends a run, it just has no metrics on it. This
        is the predicate the report uses to decide it has nothing to print, and
        the one the exit code has to agree with.
        """
        return bool(self._generate_rows())

    def _generate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for run in self.runs:
            c_suffix = ""
            if self.metadata and self.metadata.max_concurrency > 1:
                c_suffix = f" (c{run.concurrency})"

            if run.is_context_prefill_phase:
                # Context Phase Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_pp @ d{run.context_size}{c_suffix}{_re_suffix(run)}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "peak_ts": None,
                        "peak_ts_req": None,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Context Phase Token Generation
                if run.tg_throughput or run.peak_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_tg @ d{run.context_size}{c_suffix}{_re_suffix(run)}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "peak_ts": run.peak_throughput,
                        "peak_ts_req": run.peak_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
            else:
                # Standard Phase
                d_suffix = f" @ d{run.context_size}" if run.context_size > 0 else ""
                
                # Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"pp{run.prompt_size}{d_suffix}{c_suffix}{_re_suffix(run)}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "peak_ts": None,
                        "peak_ts_req": None,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Token Generation
                if run.tg_throughput or run.peak_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"tg{run.response_size}{d_suffix}{c_suffix}{_re_suffix(run)}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "peak_ts": run.peak_throughput,
                        "peak_ts_req": run.peak_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
        return rows

    def reliability_notes(self, target_rel_ci: float = 0.05) -> List[str]:
        """Flag decode metrics whose CI is too wide to support a comparison.

        A small +/- at n=3 reads as precision but usually is not: it is a
        small-sample artifact. This says so explicitly and gives the run count
        that would actually resolve the metric.
        """
        notes: List[str] = []
        for run in self.runs:
            metric = run.tg_throughput
            if metric is None or metric.n < 2:
                continue
            if metric.rel_ci95 <= target_rel_ci:
                continue
            needed = metric.runs_needed_for(target_rel_ci)
            # Same dimensions as _shape_label, or two notes under a concurrency
            # or thinking sweep are indistinguishable.
            label = self._shape_label(run)
            note = (
                f"  {label}: decode {metric.mean:.2f} t/s, 95% CI +/-{metric.rel_ci95 * 100:.1f}% "
                f"at n={metric.n}"
            )
            if needed:
                note += f" -- need ~{needed} runs for +/-{target_rel_ci * 100:.0f}%"
            notes.append(note)
        return notes

    def _legend(self, concurrency: int) -> str:
        """Explain the table that was actually printed, and only that.

        Derived from the rendered rows rather than written out as a fixed
        block, for the same reason `_shape_label` carries every swept
        dimension: a legend that describes `t/s (req)` under a `c1` table, or
        `ctx_pp` for a run that was not two-phase, teaches the reader about a
        column that is not there -- and one that silently omits a label the
        table does contain is worse, because the reader concludes the tool has
        nothing to say about it.

        The row entries are matched against `_generate_rows`' own output, so a
        new swept dimension shows up here as an unexplained suffix rather than
        as a legend that quietly contradicts the table. It is deliberately not
        a sixth surface to remember: nothing here has to be updated for a label
        this method does not recognise.

        Every entry leads with a plain sentence and puts the mechanics
        underneath it. The two audiences are genuinely different: someone
        reading a benchmark for the first time needs to know that `pp512` is a
        reading speed before any formula helps, and someone who already knows
        that needs the formula rather than the sentence. Dropping either one
        loses a reader the other keeps.
        """
        rows = self._generate_rows()
        if not rows:
            return ""
        names = [r["test_name"] for r in rows]
        joined = " ".join(names)

        def any_row(pattern: str) -> bool:
            return bool(re.search(pattern, joined))

        # (term, plain sentence, mechanics). The mechanics may be empty.
        entries: List[Tuple[str, str, str]] = []

        # --- what a row is -----------------------------------------------
        two_phase = any_row(r"\bctx_pp\b")

        def row_term(prefix: str) -> Optional[Tuple[str, str]]:
            """Name the row the way the table does, or generically if swept.

            With one size the reader gets the literal label back; with several
            a single example would be a legend that explains one row and reads
            as though it explained all of them.
            """
            seen = sorted({int(m) for m in re.findall(rf"\b{prefix}(\d+)", joined)})
            if not seen:
                return None
            if len(seen) == 1:
                return f"{prefix}{seen[0]}", str(seen[0])
            return f"{prefix}N", "N"

        pp_term, tg_term = row_term("pp"), row_term("tg")
        depths = sorted({int(m) for m in re.findall(r"@ d(\d+)", joined)})

        if pp_term:
            plain = (
                f"How fast the model reads the prompt. It was handed a "
                f"{pp_term[1]}-token prompt{_approx_words(pp_term[1])} and this "
                f"is the rate it got through it. Higher is faster."
            )
            detail = (
                "t/s on this row is prompt tokens / est_ppt -- the server-side "
                "ingest rate, with the network round trip removed."
            )
            if two_phase:
                detail += (
                    " Under --measure-cached-followup this is phase 2, the "
                    "follow-up turn over the context loaded above -- which only "
                    "means 'follow-up' if the server actually reused it. With "
                    "server-side prefix caching off the identical measurement is "
                    "a full re-prefill, and looks plausible either way."
                )
            entries.append((pp_term[0], plain, detail))
        if tg_term:
            entries.append((
                tg_term[0],
                f"How fast the model writes its answer, once it has started. "
                f"It produced "
                f"{tg_term[1]} tokens{_approx_words(tg_term[1])} on this row. "
                f"Higher is faster.",
                "t/s is tokens after the first / (last token - first token), so "
                "it is the typing speed and not the wait before typing began.",
            ))
        if depths:
            d = str(depths[0]) if len(depths) == 1 else "D"
            amount = (f"{depths[0]:,} tokens{_approx_words(str(depths[0]))}"
                      if len(depths) == 1 else "D tokens")
            entries.append((
                f"@ d{d}",
                f"How much text the model had to take in before the prompt: "
                f"{amount}. Think of it as how far into a long document the "
                f"question was asked -- the further in, the more there is to "
                f"read first.",
                "Prefill cost grows with depth, so rows are only comparable at "
                "equal d.",
            ))
        if two_phase:
            entries.append((
                "ctx_pp / ctx_tg",
                "The cost of loading the background text on its own, before any "
                "real question was asked.",
                "Phase 1 of --measure-cached-followup: the context sent alone as "
                "a system message. Always uncached, so it is the price of the "
                "context rather than of a turn over it.",
            ))
        if any_row(r"\(c\d+\)"):
            entries.append((
                "(cN)",
                "N requests were in flight at once -- what a server sees when N "
                "people use it simultaneously, rather than one at a time.",
                "",
            ))
        if any_row(r"\(think="):
            entries.append((
                "(think=on|off)",
                "Whether the model was allowed to think to itself before "
                "answering.",
                "It is in the corpus key, so the two arms read different text by "
                "design -- otherwise the second arm's prefill would be served "
                "from the cache the first one filled.",
            ))
        if any_row(r"\(re="):
            entries.append((
                "(re=...)",
                "How hard the model was told to think.",
                "Check the endpoint honours it: `tune <url>` renders the prompt "
                "under each level and reports which switches are actually live. "
                "Many deployments accept the setting and ignore it.",
            ))

        # --- what a column is ---------------------------------------------
        if concurrency > 1:
            entries.append((
                "t/s (total)",
                "Tokens per second across everything running at once -- the "
                "server's total output. Higher is better.",
                "For prefill this is sum of prompt tokens / wall-clock, with no "
                "latency subtraction, so it is not on the same basis as the c1 "
                "figure. Compare c>1 against c>1, or use t/s (req).",
            ))
            entries.append((
                "t/s (req)",
                "Tokens per second as one user experiences it, averaged over the "
                "requests. This is the number a person waiting would notice.",
                "The per-request figure, latency-subtracted as at c1.",
            ))
        else:
            entries.append((
                "t/s",
                "Tokens per second -- the speed. A token is roughly three "
                "quarters of an English word. Higher is better.",
                "Prefill speed on a pp row, decode speed on a tg row: two "
                "different formulas under one heading.",
            ))

        tg_rows = [r for r in rows if re.search(r"\btg", r["test_name"])]
        if tg_rows and all(r["peak_ts"] is None for r in tg_rows):
            entries.append((
                "peak t/s (blank)",
                "Empty on purpose, not missing.",
                "This backend stamped every token with one instant, so the "
                "tokens carry no interval and a peak over them would be a "
                "protocol artifact rather than a rate.",
            ))
        elif tg_rows:
            entries.append((
                "peak t/s",
                "The fastest one-second burst, rather than the average. The gap "
                "between this and t/s is how uneven the output was.",
                "Blank on prefill rows, which have no such window.",
            ))
            if concurrency > 1:
                entries.append((
                    "peak t/s (req)",
                    "The same burst figure, but per request.",
                    "Each request's own peak, averaged. The total is an "
                    "aggregate peak, so it is not a sum of these.",
                ))

        mode = self.metadata.latency_mode if self.metadata else None
        entries.append((
            "ttfr (ms)",
            "How long the user waited, in milliseconds, before anything at "
            "all came back. Lower is better. (1000 ms = 1 second.)",
            "Time to the first stream chunk of any kind, including a role-only "
            "one, so it includes network latency. This is what `vllm bench "
            "serve` calls TTFT.",
        ))
        subtracted = (f"the latency measured by --latency-mode {mode}"
                      if mode else "measured latency")
        entries.append((
            "est_ppt (ms)",
            "Of that wait, how much the server spent actually reading the "
            "prompt -- the network trip taken out. Lower is better.",
            f"ttfr minus {subtracted}. It is the denominator of every prefill "
            f"t/s above, so a bad latency estimate moves those too.",
        ))
        entries.append((
            "e2e_ttft (ms)",
            "How long before the first real word of the answer appeared. This "
            "is the wait a person actually feels. Lower is better.",
            "Time to the first token carrying content; ttfr can beat it by a "
            "whole chunk when the server sends an empty one first.",
        ))
        entries.append((
            "blank cell",
            "An empty cell means the number does not apply to that row -- not "
            "that it was zero.",
            "Prefill rows have no decode window; decode rows have no prompt to "
            "time.",
        ))

        width = max(len(term) for term, _, _ in entries)
        out = ["", "Legend:"]
        body = 78 - width - 4
        for term, plain, detail in entries:
            wrapped = textwrap.wrap(plain, width=body)
            out.append(f"  {term.rjust(width)}  {wrapped[0]}")
            out.extend(" " * (width + 4) + line for line in wrapped[1:])
            # The mechanics sit one step further in, so the plain sentence can
            # be read on its own and the rest skipped.
            for line in textwrap.wrap(detail, width=body - 2):
                out.append(" " * (width + 6) + line)
        return "\n".join(out)

    def _generate_md_report(self, concurrency: int, stats_display: str = "std") -> str:
        rows = self._generate_rows()
        if not rows:
            return "No results collected. Check if the model is generating tokens."

        use_ci = stats_display == "ci"
        use_median = stats_display == "median"

        def fmt(metric: Optional[BenchmarkMetric]) -> str:
            if metric is None:
                return ""
            if use_median:
                # Report what the typical run did, and how far the middle half
                # of the runs spread -- neither is moved by a single outlier.
                return f"{metric.median:.2f} ± {metric.iqr / 2:.2f}"
            spread = metric.ci95 if use_ci else metric.std
            return f"{metric.mean:.2f} ± {spread:.2f}"
            
        data = [[
            row["model"], 
            row["test_name"], 
            fmt(row["t_s"]), 
            fmt(row["t_s_req"]), 
            fmt(row["peak_ts"]),
            fmt(row["peak_ts_req"]),
            fmt(row["ttfr"]), 
            fmt(row["est_ppt"]), 
            fmt(row["e2e_ttft"])
        ] for row in rows]

        ts_header = "t/s (total)" if concurrency > 1 else "t/s"
        headers = ["model", "test", ts_header, "t/s (req)", "peak t/s", "peak t/s (req)", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
        
        if concurrency == 1:
            data = [[
                row["model"], 
                row["test_name"], 
                fmt(row["t_s"]),
                fmt(row["peak_ts"]),
                fmt(row["ttfr"]), 
                fmt(row["est_ppt"]), 
                fmt(row["e2e_ttft"])
            ] for row in rows]
            headers = ["model", "test", ts_header, "peak t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]

        return tabulate(data, headers=headers, tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right", "right", "right", "right") if concurrency > 1 else ("left", "right", "right", "right", "right", "right", "right"))

    def save_report(self, filename: Optional[str], format: str, concurrency: int = 1,
                    stats_display: str = "std", legend: bool = False):
        msg = ""
        if filename:
            msg += f"Saving results to {filename} in {format.upper()} format...\n"
        else:            
            msg += f"Printing results in {format.upper()} format:\n"

        print(f"{msg}\n")

        # The human-readable table is always printed, whatever --format saves to
        # disk. Previously `--format json --save-result f` produced no console
        # output at all, which is unhelpful for a run that takes minutes.
        table = self._generate_md_report(concurrency, stats_display)
        legend_text = self._legend(concurrency) if legend else ""
        if format != "md" or filename:
            print("\n" + table)
            if legend_text:
                print(legend_text)
            if stats_display == "std":
                print("\n(± is the sample standard deviation; pass --stats ci for the 95% "
                      "confidence interval of the mean, which is what tells you whether a "
                      "difference between two runs is real, or --stats median for a "
                      "median ± IQR/2 summary that one slow run cannot drag around.)")

        notes = self.reliability_notes()
        if notes:
            print("\nStatistical reliability warnings:")
            for note in notes:
                print(note)

        if format == "md":
            output = table
            if legend_text:
                # Fenced only when saving. Markdown reflows consecutive lines
                # into one paragraph, and the legend's two-column layout is
                # load-bearing: unfenced, the term runs into its explanation
                # and the indent separating plain sentence from mechanics is
                # lost. On a terminal there is no renderer to defend against
                # and the backticks would just be noise.
                output += ("\n\n```\n" + legend_text.strip("\n") + "\n```\n"
                           if filename else legend_text)
            # Only when saving. A saved table with no provenance cannot be traced
            # back to the run that produced it, which is the whole point of
            # recording the invocation in the JSON -- but the terminal already
            # prints a run footer, and emitting both put two nearly identical
            # footers back to back on every default run.
            if filename and self.metadata:
                meta = self.metadata
                bits = [f"llm-assay {meta.version}", f"date: {meta.timestamp}",
                        f"model: {meta.model}", f"latency mode: {meta.latency_mode}"]
                if meta.seed is not None:
                    bits.append(f"seed: {meta.seed}")
                if meta.exact_tg:
                    bits.append("--exact-tg")
                if meta.no_cache:
                    bits.append("--no-cache")
                if meta.prefix_caching_enabled:
                    bits.append("--measure-cached-followup")
                # Appended to what is already there. Reassigning `table`
                # here silently dropped the legend from every saved file.
                output += "\n\n" + " | ".join(bits) + "\n"
            if filename:
                with open(filename, "w") as f:
                    f.write(output)
            else:
                 print("\n" + output)
        
        elif format == "json":
            output_data = {}
            # Flatten metadata if present
            if self.metadata:
                output_data.update(self.metadata.model_dump())
            
            # Serialize runs
            output_data["benchmarks"] = [run.model_dump() for run in self.runs]
            
            json_str = json.dumps(output_data, indent=2)
            
            if filename:
                 with open(filename, "w") as f:
                     f.write(json_str)
            else:
                 print(json_str)
        
        elif format == "csv":
             rows = self._generate_rows()
             csv_rows = []
             # n and ci95 travel with every metric: the README's whole argument is
             # that a mean without its sample count and interval is not actionable,
             # so the machine-readable export must not drop them.
             headers = ["model", "test_name", "t_s_mean", "t_s_std", "t_s_n", "t_s_ci95", "t_s_req_mean", "t_s_req_std", "t_s_req_n", "t_s_req_ci95", "peak_ts_mean", "peak_ts_std", "peak_ts_n", "peak_ts_ci95", "peak_ts_req_mean", "peak_ts_req_std", "peak_ts_req_n", "peak_ts_req_ci95", "ttfr_mean", "ttfr_std", "ttfr_n", "ttfr_ci95", "est_ppt_mean", "est_ppt_std", "est_ppt_n", "est_ppt_ci95", "e2e_ttft_mean", "e2e_ttft_std", "e2e_ttft_n", "e2e_ttft_ci95"]
             
             for r in rows:
                 row = {
                     "model": r["model"],
                     "test_name": r["test_name"],
                     "t_s_mean": r["t_s"].mean if r["t_s"] else None,
                     "t_s_std": r["t_s"].std if r["t_s"] else None,
                     "t_s_n": r["t_s"].n if r["t_s"] else None,
                     "t_s_ci95": r["t_s"].ci95 if r["t_s"] else None,
                     "t_s_req_mean": r["t_s_req"].mean if r["t_s_req"] else None,
                     "t_s_req_std": r["t_s_req"].std if r["t_s_req"] else None,
                     "t_s_req_n": r["t_s_req"].n if r["t_s_req"] else None,
                     "t_s_req_ci95": r["t_s_req"].ci95 if r["t_s_req"] else None,
                     "peak_ts_mean": r["peak_ts"].mean if r["peak_ts"] else None,
                     "peak_ts_std": r["peak_ts"].std if r["peak_ts"] else None,
                     "peak_ts_n": r["peak_ts"].n if r["peak_ts"] else None,
                     "peak_ts_ci95": r["peak_ts"].ci95 if r["peak_ts"] else None,
                     "peak_ts_req_mean": r["peak_ts_req"].mean if r["peak_ts_req"] else None,
                     "peak_ts_req_std": r["peak_ts_req"].std if r["peak_ts_req"] else None,
                     "peak_ts_req_n": r["peak_ts_req"].n if r["peak_ts_req"] else None,
                     "peak_ts_req_ci95": r["peak_ts_req"].ci95 if r["peak_ts_req"] else None,
                     "ttfr_mean": r["ttfr"].mean if r["ttfr"] else None,
                     "ttfr_std": r["ttfr"].std if r["ttfr"] else None,
                     "ttfr_n": r["ttfr"].n if r["ttfr"] else None,
                     "ttfr_ci95": r["ttfr"].ci95 if r["ttfr"] else None,
                     "est_ppt_mean": r["est_ppt"].mean if r["est_ppt"] else None,
                     "est_ppt_std": r["est_ppt"].std if r["est_ppt"] else None,
                     "est_ppt_n": r["est_ppt"].n if r["est_ppt"] else None,
                     "est_ppt_ci95": r["est_ppt"].ci95 if r["est_ppt"] else None,
                     "e2e_ttft_mean": r["e2e_ttft"].mean if r["e2e_ttft"] else None,
                     "e2e_ttft_std": r["e2e_ttft"].std if r["e2e_ttft"] else None,
                     "e2e_ttft_n": r["e2e_ttft"].n if r["e2e_ttft"] else None,
                     "e2e_ttft_ci95": r["e2e_ttft"].ci95 if r["e2e_ttft"] else None,
                 }
                 csv_rows.append(row)
             
             if filename:
                 with open(filename, "w", newline="") as f:
                      writer = csv.DictWriter(f, fieldnames=headers)
                      writer.writeheader()
                      writer.writerows(csv_rows)
             else:
                 writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                 writer.writeheader()
                 writer.writerows(csv_rows)
