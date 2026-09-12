from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import re
import requests
import sys
from . import __version__


def _split_outside_json(text: str) -> List[str]:
    """Split on commas that are not inside a JSON object, array or string.

    A plain str.split(",") mangles a nested value: `chat_template_kwargs=
    {"enable_thinking":false,"x":1}` became two garbage keys with no error at
    all, which is how a request body ends up silently wrong.
    """
    parts: List[str] = []
    buf: List[str] = []
    depth, in_str, escaped = 0, False, False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf)); buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


class BenchmarkConfig(BaseModel):
    base_url: str = Field(..., description="OpenAI compatible endpoint URL")
    api_key: str = Field(..., description="API Key for the endpoint")
    model: str = Field(..., description="Model name to use for benchmarking")
    served_model_name: str = Field(
        ...,
        description="Model name used in API calls (defaults to --model if not specified)",
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Tokenizer to use (HF model name or local path; defaults to model name)",
    )
    pp_counts: List[int] = Field(
        ..., description="List of prompt processing token counts"
    )
    tg_counts: List[int] = Field(..., description="List of token generation counts")
    endpoint: str = Field(
        "chat",
        description="Which route to benchmark: 'chat' (/v1/chat/completions) or 'completions' (/v1/completions)",
    )
    exact_tg: bool = Field(
        False,
        description="Force generated output length to match --tg using server-specific min_tokens and ignore_eos fields",
    )
    depths: List[int] = Field(
        ..., description="List of context depths (previous conversation tokens)"
    )
    num_runs: int = Field(..., description="Number of runs per test")
    warmup_runs: int = Field(
        1,
        description="Number of discarded warmup runs per test shape; also used for generation latency probes",
    )
    cold: bool = Field(
        False,
        description="Draw different corpus text each invocation so the server's prefix cache cannot be warm",
    )
    no_cache: bool = Field(
        ..., description="Ensure unique requests to avoid prefix caching"
    )
    latency_mode: str = Field(
        ..., description="Method to measure latency: 'api', 'generation', or 'none'"
    )
    no_warmup: bool = Field(..., description="Skip warmup phase")
    skip_coherence: bool = Field(..., description="Skip coherence test after warmup")
    stall_timeout: float = Field(
        300.0,
        description="Seconds to wait for the next chunk before treating a request as stalled (0 disables)",
    )
    coherence_prompt: Optional[str] = Field(
        None, description="Question asked by the coherence check (default: an English capital-city question)"
    )
    coherence_expect: Optional[List[str]] = Field(
        None,
        description="Accepted substrings for the coherence answer; empty means accept any non-empty content",
    )
    adapt_prompt: bool = Field(
        ..., description="Adapt prompt size based on warmup token usage delta"
    )
    enable_prefix_caching: bool = Field(
        ...,
        description="Two-phase measurement: load context, then time a follow-up turn over it",
    )
    seed: Optional[int] = Field(
        None,
        description="Seed corpus sampling so runs are reproducible and A/B comparable",
    )
    target_ci: Optional[float] = Field(
        None,
        description="Keep sampling a test shape until the decode 95% CI is within this fraction of the mean",
    )
    max_runs: Optional[int] = Field(
        None, description="Upper bound on runs per shape when --target-ci is set"
    )
    stats_display: str = Field(
        "std", description="Show '+/- std' or '+/- ci' in the table"
    )
    legend: bool = Field(
        False, description="Print an explanation of the table's rows and columns beneath it"
    )
    server_metrics: bool = Field(
        True, description="Scrape the server's /metrics endpoint for cache and spec-decode counters"
    )
    thinking_modes: Optional[List[str]] = Field(
        None,
        description="Thinking modes to sweep ('on'/'off'); sets enable_thinking and reasoning_effort=none",
    )
    reasoning_efforts: Optional[List[str]] = Field(
        None, description="Reasoning effort levels to sweep; each runs the full suite"
    )
    book_url: str = Field(..., description="URL of a book to use for text generation")
    post_run_cmd: Optional[str] = Field(
        None, description="Command to execute after each test run"
    )
    concurrency_levels: List[int] = Field(..., description="List of concurrency levels")
    save_result: Optional[str] = Field(None, description="File to save results to")
    result_format: str = Field("md", description="Output format (md, json, csv)")
    save_total_throughput_timeseries: bool = Field(
        False,
        description="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.",
    )
    save_all_throughput_timeseries: bool = Field(
        False,
        description="Save calculated throughput timeseries for EACH individual request.",
    )
    exit_on_first_fail: bool = Field(
        False,
        description="Stop execution on first failed test and exit with non-zero status",
    )
    no_results_on_fail: bool = Field(
        False,
        description="Prevent saving/printing results when error is experienced, turns on --exit-on-first-fail as well",
    )
    extra_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra JSON fields to merge into benchmark chat completion requests",
    )
    emit_progress: Optional[str] = Field(
        None,
        description="Emit progress events as JSONL to PATH (or '-' for stdout). See docs/progress-schema.md.",
    )

    @staticmethod
    def _parse_extra_body(values: Optional[List[str]]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if not values:
            return extra

        for item in values:
            entries = [entry.strip() for entry in _split_outside_json(item) if entry.strip()]
            for entry in entries:
                if "=" in entry:
                    key, raw_value = entry.split("=", 1)
                elif ":" in entry:
                    key, raw_value = entry.split(":", 1)
                else:
                    raise ValueError(
                        f"Invalid --extra-body entry '{entry}'. Use key=value or key:value."
                    )

                key = key.strip()
                raw_value = raw_value.strip()
                if not key:
                    raise ValueError(f"Invalid --extra-body entry '{entry}': empty key.")

                try:
                    extra[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    extra[key] = raw_value

        return extra

    @staticmethod
    def _detect_hf_model_from_endpoint(base_url: str, api_key: str) -> Tuple[str, str]:
        """
        Fetch models from {base_url}/models endpoint and identify HF model name.

        Returns:
            tuple of (hf_model_name, served_model_name)
        """
        HF_MODEL_PATTERN = re.compile(r"^[^/]+/[^/]+$")
        # Runs off argv, before the base_url field validator has seen it.
        base_url = base_url.rstrip("/")

        try:
            headers = (
                {"Authorization": f"Bearer {api_key}"}
                if api_key and api_key != "EMPTY"
                else {}
            )
            response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ValueError(
                f"Unable to connect to {base_url}/models endpoint: {e}\n"
                "Please specify --model explicitly."
            )

        # Collect all available models and separate by HF format
        hf_formatted = []  # (hf_name, served_name)
        non_hf_formatted = []  # model names without HF format

        # Parse response based on server type
        # Parse data array first
        if "data" in data:
            for model in data["data"]:
                model_id = model.get("id", "")
                root = model.get("root", "")

                if root and HF_MODEL_PATTERN.match(root):
                    hf_formatted.append((root, model_id))
                elif HF_MODEL_PATTERN.match(model_id):
                    hf_formatted.append((model_id, model_id))
                else:
                    non_hf_formatted.append(model_id)

        # parse models array as a fallback
        # Only process if "data" is not present to avoid duplicates
        elif "models" in data:
            for model in data["models"]:
                model_name = model.get("model", "") or model.get("id", "")
                if HF_MODEL_PATTERN.match(model_name):
                    hf_formatted.append((model_name, model_name))
                else:
                    non_hf_formatted.append(model_name)

        # Guard: Multiple models available - cannot determine which one to use
        if len(hf_formatted) + len(non_hf_formatted) > 1:
            error_msg = "Multiple models available at the endpoint. Please specify --model explicitly.\n\n"

            if hf_formatted:
                error_msg += "Models with HF format:\n"
                error_msg += "\n".join(f"  - {m[0]}" for m in hf_formatted) + "\n"

            if non_hf_formatted:
                error_msg += "\nModels without HF format:\n"
                error_msg += "\n".join(f"  - {m}" for m in non_hf_formatted) + "\n"

            error_msg += "\nPlease specify --model explicitly with the model name you want to test."
            raise ValueError(error_msg)

        # No models found
        if not hf_formatted and not non_hf_formatted:
            raise ValueError(
                "No models found at the endpoint.\nPlease specify --model explicitly."
            )

        # Single non-HF model found. This is an ordinary way to serve a model --
        # vLLM's --served-model-name takes any string -- so refusing to proceed
        # was stricter than necessary. The HF name is only used to pick a
        # tokenizer, and --tokenizer overrides that anyway. `tune` already copes
        # with exactly this shape of endpoint; the main path now matches it.
        if not hf_formatted and non_hf_formatted:
            served = non_hf_formatted[0]
            print(
                f"Model '{served}' is not in HF format (namespace/model), so the "
                "tokenizer cannot be\n"
                "inferred from it and a fallback will be used. Pass --tokenizer "
                "(an HF name, a local\n"
                "directory, or a tokenizer.json) if prompt sizes need to be exact."
            )
            return (served, served)

        # Single HF-formatted model found - validate against HF Hub
        hf_name, served_name = hf_formatted[0]
        try:
            hf_response = requests.get(
                f"https://huggingface.co/api/models/{hf_name}", timeout=3
            )
            if hf_response.status_code in (200, 401):
                return (hf_name, served_name)
        except requests.RequestException:
            pass

        # The Hub lookup is advisory. A local or private checkpoint named
        # org/model is perfectly legitimate and simply is not on huggingface.co,
        # and an offline host cannot answer at all -- neither is a reason to
        # refuse to benchmark.
        print(
            f"Model '{hf_name}' was not found on huggingface.co (private, local, or "
            "no network).\n"
            "Proceeding with it; pass --tokenizer if the tokenizer cannot be resolved."
        )
        return (hf_name, served_name)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        """Canonicalise the endpoint so every consumer sees one spelling.

        --base-url .../v1/ built ".../v1//chat/completions" and the run died in
        warmup on a 404 that named no URL. Normalising the field also keeps the
        printed banner and the saved metadata from recording two spellings of
        one endpoint.
        """
        return v.rstrip("/")

    @model_validator(mode="after")
    def _shapes_must_be_positive(self) -> "BenchmarkConfig":
        """Reject shapes that cannot mean anything.

        --pp 0 or --tg -5 parsed happily and produced nonsense: pp is clamped
        with max(1, ...) at the use site, while tg flows straight into
        max_tokens. Better to refuse than to benchmark a shape the user did not
        ask for.
        """
        for name, values in (
            ("--pp", self.pp_counts),
            ("--tg", self.tg_counts),
            ("--concurrency", self.concurrency_levels),
        ):
            bad = [v for v in values if v < 1]
            if bad:
                raise ValueError(f"{name} values must be >= 1, got {bad}")
        negative_depths = [d for d in self.depths if d < 0]
        if negative_depths:
            raise ValueError(f"--depth values must be >= 0, got {negative_depths}")
        if self.num_runs < 1:
            raise ValueError(f"--runs must be >= 1, got {self.num_runs}")
        if self.warmup_runs < 0:
            raise ValueError(f"--warmup-runs must be >= 0, got {self.warmup_runs}")
        return self

    @classmethod
    def from_args(cls):
        # The subcommands are intercepted in llm-assay.py before argparse
        # ever sees them, so argparse cannot know they exist and would never
        # list them. Without this epilog, `-h` shows the benchmark's flags and
        # gives no hint that tune, compare or probe are there at all.
        parser = argparse.ArgumentParser(
            description="LLM Benchmark Script",
            epilog=(
                "subcommands:\n"
                "  tune <url>              detect an endpoint and suggest runs sized to it\n"
                "  compare a.json b.json   paired/Welch significance testing between saved runs\n"
                "  probe <url> --model M   quality probe: can the model still USE the context?\n"
                "\n"
                "Each takes its own --help, e.g. llm-assay.py probe --help\n"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )
        parser.add_argument(
            "--base-url", type=str, required=True, help="OpenAI compatible endpoint URL"
        )
        parser.add_argument(
            "--api-key", type=str, default="EMPTY", help="API Key for the endpoint"
        )
        parser.add_argument(
            "--model",
            type=str,
            required=False,
            default=None,
            help="Model name to use for benchmarking (auto-detected from endpoint if not specified)",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=None,
            help="Model name used in API calls (defaults to --model if not specified)",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Tokenizer to use (HF model name or local path; defaults to model name)",
        )
        parser.add_argument(
            "--pp",
            type=int,
            nargs="+",
            required=False,
            default=[2048],
            help="List of prompt processing token counts - default: 2048",
        )
        parser.add_argument(
            "--tg",
            type=int,
            nargs="+",
            required=False,
            default=[32],
            help="List of token generation counts - default: 32",
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default="chat",
            choices=["chat", "completions"],
            help="Route to benchmark - default: chat. 'completions' uses raw "
                 "/v1/completions: no chat template, so it isolates engine cost from "
                 "template cost and reaches stacks that never implemented the chat route. "
                 "Context and prompt are concatenated into one string.",
        )
        parser.add_argument(
            "--exact-tg",
            action="store_true",
            help="Force output length to match --tg by sending min_tokens=<tg> and ignore_eos=true in benchmark requests.",
        )
        parser.add_argument(
            "--depth",
            type=int,
            nargs="+",
            default=[0],
            help="List of context depths (previous conversation tokens) - default: 0",
        )
        parser.add_argument(
            "--runs", type=int, default=3, help="Number of runs per test - default: 3"
        )
        parser.add_argument(
            "--warmup-runs",
            type=int,
            default=1,
            help=(
                "Number of discarded warmup runs per test shape - default: 1. "
                "Also controls discarded probes for --latency-mode generation. "
                "Does not affect the initial prompt-adaptation warmup."
            ),
        )
        parser.add_argument(
            "--cold",
            action="store_true",
            help="Guarantee a cold prefix cache: shift the corpus offsets by a value "
                 "unique to this invocation, so a repeat run cannot be served from the "
                 "cache the previous one populated. Prefill and TTFT then measure real "
                 "ingest. Two --cold runs saw different text, so they cannot be compared "
                 "pairwise -- compare says so.",
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server",
        )
        parser.add_argument(
            "--post-run-cmd",
            type=str,
            default=None,
            help="Command to execute after each test run",
        )
        parser.add_argument(
            "--book-url",
            type=str,
            default="https://www.gutenberg.org/files/2600/2600-0.txt",
            help="URL of a book to use for text generation, defaults to War and Peace (Gutenberg 2600)",
        )
        parser.add_argument(
            "--latency-mode",
            type=str,
            default="api",
            choices=["api", "generation", "none"],
            help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)",
        )
        parser.add_argument(
            "--no-warmup", action="store_true", help="Skip warmup phase"
        )
        parser.add_argument(
            "--skip-coherence",
            action="store_true",
            help="Skip coherence test after warmup",
        )
        parser.add_argument(
            "--stall-timeout",
            type=float,
            default=300.0,
            metavar="SECONDS",
            help="Give up on a request that has sent nothing for SECONDS - default: 300. "
                 "There is no total cap, so a legitimately slow long-context prefill is "
                 "never cut off; 0 disables the check entirely.",
        )
        parser.add_argument(
            "--coherence-prompt",
            type=str,
            default=None,
            metavar="TEXT",
            help="Question for the coherence check. The default is English and expects "
                 "'Paris'; override it for a model instructed in another language.",
        )
        parser.add_argument(
            "--coherence-expect",
            type=str,
            nargs="*",
            default=None,
            metavar="TERM",
            help="Accepted substrings for the coherence answer (case-insensitive). "
                 "Pass with no values to accept any non-empty response, which checks "
                 "the model is generating without assuming a language.",
        )
        parser.add_argument(
            "--adapt-prompt",
            action="store_true",
            default=True,
            help="Adapt prompt size based on warmup token usage delta (default: True)",
        )
        parser.add_argument(
            "--no-adapt-prompt",
            action="store_false",
            dest="adapt_prompt",
            help="Disable prompt size adaptation",
        )
        parser.add_argument(
            "--measure-cached-followup",
            "--enable-prefix-caching",
            dest="enable_prefix_caching",
            action="store_true",
            help=(
                "Two-phase measurement: phase 1 loads the context (reported as ctx_pp/ctx_tg), "
                "phase 2 re-sends the SAME context plus a real prompt (reported as pp/tg @ depth). "
                "Phase 2 only measures a cached follow-up turn if the SERVER has prefix caching "
                "enabled -- otherwise it silently measures a full re-prefill. "
                "(--enable-prefix-caching is the old name for this flag; it never changed a server setting.)"
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Seed corpus sampling. Makes runs reproducible, and makes two configs draw "
                "IDENTICAL text for the same slot so they can be compared pairwise -- which "
                "removes content-driven variance, the dominant noise source with speculative "
                "decoding. The slot is (depth, pp, tg, concurrency, run index, reasoning "
                "effort, thinking), so pairing buys sensitivity for changes OUTSIDE that "
                "tuple. Arms of a --thinking or --reasoning-effort sweep deliberately read "
                "different text -- otherwise one arm would prefill from the cache the other "
                "just filled -- so a compare across them is unpaired however the seeds line up."
            ),
        )
        parser.add_argument(
            "--target-ci",
            type=float,
            default=None,
            metavar="FRAC",
            help=(
                "Adaptive sampling: keep running a test shape until the decode 95%% confidence "
                "interval is within FRAC of the mean (e.g. 0.03 for +/-3%%), instead of a fixed --runs. "
                "--runs becomes the minimum; --max-runs caps it."
            ),
        )
        parser.add_argument(
            "--max-runs",
            type=int,
            default=None,
            help="Upper bound on runs per shape when --target-ci is set (default: 5x --runs)",
        )
        parser.add_argument(
            "--legend",
            action="store_true",
            help=(
                "Explain the result table beneath it: what pp1024, tg128, "
                "ctx_pp @ d16384 and the (cN)/(think=)/(re=) suffixes mean, and "
                "how ttfr, est_ppt and e2e_ttft differ. Only the rows and "
                "columns actually printed are described."
            ),
        )
        parser.add_argument(
            "--stats",
            dest="stats_display",
            type=str,
            default="std",
            choices=["std", "ci", "median"],
            help=(
                "What the table shows: 'std' (mean +/- sample standard deviation, "
                "default), 'ci' (mean +/- 95%% confidence interval -- use this to judge "
                "whether a difference between two runs is real), or 'median' (median +/- "
                "half the inter-quartile range, which one slow run cannot drag around; "
                "useful because speculative-decode throughput is content-skewed)."
            ),
        )
        parser.add_argument(
            "--no-server-metrics",
            dest="server_metrics",
            action="store_false",
            default=True,
            help=(
                "Do not scrape the server's /metrics endpoint. By default llm-assay reads "
                "prefix-cache hit rate and speculative-decode acceptance per run, and warns "
                "if a cached-follow-up measurement got no cache hits."
            ),
        )
        parser.add_argument(
            "--thinking",
            dest="thinking_modes",
            nargs="+",
            choices=["on", "off"],
            default=None,
            metavar="MODE",
            help="Sweep thinking on and/or off (e.g. --thinking on off). 'off' sends both "
                 "spellings servers honour -- reasoning_effort=none and "
                 "chat_template_kwargs.enable_thinking=false -- and rows are labelled "
                 "(think=on)/(think=off). Whether graded --reasoning-effort levels do "
                 "anything depends on the deployment's chat template; this on/off switch "
                 "is the part that is portable.",
        )
        parser.add_argument(
            "--reasoning-effort",
            type=str,
            nargs="+",
            default=None,
            metavar="LEVEL",
            help=(
                "One or more reasoning effort levels to benchmark (e.g. --reasoning-effort none low high). "
                "Each level runs the full suite and results are labelled with it (e.g. 'tg1024 (re=none)'). "
                "Sent as a top-level 'reasoning_effort' field. Default: not sent."
            ),
        )
        parser.add_argument(
            "--concurrency",
            type=int,
            nargs="+",
            default=[1],
            help="List of concurrency levels (number of concurrent requests per test) - default: [1]",
        )
        parser.add_argument("--save-result", type=str, help="File to save results to")
        parser.add_argument(
            "--format",
            type=str,
            default="md",
            choices=["md", "json", "csv"],
            help="Output format",
        )
        parser.add_argument(
            "--save-total-throughput-timeseries",
            action="store_true",
            help="Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run.",
        )
        parser.add_argument(
            "--save-all-throughput-timeseries",
            action="store_true",
            help="Save calculated throughput timeseries for EACH individual request.",
        )
        parser.add_argument(
            "--exit-on-first-fail",
            action="store_true",
            help="Stop execution on first failed test and exit with non-zero status",
        )
        parser.add_argument(
            "--no-results-on-fail",
            action="store_true",
            help="Prevent saving/printing results when error is experienced, turns on --exit-on-first-fail as well",
        )
        parser.add_argument(
            "--extra-body",
            action="append",
            default=[],
            help="Extra JSON fields to merge into benchmark chat completion requests. Accepts key=value or key:value, comma-separated or repeated.",
        )
        parser.add_argument(
            "--emit-progress",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Emit benchmark progress events as JSONL to PATH (or '-' for stdout). "
                "External visualizers (live TUIs, web dashboards, post-hoc charts) "
                "consume this stream. Schema: docs/progress-schema.md."
            ),
        )

        args = parser.parse_args()

        if args.no_results_on_fail:
            args.exit_on_first_fail = True
        if args.warmup_runs < 0:
            parser.error("--warmup-runs must be >= 0")
        if args.target_ci is not None and not (0 < args.target_ci < 1):
            parser.error("--target-ci must be a fraction between 0 and 1 (e.g. 0.03 for +/-3%)")
        if args.max_runs is not None and args.max_runs < args.runs:
            parser.error("--max-runs must be >= --runs")
        if args.max_runs is not None and args.target_ci is None:
            # It is only consulted inside the adaptive-sampling branch, so on
            # its own it is a no-op. Silently ignoring a cap the user asked for
            # is worse than saying it does nothing.
            print(
                "[WARNING] --max-runs has no effect without --target-ci; runs per shape "
                "are fixed by --runs.",
                file=sys.stderr,
            )

        try:
            extra_body = BenchmarkConfig._parse_extra_body(args.extra_body)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Auto-detect model if not specified
        if args.model is None:
            print("No model specified, attempting to auto-detect from endpoint...")
            try:
                hf_model, served_model = BenchmarkConfig._detect_hf_model_from_endpoint(
                    args.base_url, args.api_key
                )
                model_to_use = hf_model
                served_model_name_to_use = (
                    args.served_model_name if args.served_model_name else served_model
                )
                print(
                    f"Auto-detected HF model: {model_to_use} (served as: {served_model_name_to_use})"
                )
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            model_to_use = args.model
            served_model_name_to_use = (
                args.served_model_name if args.served_model_name else args.model
            )

        return cls(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model_to_use,
            served_model_name=served_model_name_to_use,
            tokenizer=args.tokenizer,
            pp_counts=args.pp,
            tg_counts=args.tg,
            endpoint=args.endpoint,
            exact_tg=args.exact_tg,
            depths=args.depth,
            num_runs=args.runs,
            warmup_runs=args.warmup_runs,
            cold=args.cold,
            no_cache=args.no_cache,
            latency_mode=args.latency_mode,
            no_warmup=args.no_warmup,
            skip_coherence=args.skip_coherence,
            stall_timeout=args.stall_timeout,
            coherence_prompt=args.coherence_prompt,
            coherence_expect=args.coherence_expect,
            adapt_prompt=args.adapt_prompt,
            enable_prefix_caching=args.enable_prefix_caching,
            seed=args.seed,
            target_ci=args.target_ci,
            max_runs=args.max_runs,
            stats_display=args.stats_display,
            legend=args.legend,
            server_metrics=args.server_metrics,
            reasoning_efforts=args.reasoning_effort,
            thinking_modes=args.thinking_modes,
            book_url=args.book_url,
            post_run_cmd=args.post_run_cmd,
            concurrency_levels=args.concurrency,
            save_result=args.save_result,
            result_format=args.format,
            save_total_throughput_timeseries=args.save_total_throughput_timeseries,
            save_all_throughput_timeseries=args.save_all_throughput_timeseries,
            exit_on_first_fail=args.exit_on_first_fail,
            no_results_on_fail=args.no_results_on_fail,
            extra_body=extra_body,
            emit_progress=args.emit_progress,
        )
