# llm-assay - llama-bench style benchmarking for any OpenAI-compatible LLM endpoint

Prompt-processing and decode speed at real context depths, against vLLM, SGLang or
llama.cpp, with confidence intervals, paired A/B significance testing, prefix-cache
trap detection, and a long-context quality probe.

## Motivation

`llama-bench` is a CLI tool that is a part of a very popular [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine. It is widely used in LLM community to benchmark models and allows to perform measurement at different context sizes.
However, it is available only for llama.cpp and cannot be used with other inference engines, like vllm or SGLang.

Also, it performs measurements using the C++ engine directly which is not representative of the end user experience which can be quite different in practice.

vLLM has its own powerful benchmarking tool, but while it can be used with other inference engines, there are a few issues:

- It's very tricky and even impossible to calculate prompt processing speeds at different context lengths. You can use `vllm bench sweep serve`, but it only works well with vLLM with prefix caching disabled on the server. Even with random prompts it will reuse the same prompt between multiple runs which will hit the cache in `llama-server` for instance. So you will get very low median TTFT times and very high prompt processing speeds. 
- The TTFT measurement it uses is not actually until the first usable token, it's until the very first data chunk from the server which may not contain any generated tokens in /v1/chat/completions mode.
- Random dataset is the only ones that allows to specify an arbitrary number of tokens, but randomly generated token sequence doesn't let you adequately measure speculative decoding/MTP.

As of January 2nd, 2026, I wasn't able to find any existing benchmarking tool that brings llama-bench style measurements at different context lengths to any OpenAI-compatible endpoint.

## Features

- Measures Prompt Processing (pp) and Token Generation (tg) speeds at different context depths.
- Can measure separate context prefill and prompt processing over existing cached context at different context depths.
- Reports Time To First Response (data chunk) (TTFR), Estimated Prompt Processing Time (est_ppt), and End-to-End TTFT.
- Supports configurable prompt length (`--pp`), generation length (`--tg`), and context depth (`--depth`).
- Can run multiple iterations (`--runs`) and report mean ± std.
- Uses HuggingFace tokenizers for accurate token counts.
- Correctly handles multi-token prediction (MTP) chunks.
- Downloads a book from Project Gutenberg to use as source text for prompts to ensure better benchmarking of spec.decoding/MTP models.
- Supports executing a command after each run (e.g., to clear cache).
- Configurable latency measurement mode.
- Supports concurrent requests (`--concurrency`) to measure throughput under load.
- Seeded, reproducible corpus sampling (`--seed`) enabling *paired* A/B comparison of two configurations.
- Statistically sound reporting: sample std, Student-t confidence intervals, and warnings when a run is underpowered.
- Adaptive sampling (`--target-ci`): keep running until the result is precise enough, instead of guessing `--runs`.
- Scrapes the server's Prometheus `/metrics` for prefix-cache hit rate and speculative-decode acceptance — vLLM counters diffed per shape, SGLang-style gauges reported as server-lifetime and labelled as such, and an unrecognised endpoint says so rather than disabling silently. Warns in both directions: when a cached-follow-up measurement got no cache hits, and when an ordinary run was unexpectedly served *from* the cache.
- `llm-assay.py compare` for paired / Welch significance testing between saved runs.
- `llm-assay.py probe` for a **quality** check to sit beside the speed numbers: it
  injects invented, self-contradicting statements into the corpus at a chosen depth and
  measures whether the model still finds them. Throughput at 200k says nothing about
  whether the model can *use* 200k, and the features that make long context fast
  (quantized KV, windowed attention, chunked prefill) are the ones that can cost
  comprehension. Scored without a second model: a correct answer must name a surname
  that exists in no training set.
- Can save results to file in Markdown, JSON, or CSV format.
- Can save granular time-series data for token generation when JSON output is used (`--save-total-throughput-timeseries` and `--save-all-throughput-timeseries`).
- Runs a coherence test after warmup to verify model responds correctly (default, can be skipped with `--skip-coherence`).
- Auto-detects HuggingFace model name from the endpoint's `/models` endpoint when `--model` is not specified.

# Current Limitations

- Evaluates against `/v1/chat/completions` endpoint only.

## Setup

This project is run directly from a checkout -- there is no packaging or install
step, no wheel and no console scripts. [`uv`](https://docs.astral.sh/uv/) is the
recommended way to run it.

Install `uv` if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Recommended: `uv run` (no setup at all)

`llm-assay.py` carries [PEP 723](https://peps.python.org/pep-0723/) inline
metadata declaring its own dependencies, so `uv` resolves and caches an
environment on first use. There is nothing to install and no virtualenv to
activate or remember:

```bash
git clone https://github.com/alexeytz/llm-assay.git
cd llm-assay

uv run llm-assay.py --base-url http://localhost:8000/v1 --model my-model
```

The comparison tool is a subcommand of the same script:

```bash
uv run llm-assay.py compare baseline.json candidate.json
uv run llm-assay.py compare baseline.json candidate.json --metric e2e_ttft
```

The script is also executable directly, since its shebang defers to uv:

```bash
./llm-assay.py --base-url http://localhost:8000/v1 --model my-model
```

### Alternative: a managed virtual environment

If you would rather have a persistent environment -- to add packages, or to point
an IDE at it:

```bash
uv venv                                   # creates .venv using a suitable Python
uv pip install -r requirements.txt
uv run llm-assay.py --base-url http://localhost:8000/v1 --model my-model
```

### Alternative: plain `pip`

`uv` is not required. `requirements.txt` mirrors the dependency list embedded in
`llm-assay.py`:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python llm-assay.py --base-url http://localhost:8000/v1 --model my-model
```


### Notes

Requires Python 3.10+. `uv` will fetch a suitable interpreter automatically if
the system one is older.

`llm-assay.py` prepends the checkout's `src/` to `sys.path`, so you always
benchmark the code in front of you. The version it reports combines the `VERSION`
file with the current commit id (e.g. `0.5.0-dev+g<sha>`, plus `.dirty` for
uncommitted changes), and that string is recorded in saved results -- so a result
stays traceable to the code that produced it.

> The dependency list exists twice: as PEP 723 metadata inside `llm-assay.py`
> (for `uv`) and in `requirements.txt` (for `pip`). A test
> (`test_pep723_matches_requirements`) fails if the two ever disagree.

> `asyncio` is not listed as a dependency. It is part of the standard library on
> every supported Python; the PyPI package of that name is a deprecated
> placeholder, and older releases of it shipped a Python 3.3-era backport that
> could shadow the stdlib module.

## Usage

> Examples below are written as `llm-assay` for brevity. Run them as
> `uv run llm-assay.py` (or `python llm-assay.py` inside an activated
> environment), and `uv run llm-assay.py compare` for the comparison tool.


After installation, you can run the tool directly:

```bash
llm-assay --base-url <ENDPOINT_URL> --model <MODEL_NAME> --pp <PROMPT_TOKENS> --tg <GEN_TOKENS> [OPTIONS]
```

Example:

```bash
llm-assay \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation
```

Output:


| model               |            test |             t/s |     peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:--------------------|----------------:|----------------:|-------------:|---------------:|---------------:|----------------:|
| openai/gpt-oss-120b |          pp2048 | 8521.08 ± 69.61 |              |  297.14 ± 1.97 |  240.36 ± 1.97 |   340.65 ± 3.49 |
| openai/gpt-oss-120b |            tg32 |    73.18 ± 0.45 | 75.84 ± 0.48 |                |                |                 |
| openai/gpt-oss-120b |  pp2048 @ d4096 | 9450.36 ± 24.73 |              |  706.92 ± 1.70 |  650.14 ± 1.70 |   750.96 ± 3.08 |
| openai/gpt-oss-120b |    tg32 @ d4096 |    72.22 ± 0.83 | 74.81 ± 0.86 |                |                |                 |
| openai/gpt-oss-120b |  pp2048 @ d8192 | 8481.42 ± 38.50 |              | 1264.15 ± 5.50 | 1207.37 ± 5.50 |  1307.31 ± 6.20 |
| openai/gpt-oss-120b |    tg32 @ d8192 |    71.78 ± 0.74 | 74.36 ± 0.77 |                |                |                 |
| openai/gpt-oss-120b | pp2048 @ d16384 | 7954.96 ± 14.20 |              | 2373.83 ± 4.14 | 2317.05 ± 4.14 |  2418.63 ± 4.87 |
| openai/gpt-oss-120b |   tg32 @ d16384 |    70.48 ± 0.84 | 73.02 ± 0.86 |                |                |                 |
| openai/gpt-oss-120b | pp2048 @ d32768 |  6896.57 ± 4.62 |              | 5105.09 ± 3.38 | 5048.31 ± 3.38 |  5153.34 ± 2.87 |
| openai/gpt-oss-120b |   tg32 @ d32768 |    65.80 ± 0.79 | 68.17 ± 0.82 |                |                |                 |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)   # sample captured before the fork
date: 2026-02-06 15:52:14 | latency mode: generation

-------

It's recommended to use "generation" latency mode to get prompt processing speeds closer to real numbers, especially on shorter prompts.
By default, the script adapts the prompt size to match the specified value, regardless of the chat template applied. Use `--no-adapt-prompt` to disable this behavior.

Within a single run the probability of accidental cache hits is small, so you rarely need to disable prompt caching on the server. **Across** runs it is not small at all: a fixed `--seed` sends byte-identical prompts, so a second invocation of the same command is served warm and prefill inflates — see [the trap section](#the-trap-a-fixed---seed-warms-the-cache-for-the-next-run) and `--cold`, which prevents it. You can add `--no-cache` that will add some random noise if you get cache hits.

### Arguments

-   `--base-url`: OpenAI compatible endpoint URL (Required).
-   `--api-key`: API Key (Default: "EMPTY").
-   `--model`: Model name to use for benchmarking. If not specified, attempts to auto-detect from the endpoint's `/models` endpoint.
-   `--served-model-name`: Model name used in API calls (Defaults to --model if not specified). Tries to autodetect from the endpoint's `/models` endpoint (if supported, e.g. vLLM).
-   `--tokenizer`: HuggingFace tokenizer name or local path (Defaults to model name).
-   `--pp`: List of prompt processing token counts (Default: [2048]).
-   `--tg`: List of token generation counts (Default: [32]).
-   `--endpoint {chat,completions}`: Which route to benchmark (default `chat`). `completions` uses raw `/v1/completions` — no chat template, context and prompt concatenated — which reaches stacks that never implemented the chat route and isolates engine cost from template cost. Visible in the warmup: template overhead measures 9 tokens on the chat route and 0 on completions.
-   `--exact-tg`: Force output length to match `--tg` by sending `min_tokens=<tg>` and `ignore_eos=true` in benchmark requests. This is useful for fixed-OSL throughput runs on compatible servers such as vLLM.

    > **Caveat for speculative decoding.** `ignore_eos` makes the model continue past its natural stopping point, and long forced generations tend to degenerate into repetition. Repetitive text is unusually easy for a draft head to predict, so `--exact-tg` can *inflate* measured acceptance relative to real use. It buys determinism in output length at the cost of realism in acceptance; keep that in mind when benchmarking MTP/Eagle setups.
-   `--depth`: List of context depths (Default: [0]).
-   `--runs`: Number of runs per test (Default: 3).
-   `--warmup-runs`: Number of discarded warmup runs per test shape (Default: 1). For concurrency `N`, each warmup run sends `N` requests. Also controls the number of discarded warmup probes for `--latency-mode generation`; it does not affect the initial prompt-adaptation warmup.
-   `--cold`: Guarantee a cold prefix cache by salting the corpus offsets per invocation, so a repeat run cannot be served from the cache the previous one populated. Measured: without it a second invocation of the same command sees 69.4% hits, with it every invocation sees 0.0%. `--no-cache` cannot do this — prefix caching matches the *prefix*, and its buster is appended to the end. Two `--cold` runs read different text, so `compare` will not pair them.
-   `--no-cache`: Add noise to requests to improve prefix caching avoidance. Also sends `cache-prompt=false` to the server.
-   `--post-run-cmd`: Command to execute after each test run.
-   `--book-url`: URL of a book to use for text generation (defaults to War and Peace,
    Project Gutenberg 2600, ~787k tokens). The corpus has to be longer than the deepest
    shape you measure -- `--depth + --pp` past its end reads *repeated* text, which a
    speculative draft head predicts unusually well and which therefore flatters decode.
    The previous default (Sherlock Holmes, ~144k tokens) could not cover a context window
    past ~144k, so anything deeper was measuring repetition. A local path works too.
    The Project Gutenberg licence block at the end of the file is dropped: it is legal
    boilerplate rather than prose, and a speculative draft head predicts it unusually
    well, so a prompt drawn from it flatters decode. The corpus keeps its original
    length (the story is repeated to refill the gap), so results saved before this
    still pair with results saved after.
-   `--latency-mode`: Method to measure latency: 'api' (call list models function) - default, 'generation' (single token generation), or 'none' (skip latency measurement).
-   `--no-warmup`: Skip warmup phase.
-   `--skip-coherence`: Skip coherence test after warmup.
-   `--coherence-prompt` / `--coherence-expect`: The default sanity check asks an English question and expects "Paris", which is a false failure for a model instructed in another language. Override the question and the accepted answers, or pass `--coherence-expect` with no values to accept any non-empty response.
-   `--stall-timeout SECONDS`: Abandon a request that has sent nothing for SECONDS (default 300). There is no total cap, so a slow long-context prefill is never cut off..
-   `--adapt-prompt`: Adapt prompt size based on warmup token usage delta (Default: True).
-   `--no-adapt-prompt`: Disable prompt size adaptation.
-   `--measure-cached-followup` (formerly `--enable-prefix-caching`, still accepted): Two-phase benchmark. When enabled (and depth > 0) it first loads the context (reported as `ctx_pp`), then runs a prompt over that same context. **This is a client-side measurement mode; it does not configure the server.** If the server does not have prefix caching enabled, phase 2 re-prefills the whole context and the `pp… @ d…` rows measure a full re-prefill rather than a cached follow-up turn. llm-assay now detects this and warns (see [Prefix Caching Benchmarking](#prefix-caching-benchmarking)).
-   `--seed`: Seed corpus sampling. Makes a run reproducible, and makes two runs draw **identical** text for the same logical slot so they can be compared *pairwise*. With speculative decoding, content-driven acceptance variance is usually the dominant noise source, so pairing is far more sensitive than comparing two independent samples. Sampling is keyed, not sequential, so pairing survives even if one side issues a different number of warmup probes. The key is the *shape* -- `(depth, pp, tg, concurrency, run index, reasoning effort)` -- so pairing applies to two configs that differ in something **outside** it: thinking mode, `--extra-body`, a server-side change. Changing `--pp`, `--tg`, `--depth` or `--concurrency` draws different text *and* makes a different row in `compare`, so those sides have nothing to pair against each other; `compare` exits non-zero rather than reporting a clean run when no shape matched. Both `--reasoning-effort` and `--thinking` are *in* the key, so each arm of those sweeps reads its own text. That means a paired test across arms stays valid but gains no power from pairing -- and, more importantly, no arm warms the prefix cache for the next. `--thinking` needs this most: only `chat_template_kwargs` differs between its two requests, so sharing text would let the second arm's prefill be served from the cache the first arm built, inside a single invocation, with `--no-cache` powerless to stop it (its buster is appended *after* the prefix). `--thinking` joins the key only when it is sweeping, so a result saved without it still pairs with a new one.
-   `--target-ci FRAC`: Adaptive sampling. Keep running a test shape until the decode 95% confidence interval is within `FRAC` of the mean (e.g. `0.03` for ±3%), instead of a fixed `--runs`. `--runs` becomes the minimum and `--max-runs` the cap (default `5 × --runs`).
-   `--max-runs`: Upper bound on runs per shape when `--target-ci` is set.
-   `--legend`: Print an explanation of the table beneath it. Every entry leads with a plain sentence — `pp512` is *"how fast the model reads the prompt; it was handed a 512-token prompt (roughly 380 words of English)"* — and puts the mechanics one indent further in, so a first-time reader and someone who wants the formula are both served. Covers the `pp`/`tg`/`ctx_pp`/`@ d` row labels, the `(cN)`/`(think=)`/`(re=)` suffixes, and how `ttfr`, `est_ppt` and `e2e_ttft` differ. Only the rows and columns actually printed are described, so it never explains a `t/s (req)` column a `c1` table does not have. Off by default; with `--format md` it is appended to the saved file, fenced so the layout survives.
-   `--stats {std,ci,median}`: What the `±` column means — `std` (sample standard deviation, default) or `ci` (95% confidence interval of the mean). Use `ci` when deciding whether a difference between two runs is real.
-   `--no-server-metrics`: Disable scraping of the server's Prometheus `/metrics` endpoint. By default llm-assay records prefix-cache hit rate and speculative-decode acceptance length per test shape, and warns when a cached-follow-up measurement got no cache hits.
-   `--thinking on|off`: Sweep thinking as a benchmark dimension; rows are labelled `(think=on)`/`(think=off)`. `off` sends both spellings servers honour — `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking: false` — because deployments honour different ones. Measured on a Qwen3.8 served by vLLM: prefill unchanged, decode **48.50 t/s thinking vs 38.15 t/s not**, tracking speculative acceptance 3.03 vs 2.36 — reasoning text is repetitive, so the draft head predicts it better and each token is cheaper.
-   `--reasoning-effort`: One or more reasoning effort levels to benchmark (e.g. `--reasoning-effort none low high`). Each level runs the full suite and results are labelled with it (e.g. `tg1024 (re=none)`). Sent as a top-level `reasoning_effort` field. Default: not sent.

    > **Check the levels do anything before sweeping them.** Whether a graded level has an effect depends on the deployment's chat template, not the model card. On an unsloth conversion of Qwen3.8 served by vLLM, `low`, `medium` and `xhigh` render a *byte-identical* prompt — only `none` differs, by prefilling an empty `<think></think>` block — so sweeping them runs the suite repeatedly over a single configuration. `llm-assay.py tune <url>` probes the endpoint and reports which switches are live.
-   `--concurrency`: List of concurrency levels (number of concurrent requests per test) (Default: [1]).
-   `--save-result`: File to save results to.
-   `--format`: Output format: 'md', 'json', 'csv' (Default: 'md').
-   `--version`: Print the version (`VERSION` file plus the short commit id) and exit.
-   `--save-total-throughput-timeseries`: Save calculated TOTAL throughput for each 1 second window inside peak throughput calculation during the run (default: off). One series per reported peak, index-aligned with `peak_throughput.values`; the largest point of a series equals the peak saved beside it.
-   `--save-all-throughput-timeseries`: Save calculated throughput timeseries for EACH individual request (default: off).
-   `--exit-on-first-fail`: Stop execution on first failed test and exit with non-zero status.
-   `--no-results-on-fail`: Prevent saving/printing any results when error is experienced, turns on --exit-on-first-fail as well.
-   `--extra-body`: Extra JSON fields to merge into benchmark chat completion requests. Accepts repeated or comma-separated `key=value` / `key:value` entries, e.g. `--extra-body min_tokens=1024,ignore_eos=true`.

For fixed-output-length throughput benchmarks, prefer `--exact-tg` over manually passing `min_tokens` and `ignore_eos`:

```bash
llm-assay \
  --base-url http://localhost:8000/v1 \
  --model my-model \
  --pp 2160 \
  --tg 1024 \
  --exact-tg
```

### Metrics

The script outputs a table with the following metrics. All time measurements are in milliseconds (ms).

#### Latency Adjustment

The script attempts to estimate network or processing latency to provide "server-side" processing times.
- **Latency**: Measured based on `--latency-mode`.
  - `api`: Time to fetch `/models` (from sending request to getting first byte of the response). Eliminates network latency only.
  - `generation`: Time to generate 1 token (from sending request to getting first byte of the response). Tries to eliminate network and server overhead latency. By default this sends 4 single-token streaming probes, discards the first as a request-shape warmup, and averages the remaining 3. `--warmup-runs` controls the number of discarded generation-latency probes.
  - `none`: Assumed to be 0.
- This measured latency is subtracted from `ttfr` to calculate `est_ppt`.

#### Table Columns

-   **`t/s` (Tokens per Second)**:
    -   **For Prompt Processing (pp)**: Calculated as `Total Prompt Tokens / est_ppt`. This represents the prefill speed.
        -   *Total Prompt Tokens* is the server's reported `usage.prompt_tokens` when it is within 20% of the count llm-assay asked for, and the requested count otherwise. The two differ by the chat template's own tokens -- a few per request -- so a hand-check against the requested count lands a fraction of a percent off the reported figure. The 20% guard exists because a server that reports prompt tokens for a *cached* prefix, or reports nothing at all, would otherwise silently rescale the metric.
        -   **This formula describes `concurrency = 1` only.** At `concurrency > 1` the `t/s (total)` column is `sum of prompt tokens / (last first-token - first request start)` -- wall-clock across the batch, with **no latency subtraction**. `t/s (req)` keeps the latency-subtracted per-request formula above. So the prefill total at c>1 and the prefill figure at c=1 are not on the same basis: the c>1 number includes the client-server round trip that `est_ppt` removes. Compare c>1 against c>1, or use `t/s (req)`.
    -   **For Token Generation (tg)**: Calculated as `Tokens observed after the first token timestamp / (Time of Last Token - Time of First Token)`. This represents decode speed over the observable post-first-token interval. For block-streaming backends, all tokens that arrive with the first content timestamp are excluded because they have no observable generation interval. If the backend emits the whole response in a single content-bearing stream chunk, every token carries the same timestamp and there is no observable interval at all: the decode row is omitted rather than reporting a protocol-timing artifact. That applies to `peak t/s` as well as the mean -- 130 tokens stamped with one instant is not a throughput. The shape stays visible through its prefill row.
        -   When `concurrency` > 1:
        -   **`t/s (total)`**: Total throughput across all concurrent requests.
        -   **`t/s (req)`**: Average throughput per individual request.

- **`peak t/s` (Maximum observed Tokens per Second)**: 
    - **Only for Token Generation (tg)**: The highest token‑generation throughput observed in any 1‑second window during the run across all concurrent requests.
    - At `concurrency > 1` this is the aggregate peak; the `peak t/s (req)` column instead averages each request's own peak, so the two answer different questions and the total is not a sum of the per-request figures.

-   **`ttfr (ms)` (Time To First Response)**:
    -   Calculation: `Time of First Response Chunk - Start Time`.
    -   Represents the raw time until the client receives *any* stream data from the server (including empty chunks or role definitions, but excluding initial http response header). This includes network latency. The same measurement method is used by `vllm bench serve` to report TTFT.

-   Each metric carries its own `values` list, and **the lists can differ in length**,
    for two separate reasons.
    -   *Batch versus per-request.* `pp_throughput`, `tg_throughput` and
        `peak_throughput` are batch figures: one value per run. `pp_req_throughput`,
        `tg_req_throughput`, `ttfr`, `est_ppt` and `e2e_ttft` are per-request: one
        value per request. At `--concurrency 2` over 2 runs that is 2 values against
        4. Metrics within a family stay index-aligned with each other; across
        families they do not, and pairing them by position compares a run to a
        request.
    -   *Missing samples.* `ttfr` is recorded whenever any stream chunk arrives;
        `est_ppt` and `e2e_ttft` need a content-bearing token. A response that sends
        a role-only chunk and then stops contributes a `ttfr` sample and no
        `est_ppt` sample.

    Each mean is valid over its own list; only the zipping is unsafe.

-   **`est_ppt (ms)` (Estimated Prompt Processing Time)**:
    -   Calculation: `TTFR - Estimated Latency`.
    -   Estimated time the server spent processing the prompt. Used for calculating Prompt Processing speed.

-   **`e2e_ttft (ms)` (End-to-End Time To First Token)**:
    -   Calculation: `Time of First Content Token - Start Time`.
    -   The total time perceived by the client from sending the request to seeing the first generated content.

### Prefix Caching Benchmarking

> **Read this before interpreting `pp… @ d…` rows.**
> `--measure-cached-followup` is a *client-side* measurement mode. It never changes
> a server setting. Phase 2 re-sends the same context plus a real prompt, and the
> result is reported as `pp{tokens} @ d{depth}`. That row only means "cost of a
> follow-up turn over an existing context" **if the server actually reused the
> context**. If the server has prefix caching disabled, the identical measurement
> silently becomes "cost of re-prefilling the whole context", and the timings look
> perfectly plausible either way.
>
> This matters because several engines do not enable prefix caching by default.
> vLLM, for instance, keeps it opt-in for hybrid (Mamba/attention) models such as
> Qwen3-Next and Qwen3.5 — `--enable-prefix-caching` is required there, and without
> it a 64k follow-up turn can read as ~150 t/s when the cached figure is ~1800 t/s.
>
> llm-assay now scrapes the server's `/metrics` endpoint and prints a loud
> warning if this mode ran with ~0% cache hits, so the misreading is visible rather
> than silent. Pass `--no-server-metrics` to disable the check.


When `--enable-prefix-caching` is used (with `--depth` > 0), the script performs a two-step process for each run to measure the impact of prefix caching:

1.  **Context Load**: Sends the context tokens (as a system message) with a minimal user probe message. This forces the server to process and cache the context while staying compatible with frontends that reject empty user messages.
    -   Reported as `ctx_pp @ d{depth}` (Context Prompt Processing) and `ctx_tg @ d{depth}`.
2.  **Inference**: Sends the same context (system message) followed by the actual prompt (user message). The server should reuse the cached context.
    -   Reported as standard `pp{tokens} @ d{depth}` and `tg{tokens} @ d{depth}`.

In this case, `pp` and `tg` speeds will show an actual prompt processing / token generation speeds for a follow up prompt with a context pre-filled.

**Example**:

```bash
llm-assay \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation \
  --enable-prefix-caching
```

Output:


| model               |            test |              t/s |      peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|----------------:|-----------------:|--------------:|-----------------:|-----------------:|-----------------:|
| openai/gpt-oss-120b |          pp2048 | 8236.95 ± 134.25 |               |    298.95 ± 4.08 |    248.70 ± 4.08 |    342.07 ± 3.40 |
| openai/gpt-oss-120b |            tg32 |     73.96 ± 1.19 |  76.63 ± 1.24 |                  |                  |                  |
| openai/gpt-oss-120b |  ctx_pp @ d4096 |  9259.71 ± 76.35 |               |    492.62 ± 3.63 |    442.38 ± 3.63 |    535.67 ± 3.75 |
| openai/gpt-oss-120b |  ctx_tg @ d4096 |     73.28 ± 0.80 |  75.93 ± 0.82 |                  |                  |                  |
| openai/gpt-oss-120b |  pp2048 @ d4096 | 7467.44 ± 131.59 |               |    324.59 ± 4.84 |    274.34 ± 4.84 |    367.60 ± 4.95 |
| openai/gpt-oss-120b |    tg32 @ d4096 |     72.25 ± 0.12 |  74.86 ± 0.12 |                  |                  |                  |
| openai/gpt-oss-120b |  ctx_pp @ d8192 | 9177.24 ± 167.37 |               |   943.19 ± 16.45 |   892.94 ± 16.45 |    973.01 ± 5.31 |
| openai/gpt-oss-120b |  ctx_tg @ d8192 |     73.43 ± 0.50 |  76.09 ± 0.54 |                  |                  |                  |
| openai/gpt-oss-120b |  pp2048 @ d8192 | 6846.48 ± 135.57 |               |    349.50 ± 5.96 |    299.25 ± 5.96 |    394.26 ± 5.16 |
| openai/gpt-oss-120b |    tg32 @ d8192 |     72.62 ± 0.66 |  75.23 ± 0.68 |                  |                  |                  |
| openai/gpt-oss-120b | ctx_pp @ d16384 | 8235.05 ± 179.06 |               |  2040.75 ± 43.92 |  1990.50 ± 43.92 |  2073.53 ± 23.55 |
| openai/gpt-oss-120b | ctx_tg @ d16384 |     73.87 ± 5.04 |  76.53 ± 5.22 |                  |                  |                  |
| openai/gpt-oss-120b | pp2048 @ d16384 | 5441.56 ± 484.88 |               |   429.81 ± 35.96 |   379.57 ± 35.96 |   483.42 ± 48.92 |
| openai/gpt-oss-120b |   tg32 @ d16384 |    62.80 ± 10.73 | 65.06 ± 11.12 |                  |                  |                  |
| openai/gpt-oss-120b | ctx_pp @ d32768 | 6904.92 ± 217.24 |               | 4800.62 ± 151.68 | 4750.38 ± 151.68 | 4832.95 ± 157.53 |
| openai/gpt-oss-120b | ctx_tg @ d32768 |     69.77 ± 5.32 |  72.29 ± 5.52 |                  |                  |                  |
| openai/gpt-oss-120b | pp2048 @ d32768 | 4549.10 ± 105.92 |               |   500.69 ± 10.32 |   450.44 ± 10.32 |    548.23 ± 8.98 |
| openai/gpt-oss-120b |   tg32 @ d32768 |     62.18 ± 6.87 |  64.59 ± 6.87 |                  |                  |                  |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)   # sample captured before the fork
date: 2026-02-06 16:15:38 | latency mode: generation

### Combining multiple parameters

You can specify multiple parameters for `--depth`, `--pp`, `--tg` and `--concurrency`. The benchmarks will run using the following hierarchy: depth -> pp -> tg -> concurrency.

```bash
llm-assay \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-120b \
  --pp 128 256 \
  --tg 32 64 \
  --depth 0 1024
```

This will run benchmarks for all combinations of pp (128, 256), tg (32, 64), and depth (0, 1024).

### Concurrency measurement

To test how the server performs in concurrent requests scenario, you can specify one or more concurrency levels using `--concurrency <N>` (e.g. `--concurrency 1 2 4`).

When running with `concurrency > 1`, `llm-assay` launches N parallel clients. The results table will include:
*   **t/s (total)**: The aggregate throughput (tokens/sec) of all clients combined.
*   **t/s (req)**: The average throughput per client.

This allows you to measure how the server scales and find the saturation point where adding more clients doesn't increase the total throughput.

Please note, that currently all batches run at the same time. If `--enable-prefix-caching` is used, then all prefill requests are executed simultaneously, followed by concurrent follow up requests.
Other concurrency scenarios may be added in the future.

**Example**

```bash
llm-assay \
  --base-url http://spark:8888/v1 \
  --model openai/gpt-oss-120b \
  --depth 0 4096 \
  --latency-mode generation \
  --enable-prefix-caching \
  --concurrency 1 2
```

Output:

| model               |                test |       t/s (total) |         t/s (req) |       peak t/s |       ttfr (ms) |    est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|--------------------:|------------------:|------------------:|---------------:|----------------:|----------------:|-----------------:|
| openai/gpt-oss-120b |         pp2048 (c1) |   7803.68 ± 63.74 |   7803.68 ± 63.74 |                |   292.86 ± 2.15 |   262.46 ± 2.15 |    337.08 ± 2.48 |
| openai/gpt-oss-120b |           tg32 (c1) |      74.83 ± 0.59 |      74.83 ± 0.59 |   77.54 ± 0.62 |                 |                 |                  |
| openai/gpt-oss-120b |         pp2048 (c2) |  7198.20 ± 541.69 | 4872.80 ± 1298.05 |                |  473.18 ± 85.71 |  442.77 ± 85.71 |   568.97 ± 40.44 |
| openai/gpt-oss-120b |           tg32 (c2) |     111.27 ± 3.61 |      56.20 ± 1.27 |  115.25 ± 3.74 |                 |                 |                  |
| openai/gpt-oss-120b | ctx_pp @ d4096 (c1) |   8816.20 ± 55.40 |   8816.20 ± 55.40 |                |   495.02 ± 2.91 |   464.62 ± 2.91 |    540.31 ± 1.64 |
| openai/gpt-oss-120b | ctx_tg @ d4096 (c1) |      72.92 ± 0.63 |      72.92 ± 0.63 |   75.56 ± 0.67 |                 |                 |                  |
| openai/gpt-oss-120b | pp2048 @ d4096 (c1) | 5918.71 ± 1447.23 | 5918.71 ± 1447.23 |                | 403.38 ± 110.24 | 372.98 ± 110.24 |   432.07 ± 90.02 |
| openai/gpt-oss-120b |   tg32 @ d4096 (c1) |      65.27 ± 9.85 |      65.27 ± 9.85 |  67.62 ± 10.21 |                 |                 |                  |
| openai/gpt-oss-120b | ctx_pp @ d4096 (c2) | 7934.38 ± 1145.70 |  4665.38 ± 660.55 |                | 928.79 ± 146.59 | 898.38 ± 146.59 | 1053.95 ± 165.93 |
| openai/gpt-oss-120b | ctx_tg @ d4096 (c2) |     111.64 ± 3.25 |      56.38 ± 1.06 |  115.63 ± 3.36 |                 |                 |                  |
| openai/gpt-oss-120b | pp2048 @ d4096 (c2) |  6659.05 ± 231.76 |  3623.78 ± 278.02 |                |  598.81 ± 42.34 |  568.40 ± 42.34 |   615.33 ± 22.02 |
| openai/gpt-oss-120b |   tg32 @ d4096 (c2) |    116.93 ± 10.24 |      58.47 ± 5.12 | 121.11 ± 10.61 |                 |                 |                  |

llama-benchy (0.2.2.dev1+g52d2b0d55.d20260206)   # sample captured before the fork
date: 2026-02-06 16:36:05 | latency mode: generation

### Further analysis

To perform additional analysis or generate any visualizations, you can output results in JSON or CSV. 
JSON (`--format json`) will give you the most detailed data. If you specify `--save-total-throughput-timeseries`, then JSON will include total throughput in 1 second intervals.

- [Sample JSON file](schemas/sample.json)
- [Sample JSON file with embedded documentation](schemas/sample.jsonc)
- [JSON schema](schemas/benchmark_report_schema.json)

You can also drive llm-assay directly from Python: put `src/` on `sys.path`, build a
`BenchmarkConfig`, hand it to a `BenchmarkRunner` along with a `TokenizedCorpus`,
`PromptGenerator` and `LLMClient`, `await runner.run_suite()`, then read
`runner.results.runs` or call `runner.results.save_report(path, "json")`.



## Sizing a run to an endpoint (`tune`)

You do not have to work out which depths a server can take, or whether
`--measure-cached-followup` will measure anything real on it. `tune` asks the
endpoint and tells you:

```bash
uv run llm-assay.py tune http://localhost:8000/v1
```

```
Detected
  endpoint      http://endpoint.example.net:8000/v1  (vLLM 0.27.1)
  model         unsloth/Qwen3.8-27B  (served as MY-DEPLOYMENT)
  max ctx       262,144 tokens
  prefix cache  ENABLED -- cached-follow-up measurement is meaningful here
  spec decode   ACTIVE (accept len 2.65) -- expect wide decode spread; sample adaptively
  kv cache      463,910 tokens (~1.8 seqs at max ctx)
  thinking      reasoning_effort=none, enable_thinking=false
                inert here: reasoning_effort=low, reasoning_effort=medium, reasoning_effort=xhigh




Suggested runs

  # smoke -- is the endpoint alive and sane -- run this first
  uv run llm-assay.py --base-url http://endpoint.example.net:8000/v1 --model unsloth/Qwen3.8-27B \
      --served-model-name MY-DEPLOYMENT --pp 512 --tg 64 --exact-tg --depth 0 --runs 3 \
      --latency-mode generation --seed $RANDOM
  …
```

Where the numbers come from, and why they matter:

| Detected | Source | What it changes |
|---|---|---|
| model + served name | `/v1/models` `root` / `id` | Fills in `--model` and `--served-model-name` |
| `max ctx` | `/v1/models` `max_model_len` | Every suggested depth fits under it, with room for prompt and generation |
| `prefix cache` | `vllm:cache_config_info` label | `--measure-cached-followup` is only suggested when the server can actually honour it |
| `spec decode` | `vllm:spec_decode_*` counters | Decode presets switch to `--target-ci` sampling instead of a fixed `--runs` |
| `kv cache` | `kv_cache_size_tokens` | Concurrency levels are capped at what fits, so no request sits queued |
| thinking switches | `/v1/chat/completions/render` | Which of `reasoning_effort=none`, `enable_thinking=false` and the graded levels actually change the prompt |

Detection is best-effort. A server with no readable `/metrics` still gets
suggestions — built from conservative defaults and labelled as such — rather than
an error.

### Machine-readable detection

```bash
uv run llm-assay.py tune http://localhost:8000/v1 --json
```

Emits the detection and the suggested presets as JSON. Each preset carries both an
argv list you can exec directly and a rendered command string for logs. `null` means
*unknown* rather than *off* — a server with no readable `/metrics` reports
`"prefix_caching": null` and explains itself in `notes`. stdout is only ever JSON;
the report and any error go to stderr.

### Saving a runner

```bash
uv run llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh
./run-local.sh --help
./run-local.sh smoke
```

The generated script has the detected values already resolved, records what was
detected in a header comment, and tells you how to regenerate itself after the
server is restarted with different settings. It takes env overrides
(`SEED RUNS OUTDIR EXTRA TAG DRY_RUN BENCHY`) and saves timestamped JSON to
`results/`:

```
==> preset=smoke seed=4377
    result: results/smoke-20260820-003633.json
```

Presets are sized to the endpoint, so the list varies: `smoke` and `sweep` always
appear, `concurrency` only when more than one request fits the KV cache, and
`long` only when the model's context is deep enough for it to differ from the
sweep.

`--seed $RANDOM` is deliberate in the suggestions. A fixed seed sends
byte-identical prompts, so a repeat invocation is served from the prefix cache and
prefill inflates; see the prefix-caching notes above.
For a paired A/B, use the same fixed seed on both sides and prime once first.

### The trap: a fixed `--seed` warms the cache for the next run

`--seed` is what makes a paired comparison possible, and it does so by making corpus
sampling deterministic. The consequence is that **re-running the same command sends
byte-identical prompts**, so the second invocation is served from the prefix cache the
first one populated.

Measured on a 262k-context vLLM host, same shape, varying only whether that seed had
been used before:

| seed | invocation | cache hit rate | prefill t/s | est_ppt |
|---|---|---|---|---|
| 8801 | 1st | **0.0%** | 3,888 | 1185 ms |
| 8801 | 2nd | 69.4% | 9,707 | 475 ms |
| 8810 | 1st | **0.0%** | 3,823 | 1206 ms |
| 8810 | 2nd | 69.4% | 9,795 | 471 ms |
| 8810 | 3rd | 69.4% | 9,487 | 487 ms |

Prefill inflated ~2.5×, est_ppt halved, and it stays warm for every later run. `--no-cache` does **not** help — its cache-buster
is seed-derived, so it too is identical across invocations, and only the first use of
it is cold.

Run the same config twice and `compare` will tell you, with a significance verdict,
that it beat itself:

```
metric: pp_throughput   test: paired   alpha=0.01
d4096 pp512 tg32 inf                 3888.30         9707.13   +149.6%   0.0003  BETTER
```

llm-assay warns above a 10% hit rate in a plain run, so this is no longer silent —
and `--cold` prevents it outright, salting the corpus per invocation so every run
measures a genuine ingest.
Prefill and TTFT move most, but the ordering is still detectable in decode: a paired
compare of a config against *itself*, cold side versus warm side, has been observed to
report a significant `WORSE` at 3 runs per side. Prime with one throwaway invocation so
both sides are warm before comparing. To avoid it: compare decode
across invocations freely; for prefill or TTFT either prime with one throwaway run so
both sides are equally warm, or use a fresh `--seed` each time and accept an unpaired
test. `tune` suggests `--seed $RANDOM` for exactly this reason.

## Comparing two runs statistically

`llm-assay` characterises one endpoint. Comparing two *configurations* is a
different job, and eyeballing overlapping `mean ± std` bars is a poor way to do it —
especially with speculative decoding, where run-to-run spread is driven by
content-dependent acceptance and routinely exceeds the effect you are looking for.

```bash
# Run both sides with the SAME seed so the comparison can be paired
llm-assay --base-url … --seed 1234 --format json --save-result baseline.json
# … change one server flag …
llm-assay --base-url … --seed 1234 --format json --save-result candidate.json

uv run llm-assay.py compare baseline.json candidate.json
uv run llm-assay.py compare baseline.json candidate.json --metric e2e_ttft --alpha 0.01
```

Output:

```
metric: tg_throughput   test: paired   alpha=0.05

test                                baseline       candidate     delta        p  verdict
------------------------------------------------------------------------------------------------
d0 pp2048 tg32 inf                     65.10           60.69     -6.8%   0.0018  WORSE
d16384 pp2048 tg32 inf                 54.88           54.05     -1.5%   0.5396  no difference
```

- **Paired** test when both runs share a `--seed` (each run index saw identical
  text, so content variance cancels). **Welch's** unequal-variance test otherwise.
  Saved results carry a `corpus_fingerprint`, and a row whose two sides recorded
  different fingerprints falls back to Welch no matter what the seeds say — the
  seed is a proxy for "same text", and the fingerprint is the thing itself.
  `--unpaired` forces Welch even when the seeds match — use it when the two runs
  are not genuinely paired (different corpora, or `--cold` on either side, which
  `compare` refuses to pair anyway).
- Rows are keyed by depth, prompt size, generation size and phase, so a sweep over
  `--pp` or `--tg` compares shape against matching shape instead of collapsing.
- `--thinking` and `--reasoning-effort` are in that key too, so two files differing
  only in one of them share no rows and nothing is compared (which exits non-zero).
  To A/B the dimension itself, pass `--across thinking` or
  `--across reasoning_effort`, which drops it from the key so the arms line up:

  ```bash
  uv run llm-assay.py compare think-on.json think-off.json --across thinking
  ```

  That forces an **unpaired** test and says so. Both dimensions are part of the
  corpus key, so the two sides read different text and run *i* of each never saw
  the same content — pairing would have nothing to cancel. If a single file swept
  the dimension itself it has two rows per shape, and `compare` refuses rather than
  silently keeping one.
- Settings that differ between the two files are called out before the table —
  comparing across a change of `--exact-tg`, `--no-cache`, model, tokenizer, corpus
  or latency mode measures the setting, not the thing you meant to test:

  ```
  warning: --exact-tg differs between runs (base=False, cand=True)
  ```
- `--metric` selects `tg_throughput` (default), `pp_throughput`, `e2e_ttft`,
  `ttfr`, `est_ppt` or `peak_throughput`. Latency metrics are scored so that lower
  is `BETTER`.
- `untestable (n<2)` means the test never ran: a t-test needs at least two
  samples per side to have any spread to divide by. A one-run A/B reports this
  rather than `no difference`, which would hide an arbitrarily large gap behind
  a word that reads like a measurement. `--equivalence-margin` leaves the row
  untestable too, and marks `equivalence.testable: false` in the JSON, rather
  than claiming `EQUIVALENT` off a sample that cannot support it.
- "no difference" is **not** evidence of equivalence — it may simply be
  underpowered. Raise `--runs`, or use `--target-ci`.

### Exit codes

A benchmark that produced nothing must not report success — anything automated
reading `$?` would take an empty table for a healthy run.

| Situation | Exit code |
|---|---|
| Every shape measured | `0` |
| Some requests failed, but every shape still measured | `0`, with a `Failures:` summary |
| Any shape produced no measurement at all | `1` |
| Nothing measured, endpoint unreachable, or warmup failed | `1` |

Losing a run of a shape is a weaker measurement, not a missing one, so it does not
fail the suite. Losing a whole shape leaves a hole the table prints around, so it
does. Use `--exit-on-first-fail` to stop at the first failed request instead.

### Proving a change is harmless, not just "not proven harmful"

A t-test can only fail to reject "the difference is zero", so `no difference`
conflates two opposite situations: the change really is harmless, or you did not
measure enough. `--equivalence-margin` settles it with **TOST** (Two One-Sided
Tests): declare a margin, and the difference must be demonstrably inside it.

```bash
uv run llm-assay.py compare baseline.json candidate.json --equivalence-margin 0.03
```

`no difference` then splits into `EQUIVALENT` (demonstrably within ±3%) and
`INCONCLUSIVE` (too noisy to tell — raise `--runs` or use `--target-ci`).

### Using A/B as a CI gate

```bash
uv run llm-assay.py compare baseline.json candidate.json \
    --metric tg_throughput --fail-on-regression
```

Exits non-zero when any shape is significantly **worse**, so a perf regression can
block a merge or a deploy. **Use at least 3 runs per side**: a paired test divides by
the spread of the *differences*, so two runs that drifted together — which is what
speculative-decode acceptance does between invocations — read as certain at any shift
size. llm-assay warns when a significant verdict rests on fewer than 3 runs. Improvements never trip it, and direction is metric-aware
— for `e2e_ttft`, `ttfr` and `est_ppt`, lower is better. Add `--json` to get the
verdicts as data; both the table and the JSON come from one analysis pass, so they
cannot disagree.

### How many runs do you actually need?

Decode throughput on a speculative-decoding setup can easily have a sample standard
deviation of ~7% of the mean. At `--runs 3` the standard error is large enough that
two runs of an *unchanged* config can land 10% apart — and, worse, a small `±` at
n=3 reads as precision when it is a small-sample artifact.

llm-assay therefore:

- computes the **sample** standard deviation (`ddof=1`) and a **Student-t** 95%
  confidence interval (at n=3 the critical value is 4.30, not 1.96);
- prints a reliability warning naming any decode metric whose CI is too wide,
  together with the run count that would resolve it:

```
Statistical reliability warnings:
  d0: decode 64.67 t/s, 95% CI +/-8.6% at n=3 -- need ~9 runs for +/-5%
```

- and can reach a target automatically with `--target-ci 0.05`.

## Checking quality, not just speed (`probe`)

Throughput at 200k says nothing about whether the model can still *use* 200k, and
the features that make long context cheap — quantized KV, windowed attention,
chunked prefill — are the ones that can quietly cost comprehension. `probe`
generates a synthetic event log to a chosen depth, injects a known anomaly into
it, and checks the model still finds it.

```bash
uv run llm-assay.py probe http://localhost:8000/v1 --model my-model \
  --depths 4096 8192 32768 --trials 6 --thinking on
```

```
| depth | rung | at | sep | accuracy | 95% CI | partial | false+ | unanswered |
|------:|:-----|---:|----:|---------:|:-------|--------:|-------:|-----------:|
| 2048 | clean | - | - | 10/10 | [0.72, 1.00] | 0 | 0 | 0 |
| 2048 | outlier | 0.50 | - | 10/10 | [0.72, 1.00] | 0 | 0 | 0 |
| 2048 | log | 0.50 | 0.60 | 10/10 | [0.72, 1.00] | 0 | 0 | 0 |
```

`tune` suggests a probe sized to the endpoint it detected, and `--write` puts it in
the generated runner as its own preset — so the quality check travels with the
throughput suite rather than being something you have to remember.

No second model and no ground truth about any text are needed: the haystack and
the anomalies are both generated here, so a correct answer has to contain a
surname that exists in no training set. The answer is checked against the log's
own name dictionary and must match exactly one, so an answer that hedges across
several suspects fails rather than scoring on the one it happened to include.

-   `--depths N …`: context depths to probe (default `4096 32768`).
-   `--rungs R …`: which of `clean`, `outlier`, `log` to run (default: all).
    `clean` injects nothing and is the control — asking "is anything wrong?" primes a
    model to find something, and only these trials say how often it invents one.
    `outlier` injects a passage from another domain and is *meant* to be easy: it
    separates "cannot see the text at this depth" from "sees it and cannot reason
    about it". `log` injects the contradiction — one person in two places on one
    date — and is the measurement. All three draw the same generated haystack and
    are put the same question, which is what makes `clean` a control: a control
    drawn from a different distribution than the measurement measures a
    false-positive rate that does not transfer. Every line shares one template, so
    the needle cannot stand out by style, repetition or familiarity.
-   `--save-transcripts PATH`: write each trial's answer and reasoning trace to PATH.
    The accuracy number cannot tell a model that never saw the injection from one
    that saw it and judged it consistent; the trace can, and that distinction is
    what separated the rungs' failure modes.
-   `--positions F …`: where to inject, as fractions of the context (default `0.5`).
    Attention is strongly position-dependent, so `0.1 0.5 0.9` separates that effect
    from depth rather than letting it hide inside every other number. For `log` the
    position is the *centre* between the contradiction's two halves, and a pair
    centred at 0.9 cannot also span 0.6 of the log — so the span narrows
    symmetrically near the ends and the `sep` column reports how far apart they
    actually were. Position and distance cannot both be held fixed there; the table
    publishes both rather than letting one be read as the other.
-   `--trials N`: trials per (depth, rung) cell (default `6`). Deep trials are slow,
    so the default buys a wide interval; raise it for anything you intend to act on.
-   `--max-tokens N`: answer budget (default `40000`). A thinking model scans the
    whole log before answering, so this grows with depth. Too small a budget does
    not merely lose trials, it loses the *hard* ones — a trial the model struggles
    with reasons longer and hits the ceiling — so the reported accuracy rises. The
    tool says so out loud past 20% unanswered. The `clean` control is the most
    expensive rung, because proving absence means checking everything.
-   `--thinking on|off`, `--seed`, `--served-model-name`, `--json`, `--save-result`
    behave as they do elsewhere.

Accuracy is a **proportion**, so the interval is Wilson rather than the Student-t
used for throughput means: on a 0/1 rate a t-interval is wrong, and worst exactly
where a working probe sits — at 0/6 it would report ±0.

Measured on a 262k-context Qwen3.8-27B, same haystack and same question at each
depth — only what is asked of the model changes:

| depth | retrieve (`outlier`) | reason (`log`) | invent (`clean`) |
|---|---|---|---|
| 16,384 | 8/9 | 5/9 | 12% |
| 65,536 | 7/10 | 3/5 | 89% |
| 131,072 | 6/9 | **0/8** | 89% |
| 200,000 | **8/9** | **0/9** | **100%** |

**At 200k it retrieves and does not reason.** The model finds the injected
foreign line and names it in its own reasoning 8 times in 9, never finds a
logical contradiction, and invents one in every clean log it is shown
(retrieval vs reasoning, p=0.0004; retrieval does not decay across the range,
p=0.58).

A needle-in-a-haystack test is exactly the `outlier` rung — at 200k it would
report near-perfect recall and call this context healthy. The failure is not
silence but confident fabrication, and only the `clean` control makes it
visible.

**Always read `clean` beside any deep score**, and expect it to be the sharper
signal. Its false-positive rate breaks once and then saturates — 0% at 2k, 12% at
16k, **71% at 32k** (p=0.0056 against 16k), then 85%, 89%, 100%. That single step
is the tool's actual answer for this deployment: **trustworthy to ~16k,
confabulating by 32k.** Once the control saturates, an accuracy figure stops
meaning comprehension — it measures whether a fabrication landed on the right
name. Deep cells also lose trials the probe cannot score (a finished reasoning
trace with empty content — 50% of `log` trials at 65k), counted as unanswered
rather than guessed at.
With thinking off at depth 4096 it scores **0/10** against **6/10** on — but that
is a property of *this probe's answer format*, not of the model. Asked the same
question with working permitted, thinking off scores **9/10**, slightly ahead of
thinking on. The model needs *a* scratchpad and the think block is only one of
them; the probe's question demands one line containing only a surname, so closing
the block leaves it nowhere to work at all. `--thinking off` here measures "can it
do this with no scratchpad", which is a real question and not the same as
"thinking off costs long-context quality".

## Live progress stream (for external visualizers)

`--emit-progress PATH` writes a stream of newline-delimited JSON events to
`PATH` (or `-` for stdout) while the benchmark runs. External visualizers —
live TUIs, web dashboards, post-hoc analyzers — consume that stream and
render whatever they like.

```bash
# Emit to a file alongside the normal benchmark
llm-assay --base-url http://localhost:8000/v1 --model … \
             --emit-progress /tmp/progress.jsonl

# Pipe straight into a visualizer (status output goes to stderr in this mode)
llm-assay --base-url http://localhost:8000/v1 --model … \
             --emit-progress - | my-visualizer
```

Default behavior is unchanged when `--emit-progress` is omitted. Schema
spec: [`docs/progress-schema.md`](docs/progress-schema.md).

When the server returns `token_ids`, per-chunk `tokens.count` values are
exact. When token IDs are unavailable, `tokens` events are marked
`estimated: true` and should be treated as live progress hints; use
`request_end.total_tokens` as the authoritative generated-token total.

Thanks to [@alexziskind1](https://github.com/alexziskind1) for contributing
the progress stream functionality and reference visualizer integration.


## License and provenance

MIT. See [LICENSE](LICENSE).

llm-assay is derived from [eugr/llama-benchy](https://github.com/eugr/llama-benchy)
at commit `e9be344`, by Eugene Rakhmatulin, which is MIT licensed; that copyright
notice is retained unmodified. The benchmark loop, the streaming client and the
result aggregation are substantially his work, and roughly 83% of the upstream
source is still present here.

Added since the fork: seeded and paired corpus sampling; `compare`, for
paired/Welch significance testing between saved runs; `servermetrics`, which
scrapes prefix-cache and speculative-decode counters from vLLM and llama.cpp;
`tune`, for endpoint detection and run suggestion; `probe`, the long-context
quality probe; Student-t confidence intervals and adaptive sampling; corpus
fingerprinting; `--cold`, `--thinking` and `--reasoning-effort`; and an
exit-code contract. Packaging was removed -- upstream still ships a
`pyproject.toml`, so upstream install instructions do not apply here.
