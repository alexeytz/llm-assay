# llm-assay - llama-bench style benchmarking for any OpenAI-compatible LLM endpoint

Prompt-processing and decode speed at real context depths, against vLLM, SGLang or
llama.cpp, with confidence intervals, paired A/B significance testing, prefix-cache
trap detection, and a long-context quality probe.

Inspired by llama.cpp's [`llama-bench`](https://github.com/ggml-org/llama.cpp) and
[llama-benchy](https://github.com/eugr/llama-benchy).

## TL;DR

```bash
# 1. Install uv (macOS / Linux; for Windows see Setup below)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get llm-assay. There is no install step: uv resolves the dependencies on first run
git clone https://github.com/alexeytz/llm-assay.git
cd llm-assay

# 3. Point tune at your endpoint. It detects the model, context window, prefix
#    caching and speculative decoding, and suggests runs sized to that server
uv run llm-assay.py tune http://localhost:8000/v1

# 4. Save those runs as a script and start with the smoke test
uv run llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh
./run-local.sh smoke
```

`./run-local.sh --help` lists the other presets: a depth sweep, decode, concurrency,
the deepest context the server accepts, and `probe` for quality. Results land in
`results/` as JSON. From there, [`compare`](#comparing-two-runs-statistically) A/B
tests two saved runs, and [Usage](#usage) covers every flag.

## Motivation

*Kept as written in [llama-benchy](https://github.com/eugr/llama-benchy)'s README;
the "I" below is its author.*

`llama-bench` is a CLI tool that is a part of a very popular [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine. It is widely used in LLM community to benchmark models and allows to perform measurement at different context sizes.
However, it is available only for llama.cpp and cannot be used with other inference engines, like vllm or SGLang.

Also, it performs measurements using the C++ engine directly which is not representative of the end user experience which can be quite different in practice.

vLLM has its own powerful benchmarking tool, but while it can be used with other inference engines, there are a few issues:

- It's very tricky and even impossible to calculate prompt processing speeds at different context lengths. You can use `vllm bench sweep serve`, but it only works well with vLLM with prefix caching disabled on the server. Even with random prompts it will reuse the same prompt between multiple runs which will hit the cache in `llama-server` for instance. So you will get very low median TTFT times and very high prompt processing speeds. 
- The TTFT measurement it uses is not actually until the first usable token, it's until the very first data chunk from the server which may not contain any generated tokens in /v1/chat/completions mode.
- Random dataset is the only ones that allows to specify an arbitrary number of tokens, but randomly generated token sequence doesn't let you adequately measure speculative decoding/MTP.

As of January 2nd, 2026, I wasn't able to find any existing benchmarking tool that brings llama-bench style measurements at different context lengths to any OpenAI-compatible endpoint.

### Why llm-assay

It started with an itch: two RTX 3090s and the NVLink bridge on the side that
would not let me sleep. Reddit said it helps a lot. Every LLM I asked said the
same. Actual numbers? Nobody had any. Plenty of confident opinions, no data.

So I needed a tool I could test and trust. Cloned [llama-benchy](https://github.com/eugr/llama-benchy) as the base
and rebuilt it - my way or the llm-assay.

The first lesson: the obvious way to measure will lie to you with a straight
face. `NCCL_P2P_DISABLE=1` does not actually turn NVLink off. And an A/B test of
a config against itself picked a winner: same command, run twice, and `compare`
reported the second one +110% on prefill, p = 0.0000, BETTER. It was served from
the prefix cache the first run had warmed up, for free. Even with the cache out
of the way, two runs of an unchanged config at `--runs 3` on a speculative
decoding endpoint routinely land 10% apart on decode... and a tight-looking
`±` says nothing about it. Hence the confidence intervals, the paired tests,
and a tool that reads the server's own counters before believing a word it
says.

Then a different question moved in: how much can you rely on an LLM you only
reach through an endpoint? It came out of a discussion about an AI agent that
crashed mid-refund, came back with no record of it, and refunded again. Double
refund. Which weights, which precision, which settings, how long it was allowed
to think - none of that is yours to see. So `probe` hands a model a log where
the right answer is known, and asks what is wrong with it. The verdict is in
[`docs/fabrication-programme/fabrication-findings.md`](docs/fabrication-programme/fabrication-findings.md):
up to about 4,000 tokens it is boring, in the good way. Past that, you get a
sample.

Measure your own endpoint.

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
- Scrapes the server's Prometheus `/metrics` for prefix-cache hit rate and speculative-decode acceptance - vLLM counters diffed per shape, SGLang-style gauges reported as server-lifetime and labelled as such, and an unrecognised endpoint says so rather than disabling silently. Warns in both directions: when a cached-follow-up measurement got no cache hits, and when an ordinary run was unexpectedly served *from* the cache.
- Records what the **server** says it is serving, not just what was asked for.
  `--model` and `--served-model-name` are both the caller's side of the
  conversation, and an alias survives a reload that changes the weights
  underneath it. Every saved result carries `served_model` (the endpoint's own
  answer from `/v1/models`) and, where the engine publishes it, `served_build`
  - llama.cpp's `/props` names the GGUF, the quantisation and the server build.
  A missing value reads as *unknown*, never as "the same build".
- `llm-assay.py compare` for paired / Welch significance testing between saved runs.
- `llm-assay.py probe` for a **quality** check to sit beside the speed numbers: it
  generates a synthetic event log to a chosen depth, injects a known anomaly into it,
  and measures whether the model still finds it. It needs no corpus and no second
  model. Throughput at 200k says nothing about
  whether the model can *use* 200k, and the features that make long context fast
  (quantized KV, windowed attention, chunked prefill) are the ones that can cost
  comprehension. Scored without a second model: a correct answer must name a surname
  that exists in no training set.
- Can save results to file in Markdown, JSON, or CSV format.
- Can save granular time-series data for token generation when JSON output is used (`--save-total-throughput-timeseries` and `--save-all-throughput-timeseries`).
- Runs a coherence test after warmup to verify model responds correctly (default, can be skipped with `--skip-coherence`).
- Auto-detects HuggingFace model name from the endpoint's `/models` endpoint when `--model` is not specified.

## How it differs from upstream

llm-assay started from [llama-benchy](https://github.com/eugr/llama-benchy) at
`e9be344`. Its core is still upstream's: the streaming client, prompt
construction, the result tables and the progress stream grew out of that code.
Every upstream flag still works (`--enable-prefix-caching` is accepted under
its new name, `--measure-cached-followup`), and none was removed. What changed:

**Measurement and statistics.**
- Sample standard deviation (`ddof=1`) with a Student-t 95% confidence interval,
  where upstream reports the population standard deviation and no interval.
  At three runs the t multiplier is 4.30, not 1.96.
- `--target-ci` samples until the interval is tight enough (`--max-runs` caps
  it), and underpowered runs are flagged with the run count that would resolve
  them.
- `--seed` makes corpus sampling deterministic and keyed to each shape, so two
  configurations read identical text and `compare` can run a paired test.
  Upstream picks the slice with an unseeded random start. `--cold` adds a
  per-process salt so repeated runs never share a cached prefix.
- `--reasoning-effort` and `--thinking on|off` as swept dimensions;
  `--endpoint completions` for the raw `/v1/completions` route, which skips the
  chat template; `--stall-timeout`, a configurable coherence check, `--stats`
  and `--legend`.

**Knowing what the server actually did.**
- Reads the server's Prometheus `/metrics` (vLLM, llama.cpp, SGLang) for
  prefix-cache hit rate and speculative-decode acceptance, per shape, and per
  phase in the cached-follow-up mode.
- Warns when a cached-follow-up measurement got no cache hits, when an ordinary
  run was served from a cache an earlier run filled (the trap where the same
  command run twice reports a large "improvement" against itself), and when
  another client's traffic shares the measurement window.
- Saved results record what the server says it is serving (`served_model`,
  `served_build`), a fingerprint of the text each shape read, error messages
  from failed requests, any tokenizer fallback, and the full invocation, so two
  result files can be checked for comparability.

**Honest output.**
- A run that measured nothing, or lost a whole shape, exits non-zero;
  upstream exits 0 unless interrupted.
- `peak t/s` is omitted when the server delivers tokens without timing to
  derive a rate from, rather than printing an artifact.

**New subcommands.**
- `compare`: paired or Welch significance testing between saved runs, with an
  equivalence test for proving a change harmless, usable as a CI gate.
- `tune`: detects the model, context window, prefix caching, speculative
  decoding, KV-cache capacity and which thinking switches are live, then
  suggests runs sized to that server and can write them as a script.
- `probe`: a long-context quality check, with the study it was built for in
  [`docs/fabrication-programme/fabrication-findings.md`](docs/fabrication-programme/fabrication-findings.md).

**Packaging.** Upstream installs as a package with its own console
command. llm-assay is a single `llm-assay.py` carrying its own dependency list
(PEP 723), run with `uv` and nothing to install.

# Current Limitations

- Assumes an OpenAI-compatible API. `--endpoint chat` (the default) uses
  `/v1/chat/completions`; `--endpoint completions` uses raw `/v1/completions`,
  which skips the chat template and so separates engine cost from template cost.
- Most "non-standard" endpoints only differ in their prefix, which `--base-url`
  already handles. A gateway at
  `https://gw.example.net/acct/gw/openai/chat/completions` just needs
  `--base-url https://gw.example.net/acct/gw/openai`.

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
  --base-url http://localhost:8000/v1 \
  --model cyankiwi/Qwen3.8-27B-AWQ-FP8 \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation
```

Output:


| model                        |            test |             t/s |      peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-----------------------------|----------------:|----------------:|--------------:|-----------------:|-----------------:|-----------------:|
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |          pp2048 | 2124.54 ± 25.36 |               |  1046.17 ± 11.33 |   964.38 ± 11.33 |  1046.17 ± 11.33 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |            tg32 |    85.59 ± 8.36 |  88.35 ± 8.63 |                  |                  |                  |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  pp2048 @ d4096 |  2081.16 ± 5.93 |               |   3034.50 ± 8.40 |   2952.70 ± 8.40 |   3034.50 ± 8.40 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |    tg32 @ d4096 |   75.21 ± 17.20 | 77.64 ± 17.75 |                  |                  |                  |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  pp2048 @ d8192 |  2037.53 ± 4.45 |               |  5108.16 ± 10.90 |  5026.36 ± 10.90 |  5108.16 ± 10.90 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |    tg32 @ d8192 |    71.17 ± 0.59 |  73.47 ± 0.61 |                  |                  |                  |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d16384 |  1950.05 ± 3.99 |               |  9534.42 ± 19.79 |  9452.62 ± 19.79 |  9534.42 ± 19.79 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d16384 |   83.18 ± 14.50 | 85.86 ± 14.97 |                  |                  |                  |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d32768 |  1816.41 ± 2.51 |               | 19249.80 ± 26.48 | 19168.00 ± 26.48 | 19249.80 ± 26.48 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d32768 |    72.15 ± 7.08 |  74.48 ± 7.30 |                  |                  |                  |

llm-assay (0.5.0-dev+g0896539)
date: 2026-09-12 08:52:09 | latency mode: generation

-------

It's recommended to use "generation" latency mode to get prompt processing speeds closer to real numbers, especially on shorter prompts.
By default, the script adapts the prompt size to match the specified value, regardless of the chat template applied. Use `--no-adapt-prompt` to disable this behavior.

Within a single run the probability of accidental cache hits is small, so you rarely need to disable prompt caching on the server. **Across** runs it is not small at all: a fixed `--seed` sends byte-identical prompts, so a second invocation of the same command is served warm and prefill inflates - see [the trap section](#the-trap-a-fixed---seed-warms-the-cache-for-the-next-run) and `--cold`, which prevents it. You can add `--no-cache` that will add some random noise if you get cache hits.

### Arguments

-   `--base-url`: OpenAI compatible endpoint URL (Required).
-   `--api-key`: API Key (Default: "EMPTY").
-   `--model`: Model name to use for benchmarking. If not specified, attempts to auto-detect from the endpoint's `/models` endpoint.
-   `--served-model-name`: Model name used in API calls (Defaults to --model if not specified). Tries to autodetect from the endpoint's `/models` endpoint (if supported, e.g. vLLM).
-   `--tokenizer`: HuggingFace tokenizer name or local path (Defaults to model name).
-   `--pp`: List of prompt processing token counts (Default: [2048]).
-   `--tg`: List of token generation counts (Default: [32]).
-   `--endpoint {chat,completions}`: Which route to benchmark (default `chat`). `completions` uses raw `/v1/completions` - no chat template, context and prompt concatenated - which reaches stacks that never implemented the chat route and isolates engine cost from template cost. Visible in the warmup: template overhead measures 9 tokens on the chat route and 0 on completions.
-   `--exact-tg`: Force output length to match `--tg` by sending `min_tokens=<tg>` and `ignore_eos=true` in benchmark requests. This is useful for fixed-OSL throughput runs on compatible servers such as vLLM.

    > **Caveat for speculative decoding.** `ignore_eos` makes the model continue past its natural stopping point, and long forced generations tend to degenerate into repetition. Repetitive text is unusually easy for a draft head to predict, so `--exact-tg` can *inflate* measured acceptance relative to real use. It buys determinism in output length at the cost of realism in acceptance; keep that in mind when benchmarking MTP/Eagle setups.
-   `--depth`: List of context depths (Default: [0]).
-   `--runs`: Number of runs per test (Default: 3).
-   `--warmup-runs`: Number of discarded warmup runs per test shape (Default: 1). For concurrency `N`, each warmup run sends `N` requests. Also controls the number of discarded warmup probes for `--latency-mode generation`; it does not affect the initial prompt-adaptation warmup.
-   `--cold`: Guarantee a cold prefix cache by salting the corpus offsets per invocation, so a repeat run cannot be served from the cache the previous one populated. Measured: without it a second invocation of the same command sees 69.4% hits, with it every invocation sees 0.0%. `--no-cache` cannot do this - prefix caching matches the *prefix*, and its buster is appended to the end. Two `--cold` runs read different text, so `compare` will not pair them.
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
-   `--legend`: Print an explanation of the table beneath it. Every entry leads with a plain sentence - `pp512` is *"how fast the model reads the prompt; it was handed a 512-token prompt (roughly 380 words of English)"* - and puts the mechanics one indent further in, so a first-time reader and someone who wants the formula are both served. Covers the `pp`/`tg`/`ctx_pp`/`@ d` row labels, the `(cN)`/`(think=)`/`(re=)` suffixes, and how `ttfr`, `est_ppt` and `e2e_ttft` differ. Only the rows and columns actually printed are described, so it never explains a `t/s (req)` column a `c1` table does not have. Off by default; with `--format md` it is appended to the saved file, fenced so the layout survives.
-   `--stats {std,ci,median}`: What the `±` column means - `std` (sample standard deviation, default), `ci` (95% confidence interval of the mean), or `median` (median ± IQR/2). Use `ci` when deciding whether a difference between two runs is real.

    **This chooses the display only.** Every statistic is computed and saved
    whichever you pick: `mean`, `std`, `median`, `iqr`, `ci95`, `n` and the raw
    per-run `values[]` are all in the JSON regardless. And every decision the
    tool makes is mean-based - `compare` runs paired-t and Welch on means,
    `--target-ci` chases the interval on the mean, and `reliability_notes()`
    reasons about that same interval. So `--stats median` changes what you read
    and nothing the tool concludes; switching to it will not change an A/B
    result.
-   `--no-server-metrics`: Disable scraping of the server's Prometheus `/metrics` endpoint. By default llm-assay records prefix-cache hit rate and speculative-decode acceptance length per test shape, and warns when a cached-follow-up measurement got no cache hits.
-   `--thinking on|off`: Sweep thinking as a benchmark dimension; rows are labelled `(think=on)`/`(think=off)`. `off` sends both spellings servers honour - `reasoning_effort: "none"` and `chat_template_kwargs.enable_thinking: false` - because deployments honour different ones. Measured on a Qwen3.8 served by vLLM: prefill unchanged, decode **48.50 t/s thinking vs 38.15 t/s not**, tracking speculative acceptance 3.03 vs 2.36 - reasoning text is repetitive, so the draft head predicts it better and each token is cheaper.
-   `--reasoning-effort`: One or more reasoning effort levels to benchmark (e.g. `--reasoning-effort none low high`). Each level runs the full suite and results are labelled with it (e.g. `tg1024 (re=none)`). Sent as a top-level `reasoning_effort` field. Default: not sent.

    > **Check the levels do anything before sweeping them.** Whether a graded level has an effect depends on the deployment's chat template, not the model card. On an unsloth conversion of Qwen3.8 served by vLLM, `low`, `medium` and `xhigh` render a *byte-identical* prompt - only `none` differs, by prefilling an empty `<think></think>` block - so sweeping them runs the suite repeatedly over a single configuration. `llm-assay.py tune <url>` probes the endpoint and reports which switches are live.
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
> Qwen3-Next and Qwen3.5 - `--enable-prefix-caching` is required there, and without
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
  --base-url http://localhost:8000/v1 \
  --model cyankiwi/Qwen3.8-27B-AWQ-FP8 \
  --depth 0 4096 8192 16384 32768 \
  --latency-mode generation \
  --enable-prefix-caching
```

Output:


| model                        |            test |             t/s |      peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:-----------------------------|----------------:|----------------:|--------------:|----------------:|----------------:|----------------:|
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |          pp2048 | 2114.72 ± 24.74 |               | 1052.42 ± 11.57 |  968.86 ± 11.57 | 1052.42 ± 11.57 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |            tg32 |    73.30 ± 3.31 |  75.66 ± 3.42 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  ctx_pp @ d4096 |  2092.53 ± 6.52 |               |  2041.65 ± 6.60 |  1958.09 ± 6.60 |  2041.65 ± 6.60 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  ctx_tg @ d4096 |    72.23 ± 6.82 |  74.56 ± 7.04 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  pp2048 @ d4096 | 1127.94 ± 17.50 |               | 1899.55 ± 27.94 | 1815.99 ± 27.94 | 1899.55 ± 27.94 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |    tg32 @ d4096 |    79.87 ± 7.83 |  82.45 ± 8.08 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  ctx_pp @ d8192 |  2042.64 ± 2.05 |               |  4095.04 ± 4.03 |  4011.48 ± 4.03 |  4095.04 ± 4.03 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  ctx_tg @ d8192 |    75.08 ± 4.08 |  77.50 ± 4.21 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |  pp2048 @ d8192 |  1100.08 ± 1.94 |               |  1945.25 ± 3.29 |  1861.69 ± 3.29 |  1945.25 ± 3.29 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |    tg32 @ d8192 |    65.97 ± 3.84 |  68.09 ± 3.96 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_pp @ d16384 |  1956.85 ± 2.09 |               |  8457.41 ± 8.69 |  8373.85 ± 8.69 |  8457.41 ± 8.69 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_tg @ d16384 |   61.38 ± 11.93 | 63.36 ± 12.32 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d16384 |  1035.34 ± 2.46 |               |  2061.67 ± 4.69 |  1978.10 ± 4.69 |  2061.67 ± 4.69 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d16384 |   79.84 ± 11.98 | 82.41 ± 12.37 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_pp @ d32768 |  1821.29 ± 0.85 |               | 18076.13 ± 8.21 | 17992.56 ± 8.21 | 18076.13 ± 8.21 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_tg @ d32768 |    82.05 ± 8.21 |  84.70 ± 8.47 |                 |                 |                 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d32768 |   986.29 ± 6.57 |               | 2160.09 ± 13.78 | 2076.52 ± 13.78 | 2160.09 ± 13.78 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d32768 |   88.41 ± 13.67 | 91.27 ± 14.11 |                 |                 |                 |

llm-assay (0.5.0-dev+g0896539)
date: 2026-09-12 08:54:56 | latency mode: generation

### Combining multiple parameters

You can specify multiple parameters for `--depth`, `--pp`, `--tg` and `--concurrency`. The benchmarks will run using the following hierarchy: depth -> pp -> tg -> concurrency.

```bash
llm-assay \
  --base-url http://localhost:8000/v1 \
  --model cyankiwi/Qwen3.8-27B-AWQ-FP8 \
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
  --base-url http://localhost:8000/v1 \
  --model cyankiwi/Qwen3.8-27B-AWQ-FP8 \
  --depth 0 4096 \
  --latency-mode generation \
  --enable-prefix-caching \
  --concurrency 1 2
```

Output:

| model                        |                test |     t/s (total) |        t/s (req) |     peak t/s |   peak t/s (req) |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:-----------------------------|--------------------:|----------------:|-----------------:|-------------:|-----------------:|------------------:|------------------:|------------------:|
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |         pp2048 (c1) | 2102.53 ± 24.03 |  2102.53 ± 24.03 |              |                  |   1055.26 ± 11.21 |    974.63 ± 11.21 |   1055.26 ± 11.21 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |           tg32 (c1) |    83.27 ± 0.77 |     83.27 ± 0.77 | 85.96 ± 0.79 |     85.96 ± 0.79 |                   |                   |                   |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |         pp2048 (c2) | 1965.05 ± 30.91 | 1465.11 ± 530.95 |              |                  |  1633.06 ± 515.14 |  1552.43 ± 515.14 |  1633.06 ± 515.14 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |           tg32 (c2) |   48.37 ± 10.33 |    50.52 ± 27.87 | 60.67 ± 2.08 |    53.69 ± 26.96 |                   |                   |                   |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_pp @ d4096 (c1) |  2080.09 ± 2.50 |   2080.09 ± 2.50 |              |                  |    2050.74 ± 2.37 |    1970.11 ± 2.37 |    2050.74 ± 2.37 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_tg @ d4096 (c1) |    74.76 ± 3.69 |     74.76 ± 3.69 | 77.17 ± 3.81 |     77.17 ± 3.81 |                   |                   |                   |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d4096 (c1) |  1114.17 ± 3.81 |   1114.17 ± 3.81 |              |                  |    1918.78 ± 6.28 |    1838.15 ± 6.28 |    1918.78 ± 6.28 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d4096 (c1) |    68.72 ± 7.41 |     68.72 ± 7.41 | 70.93 ± 7.65 |     70.93 ± 7.65 |                   |                   |                   |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_pp @ d4096 (c2) |  1978.56 ± 2.68 | 1543.85 ± 585.84 |              |                  | 3097.23 ± 1144.70 | 3016.60 ± 1144.70 | 3097.23 ± 1144.70 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | ctx_tg @ d4096 (c2) |    24.39 ± 0.44 |    40.93 ± 31.17 | 58.00 ± 1.00 |    48.82 ± 25.09 |                   |                   |                   |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 | pp2048 @ d4096 (c2) |  1083.72 ± 2.04 |  742.75 ± 207.07 |              |                  |  3028.92 ± 821.94 |  2948.29 ± 821.94 |  3028.92 ± 821.94 |
| cyankiwi/Qwen3.8-27B-AWQ-FP8 |   tg32 @ d4096 (c2) |    32.46 ± 0.02 |    46.28 ± 32.32 | 61.00 ± 0.00 |    53.95 ± 26.60 |                   |                   |                   |

llm-assay (0.5.0-dev+g0896539)
date: 2026-09-12 08:58:06 | latency mode: generation

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
  endpoint      http://localhost:8000/v1  (vLLM 0.27.1)
  model         unsloth/Qwen3.8-27B  (served as MY-DEPLOYMENT)
  max ctx       262,144 tokens
  prefix cache  ENABLED -- cached-follow-up measurement is meaningful here
  spec decode   ACTIVE (accept len 2.65) -- expect wide decode spread; sample adaptively
  kv cache      463,910 tokens (~1.8 seqs at max ctx)
  thinking      reasoning_effort=none, enable_thinking=false
                inert here: reasoning_effort=low, reasoning_effort=medium, reasoning_effort=xhigh




Suggested runs

  # smoke -- is the endpoint alive and sane -- run this first
  uv run llm-assay.py --base-url http://localhost:8000/v1 --model unsloth/Qwen3.8-27B \
      --served-model-name MY-DEPLOYMENT --pp 512 --tg 64 --exact-tg --depth 0 --runs 3 \
      --latency-mode generation --seed $RANDOM
  …
```

Where the numbers come from, and why they matter:

| Detected | Source | What it changes |
|---|---|---|
| model + served name | `/v1/models` `root` / `id` | Fills in `--model` and `--served-model-name` |
| `build` | llama.cpp's `/props` (`model_path`, `model_ftype`, `build_info`) | Names the weights behind a pinned alias, which `/v1/models` cannot |
| `max ctx` | `/v1/models` `max_model_len`; else llama.cpp's `/props` slot `n_ctx`, else `meta.n_ctx_train` | Every suggested depth fits under it, with room for prompt and generation |
| `prefix cache` | `vllm:cache_config_info` label; else cached-token counters | `--measure-cached-followup` is only suggested when the server can actually honour it |
| `spec decode` | `vllm:spec_decode_*` / `llamacpp:spec_decode_*` counters | Decode presets switch to `--target-ci` sampling instead of a fixed `--runs` |
| `kv cache` | `kv_cache_size_tokens` | Concurrency levels are capped at what fits, so no request sits queued |
| thinking switches | `/v1/chat/completions/render` | Which of `reasoning_effort=none`, `enable_thinking=false` and the graded levels actually change the prompt |

Every suggested run carries the consequence of that detection with it, in the
console output and in the generated runner, because the fact and the commands
were previously in two places and the connection had to be supplied by the
reader:

```
Suggested runs

  prefix cache is ENABLED here, and two things below follow from it.
  Every preset uses --seed $RANDOM: a fixed seed sends byte-identical
  prompts, so a repeat invocation is served from the cache the previous
  one filled. …
  And --measure-cached-followup is included where it belongs, because
  this server can actually honour it. Only prefill and TTFT move;
  decode numbers are unaffected either way.
```

With caching off it says so instead, and explains why no cached-follow-up
preset appears. With the status unknown it tells you how to settle it: run the
smoke preset twice with a **fixed** seed and compare prefill - a large jump on
the second run is the cache, not the server.

**Only vLLM states prefix caching as a fact.** `cache_config_info` carries an
`enable_prefix_caching` label, and that is the one place the status is asserted
rather than inferred. Other engines publish the *evidence* instead: llama.cpp
counts prompt tokens it served from cache, so a counter above zero proves
caching is both enabled and working - which is what you need before trusting
`--measure-cached-followup`. A counter reading zero is reported as **unknown,
not off**, because it is cumulative and a freshly restarted server has simply
not been hit yet. An engine publishing neither says so, and says it without
blaming the URL.

To settle it by hand on any engine: send one prompt twice and diff the
cached-token counters. The benchmark already does exactly that per shape, which
is what both prefix-cache warnings are built on.

Detection is best-effort. A server with no readable `/metrics` still gets
suggestions - built from conservative defaults and labelled as such - rather than
an error.

### Machine-readable detection

```bash
uv run llm-assay.py tune http://localhost:8000/v1 --json
```

Emits the detection and the suggested presets as JSON. Each preset carries both an
argv list you can exec directly and a rendered command string for logs. `null` means
*unknown* rather than *off* - a server with no readable `/metrics` reports
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

Prefill inflated ~2.5×, est_ppt halved, and it stays warm for every later run. `--no-cache` does **not** help - its cache-buster
is seed-derived, so it too is identical across invocations, and only the first use of
it is cold.

Run the same config twice and `compare` will tell you, with a significance verdict,
that it beat itself:

```
metric: pp_throughput   test: paired   alpha=0.01
d4096 pp512 tg32 inf                 3888.30         9707.13   +149.6%   0.0003  BETTER
```

llm-assay warns above a 10% hit rate in a plain run, so this is no longer silent -
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
different job, and eyeballing overlapping `mean ± std` bars is a poor way to do it -
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
  different fingerprints falls back to Welch no matter what the seeds say - the
  seed is a proxy for "same text", and the fingerprint is the thing itself.
  `--unpaired` forces Welch even when the seeds match - use it when the two runs
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
  the same content - pairing would have nothing to cancel. If a single file swept
  the dimension itself it has two rows per shape, and `compare` refuses rather than
  silently keeping one.
- Settings that differ between the two files are called out before the table -
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
- "no difference" is **not** evidence of equivalence - it may simply be
  underpowered. Raise `--runs`, or use `--target-ci`.

### Exit codes

A benchmark that produced nothing must not report success - anything automated
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
`INCONCLUSIVE` (too noisy to tell - raise `--runs` or use `--target-ci`).

### Using A/B as a CI gate

```bash
uv run llm-assay.py compare baseline.json candidate.json \
    --metric tg_throughput --fail-on-regression
```

Exits non-zero when any shape is significantly **worse**, so a perf regression can
block a merge or a deploy. **Use at least 3 runs per side**: a paired test divides by
the spread of the *differences*, so two runs that drifted together - which is what
speculative-decode acceptance does between invocations - read as certain at any shift
size. llm-assay warns when a significant verdict rests on fewer than 3 runs. Improvements never trip it, and direction is metric-aware
- for `e2e_ttft`, `ttfr` and `est_ppt`, lower is better. Add `--json` to get the
verdicts as data; both the table and the JSON come from one analysis pass, so they
cannot disagree.

### How many runs do you actually need?

Decode throughput on a speculative-decoding setup can easily have a sample standard
deviation of ~7% of the mean. At `--runs 3` the standard error is large enough that
two runs of an *unchanged* config can land 10% apart - and, worse, a small `±` at
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
the features that make long context cheap - quantized KV, windowed attention,
chunked prefill - are the ones that can quietly cost comprehension. `probe`
generates a synthetic event log to a chosen depth, injects a known anomaly into
it, and checks the model still finds it.

```bash
uv run llm-assay.py probe http://localhost:8000/v1 --model my-model \
  --depths 4096 8192 32768 --trials 6 --thinking on
```

```
| depth | rung | at | sep | accuracy | 95% CI | partial | false+ | fabricated | unanswered |
|------:|:-----|---:|----:|---------:|:-------|--------:|-------:|-----------:|-----------:|
| 2048 | clean | - | - | 10/10 | [0.72, 1.00] | 0 | 0 | - | 0 |
| 2048 | outlier | 0.50 | - | 10/10 | [0.72, 1.00] | 0 | 0 | - | 0 |
| 2048 | log | 0.50 | 0.60 | 10/10 | [0.72, 1.00] | 0 | 0 | - | 0 |
```

`tune` suggests a probe sized to the endpoint it detected, and `--write` puts it in
the generated runner as its own preset - so the quality check travels with the
throughput suite rather than being something you have to remember.

What this subcommand found, across eight models and twenty-one pre-registered
arms, is written up in
[`docs/fabrication-programme/fabrication-findings.md`](docs/fabrication-programme/fabrication-findings.md):
fabrication rises with document depth, detection falls, a low fabrication rate
on its own means nothing, and every rule, exclusion and error is in the data
appendix at the end of it.

### What the model is actually shown

The haystack is generated, not sampled from anything. Every line is one template
filled from five fixed vocabularies with a seeded RNG:

```
[{date}] At {location}, {rank} {name} {action} {object}.
```
```
[2142-09-12] At Harbour Vault, Engineer Vasquez monitored the plasma conduits.
[2142-06-25] At Sector Four, Steward Vorst inspected the coolant loop.
[2142-09-12] At Harbour Vault, Engineer Vasquez tested the plasma conduits.
```

5 ranks, 10 locations, 10 actions, 9 objects, 50 invented surnames, and dates
across one fictional year. Lines are added until the log reaches the requested
depth **measured with the real tokenizer**, then trimmed - built to length rather
than sliced to it, so no entry is cut in half at either end.

Three properties make this worth doing rather than lifting text from a book:

-   **The surnames exist in no training set**, which is what lets scoring be
    judge-free. A correct answer must name one of them, checked against the log's
    own dictionary and required to match exactly one - so a reply hedging across
    several suspects fails rather than scoring on the one it happened to include.
    The model cannot supply the answer from parametric memory.
-   **The needle is drawn from the same distribution as the haystack.** Every line
    shares one template, so an injection cannot stand out by style, vocabulary or
    familiarity. Three earlier rungs injected authored sentences into a real book
    and were deleted for exactly this: they measured how eye-catching the injected
    prose was. `docs/deprecated-rungs.md` records the experiments that showed it.
-   **The generator guarantees the answer.** Background entries are constrained so
    every `(name, date)` resolves to exactly one location, so the only impossibility
    is the injected one and a "wrong" answer is genuinely wrong. It also plants
    **benign same-name, same-date repeats at a single location** - deliberately,
    because without them "the name that appears twice on one date" would solve the
    task without ever reading a location. A generated 8k log runs ~308 lines with
    ~42 such repeats and zero two-location pairs.

Those repeats are also what fabrications are built out of: the documented failure
mode is the model taking a real repeat and re-dating or re-locating one half, so
it acquires the second location a contradiction needs.

This is why `clean` is the expensive rung - proving absence means checking
everything, and a model that cross-checks exhaustively can spend tens of
thousands of reasoning tokens per trial at modest depths.

No second model and no ground truth about any text are needed: the haystack and
the anomalies are both generated here, so a correct answer has to contain a
surname that exists in no training set. The answer is checked against the log's
own name dictionary and must match exactly one, so an answer that hedges across
several suspects fails rather than scoring on the one it happened to include.

-   `--depths N …`: context depths to probe (default `4096 32768`).
-   `--rungs R …`: which of `clean`, `outlier`, `log` to run (default: all).
    `clean` injects nothing and is the control - asking "is anything wrong?" primes a
    model to find something, and only these trials say how often it invents one.
    `outlier` injects a passage from another domain and is *meant* to be easy: it
    separates "cannot see the text at this depth" from "sees it and cannot reason
    about it". `log` injects the contradiction - one person in two places on one
    date - and is the measurement. All three draw the same generated haystack and
    are put the same question, which is what makes `clean` a control: a control
    drawn from a different distribution than the measurement measures a
    false-positive rate that does not transfer. Every line shares one template, so
    the needle cannot stand out by style, repetition or familiarity.
-   `--reasoning-effort LEVEL`: send a top-level `reasoning_effort` and record
    it as `reasoning_effort` in the result (an `--extra-body` value wins, being
    merged last, and is recorded the same way - either a top-level
    `reasoning_effort` or a router's `reasoning: {"effort": ...}`). Unset means the vendor's
    default, and defaults differ: grok-4.3's is `low`, and its 0% false-positive
    rate at d131,072 became 20% at `high` on the same haystacks, so two results
    run at different settings - the default counting as a setting - are not a
    comparison of models. Cannot be combined with `--thinking off`, which
    already sends `none`.
-   `--allow-binding-budget`: run even when `--max-tokens` cannot fit under the
    endpoint's ceiling at some depth. Before the first trial the probe reads the
    context window (vLLM's `max_model_len`, a router's `context_length`, or
    llama.cpp's `n_ctx`) and a router's declared completion cap, and refuses a
    budget that cannot fit: such a request is rejected or silently clamped. A
    budget within 80% of the binding ceiling is allowed but warned about,
    because it has no room to grow if trials truncate - v24 set its budget
    *equal* to the provider's cap and lost 15 of 25 cells, and v32 used 87% of
    the window at its deepest depth and lost 11 of 17.
-   `--max-spend USD`: stop before the next trial once the cost the endpoint
    reports in `usage.cost` reaches this many dollars. Checked between trials,
    never during one, so a stop leaves a smaller cell rather than money spent
    with nothing recorded. An endpoint whose pre-flight reply carries no cost
    refuses the run (exit 3), because a cap the tool cannot measure is no cap.
    The result records `max_spend`, `spent`, and `halted` - why the run stopped
    early, or `null` - and a cell cut short records the trials it actually ran.
-   `--calibrate-depth`: size each depth in the endpoint's own tokens. Hosted
    haystacks are otherwise sized with a local tokenizer, and the log is
    date-dense, so a vendor whose tokenizer merges digits sees far fewer tokens
    than the label says - grok's billing implied about half. Two `max_tokens 1`
    requests per depth (the question alone, and with a haystack) give the
    endpoint's count of the haystack; the cells are then built at the scaled
    size and keep their label. Recorded as `calibration`: per depth, the local
    count, the endpoint's count, their `ratio` and the `build_depth` used.
    Refuses on an endpoint that reports no `usage.prompt_tokens`. Hosted results
    banked so far are nominal, and a calibrated run does not compare with them.
-   `--answer-format line|cited`: `line` (the default, and every earlier result)
    asks for one line - the surname, `OFFTOPIC` or `CONSISTENT`. `cited` asks for
    that line and then the two log lines that show the impossibility, copied
    verbatim. The verdict is scored from line 1 alone, exactly as `line` scores
    it; lines 2-3 are only checked against the haystack. With the one-line
    format, citation checking depends entirely on what a model volunteers while
    reasoning, so fabricated evidence is visible on models that quote and
    invisible on those that paraphrase or return no trace. Each cell under
    `cited` records `evidence`: how many trials `offered` a log line, how many
    cited a line the log lacks (`absent`), and how many `proves` an
    impossibility with two real lines - which on `clean` cannot exist. Recorded
    as `answer_format`; the two formats do not pool.
-   `--vocabulary builtin|seeded`: which surnames fill the haystack. `builtin`
    (the default) is the fixed list of fifty every earlier result used. `seeded`
    generates fifty pronounceable nonsense surnames from `--seed` - the same
    count, so the generator's guarantees hold unchanged - none of which contains
    another or any other word in the log, because the scorer matches the
    expected surname by substring. The scoring rests on a correct answer naming
    a surname no training set contains, and a list published in a repository
    cannot stay that way forever; a seeded list exists nowhere until the run
    makes it. The two read different text from the same seed and do not compare,
    which `vocabulary` and `vocabulary_kind` record.
-   `--build-corpus DIR`: write every haystack the run would read - one file per
    depth, rung, position and trial - plus a `manifest.json` holding each file's
    sha256 and expected answer, the generator version, the tokenizer and the
    vocabulary digest; then exit without contacting any endpoint, so no
    `base_url` is needed. A haystack built from a seed is otherwise an
    *inference* about what a trial read, true only while the generator, word
    lists and sizing tokenizer are unchanged - and only the seed is recorded.
-   `--corpus-dir DIR`: read the haystacks from such an archive instead of
    generating them. Every file is checked against the manifest and the run
    refuses (exit 3) on any mismatch, so an edited haystack fails loudly rather
    than quietly becoming a different experiment. Depths, rungs, positions,
    trials and seed come from the manifest; passing a different one is an error.
    A replay reads exactly the bytes a live run with the same seed would build.
-   `--no-preflight`: skip the one request sent before the first trial. The
    pre-flight sends a tiny question exactly as the trials will - same model
    name, thinking switch and `--extra-body` provider pin - and refuses to start
    if it is rejected: HTTP 4xx or a reply with no `choices` exits **3**, and an
    endpoint that does not answer at all exits **1**. A wrong name or pin fails
    every trial identically, and those failures were once recorded as the model
    declining to answer. When the endpoint lists the requested repository under
    an alias, the refusal names the alias to use.
-   `--save-transcripts PATH`: write each trial's answer and reasoning trace to PATH.
    The accuracy number cannot tell a model that never saw the injection from one
    that saw it and judged it consistent; the trace can, and that distinction is
    what separated the rungs' failure modes.
-   `--positions F …`: where to inject, as fractions of the context (default `0.5`).
    Attention is strongly position-dependent, so `0.1 0.5 0.9` separates that effect
    from depth rather than letting it hide inside every other number. For `log` the
    position is the *centre* between the contradiction's two halves, and a pair
    centred at 0.9 cannot also span 0.6 of the log - so the span narrows
    symmetrically near the ends and the `sep` column reports how far apart they
    actually were. Position and distance cannot both be held fixed there; the table
    publishes both rather than letting one be read as the other.
-   `--trials N`: trials per (depth, rung) cell (default `6`). Deep trials are slow,
    so the default buys a wide interval; raise it for anything you intend to act on.
-   `--max-tokens N`: answer budget (default `40000`). A thinking model scans the
    whole log before answering, so this grows with depth. Too small a budget does
    not merely lose trials, it loses the *hard* ones - a trial the model struggles
    with reasons longer and hits the ceiling - so the reported accuracy rises. The
    tool says so out loud past 20% unanswered. The `clean` control is the most
    expensive rung, because proving absence means checking everything.
-   `--thinking on|off`, `--seed`, `--served-model-name`, `--json`, `--save-result`
    behave as they do elsewhere.

### A false positive is not one phenomenon

When the control rung answers wrongly, the model has done one of two very
different things, and the `false+` count alone cannot say which. Sometimes it
**manufactures the evidence** - quoting a log line the haystack does not
contain, most often a real entry with one digit of its date changed or its
location swapped. Sometimes it quotes only real lines and simply reasons wrongly
- or reaches the *right* verdict and the one-line answer format extracts a
surname anyway, which measures this probe rather than the model.

The `fabricated` column splits them, mechanically and at trial time, against the
haystack that trial actually read. The rule is that the **claim** a citation
makes is `(date, place, surname)`: a false impossibility is precisely the
assertion that one person was in two places on one date, so the rank, the verb
and the object are decoration. The model is held to the claim, not to its
wording - it paraphrases constantly while reasoning, and a rule that can be
satisfied by loose typing is not measuring fabrication.

**A quoted line the log lacks is not by itself a fabrication**, which is the
part that takes measuring. In a 50-trial run at 8k, **10 of the 48 trials the
model got right** still quoted at least one absent line: it misquotes while
reasoning and then does not act on the misquote. So the absent claim must be
*load-bearing* - it has to name the surname the reply names, or, for an
`OFFTOPIC` answer, be the line the reply is calling foreign. Anything else is
recorded as `misquoted` rather than counted as fabrication, which is the
difference between measuring the model and measuring its typing.

The saved result carries the split per cell as `false_positives` -
`fabricated` / `misquoted` / `only_real` / `uncited`, a reply quoting nothing
checkable being counted apart rather than folded in. `--save-transcripts`
records each trial's `citations`: every absent claim, the nearest real line to
each, and the whole reply. That is both what tells a changed digit from a line
invented whole, and what lets a later refinement of the rule be applied to old
results instead of re-measuring them.

### What a saved probe result contains

`--save-result` writes the table's data plus the provenance needed to say what
produced it. At the top level, `floors` holds the retrieval floor at each depth
that ran an `outlier` cell - `misses`, `trials` and `passed`, gated on MISS at
no more than 30% of trials - so the qualification of every `clean` and `log`
rate travels with the file instead of being recomputed by hand. Per cell, beside
the counts shown in the table:

-   `ci95_low` / `ci95_high`: the Wilson interval on the accuracy.
-   `mentioned_while_reasoning`: how many trials named the injected text while
    reasoning. This separates the two ways a rung fails - never saw the text,
    against saw it and judged it consistent - which the accuracy number cannot.
-   `traced`: how many trials returned any reasoning at all - the denominator
    `mentioned_while_reasoning` needs. On an endpoint that withholds or
    summarises the trace, "named the injection 0 times" says what the vendor
    returns, not what the model saw: gpt-6-luna named it 0 times in 10 with a
    trace on only 2.
-   `false_positives`: the `fabricated` / `only_real` / `uncited` split above.
    It is gated on the verdict, so read it with the next entry: a `FALSE+` is
    only reachable on `clean`, where the question has no expected answer. On
    `log` and `outlier` these four counts are structurally zero and say nothing
    about whether the model invented anything.
-   `absent_claim_trials`: how many trials quoted a log line the haystack does
    not contain - on any rung, at any verdict, including unanswered ones. This
    is the fabrication measure that survives the rung, and it exists because the
    bucket above cannot see an invented line on a rung where the reply scores
    `PARTIAL`. **Both are per-rung quantities.** Summing `fabricated` across
    rungs divides a real rate by trials that could never contribute to it;
    `probe.fabrication_rate(results, rung)` is the supported way to compute it,
    and takes one rung by design.
-   `uncited_with_trace`: of the `uncited` false positives, how many came with a
    reasoning trace the citation check could not parse. "Uncited" alone
    conflates two opposite things: a reply with no trace is unverifiable in
    principle, while a trace that paraphrases its evidence - grok-4.3 wrote
    "Achebe appears at both Cinder Yard and Lunar Base" rather than quoting the
    log - is a gap in this tool. The table prints a note when it is non-zero.
-   `usage`: the endpoint's own token counts summed over the cell -
    `prompt_tokens`, `completion_tokens`, and `reasoning_tokens`,
    `cached_tokens` and `cost` where the endpoint reports them - each with a
    `_trials` count of how many trials reported it, plus `trials` for how many
    reported any usage at all; `null` when none did. This is the billed unit,
    which `reasoning_tokens` in a transcript is not: that one is the local
    tokenizer's count of whatever trace the vendor chose to return. It is also
    the **actual depth**: when the endpoint's mean `prompt_tokens` differs from
    the nominal depth by more than 10%, the table says so, because hosted
    haystacks are sized with a local tokenizer and a vendor's may count the same
    text very differently. Each transcript entry carries its own `usage`.
-   `separation`: the `sep` column; `null` for rungs with nothing to separate.
-   `started_at` / `finished_at` / `duration_s`: when the cell ran and how long
    it took. Recorded because a cell that ran slowly on a shared endpoint should
    say so in its own artifact, rather than being reconstructed afterwards from
    file timestamps - which is wrong by the length of any pause between trials.
-   `degenerate`: how many of the cell's unanswered trials were **looping**
    rather than working - a trace that repeats instead of progressing. Reported
    apart from the rest because the advice differs and one version of it is
    actively wrong: a trial that hit `--max-tokens` is told to raise the budget,
    which for a loop simply buys a longer loop. Measured, genuine reasoning runs
    11-26% unique words over its last 3,000 characters while observed loops read
    1.6% and 2.5%, so the threshold does not need to be delicate.
-   `spec_decode`: what speculative decoding did across the cell - `drafts`,
    `draft_tokens`, `accepted` and the resulting `accept_length`. A generation
    setting such as a draft-length cap is invisible to everything else a result
    records: it moves neither the weights nor the engine build, so two cells
    measured either side of a change look identical in the artifact and are not
    comparable. `null` where the endpoint publishes no such counters, where a
    counter went backwards (a restart inside the window), or where any one of
    them is missing - a partial family gives a plausible wrong number. The
    counters are server-global, so this detects a *change* in configuration and
    is not a throughput measurement.

Top level: `served_model`, `served_build`, `engine_version`,
`tokenizer_fallback`, `extra_body` and `providers`. `engine_version` is the inference engine's own version
where it publishes one (vLLM's `/version`) and `null` otherwise - an engine
upgrade changes kernels, sampling and prefix caching, so cells measured across
one do not pool. `tokenizer_fallback` is the
substituted tokenizer's name when the requested one would not load and `null`
when nothing fell back. **A depth is only a depth if the model's own tokenizer
counted it**, so a fallback makes every depth an approximation - the probe warns
before the first trial and heads the table with it.

`extra_body` records any extra JSON fields `--extra-body` merged into each
request (`{}` when none), and `providers` lists every backend that answered,
taken from the response body rather than from `/v1/models`. Both matter against
a **router**: an aggregator reports one model id for backends that may run
different quantisations, so `served_model` cannot distinguish them. Pin one with
`--extra-body 'provider={"only":["…"],"allow_fallbacks":false}'` - and if
`providers` holds more than one name the pin did not hold, the cell mixed
backends, and where those backends differ in precision it mixed that too. An
empty list means no response named a provider, which is normal for a direct
endpoint and reads as unknown, never as "all the same".

`max_tokens` is the output budget the run used - not recorded before, although
every void rule in the programme is a statement about it. `limits` records the
ceilings the endpoint declared (`context_window`, `completion_cap`; `null`
means it said nothing, not that there is no limit), and `budget_warnings` what
was said about the budget against them. Each cell's `truncated` counts the
trials that finished `length` - the budget ended them - and the table notes any
cell where that is non-zero.

`vocabulary` is a digest of the generator's word lists, the outlier snippet and
the question: two results with different values read different text from the
same seed. `vocabulary_kind` says which surname list it was, `builtin` or
`seeded`; a corpus records the list itself, so a replay scores against the
names it was built from. `corpus` is the digest of the archive manifest a `--corpus-dir` run
read, `null` when the haystacks were generated. Each cell's `haystack_digest`
covers every haystack it read, in order, so equal digests mean byte-identical
inputs - what a paired comparison assumes and could otherwise only infer - and
each transcript entry carries its own `haystack_sha256`.

`preflight` records what the pre-flight request saw - its `finish_reason` and
the `provider` that answered it - or `null` when `--no-preflight` skipped it.

`--save-transcripts` writes one entry per trial, failures included:

-   `trace_repetition`: the unique-word ratio over the trace's last 3,000
    characters, or `null` when there is too little text to judge. Alphabetic
    words only - one observed loop repeated `[date] - already checked` with the
    date changing every time, so counting numbers would have hidden a phrase
    that never varied.
-   `started_at`, `duration_s`, `spec_decode`: the same three facts as above,
    for the single trial rather than the cell. One trial per process is the
    normal way to run a deep cell, so this is where a configuration change
    becomes visible as a step between one trial and the next.
-   `reply`, `finish_reason`: what was scored, and whether the server called the
    reply complete. Only a clean stop counts as an answer.
-   `reasoning_excerpt`, `reasoning_chars`, `reasoning_tokens`: the tail of the
    reasoning trace and its size. The trace is where a model's working shows,
    and reading it is what caught four scoring bugs that left the table looking
    perfectly reasonable.
-   `named_in_reasoning`: whether this trial named the injected text.
-   `citations`: the fabrication check for this trial - how many log lines it
    quoted, how many of those the haystack does not contain, and the nearest
    real line to each invented one.

Accuracy is a **proportion**, so the interval is Wilson rather than the Student-t
used for throughput means: on a 0/1 rate a t-interval is wrong, and worst exactly
where a working probe sits - at 0/6 it would report ±0.

Measured on a 262k-context Qwen3.8-27B, same haystack and same question at each
depth - only what is asked of the model changes:

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

A needle-in-a-haystack test is exactly the `outlier` rung - at 200k it would
report near-perfect recall and call this context healthy. The failure is not
silence but confident fabrication, and only the `clean` control makes it
visible.

**Always read `clean` beside any deep score**, and expect it to be the sharper
signal. Its false-positive rate breaks once and then saturates - 0% at 2k, 12% at
16k, **71% at 32k** (p=0.0056 against 16k), then 85%, 89%, 100%. That single step
is the tool's actual answer for this deployment: **trustworthy to ~16k,
confabulating by 32k.** Once the control saturates, an accuracy figure stops
meaning comprehension - it measures whether a fabrication landed on the right
name. Deep cells also lose trials the probe cannot score (a finished reasoning
trace with empty content - 50% of `log` trials at 65k), counted as unanswered
rather than guessed at.
With thinking off at depth 4096 it scores **0/10** against **6/10** on - but that
is a property of *this probe's answer format*, not of the model. Asked the same
question with working permitted, thinking off scores **9/10**, slightly ahead of
thinking on. The model needs *a* scratchpad and the think block is only one of
them; the probe's question demands one line containing only a surname, so closing
the block leaves it nowhere to work at all. `--thinking off` here measures "can it
do this with no scratchpad", which is a real question and not the same as
"thinking off costs long-context quality".

## Live progress stream (for external visualizers)

`--emit-progress PATH` writes a stream of newline-delimited JSON events to
`PATH` (or `-` for stdout) while the benchmark runs. External visualizers -
live TUIs, web dashboards, post-hoc analyzers - consume that stream and
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
