---
name: llm-assay
description: Drive llm-assay for someone using it for the first time - measure an OpenAI-compatible LLM endpoint's speed (prefill and decode at real context depths), A/B two server configurations with a significance test, check long-context quality (`probe`), grade a model's behaviour in a coding harness (`agent`), or test whether a serving change moves the model (`divergence`). Use when the user asks to benchmark, compare, size, or quality-check a vLLM, SGLang, llama.cpp or hosted endpoint with llm-assay, or asks what an llm-assay result means.
---

# Driving llm-assay

llm-assay measures an OpenAI-compatible endpoint and refuses to flatter it: confidence
intervals instead of bare means, paired significance tests, and loud warnings when a
number does not mean what it looks like. Your job is to pick the right subcommand,
size it to the user's server, run it, and **pass the warnings on** - they are the
point of the tool, not noise to summarise away.

`README.md` in the checkout is the full reference; section names are given below so
you can read the part you need instead of all of it.

## Ground rules

- **Run from the checkout with `uv`**: `uv run llm-assay.py ...`. There is no install
  step and no console command; uv resolves dependencies on first run. If `uv` is
  missing, the README's *Setup* section has the install line and the pip fallback.
- **Keys go in the environment**: `export LLM_ASSAY_API_KEY=...`. `tune`, the
  benchmark, `probe`, `agent`, `divergence` and `scenario` all read it (`scenario`
  reads it nowhere else). Never put a key on a command line (it is
  visible in `ps`) or in a file you write. In a models file use `api_key_env:`.
- **Ask before loading someone else's endpoint.** A benchmark saturates the server;
  on a hosted API every request costs money. Confirm the endpoint is the user's, and
  whether anything else is using it - another client's traffic corrupts the server
  counters llm-assay reads (it warns when it sees it).
- **Start small.** Every workflow below has a smoke step. Run it, read it, then scale.
- **Results are JSON in `results/`** (or wherever `--save-result` says). Keep them:
  `compare` works on saved files, and a result records the invocation and what the
  server said it was serving, so it can be checked later.

## Step 0 - what does the user actually want?

| they want to know | use | README section |
|---|---|---|
| how fast is my server, at the context lengths I use | `tune`, then the benchmark | *Sizing a run to an endpoint (`tune`)*, *Usage* |
| did this change (flag, engine version, quant) make it faster or slower | benchmark twice + `compare` | *Comparing two runs statistically* |
| can the model still *use* a long context, or does it make things up | `probe` | *Checking quality, not just speed (`probe`)* |
| does the model behave safely with a shell (in Pi / Hermes / OpenCode / OMP) | `agent` | *Doing the right thing with tools (`agent`)* |
| did a serving change (KV dtype, quantisation, TP) change the model's outputs | `divergence` | *Does a serving change move the model?* |

If they just say "benchmark my server", that is the first row. Always begin with `tune`.

## Step 1 - `tune`: let the endpoint size the run

```bash
uv run llm-assay.py tune http://localhost:8000/v1
```

It reads `/v1/models`, `/metrics` and (where available) the render endpoint, and
prints what it detected and a list of suggested runs. Tell the user what it found,
in particular:

- **prefix cache** ENABLED / DISABLED / unknown - decides whether cached-follow-up
  rows mean anything, and why every preset uses `--seed $RANDOM`.
- **spec decode** ACTIVE - decode will be noisy (acceptance depends on content), so
  decode presets sample adaptively (`--target-ci`) instead of a fixed `--runs`.
- **kv cache** size - caps the concurrency levels it suggests.
- **thinking** - which switches actually change the prompt here. A level listed as
  "same as sending nothing" renders the prompt the server builds with no level at all,
  so sweeping it beside the default measures one config twice; "rejected" levels make
  every request fail, and the benchmark, probe and agent refuse to start on them.
- a **tokenizer** note under the suggestions when the served name is an alias or a
  GGUF - shapes are then counted in the right vocabulary only if `--tokenizer` is
  passed (tune adds it when it found one that loads; otherwise it says to).

Then save the suggestions as a script and run the smoke preset:

```bash
uv run llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh
./run-local.sh --help      # lists the presets this server got
./run-local.sh smoke
```

`tune --json` gives the same as data (argv lists you can exec). Re-run `tune` after
the server is restarted with different settings; the script says so in its header.

**On a deployment nobody has measured yet, run `./run-local.sh qualify` first**
(`scenario qualify`, README *One question, end to end*): a canary reply with the
deployment's default thinking (and with thinking off, where an off switch works), the smoke benchmark, a short probe and, when tune found a harness, three agent
runs, graded in `VERDICT.txt`. A FAIL there - the think block in the content,
an answer only in reasoning - means every later number is wrong; fix the server first.
To A/B a change to the server, `ab-before`, change it, `ab-after`: same seed both
sides, and the verdict says what the endpoint reports changed.

## Step 2 - speed: the benchmark

After `smoke` looks sane, run the presets the user cares about (`sweep` for context
depth, `concurrency`, `long`, decode). Or build a command by hand:

```bash
uv run llm-assay.py --base-url http://localhost:8000/v1 --model MY-MODEL \
  --depth 0 4096 16384 --pp 2048 --tg 128 --latency-mode generation \
  --seed $RANDOM --save-result results/sweep.json
```

Reading the table (README *Metrics*):

- `pp2048 @ d16384` - prefill: a 2048-token prompt on top of 16k of context. `t/s` is
  prompt tokens / `est_ppt` (time to first generated token minus measured latency).
- `tg128 @ d16384` - decode speed over the generated tokens.
- `ctx_pp` / `ctx_tg` rows appear only with `--measure-cached-followup`: they are the
  cold context load, and the `pp`/`tg` rows beside them are then a follow-up over the
  cached context.
- `±` is the sample std by default; `--stats ci` shows the 95% interval, which is
  what to look at when asking "is this difference real". `--legend` explains every
  printed row and column in plain words - offer it to a first-time user.

**Warnings to relay, always** (each is a reason the number may be wrong):

| warning says | what it means for the user |
|---|---|
| cached-follow-up ran with ~0% cache hits | the follow-up rows measured a full re-prefill; prefix caching is off or not working |
| run was served from the prefix cache (stale cache) | a fixed `--seed` re-sent identical prompts; prefill is inflated. Re-run with a fresh seed or `--cold` |
| another client / foreign traffic | the server's counters include someone else's requests |
| statistical reliability: CI too wide, need ~N runs | the mean is not precise enough to quote; run more or use `--target-ci` |
| endpoint reports serving X (differs from `--model`) | normal for an alias; tell the user which weights were actually measured |
| corpus shorter than depth + pp | deep shapes read repeated text, which flatters speculative decode |
| tokenizer fell back to gpt2 | the shape labels are approximate; pass `--tokenizer` |

**Exit codes**: 0 measured; 1 a whole shape produced nothing, or nothing was
measured; 2 a usage error, or a `--thinking` / `--reasoning-effort` setting the server
rejects (checked before anything is measured); 4 void - the endpoint changed identity mid-run, so the numbers belong to no
single model. Do not report a non-zero run as a result.

## Step 3 - A/B two configurations

The trap (README *The trap: a fixed `--seed` warms the cache*): the same command run
twice with the same seed is served from the cache the first run filled, and `compare`
will declare it significantly faster than itself (+150% prefill in the README's
example). So:

```bash
# prime once so both sides start equally warm (or use --cold on both and accept unpaired)
uv run llm-assay.py --base-url ... --model ... --seed 1234 --runs 5 --save-result /tmp/prime.json
uv run llm-assay.py --base-url ... --model ... --seed 1234 --runs 5 --save-result results/baseline.json
# ... the user changes ONE thing on the server, restarts it, and primes again
#     (a restart empties the cache, which would leave only the baseline warm) ...
uv run llm-assay.py --base-url ... --model ... --seed 1234 --runs 5 --save-result /tmp/prime2.json
uv run llm-assay.py --base-url ... --model ... --seed 1234 --runs 5 --save-result results/candidate.json
uv run llm-assay.py compare results/baseline.json results/candidate.json
uv run llm-assay.py compare results/baseline.json results/candidate.json --metric pp_throughput
```

- Same `--seed` on both sides makes it a **paired** test (much more sensitive); the
  shape flags (`--depth --pp --tg --concurrency`) must match too, or no rows line up
  and `compare` exits 1.
- At least 3 runs per side; 5+ on a speculative-decoding server.
- `no difference` is **not** "the same". To claim a change is harmless use
  `--equivalence-margin 0.03` (EQUIVALENT vs INCONCLUSIVE).
- To A/B thinking on/off or reasoning effort: `compare ... --across thinking` or
  `--across reasoning_effort` (unpaired by design).
- `--fail-on-regression` turns it into a CI gate.
- Read the comparability warnings `compare` prints first: a setting that differs
  between the files other than the one under test means the comparison measures it.

## Step 4 - quality: `probe`

`probe` builds a synthetic log at a chosen depth with a known answer and asks the model
what is wrong with it. Three rungs: `log` (one real contradiction - the measurement),
`outlier` (one foreign line - can it *see* that deep), `clean` (nothing wrong - how
often it **invents** a problem). The clean rung is the important control.

```bash
uv run llm-assay.py probe http://localhost:8000/v1 --model MY-MODEL \
  --depths 4096 32768 --trials 6 --save-result results/probe.json
```

Prefer the preset `tune` wrote (`./run-local.sh probe`): it sizes `--max-tokens` to
the context window and adds `--tokenizer` / `--calibrate-depth` / `--max-spend` where
needed. Things to tell the user:

- **It is slow on reasoning models** - the model reads the whole log before answering;
  tune estimates the time. Start with one depth and 3-6 trials.
- **`unanswered` depends on the budget.** A trial that hit `--max-tokens` is
  unanswered, and the ones lost are the hard ones, so a small budget *raises* the
  score. Quote any rate with its `--max-tokens` beside it.
- **Small n means wide intervals** - read the Wilson 95% CI column, not the fraction.
  6/6 is compatible with a model that fails one time in three.
- `--thinking off` scores near 0 by the probe's one-line answer format, not because
  the model cannot do it without thinking.
- On a router or hosted API, pin one backend with `--extra-body`, and set `--max-spend`.

## Step 5 - behaviour: `agent`

Hands an installed coding harness a small repository with a trap in it and grades the
**files** afterwards, never the model's own account. Needs: Linux with bubblewrap
(the sandbox; `--unsandboxed` runs the agent as the user on their machine - say so
plainly before using it), and one harness installed: `pi` (default), `hermes`,
`opencode` or `omp` (`--harness`, `--harness-cmd` if not on PATH). The runner `tune`
wrote has an `agent-<harness>` preset for each harness it found on PATH
(`RUNS=1 EXTRA="--tasks sync" ./run-local.sh agent-pi` is the smoke); none listed
means none was found.

```bash
# smoke: one run of one task
uv run llm-assay.py agent http://localhost:8000/v1 --model MY-MODEL --tasks sync --runs 1 --keep /tmp/agent-keep
# then the real thing
uv run llm-assay.py agent http://localhost:8000/v1 --model MY-MODEL --tasks all --runs 10 --workers 4 \
  --sampling temperature=1.0 top_p=0.95 top_k=20 --save-result results/agent.json
```

- **Pin sampling from the model card** (`--sampling`), or two deployments differ in
  sampling alone. Check whether the card gives different values with thinking off.
- **Workers share the endpoint.** Pass rates hold until the KV cache runs out; then
  the server preempts, runs crawl and time out, and you are measuring the box. If
  timeouts appear, lower `--workers`. `wall_s` compares only at equal worker counts.
- **Void is not failed**: a run where the endpoint changed, the harness crashed or
  changed version, the server failed its last turn, or a request was refused is void
  and left out (a timeout is a failure, not a void). Report voids; many voids means fix the
  setup, not the model.
- **Read a few kept traces** (`--keep`) before trusting the grade on a new model.
  Each task's page in `docs/agent-tasks/` gives its prompt, trap and checks.
- n matters: even at 20 runs per task, only a large gap (20/20 against 15/20) is
  significant on one task. Compare two results with `compare` (Fisher per task,
  stratified across tasks), and say "not shown to differ", not "the same".

## Step 6 - `divergence` (vLLM only)

Whether a serving change moves the next-token distribution, on the prefixes an `agent`
run sent. It needs an `agent` result **with its `--keep` directory**, and three
server states: reference, candidate, and the reference again (the floor - an unchanged
config already disagrees with itself at a few percent of positions). The commands are
in README *Does a serving change move the model?*. Divergence is not harm: read it
beside the `agent` pass rates.

## Reporting back

Give the user: what was run (endpoint, the model the server *reported*, key flags),
the numbers that answer their question with their intervals or p-values, every
warning the tool printed in plain words, and what you would run next to firm it up.
Never round a `no difference` into "the same", and never drop a warning to make the
answer cleaner.
