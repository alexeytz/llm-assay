# Fabrication programme: use cases, and how to run them

How the programme's tests are run, written as use cases: what each one is for,
what it does, and one real execution of it, so you can adapt it to your own
model. `fabrication-findings.md` holds the results; this file shows how to
produce results like them. It is a selection, not a history: the programme ran
33 pre-registered arms, and each use case below shows one of them, with the
others that used the same method named in a line.

---

## The building blocks

Every use case is made of the same few parts.

### Running one cell: `scripts/probe-cell.sh` (Appendix A)

```
# usage: scripts/probe-cell.sh DEPTH BASE_SEED [N_TRIALS] [MAX_TOKENS] [RUNGS]
```

A *cell* is one depth and one rung, run N times. The script runs
`llm-assay.py probe` **once per trial**, one process each, so a crash or a
restarted server costs one trial instead of a whole cell, and a cell resumes
where it stopped (a trial whose result file exists is skipped). Defaults: N=10,
MAX_TOKENS=80000, RUNGS="log clean". Trial *i* on a rung gets seed
`BASE_SEED + i*100 + offset` (offset 0 for `log`, 7 otherwise), so a trial is
reproducible from its file name, and two arms with the same `BASE_SEED` read
byte-identical haystacks. The rungs: `clean` plants nothing (any surname named is
invented), `log` plants one real impossibility (finding it is detection),
`outlier` plants a foreign Python snippet (missing it means the model was not
reading).

Configured through the environment:

| variable | meaning | default |
|---|---|---|
| `PROBE_BASE_URL` | endpoint | `http://localhost:8080/v1` |
| `PROBE_MODEL` | name the API is called with (the serving alias on vLLM) | `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` |
| `PROBE_TOKENIZER` | tokenizer that sizes the haystack | `Qwen/Qwen3.8-Flash-Next` |
| `PROBE_OUTDIR` | the cell's directory | `results/cell-d${DEPTH}` |
| `PROBE_API_KEY` | key, read by probe itself - never passed as `--api-key` | - |
| `PROBE_EXTRA_BODY` | extra JSON fields, e.g. a router provider pin | - |
| `PROBE_TRIAL_TIMEOUT` | wall-clock limit per trial | 28800 |
| `PROBE_MAX_ATTEMPTS`, `PROBE_BACKOFF`, `PROBE_BACKOFF_CAP` | retries of an endpoint failure | 5, 60, 900 |
| `PROBE_DEADLINE` | epoch seconds; no trial starts after it | - |
| `PROBE_MAX_SPEND` | dollars, summed from banked `usage.cost` | - |

A failed trial is handled by who failed. A non-answer the *model* produced
(`incomplete answer: ...`, `no answer content`) is banked and counts as
unanswered, because the unanswered rate is part of the measurement. Anything
else - the endpoint failing - is moved to `interrupted/` and retried on the same
seed. Exit 3 from probe (a wrong model name or provider pin, or a budget that
cannot fit) stops the cell at once.

### Pooling a cell, and testing two: `scripts/probe-pool.py` (Appendix B)

```
usage: uv run scripts/probe-pool.py DIR [DIR ...]
       uv run scripts/probe-pool.py --compare SHALLOW_DIR DEEP_DIR
       uv run scripts/probe-pool.py --paired ARM_A ARM_B [RUNG]
```

`DIR` sums every one-trial file into a table per (depth, rung), with the
retrieval floor beside each cell. `--compare` is Fisher's exact test on
fabricated against answered, between two arms. `--paired` matches trials by
(seed, depth) and runs exact McNemar, over all trials and on complete pairs.

### Checking what is being served

A run against the wrong weights or engine is worse than no run. The checks the
runners made before the first trial:

```
# engine version (runner `v13-runner`)
v=$(curl -s -m 10 http://vllm-host2.example:8000/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)

# alias and the repository behind it (runner `v25-arm`)
models=$(curl -s -m 15 "$PROBE_BASE_URL/models")
served=$(echo "$models" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'][0];print(d.get('root') or d.get('id'))" 2>/dev/null)
alias=$(echo "$models" | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0].get('id'))" 2>/dev/null)

# KV-cache dtype (runner `v26-cell`)
kv=$(curl -s -m 15 "http://${HOST}:8000/metrics" | grep -oE 'cache_dtype="[^"]*"' | head -1)
```

On llama.cpp, whose `/v1/models` answers with a pinned alias, the build is
read from `/props`:

```
curl -s http://llamacpp-host1.example:8080/props \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['total_slots'], d['model_ftype'], d['model_path'])"
```

### Freezing the design before the first trial

Each arm's design, decision rules, seeds and power were written down and frozen
**before its first trial**, by committing the document or by publishing its
sha256 and committing it unchanged later. A runner records the hash of the
registration it runs under, so a changed design cannot pass unnoticed:

```
# runner `v25-arm`, header: sha256 of prereg v25
ffc4da72e4d7b75de43589d6a5fa3fb209ce8c23b5c5c825bc681ced0e1ec713
```

That is a hash of a document, not a seed; seeds are the `BASE_SEED` arguments.

### Stopping a running arm

```
kill -- -<pgid>                           # stop a worker group (runner `v31-boost`)
```

Kill the process group, not the runner: `timeout` puts each trial in its own
group, so killing only the runner orphans the trial in flight. And any
`pgrep -f`/`grep` over command lines also matches the shell running the check.

---

## Use case 1: does a model invent more as the document grows?

**What it does.** One model, a shallow and a deep `clean` cell, and Fisher's test
between them. The simplest form of the programme's question.

**Example** (prereg v2, Flash-Next GGUF on `llamacpp-host1`):

```
PROBE_OUTDIR=results/prereg2-fabrication-d32768 \
  scripts/probe-cell.sh 32768 327680000 15 190000 "clean"

uv run scripts/probe-pool.py --compare \
  results/prereg-fabrication-d8192 results/prereg2-fabrication-d32768
```

The shallow arm was the same command at depth 8192 with n=50 (prereg v1).
**Result:** 2.0% (1/49) at 8,192 against 38.5% (5/13) at 32,768, p = 0.0011.

---

## Use case 2: map where it starts - a depth ladder

**What it does.** Many depths for one model, to see where invention begins and how
it grows; a `log` rung at the same depths shows detection on the same curve.

**Example** (prereg v3, `cyankiwi/Qwen3.8-27B-AWQ-FP8` on `vllm-host2`):

```
for d in 2048 4096 6144 8192 12288 16384 24576 32768 65536; do
  PROBE_OUTDIR=results/prereg3-ladder-d$d \
    scripts/probe-cell.sh $d $((30000000 + d)) 30 190000 "clean log"
done
```

For a long ladder, call the script one trial at a time so a stop file or a
deadline is checked between trials (runner `v6-runner`, inside
`for i in $(seq 1 30)`):

```
PROBE_OUTDIR=results/prereg6-flashnext-d$d \
  scripts/probe-cell.sh "$d" "$((99000000 + d))" "$i" 225000 "clean" >> "$LOG" 2>&1
```

Also used by v4-v8, v10, v11 (Flash-Next), v24 (GLM 5.3), v31-v32.
**Result:** the 27B invented nothing through 8,192 and 88% by 65,536;
Flash-Next's curve turned out to be a ramp, not a cliff.

---

## Use case 3: tell two models apart at one depth

**What it does.** Runs a new model's cell at the same depth, budget and engine as
a banked cell of another model, and tests the two with `--compare`.

**Example** (prereg v12, the 27B against Flash-Next's banked 32k cell):

```
export PROBE_BASE_URL=http://vllm-host2.example:8000/v1
export PROBE_MODEL=VLLM-HOST2
export PROBE_TOKENIZER=Qwen/Qwen3.8-27B
PROBE_OUTDIR=results/prereg12-27b-d32768 \
  scripts/probe-cell.sh 32768 90032768 40 225000 "clean"
```

Also v14, v16 (hosted families). **Result:** 18/32 against 21/28, p = 0.18 - a
null the registration had predicted at its power of 0.49. Compute the power
before running; it decides whether a null means anything.

---

## Use case 4: a hosted model through a router, pinned to one provider

**What it does.** Measures a model you cannot run, as a customer reaches it. A
router serves one model id from several providers with different
quantisations, so the cell is pinned to one, and the key is read from the
environment, never put on a command line.

**Example** (prereg v16, `poolside/laguna-xs-2.1`; runner `v16-runner`):

```
set -a; . "<keyfile>"; set +a
MODEL=poolside/laguna-xs-2.1
PIN=poolside/fp8
BUDGET=32000          # just under the provider's declared 32,768 ceiling
export PROBE_API_KEY="$OPENROUTER_API_KEY"
export PROBE_BASE_URL=https://openrouter.ai/api/v1
export PROBE_TOKENIZER=Qwen/Qwen3.8-27B
export PROBE_MODEL="$MODEL"
export PROBE_EXTRA_BODY="provider={\"only\":[\"$PIN\"],\"allow_fallbacks\":false}"
PROBE_OUTDIR="results/prereg16-laguna-d${depth}" \
  scripts/probe-cell.sh "$depth" "$seed" "$i" "$BUDGET" "$rung" >> "$LOG" 2>&1
cell 32768  69032768 "clean"   20 || true
cell 32768  69032768 "outlier" 10 || true
```

Before spending a cell, check the pin answers (runner `v24-worker`):

```
pf=$(curl -s -m 60 https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"provider\":{\"only\":[\"$PIN\"],\"allow_fallbacks\":false}}")
```

Also v9, v14-v20, v22, v24, v29. For a model with its own tokenizer, use it:
GLM 5.3 ran with `PROBE_TOKENIZER=zai-org/GLM-5.3`. **Result:** at 32k,
gpt-6-sol-pro invented 5% and found 35%; laguna invented 75% and found nothing.

**With today's tool,** `llm-assay.py probe` refuses a dead pin before the first
trial, `--max-spend` caps the bill, `--reasoning-effort` sets and records effort,
and `--calibrate-depth` counts depth in the vendor's own tokens.

---

## Use case 5: prove the model was reading - the retrieval floor

**What it does.** A low invention rate is either careful checking or no checking.
The `outlier` cell at the same depth says which: at most 3 misses in 10 passes.
A rate with no passing floor beside it is not interpreted.

**Example** (prereg v17, hosted floors at 32k; runner `v17-hosted-runner`):

```
PROBE_OUTDIR="results/prereg17-${tag}-d32768" \
  scripts/probe-cell.sh "$DEPTH" "$SEED" "$i" "$budget" "outlier" >> "$LOG" 2>&1
cell "openai/gpt-6-luna"        "openai"    120000 "gpt6luna"      || true
cell "x-ai/grok-4.3"            "xai"       120000 "grok43"        || true
```

A setting can be tested the same way, on the same seeds - v15 re-ran grok's cell
at high effort by adding to the pin (runner `v15-runner`):

```
run_cell "grok43-clean-hi-d131072"  "clean"   20 "$pin,reasoning={\"effort\":\"high\"}" || true
```

**Result:** grok's perfect 0/20 at 131k came with a floor that missed 4 of 9 -
it was mostly not reading - and at high effort the same cell invented 4 of 20.

---

## Use case 6: can it find a real problem - detection

**What it does.** The `log` rung plants one real impossibility - one person in two
places on one date - and scores whether the model names that person.

**Example** (prereg v18, hosted at 32k; runner `v18-hosted-runner`):

```
PROBE_OUTDIR="results/prereg18-${tag}-d32768" \
  scripts/probe-cell.sh "$DEPTH" "$seed" "$i" "$budget" "log" >> "$LOG" 2>&1
cell "openai/gpt-6-sol-pro"     "openai"       120000 67032768 "gpt6solpro"    || true
```

Also v19, v20, and the `log` cells of every ladder. **Result:** at 32k detection
ran from 35% (gpt-6-sol-pro) to 0% (grok, laguna).

---

## Use case 7: change one thing, keep the haystacks identical - a paired A/B

**What it does.** Tests whether one setting changes what a model claims. Both arms
use the same `BASE_SEED` and depth, so they read byte-identical haystacks, and a
paired test cancels the content variance. The outcome is "produced the correct
answer" over all trials, which the unanswered rule cannot void.

**Example** (prereg v26, the same weights on two hosts and an NVFP4 build;
runner `v26-cell`):

```
H1) HOST=vllm-host2; EXPECT=cyankiwi/Qwen3.8-27B-AWQ-FP8 ;;
C1) HOST=vllm-host1; EXPECT=nvidia/Qwen3.8-27B-NVFP4 ;;
DEPTH=12288
SEED=26012288          # shared by all three cells: byte-identical haystacks
BUDGET=225000
export PROBE_BASE_URL="http://${HOST}:8000/v1"
export PROBE_TOKENIZER=Qwen/Qwen3.8-27B
export PROBE_OUTDIR="results/prereg26-${CELL}-d${DEPTH}"
scripts/probe-cell.sh "$DEPTH" "$SEED" "$2" "$BUDGET" "$1" >>"$LOG" 2>&1 \
run_cell outlier 10 || exit 1     # floor first
run_cell clean   60 || exit 1
```

```
uv run scripts/probe-pool.py --paired results/prereg26-H1-d12288 results/prereg26-C1-d12288
```

Also v13 (engine version), v21 and v25 (quantisation), v28 (speculative draft
head, served with and without `--speculative-config`). **Result:** none of them
moved the result: AWQ-FP8 48/60 against NVFP4 47/60, p = 1.0000.

---

## Use case 8: size the output budget before a ladder

**What it does.** A budget that cuts trials removes the hard ones, so the result
reports the easy subset. Run a few trials with a budget far above anything
expected, and see how far the model's reasoning actually runs.

**Example** (prereg v29, GLM 5.3's tail at 400,000; runner `v29-arm`):

```
DEPTH=32768
SEED=24032768
N=10
BUDGET=400000
MODEL=z-ai/glm-5.3
PIN=morph/fp8
export PROBE_OUTDIR="results/prereg29-glm53-morph-d${DEPTH}"
export PROBE_TOKENIZER=zai-org/GLM-5.3
export PROBE_EXTRA_BODY="provider={\"only\":[\"$PIN\"],\"allow_fallbacks\":false},reasoning={\"enabled\":true}"
scripts/probe-cell.sh "$DEPTH" "$SEED" "$i" "$BUDGET" clean >>"$LOG" 2>&1 \
```

**Result:** output length on identical input spanned 856 to 179,847 tokens. No
fixed budget is safe; check it against the model's tail *and* the context
window, which caps it at depth. `llm-assay.py probe` now does the second check
itself and counts `truncated` trials per cell.

---

## Use case 9: a new model family - serve it right, then measure it

**What it does.** A ladder on a family not measured before. The serving
configuration is part of the measurement: a missing reasoning parser does not
fail, it puts the model's working notes in the answer field.

**Serve** (prereg v32, runner `v32-moe-launch`):

```
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source /root/vllm/bin/activate
exec vllm serve unsloth/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name VLLM-HOST1 \
  --trust-remote-code \
  --language-model-only \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --attention-backend TRITON_ATTN
```

**A canary first** - one real trial whose reply must be short, so a parser
leaking the whole thought block is caught at trial one (runner `v31-arm`):

```
export PROBE_OUTDIR="results/prereg31-gemma4-d2048"
scripts/probe-cell.sh 2048 30002048 1 "$BUDGET" log >>"$LOG" 2>&1 \
```

**Then the ladder,** floor first at each depth (runner `v31-arm`):

```
for d in $DEPTHS; do
  SEED=$(( 30000000 + d ))
  has_floor "$d" && { run_cell "$d" "$SEED" outlier 10 || exit 1; }
  run_cell "$d" "$SEED" log   30 || exit 1
  run_cell "$d" "$SEED" clean 30 || exit 1
```

**Result:** on the Gemma 4 finetune, detection fell from 100% at 2,048 to 1/30 at
24,576, p = 3.4e-09. The arm before it, run without `--reasoning-parser gemma4`,
scored a uniform 30/30 PARTIAL off scratchpads and was voided. Reasoning parsers
are per family (`gemma4`, `qwen3`); on vLLM, `--kv-cache-dtype fp8` may also
need `--attention-backend TRITON_ATTN`.

---

## Use case 10: verify that a claimed fabrication really is one

**What it does.** Checks that the lines a false positive cites do not exist in the
haystack the trial read. A one-digit date edit can turn an innocent repeat into
an apparent impossibility.

The probe does this at trial time against the exact haystack: each transcript
records its `citations`, and each cell splits false positives into
`fabricated`, `misquoted`, `only_real` and `uncited`. To re-check one by hand,
rebuild the haystack from the seed in the trial's file name with
`build_context(tok, depth, rung, random.Random(seed), position)` from
`llm_assay.probe`. Better still, archive the haystacks before running
(`--build-corpus DIR`) and replay them (`--corpus-dir DIR`), so the check reads a
record rather than a regeneration; `--answer-format cited` asks the model for
the two lines that show its claim and checks them.

---

## Alongside: a throughput A/B on the same machine

Not the probe, but the same endpoints: two server configurations, each
benchmarked with a fresh seed, then compared (runner `gdn-ab`):

```
uv run llm-assay.py --base-url http://vllm-host1.example:8000/v1 \
  --model VLLM-HOST1 --tokenizer Qwen/Qwen3.8-Flash-Next \
  --pp 512 --tg 1024 --depth 0 --runs 8 --seed $RANDOM \
  --format json --save-result "results/gdn-$1.json" >> "$LOG" 2>&1
uv run llm-assay.py compare results/gdn-cuda.json results/gdn-triton.json >> "$LOG" 2>&1
```

`$RANDOM` matters: a fixed seed sends byte-identical prompts, and the second run
reads the first one's prefix cache.

---

## Appendix A: `probe-cell.sh`

The cell runner every command above calls, as `scripts/probe-cell.sh` in the
repository. Save it there to run the commands as written; it needs the
repository's `llm-assay.py` beside it.

````bash
#!/usr/bin/env bash
# Run a probe cell one trial at a time, persisting each.
#
# A deep cell runs for hours and `probe` writes its --save-result only when the
# whole cell finishes, so anything that kills the process loses every trial.
# That is not hypothetical: a 4k cell died 7 trials in and wrote nothing.
#
# Each (rung, trial) is its own invocation with its own seed and its own file,
# so a death costs one trial. Re-running skips files that already exist, which
# makes the whole thing resumable by just running it again.
#
# That skip is why an endpoint outage has to be told apart from a model that
# did not answer: probe writes a complete result either way, and the wrong one
# banked is a restart recorded as the model's failure. See the classification
# at the trial itself. A transport failure is quarantined under
# `interrupted/` -- out of reach of probe-pool.py, which lists one level -- and
# the same seed is retried; a genuine non-answer is kept exactly once.
#
#   PROBE_TRIAL_TIMEOUT  seconds before a silent trial reads as a hung
#                        endpoint rather than a working model (default 28800).
#                        Sized never to bind -- see the constant.
#   PROBE_MAX_ATTEMPTS   attempts at one trial before the cell stops,
#                        resumably, rather than burning the rest against a
#                        dead endpoint (default 5)
#   PROBE_BACKOFF        first wait between attempts, doubling (default 60)
#   PROBE_BACKOFF_CAP    ceiling on that wait (default 900)
#
# Pool the per-trial files with scripts/probe-pool.py.
#
# usage: scripts/probe-cell.sh DEPTH BASE_SEED [N_TRIALS] [MAX_TOKENS] [RUNGS]
#
# RUNGS defaults to "log clean". Pass "clean" alone where only the control is
# in question -- at depth the log rung terminates fast and has been 10/10
# everywhere, so its trials are cheap but buy nothing, while clean trials are
# the expensive ones and the only ones still carrying uncertainty.
set -uo pipefail

DEPTH="${1:?usage: probe-cell.sh DEPTH BASE_SEED [N_TRIALS] [MAX_TOKENS] [RUNGS]}"
BASE_SEED="${2:?base seed required}"
N="${3:-10}"
MAX_TOKENS="${4:-80000}"
RUNGS="${5:-log clean}"

# A deployment's own endpoint is not the repository's business, so the default
# is localhost and the real one is supplied out of tree. `local/` is
# gitignored; put the exports there and every invocation picks them up:
#
#   # local/probe-env.sh
#   export PROBE_BASE_URL=http://your-host:8080/v1
#   export PROBE_MODEL=unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
#
# Record in that file *which build* the endpoint serves, too. A cell measured
# against a different quantisation or conversion is a separate arm and does not
# pool with cells measured against another, and nothing in the filenames says
# so.
# Precedence is explicit env > private file > public default. The file is
# sourced, so its `export`s would otherwise clobber a variable set on the
# command line -- which is the wrong way round, and silently: the run would
# go to the file's endpoint while the invocation said otherwise. So the
# environment is captured first and restored after.
_env_base="${PROBE_BASE_URL:-}"
_env_model="${PROBE_MODEL:-}"
_env_tokenizer="${PROBE_TOKENIZER:-}"

_local_env="$(dirname "$0")/../local/probe-env.sh"
# shellcheck source=/dev/null
[ -f "$_local_env" ] && . "$_local_env"

BASE_URL="${_env_base:-${PROBE_BASE_URL:-http://localhost:8080/v1}}"
MODEL="${_env_model:-${PROBE_MODEL:-unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL}}"
# Mandatory, not a nicety: the served id is a GGUF repo carrying a
# quantisation tag, so it is not a valid HF repo id and load_tokenizer
# falls back to gpt2 -- which counts a different number of tokens, so
# every --depths value silently stops being the depth it claims.
TOKENIZER="${_env_tokenizer:-${PROBE_TOKENIZER:-Qwen/Qwen3.8-Flash-Next}}"
OUTDIR="${PROBE_OUTDIR:-results/cell-d${DEPTH}}"
# A hosted endpoint needs a key, and a *router* needs a backend pin: without
# one, OpenRouter serves different trials from providers running different
# quantisations and the cell mixes precisions with nothing recording it.
#
# PROBE_API_KEY is deliberately *not* forwarded as `--api-key`. It was, for one
# commit, and the key then appeared in full in `ps` output for the life of every
# trial -- readable by any user on the box, which is worse than a shell history.
# probe defaults --api-key from this same variable, so exporting it is enough
# and the key never reaches an argument vector.
EXTRA_BODY="${PROBE_EXTRA_BODY:-}"

# How long a single trial may take before it is read as a hung endpoint rather
# than a working model.
#
# This must not bind. The design censors on *tokens* -- --max-tokens, which is
# the same number on every run -- so a wall-clock ceiling that fires censors on
# time instead, and time depends on whether anything else is using the server.
# That would put a confound in the measurement rather than a guard around it.
#
# Sized against the worst case actually reachable: a trial that burns the whole
# 190,000-token budget, at the rate observed while a second client shares the
# endpoint. Measured on this deployment -- 13.38 t/s alone, 6.19 t/s sharing,
# because two sequences through a 10-of-512 MoE route to near-disjoint experts
# and the batch step doubles -- the full budget is 3.9h alone and 8.5h shared.
# 5h was the first value here and would have fired on a shared box.
TRIAL_TIMEOUT="${PROBE_TRIAL_TIMEOUT:-28800}"
# A llama.cpp restart reloads a 176B model, so the first few retries are
# expected to fail; the backoff is sized for that rather than for a blip.
MAX_ATTEMPTS="${PROBE_MAX_ATTEMPTS:-5}"
BACKOFF_START="${PROBE_BACKOFF:-60}"
BACKOFF_CAP="${PROBE_BACKOFF_CAP:-900}"
backoff="$BACKOFF_START"

_transport_failure() {
  # "no" (keep it) only when the saved trial is a measurement: something was
  # answered, or every recorded error is the model's own non-answer. Anything
  # else is "yes" -- the endpoint's, or not provably the model's.
  #
  # Classified on the recorded message rather than on the exception type,
  # because an HTTP 503 from a server that is still loading arrives as the same
  # ProbeError as a reply the model did not finish.
  python3 - "$1" <<'PY'
import json, re, sys

try:
    with open(sys.argv[1]) as fh:
        report = json.load(fh)
except (OSError, ValueError):
    # Unreadable or half-written: the process died rather than measured.
    print("yes")
    raise SystemExit

results = report.get("results", [])
errors = [e for r in results for e in r.get("errors", [])]
answered = sum(r.get("answered", 0) for r in results)
transport = re.compile(
    r"cannot connect|connection reset|connection refused|server disconnected"
    r"|broken pipe|timeout|timed out|HTTP 5\d\d|HTTP 429|HTTP 408"
    # A router reports an upstream provider failure as a 200 whose choice
    # carries finish_reason='error'. Measured against OpenRouter: a d32,768
    # trial ran 641s, produced 31,303 reasoning tokens, and then returned
    # finish_reason='error' with no provider named. That is the endpoint
    # failing, not the model declining to answer -- and without this it banked
    # as an unanswered trial, inflating the very rate that decides whether a
    # cell is excluded. `length` is deliberately NOT here: truncation at
    # --max-tokens is the model hitting the budget and is a real outcome.
    r"|finish_reason='error'"
    # 402 is a router out of credits; a truncated transfer is a stream the
    # connection dropped. Both were banked as the model declining before the
    # default below was inverted (gpt-6-sol-pro and GLM respectively).
    r"|HTTP 40[01234]|no choices|payload is not completed", re.I)
# The default is inverted: a non-answer is kept only when it is recognisably
# the *model's*. An endpoint's failure modes are open-ended and a model's are
# not -- probe raises exactly these two for a reply that reached the model --
# so anything unrecognised is quarantined. The old default banked whatever the
# transport list missed, and it missed three: an HTTP 404 from addressing the
# model by repository instead of the serving alias (23 trials in v25), and a
# 200 with no `choices` (2 in v29), each recorded as the model declining to
# answer. Quarantine is cheap -- the file survives under interrupted/ and the
# seed is fixed -- while banking inflates the rate that excludes a cell.
# `transport` still wins over `model`, because finish_reason='error' arrives
# phrased as an incomplete answer.
model = re.compile(r"^incomplete answer: |^no answer content", re.I)
kept = answered or (errors and all(
    model.search(e) and not transport.search(e) for e in errors))
print("no" if kept else "yes")
PY
}

mkdir -p "$OUTDIR"
echo "cell d${DEPTH}: ${N} trials/rung [${RUNGS}] -> ${OUTDIR}"

for rung in ${RUNGS}; do
  for i in $(seq 1 "$N"); do
    # Distinct per (rung, trial) so the two rungs do not draw the same
    # haystack, and so a trial's seed is reproducible from its filename.
    # The offset is resolved outside $(( )), which is arithmetic-only: a
    # string compare there reads the operands as variable names and dies
    # under `set -u`.
    case "$rung" in
      log) rung_offset=0 ;;
      *)   rung_offset=7 ;;
    esac
    seed=$(( BASE_SEED + i * 100 + rung_offset ))
    out="${OUTDIR}/${rung}-t$(printf '%02d' "$i")-s${seed}.json"
    if [ -f "$out" ]; then
      echo "  skip ${rung} t${i} (have $(basename "$out"))"
      continue
    fi
    # A deep trial runs tens of minutes. Starting one that cannot finish before
    # the machine is needed wastes the slot and leaves nothing behind, so the
    # deadline is checked before starting rather than interrupting mid-trial.
    # Every completed trial is already on disk, so stopping early yields a
    # smaller cell, not a lost one.
    if [ -n "${PROBE_DEADLINE:-}" ] && [ "$(date +%s)" -ge "$PROBE_DEADLINE" ]; then
      echo "  STOP: deadline $(date -d "@$PROBE_DEADLINE" '+%F %T') reached;"\
           "$(ls "$OUTDIR"/*.json 2>/dev/null | grep -vc transcripts) trials banked"
      exit 0
    fi
    # A spend cap across the cell. Each trial is its own process, so the probe's
    # own --max-spend would never bind; the banked files carry what each trial
    # cost (`usage.cost` per cell), and that sum is the cell's spend so far.
    # Checked before starting, like the deadline: stopping between trials
    # leaves a smaller cell, never a trial paid for and not recorded.
    if [ -n "${PROBE_MAX_SPEND:-}" ]; then
      spent=$(python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
total = 0.0
for path in glob.glob(os.path.join(sys.argv[1], "*.json")):
    if path.endswith("-transcripts.json"):
        continue
    try:
        report = json.load(open(path))
    except (OSError, ValueError):
        continue
    total += ((report.get("preflight") or {}).get("usage") or {}).get("cost") or 0.0
    for r in report.get("results", []):
        total += (r.get("usage") or {}).get("cost") or 0.0
print(f"{total:.6f}")
PY
)
      if python3 -c "import sys; sys.exit(0 if float('$spent') >= float('$PROBE_MAX_SPEND') else 1)"; then
        echo "  STOP: spend cap \$${PROBE_MAX_SPEND} reached (\$${spent} banked in this cell);"\
             "$(ls "$OUTDIR"/*.json 2>/dev/null | grep -vc transcripts) trials banked"
        exit 0
      fi
    fi
    # One process per trial, enforced rather than assumed.
    #
    # `timeout` calls setpgid, so the probe it starts is the leader of its own
    # process group, *not* a member of this script's. Killing this script's
    # group therefore leaves the running trial alive as an orphan -- and a
    # resumed cell, seeing no result file yet, starts the same trial again.
    # Observed: two processes on seed 327680407 for 2.5 hours, halving each
    # other's throughput by competing for the endpoint, and racing to write one
    # `--save-result` and one `--save-transcripts` between them -- an interleave
    # can pair one process's result with the other's transcripts.
    #
    # The lock makes that impossible however the duplicate arises, including a
    # second cell started by hand in the same PROBE_OUTDIR. It carries no
    # `.json` suffix so probe-pool.py cannot read it as a result.
    exec 9>"${OUTDIR}/.lock-$(basename "${out%.json}")"
    if ! flock -n 9; then
      echo "  skip ${rung} t${i} (another process holds this trial)"
      exec 9>&-
      continue
    fi
    attempt=1
    while : ; do
      if [ "$attempt" -eq 1 ]; then
        echo "  run  ${rung} t${i} seed=${seed}"
      else
        echo "  run  ${rung} t${i} seed=${seed} (attempt ${attempt}/${MAX_ATTEMPTS})"
      fi
      # `timeout` is the guard against a hang rather than a failure. The
      # request is not streamed -- probe awaits one whole JSON body -- so its
      # socket read timeout is necessarily None: a legitimate deep trial sends
      # nothing for over an hour and any read deadline short enough to catch a
      # half-open connection would kill it. The wall clock is the only tell
      # left, so it is checked out here and set generously.
      timeout "$TRIAL_TIMEOUT" uv run llm-assay.py probe "$BASE_URL" \
        --model "$MODEL" \
        ${EXTRA_BODY:+--extra-body "$EXTRA_BODY"} \
        --tokenizer "$TOKENIZER" \
        --depths "$DEPTH" \
        --rungs "$rung" \
        --trials 1 \
        --seed "$seed" \
        --thinking on \
        --max-tokens "$MAX_TOKENS" \
        --save-result "$out" \
        --save-transcripts "${out%.json}-transcripts.json" \
        >>"${OUTDIR}/run.log" 2>&1
      status=$?
      [ $status -eq 0 ] && break

      # Exit 3 is the probe's pre-flight refusing the configuration -- a wrong
      # model name or provider pin, rejected with an HTTP 4xx before any trial.
      # It fails identically on every attempt, so the backoff below would only
      # spend up to an hour proving it. Nothing was measured and nothing is
      # banked; fix the command and re-run.
      if [ "$status" -eq 3 ]; then
        echo "  STOP: the endpoint rejected this configuration before any trial (see ${OUTDIR}/run.log)."
        echo "        Not retried: a wrong model name or provider pin fails the same way every time."
        exit 3
      fi

      # A probe that measured nothing exits non-zero by design, and two very
      # different things arrive here wearing that exit code.
      #
      # A trial the *model* did not answer -- `incomplete answer`, `no answer
      # content` -- is a real outcome the design reads: the unanswered rate is
      # reported per cell and past 20% it invalidates the arm. It must be kept,
      # and kept exactly once. Retrying it until it answers would select for
      # the trials that answer and quietly destroy that rate.
      #
      # A trial that never reached the model is not a trial at all. Measured
      # against a dead port: probe exits 1 and leaves a *complete, well-formed*
      # file recording `answered: 0` with `Cannot connect to host` in `errors`
      # -- which the skip check above then banks forever. Over an arm that runs
      # for a day, one llama.cpp restart would enter the record as the model
      # failing to answer.
      #
      # Retrying that is not resampling: the seed is fixed, so the retry builds
      # the identical haystack and asks the identical question. What repeats is
      # an attempt, not a measurement.
      if [ -f "$out" ] && [ "$(_transport_failure "$out")" = "no" ]; then
        echo "    ^ exit ${status}: the model did not answer (kept; counts as unanswered)"
        break
      fi

      # Quarantined rather than deleted, and into a subdirectory because
      # probe-pool.py lists one level and pools `*.json`: the evidence survives
      # for an auditor without ever reaching a pooled number.
      mkdir -p "${OUTDIR}/interrupted"
      stamp="$(date +%Y%m%d-%H%M%S)"
      for f in "$out" "${out%.json}-transcripts.json"; do
        [ -f "$f" ] && mv "$f" "${OUTDIR}/interrupted/$(basename "${f%.json}")-${stamp}.json"
      done
      if [ "$status" -eq 124 ]; then
        echo "    ^ no result after ${TRIAL_TIMEOUT}s; treating as a hung endpoint"
      else
        echo "    ^ exit ${status}: the endpoint did not answer (not the model); retrying same seed"
      fi

      attempt=$(( attempt + 1 ))
      if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        # Marching the remaining trials into a dead endpoint would quarantine
        # every one of them: nothing lost, but nothing gained either, and the
        # arm would look finished. Stopping is resumable -- re-run the same
        # line and it picks up at this trial.
        echo "  STOP: ${MAX_ATTEMPTS} attempts at ${rung} t${i} and the endpoint never answered."
        echo "        $(ls "$OUTDIR"/*.json 2>/dev/null | grep -vc transcripts) trials banked."
        echo "        Resume with the same command once it is back."
        exit 1
      fi
      echo "    waiting ${backoff}s for the endpoint"
      sleep "$backoff"
      backoff=$(( backoff * 2 ))
      [ "$backoff" -gt "$BACKOFF_CAP" ] && backoff="$BACKOFF_CAP"
    done
    exec 9>&-
    backoff="$BACKOFF_START"
  done
done

echo "done. pool with: uv run scripts/probe-pool.py ${OUTDIR}"
````

## Appendix B: `probe-pool.py`

The pooling and testing script, as `scripts/probe-pool.py` in the repository:
`DIR` pools cells, `--compare` runs the registered Fisher test, `--paired` the
McNemar tests on trials matched by seed.

````python
#!/usr/bin/env python3
"""Pool per-trial probe results into a cell.

`scripts/probe-cell.sh` writes one file per (rung, trial) so a crash costs one
trial instead of a whole cell. This sums them back into the table `probe` would
have printed, recomputing the interval with the same Wilson function rather
than a second implementation of it.

Accepts whole-cell files too, so a directory holding a mix pools correctly.

With --compare it runs the pre-registered comparison in
the programme's pre-registration instead: Fisher exact, two-sided, on
fabricated against non-fabricated *answered* trials, between exactly two arms.
The statistics live in `llm_assay.probe` rather than here, because this file is
outside mypy and pytest and these are published numbers.

With --paired it matches two arms' per-trial files by (seed, depth) -- the same
haystack, byte for byte -- and runs exact McNemar twice: on "produced the
correct answer" over every pair, which the unanswered rule cannot void, and on
the pairs where both arms answered.

usage: uv run scripts/probe-pool.py DIR [DIR ...]
       uv run scripts/probe-pool.py --compare SHALLOW_DIR DEEP_DIR
       uv run scripts/probe-pool.py --paired ARM_A ARM_B [RUNG]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from llm_assay.probe import (  # noqa: E402
    FALSE_POSITIVE_KINDS, _build_summary, fisher_exact, floor_note, wilson,
)
from llm_assay.pooling import compare_arms, paired_compare  # noqa: E402
from llm_assay.pooling import pool as _pool  # noqa: E402


def pool(targets):
    """The six-tuple this script has always unpacked.

    The pooling itself moved to `llm_assay.pooling` once these numbers became
    published ones: `scripts/` is outside mypy and pytest, and the standing rule
    is that such code moves into `src/` with tests -- the same rule that put
    `wilson` and `fisher_exact` in `probe.py`.
    """
    return _pool(targets).as_tuple()


def main():
    targets = sys.argv[1:]
    if targets and targets[0] == "--compare":
        if len(targets) != 3:
            print("--compare takes exactly two directories", file=sys.stderr)
            return 2
        return compare(targets[1], targets[2])
    if targets and targets[0] == "--paired":
        if len(targets) not in (3, 4):
            print("--paired takes two directories and an optional rung", file=sys.stderr)
            return 2
        return paired(targets[1], targets[2], targets[3] if len(targets) == 4 else "clean")
    if not targets:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    cells, fallbacks, served, builds, identified, sources = pool(targets)
    if not cells:
        print("no results found", file=sys.stderr)
        return 1

    # A pooled cell whose trials were measured against different weights, or
    # with a substituted tokenizer, is not one cell. Say so above the table
    # rather than letting the sum imply it was.
    if fallbacks:
        print(f"WARNING: tokenizer fallback in some trials: {sorted(fallbacks)}")
    if len(served) > 1:
        print(f"WARNING: pooled across different served models: {sorted(served)}")
    if len(builds) > 1:
        # The check that bites on a box whose alias is pinned: `served` is one
        # string there whatever was loaded, so this is the only tell that the
        # weights changed under a resumed cell.
        print("WARNING: pooled across different server builds:")
        for build in sorted(builds):
            print(f"         {_build_summary(dict(build))}")

    def _of(label, values, kind):
        """Name the one value found, and say if it was not found everywhere."""
        if len(values) != 1:
            return None
        shown = values.pop()
        if kind == "build":
            shown = _build_summary(dict(shown))
        if identified[kind] < sources:
            return f"{label}: {shown} (recorded in {identified[kind]} of {sources}; rest unknown)"
        return f"{label}: {shown}"

    pooled_all = _pool(targets)
    if len(pooled_all.efforts) > 1 or (
            pooled_all.efforts and pooled_all.identified["effort"] < sources):
        print("WARNING: pooled across different reasoning_effort settings: "
              f"{sorted(pooled_all.efforts)}"
              + (" and the vendor default" if pooled_all.identified["effort"] < sources else ""))
    print(f"\npooled {sources} file(s)")
    if len(pooled_all.efforts) == 1 and pooled_all.identified["effort"] == sources:
        print(f"reasoning_effort: {next(iter(pooled_all.efforts))}")
    for line in (_of("served", served, "served"), _of("build", builds, "build")):
        if line:
            print(line)
    print()
    pooled = _pool(targets)
    floors = pooled.floors()
    print("| depth | rung | accuracy | 95% CI | false+ | fabricated | misquoted "
          "| only_real | uncited | unanswered | floor |")
    print("|------:|:-----|---------:|:-------|-------:|-----------:|----------:"
          "|----------:|--------:|-----------:|:------|")
    for (depth, rung) in sorted(cells):
        c = cells[(depth, rung)]
        lo, hi = wilson(c["ok"], c["answered"])
        print(f"| {depth} | {rung} | {c['ok']}/{c['answered']} | "
              f"[{lo:.2f}, {hi:.2f}] | {c['false+']} | {c['fabricated']} | "
              f"{c['misquoted']} | {c['only_real']} | {c['uncited']} | "
              f"{c['trials'] - c['answered']} | "
              f"{floor_note(depth, rung, floors) or '-'} |")
    print()
    # The gate every pre-registration from v17 on computed by hand. Pool the
    # floor's directory with the cell's to have it checked here.
    for w in pooled.floor_warnings():
        print(f"WARNING: {w}")
    for n in pooled.notes():
        print(f"note: {n}")
    # The three buckets have to account for every false positive, or one of
    # them is being dropped silently. A file written before they existed shows
    # up here rather than as a quietly wrong rate.
    for (depth, rung), c in sorted(cells.items()):
        classified = sum(c[k] for k in FALSE_POSITIVE_KINDS)
        if classified != c["false+"]:
            print(f"WARNING: d{depth} {rung}: {c['false+']} false positive(s) but "
                  f"{classified} classified -- some trials predate the buckets")
    return 0


def compare(shallow, deep):
    """Render the pre-registered test. The decisions live in llm_assay.pooling.

    Everything that decides a number, or refuses to produce one, moved into
    `src/` once these became published numbers -- the same rule that put
    `wilson` and `fisher_exact` in `probe.py`. What is left here is printing.
    """
    result = compare_arms(shallow, deep)
    for w in result.warnings:
        print(f"WARNING: {w}")
    if result.error:
        print(result.error, file=sys.stderr)
        return 1

    print("\npre-registered comparison "
          "(the programme's pre-registration)\n")
    print("| arm | answered | unanswered | fabricated | rate | 95% CI | misquoted |")
    print("|:----|---------:|-----------:|-----------:|-----:|:-------|----------:|")
    for arm in result.arms:
        c = arm.cell
        lo, hi = wilson(arm.fabricated, arm.answered)
        rate = f"{arm.fabricated / arm.answered:.1%}" if arm.answered else "--"
        print(f"| {os.path.basename(arm.target.rstrip('/'))} | {arm.answered} "
              f"| {arm.unanswered} | {arm.fabricated} | {rate} "
              f"| [{lo:.2f}, {hi:.2f}] | {c['misquoted']} |")
    p = result.p
    print(f"\nFisher exact, two-sided: p = {p:.4f}  "
          f"({'reject' if p < 0.05 else 'do not reject'} H0 at 0.05)")
    if p >= 0.05:
        print("A null at this n is a statement about the experiment: power is "
              "0.50 against\nthe effect the prior estimated. Report the intervals, "
              "not 'no effect'.")
    return 0


def paired(a, b, rung):
    """Render the paired test. The decisions live in llm_assay.pooling."""
    result = paired_compare(a, b, rung)
    for w in result.warnings:
        print(f"WARNING: {w}")
    if result.error:
        print(result.error, file=sys.stderr)
        return 1
    names = [os.path.basename(t.rstrip("/")) for t in (a, b)]
    print(f"\npaired on (seed, depth), rung {rung}: {result.pairs} pair(s)\n")
    print(f"| test | pairs | correct: {names[0]} | correct: {names[1]} "
          "| discordant | McNemar p |")
    print("|:-----|------:|------:|------:|:----|------:|")
    for label, t in (("all trials (unanswered = not correct)", result.all_trials),
                     ("complete pairs (both answered)", result.complete_pairs)):
        print(f"| {label} | {t['pairs']} | {t['correct'][0]} | {t['correct'][1]} "
              f"| {t['discordant'][0]} vs {t['discordant'][1]} | {t['p']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````
