# Quick start

From a fresh machine to a first verdict on your endpoint. Each step stands on its own,
so stop wherever you have your answer. The [README](../README.md) explains what every
number means; this page only gets you running.

You need: Linux or macOS, `git`, `curl`, and an OpenAI-compatible endpoint (vLLM,
llama.cpp, SGLang, or a hosted API). The examples use `http://localhost:8000/v1`;
put your own URL in its place, including the `/v1`.

## 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv, if you do not have it
git clone https://github.com/alexeytz/llm-assay.git
cd llm-assay
uv run llm-assay.py --help                           # first run resolves the dependencies
```

There is no install step beyond that. If the endpoint needs a key, put it in the
environment, never on the command line, where every user on the machine can read it:

```bash
export LLM_ASSAY_API_KEY=...
```

## 2. Let `tune` size the runs

```bash
uv run llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh
./run-local.sh --help
```

`tune` reads the model, context window, prefix caching, speculative decoding, KV cache
and which thinking settings the server honours, and writes a runner whose presets are
sized to that server. Read its report once: a `note:` line or a "rejected" thinking
level is something to know before measuring.

## 3. Check the deployment is served right

```bash
./run-local.sh qualify
```

A canary reply (with thinking off as well, where the server can switch it off), the smoke benchmark, a short probe and, if
`tune` found an agent harness on `PATH`, three agent runs. It ends with a verdict
(`results/qualify-<time>/VERDICT.txt`):

- **PASS** - go on.
- **WARN** - read the line; it names something that will colour the numbers (another
  client on the server, a tokenizer that did not load).
- **FAIL** - fix the server first. The usual one is the think block arriving inside the
  answer, which means the server's reasoning parser is missing or wrong; every later
  number would be measuring that.

On a reasoning model the probe step takes a few minutes. `EXTRA=--no-probe` skips it.

## 4. Measure speed

```bash
./run-local.sh smoke     # alive and sane: a minute
./run-local.sh sweep     # how prefill and decode cost grow with context depth
./run-local.sh decode    # decode only, sampled until the confidence interval is tight
```

Results are JSON under `results/`. `RUNS=10 ./run-local.sh sweep` overrides the run
count; `EXTRA="..."` appends any benchmark flag.

## 5. Did a change to the server help?

Measure, change the server, measure again. The A/B keeps its seed, so both sides read
the same text and the comparison is paired:

```bash
NOTE="kv auto" ./run-local.sh ab-before
# ... change the server and restart it ...
NOTE="kv fp8" ./run-local.sh ab-after
cat results/ab-deploy/VERDICT.txt
```

The verdict says what the endpoint reports changed, compares prefill and decode, and
warns when the after side was served from the prefix cache (the server was not
restarted, so its prefill is not a result). `AB=name` keeps several A/Bs apart.
`EXTRA="--agent auto"` on `ab-before` adds an agent arm to both sides, for a change
that is meant to leave the model's behaviour alone.

To compare any two saved runs yourself:

```bash
uv run llm-assay.py compare results/a.json results/b.json
```

## 6. Does the model still use its context?

```bash
./run-local.sh probe
```

A haystack at several depths with something to find, something to reason about, and a
control with nothing wrong in it. The control is the one to watch: a model that invents
a problem in a clean document is not reading it. On a reasoning model this preset runs
for hours; `tune`'s report estimates how long from the decode rate it measured.

## 7. Does it do the right thing with tools?

The agent suite hands a coding harness a small repository with a trap in it and grades
the files afterwards. It needs Linux, `bubblewrap` and a harness. Pi is the default:

```bash
sudo apt install bubblewrap                          # or your distribution's package
npm install -g @earendil-works/pi-coding-agent       # needs Node.js 22.19 or later
uv run llm-assay.py tune http://localhost:8000/v1 --write ./run-local.sh   # now finds pi
RUNS=1 EXTRA="--tasks sync" ./run-local.sh agent-pi  # one quick run first
./run-local.sh agent-pi                              # the hard tasks, 5 runs each
```

On Ubuntu 23.10 and later the sandbox's self-test may fail because AppArmor blocks
bubblewrap's user namespace; it prints the profile to install. Pin the model card's
sampling with `EXTRA="--sampling temperature=... top_p=..."`, or the result says it was
not pinned. Hermes, OpenCode and OMP work the same way once installed (`agent-hermes`,
...).

## Next

- [README](../README.md): what each metric, warning and exit code means, and the traps
  that make a plausible number wrong.
- [`skills/llm-assay`](../skills/llm-assay/SKILL.md): the same workflow as an Agent
  Skill, for a coding agent to drive.
- [Agent tasks](agent-tasks/) and [what the suite found across harnesses](agent-harnesses.md).
