# Removed probe rungs, and why

Three rungs were built, measured, and deleted: `near`, `far` and `corpus`. They
are recorded here because the reason they failed is more useful than the code
was, and because anyone building a contradiction-based long-context probe on
natural text will reach for the same designs.

All numbers below are Qwen3.8-27B-FP8 on vLLM, 262k context.

## `near` / `far` — injected invented sentences

Two invented statements contradicting each other about an invented person, one
paragraph apart (`near`) or spanning most of the context (`far`).

**What they actually measured: how eye-catching the injected prose was.**

The second half was originally reported speech — "Count X *was saying* that he
had wintered at Arkhangelsk". Rewriting both halves as narrator assertions was a
correctness fix, made for its own sake: reported speech is a character lying,
not a textual contradiction, so a model answering `CONSISTENT` had a defensible
case and the scorer was marking it wrong.

That fix erased the headline result:

| injection | near @0.5 | far | near vs far |
|---|---|---|---|
| reported speech | 4/20 | 17/20 | p = 3.9e-05 |
| narrator only | 6/20 | 8/20 | p = 0.51 |

The gap closed because `far` **fell**, 17/20 to 8/20 (p = 0.003), not because
`near` rose. The original sentence was vivid and distinctive; at distance you
must notice that half on its own, so conspicuousness carried it, while adjacent
placement finds it either way.

Two hypotheses were tested against this and refuted. Position: controlling it
made the gap *larger*, and `near @0.1` on the supposedly privileged start scored
worse than `@0.5`. Defeasibility: removing the lying-character reading moved
`near` by 4/20 → 6/20, p = 0.47.

## `corpus` — a real sentence with one word swapped

A sentence lifted from the corpus with one word swapped for its antonym
(`began` → `ceased`), so register and vocabulary matched by construction rather
than by the author's skill.

**What it actually measured: structural repetition.**

The injected sentence is a near duplicate of a real one, and duplication is
conspicuous in its own right. That shows in the failure mode: `corpus` keeps
*noticing* at depth where `near` does not, and instead gets the logic wrong.

| rung | depth | noticed | correct when noticed |
|---|---|---|---|
| near | 2k → 16k | 8/10 → 1/10 | 8/8 → 1/1 |
| corpus | 2k → 16k | 9/10 → 8/10 | 7/9 → 5/8 |

So it traded "the author's prose is salient" for "duplication is salient". An
improvement, since it no longer depends on anyone's writing, but not a solution.

## The tension this exposes

A contradiction needs two statements about one fact. Two corpus-derived
statements about one fact are near-identical by construction, so duplication
shows. Make them differ enough to avoid that and you are authoring one again,
so authorial salience returns. On natural narrative text there does not appear
to be a way out.

## Why deleted rather than documented

Both rungs stayed at 7–8/10 at 2,048 tokens, where the task should be trivial —
so neither was calibrated well enough for a depth measurement to mean anything.
They were kept for a while as documented negative results.

Documentation is weaker than deletion. Someone runs `--rungs near`, gets a
number, and publishes it. This codebase already refuses to publish misleading
figures elsewhere: `peak t/s` is omitted entirely when the tokens carry no
interval, and `compare` exits non-zero rather than printing a table where
nothing was compared. Shipping a rung known to measure an artifact contradicts
that.

The replacement is the `log` rung, which generates a uniform synthetic corpus
where the needle is drawn from the same distribution as the haystack. It reaches
10/10 at 2,048 tokens.
