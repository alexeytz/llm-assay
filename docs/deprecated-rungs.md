# Removed probe rungs, and why

Three rungs were built, measured, and deleted: `near`, `far` and `corpus`. They
are recorded here because the reason they failed is more useful than the code
was, and because anyone building a contradiction-based long-context probe on
natural text will reach for the same designs.

All numbers below are Qwen3.8-27B-FP8 on vLLM, 262k context.

## `near` / `far` - injected invented sentences

Two invented statements contradicting each other about an invented person, one
paragraph apart (`near`) or spanning most of the context (`far`).

**What they actually measured: how eye-catching the injected prose was.**

The second half was originally reported speech - "Count X *was saying* that he
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

## `corpus` - a real sentence with one word swapped

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

Both rungs stayed at 7-8/10 at 2,048 tokens, where the task should be trivial -
so neither was calibrated well enough for a depth measurement to mean anything.
They were kept for a while as documented negative results.

Documentation is weaker than deletion. Someone runs `--rungs near`, gets a
number, and publishes it. This codebase already refuses to publish misleading
figures elsewhere: `peak t/s` is omitted entirely when the tokens carry no
interval, and `compare` exits non-zero rather than printing a table where
nothing was compared. Shipping a rung known to measure an artifact contradicts
that.

## What replaced them

The `log` rung, which abandons natural text entirely. It generates its own
haystack, every line from one template:

```
[{date}] At {location}, {rank} {name} {action} {object}.
```
```
[2142-09-12] At Harbour Vault, Engineer Vasquez monitored the plasma conduits.
[2142-06-25] At Sector Four, Steward Vorst inspected the coolant loop.
```

That design answers each failure above directly:

-   **Authorial salience** cannot arise, because no line is authored. The needle
    is two lines in the same template as the other three hundred, differing only
    in which vocabulary items they draw.
-   **Structural repetition** cannot betray the needle, because repetition is
    everywhere by construction - and the generator plants *benign* same-name,
    same-date repeats at a single location on purpose. Without them, "the name
    that occurs twice on one date" would solve the task without reading a
    location at all, which is the `corpus` rung's shortcut in a new costume.
-   **The tension is dissolved rather than escaped.** A contradiction needs two
    statements about one fact; on natural text those are either near-identical
    (so duplication shows) or differently worded (so authorship shows). On
    generated text the two halves are *identical in form* and differ only in the
    field that matters - the location - so neither tell is available.
-   **Parametric memory is cut out.** The fifty surnames are invented, so a
    correct answer cannot come from anywhere but the context, and scoring needs
    no second model: the answer is checked against the log's own dictionary and
    must match exactly one name.

The generator also guarantees what the corpus rungs could only hope for: every
`(name, date)` in the background resolves to exactly one location, so the only
impossibility is the injected one. That is checkable after the fact - regenerate
the haystack from the seed and scan it - and has been checked on every false
positive worth believing.

It reaches 10/10 at 2,048 tokens, where `near` and `corpus` sat at 7-8/10 and
therefore could not support a depth measurement at all.

`clean` and `outlier` were later moved onto this same generated haystack for the
same reason. A control drawn from a different distribution than the measurement
reports a false-positive rate that does not transfer: `clean` once sliced the
book, so it measured how often the model invents a contradiction *in Tolstoy*,
which says nothing about how often it invents one in an event log.
