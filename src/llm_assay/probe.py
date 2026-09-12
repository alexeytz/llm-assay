"""Anomaly-injection quality probe: can the model still *use* the context?

llm-assay measures how fast tokens arrive. It says nothing about whether they
are any good, which means it cannot tell "faster" from "faster because worse" --
and every knob it sweeps has a quality dimension. Quantized KV, sliding-window
attention and chunked prefill all make long context cheaper, and all of them can
quietly cost comprehension at exactly the depths the throughput table is
celebrating.

This probe generates a synthetic event log to a controlled depth and asks the
model to find what is wrong with it. Scoring needs no second model and no ground
truth about any corpus, because the anomalies are invented here: a correct
answer has to contain a name that exists in no training set.

Three rungs over one generated haystack and one question:

``clean``    nothing injected. Not optional -- asking "is anything wrong?" primes
             a model to find something, and only these trials say how often it
             invents one.
``outlier``  one line from another domain entirely. This is needle-in-a-
             haystack, and it is here as a *floor*: it separates "cannot see the
             text at that depth" from "sees it and cannot reason about it".
``log``      one logical impossibility -- a person in two places on one date.
             This is the measurement.

Measured on a 262k-context Qwen3.8-27B: at 32k the model found an injected
outlier 3/3 while missing a contradiction 0/3. A needle-in-a-haystack test would
have called that context healthy. That gap is the reason the `log` rung exists.

Rungs sharing one haystack is what makes `clean` a control: three earlier rungs
sliced a book while `log` generated its own text, so the "control" measured how
often the model invents a contradiction *in Tolstoy*, a rate that does not
transfer. `docs/deprecated-rungs.md` records why those three were deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from . import __version__
from .client import LLMClient, _answer_text
from .corpus import load_tokenizer, tokenizer_fallback as _tokenizer_fallback

# Written in the register of the surrounding prose on purpose. An injection that
# reads as foreign is detectable without understanding it, which measures
# retrieval and calls it comprehension.
OUTLIER = (
    ">>> import os\n>>> os.path.join('usr', 'local')\n'usr/local'\n"
    "Parameters: *paths* (str) -- path components to join. Returns: str."
)


# --- the synthetic event log -------------------------------------------------
# Natural prose cannot escape salience: an authored injection carries the
# author's voice, and a mutated corpus sentence carries structural repetition.
# Both were measured doing exactly that. A procedurally generated log removes
# the choice -- every sentence shares one template, so the needle is drawn from
# the same distribution as the haystack and has nothing to stand out by.
_LOG_LOCATIONS = ("Outpost Delta", "Station Prime", "Sector Four", "Lunar Base",
                  "Relay Nine", "Harbour Vault", "Ridge Post", "Deep Array",
                  "Cinder Yard", "North Pylon")
_LOG_RANKS = ("Engineer", "Technician", "Warden", "Analyst", "Steward")
_LOG_NAMES = (
    "Vance", "Thorne", "Kaelen", "Elara", "Rusk", "Palgrave", "Oduya", "Sarn",
    "Trevik", "Mowbray", "Cassar", "Ibsen", "Renko", "Halloway", "Zabel",
    "Oyelaran", "Petrak", "Fenn", "Muraoka", "Aldritch", "Sowinski", "Behr",
    "Calloway", "Nakamura", "Ferreira", "Odell", "Vasquez", "Lindqvist",
    "Achebe", "Brannigan", "Sokolov", "Marchetti", "Nwosu", "Farr", "Tiller",
    "Ravel", "Oyelowo", "Kestrel", "Danforth", "Ashby", "Imani", "Vorst",
    "Quillon", "Sabra", "Teague", "Ulrich", "Wexler", "Yarrow", "Zorin", "Bex",
)
_LOG_ACTIONS = ("calibrated", "inspected", "replaced", "monitored", "repaired",
                "logged", "sealed", "tested", "realigned", "catalogued")
_LOG_OBJECTS = ("the thermal manifold", "the orbital tethers", "the sensor arrays",
                "the plasma conduits", "the coolant loop", "the docking clamps",
                "the relay housing", "the pressure seals", "the guidance stack")

# One question for every rung, which is what makes `clean` a control rather than
# a separate experiment: the model is asked exactly the same thing whether or not
# anything was injected, so its answer on an untouched log *is* its
# false-positive rate. Note it does not promise an impossibility exists -- saying
# "there is exactly one" would make CONSISTENT unavailable and destroy the
# control.
LOG_QUESTION = (
    "You are analyzing a chronological security log.\n\n"
    "A logical impossibility would be a single person recorded in two completely "
    "different locations on the exact same date.\n\n"
    "Read the log carefully, then reply with ONE line:\n"
    "1. If you find such an impossibility, reply with ONLY the surname of the "
    "person involved.\n"
    "2. Otherwise, if some line does not belong in a log of this kind at all, "
    "reply with ONLY: OFFTOPIC\n"
    "3. Otherwise reply with ONLY: CONSISTENT"
)

# Which needle each rung puts in the same generated haystack.
_NEEDLES = {"clean": "none", "outlier": "foreign", "log": "impossible"}

RUNGS = ("clean", "outlier", "log")

# What a reasoning trace has to contain to prove the model *located* the
# injection, per rung. Deliberately not `expected`: `outlier` expects the
# literal OFFTOPIC, which a model writes whenever it settles on that answer, so
# finding it in the trace proves it answered, not that it read anything. The
# injected snippet occurs nowhere else in a generated log, so finding *that* is
# retrieval, and it separates the two ways a rung can fail -- never saw the
# text, versus saw it and judged it fine.
#
# `log` has no entry on purpose: its culprit's surname recurs throughout the
# benign entries -- that repetition is the shortcut defence -- so the name is in
# the trace whether or not the collision was found. A run reported saw=10/10
# against accuracy 5/10, which reads as flawless retrieval and was an artifact
# of the name simply being common in the text.
TRACE_MARKERS: Dict[str, str] = {"outlier": "os.path.join"}

# Rungs whose injection can be placed anywhere. `clean` injects nothing, so it
# has no position to sweep and is pinned.
POSITIONABLE = ("outlier", "log")


class ProbeError(RuntimeError):
    """A request that did not answer.

    Kept distinct from a wrong answer on purpose. Scoring a 404 as a MISS turns
    a broken invocation into a confident quality measurement -- every rung reads
    0/N, including the retrieval floor that is supposed to be trivial, and the
    table looks like a model that has collapsed rather than a model that was
    never asked.

    It carries whatever the model *did* produce. A trial that fails this way is
    the one a reader most wants to look at -- "did it run out of budget, or stop
    without answering, and was it a hard trial?" -- and discarding the trace at
    the raise leaves the transcript blind to exactly those trials.
    """

    def __init__(self, message: str, reasoning: str = "",
                 finish_reason: Optional[str] = None):
        super().__init__(message)
        self.reasoning = reasoning
        self.finish_reason = finish_reason


def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% interval for a proportion.

    A Student-t interval is for a mean and is wrong on a 0/1 rate -- badly so
    near 0 and 1, which is exactly where a probe that is working sits. At 0/6 it
    would report +/-0, claiming certainty from the least informative result
    there is.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))



def span(position: float, gap: float = 0.6) -> Tuple[float, float]:
    """Where the contradiction's two halves go, as fractions of the log.

    A contradiction has two ends, so "position" has to mean the centre between
    them -- and a pair centred at 0.9 cannot also be 0.6 of the log wide. The
    span is therefore narrowed symmetrically until it fits, which keeps the
    requested centre exact and makes *separation* the thing that varies.

    The first version clamped the two indices independently instead. That moved
    the centre as well: `--positions 0.1` produced a pair centred at 0.20 with
    its halves 0.40 apart rather than 0.60, and the table printed `at 0.10`.
    A sweep meant to isolate position was varying position *and* distance, and
    reporting a placement the log did not have -- the same confound that got
    the `near`/`far` rungs deleted, in mirror image.

    The residual trade is inherent and is reported rather than hidden: the
    `sep` column says how far apart the halves actually were.
    """
    half = min(gap / 2.0, position, 1.0 - position)
    return position - half, position + half


def _at(paragraphs: List[str], fraction: float) -> int:
    """Paragraph index for a position given as a fraction of the context."""
    last = max(1, len(paragraphs) - 1)
    return max(1, min(last, int(round(fraction * last))))


def build_context(
    tokenizer: Any, depth: int, rung: str, rng: random.Random,
    position: float = 0.5,
) -> Tuple[str, Optional[str]]:
    """Return (context, the marker a correct answer must contain).

    ``position`` is where the injection goes, as a fraction of the context.
    It is a parameter rather than a random draw because attention is strongly
    position-dependent -- the ends are privileged and the middle is not -- so
    leaving it to chance buries a large effect inside every other number.

    An earlier pair of rungs drew one injection's position at random while
    pinning the other's to 0.2 and 0.8, which are both privileged. The
    comparison that was supposed to isolate *distance* was therefore partly
    measuring *position*. Position is a parameter here for that reason.
    """
    needle = _NEEDLES.get(rung)
    if needle is None:
        raise ValueError(f"unknown rung: {rung}")
    return build_log(depth, position, rng, tokenizer, needle=needle)



def _log_entry(rng: random.Random, name: str, date: str, place: str) -> str:
    return (f"[{date}] At {place}, {rng.choice(_LOG_RANKS)} {name} "
            f"{rng.choice(_LOG_ACTIONS)} {rng.choice(_LOG_OBJECTS)}.")


def build_log(
    depth: int, position: float, rng: random.Random, tokenizer: Any,
    gap: float = 0.6, needle: str = "impossible",
) -> Tuple[str, Optional[str]]:
    """A procedurally generated log with exactly one impossible entry.

    Every line shares one template, so the needle is drawn from the same
    distribution as the haystack: no authorial voice to stand out by, no
    structural repetition to act as an attention magnet, and no famous text for
    the model's parametric memory to override the context with.

    Two properties the generator has to guarantee, or the measurement is void:

    * **No accidental impossibility.** Background entries are constrained so a
      given (name, date) always resolves to one location. Without that, a slice
      can contain a second violation and a "wrong" answer may be right.
    * **Benign same-name, same-date repeats.** Otherwise "the name that appears
      twice on one date" is a shortcut that solves the task without reading the
      locations at all.
    """
    days = [f"2142-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    placed: Dict[Tuple[str, str], str] = {}

    def background() -> str:
        # Reuse an existing (name, date) about a fifth of the time. Chance
        # collisions are far too rare at these sizes -- one per 90 entries --
        # and without deliberate benign repeats "the name and date that occur
        # twice" solves the task without ever reading a location.
        if placed and rng.random() < 0.2:
            name, date = rng.choice(list(placed))
        else:
            name, date = rng.choice(_LOG_NAMES), rng.choice(days)
        # One location per (name, date), so the only impossibility is the one
        # deliberately injected below.
        place = placed.setdefault((name, date), rng.choice(_LOG_LOCATIONS))
        return _log_entry(rng, name, date, place)

    lines: List[str] = []
    budget = max(1, depth)
    # Grow to the token budget, measuring rather than guessing, then trim. The
    # log is built to length instead of sliced to length, so no entry is cut in
    # half at either boundary.
    while True:
        lines.extend(background() for _ in range(64))
        if len(tokenizer.encode("\n".join(lines), add_special_tokens=False)) >= budget:
            break
    while len(lines) > 2 and len(
            tokenizer.encode("\n".join(lines), add_special_tokens=False)) > budget:
        lines.pop()

    if needle == "none":
        # The control has to come from the same distribution as the
        # measurement. A false-positive rate measured on Tolstoy says nothing
        # about how often the model invents an impossibility in an event log:
        # different text, different question, different prior.
        return "\n".join(lines), None

    if needle == "foreign":
        # The retrieval floor, on the same haystack. A line from another domain
        # entirely -- conspicuous the way a needle-in-a-haystack needle is --
        # so that failing here means the context is not reaching the model,
        # rather than that the impossibility was too subtle.
        lines.insert(_at(lines, position), OUTLIER.replace("\n", " "))
        return "\n".join(lines), "OFFTOPIC"

    # The needle: one person, one date, two different locations.
    culprit = rng.choice(_LOG_NAMES)
    date = rng.choice(days)
    here, there = rng.sample(_LOG_LOCATIONS, 2)
    placed[(culprit, date)] = here

    lo_frac, hi_frac = span(position, gap)
    lo_at = max(0, min(len(lines) - 1, int(lo_frac * len(lines))))
    hi_at = max(lo_at + 1, min(len(lines), int(hi_frac * len(lines))))
    lines.insert(lo_at, _log_entry(rng, culprit, date, here))
    lines.insert(hi_at, _log_entry(rng, culprit, date, there))
    return "\n".join(lines), culprit



def score(reply: str, expected: Optional[str],
          decoys: Optional[Sequence[str]] = None) -> str:
    """OK / PARTIAL / MISS / FALSE+.

    Three outcomes for an injected trial, not two. A model that spots the
    injected text but reports it as an odd passage instead of naming the person
    has not failed to comprehend, it has answered in the wrong shape; folding
    that into MISS understates the model and overstates how fast quality decays.
    """
    consistent = bool(re.search(r"\bCONSISTENT\b", reply, re.I))
    offtopic = bool(re.search(r"\bOFFTOPIC\b", reply, re.I))

    if expected is None:
        return "OK" if consistent else "FALSE+"

    if expected == "OFFTOPIC":
        return "OK" if offtopic else ("MISS" if consistent else "PARTIAL")

    # Classification is checked before the name, or the score leaks: a reply of
    # "The passage about Count Vasnetsov is in a different style. OFFTOPIC"
    # contains the surname while explicitly declining to call it a
    # contradiction, and a bare substring test scored that as a full pass.
    if offtopic:
        return "PARTIAL"
    if expected.lower() in reply.lower():
        # Naming the right person among several is not answering. "The
        # inconsistency is between Oyelaran and Imani" contains the expected
        # surname and scored a full pass, which rewards listing suspects: with
        # fifty names a shotgun of five carries a one-in-ten chance of holding
        # the answer for no understanding at all. One surname was asked for.
        others = [n for n in decoys or ()
                  if n.lower() != expected.lower()
                  and re.search(rf"\b{re.escape(n)}\b", reply, re.I)]
        return "PARTIAL" if others else "OK"
    return "MISS" if consistent else "PARTIAL"


def _question_for(rung: str) -> str:
    """Which question a rung asks.

    Every rung asks the same question. That is what makes `clean` a control: an
    untouched log put to the identical question yields the model's
    false-positive rate directly, where a different question would only yield a
    different experiment.
    """
    return LOG_QUESTION


async def served_model(
    session: aiohttp.ClientSession, client: LLMClient,
) -> Optional[str]:
    """What the endpoint says it is actually serving, from /v1/models.

    Recorded so a run cannot be mislabelled after the fact. `--model` is what
    the *caller* asked for and doubles as the tokenizer id; `--served-model-name`
    is an alias the API answers to. Neither is evidence of which weights were
    loaded, and an alias is stable across a reload that changes them.

    Measured: a result recorded `model: "MY-DEPLOYMENT"` while `tune` had reported
    `unsloth/Qwen3.8-27B-FP8` at launch and the endpoint later served
    `cyankiwi/Qwen3.8-27B-AWQ-FP8`. The box had been re-loaded, and nothing in
    the artifact could say which build the numbers belonged to.

    Best-effort: a server that does not publish `root`, or does not answer, is
    recorded as unknown rather than failing the run. Absent means unknown here,
    never "the same as the alias" -- the whole point is not to guess.
    """
    url = client.base_url.rstrip("/") + "/models"
    try:
        async with session.get(url, headers=client.headers) as response:
            if response.status != 200:
                return None
            body = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return None
    for entry in body.get("data") or []:
        root = entry.get("root") or entry.get("id")
        if root:
            return str(root)
    return None


async def _ask(
    client: LLMClient, session: aiohttp.ClientSession, context: str,
    max_tokens: int, thinking: Optional[str], question: str = LOG_QUESTION,
) -> Tuple[str, str, str]:
    payload = client._probe_payload(question, max_tokens, context=context)
    if thinking == "off":
        # Both spellings, as elsewhere: servers honour one or the other.
        payload["reasoning_effort"] = "none"
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif thinking == "on":
        # Asserted, not assumed. This used to send nothing at all, on the
        # reasoning that the chat template opens `<think>` by default -- which
        # is true of the reference deployment and is not a property of the
        # flag. On a server defaulting to thinking-off, `--thinking on` would
        # have measured the closed condition while the saved result recorded
        # `thinking: "on"`: an artifact asserting something the request never
        # said. Matches what the benchmark's own payload builder does.
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    async with session.post(client._url, json=payload, headers=client.headers) as response:
        if response.status != 200:
            raise ProbeError(f"HTTP {response.status}: {(await response.text())[:200]}")
        body = await response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ProbeError("no choices in response")
    # A reply cut off mid-reasoning is not a wrong answer, it is an unfinished
    # one. Scoring it as a miss blames the model for a budget this tool chose:
    # observed on the log rung, where scanning ~90 entries overran 1024 tokens
    # and produced two "failures" that were the model still working.
    #
    # Only a clean stop is an answer. Testing for "length" alone let everything
    # else through, including a missing field, and a verdict was then published
    # for a reply the server never said was complete. Whatever the value is, it
    # is named in the error rather than folded into one message, because "not
    # stop" is the whole diagnosis.
    finish = choices[0].get("finish_reason")
    content, reasoning = _answer_text(choices[0])
    if finish != "stop":
        detail = ("truncated at --max-tokens; raise it" if finish == "length"
                  else f"finish_reason={finish!r}, not 'stop'")
        raise ProbeError(f"incomplete answer: {detail}", reasoning, finish)
    # A reply with no content is the model's scratchpad, not its answer.
    #
    # This used to fall back to the reasoning trace, on the grounds that a
    # thinking model may put the answer there. Measured at 16k, that fallback
    # scored 4 of 30 trials off the trace, and every one of those traces ended
    # mid-work -- "[2142-11-26] At Deep Array, W". A scratchpad names every
    # candidate the model considered, so scoring it produced two false
    # positives on the *control* and one hedge on the measurement: the control
    # read 6/10 when the trials it could actually judge were 6/8.
    #
    # Counted as unanswered, which is loud: the rate is reported per cell and
    # warned about past 20%. A model whose answer genuinely lives in
    # reasoning_content therefore reports 100% unanswered rather than a table
    # of verdicts read from working notes -- the tool saying it cannot read the
    # answers, which is the failure that can be acted on.
    if not content.strip():
        raise ProbeError("no answer content; the reply was reasoning only",
                         reasoning, finish)
    return content, reasoning, finish


def _count(tokenizer: Any, text: str) -> Optional[int]:
    """Length of a reasoning trace in the same unit as --max-tokens.

    Transcripts recorded characters, which cannot be compared to the budget:
    this text runs ~2.4 chars/token (dates and names tokenize densely), so a
    188,417-character trace that *finished* looks longer than a 148,030-
    character one that was truncated. Counting tokens makes "how much did it
    think" answerable and makes truncation legible as budget exhaustion.

    Best-effort: a tokenizer that will not count must not fail the run.
    """
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


async def run(
    base_url: str, api_key: str, model: str, tokenizer: Any,
    depths: List[int], rungs: List[str], trials: int, seed: int,
    thinking: Optional[str], max_tokens: int, endpoint: str,
    positions: List[float], transcripts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    client = LLMClient(base_url, api_key, model, None, False, endpoint)
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    # A depth is only what it claims to be if it was counted with the model's
    # own tokenizer. gpt2 disagrees with every other vocabulary on how many
    # tokens a text is, so a silent substitution here (corpus.load_tokenizer's
    # last resort, e.g. a GGUF-only repo with no tokenizer.json to download)
    # makes every depth in the run an unlabelled approximation -- found only
    # by an unexplained wall of truncated trials that looked like the model
    # struggling, not like a miscounted context.
    tokenizer_fallback = _tokenizer_fallback(tokenizer)
    if tokenizer_fallback:
        print(f"  WARNING: tokenizer '{tokenizer_fallback}' failed to load; "
              f"depths below are counted with gpt2 and are approximate",
              file=sys.stderr, flush=True)

    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        # Asked once, before any trial, so the recorded identity belongs to the
        # weights the run actually measured rather than to whatever is loaded
        # when someone reads the file later.
        served = await served_model(session, client)
        if served:
            print(f"  serving: {served}", file=sys.stderr, flush=True)
        for depth in depths:
            for rung in rungs:
              # A position sweep is meaningless for a rung that injects nothing.
              for position in (positions if rung in POSITIONABLE else [0.5]):
                tally = {"OK": 0, "PARTIAL": 0, "MISS": 0, "FALSE+": 0}
                errors: List[str] = []
                seen = 0
                # A deep cell can take hours -- a single trial at 131k tokens
                # with a large reasoning budget runs tens of minutes -- and
                # printing only on completion makes a working run look identical
                # to a hung one. Progress goes to stderr so it cannot corrupt a
                # piped table.
                print(f"  d{depth} {rung} @{position}: {trials} trials",
                      file=sys.stderr, flush=True)
                # Cell-level, because the summary line below needs it too.
                marker = TRACE_MARKERS.get(rung)
                # Counts every trial as it finishes, success or not -- unlike
                # len(errors) (capped at 3 distinct messages for the summary),
                # this always advances by exactly 1, so the live numbering
                # cannot fall behind or repeat.
                completed = 0
                for _ in range(trials):
                    context, expected = build_context(
                        tokenizer, depth, rung, rng, position)
                    try:
                        reply, reasoning, finish = await _ask(
                            client, session, context, max_tokens, thinking,
                            _question_for(rung))
                    except (ProbeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        # Not a wrong answer: a question that was never answered.
                        if len(errors) < 3 and str(exc) not in errors:
                            errors.append(str(exc))
                        completed += 1
                        # A cell can run for hours; a trial that fails printed
                        # nothing before this, so a run watched live looked
                        # like it had skipped a trial number with no
                        # explanation until the cell finished.
                        print(f"    trial {completed}/{trials}: UNANSWERED  "
                              f"({str(exc)[:150]})", file=sys.stderr, flush=True)
                        # Recorded like any other trial, because "which trials
                        # did not answer, and what was the model doing" is the
                        # question the unanswered column raises and cannot
                        # answer. Dropping them left the transcript blind to
                        # precisely the trials worth reading.
                        if transcripts is not None:
                            trace = getattr(exc, "reasoning", "")
                            transcripts.append({
                                "depth": depth, "rung": rung, "position": position,
                                "expected": expected, "verdict": "UNANSWERED",
                                "error": str(exc),
                                "named_in_reasoning": bool(
                                    marker and marker.lower() in trace.lower()),
                                "reasoning_chars": len(trace),
                                "reasoning_tokens": _count(tokenizer, trace),
                                "finish_reason": getattr(exc, "finish_reason", None),
                                "reply": "",
                                "reasoning_excerpt": trace.strip()[-3000:],
                            })
                        continue
                    verdict = score(reply.strip().replace("\n", " "), expected,
                                    _LOG_NAMES if rung == "log" else None)
                    tally[verdict] += 1
                    # Did it *see* the injection? A model that never names the
                    # marker while reasoning failed to retrieve; one that names
                    # it and still answers CONSISTENT failed to reason. Those are
                    # different defects and the accuracy number cannot tell them
                    # apart.
                    # Computed on the *whole* trace. The stored excerpt below is
                    # for reading; deriving this from the excerpt instead made
                    # the count a lower bound and quietly understated it.
                    named = bool(marker and marker.lower() in reasoning.lower())
                    if named:
                        seen += 1
                    completed += 1
                    print(f"    trial {completed}/{trials}: {verdict}",
                          file=sys.stderr, flush=True)
                    if transcripts is not None:
                        transcripts.append({
                            "depth": depth, "rung": rung, "position": position,
                            "expected": expected, "verdict": verdict,
                            "named_in_reasoning": named,
                            "reasoning_chars": len(reasoning),
                            "reasoning_tokens": _count(tokenizer, reasoning),
                            # Which field the verdict was read from, and what
                            # the server said about completeness. Without both,
                            # a transcript cannot distinguish an answer from a
                            # scratchpad -- which is how four scored trials
                            # turned out to be the model still working.
                            "finish_reason": finish,
                            "reply": reply.strip()[:400],
                            "reasoning_excerpt": reasoning.strip()[-3000:],
                        })
                answered = sum(tally.values())
                lo, hi = wilson(tally["OK"], answered)
                lo_frac, hi_frac = span(position)
                results.append({
                    "depth": depth, "rung": rung, "position": position,
                    # Only the log rung has two halves to separate. Recorded so
                    # a sweep cannot publish a position without also publishing
                    # how much distance changed underneath it.
                    "separation": round(hi_frac - lo_frac, 3) if rung == "log" else None,
                    "trials": trials, "answered": answered, "errors": errors,
                    "mentioned_while_reasoning": seen,
                    "accuracy": tally["OK"] / answered if answered else None,
                    "ci95_low": lo, "ci95_high": hi, **{k.lower(): v for k, v in tally.items()},
                })
                # Unanswered trials are not missing at random. A trial the
                # model reasons through quickly finishes; one it struggles with
                # runs long and hits the token ceiling. Dropping those biases
                # accuracy *upward*, so a high rate has to be said out loud
                # rather than left as a quiet column.
                lost = trials - answered
                note = "" if not lost else f"  [{lost} unanswered]"
                if lost and lost / trials >= 0.2:
                    # Name the fix that fits the failure. "raise --max-tokens"
                    # is right for truncation and useless for a reply that
                    # stopped cleanly without answering -- and advice that does
                    # not apply sends the reader to change the one setting that
                    # then makes the run longer for no reason.
                    fix = ("raise --max-tokens" if any("truncated" in e for e in errors)
                           else "see the errors below")
                    note += f"  <- accuracy is on a non-random subset; {fix}"
                saw = ""
                if marker and answered:
                    saw = f"  saw={seen}/{answered}"
                print(f"  d{depth:<7} {rung:8} @{position:<4} {tally['OK']}/{answered}  "
                      f"95% CI [{lo:.2f}, {hi:.2f}]{saw}{note}", flush=True)
                for err in errors:
                    print(f"      error: {err}", file=sys.stderr)
    return {
        "version": __version__, "model": model, "served_model": served,
        "base_url": base_url,
        "seed": seed, "thinking": thinking, "trials": trials,
        "depths": depths, "rungs": rungs, "positions": positions,
        # None means the requested tokenizer loaded; a string names the
        # tokenizer that failed and says every depth above is a gpt2 count,
        # not a real one. Absent-from-older-files means unknown, not "loaded
        # fine" -- the fallback existed long before this field did.
        "tokenizer_fallback": tokenizer_fallback,
        "results": results,
    }


def render(report: Dict[str, Any]) -> str:
    lines = [
        "",
        f"quality probe | model: {report['model']} | trials per cell: {report['trials']}"
        + (f" | thinking: {report['thinking']}" if report["thinking"] else ""),
        "",
    ]
    if report.get("tokenizer_fallback"):
        # Printed on the console table too, not just stderr scrollback -- a
        # table that looks complete is exactly what got this bug missed the
        # first time.
        lines.append(
            f"WARNING: tokenizer '{report['tokenizer_fallback']}' failed to "
            f"load; every depth above is a gpt2 count, not a real one.")
        lines.append("")
    lines += [
        "| depth | rung | at | sep | accuracy | 95% CI | partial | false+ | unanswered |",
        "|------:|:-----|---:|----:|---------:|:-------|--------:|-------:|-----------:|",
    ]
    for r in report["results"]:
        unanswered = r["trials"] - r["answered"]
        at = "-" if r["rung"] == "clean" else f"{r['position']:.2f}"
        sep = "-" if r.get("separation") is None else f"{r['separation']:.2f}"
        lines.append(
            f"| {r['depth']} | {r['rung']} | {at} | {sep} | {r['ok']}/{r['answered']} |"
            f" [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] |"
            f" {r['partial']} | {r['false+']} | {unanswered} |"
        )
    lines += [
        "",
        "`at` is where the injection went, as a fraction of the context. The ends are",
        "attention-privileged and the middle is not, so a position sweep separates that",
        "effect from depth. `clean` has no injection, so it shows `-`.",
        "",
        "`sep` is how far apart the contradiction's two halves actually were, as a",
        "fraction of the log. A pair centred at 0.9 cannot be 0.6 wide, so a position",
        "sweep necessarily varies separation near the ends -- the column says by how",
        "much rather than leaving it to be read as a pure position effect.",
        "",
        "`clean` is the control: its false+ column is how often the model invents an",
        "anomaly when asked to look for one. `outlier` is the retrieval floor -- if it",
        "fails, the context is not reaching the model at all and `log` says nothing.",
        "`log` is the measurement.",
        "",
        "Accuracy is a proportion, so the interval is Wilson, not Student-t. At small",
        "trial counts it is wide, and honestly so: 0/6 is [0.00, 0.39], not certainty.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py probe",
        description="Inject known anomalies at depth and check the model still finds them.",
    )
    ap.add_argument("base_url", help="OpenAI-compatible endpoint URL, including /v1")
    ap.add_argument("--model", required=True, help="Model to probe")
    ap.add_argument("--served-model-name", default=None,
                    help="Name the API expects, when it differs from --model "
                         "(which also selects the tokenizer)")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the endpoint")
    ap.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    ap.add_argument("--depths", type=int, nargs="+", default=[4096, 32768],
                    help="Context depths to probe (default: 4096 32768)")
    ap.add_argument("--rungs", nargs="+", choices=RUNGS, default=list(RUNGS),
                    help="Which rungs to run (default: all three)")
    ap.add_argument("--positions", type=float, nargs="+", default=[0.5],
                    metavar="F",
                    help="Where to inject, as fractions of the context "
                         "(default: 0.5). Attention is strongly position "
                         "dependent -- the ends are privileged and the middle is "
                         "not -- so 0.1 0.5 0.9 separates that from depth. "
                         "Applies to the outlier and log rungs; clean injects "
                         "nothing")
    ap.add_argument("--trials", type=int, default=6,
                    help="Trials per (depth, rung) cell (default: 6)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for haystack generation and name choice (default: random)")
    ap.add_argument("--thinking", choices=("on", "off"), default=None,
                    help="Force thinking mode. `off` scores 0/10 against 6/10 at "
                         "depth 4096 -- but that is this probe's one-line answer "
                         "format, not a capability loss: allowed to show working, "
                         "`off` scores 9/10. It measures the model with no "
                         "scratchpad anywhere")
    ap.add_argument("--max-tokens", type=int, default=40000,
                    help="Answer budget. A thinking model scans the whole log "
                         "before answering, so this grows with --depths: 1024 is "
                         "enough at 2k and truncated every cell at 16k. Too small "
                         "a budget does not just lose trials, it loses the hard "
                         "ones, which raises the reported score")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer name or path")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit the report as JSON instead of a table")
    ap.add_argument("--save-result", default=None, help="Write the JSON report to this path")
    ap.add_argument("--save-transcripts", default=None, metavar="PATH",
                    help="Write each trial's answer and reasoning trace to PATH as "
                         "JSON. The accuracy number cannot distinguish a model "
                         "that never saw the injection from one that saw it and "
                         "judged it consistent; the trace can")
    args = ap.parse_args()

    if args.trials < 1:
        ap.error("--trials must be >= 1")
    if any(d < 0 for d in args.depths):
        ap.error("--depths must be >= 0")
    if any(not 0.0 <= f <= 1.0 for f in args.positions):
        ap.error("--positions are fractions of the context, so 0.0 to 1.0")

    # No corpus. Every rung generates its own haystack, so a token counter is
    # all that is needed and there is no book to download, no slice to warn
    # about running past the end of, and no --book-url to imply the source text
    # affected the result.
    tokenizer = load_tokenizer(args.model, args.tokenizer)

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    transcripts: Optional[List[Dict[str, Any]]] = [] if args.save_transcripts else None
    report = asyncio.run(run(
        args.base_url, args.api_key, args.served_model_name or args.model, tokenizer, args.depths,
        list(args.rungs), args.trials, seed, args.thinking, args.max_tokens,
        args.endpoint, list(args.positions), transcripts,
    ))

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    if args.save_transcripts and transcripts is not None:
        with open(args.save_transcripts, "w") as fh:
            json.dump(transcripts, fh, indent=2)
        print(f"\nTranscripts: {args.save_transcripts}")
    if args.save_result:
        with open(args.save_result, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved to {args.save_result}")

    # A probe that answered nothing is a failed probe, not a model that scored
    # zero. Reporting 0/N for every rung -- including the trivial retrieval
    # floor -- reads exactly like a collapsed model, so it must not exit 0.
    answered = sum(r["answered"] for r in report["results"])
    if not answered:
        print("error: no probe request was answered; the results above measure "
              "nothing. Check --model names a model the endpoint serves.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
