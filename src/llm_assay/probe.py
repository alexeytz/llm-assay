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
import difflib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import aiohttp

from . import __version__
from .client import LLMClient, _answer_text
from .config import BenchmarkConfig
from .corpus import load_tokenizer, tokenizer_fallback as _tokenizer_fallback
from .servermetrics import (
    ServerCounters, ServerMetricsProbe, _build_summary, engine_version,
    served_model, server_build,
)

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

#: Syllables for seeded surnames. Pronounceable so they read as names in a
#: log, built from parts so no list of them exists anywhere to be trained on.
_ONSETS = ("b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t",
           "v", "z", "br", "dr", "gr", "kr", "st", "tr", "th", "sk")
_VOWELS = ("a", "e", "i", "o", "u", "ae", "io")
_CODAS = ("", "", "", "n", "r", "s", "l", "x", "k")
VOCABULARIES = ("builtin", "seeded")


def seeded_names(seed: int, count: Optional[int] = None) -> Tuple[str, ...]:
    """Nonsense surnames generated from the seed: contamination-proof by
    construction, and reproducible from nothing but the seed.

    The built-in fifty sit in a public repository, which is a slow leak in the
    instrument's foundation: the scorer only works because a correct answer
    names a surname no training set contains. Generated names keep that true.

    Same count as the built-in list, so the generator's two invariants -- no
    accidental (name, date) clash, and deliberate benign same-name repeats --
    hold exactly as they do now. The scorer matches the expected surname by
    substring, so no name may contain another or any other vocabulary word,
    and none may be a built-in name, or a seeded and a built-in result could
    share a surname and look comparable.
    """
    count = len(_LOG_NAMES) if count is None else count
    rng = random.Random(f"llm-assay vocabulary {seed}")
    others = [w.lower() for w in (*_LOG_LOCATIONS, *_LOG_RANKS, *_LOG_ACTIONS,
                                  *_LOG_OBJECTS, *_LOG_NAMES, "offtopic",
                                  "consistent", "the")]
    names: List[str] = []
    while len(names) < count:
        name = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS)
                       for _ in range(rng.choice((2, 2, 3)))) + rng.choice(_CODAS)
        low = name.lower()
        if not 5 <= len(low) <= 10:
            continue
        clash = [w for w in [n.lower() for n in names] + others
                 if low in w or any(part in low for part in w.split() if len(part) > 2)]
        if clash:
            continue
        names.append(name.capitalize())
    return tuple(names)


def names_for(vocabulary: str, seed: int) -> Tuple[str, ...]:
    """The surname list a run with this vocabulary and seed uses."""
    if vocabulary == "builtin":
        return _LOG_NAMES
    if vocabulary == "seeded":
        return seeded_names(seed)
    raise ValueError(f"unknown vocabulary {vocabulary!r} (expected one of {VOCABULARIES})")


def vocabulary_digest(names: Optional[Sequence[str]] = None) -> str:
    """A digest of everything the generator and the question are made of.

    Two results built from different word lists or a different template read
    different text from the same seed, and nothing else in a result says so.
    Recorded beside `tokenizer_fallback` for the same reason that one is: the
    comparison is silently wrong otherwise.
    """
    import hashlib
    material = json.dumps([_LOG_LOCATIONS, _LOG_RANKS,
                           list(names) if names is not None else _LOG_NAMES,
                           _LOG_ACTIONS, _LOG_OBJECTS, OUTLIER, LOG_QUESTION])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


#: The same question, asking for the evidence as well. Line 1 is the verdict
#: and is scored exactly as the one-line format is; lines 2-3 are checked
#: against the haystack and never scored. Relaxing the format any further is
#: known to break scoring: full working scored 0/10 in both thinking arms,
#: because a derivation names many suspects and the hedging rule rejects them.
CITED_QUESTION = (
    "You are analyzing a chronological security log.\n\n"
    "A logical impossibility would be a single person recorded in two completely "
    "different locations on the exact same date.\n\n"
    "Read the log carefully, then reply in this format:\n"
    "Line 1. If you find such an impossibility, ONLY the surname of the person "
    "involved. Otherwise, if some line does not belong in a log of this kind at "
    "all, ONLY: OFFTOPIC. Otherwise ONLY: CONSISTENT\n"
    "Lines 2 and 3. Only if line 1 is a surname: the two log lines that show it, "
    "copied exactly as they appear in the log, one per line. Otherwise write "
    "nothing after line 1."
)
ANSWER_FORMATS = ("line", "cited")


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
                 finish_reason: Optional[str] = None,
                 usage: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reasoning = reasoning
        self.finish_reason = finish_reason
        # What the endpoint billed for the attempt. A truncated trial spent its
        # whole budget, and that is exactly the trial whose cost and length a
        # reader wants.
        self.usage = usage


def usage_of(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The API's own token counts for one response, or None if it gave none.

    `reasoning_tokens` elsewhere in a transcript is the *local* tokenizer's
    count of the trace the endpoint chose to return -- the wrong unit twice
    over on a hosted model. Vendors summarise or withhold reasoning
    (gpt-6-sol-pro returned a 261-token trace for roughly 15,000 billed), and a
    provider's output ceiling and its bill are both in its own tokens. Settling
    a budget question in v22 needed one hand-written API call to read this
    block; recording it makes that answerable from the artifact.

    `prompt_tokens` is also the true depth on a model whose tokenizer is not
    public: the haystack is sized in Qwen tokens, and a vendor counting the
    same text differently reports it here. `cost` is kept when a router
    reports one.
    """
    raw = body.get("usage")
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens"):
        if isinstance(raw.get(key), int):
            out[key] = raw[key]
    details = raw.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        out["reasoning_tokens"] = details["reasoning_tokens"]
    cached = raw.get("prompt_tokens_details")
    if isinstance(cached, dict) and isinstance(cached.get("cached_tokens"), int):
        out["cached_tokens"] = cached["cached_tokens"]
    if isinstance(raw.get("cost"), (int, float)):
        out["cost"] = raw["cost"]
    return out or None


#: Usage fields summed per cell. `trials` beside them counts the trials that
#: reported usage at all, so a partial report is visible rather than averaged.
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                "cached_tokens", "cost")


def sum_usage(usages: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Per-cell totals of the trials' usage blocks; None when none reported."""
    reported = [u for u in usages if u]
    if not reported:
        return None
    total: Dict[str, Any] = {"trials": len(reported)}
    for key in USAGE_FIELDS:
        values = [u[key] for u in reported if key in u]
        if values:
            total[key] = round(sum(values), 6) if key == "cost" else sum(values)
            # How many trials reported this field. A vendor that omits
            # reasoning_tokens on some replies must not read as zero there.
            total[f"{key}_trials"] = len(values)
    return total


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



def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p for a 2x2 table, by summing tables no likelier than this one.

    Beside `wilson` for the same reason it is: the counts a probe produces are
    small and one-sided, and the normal approximations are wrong exactly there.
    A chi-square on `0/50` against `3/30` is not a test of anything.

    "As or more extreme" is by *point probability* rather than by one tail
    doubled. The doubling convention can report a p above 1 on a lopsided table,
    and the two disagree most on the small, unbalanced tables this is for.

    The fabrication programme's pre-registered comparison is this function on
    (fabricated, answered - fabricated) in each of two arms.
    """
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        # A table with an empty margin holds no comparison: both arms answered
        # the same way, or one of them produced nothing at all. Reporting 1.0
        # says "no evidence of a difference", which is the truthful reading of
        # a table that cannot show one.
        return 1.0
    row1, row2, col1 = a + b, c + d, a + c

    def probability(x: int) -> float:
        return (math.comb(row1, x) * math.comb(row2, col1 - x)
                / math.comb(n, col1))

    observed = probability(a)
    # The float comparison is nudged, or a table symmetric to the observed one
    # is dropped for a last-bit difference and the p-value comes out too small.
    return min(1.0, sum(
        probability(x) for x in range(max(0, col1 - row2), min(row1, col1) + 1)
        if probability(x) <= observed * (1 + 1e-9)
    ))


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
    position: float = 0.5, names: Optional[Sequence[str]] = None,
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
    return build_log(depth, position, rng, tokenizer, needle=needle, names=names)



def _log_entry(rng: random.Random, name: str, date: str, place: str) -> str:
    return (f"[{date}] At {place}, {rng.choice(_LOG_RANKS)} {name} "
            f"{rng.choice(_LOG_ACTIONS)} {rng.choice(_LOG_OBJECTS)}.")


def build_log(
    depth: int, position: float, rng: random.Random, tokenizer: Any,
    gap: float = 0.6, needle: str = "impossible",
    names: Optional[Sequence[str]] = None,
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
    pool = tuple(names) if names is not None else _LOG_NAMES

    def background() -> str:
        # Reuse an existing (name, date) about a fifth of the time. Chance
        # collisions are far too rare at these sizes -- one per 90 entries --
        # and without deliberate benign repeats "the name and date that occur
        # twice" solves the task without ever reading a location.
        if placed and rng.random() < 0.2:
            name, date = rng.choice(list(placed))
        else:
            name, date = rng.choice(pool), rng.choice(days)
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
    culprit = rng.choice(pool)
    date = rng.choice(days)
    here, there = rng.sample(_LOG_LOCATIONS, 2)
    placed[(culprit, date)] = here

    lo_frac, hi_frac = span(position, gap)
    lo_at = max(0, min(len(lines) - 1, int(lo_frac * len(lines))))
    hi_at = max(lo_at + 1, min(len(lines), int(hi_frac * len(lines))))
    lines.insert(lo_at, _log_entry(rng, culprit, date, here))
    lines.insert(hi_at, _log_entry(rng, culprit, date, there))
    return "\n".join(lines), culprit



# Every line the generator writes has one shape: "[date] At place, rank name
# action object." -- no internal full stop, no brackets. So a quotation of one
# is recognisable without knowing which log it came from, which is what makes
# the check below mechanical rather than a reading.
# The body must look like the generator's own: "<place>, <rank> <name> <verb>
# <object>." and nothing else. The first version allowed any run of characters
# up to a full stop, which swept up the model's half-written prose as though it
# were a quotation -- measured at 32k, it captured
# `... inspected coolant loop? i might b` and `... technician bex .`, both of
# which then counted as lines the log does not contain. A citation matcher that
# matches non-citations manufactures absent claims.
_CITATION_RE = re.compile(
    r"\[\d{4}-\d{2}-\d{2}\]\s*At\s"          # the stamp
    r"[A-Za-z][A-Za-z ]{1,40},\s*"              # place, then its comma
    r"[A-Za-z]+\s+[A-Za-z]+\s+"                # rank, name
    r"[a-z]+\s+"                                # verb
    # The generator always writes "the <two words>"; a paraphrase may drop the
    # article or swap it, so any of them is allowed and none is required.
    r"(?:(?:the|a|an)\s+)?[a-z]+\s+[a-z]+\.",   # object, ending the line
    # Case-insensitive: a model that re-types a line in lower case is quoting it
    # just the same, and an invisible citation is counted as "cited nothing
    # checkable" -- the one bucket that should mean the model offered no
    # evidence at all.
    re.IGNORECASE,
)


# How a control-rung false positive can go, named once. The programme's
# pre-registration fixed this classification in advance, and a test reads these
# names back out of here, so a rename cannot leave that document describing a
# rule nothing implements.
#
# `misquoted` exists because the first three buckets could not tell a
# manufactured contradiction from an incidental slip. Measured across the 8k
# arm, **21% of trials the model got right** still quoted at least one line the
# log does not contain -- it misquotes while reasoning and then does not act on
# it -- so "cited an absent line" on its own would bucket roughly one false
# positive in five as fabrication for a misquote that had nothing to do with its
# answer.
FALSE_POSITIVE_KINDS = ("fabricated", "misquoted", "only_real", "uncited")


def _normalise_line(line: str) -> str:
    """One spelling for a log line, so quoting it in prose still matches.

    Whitespace and case only. Every other field -- the date, the place, the
    rank, the name, the verb, the object -- is left significant on purpose:
    those are exactly the fields a fabrication alters, and a comparison lenient
    enough to forgive a changed digit cannot detect the failure it is for.
    """
    return " ".join(line.replace("*", "").replace('"', "").split()).lower()


def cited_lines(text: str) -> List[str]:
    """Every log line the reply or the reasoning quotes, de-duplicated.

    A long trace quotes the same line many times over; the question is which
    distinct lines were cited, not how often.
    """
    seen: List[str] = []
    for match in _CITATION_RE.finditer(text):
        line = _normalise_line(match.group(0))
        if line not in seen:
            seen.append(line)
    return seen


# The three fields a false impossibility is *made of*: a date, a place, and a
# person. The rank, the verb and the object are decoration -- changing one of
# them cannot turn a consistent log into a contradictory one.
_CLAIM_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2})\] at ([^,]+), (\w+) (\w+)\b")


def _claim(line: str) -> Optional[Tuple[str, str, str]]:
    """(date, place, surname) asserted by a normalised log line, if it parses."""
    match = _CLAIM_RE.match(line)
    if not match:
        return None
    return (match.group(1), match.group(2).strip(), match.group(4))


def fabrication_rate(
    results: Sequence[Dict[str, Any]], rung: str = "clean",
) -> Tuple[int, int]:
    """``(fabricated, answered)`` for ONE rung. Never pooled across rungs.

    **`fabricated` is only defined on `clean`.** ``score`` returns ``FALSE+``
    solely when ``expected is None``, which is the control rung -- so on ``log``
    and ``outlier`` the bucket is *structurally* zero and cannot be anything
    else. Summing it across rungs therefore divides a real rate by trials that
    could never have contributed to it.

    That is not hypothetical. A banked 27B cell whose control rung read
    **12/19 = 63.2%** was published for months as **12/69 = 17.4%**, because an
    analysis script summed ``false_positives`` and ``answered`` over a result
    file's whole ``results`` list. The dilution factor, 69/19 = 3.6x, was then
    mistaken for a threefold effect and an entire arm was pre-registered to
    chase it. The pooling itself was never wrong -- it keys cells by
    ``(depth, rung)``. The ad-hoc summing was.

    So this function takes one rung and returns one pair. Pass ``rung``
    explicitly; there is deliberately no way to ask it for a pooled number.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung: {rung!r} (expected one of {RUNGS})")
    fabricated = answered = 0
    for r in results:
        if r.get("rung") != rung:
            continue
        fabricated += (r.get("false_positives") or {}).get("fabricated", 0)
        answered += r.get("answered", 0)
    return fabricated, answered


#: The retrieval floor passes at no more than this share of MISS verdicts. It is
#: the gate every pre-registration from v17 on wrote by hand -- MISS <= 3/10,
#: and MISS <= 6/20 where floors were doubled -- stated once as a fraction.
FLOOR_MISS_MAX = 0.3


def floor_verdicts(cells: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """The retrieval floor at each depth that has one: misses, trials, passed.

    A `clean` rate means nothing on its own. A model that is not reading the
    log answers CONSISTENT and scores perfectly on the control, so a 0% false
    positive rate is either careful checking or no checking at all, and only
    the `outlier` cell at the same depth says which. grok-4.3's 0/20 at
    d131,072 came within one cheap cell of being published as the arm's best
    number; its floor, re-run, missed the planted line 4 times in 9.

    Keyed on **MISS**, not OK. MISS is the model answering CONSISTENT with a
    foreign line in front of it -- it did not see it. PARTIAL means it flagged
    something in the wrong shape, so it did see it. An OK-based gate scores
    answer format along with retrieval: laguna's deep floor read 0/10 on OK
    and looked like abstention, and 2/10 on MISS with the model engaging in 8
    of 10 trials.

    Several outlier cells at one depth (a position sweep, or pooled files) are
    summed. Takes the probe's own result dicts or pooled cells; both carry
    `depth`, `rung`, `trials` and `miss`.
    """
    floors: Dict[int, Dict[str, Any]] = {}
    for c in cells:
        if c.get("rung") != "outlier":
            continue
        f = floors.setdefault(c["depth"], {"misses": 0, "trials": 0})
        f["misses"] += c.get("miss", 0) or 0
        f["trials"] += c.get("trials", 0) or 0
    for f in floors.values():
        f["passed"] = bool(f["trials"]) and f["misses"] <= FLOOR_MISS_MAX * f["trials"]
    return floors


def floor_note(depth: int, rung: str,
               floors: Dict[int, Dict[str, Any]]) -> Optional[str]:
    """What qualifies a `clean` or `log` cell at `depth`, or None for others.

    ``"MISS 0/10 ok"``, ``"MISS 4/9 FAILED"``, or ``"none"`` when no floor was
    run at that depth -- which leaves the cell's rate unqualified, not wrong.
    """
    if rung == "outlier":
        return None
    f = floors.get(depth)
    if f is None:
        return "none"
    return f"MISS {f['misses']}/{f['trials']} {'ok' if f['passed'] else 'FAILED'}"


def floor_warnings(cells: Iterable[Dict[str, Any]]) -> List[str]:
    """One line per (depth, rung) whose rate the floor does not qualify.

    Recording and reporting only: nothing here changes a number. It says out
    loud what was previously left for a reader to notice, the same shape as
    the benchmark's cache warnings.
    """
    cells = list(cells)
    floors = floor_verdicts(cells)
    seen: Set[Tuple[int, str]] = set()
    out: List[str] = []
    for c in cells:
        key = (c["depth"], c["rung"])
        if c["rung"] == "outlier" or key in seen:
            continue
        seen.add(key)
        f = floors.get(c["depth"])
        if f is None:
            out.append(f"d{c['depth']} {c['rung']}: no retrieval floor at this "
                       "depth, so the rate is unqualified -- a model that is "
                       "not reading scores perfectly on `clean`")
        elif not f["passed"]:
            out.append(f"d{c['depth']} {c['rung']}: the retrieval floor FAILED "
                       f"(MISS {f['misses']}/{f['trials']}, gate "
                       f"{FLOOR_MISS_MAX:.0%}), so the model was not reliably "
                       "reading the log and this rate is not interpretable")
    return out


def classify_false_positive(reply: str, citations: Dict[str, Any],
                            names: Optional[Sequence[str]] = None) -> str:
    """Which of FALSE_POSITIVE_KINDS this wrong answer is.

    An absent citation alone is not evidence of fabrication, which is the thing
    the first version of this got wrong. Measured across the 8k arm, **10 of 48
    trials the model answered correctly** still quoted at least one line the log
    does not contain: it misquotes while reasoning and then does not act on the
    misquote. Bucketing on presence therefore mislabels roughly one false
    positive in five.

    So the absent claim has to be *load-bearing*, and there is a mechanical test
    for that after all -- the pre-registration asserted there was not:

    * The reply names a surname. The manufactured evidence has to be about that
      person, or it is not what the answer rests on. Verified on every case
      available: the deep false positives cite an absent line naming exactly the
      surname they answer, and the incidental misquotes name other people.
    * The reply is ``OFFTOPIC``. The claim being made is "this line does not
      belong", so a cited line that is not in the log *is* the fabricated
      evidence, by construction.

    Anything else with absent claims is ``misquoted`` -- wrong for some other
    reason, with a slip in its working. That bucket exists so the distinction is
    published rather than absorbed into the headline number.
    """
    if not citations.get("quoted"):
        return "uncited"
    absent = [tuple(c) for c in citations.get("absent_claim_list") or []]
    if not absent:
        return "only_real"
    if re.search(r"\bOFFTOPIC\b", reply, re.I):
        return "fabricated"
    named = {n.lower() for n in (names if names is not None else _LOG_NAMES)
             if re.search(rf"\b{n}\b", reply, re.I)}
    if named and any(claim[2].lower() in named for claim in absent):
        return "fabricated"
    return "misquoted"


def check_citations(text: str, context: str, examples: int = 5) -> Dict[str, Any]:
    """Which of the model's quoted evidence is actually in the haystack.

    The control rung injects nothing, so every FALSE+ rests on evidence the log
    does not contain -- and reading transcripts showed the model does not merely
    guess a name, it *manufactures the line*: a real entry with one digit of its
    date changed, or an entry invented whole. Both are built to survive casual
    checking, so the only reliable way to tell them from an answer-format slip
    is to go back to the source. That is a mechanical question, so the probe
    answers it at trial time, when it still holds both the whole trace and the
    exact haystack it built.

    Doing it here rather than afterwards settles two things that otherwise
    weaken the answer. The saved transcript keeps only the trace's last few
    thousand characters, and a citation earlier than that is unrecoverable; and
    a check run later has to rebuild the haystack from a seed, which is correct
    but is an inference about what the trial read rather than a record of it.

    **The verdict is on the claim, not on the wording.** ``absent_claims`` counts
    quoted (date, place, surname) triples the log does not contain, which is
    exactly what a false impossibility is made of. ``absent`` counts whole lines
    that do not match verbatim, and is description only: the model paraphrases
    constantly while reasoning -- dropping "the" from "inspected the coolant
    loop" is its commonest -- and one measured trial quoted three real lines
    that way beside one genuinely invented, so a line-level rule called it
    right for three wrong reasons. A rule that can be satisfied by loose typing
    is not measuring fabrication.

    ``absent_examples`` records the nearest real line beside each absent claim,
    for the reader rather than for the verdict: it is what distinguishes a
    changed digit from a line invented whole.
    """
    real_lines = {_normalise_line(line) for line in context.split("\n")}
    real_claims = {c for c in (_claim(line) for line in real_lines) if c}
    quoted = cited_lines(text)
    absent_lines = [line for line in quoted if line not in real_lines]

    claims: List[Tuple[str, str, str]] = []
    for line in quoted:
        claim = _claim(line)
        if claim and claim not in claims:
            claims.append(claim)
    absent_claims = [c for c in claims if c not in real_claims]

    detail: List[Dict[str, str]] = []
    for claim in absent_claims[:examples]:
        cited = next(line for line in quoted if _claim(line) == claim)
        near = difflib.get_close_matches(cited, real_lines, n=1, cutoff=0.5)
        detail.append({"cited": cited, "nearest": near[0] if near else ""})
    return {
        "quoted": len(quoted), "absent": len(absent_lines),
        "claims": len(claims), "absent_claims": len(absent_claims),
        "absent_examples": detail,
        # Every absent claim, not the five that fit the examples list, and as
        # the triple rather than the prose. A verdict that cannot be recomputed
        # has to be re-measured: the 8k arm survived a rule change only because
        # it happened to hold one false positive whose single example fit. The
        # triples are three short strings each, so recording all of them costs
        # almost nothing and buys the next refinement a reclassification
        # instead of another 32 hours.
        "absent_claim_list": [list(c) for c in absent_claims],
    }


def evidence_check(reply: str, context: str) -> Dict[str, Any]:
    """What a `cited` reply's lines 2-3 actually show.

    ``lines``: log lines parsed after the verdict line. ``absent``: claims among
    them the haystack does not contain. ``proves``: two cited lines, both real,
    naming one surname on one date at two places -- an impossibility the model
    can actually produce. On the `clean` rung none exists, so a false positive
    never proves, and its evidence is either absent or beside the point. This is
    what the structured format buys on models that never quote while
    reasoning: their evidence becomes checkable at all.
    """
    rest = "\n".join(reply.strip().splitlines()[1:])
    real = {c for c in (_claim(_normalise_line(line)) for line in context.split("\n")) if c}
    claims = [c for c in (_claim(line) for line in cited_lines(rest)) if c]
    shown = [c for c in claims if c in real]
    proves = any(a[0] == b[0] and a[2] == b[2] and a[1] != b[1]
                 for i, a in enumerate(shown) for b in shown[i + 1:])
    return {"lines": len(claims), "absent": len(claims) - len(shown), "proves": proves}


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


def _question_for(rung: str, answer_format: str = "line") -> str:
    """Which question a rung asks.

    Every rung asks the same question. That is what makes `clean` a control: an
    untouched log put to the identical question yields the model's
    false-positive rate directly, where a different question would only yield a
    different experiment.
    """
    return CITED_QUESTION if answer_format == "cited" else LOG_QUESTION


def effective_effort(flag: Optional[str],
                     extra_body: Optional[Dict[str, Any]]) -> Optional[str]:
    """The `reasoning_effort` the requests actually carried.

    `--extra-body` is merged last, so a value there overrides the flag; v15 set
    grok's effort that way before the flag existed, and its result must read
    the same as one run with the flag. It used a router's nested spelling,
    `reasoning: {"effort": "high"}`, so that is read too -- reading only the
    top-level field left the one real case this exists for unseen.
    """
    body = extra_body or {}
    nested = body.get("reasoning")
    value = body.get("reasoning_effort")
    if value is None and isinstance(nested, dict):
        value = nested.get("effort")
    if value is None:
        value = flag
    return None if value is None else str(value)


def _request_body(
    client: LLMClient, context: str, max_tokens: int, thinking: Optional[str],
    question: str, extra_body: Optional[Dict[str, Any]],
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """One probe request body. Shared by the trials and the pre-flight, so the
    pre-flight exercises exactly the thinking switch and provider pin the
    trials will send -- a pin that 404s must fail there, not on trial one."""
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
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    # Merged last so an explicit field beats the flag's own spelling. This is
    # what carries a router's provider pin: without it OpenRouter picks freely
    # among providers serving different quantisations, so one cell would mix
    # precisions and the artifact would name none of them.
    if extra_body:
        payload.update(extra_body)
    return payload


async def _post(client: LLMClient, session: aiohttp.ClientSession,
                payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST one request; a non-200 or a reply with no choices is a ProbeError."""
    async with session.post(client._url, json=payload, headers=client.headers) as response:
        if response.status != 200:
            raise ProbeError(f"HTTP {response.status}: {(await response.text())[:200]}")
        body = await response.json()
    if not (body.get("choices") or []):
        raise ProbeError("no choices in response")
    return dict(body)


async def _ask(
    client: LLMClient, session: aiohttp.ClientSession, context: str,
    max_tokens: int, thinking: Optional[str], question: str = LOG_QUESTION,
    extra_body: Optional[Dict[str, Any]] = None,
    usage_sink: Optional[List[Optional[Dict[str, Any]]]] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, str, str, Optional[str]]:
    body = await _post(client, session, _request_body(
        client, context, max_tokens, thinking, question, extra_body,
        reasoning_effort))
    usage = usage_of(body)
    if usage_sink is not None:
        usage_sink.append(usage)
    choices = body["choices"]
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
        raise ProbeError(f"incomplete answer: {detail}", reasoning, finish, usage)
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
                         reasoning, finish, usage)
    provider = body.get("provider")
    return content, reasoning, finish, (str(provider) if provider else None)


class PreflightRefused(ProbeError):
    """The endpoint, model name or provider pin is wrong: run no trial at all.

    Distinct from a transport failure, which is worth retrying. A misconfigured
    request fails identically every time, so retrying it only spends the backoff
    -- and banking it, which is what happened before this existed, records an
    operator's mistake as the model declining to answer.
    """


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Small on purpose: the pre-flight proves the request is accepted, not that
#: the model can do the task. A thinking model will usually stop on `length`
#: here, and that is a pass.
PREFLIGHT_TOKENS = 16
PREFLIGHT_QUESTION = "Reply with the single word OK."


async def _model_listing(
    session: aiohttp.ClientSession, base_url: str, api_key: str,
) -> List[Tuple[str, Optional[str]]]:
    """``(id, root)`` for each entry in ``/v1/models``; empty when unreadable."""
    try:
        async with session.get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            if response.status != 200:
                return []
            body = await response.json(content_type=None)
    except Exception:
        return []
    entries = body.get("data") if isinstance(body, dict) else None
    return [(str(e["id"]), e.get("root")) for e in entries or []
            if isinstance(e, dict) and e.get("id")]


def _name_hint(requested: str, listing: List[Tuple[str, Optional[str]]]) -> str:
    """Say which name the endpoint answers to, when the listing can tell.

    The v25 case: vLLM reports the repository as `root` and serves it under an
    alias `id`, requests must name the `id`, and naming the repository returns
    HTTP 404. 23 trials were banked as the model declining before anyone looked.
    """
    ids = [i for i, _ in listing]
    if not ids or requested in ids:
        return ""
    for alias, root in listing:
        if root == requested:
            return (f" -- this endpoint serves {requested} under the alias "
                    f"'{alias}', and requests must name the alias "
                    f"(--served-model-name {alias})")
    if len(ids) <= 5:
        return f" -- the endpoint lists: {', '.join(ids)}"
    return f" -- '{requested}' is not among the {len(ids)} models it lists"


async def preflight(
    client: LLMClient, session: aiohttp.ClientSession, api_key: str,
    thinking: Optional[str], extra_body: Optional[Dict[str, Any]],
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Send one real request, exactly as the trials will, before any trial.

    Every hosted arm re-implemented this by hand; the local arms had none, which
    is why an HTTP 404 bit there. The request is authoritative and the listing
    only explains a failure, because some servers -- llama.cpp among them --
    answer to any model name while listing one alias.

    Raises `PreflightRefused` when the request is rejected (HTTP 4xx, or a
    reply with no choices): the configuration is wrong and every trial would
    fail the same way. A transport failure raises the plain `ProbeError` or
    aiohttp error, which a caller may retry.
    """
    try:
        body = await _post(client, session, _request_body(
            client, "", PREFLIGHT_TOKENS, thinking, PREFLIGHT_QUESTION, extra_body,
            reasoning_effort))
    except ProbeError as exc:
        message = str(exc)
        if re.match(r"HTTP 4\d\d", message) and not message.startswith(("HTTP 408", "HTTP 429")):
            listing = await _model_listing(session, client.base_url, api_key)
            raise PreflightRefused(
                f"pre-flight refused: {message}{_name_hint(client.model_name, listing)}"
            ) from exc
        if message.startswith("no choices"):
            raise PreflightRefused(f"pre-flight refused: {message}") from exc
        raise
    choice = body["choices"][0]
    finish = choice.get("finish_reason")
    if finish == "error":
        # A router reporting an upstream failure as a 200. Transport, not
        # configuration: the same request may succeed in a minute.
        raise ProbeError("pre-flight: finish_reason='error' from upstream")
    provider = body.get("provider")
    return {"finish_reason": finish,
            "provider": str(provider) if provider else None,
            "usage": usage_of(body)}


class BudgetRefused(PreflightRefused):
    """--max-tokens cannot fit under the endpoint's ceiling at some depth."""


#: Tokens the chat template, the question and the system wrapper add on top
#: of the haystack's nominal depth. Measured on the reference deployment at
#: about 150; rounded up so the headroom estimate errs toward caution.
PROMPT_OVERHEAD = 512

#: Past this share of the binding ceiling, the budget has no room to grow. A
#: model whose output reaches it finishes `length`, and the void-on-length
#: rule kills the cell -- with nothing left to raise.
BUDGET_NEAR = 0.8


async def endpoint_limits(
    session: aiohttp.ClientSession, base_url: str, api_key: str, model: str,
) -> Dict[str, Optional[int]]:
    """The ceilings an output budget runs into, as far as the endpoint says.

    ``context_window``: vLLM's ``max_model_len`` or a router's
    ``context_length`` from ``/v1/models``, else llama.cpp's per-slot
    ``n_ctx`` from ``/props``. ``completion_cap``: a router's declared
    ``max_completion_tokens`` for its default provider. Either is None when
    nothing says, which reads as unknown -- the check below then has nothing
    to check, and says so in the record rather than inventing a limit.
    """
    limits: Dict[str, Optional[int]] = {"context_window": None, "completion_cap": None}
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with session.get(base_url.rstrip("/") + "/models",
                               headers=headers, timeout=timeout) as response:
            body = await response.json(content_type=None) if response.status == 200 else None
    except Exception:
        body = None
    entries = [e for e in (body.get("data") or [])
               if isinstance(e, dict)] if isinstance(body, dict) else []
    entry = next((e for e in entries if e.get("id") == model), None)
    if entry is None and len(entries) == 1:
        entry = entries[0]
    if entry:
        raw_top = entry.get("top_provider")
        top: Dict[str, Any] = raw_top if isinstance(raw_top, dict) else {}
        for key in ("max_model_len", "context_length"):
            if isinstance(entry.get(key), int):
                limits["context_window"] = entry[key]
                break
        else:
            if isinstance(top.get("context_length"), int):
                limits["context_window"] = top["context_length"]
        if isinstance(top.get("max_completion_tokens"), int):
            limits["completion_cap"] = top["max_completion_tokens"]
    if limits["context_window"] is None:
        root = base_url.rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root
        try:
            async with session.get(root + "/props", headers=headers,
                                   timeout=timeout) as response:
                props = await response.json(content_type=None) if response.status == 200 else None
        except Exception:
            props = None
        if isinstance(props, dict):
            gen = props.get("default_generation_settings")
            n_ctx = gen.get("n_ctx") if isinstance(gen, dict) else None
            if not isinstance(n_ctx, int):
                n_ctx = props.get("n_ctx")
            if isinstance(n_ctx, int):
                limits["context_window"] = n_ctx
    return limits


def budget_check(
    max_tokens: int, depths: Sequence[int], limits: Dict[str, Optional[int]],
) -> Tuple[List[str], List[str]]:
    """``(refusals, warnings)`` for a budget against the endpoint's ceilings.

    Three arms died of this. v24 set the budget *equal* to its provider's
    131,072 output cap, which looked like using everything available and is
    the one value guaranteed to produce `length` finishes if the model gets
    there -- 15 of 25 cells void. v32 set 200,000 against a 262,144 window, so
    at depth 32,768 the budget sat at 87% of the ~229,000 that fit, and 11 of
    17 cells voided. In both the budget had nowhere left to go.

    A budget that cannot fit at all is refused: the request is rejected or
    silently clamped, either way not what was asked. A budget within
    BUDGET_NEAR of the binding ceiling is warned about and recorded, because
    it may be a deliberate choice -- but it must not be an unnoticed one.
    """
    refusals: List[str] = []
    warnings: List[str] = []
    window, cap = limits.get("context_window"), limits.get("completion_cap")
    for depth in sorted(set(depths)):
        ceilings = []
        if window:
            ceilings.append((window - depth - PROMPT_OVERHEAD,
                             f"the {window:,}-token context window minus d{depth}"))
        if cap:
            ceilings.append((cap, f"the declared {cap:,}-token completion cap"))
        if not ceilings:
            continue
        ceiling, what = min(ceilings)
        if max_tokens > ceiling:
            refusals.append(
                f"--max-tokens {max_tokens:,} does not fit at d{depth}: "
                f"{what} leaves {max(ceiling, 0):,}")
        elif max_tokens >= BUDGET_NEAR * ceiling:
            warnings.append(
                f"d{depth}: --max-tokens {max_tokens:,} is {max_tokens / ceiling:.0%} "
                f"of {what} ({ceiling:,}). A trial that reaches it finishes "
                "`length`, and there is no room to raise the budget -- v32 "
                "lost 11 of 17 cells this way")
    return refusals, warnings


async def calibrate_depth(
    client: LLMClient, session: aiohttp.ClientSession, tokenizer: Any,
    depth: int, question: str, thinking: Optional[str],
    extra_body: Optional[Dict[str, Any]], reasoning_effort: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    """How large to build a haystack so the *endpoint* counts `depth` tokens.

    Hosted depths are sized with a local tokenizer, and no public one exists
    for several hosted models. The haystack is date-dense and Qwen splits
    digits one per token, so a vendor that merges them sees far fewer: grok's
    billing implied a prompt about half its nominal size, which made every
    hosted depth so far nominal. Two requests answer it exactly -- the question
    alone, and the question with a haystack built at the nominal size -- and
    the difference in the endpoint's own `prompt_tokens` is what the haystack
    costs in its tokens. `max_tokens` 1: only the prompt is wanted.

    Built from its own RNG, so calibrating does not move the run's haystacks.
    """
    rng = random.Random(f"llm-assay calibration {seed} {depth}")
    context, _ = build_context(tokenizer, depth, "clean", rng, 0.5)
    sized = _count(tokenizer, context) or depth
    counts = []
    for ctx in ("", context):
        body = await _post(client, session, _request_body(
            client, ctx, 1, thinking, question, extra_body, reasoning_effort))
        usage = usage_of(body) or {}
        if "prompt_tokens" not in usage:
            raise PreflightRefused(
                "--calibrate-depth needs the endpoint's usage.prompt_tokens, "
                "and this endpoint reports none")
        counts.append(usage["prompt_tokens"])
    haystack = counts[1] - counts[0]
    if haystack <= 0:
        raise PreflightRefused(
            f"calibration at d{depth} measured {haystack} haystack tokens; "
            "the endpoint's prompt counts are not usable")
    ratio = haystack / sized
    return {"sized_with": sized, "endpoint_counted": haystack,
            "ratio": round(ratio, 4), "build_depth": max(1, round(depth / ratio))}


#: A trace tail this repetitive is a loop, not work in progress. Measured:
#: genuine reasoning runs 11-26% unique words over its last 3,000 characters,
#: while the loops observed read 1.6% and 2.5%, and a Flash-Next trial that
#: burned its whole 190,000-token budget managed 396 words with 10 distinct.
#: The gap is wide enough that the threshold does not need to be delicate.
_LOOP_RATIO = 0.05
_LOOP_TAIL = 3000
_LOOP_MIN_WORDS = 50


def trace_repetition(trace: str) -> Optional[float]:
    """Unique-word ratio over the tail of a reasoning trace.

    Alphabetic words only, deliberately: one observed loop repeated
    ``[date] - already checked`` with the date changing every time, so counting
    numbers would have hidden a phrase that never varied. Cycling a small
    vocabulary counts as a loop too -- another repeated eight verbs in rotation
    without ever emitting the same line twice, which a substring check misses.

    ``None`` when there is too little text to judge.
    """
    words = re.findall(r"[a-zA-Z']+", (trace or "")[-_LOOP_TAIL:])
    if len(words) < _LOOP_MIN_WORDS:
        return None
    return round(len(set(w.lower() for w in words)) / len(words), 4)


def looks_degenerate(trace: str) -> bool:
    """Is this trace looping rather than working?

    Worth separating because the advice differs and one of them is actively
    wrong: a trial truncated at the budget is told to raise ``--max-tokens``,
    which for a loop simply buys a longer loop. It also keeps a depth curve
    honest -- a loop at 4,096 was otherwise counted alongside the deep trials
    that abandon a scan mid-enumeration, which is a different failure and the
    only one that tracks depth.
    """
    ratio = trace_repetition(trace)
    return ratio is not None and ratio < _LOOP_RATIO


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


def _spec_delta(
    before: Optional[ServerCounters], after: Optional[ServerCounters],
) -> Optional[Dict[str, Any]]:
    """What speculative decoding did across one trial, or None if unknowable.

    This exists because a generation setting is invisible to everything else a
    result records. `served_build` names the llama.cpp build and `served_model`
    the weights; neither moves when `--spec-draft-n-max` does. Measured
    mid-arm: the cap was raised between two trials of one pre-registered cell,
    the accepted run length went 3.37 to 4.22, and the only way to see it was
    to diff these counters from outside the tool. A saved trial should not need
    an outside observer to say what configuration produced it.

    All-or-nothing, like `_SUMMED`: a partial family yields a plausible wrong
    number. A counter that went backwards means the server restarted inside the
    window, which makes the difference meaningless rather than negative.

    The counters are server-global, so a second client on the endpoint lands
    inside this window too. That is tolerable for the purpose -- a
    configuration *change* shows up as a step far larger than foreign traffic
    moves the mean -- but it is why this is not published as a throughput
    measurement.
    """
    if before is None or after is None:
        return None
    delta: Dict[str, Any] = {}
    for logical, key in (("spec_drafts", "drafts"),
                         ("spec_draft_tokens", "draft_tokens"),
                         ("spec_accepted", "accepted")):
        b, a = before.get(logical), after.get(logical)
        if b is None or a is None:
            return None
        moved = a - b
        if moved < 0:
            return None
        delta[key] = int(moved)
    # The bonus token is excluded from the accepted counter by both vLLM and
    # llama.cpp, hence the +1 -- the same convention `acc_len` uses elsewhere.
    delta["accept_length"] = (
        round(delta["accepted"] / delta["drafts"] + 1, 3) if delta["drafts"] else None
    )
    return delta


def _sum_spec(deltas: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Cell total, only when every trial in it reported."""
    if not deltas or any(d is None for d in deltas):
        return None
    total: Dict[str, Any] = {
        k: sum(int(d[k]) for d in deltas if d)
        for k in ("drafts", "draft_tokens", "accepted")
    }
    total["accept_length"] = (
        round(total["accepted"] / total["drafts"] + 1, 3) if total["drafts"] else None
    )
    return total


async def run(
    base_url: str, api_key: str, model: str, tokenizer: Any,
    depths: List[int], rungs: List[str], trials: int, seed: int,
    thinking: Optional[str], max_tokens: int, endpoint: str,
    positions: List[float], transcripts: Optional[List[Dict[str, Any]]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    run_preflight: bool = True,
    reasoning_effort: Optional[str] = None,
    allow_binding_budget: bool = False,
    max_spend: Optional[float] = None,
    corpus: Optional[Any] = None,
    vocabulary: str = "builtin",
    answer_format: str = "line",
    calibrate: bool = False,
) -> Dict[str, Any]:
    client = LLMClient(base_url, api_key, model, None, False, endpoint)
    rng = random.Random(seed)
    # An archive fixes its own names: replaying a seeded corpus must score
    # against the surnames it was built from, whatever this run was told.
    if corpus is not None and corpus.manifest.get("names"):
        names: Tuple[str, ...] = tuple(corpus.manifest["names"])
        vocabulary = corpus.manifest.get("vocabulary_kind", vocabulary)
    else:
        names = names_for(vocabulary, seed)
    results: List[Dict[str, Any]] = []
    # Which backend actually served each trial. A router reports this per
    # response and nothing else does: /v1/models on OpenRouter lists the whole
    # catalogue, so `served_model` cannot say. More than one name here means
    # the pin failed and the cell mixed backends -- which, on a router whose
    # providers run different quantisations, means it mixed precisions too.
    providers_seen: Set[str] = set()

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
        served = await served_model(session, client.base_url, api_key, model)
        if served:
            print(f"  serving: {served}", file=sys.stderr, flush=True)
        # An alias pinned to something stable -- llama.cpp's default is the
        # literal string "llama.cpp" -- makes `served` useless for provenance
        # while costing the operator nothing, which is why it is worth pinning.
        # /props names the build the alias hides.
        build = await server_build(session, client.base_url, api_key)
        if build:
            print("  build: " + _build_summary(build), file=sys.stderr, flush=True)
        # Which engine, as against which weights. An engine upgrade changes
        # kernels, sampling and prefix caching, so cells measured across one do
        # not pool -- and until now no saved probe result could say.
        engine = await engine_version(session, client.base_url, api_key)
        if engine:
            print(f"  engine: {engine}", file=sys.stderr, flush=True)
        # Server-global counters, diffed per trial. Disables itself on a
        # definitive negative and returns None on a transient one, so an
        # endpoint publishing no /metrics costs nothing.
        checked: Optional[Dict[str, Any]] = None
        if max_spend is not None and not run_preflight:
            raise PreflightRefused(
                "--max-spend needs the pre-flight: it is what shows the endpoint "
                "reports a cost at all")
        if run_preflight:
            checked = await preflight(client, session, api_key, thinking, extra_body,
                                      reasoning_effort)
            via = f" via {checked['provider']}" if checked["provider"] else ""
            print(f"  pre-flight ok: '{model}' answered{via}",
                  file=sys.stderr, flush=True)
        # Money spent so far, from the endpoint's own `usage.cost`. A cap the
        # tool cannot measure is no cap, so an endpoint that reports no cost
        # refuses the run rather than letting it proceed unguarded.
        spent = 0.0
        if max_spend is not None:
            cost = ((checked or {}).get("usage") or {}).get("cost")
            if cost is None:
                raise PreflightRefused(
                    "--max-spend set, but this endpoint reports no cost in its "
                    "usage block, so the cap cannot be enforced")
            spent += cost
        halted: Optional[str] = None
        limits = await endpoint_limits(session, client.base_url, api_key, model)
        refusals, budget_warnings = budget_check(max_tokens, depths, limits)
        if refusals and not allow_binding_budget:
            raise BudgetRefused("; ".join(refusals)
                                + " (--allow-binding-budget to run anyway)")
        for w in refusals + budget_warnings:
            print(f"  WARNING: {w}", file=sys.stderr, flush=True)
        # Depths as the endpoint counts them, when asked for. The cell keeps
        # its label; what changes is how large its haystack is built.
        calibration: Optional[Dict[str, Any]] = None
        build_depth = {d: d for d in depths}
        if calibrate:
            if corpus is not None:
                raise PreflightRefused("--calibrate-depth cannot resize an archived corpus")
            calibration = {}
            for d in depths:
                c = await calibrate_depth(
                    client, session, tokenizer, d, _question_for("clean", answer_format),
                    thinking, extra_body, reasoning_effort, seed)
                calibration[str(d)] = c
                build_depth[d] = c["build_depth"]
                print(f"  calibrated d{d}: the endpoint counts {c['ratio']:.4f} of "
                      f"a local token; building {c['build_depth']:,}",
                      file=sys.stderr, flush=True)
        metrics = ServerMetricsProbe(client.base_url, api_key)
        for depth in depths:
            if halted:
                break
            for rung in rungs:
              if halted:
                break
              # A position sweep is meaningless for a rung that injects nothing.
              for position in (positions if rung in POSITIONABLE else [0.5]):
                if halted:
                    break
                tally = {"OK": 0, "PARTIAL": 0, "MISS": 0, "FALSE+": 0}
                errors: List[str] = []
                seen = 0
                # A wrong answer on the control rung splits three ways, and the
                # count alone hides which one you have: evidence manufactured
                # for the occasion, a right answer the one-line format extracted
                # a name from anyway, or a reply that cited nothing checkable.
                # Kept on the cell so a result is classifiable without the
                # optional transcripts beside it.
                cited = {kind: 0 for kind in FALSE_POSITIVE_KINDS}
                # Counted on every trial, not only the FALSE+ ones.
                # `cited` is gated on the verdict, and a verdict of
                # FALSE+ is only reachable when `expected is None` --
                # the `clean` rung. So a model that invents a line on
                # `log` or `outlier` scores PARTIAL and never reaches
                # the buckets at all. Measured: a vision pilot produced
                # exactly that, an `outlier` trial citing a real entry
                # with its date altered, invisible in `false_positives`.
                # This counter is additive and deliberately does not
                # change what `fabricated` means, because every banked
                # cell's number rests on the old definition.
                absent_claim_trials = 0
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
                cell_started = datetime.now(timezone.utc)
                cell_clock = time.monotonic()
                spec_deltas: List[Optional[Dict[str, Any]]] = []
                looped = 0
                # Trials that returned any reasoning at all. `seen` below is
                # structurally zero on an endpoint that withholds the trace --
                # gpt-6-luna named the injection 0 times in 10 with a trace on
                # only 2 -- so alone it describes what the vendor returns
                # rather than what the model saw. This is its denominator.
                traced = 0
                # FALSE+ trials bucketed `uncited` that nonetheless had a
                # trace. "Uncited" conflates two opposite things: no trace at
                # all is unverifiable in principle, while a trace the citation
                # extractor could not parse is a gap in this tool. grok-4.3's
                # four false positives at d32,768 were all the second kind --
                # the same one-field fabrication, paraphrased rather than
                # quoted -- and `fabricated` read 0. Additive, so the four
                # existing buckets keep the meaning banked cells rely on.
                uncited_with_trace = 0
                usages: List[Optional[Dict[str, Any]]] = []
                # Trials that finished `length`: the budget, not the model,
                # ended them. Every pre-registration from v14 voids a cell on
                # these, and until now the count had to be dug out of error
                # strings and transcripts.
                truncated = 0
                attempted = 0
                # `cited` format only: trials whose lines 2-3 offered any log
                # line, offered one the log lacks, and actually showed an
                # impossibility.
                evidence = {"offered": 0, "absent": 0, "proves": 0}
                # One digest per haystack read, so the cell can say which
                # bytes it measured whether they came from the generator or
                # an archive.
                read: List[str] = []
                for trial_index in range(trials):
                    # Checked before a trial, never during one: stopping
                    # between trials leaves a smaller cell, while killing one
                    # mid-request leaves money spent and nothing recorded.
                    if max_spend is not None and spent >= max_spend:
                        halted = (f"spend cap reached: ${spent:.4f} of "
                                  f"${max_spend:.4f}; stopped before trial "
                                  f"{attempted + 1} of d{depth} {rung}")
                        print(f"  STOP: {halted}", file=sys.stderr, flush=True)
                        break
                    attempted += 1
                    if corpus is not None:
                        context, expected = corpus.take(depth, rung, position,
                                                        trial_index)
                    else:
                        context, expected = build_context(
                            tokenizer, build_depth[depth], rung, rng, position, names)
                    read.append(_sha256(context))
                    # Wall clock for "when", monotonic for "how long": the two
                    # answer different questions and a clock adjustment breaks
                    # the second if it is derived from the first.
                    t_started = datetime.now(timezone.utc)
                    t_clock = time.monotonic()
                    t_before = await metrics.snapshot(session)
                    trial_usage: List[Optional[Dict[str, Any]]] = []
                    try:
                        reply, reasoning, finish, provider = await _ask(
                            client, session, context, max_tokens, thinking,
                            _question_for(rung, answer_format), extra_body, trial_usage,
                            reasoning_effort)
                        if provider:
                            providers_seen.add(provider)
                        t_elapsed = time.monotonic() - t_clock
                    except (ProbeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        t_elapsed = time.monotonic() - t_clock
                        usages.append(getattr(exc, "usage", None))
                        spent += (usages[-1] or {}).get("cost") or 0.0
                        if getattr(exc, "finish_reason", None) == "length":
                            truncated += 1
                        # Not a wrong answer: a question that was never answered.
                        if len(errors) < 3 and str(exc) not in errors:
                            errors.append(str(exc))
                        completed += 1
                        # A cell can run for hours; a trial that fails printed
                        # nothing before this, so a run watched live looked
                        # like it had skipped a trial number with no
                        # explanation until the cell finished.
                        # Two different failures wear the same "unanswered"
                        # label, and only one of them tracks depth. Naming the
                        # loop at the trial keeps it out of a depth curve it
                        # does not belong to.
                        if looks_degenerate(getattr(exc, "reasoning", "")):
                            looped += 1
                            print(f"    trial {completed}/{trials}: UNANSWERED  "
                                  f"(degenerate loop -- the trace repeats; a bigger "
                                  f"--max-tokens buys a longer loop)",
                                  file=sys.stderr, flush=True)
                        else:
                            print(f"    trial {completed}/{trials}: UNANSWERED  "
                                  f"({str(exc)[:150]})", file=sys.stderr, flush=True)
                        # Recorded like any other trial, because "which trials
                        # did not answer, and what was the model doing" is the
                        # question the unanswered column raises and cannot
                        # answer. Dropping them left the transcript blind to
                        # precisely the trials worth reading.
                        spec_deltas.append(
                            _spec_delta(t_before, await metrics.snapshot(session)))
                        trace = getattr(exc, "reasoning", "")
                        # An unanswered trial can still have quoted the log
                        # while working, so it is checked here too -- and the
                        # stored excerpt is far too short to check afterwards.
                        err_citations = check_citations(trace, context)
                        if err_citations["absent_claims"]:
                            absent_claim_trials += 1
                        if transcripts is not None:
                            transcripts.append({
                                "depth": depth, "rung": rung, "position": position,
                                "expected": expected, "verdict": "UNANSWERED",
                                "error": str(exc),
                                "started_at": t_started.isoformat(),
                                "duration_s": round(t_elapsed, 3),
                                "spec_decode": spec_deltas[-1],
                                "named_in_reasoning": bool(
                                    marker and marker.lower() in trace.lower()),
                                "reasoning_chars": len(trace),
                                "reasoning_tokens": _count(tokenizer, trace),
                                "finish_reason": getattr(exc, "finish_reason", None),
                                "usage": usages[-1],
                                "haystack_sha256": read[-1],
                                "reply": "",
                                # Checked even here: an unanswered trial can
                                # still have quoted the log while working, and
                                # the excerpt below is too short to check later.
                                "citations": err_citations,
                                "trace_repetition": trace_repetition(trace),
                                "reasoning_excerpt": trace.strip()[-3000:],
                            })
                        continue
                    spec_deltas.append(
                        _spec_delta(t_before, await metrics.snapshot(session)))
                    usages.append(trial_usage[0] if trial_usage else None)
                    spent += (usages[-1] or {}).get("cost") or 0.0
                    # Under `cited` the verdict is line 1 alone: lines 2-3 name
                    # people by design, and scoring them would trip the hedging
                    # rule on every answer that shows its evidence.
                    if answer_format == "cited":
                        verdict_text = (reply.strip().splitlines() or [""])[0]
                    else:
                        verdict_text = reply.strip().replace("\n", " ")
                    verdict = score(verdict_text, expected,
                                    names if rung == "log" else None)
                    shown = (evidence_check(reply, context)
                             if answer_format == "cited" else None)
                    if shown:
                        evidence["offered"] += bool(shown["lines"])
                        evidence["absent"] += bool(shown["absent"])
                        evidence["proves"] += bool(shown["proves"])
                    tally[verdict] += 1
                    # Over the whole trace, not the stored excerpt: the line a
                    # trial invented is usually quoted where it was first
                    # reasoned about, which at these trace lengths is tens of
                    # thousands of characters before the end.
                    citations = check_citations(reply + "\n" + reasoning, context)
                    trial_provider = provider
                    if verdict == "FALSE+":
                        kind = classify_false_positive(reply, citations, names)
                        cited[kind] += 1
                        if kind == "uncited" and reasoning.strip():
                            uncited_with_trace += 1
                    if citations["absent_claims"]:
                        absent_claim_trials += 1
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
                    if reasoning.strip():
                        traced += 1
                    completed += 1
                    print(f"    trial {completed}/{trials}: {verdict}",
                          file=sys.stderr, flush=True)
                    if transcripts is not None:
                        transcripts.append({
                            "depth": depth, "rung": rung, "position": position,
                            "expected": expected, "verdict": verdict,
                            "named_in_reasoning": named,
                            "started_at": t_started.isoformat(),
                            "duration_s": round(t_elapsed, 3),
                            "spec_decode": spec_deltas[-1],
                            "reasoning_chars": len(reasoning),
                            "reasoning_tokens": _count(tokenizer, reasoning),
                            # Which field the verdict was read from, and what
                            # the server said about completeness. Without both,
                            # a transcript cannot distinguish an answer from a
                            # scratchpad -- which is how four scored trials
                            # turned out to be the model still working.
                            "finish_reason": finish,
                            # Which backend served this trial. None on a direct
                            # endpoint, which reads as unknown, never as "the
                            # same one as the others".
                            "provider": trial_provider,
                            "usage": usages[-1],
                            "haystack_sha256": read[-1],
                            "evidence": shown,
                            # Whole, not the first 400 characters. The
                            # classification above reads the reply for the
                            # surname it names, so a truncated one can change a
                            # verdict -- and at a large budget the model answers
                            # in paragraphs, not a word.
                            "reply": reply.strip(),
                            "citations": citations,
                            "trace_repetition": trace_repetition(reasoning),
                            "reasoning_excerpt": reasoning.strip()[-3000:],
                        })
                answered = sum(tally.values())
                lo, hi = wilson(tally["OK"], answered)
                lo_frac, hi_frac = span(position)
                results.append({
                    "depth": depth, "rung": rung, "position": position,
                    # When, and for how long. A cell that ran slow because the
                    # box was shared should say so in its own artifact instead
                    # of being reconstructed from file mtimes -- which breaks
                    # the moment a run is paused between two trials.
                    "started_at": cell_started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_s": round(time.monotonic() - cell_clock, 3),
                    "spec_decode": _sum_spec(spec_deltas),
                    # Of the unanswered trials, how many were looping rather
                    # than working. Reported separately because raising the
                    # budget helps one and not the other.
                    "degenerate": looped,
                    # Only the log rung has two halves to separate. Recorded so
                    # a sweep cannot publish a position without also publishing
                    # how much distance changed underneath it.
                    "separation": round(hi_frac - lo_frac, 3) if rung == "log" else None,
                    # Trials actually run. Equal to the requested count unless
                    # the spend cap stopped the cell short.
                    "trials": attempted, "answered": answered, "errors": errors,
                    "mentioned_while_reasoning": seen,
                    "traced": traced,
                    # Only the control rung has a false positive to classify;
                    # elsewhere this is three zeros and says so.
                    "false_positives": dict(cited),
                    # How many trials cited a line the haystack does not
                    # contain, on any rung and at any verdict. `false_positives`
                    # above answers "of the wrong answers, what kind"; this
                    # answers "how often did it invent evidence at all".
                    "absent_claim_trials": absent_claim_trials,
                    "uncited_with_trace": uncited_with_trace,
                    "truncated": truncated,
                    "evidence": evidence if answer_format == "cited" else None,
                    # A digest over every haystack the cell read, in order.
                    # Equal digests mean byte-identical inputs, which is what a
                    # paired comparison assumes and could only infer before.
                    "haystack_digest": _sha256("".join(read)),
                    # The endpoint's own token counts, summed. None when no
                    # trial's response carried a usage block.
                    "usage": sum_usage(usages),
                    "accuracy": tally["OK"] / answered if answered else None,
                    "ci95_low": lo, "ci95_high": hi, **{k.lower(): v for k, v in tally.items()},
                })
                # Unanswered trials are not missing at random. A trial the
                # model reasons through quickly finishes; one it struggles with
                # runs long and hits the token ceiling. Dropping those biases
                # accuracy *upward*, so a high rate has to be said out loud
                # rather than left as a quiet column.
                lost = attempted - answered
                note = "" if not lost else f"  [{lost} unanswered]"
                if lost and attempted and lost / attempted >= 0.2:
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
                    saw = f"  saw={seen}/{answered} (traced {traced})"
                print(f"  d{depth:<7} {rung:8} @{position:<4} {tally['OK']}/{answered}  "
                      f"95% CI [{lo:.2f}, {hi:.2f}]{saw}{note}", flush=True)
                for err in errors:
                    print(f"      error: {err}", file=sys.stderr)
    return {
        "version": __version__, "model": model, "served_model": served,
        "engine_version": engine,
        # What the endpoint says it loaded, when it says anything: llama.cpp's
        # /props names the GGUF, the quantisation and the server build, which is
        # the only provenance available on a box whose alias is pinned. None
        # means the endpoint published none of it; absent, in a file written
        # before this field existed, means unknown.
        "served_build": build,
        "base_url": base_url,
        "seed": seed, "thinking": thinking, "trials": trials,
        # The output budget. Never recorded before, although every void rule
        # in the programme is a statement about it.
        "max_tokens": max_tokens,
        # What the endpoint said its ceilings were, and what was said about
        # the budget against them. None values are unknown, not unlimited.
        "limits": limits,
        "budget_warnings": refusals + budget_warnings,
        # Why the run stopped early, or None when it ran every trial asked of
        # it. Cells after the stop are absent; the one it stopped in records
        # only the trials it ran.
        "halted": halted,
        "max_spend": max_spend,
        "spent": round(spent, 6) if max_spend is not None else None,
        # The effort level the requests asked for, from --reasoning-effort or
        # an --extra-body that carried it (which wins, being merged last);
        # None when neither did, meaning the vendor's default -- which is not
        # one value across vendors. grok-4.3's defaults to `low`, so v14
        # compared deepseek at up to 232,000 reasoning tokens against grok at
        # ~122: default deployments, not capabilities.
        "reasoning_effort": effective_effort(reasoning_effort, extra_body),
        "depths": depths, "rungs": rungs, "positions": positions,
        # None means the requested tokenizer loaded; a string names the
        # tokenizer that failed and says every depth above is a gpt2 count,
        # not a real one. Absent-from-older-files means unknown, not "loaded
        # fine" -- the fallback existed long before this field did.
        "tokenizer_fallback": tokenizer_fallback,
        # What the haystacks are made of, and where they came from: the
        # generator's word lists and template, and -- when replayed from an
        # archive -- the digest of its manifest, which pins every byte read.
        "vocabulary": vocabulary_digest(names),
        "vocabulary_kind": vocabulary,
        # Which question: `line` (one line, every banked result) or `cited`
        # (the verdict plus the two lines that show it). The verdict is scored
        # identically, but the reply quotes far more under `cited`, so the
        # fabrication buckets see more -- the two do not pool.
        "answer_format": answer_format,
        # Per depth, when --calibrate-depth ran: the local count of a probe
        # haystack, the endpoint's count of it, their ratio, and the size the
        # cells were built at. None means depths are in the sizing
        # tokenizer's units -- nominal, for any other vocabulary.
        "calibration": calibration,
        "corpus": corpus.digest if corpus is not None else None,
        # What was asked for beyond the standard body, and which backends
        # actually answered. On a router these are the provenance: `providers`
        # holding more than one name means the pin did not hold and the cell
        # mixed backends -- and where those backends run different
        # quantisations, it mixed precisions with nothing else recording it.
        # Empty dict means nothing extra was sent, the same rule as
        # `extra_body`'s in BenchmarkMetadata; an empty list means no response
        # named a provider, which is normal for a direct endpoint.
        "extra_body": dict(extra_body or {}),
        "providers": sorted(providers_seen),
        # What the pre-flight saw, or None when --no-preflight skipped it.
        "preflight": checked,
        # The retrieval floor at each depth that ran one, keyed on MISS. Saved
        # beside the cells it qualifies so a reader of the file does not have
        # to recompute the gate by hand, which every pre-registration did.
        # String keys because JSON has no others.
        "floors": {str(d): f for d, f in floor_verdicts(results).items()},
        "results": results,
    }


#: Past this relative gap between the nominal depth and the endpoint's own
#: prompt count, the depth label is saying something the endpoint disagrees
#: with. The chat template and question add a few hundred tokens, so a small
#: excess is expected; a vendor tokenizer counting the text differently is not.
DEPTH_GAP = 0.10


def depth_notes(cells: Iterable[Dict[str, Any]]) -> List[str]:
    """Where the endpoint's prompt count disagrees with the nominal depth.

    Hosted depths are sized with a local tokenizer, and no public tokenizer
    exists for several hosted models. The haystack is date-dense and Qwen
    splits digits one per token, so a vendor whose tokenizer merges them sees
    far fewer tokens: grok's billing implied a prompt about half its nominal
    size. Reading `prompt_tokens` back from the usage block makes the real
    depth part of the artifact rather than an inference from a bill.
    """
    out = []
    for c in cells:
        u = c.get("usage") or {}
        n = u.get("prompt_tokens_trials")
        depth = c.get("depth") or 0
        if not n or depth <= 0:
            continue
        mean = u["prompt_tokens"] / n
        if abs(mean - depth) / depth > DEPTH_GAP:
            out.append(f"d{depth} {c['rung']}: the endpoint counted "
                       f"{mean:,.0f} prompt tokens per trial against a nominal "
                       f"{depth:,} -- its tokenizer and the sizing one disagree, "
                       "so compare models by this figure, not the label")
    return out


def truncation_notes(cells: Iterable[Dict[str, Any]],
                     max_tokens: Optional[int]) -> List[str]:
    """Cells where the budget, not the model, ended some trials."""
    out = []
    for c in cells:
        n = c.get("truncated") or 0
        if n:
            at = f" at {max_tokens:,} tokens" if max_tokens else ""
            out.append(f"d{c['depth']} {c['rung']}: {n} of {c['trials']} trial(s) "
                       f"finished `length`{at} -- the budget ended them, and the "
                       "rest of the cell is the subset that finished sooner")
    return out


def evidence_notes(cells: Iterable[Dict[str, Any]]) -> List[str]:
    """Under `--answer-format cited`: what the offered evidence showed, per cell."""
    out = []
    for c in cells:
        e = c.get("evidence")
        if not e or not e.get("offered"):
            continue
        out.append(f"d{c['depth']} {c['rung']}: {e['offered']} trial(s) cited "
                   f"evidence; {e['proves']} showed a real impossibility, "
                   f"{e['absent']} cited a line the log does not contain")
    return out


def uncited_notes(cells: Iterable[Dict[str, Any]]) -> List[str]:
    """False positives the citation check could not read, though a trace existed."""
    out = []
    for c in cells:
        n = c.get("uncited_with_trace") or 0
        if n:
            out.append(f"d{c['depth']} {c['rung']}: {n} uncited false "
                       "positive(s) had a reasoning trace the citation check "
                       "could not parse -- paraphrased evidence, unverified "
                       "by this tool, not absent")
    return out


def render(report: Dict[str, Any]) -> str:
    lines = [
        "",
        f"quality probe | model: {report['model']} | trials per cell: {report['trials']}"
        + (f" | thinking: {report['thinking']}" if report["thinking"] else "")
        + (f" | effort: {report['reasoning_effort']}"
           if report.get("reasoning_effort") else ""),
    ]
    # The table is what gets pasted into a note; if the endpoint named its
    # build, the note should carry it rather than leaving the reader to open
    # the JSON. `served_model` is deliberately not printed here -- on a pinned
    # alias it says "llama.cpp" and reads as provenance while being none.
    if report.get("served_build"):
        lines.append(f"build: {_build_summary(report['served_build'])}")
    lines.append("")
    if report.get("tokenizer_fallback"):
        # Printed on the console table too, not just stderr scrollback -- a
        # table that looks complete is exactly what got this bug missed the
        # first time.
        lines.append(
            f"WARNING: tokenizer '{report['tokenizer_fallback']}' failed to "
            f"load; every depth above is a gpt2 count, not a real one.")
        lines.append("")
    lines += [
        "| depth | rung | at | sep | accuracy | 95% CI | partial | false+ | fabricated "
        "| unanswered | floor |",
        "|------:|:-----|---:|----:|---------:|:-------|--------:|-------:|-----------:"
        "|-----------:|:------|",
    ]
    floors = floor_verdicts(report["results"])
    for r in report["results"]:
        unanswered = r["trials"] - r["answered"]
        at = "-" if r["rung"] == "clean" else f"{r['position']:.2f}"
        sep = "-" if r.get("separation") is None else f"{r['separation']:.2f}"
        # A cell with no false positive has nothing to classify, and printing a
        # 0 there invites it to be read as "checked, and none were invented".
        buckets = r.get("false_positives") or {}
        fabricated = f"{buckets.get('fabricated', 0)}" if r["false+"] else "-"
        lines.append(
            f"| {r['depth']} | {r['rung']} | {at} | {sep} | {r['ok']}/{r['answered']} |"
            f" [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] |"
            f" {r['partial']} | {r['false+']} | {fabricated} | {unanswered} |"
            f" {floor_note(r['depth'], r['rung'], floors) or '-'} |"
        )
    warnings = floor_warnings(report["results"])
    # Saved since the router arms, printed only now. On an aggregator the
    # providers behind one model id run different quantisations, so a cell
    # served by two of them mixed precisions with nothing else recording it.
    providers = report.get("providers") or []
    if len(providers) > 1:
        warnings.append(f"trials were served by {len(providers)} different "
                        f"providers ({', '.join(providers)}): the pin did not "
                        "hold, and the cells may mix precisions")
    warnings += report.get("budget_warnings") or []
    if report.get("halted"):
        warnings.append(f"run halted early -- {report['halted']}; later cells "
                        "were not run")
    if warnings:
        lines.append("")
        lines += [f"WARNING: {w}" for w in warnings]
    notes = (depth_notes(report["results"]) + uncited_notes(report["results"])
             + truncation_notes(report["results"], report.get("max_tokens"))
             + evidence_notes(report["results"]))
    if notes:
        lines.append("")
        lines += [f"note: {n}" for n in notes]
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
        f"`floor` is the outlier cell at the same depth, gated on MISS (at most "
        f"{FLOOR_MISS_MAX:.0%}",
        "of trials answering CONSISTENT with a foreign line planted). PARTIAL is not a",
        "miss: the model flagged something, in the wrong shape. A `clean` or `log` rate",
        "beside `none` or `FAILED` is unqualified, however good it looks.",
        "",
        "`fabricated` splits that false+ count, because it hides two different",
        "failures. A reply quoting a log line the haystack does not contain has",
        "*manufactured its evidence*; one that quotes only real lines reached a wrong",
        "verdict, or the one-line answer format extracted a surname from a reply that",
        "had actually concluded CONSISTENT. The check is mechanical -- the claim a",
        "citation makes is (date, place, surname), and the model is held to it, not to",
        "its wording. `--save-transcripts` records the per-trial detail, including the",
        "nearest real line to each invented one.",
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
    ap.add_argument("base_url", nargs="?", default=None,
                    help="OpenAI-compatible endpoint URL, including /v1 "
                         "(not needed with --build-corpus)")
    ap.add_argument("--model", required=True, help="Model to probe")
    ap.add_argument("--served-model-name", default=None,
                    help="Name the API expects, when it differs from --model "
                         "(which also selects the tokenizer)")
    # Defaulted from the environment because a key passed on the command line
    # is world-readable in `ps` for the life of the trial. Measured: a probe
    # against a hosted endpoint showed the full key in a process listing, which
    # is worse than a shell history because every user on the box can read it
    # without trying. PROBE_API_KEY is how a wrapper script should supply it.
    ap.add_argument("--api-key", default=None,
                    help="API key for the endpoint. Prefer the LLM_ASSAY_API_KEY "
                         "environment variable (PROBE_API_KEY also works): a key "
                         "given here is visible in `ps` to every user on the machine")
    ap.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    # Defaults are applied after parsing, so that --corpus-dir can tell a value
    # the user chose from one argparse filled in.
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="Context depths to probe (default: 4096 32768)")
    ap.add_argument("--rungs", nargs="+", choices=RUNGS, default=None,
                    help="Which rungs to run (default: all three)")
    ap.add_argument("--positions", type=float, nargs="+", default=None,
                    metavar="F",
                    help="Where to inject, as fractions of the context "
                         "(default: 0.5). Attention is strongly position "
                         "dependent -- the ends are privileged and the middle is "
                         "not -- so 0.1 0.5 0.9 separates that from depth. "
                         "Applies to the outlier and log rungs; clean injects "
                         "nothing")
    ap.add_argument("--trials", type=int, default=None,
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
    ap.add_argument("--extra-body", action="append", default=None, metavar="K=V",
                    help="Extra JSON fields merged into each request, key=value "
                         "with a JSON value, repeatable. Required to pin a "
                         "router to one backend: "
                         "--extra-body 'provider={\"only\":[\"deepinfra\"],"
                         "\"allow_fallbacks\":false}'. Without a pin a router "
                         "may serve different trials from backends running "
                         "different quantisations, and the cell silently mixes "
                         "them")
    ap.add_argument("--reasoning-effort", default=None, metavar="LEVEL",
                    help="Send this top-level reasoning_effort (e.g. low, "
                         "medium, high) and record it. Unset means the "
                         "vendor's default, which differs between vendors: "
                         "grok-4.3 defaults to low, and its 0%% false-positive "
                         "rate at d131,072 became 20%% at high. Cannot be "
                         "combined with --thinking off, which sends 'none'")
    ap.add_argument("--allow-binding-budget", action="store_true",
                    help="Run even when --max-tokens cannot fit under the "
                         "endpoint's context window or declared completion "
                         "cap at some depth. Refused by default: such a "
                         "request is rejected or silently clamped")
    ap.add_argument("--max-spend", type=float, default=None, metavar="USD",
                    help="Stop before the next trial once the endpoint's "
                         "reported cost (usage.cost) reaches this many dollars. "
                         "Refuses to start on an endpoint that reports no cost, "
                         "since the cap could not be enforced there")
    ap.add_argument("--vocabulary", choices=VOCABULARIES, default="builtin",
                    help="Surnames for the haystack: `builtin`, the fifty "
                         "fixed names every banked result used, or `seeded`, "
                         "nonsense names generated from --seed that exist in "
                         "no training set. The two read different text from "
                         "the same seed and do not compare")
    ap.add_argument("--answer-format", choices=ANSWER_FORMATS, default="line",
                    help="`line` (default, every banked result): one line, the "
                         "surname, OFFTOPIC or CONSISTENT. `cited`: that line, "
                         "then the two log lines that show the impossibility, "
                         "copied verbatim. The verdict is scored from line 1 "
                         "alone; the evidence is checked against the haystack. "
                         "Makes fabricated evidence visible on models that "
                         "never quote while reasoning")
    ap.add_argument("--calibrate-depth", action="store_true",
                    help="Size each depth in the endpoint's own tokens rather "
                         "than the local tokenizer's. Sends two max_tokens=1 "
                         "requests per depth -- the question alone, and with a "
                         "haystack -- and rebuilds at the measured ratio. For "
                         "hosted models with no public tokenizer, where a "
                         "nominal depth can be half the real one or less")
    ap.add_argument("--build-corpus", default=None, metavar="DIR",
                    help="Write every haystack this run would read into DIR, "
                         "one file per (depth, rung, position, trial), with a "
                         "manifest of sha256 digests and expected answers, "
                         "then exit without contacting any endpoint")
    ap.add_argument("--corpus-dir", default=None, metavar="DIR",
                    help="Read haystacks from an archive written by "
                         "--build-corpus instead of generating them. Every "
                         "file is checked against the manifest and the run "
                         "refuses on any mismatch. Depths, rungs, positions, "
                         "trials and seed come from the manifest")
    ap.add_argument("--no-preflight", action="store_true",
                    help="Skip the one-request check sent before the first "
                         "trial. It exists because a wrong model name or "
                         "provider pin fails every trial identically, and "
                         "those failures were once banked as the model "
                         "declining to answer")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit the report as JSON instead of a table")
    ap.add_argument("--save-result", default=None, help="Write the JSON report to this path")
    ap.add_argument("--save-transcripts", default=None, metavar="PATH",
                    help="Write each trial's answer and reasoning trace to PATH as "
                         "JSON. The accuracy number cannot distinguish a model "
                         "that never saw the injection from one that saw it and "
                         "judged it consistent; the trace can")
    args = ap.parse_args()
    from .config import resolve_api_key
    args.api_key = resolve_api_key(args.api_key, "PROBE_API_KEY")

    from .haystacks import Archive, CorpusError, build as build_corpus
    corpus = None
    if args.corpus_dir:
        if args.build_corpus:
            ap.error("--corpus-dir and --build-corpus are opposites; pass one")
        try:
            corpus = Archive(args.corpus_dir)
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        m = corpus.manifest
        for flag, given, recorded in (
                ("--depths", args.depths, m["depths"]), ("--rungs", args.rungs, m["rungs"]),
                ("--positions", args.positions, m["positions"]),
                ("--trials", args.trials, m["trials"]), ("--seed", args.seed, m["seed"])):
            if given is not None and given != recorded:
                ap.error(f"{flag} {given} disagrees with the corpus manifest "
                         f"({recorded}); the archive fixes it")
        args.depths, args.rungs, args.positions = m["depths"], m["rungs"], m["positions"]
        args.trials, args.seed = m["trials"], m["seed"]
    if args.depths is None:
        args.depths = [4096, 32768]
    if args.rungs is None:
        args.rungs = list(RUNGS)
    if args.positions is None:
        args.positions = [0.5]
    if args.trials is None:
        args.trials = 6
    if args.base_url is None and not args.build_corpus:
        ap.error("base_url is required unless --build-corpus is given")

    if args.trials < 1:
        ap.error("--trials must be >= 1")
    if any(d < 0 for d in args.depths):
        ap.error("--depths must be >= 0")
    if any(not 0.0 <= f <= 1.0 for f in args.positions):
        ap.error("--positions are fractions of the context, so 0.0 to 1.0")
    if args.max_spend is not None and args.max_spend <= 0:
        ap.error("--max-spend must be a positive number of dollars")
    if args.reasoning_effort is not None and args.thinking == "off":
        ap.error("--thinking off already sends reasoning_effort=none; "
                 "do not combine it with --reasoning-effort")

    # No corpus. Every rung generates its own haystack, so a token counter is
    # all that is needed and there is no book to download, no slice to warn
    # about running past the end of, and no --book-url to imply the source text
    # affected the result.
    tokenizer = load_tokenizer(args.model, args.tokenizer)

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    if args.build_corpus:
        try:
            manifest = build_corpus(
                args.build_corpus, tokenizer, args.tokenizer or args.model, seed,
                list(args.depths), list(args.rungs), list(args.positions), args.trials,
                args.vocabulary)
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {len(manifest['files'])} haystacks and {args.build_corpus}/"
              f"manifest.json (seed {seed}, vocabulary {manifest['vocabulary']})")
        return 0
    transcripts: Optional[List[Dict[str, Any]]] = [] if args.save_transcripts else None
    try:
        report = asyncio.run(run(
            args.base_url, args.api_key, args.served_model_name or args.model, tokenizer,
            args.depths, list(args.rungs), args.trials, seed, args.thinking,
            args.max_tokens, args.endpoint, list(args.positions), transcripts,
            BenchmarkConfig._parse_extra_body(args.extra_body),
            run_preflight=not args.no_preflight,
            reasoning_effort=args.reasoning_effort,
            allow_binding_budget=args.allow_binding_budget,
            max_spend=args.max_spend, corpus=corpus, vocabulary=args.vocabulary,
            answer_format=args.answer_format, calibrate=args.calibrate_depth,
        ))
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except PreflightRefused as exc:
        # Exit 3, distinct from 1, so a wrapper can stop instead of retrying:
        # a rejected configuration fails the same way on every attempt.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ProbeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        # Trial failures are caught inside run(); only the pre-flight reaches
        # here. The endpoint did not answer, which is worth retrying.
        print(f"error: pre-flight could not reach the endpoint: {exc}",
              file=sys.stderr)
        return 1

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
