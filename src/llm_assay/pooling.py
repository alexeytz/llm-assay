"""Sum per-trial probe results back into cells.

A cell run one trial per process writes one file per (rung, trial), so a crash
costs one trial instead of a whole cell, and the pooled totals are what get
published. That is the reason this lives in the package rather than in a
wrapper script: the package is type-checked and tested, and the standing rule
is that pooled numbers which become published numbers move here with tests. The same rule already put
`wilson` and `fisher_exact` in `probe.py`.

Two distinctions are load-bearing and easy to lose:

* **Absent means unknown, never different.** A file written before
  `served_model`, `served_build` or the false-positive buckets existed says
  nothing about which build produced it. Counting such a file as a second build
  would invent a conflict; counting its missing buckets as zeros would
  manufacture a clean arm out of unrecorded data. So the identified counts are
  reported alongside the values, and a caller that needs "did every file say?"
  can compare them against `sources`.
* **Transcripts are detail, not results.** They sit beside each result file and
  pooling them would double-count every trial.
"""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import json
import os

from .probe import FALSE_POSITIVE_KINDS, floor_verdicts, floor_warnings

#: Counters summed straight across files. `false+` is the total the buckets
#: then subdivide, and is kept because a file predating the buckets still has
#: it -- which is what makes "unclassified" detectable rather than silent.
_SUMMED_FIELDS = ("trials", "answered", "ok", "partial", "miss", "false+")

CellKey = Tuple[int, str]
Cell = Dict[str, int]
BuildKey = Tuple[Tuple[str, Any], ...]


class PooledCells:
    """Cells plus everything needed to say whether they may be pooled at all."""

    def __init__(self) -> None:
        self.cells: Dict[CellKey, Cell] = {}
        self.fallbacks: Set[str] = set()
        self.served: Set[str] = set()
        self.builds: Set[BuildKey] = set()
        #: How many files actually recorded each fact, as against how many were
        #: pooled. One file naming a build says nothing about the other ten, and
        #: a header printing that name alone reads as though it did.
        self.identified: Dict[str, int] = {"served": 0, "build": 0, "effort": 0}
        self.sources: int = 0
        #: Effort levels the pooled files recorded. A file that recorded none
        #: -- older, or run at the vendor default -- is counted apart in
        #: `identified["effort"]`, never as a level of its own.
        self.efforts: Set[str] = set()
        #: Answer formats seen. A file predating the field asked the one-line
        #: question, so absent reads as `line` -- the only format there was.
        self.formats: Set[str] = set()

    def as_rows(self) -> List[Dict[str, Any]]:
        """Cells as the probe's own result rows, so its floor logic applies."""
        return [{"depth": d, "rung": r, **c}
                for (d, r), c in sorted(self.cells.items())]

    def floors(self) -> Dict[int, Dict[str, Any]]:
        """The retrieval floor at each depth that pooled one, keyed on MISS."""
        return floor_verdicts(self.as_rows())

    def notes(self) -> List[str]:
        """Depth and citation notes over the pooled cells, as the probe prints."""
        from .probe import depth_notes, truncation_notes, uncited_notes
        rows = []
        for row in self.as_rows():
            row = dict(row)
            row["usage"] = {"prompt_tokens": row.get("prompt_tokens", 0),
                            "prompt_tokens_trials": row.get("prompt_tokens_trials", 0)}
            rows.append(row)
        return (depth_notes(rows) + uncited_notes(rows)
                + truncation_notes(rows, None))

    def floor_warnings(self) -> List[str]:
        """Each clean/log cell the floor does not qualify, said out loud."""
        return floor_warnings(self.as_rows())

    def as_tuple(self) -> Tuple[Dict[CellKey, Cell], Set[str], Set[str],
                                Set[BuildKey], Dict[str, int], int]:
        """The original six-tuple, so the CLI keeps its existing shape."""
        return (self.cells, self.fallbacks, self.served, self.builds,
                self.identified, self.sources)


def result_files(target: str) -> List[str]:
    """Result files under `target`, or `target` itself when it is a file.

    Skips `-transcripts.json`: those are per-trial detail written beside each
    result, and pooling them would count every trial twice.
    """
    if os.path.isfile(target):
        return [target]
    out = []
    for name in sorted(os.listdir(target)):
        if name.endswith(".json") and not name.endswith("-transcripts.json"):
            out.append(os.path.join(target, name))
    return out


def pool(targets: Iterable[str]) -> PooledCells:
    """Sum every result under `targets` into one cell per (depth, rung)."""
    pooled = PooledCells()
    for target in targets:
        for path in result_files(target):
            with open(path) as fh:
                report = json.load(fh)
            pooled.sources += 1
            _absorb_provenance(pooled, report)
            for r in report.get("results", []):
                _absorb_cell(pooled, r)
    return pooled


def _absorb_provenance(pooled: PooledCells, report: Dict[str, Any]) -> None:
    from .probe import effective_effort
    effort = report.get("reasoning_effort") or effective_effort(
        None, report.get("extra_body"))
    if effort:
        pooled.efforts.add(effort)
        pooled.identified["effort"] += 1
    pooled.formats.add(report.get("answer_format") or "line")
    if report.get("tokenizer_fallback"):
        pooled.fallbacks.add(report["tokenizer_fallback"])
    if report.get("served_model"):
        pooled.served.add(report["served_model"])
        pooled.identified["served"] += 1
    # Only files that recorded one. A file written before the field existed, or
    # measured against a server publishing no /props, is unknown -- pooling it
    # must not look like a second build.
    build = report.get("served_build")
    if build:
        pooled.builds.add(tuple(sorted(build.items())))
        pooled.identified["build"] += 1


def _absorb_cell(pooled: PooledCells, r: Dict[str, Any]) -> None:
    key: CellKey = (r["depth"], r["rung"])
    cell = pooled.cells.setdefault(key, {
        **{f: 0 for f in _SUMMED_FIELDS},
        **{k: 0 for k in FALSE_POSITIVE_KINDS},
    })
    for field in _SUMMED_FIELDS:
        cell[field] += r.get(field, 0) or 0
    # Absent in a file written before the buckets existed, which leaves them
    # zero -- the same reading as "no false positive to classify", and
    # distinguishable from it only by the false+ column.
    for kind, count in (r.get("false_positives") or {}).items():
        if kind in cell:
            cell[kind] += count
    # Additive fields from later versions. Absent in an older file reads as
    # zero here, the same convention as the buckets: a pooled zero is a lower
    # bound, never evidence that nothing was there.
    for name in ("uncited_with_trace", "truncated"):
        cell[name] = cell.get(name, 0) + (r.get(name) or 0)
    usage = r.get("usage") or {}
    for name in ("prompt_tokens", "prompt_tokens_trials"):
        cell[name] = cell.get(name, 0) + (usage.get(name) or 0)


def unclassified_false_positives(cell: Cell) -> int:
    """`false+` trials that no bucket accounts for.

    A file predating the buckets reports `fabricated: 0` for a cell that has
    false positives, and a comparison run over it would manufacture a null out
    of missing data. That is why `--compare` refuses rather than reports.
    """
    bucketed = sum(cell.get(k, 0) for k in FALSE_POSITIVE_KINDS)
    return max(0, cell.get("false+", 0) - bucketed)


#: Past this share of unanswered trials, an arm is measuring its survivors --
#: which are not a random subset of what was run. Fixed by the pre-registration,
#: not chosen here.
UNANSWERED_CEILING = 0.2


class Arm:
    """One side of a comparison: a pooled `clean` cell plus its provenance."""

    def __init__(self, target: str, cell: Cell, builds: Set[BuildKey],
                 sources: int, fallbacks: Set[str],
                 efforts: Optional[Set[str]] = None) -> None:
        self.target = target
        self.efforts = efforts or set()
        self.cell = cell
        self.builds = builds
        self.sources = sources
        self.fallbacks = fallbacks

    @property
    def answered(self) -> int:
        return self.cell["answered"]

    @property
    def fabricated(self) -> int:
        return self.cell["fabricated"]

    @property
    def unanswered(self) -> int:
        return self.cell["trials"] - self.cell["answered"]

    @property
    def over_ceiling(self) -> bool:
        t = self.cell["trials"]
        return bool(t) and self.unanswered / t > UNANSWERED_CEILING


class Comparison:
    """The result of the pre-registered test, or the reason there isn't one.

    `error` is a refusal: the comparison was not run and must not be reported.
    `warnings` are conditions worth printing beside a result that *was* run.
    """

    def __init__(self) -> None:
        self.arms: List[Arm] = []
        self.warnings: List[str] = []
        self.error: Optional[str] = None
        self.p: Optional[float] = None


def load_arm(target: str) -> Tuple[Optional[Arm], List[str], Optional[str]]:
    """Pool one arm and decide whether it may be compared at all."""
    warnings: List[str] = []
    pooled = pool([target])
    if pooled.fallbacks:
        warnings.append(
            f"{target}: tokenizer fallback: {sorted(pooled.fallbacks)}")
    if len(pooled.builds) > 1:
        warnings.append(f"{target}: pooled across different server builds")
    if len(pooled.formats) > 1:
        warnings.append(
            f"{target}: pooled across answer formats {sorted(pooled.formats)}; "
            "`cited` replies quote far more, so their fabrication counts are "
            "not the same measurement")
    if len(pooled.efforts) > 1 or (pooled.efforts
                                   and pooled.identified["effort"] < pooled.sources):
        warnings.append(
            f"{target}: pooled across different reasoning_effort settings "
            f"({', '.join(sorted(pooled.efforts))}"
            + (", and the vendor default" if pooled.identified["effort"] < pooled.sources
               else "") + ")")

    clean = [c for (_, rung), c in pooled.cells.items() if rung == "clean"]
    if len(clean) != 1:
        return None, warnings, (
            f"{target}: expected exactly one clean cell, found {len(clean)}")

    cell = clean[0]
    # A file written before the buckets existed classifies nothing, and an
    # unclassified false positive is indistinguishable from an absent one: the
    # arm reports `fabricated: 0` and the test duly reports p=1. That is a null
    # manufactured out of missing data, so it refuses rather than reports.
    unclassified = unclassified_false_positives(cell)
    if unclassified:
        bucketed = cell["false+"] - unclassified
        return None, warnings, (
            f"{target}: {cell['false+']} false positive(s) but {bucketed} "
            "classified -- these trials predate the fabrication buckets and "
            "cannot be compared")

    # Every pre-registration from v17 computed this gate by hand. A clean rate
    # with no floor beside it, or a failed one, is unqualified; the arm still
    # compares, because the floor is often run in a directory of its own, but
    # the reader is told rather than left to notice.
    depth = next(d for (d, rung) in pooled.cells if rung == "clean")
    floor = pooled.floors().get(depth)
    if floor is None:
        warnings.append(
            f"{target}: no retrieval floor pooled at d{depth}; the clean rate is "
            "unqualified unless a floor at this depth passed elsewhere")
    elif not floor["passed"]:
        warnings.append(
            f"{target}: the retrieval floor FAILED at d{depth} (MISS "
            f"{floor['misses']}/{floor['trials']}); the clean rate is not "
            "interpretable")

    arm = Arm(target, cell, pooled.builds, pooled.sources, pooled.fallbacks,
              pooled.efforts)
    if arm.over_ceiling:
        warnings.append(
            f"{target}: {arm.unanswered}/{cell['trials']} unanswered -- above "
            f"the {UNANSWERED_CEILING:.0%} the pre-registration sets as "
            "invalidating")
    return arm, warnings, None


def compare_arms(shallow: str, deep: str) -> Comparison:
    """Fisher exact, two-sided, on fabricated against answered-minus-fabricated.

    Rendering is left to the caller; everything that decides a number or
    refuses to produce one lives here, because these are published numbers and
    the package is where they are type-checked and tested.
    """
    from .probe import fisher_exact

    result = Comparison()
    for target in (shallow, deep):
        arm, warnings, error = load_arm(target)
        result.warnings.extend(warnings)
        if error:
            result.error = error
            return result
        assert arm is not None
        result.arms.append(arm)

    a, b = result.arms
    # Two arms measured against different weights are not a comparison at all.
    if a.builds and b.builds and a.builds != b.builds:
        result.warnings.append(
            "the two arms were measured against different server builds")
    # An effort difference changes what is being compared: v14 set deepseek's
    # default against grok's `low` and read it as a capability gap. Unset on
    # one side means the vendor default, which is a setting too.
    if a.efforts != b.efforts:
        def _say(efforts: Set[str]) -> str:
            return ", ".join(sorted(efforts)) or "vendor default"
        result.warnings.append(
            f"the two arms were run at different reasoning_effort "
            f"({_say(a.efforts)} against {_say(b.efforts)})")

    result.p = fisher_exact(a.fabricated, a.answered - a.fabricated,
                            b.fabricated, b.answered - b.fabricated)
    return result


# --- paired comparison on shared seeds ---------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar: a binomial test on the discordant pairs.

    `b` and `c` are the two discordant counts. With none, nothing disagreed and
    there is no evidence of a difference: p = 1.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


class PairedComparison:
    """Two McNemar tests over trials matched by seed, or the reason there are none.

    `all_trials` scores "this trial produced the correct answer" over every
    pair, an unanswered trial counting as not correct. It does not condition on
    answering, so the 20% unanswered rule cannot void it -- which is exactly
    why v25 registered it after v21's marginal test died of censoring.
    `complete_pairs` keeps only pairs where both arms answered, v13's design;
    it is robust to censoring the arms share, not to censoring that differs.
    """

    def __init__(self) -> None:
        self.error: Optional[str] = None
        self.warnings: List[str] = []
        self.pairs = 0
        self.unmatched: Tuple[int, int] = (0, 0)
        self.all_trials: Dict[str, Any] = {}
        self.complete_pairs: Dict[str, Any] = {}


def _trials_by_seed(target: str, rung: str) -> Tuple[Dict[Tuple[int, int], Dict[str, int]], int]:
    """{(seed, depth): {answered, ok}} for one-trial files; and how many were not."""
    out: Dict[Tuple[int, int], Dict[str, int]] = {}
    multi = 0
    for path in result_files(target):
        with open(path) as fh:
            report = json.load(fh)
        seed = report.get("seed")
        for r in report.get("results", []):
            if r.get("rung") != rung:
                continue
            if r.get("trials") != 1 or seed is None:
                multi += 1
                continue
            out[(seed, r["depth"])] = {"answered": r.get("answered", 0) or 0,
                                       "ok": r.get("ok", 0) or 0}
    return out, multi


def paired_compare(a_target: str, b_target: str, rung: str = "clean") -> PairedComparison:
    """Match two arms' per-trial files by (seed, depth) and run both tests.

    A seed and a depth fix the haystack byte for byte, so a matched pair asked
    two configurations the identical question -- what lets a paired test cancel
    the content variance a marginal one carries. Only one-trial files pair:
    a whole-cell file holds several haystacks under one seed field and cannot
    say which trial read which.
    """
    result = PairedComparison()
    a, a_multi = _trials_by_seed(a_target, rung)
    b, b_multi = _trials_by_seed(b_target, rung)
    if a_multi or b_multi:
        result.warnings.append(
            f"{a_multi + b_multi} multi-trial cell(s) skipped: only one-trial "
            "result files can be matched by seed")
    shared = sorted(set(a) & set(b))
    result.unmatched = (len(set(a) - set(b)), len(set(b) - set(a)))
    if any(result.unmatched):
        result.warnings.append(
            f"unmatched trials: {result.unmatched[0]} only in the first arm, "
            f"{result.unmatched[1]} only in the second; they are left out")
    if not shared:
        result.error = (f"no trials share a (seed, depth) on the {rung} rung, so "
                        "there is nothing to pair; a paired test needs both arms "
                        "run on the same seed base")
        return result
    for target in (a_target, b_target):
        fallbacks = pool([target]).fallbacks
        if fallbacks:
            result.warnings.append(
                f"{target}: tokenizer fallback {sorted(fallbacks)} -- the same "
                "seed may not have built the same haystack")
    result.pairs = len(shared)

    def correct(t: Dict[str, int]) -> bool:
        return bool(t["answered"]) and bool(t["ok"])

    a_only = sum(1 for k in shared if correct(a[k]) and not correct(b[k]))
    b_only = sum(1 for k in shared if correct(b[k]) and not correct(a[k]))
    result.all_trials = {
        "pairs": len(shared),
        "correct": (sum(correct(a[k]) for k in shared), sum(correct(b[k]) for k in shared)),
        "discordant": (a_only, b_only), "p": mcnemar_exact(a_only, b_only)}

    complete = [k for k in shared if a[k]["answered"] and b[k]["answered"]]
    ca = sum(1 for k in complete if a[k]["ok"] and not b[k]["ok"])
    cb = sum(1 for k in complete if b[k]["ok"] and not a[k]["ok"])
    result.complete_pairs = {
        "pairs": len(complete),
        "correct": (sum(a[k]["ok"] for k in complete), sum(b[k]["ok"] for k in complete)),
        "discordant": (ca, cb), "p": mcnemar_exact(ca, cb)}
    return result
