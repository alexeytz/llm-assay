"""Statistical comparison of two or more saved benchmark runs.

``llm-assay`` characterises one endpoint. Comparing two *configurations* is a
different job, and eyeballing overlapping ``mean ± std`` bars is a poor way to do
it -- especially with speculative decoding, where run-to-run spread is driven by
content-dependent acceptance and can easily exceed the effect being looked for.

This performs an explicit test:

* **Paired** (preferred) when both runs were produced with the same ``--seed``.
  Each run index saw identical corpus text, so content variance cancels and the
  test is far more sensitive.
* **Welch's t-test** otherwise -- unequal variances, no pooling assumption.

Usage::

    python llm-assay.py compare baseline.json candidate.json
    python llm-assay.py compare a.json b.json --metric e2e_ttft --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

METRICS = (
    "tg_throughput",
    "pp_throughput",
    "e2e_ttft",
    "ttfr",
    "est_ppt",
    "peak_throughput",
)

# Metrics where a smaller number is better (latencies).
LOWER_IS_BETTER = {"e2e_ttft", "ttfr", "est_ppt"}


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """Two-sided tail probability of Student's t with ``df`` degrees of freedom.

    This has to be the t distribution, not the normal. ``--runs 3`` gives df=2,
    where the two disagree violently: t=3.474 is p=0.0005 under the normal and
    p=0.0738 under t -- a 144x understatement that turns "no difference" into a
    confident verdict at alpha=0.05. ``results.py`` already carries a Student-t
    table for exactly this reason; a p-value computed from the normal threw that
    care away at the last step.

    scipy is not a dependency and is not worth adding for one function, so this
    is the standard incomplete-beta identity: p = I_{df/(df+t^2)}(df/2, 1/2).
    """
    if df <= 0 or math.isnan(t):
        return 1.0
    if math.isinf(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _t_upper(t: float, df: float) -> float:
    """One-sided P(T > t). ``_t_sf`` is two-sided, so halve and pick the side."""
    two = _t_sf(t, df)
    return two / 2.0 if t >= 0 else 1.0 - two / 2.0


def welch_parts(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """(mean difference, standard error, df) for the unequal-variance comparison.

    Split out from :func:`welch` because equivalence testing needs the same three
    estimates as the difference test -- it just asks a different question of them.
    """
    na, nb = len(a), len(b)
    va, vb = _var(a), _var(b)
    se2 = va / na + vb / nb
    md = _mean(b) - _mean(a)
    if se2 <= 0:
        return md, 0.0, float(na + nb - 2)
    df = se2**2 / (
        ((va / na) ** 2 / (na - 1) if na > 1 else 0.0)
        + ((vb / nb) ** 2 / (nb - 1) if nb > 1 else 0.0)
    )
    return md, math.sqrt(se2), df


def paired_parts(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """(mean difference, standard error, df) for matched samples."""
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return (_mean(diffs) if diffs else 0.0), 0.0, 0.0
    sd = math.sqrt(_var(diffs))
    return _mean(diffs), sd / math.sqrt(n), float(n - 1)


def tost(md: float, se: float, df: float, bound: float) -> float:
    """Two One-Sided Tests for equivalence. Returns the TOST p-value.

    A plain t-test can only ever *fail to reject* "the difference is zero", which
    is why "no difference" is not evidence of equivalence -- it conflates "this
    change is harmless" with "you did not measure enough". TOST asks the question
    that can actually be answered: given a margin you declare up front, is the
    difference confined inside +/-bound?

    Two one-sided tests, hence the name: reject "difference <= -bound" and reject
    "difference >= +bound". Equivalence is claimed only if both are rejected, so
    the p-value is the *larger* of the two -- the weaker piece of evidence.
    """
    if bound <= 0:
        return 1.0
    if se <= 0:
        # No spread at all: the point estimate decides it outright.
        return 0.0 if abs(md) < bound else 1.0
    p_above_lower = _t_upper((md + bound) / se, df)      # H0: diff <= -bound
    p_below_upper = 1.0 - _t_upper((md - bound) / se, df)  # H0: diff >= +bound
    return max(p_above_lower, p_below_upper)


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Returns (t, df, p) for unequal-variance two-sample comparison."""
    na, nb = len(a), len(b)
    va, vb = _var(a), _var(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        # Zero variance on both sides: any non-zero gap is a certain difference,
        # an identical mean is a certain non-difference. Reporting p=1 here would
        # invert the most clear-cut case there is.
        gap = _mean(b) - _mean(a)
        if gap == 0:
            return 0.0, float(na + nb - 2), 1.0
        return math.inf if gap > 0 else -math.inf, float(na + nb - 2), 0.0
    se = math.sqrt(se2)
    t = (_mean(b) - _mean(a)) / se
    df = se2**2 / (
        ((va / na) ** 2 / (na - 1) if na > 1 else 0.0)
        + ((vb / nb) ** 2 / (nb - 1) if nb > 1 else 0.0)
    )
    return t, df, _t_sf(t, df)


def paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Paired t-test on matched samples."""
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0, 1.0
    sd = math.sqrt(_var(diffs))
    md = _mean(diffs)
    if sd == 0:
        # Every pair moved by exactly the same amount. That is the strongest
        # possible evidence of a real effect, not the weakest.
        if md == 0:
            return 0.0, float(n - 1), 1.0
        return math.inf if md > 0 else -math.inf, float(n - 1), 0.0
    t = md / (sd / math.sqrt(n))
    return t, float(n - 1), _t_sf(t, float(n - 1))


# Dimensions a comparison can be run *across* -- dropped from the row key so the
# two sides line up instead of landing in separate rows that never meet.
ACROSS_DIMENSIONS = ("thinking", "reasoning_effort")


def shape_key(run: dict, across: Sequence[str] = ()) -> str:
    """Identify a benchmarked shape.

    Prompt and generation size belong in here. Without them a `--pp 256 1024`
    sweep collapses onto one key and all but the last shape is silently
    overwritten, and a pp512 baseline lines up against a pp2048 candidate as
    though they were the same measurement.

    ``across`` names dimensions to leave out, which is what makes an A/B *of*
    that dimension possible: with `thinking` in the key, a thinking=on file and
    a thinking=off file share no rows at all and nothing is compared.
    """
    phase = "ctx" if run.get("is_context_prefill_phase") else "inf"
    parts = [
        f"d{run.get('context_size', 0)}",
        f"pp{run.get('prompt_size', 0)}",
        f"tg{run.get('response_size', 0)}",
        phase,
    ]
    if run.get("concurrency", 1) > 1:
        parts.append(f"c{run['concurrency']}")
    # Every swept dimension must be in the key. Leaving one out is how a sweep
    # silently collapses: two shapes hash the same and only the last survives.
    if run.get("thinking") and "thinking" not in across:
        parts.append(f"think={run['thinking']}")
    if run.get("reasoning_effort") and "reasoning_effort" not in across:
        parts.append(f"re={run['reasoning_effort']}")
    return " ".join(parts)


def load(path: str, across: Sequence[str] = ()) -> Tuple[Dict[str, dict], Optional[int], Dict[str, Any]]:
    with open(path) as fh:
        doc = json.load(fh)
    table: Dict[str, dict] = {}
    for run in doc.get("benchmarks", []):
        key = shape_key(run, across)
        # Dropping a dimension can collide two rows of the *same* file -- a file
        # that swept --thinking on off compared across thinking has two rows per
        # shape. Keeping the last would silently discard the other, which is the
        # bug the key exists to prevent, so refuse instead.
        if key in table:
            raise ValueError(
                f"{path} has more than one row for '{key}' once "
                f"{', '.join(across)} is dropped from the key: it swept that "
                "dimension itself, so there is no single row to compare. Compare "
                "files that each hold one arm."
            )
        table[key] = run
    return table, doc.get("seed"), doc


# Settings that change the numbers, so comparing across a difference in one of
# them measures the setting rather than whatever the user meant to test.
_COMPARABILITY: Tuple[Tuple[str, str], ...] = (
    ("model", "model"),
    ("latency_mode", "latency mode"),
    ("endpoint", "--endpoint"),
    ("exact_tg", "--exact-tg"),
    ("no_cache", "--no-cache"),
    ("cold", "--cold"),
    ("prefix_caching_enabled", "--measure-cached-followup"),
    ("adapt_prompt", "--adapt-prompt"),
    ("reasoning_efforts", "--reasoning-effort"),
    ("thinking_modes", "--thinking"),
    ("book_url", "corpus"),
    ("tokenizer", "tokenizer"),
    ("tokenizer_fallback", "tokenizer fallback to gpt2"),
    ("extra_body", "--extra-body"),
    ("version", "llm-assay version"),
)


def comparability_warnings(docs: List[Dict[str, Any]], paths: List[str]) -> List[str]:
    """Flag settings that differ between the files being compared.

    Older result files predate most of these fields; a missing value is unknown,
    not a difference, so it is skipped rather than reported.
    """
    notes = []
    for field, label in _COMPARABILITY:
        values = [doc.get(field) for doc in docs]
        if any(v is None for v in values):
            continue
        if len(set(map(repr, values))) == 1:
            continue
        shown = ", ".join(f"{_short(p)}={v!r}" for p, v in zip(paths, values))
        notes.append(f"warning: {label} differs between runs ({shown})")
    return notes


def analyse(
    paths: List[str],
    metric: str,
    alpha: float,
    force_unpaired: bool,
    equivalence_margin: Optional[float] = None,
    across: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run every comparison and return the result as data.

    Rendering (table or JSON) is deliberately separate: a verdict a human reads
    and a verdict CI gates on must come from the same arithmetic, or the two will
    eventually disagree about the same run.
    """
    tables: List[Dict[str, dict]] = []
    seeds: List[Optional[int]] = []
    docs: List[Dict[str, Any]] = []
    for path in paths:
        table, seed, doc = load(path, across)
        tables.append(table)
        seeds.append(seed)
        docs.append(doc)

    base = tables[0]
    # --cold salts the corpus offsets per invocation, so two cold runs read
    # different text. A paired test assumes run i of each side saw the same
    # content; pairing them would compare noise to noise and call it precision.
    cold_salts = [doc.get("cold_salt") for doc in docs]
    cold_blocks_pairing = any(doc.get("cold") for doc in docs) and len(set(cold_salts)) > 1
    # Comparing *across* a dimension that is in the corpus key means the two
    # sides read different text, so run i of each side never saw the same
    # content and pairing has nothing to cancel. Labelling that "paired" would
    # advertise a sensitivity the test does not have.
    across_blocks_pairing = bool(across)
    pairable = (
        not force_unpaired
        and len(set(seeds)) == 1
        and seeds[0] is not None
        and not cold_blocks_pairing
        and not across_blocks_pairing
    )
    shared_seed = seeds[0] if (len(set(seeds)) == 1 and seeds[0] is not None) else None

    notes: List[str] = []
    mismatched_text: set = set()
    if not pairable and any(s is not None for s in seeds):
        # Two different reasons to be unpaired. Saying "no shared seed" when the
        # runs plainly share one tells the user something false about their own
        # data, and the advice that follows it is already-done work.
        # Order matters: a cold run is unpairable whatever the seeds say, and
        # only --unpaired can be blamed on --unpaired.
        if across_blocks_pairing:
            notes.append(
                "note: comparing across " + ", ".join(across) + ", which is part of "
                "the corpus key -- the two sides read different text, so the test is "
                "unpaired however the seeds line up."
            )
        elif cold_blocks_pairing:
            notes.append(
                "note: these runs used --cold, which salts the corpus per invocation, so "
                "each side saw different text and a paired test is not valid. Cold runs "
                "measure honest prefill; for a paired comparison drop --cold and prime "
                "the cache once instead."
            )
        elif force_unpaired and shared_seed is not None:
            notes.append(
                f"note: --unpaired forced Welch even though both runs share --seed "
                f"{shared_seed}. Drop --unpaired for a paired test, which is much "
                "more sensitive."
            )
        else:
            notes.append(
                "note: runs do not share a --seed, so a paired test is not valid. "
                "Re-run both sides with the same --seed for a much more sensitive comparison."
            )
    notes.extend(comparability_warnings(docs, paths))

    # Shapes the baseline does not have are never tested, so say so rather than
    # letting them disappear.
    for path, table in zip(paths[1:], tables[1:]):
        missing = [key for key in table if key not in base]
        if missing:
            notes.append(
                f"note: {_short(path)} has {len(missing)} shape(s) absent from the baseline, "
                f"not compared: {', '.join(sorted(missing)[:3])}"
                + (" ..." if len(missing) > 3 else "")
            )

    lower_is_better = metric in LOWER_IS_BETTER
    rows: List[Dict[str, Any]] = []
    counts = {"significant": 0, "regressions": 0, "improvements": 0,
              "equivalent": 0, "inconclusive": 0, "fragile": 0}

    for key in base:
        a_metric = base[key].get(metric) or {}
        a_vals = a_metric.get("values") or []
        if not a_vals:
            continue
        a_mean = _mean(a_vals)
        row: Dict[str, Any] = {
            "shape": key,
            "baseline_mean": a_mean,
            "baseline_n": len(a_vals),
            "comparisons": [],
        }
        for path, other in zip(paths[1:], tables[1:]):
            b_metric = (other.get(key) or {}).get(metric) or {}
            b_vals = b_metric.get("values") or []
            if not b_vals:
                row["comparisons"].append({"file": _short(path), "measured": False})
                continue

            # A paired test assumes run i of each side saw the same text. That
            # was previously taken on trust from the seed; when both files
            # record a corpus fingerprint it can simply be checked, and a
            # mismatch means the assumption is false whatever the seeds say.
            # Only ever downgrades: a missing fingerprint is unknown, not equal,
            # and a matching one is not treated as licence to pair runs the seed
            # rule already rejected.
            a_print = (base.get(key) or {}).get("corpus_fingerprint")
            b_print = (other.get(key) or {}).get("corpus_fingerprint")
            text_differs = bool(a_print and b_print and a_print != b_print)
            if text_differs:
                mismatched_text.add(key)

            use_paired = pairable and len(a_vals) == len(b_vals) and not text_differs
            if use_paired:
                _stat, df, pvalue = paired(a_vals, b_vals)
                md, se, _df = paired_parts(a_vals, b_vals)
            else:
                _stat, df, pvalue = welch(a_vals, b_vals)
                md, se, _df = welch_parts(a_vals, b_vals)

            b_mean = _mean(b_vals)
            delta = (b_mean / a_mean - 1) * 100 if a_mean else 0.0
            # df < 1 means the test never ran -- welch()'s zero-variance path
            # returns p=0.0 with df=0 at one sample per side, which would
            # otherwise count as a certain difference on the same row the
            # verdict labels untestable, and exit 1 under --fail-on-regression.
            # At n>=2 that path returns df=na+nb-2>=2, so a genuine
            # zero-variance difference still reports as significant.
            significant = pvalue < alpha and df >= 1
            better = (delta < 0) if lower_is_better else (delta > 0)

            entry: Dict[str, Any] = {
                "file": _short(path),
                "measured": True,
                # The suite-level choice is not the row-level one: pairing needs
                # equal sample counts, so a shape that lost a run on one side
                # falls back to Welch. A consumer acting on the p-value has to be
                # able to see which test produced it.
                "method": "paired" if use_paired else "welch",
                "mean": b_mean,
                "n": len(b_vals),
                "delta_pct": delta,
                "p": pvalue,
                "df": df,
                "significant": significant,
                # A t-test needs df >= 1. At one sample per side there is no
                # spread to divide by, so the test is not merely inconclusive --
                # it was never run. Calling that "no difference" hides an
                # arbitrarily large visible gap behind a word that reads like a
                # measurement, which is the opposite of the zero-variance path's
                # behaviour and the more dangerous direction of the two.
                "verdict": (
                    "untestable (n<2)" if df < 1
                    else ("BETTER" if better else "WORSE") if significant
                    else "no difference"
                ),
            }

            # A paired test divides by the spread of the *differences*, so two
            # runs that drifted together by a near-constant amount give p~0 for
            # any shift size. Speculative-decode acceptance drifts exactly like
            # that between invocations, so at n<3 a "significant" verdict can be
            # measuring drift rather than the change under test. The arithmetic
            # is right; the variance estimate behind it is not credible yet.
            fragile = significant and min(len(a_vals), len(b_vals)) < 3
            entry["fragile"] = fragile
            if fragile:
                counts["fragile"] += 1

            if significant:
                counts["significant"] += 1
                counts["improvements" if better else "regressions"] += 1

            if equivalence_margin:
                bound = abs(equivalence_margin * a_mean)
                tost_p = tost(md, se, _df, bound)
                equivalent = tost_p < alpha
                entry["equivalence"] = {
                    "margin_frac": equivalence_margin,
                    "margin_abs": bound,
                    "p": tost_p,
                    "equivalent": equivalent,
                    # False when df < 1: the numbers above come from tost()'s
                    # zero-spread branch on a sample that cannot support them.
                    "testable": df >= 1,
                }
                if not significant and df >= 1:
                    # Only meaningful where the difference test found nothing:
                    # this is what separates "harmless" from "not measured enough".
                    #
                    # df >= 1 for the same reason the verdict and the significance
                    # flag carry it. tost()'s zero-spread branch decides on the
                    # point estimate alone, which is right when n>=2 samples
                    # genuinely agree and unsound when there is no spread because
                    # there is no sample: at one run per side it would upgrade
                    # "untestable" into the strongest equivalence claim available.
                    entry["verdict"] = "EQUIVALENT" if equivalent else "INCONCLUSIVE"
                    counts["equivalent" if equivalent else "inconclusive"] += 1

            row["comparisons"].append(entry)
        rows.append(row)

    if mismatched_text:
        notes.append(
            f"note: {len(mismatched_text)} shape(s) recorded different corpus "
            "fingerprints, so the two sides did not read the same text; those rows "
            "fall back to Welch. A paired test there would divide by the spread of "
            "differences between unrelated samples."
        )

    methods = {
        c["method"]
        for row in rows for c in row["comparisons"] if c.get("measured")
    }
    return {
        "metric": metric,
        # "mixed" when rows disagree, so the header cannot claim a test that half
        # the table did not use.
        "test": methods.pop() if len(methods) == 1 else ("mixed" if methods else
                ("paired" if pairable else "welch")),
        "methods": sorted(methods) or None,
        "alpha": alpha,
        "equivalence_margin": equivalence_margin,
        "baseline": _short(paths[0]),
        "files": [_short(p) for p in paths],
        "notes": notes,
        "rows": rows,
        "summary": counts,
    }


def render_table(result: Dict[str, Any], paths: List[str]) -> None:
    if result["test"] == "mixed":
        mode = "mixed -- paired where both sides have equal runs, Welch elsewhere"
    elif result["test"] == "paired":
        mode = "paired"
    else:
        mode = "Welch (unpaired)"
    print(f"metric: {result['metric']}   test: {mode}   alpha={result['alpha']}")
    if result["equivalence_margin"]:
        print(
            f"equivalence margin: +/-{result['equivalence_margin'] * 100:.1f}% "
            "(TOST; EQUIVALENT means the difference is demonstrably inside it)"
        )
    for note in result["notes"]:
        print(note)
    print()

    header = f"{'test':<30}{'baseline':>14}"
    for path in paths[1:]:
        header += f"{_short(path):>16}{'delta':>10}{'p':>9}  verdict"
    print(header)
    print("-" * (len(header) + 8))

    for row in result["rows"]:
        line = f"{row['shape']:<30}{row['baseline_mean']:>14.2f}"
        for entry in row["comparisons"]:
            if not entry.get("measured"):
                line += f"{'-':>16}{'-':>10}{'-':>9}"
                continue
            line += (
                f"{entry['mean']:>16.2f}{entry['delta_pct']:>+9.1f}%"
                f"{entry['p']:>9.4f}  {entry['verdict']}"
            )
        print(line)

    s = result["summary"]
    print(f"\n{s['significant']} statistically significant difference(s) at "
          f"alpha={result['alpha']}.")
    if s.get("fragile"):
        print(
            f"[WARNING] {s['fragile']} of those rest on fewer than 3 runs per side. A paired\n"
            "          test divides by the spread of the differences, so two runs that drifted\n"
            "          together -- which is what speculative-decode acceptance does between\n"
            "          invocations -- read as certain regardless of the size of the shift.\n"
            "          Use at least 3 runs, or --target-ci, before gating on this."
        )
    if result["equivalence_margin"]:
        print(
            f"{s['equivalent']} shape(s) demonstrably equivalent, "
            f"{s['inconclusive']} inconclusive -- an inconclusive row has not shown "
            "equivalence,\nit has shown that the run was too noisy to tell. Raise "
            "--runs or use --target-ci."
        )
    else:
        print(
            "Rows marked 'no difference' are not evidence of equivalence -- they may "
            "just be underpowered; raise --runs or use --target-ci, or pass "
            "--equivalence-margin\nto test for equivalence directly."
        )


def _across_hint(
    paths: List[str], metric: str, alpha: float, across: Sequence[str]
) -> str:
    """Name --across when it is what would have made the rows line up.

    Blaming "depth, pp, tg, concurrency and phase" when the real difference is
    --thinking tells the user something false about their own data, and hides
    the one flag that fixes it.
    """
    remaining = [d for d in ACROSS_DIMENSIONS if d not in across]
    if not remaining:
        return ""

    def matches(extra: Sequence[str]) -> bool:
        try:
            retry = analyse(
                paths, metric, alpha, True, None, tuple(across) + tuple(extra)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return any(
            c.get("measured") for row in retry["rows"] for c in row["comparisons"]
        )

    # Smallest set that works, so the advice names the dimension that actually
    # differs rather than every dimension it could have been.
    candidates = [[d] for d in remaining]
    if len(remaining) > 1:
        candidates.append(remaining)
    for extra in candidates:
        if matches(extra):
            flags = " ".join(f"--across {d}" for d in extra)
            noun = "which is" if len(extra) == 1 else "which are"
            return (
                f" The files differ in {' and '.join(extra)}, {noun} part of the "
                f"row key; pass {flags} to compare across it."
            )
    return ""


def compare(
    paths: List[str],
    metric: str,
    alpha: float,
    force_unpaired: bool,
    equivalence_margin: Optional[float] = None,
    as_json: bool = False,
    fail_on_regression: bool = False,
    across: Sequence[str] = (),
) -> int:
    result = analyse(paths, metric, alpha, force_unpaired, equivalence_margin, across)
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        render_table(result, paths)

    # A comparison that compared nothing is not a passing comparison. Two files
    # whose shapes do not overlap -- an A/B where the generation length changed,
    # say -- produce a table of baseline-only rows, "0 statistically significant
    # difference(s)", and exit 0, which reads to a CI gate exactly like a clean
    # result. This mirrors the benchmark's own rule that a suite which measured
    # nothing must not report success.
    compared = sum(
        1
        for row in result["rows"]
        for c in row["comparisons"]
        if c.get("measured")
    )
    if not compared:
        print(
            f"error: no shape in the candidate file(s) matched the baseline on "
            f"{metric}; nothing was compared.{_across_hint(paths, metric, alpha, across)} "
            "Shapes must agree on depth, pp, tg, concurrency and phase to be "
            "compared -- changing any of them makes a different measurement, not a "
            "comparable one.",
            file=sys.stderr,
        )
        return 1

    # Gating on a p-value is only defensible now that p comes from the t
    # distribution; on the normal it understated p by ~144x at --runs 3 and a
    # gate would have blocked on noise constantly.
    if fail_on_regression and result["summary"]["regressions"]:
        n = result["summary"]["regressions"]
        print(
            f"error: {n} significant regression(s) on {metric} at alpha={alpha}.",
            file=sys.stderr,
        )
        return 1
    return 0


def _short(path: str, width: int = 15) -> str:
    """A column-width label for a result file.

    Timestamped names from a generated runner are routinely longer than the
    column, and an overflowing label runs into the neighbouring header. Keep the
    tail, which is the part that distinguishes two runs of the same preset.
    """
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".json"):
        name = name[:-5]
    return name if len(name) <= width else "…" + name[-(width - 1):]


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="llm-assay.py compare",
        description="Statistically compare saved llm-assay JSON results.",
    )
    ap.add_argument("files", nargs="+", help="JSON result files; the first is the baseline")
    ap.add_argument(
        "--metric",
        default="tg_throughput",
        choices=METRICS,
        help="Metric to compare (default: tg_throughput)",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    ap.add_argument(
        "--unpaired",
        action="store_true",
        help="Force Welch's test even when both runs share a --seed",
    )
    ap.add_argument(
        "--across",
        action="append",
        choices=list(ACROSS_DIMENSIONS),
        default=None,
        metavar="DIM",
        help=(
            "Compare across a swept dimension by dropping it from the row key "
            "(repeatable). Without this, a thinking=on file and a thinking=off "
            "file share no rows and nothing is compared. Implies an unpaired "
            "test: the dimension is part of the corpus key, so the two sides "
            "read different text."
        ),
    )
    ap.add_argument(
        "--equivalence-margin",
        type=float,
        default=None,
        metavar="FRAC",
        help="Test for equivalence within +/-FRAC of the baseline (e.g. 0.03 for "
             "+/-3%%) using TOST. Turns 'no difference' into either EQUIVALENT -- the "
             "difference is demonstrably inside the margin -- or INCONCLUSIVE, which "
             "means the run was too noisy to tell.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the verdicts as JSON instead of a table, for CI to act on",
    )
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any shape is significantly WORSE on --metric, "
             "turning an A/B report into a deploy gate",
    )
    args = ap.parse_args()
    if len(args.files) < 2:
        ap.error("need at least two result files to compare")
    try:
        return compare(
            args.files,
            args.metric,
            args.alpha,
            args.unpaired,
            equivalence_margin=args.equivalence_margin,
            as_json=args.as_json,
            fail_on_regression=args.fail_on_regression,
            across=tuple(args.across or ()),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        # ValueError covers load()'s refusal to collapse two rows of one file
        # onto a single key when --across drops the dimension it swept.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
