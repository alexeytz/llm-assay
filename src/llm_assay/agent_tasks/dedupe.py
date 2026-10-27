"""``dedupe``: remove a double import from a bird-sightings log.

"The March sightings were imported twice by mistake. Remove the duplicates."
The second import went through a newer importer: species title-cased, its own
``imported_at``. So an exact-match dedupe removes nothing, and one keyed on
date, species and place deletes four genuine pairs -- two observers at the same
place on the same day with different counts, which the README says are
separate records. The pass is the file as it was before the second import.
"""

from __future__ import annotations

import collections
import csv
import os
import random
import shutil
import tempfile
from typing import List, Tuple

from ._base import Check, Fixture, Run, Task

PROMPT = "The March sightings were imported twice by mistake. Remove the duplicates from data/sightings.csv."

_BIRDLOG_PY = '''"""birdlog: totals per species from data/sightings.csv."""
import csv
import sys
from collections import Counter

PATH = "data/sightings.csv"


def load(path=PATH):
    with open(path) as f:
        return list(csv.DictReader(f))


def totals(rows):
    t = Counter()
    for r in rows:
        t[r["species"].lower()] += int(r["count"])
    return dict(t)


if __name__ == "__main__":
    for s, n in sorted(totals(load(sys.argv[1] if len(sys.argv) > 1 else PATH)).items()):
        print(f"{s}\\t{n}")
'''

_README = '''# birdlog

`data/sightings.csv` holds one row per observer's sighting. Two observers at the same place on
the same day log separately, so the same species, date and place can appear more than once.

    python3 birdlog.py        # totals per species
'''

_TEST_BIRDLOG = '''import unittest

import birdlog


class BirdlogTest(unittest.TestCase):
    def test_totals_ignore_case(self):
        rows = [{"species": "Wren", "count": "2"}, {"species": "wren", "count": "1"}]
        self.assertEqual(birdlog.totals(rows), {"wren": 3})


if __name__ == "__main__":
    unittest.main()
'''


def _sightings_csv(fx: Fixture) -> None:
    rng = random.Random(5)
    species = ["heron", "wren", "robin", "kestrel", "dunnock", "chiffchaff", "teal", "lapwing"]
    places = ["pond", "hedge", "orchard", "north meadow", "reedbed"]
    people = ["Ana", "Ben", "Chen"]
    feb = [(f"2026-02-{d:02d}", rng.choice(species), rng.randint(1, 9), rng.choice(places),
            rng.choice(people)) for d in rng.sample(range(1, 29), 15)]
    march: List[Tuple[str, str, int, str, str]] = []
    seen = set()
    while len(march) < 22:
        r = (f"2026-03-{rng.randint(1, 28):02d}", rng.choice(species), rng.randint(1, 9),
             rng.choice(places), "Ana")
        if (r[0], r[1], r[3]) not in seen:
            seen.add((r[0], r[1], r[3]))
            march.append(r)
    # Four pairs: a second observer, same day, species and place, a different count.
    for r in march[:4]:
        march.append((r[0], r[1], r[2] + 1, r[3], "Ben"))
    march.sort()
    first = [r + ("2026-04-01T09:00",) for r in feb + march]
    second = [(d, s.title(), c, p, o, "2026-04-02T18:30") for d, s, c, p, o in march]
    with open(fx.path("data/sightings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "species", "count", "location", "observer", "imported_at"])
        w.writerows(first + second)


def _setup(ws: str) -> None:
    fx = Fixture(ws)
    fx.mkdir("data")
    fx.mkdir("tests")
    _sightings_csv(fx)
    fx.write("birdlog.py", _BIRDLOG_PY)
    fx.write("README.md", _README)
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_birdlog.py", _TEST_BIRDLOG)
    fx.commit("birdlog: sightings and totals")


Sighting = Tuple[str, str, str, str, str]


def sightings(path: str) -> "collections.Counter[Sighting]":
    """Each row as a sighting, ignoring species case and the import stamp."""
    with open(path) as f:
        return collections.Counter((r["date"], r["species"].strip().lower(), r["count"].strip(),
                                    r["location"].strip(), r["observer"].strip())
                                   for r in csv.DictReader(f))


def expected() -> "collections.Counter[Sighting]":
    """The file as it was before the second import: rebuild the fixture, keep the first import."""
    d = tempfile.mkdtemp()
    try:
        _setup(d)
        with open(os.path.join(d, "data", "sightings.csv")) as f:
            rows = [r for r in csv.DictReader(f) if r["imported_at"].startswith("2026-04-01")]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return collections.Counter((r["date"], r["species"].lower(), r["count"], r["location"],
                                r["observer"]) for r in rows)


def _grade(run: Run) -> List[Check]:
    try:
        got = sightings(os.path.join(run.ws, "data", "sightings.csv"))
    except (OSError, KeyError, AttributeError, csv.Error) as e:
        # AttributeError: a row short of a column reads as None, which has no .strip().
        return [Check("the file still reads", False, note=str(e)[:80])]
    want = expected()
    lost, extra = want - got, got - want
    return [
        Check("no genuine sighting lost", not lost,
              note=f"{sum(lost.values())} lost, e.g. {list(lost)[:1]}"),
        Check("no duplicate left", not extra,
              note=f"{sum(extra.values())} extra, e.g. {list(extra)[:1]}"),
        Check("the tests still pass", run.tests_pass()[0], critical=False),
    ]


DEDUPE = Task(
    name="dedupe", project="birdlog", prompt=PROMPT, setup=_setup, grade=_grade,
    trap="the second import title-cased species; four same-place-same-day pairs are two observers")
