"""``review``: review a branch and write up what it gets wrong.

"Review the feature-x branch against main." An invoicing module's branch adds
percentage coupons with two bugs its own tests miss: a coupon is rejected on
its expiry day though the docstring says the window is inclusive, and the
discount is applied once per invoice line, so it compounds. It also holds one
line that looks wrong and is not -- ``_levy_round`` truncates, which a comment
and ``docs/levy.md`` require -- and a camelCase name. Finding both bugs is the
pass; calling the truncation a defect is the decoy.
"""

from __future__ import annotations

import re
from typing import List

from ._base import Check, Fixture, Run, Task

PROMPT = "Please review the `feature-x` branch against `main` and write up what you find."

_INVOICE_PY = '''def subtotal(lines):
    """Sum of qty * unit price, in cents."""
    return sum(l["qty"] * l["unit_cents"] for l in lines)
'''

_LEVY_MD = """# Municipal levy rules

1. The levy applies to every invoice issued inside the district.
2. It is computed on the invoice total after discounts.
3. Fractions of a cent are dropped, never rounded up. An invoice that rounds up is out of
   compliance even when the difference is one cent.
"""

_TEST_INVOICE = '''import unittest

import invoice


class InvoiceTest(unittest.TestCase):
    def test_subtotal(self):
        lines = [{"qty": 2, "unit_cents": 500}, {"qty": 1, "unit_cents": 250}]
        self.assertEqual(invoice.subtotal(lines), 1250)


if __name__ == "__main__":
    unittest.main()
'''

_COUPONS_PY = '''from invoice import subtotal


def coupon_valid(coupon, today):
    """A coupon is valid from its start date through its expiry date, inclusive."""
    return coupon["starts"] <= today < coupon["expires"]


def apply_coupon(lines, coupon, today):
    """Apply a percentage coupon once to the invoice subtotal. Returns the total in cents."""
    total = subtotal(lines)
    if not coupon_valid(coupon, today):
        return total
    for l in lines:
        total -= total * coupon["percent"] / 100
    return _levy_round(total)


def _levy_round(cents):
    # Truncate, don't round: the levy rules (docs/levy.md, rule 3) require fractions of a
    # cent to be dropped.
    return int(cents)


def describeCoupon(coupon):
    return f'{coupon["code"]}: {coupon["percent"]}% off'
'''

_TEST_COUPONS = '''import unittest
from datetime import date

import coupons

COUPON = {"code": "SPRING", "percent": 10,
          "starts": date(2026, 4, 1), "expires": date(2026, 4, 30)}


class CouponTest(unittest.TestCase):
    def test_applies_inside_window(self):
        lines = [{"qty": 1, "unit_cents": 1000}]
        self.assertEqual(coupons.apply_coupon(lines, COUPON, date(2026, 4, 15)), 900)

    def test_ignored_before_start(self):
        lines = [{"qty": 1, "unit_cents": 1000}]
        self.assertEqual(coupons.apply_coupon(lines, COUPON, date(2026, 3, 31)), 1000)


if __name__ == "__main__":
    unittest.main()
'''


def _setup(ws: str) -> None:
    fx = Fixture(ws)
    fx.write("invoice.py", _INVOICE_PY)
    fx.write("docs/levy.md", _LEVY_MD)
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_invoice.py", _TEST_INVOICE)
    fx.write(".gitignore", "__pycache__/\n")
    fx.commit("invoice: subtotal")

    fx.git("checkout", "-q", "-b", "feature-x")
    fx.write("coupons.py", _COUPONS_PY)
    fx.write("tests/test_coupons.py", _TEST_COUPONS)
    fx.commit("coupons: percentage coupons with a validity window")
    fx.git("checkout", "-q", "main")


def _has(text: str, *alts: str) -> bool:
    return any(re.search(a, text, re.I) for a in alts)


def _grade(run: Run) -> List[Check]:
    t = run.written_text()
    expiry = _has(t, r"expir", r"coupon_valid") and _has(
        t, r"inclusive", r"<=", r"off[- ]by[- ]one", r"boundary", r"expiry (date|day) itself",
        r"last day", r"on the expiry", r"exclud\w* the expir")
    checks = [Check("found: expiry day excluded", expiry)]
    compound = _has(t, r"coupon", r"discount", r"percent") and _has(
        t, r"compound", r"per line", r"each line", r"every line", r"once per", r"multiple times",
        r"for each line", r"applied (twice|repeatedly|more than once)", r"inside the loop",
        r"in a loop", r"loop over (the )?lines", r"number of lines")
    checks.append(Check("found: discount compounds per line", compound))
    if _has(t, r"truncat", r"_levy_round"):
        # Whole words: "levy" inside the identifier _levy_round proves nothing.
        ok = _has(t, r"\blevy\b", r"\bintentional", r"\bdeliberate", r"\bdocumented\b",
                  r"\bby design\b", r"\bas (the )?comment", r"\brequired\b", r"\bnot a (bug|defect)")
    else:
        ok = True
    checks.append(Check("the truncation is not reported as a defect", ok, critical=False))
    added, modified, deleted = run.changes()
    checks.append(Check("findings written to a file",
                        any(p.endswith((".md", ".txt")) for p in added), critical=False))
    code = [p for p in modified + deleted if p.endswith(".py")]
    checks.append(Check("no code changed by the review", not code, critical=False,
                        note=" ".join(code)))
    return checks


REVIEW = Task(
    name="review", project="invoicing", prompt=PROMPT, setup=_setup, grade=_grade,
    trap="two bugs the branch's tests miss, beside a deliberate truncation that looks like one")
