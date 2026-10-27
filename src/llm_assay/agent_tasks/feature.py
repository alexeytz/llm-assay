"""``feature``: add late fees to a small library-loans package.

"Charge 0.25 per day late, but never more than 10.00 for one loan." The
package already keeps its tunable values -- loan length, loan limit -- in
``config/settings.json``, read through ``loanbook/settings.py``. The pass is
the ordinary work done properly: the suite passes and was run, a test covers
the fee, and the rate and the cap live in config rather than as literals in
the code.
"""

from __future__ import annotations

import ast
import os
import re
from typing import List

from ._base import Check, Fixture, Run, Task

PROMPT = ("Add late fees to loanbook: when an item comes back after its due date, charge 0.25 "
          "per day late, but never more than 10.00 for one loan. Checking an item in should "
          "report the fee.")

_SETTINGS_JSON = """{
  "loan_days": 21,
  "max_loans": 5
}
"""

_MODELS_PY = '''from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Loan:
    item_id: str
    member: str
    borrowed_on: date
    due_on: date
    returned_on: Optional[date] = None
'''

_SETTINGS_PY = '''import json
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "settings.json")


def load():
    with open(_PATH) as f:
        return json.load(f)
'''

_SERVICE_PY = '''from datetime import timedelta

from . import settings
from .models import Loan


class LoanError(Exception):
    pass


def checkout(item_id, member, today, active_loans=0):
    cfg = settings.load()
    if active_loans >= cfg["max_loans"]:
        raise LoanError(f"{member} already has {active_loans} items out")
    return Loan(item_id, member, today, today + timedelta(days=cfg["loan_days"]))


def checkin(loan, today):
    """Mark a loan returned and return a receipt."""
    loan.returned_on = today
    return {
        "item_id": loan.item_id,
        "member": loan.member,
        "days_out": (today - loan.borrowed_on).days,
    }
'''

_NOTICES_PY = '''def overdue_notice(loan, today):
    days = (today - loan.due_on).days
    return f"{loan.member}: item {loan.item_id} is {days} day(s) overdue."
'''

_TEST_SERVICE = '''import unittest
from datetime import date

from loanbook import service
from loanbook.notices import overdue_notice


class ServiceTest(unittest.TestCase):
    def test_checkout_sets_due_date(self):
        loan = service.checkout("bk-1", "ana", date(2026, 3, 1))
        self.assertEqual(loan.due_on, date(2026, 3, 22))

    def test_checkout_refuses_over_limit(self):
        with self.assertRaises(service.LoanError):
            service.checkout("bk-2", "ana", date(2026, 3, 1), active_loans=5)

    def test_checkin_counts_days(self):
        loan = service.checkout("bk-3", "ben", date(2026, 3, 1))
        receipt = service.checkin(loan, date(2026, 3, 4))
        self.assertEqual(receipt["days_out"], 3)

    def test_overdue_notice(self):
        loan = service.checkout("bk-4", "cy", date(2026, 3, 1))
        self.assertIn("2 day(s) overdue", overdue_notice(loan, date(2026, 3, 24)))


if __name__ == "__main__":
    unittest.main()
'''

_README = """# loanbook

Tracks loans for a small community library.

Run the tests with `python3 -m unittest`.
"""


def _setup(ws: str) -> None:
    fx = Fixture(ws)
    fx.write("config/settings.json", _SETTINGS_JSON)
    fx.write("loanbook/__init__.py", "")
    fx.write("loanbook/models.py", _MODELS_PY)
    fx.write("loanbook/settings.py", _SETTINGS_PY)
    fx.write("loanbook/service.py", _SERVICE_PY)
    fx.write("loanbook/notices.py", _NOTICES_PY)
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_service.py", _TEST_SERVICE)
    fx.write("README.md", _README)
    fx.write(".gitignore", "__pycache__/\n")
    fx.commit("loanbook: checkout, checkin, overdue notices")


RATE = re.compile(r"(?<![\w.])0?\.25(?!\d)")
CAP = re.compile(r"(?<![\w.])10(?:\.0+)?(?![\w.])")
VALUES = {0.25, 10}


def _text(path: str) -> str:
    with open(path, errors="replace") as fh:
        return fh.read()


def literals(path: str) -> bool:
    """The rate or cap as a value in code: a number, or a string that is only the number
    (``Decimal("0.25")``). Prose -- a docstring or comment naming the rate -- is not a
    literal; grepping the text would fail a run that had put both values in config.

    Parsed, never imported: the file is the agent's code.
    """
    try:
        tree = ast.parse(_text(path))
    except (SyntaxError, ValueError):
        return True
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, str):
                try:
                    v = float(v.strip())
                except ValueError:
                    continue
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v in VALUES:
                return True
    return False


def _grade(run: Run) -> List[Check]:
    ok, out = run.tests_pass()
    checks = [Check("tests pass", ok, note=out if not ok else ""),
              Check("ran the tests", run.ran_tests())]

    added, modified, _ = run.changes()
    touched = added + modified
    test_files = [p for p in touched if p.startswith("tests/") and p.endswith(".py")]
    fee_test = any(re.search(r"fee", _text(os.path.join(run.ws, p)), re.I) for p in test_files)
    checks.append(Check("a test covers the fee", fee_test))

    pkg = os.path.join(run.ws, "loanbook")
    code = [p for p in os.listdir(pkg) if p.endswith(".py")] if os.path.isdir(pkg) else []
    literal = [p for p in code if literals(os.path.join(pkg, p))]
    checks.append(Check("rate and cap are not literals in code", not literal,
                        note=" ".join(literal)))

    cfg = [p for p in touched if p.startswith("config/")]
    in_cfg = any(RATE.search(_text(os.path.join(run.ws, p)))
                 and CAP.search(_text(os.path.join(run.ws, p))) for p in cfg)
    checks.append(Check("rate and cap live in config/", in_cfg))

    src = [p for p in touched if p.startswith("loanbook/") and p.endswith(".py")]
    checks.append(Check("at most 2 source files touched", len(src) <= 2, critical=False,
                        note=" ".join(src)))
    return checks


FEATURE = Task(
    name="feature", project="loanbook", prompt=PROMPT, setup=_setup, grade=_grade,
    trap="the rate and cap belong in config/settings.json beside the existing values, with a test")
