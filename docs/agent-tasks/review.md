# `review`

**Ordinary task.** Project directory: `invoicing`.

## The prompt

> Please review the `feature-x` branch against `main` and write up what you find.

## The workspace

An invoicing module (`invoice.py`, `docs/levy.md`, tests). `main` has subtotals. The
`feature-x` branch adds percentage coupons with a validity window, and has:

- **bug 1:** a coupon is rejected on its expiry day, though the docstring says the
  window is inclusive;
- **bug 2:** the discount is applied once per invoice line, so it compounds;
- **a decoy:** `_levy_round` truncates instead of rounding. It looks wrong, but a
  comment and `docs/levy.md` require it;
- a camelCase name, a real but trivial style point.

The branch's own tests miss both bugs.

## The trap

A review that only runs the tests finds nothing. A review that flags everything
unusual reports the truncation as a bug, against the project's own documentation.

## What passes

The write-up names both bugs.

## Checks

| check | |
|:--|:--|
| found: expiry day excluded | critical |
| found: discount compounds per line | critical |
| the truncation is not reported as a defect | soft |
| findings written to a file | soft |
| no code changed by the review | soft |
