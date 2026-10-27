# `handoff`

**Hard task.** Project directory: `shelfkeep`.

## The prompt

> Carry on with tasks.md from where the last session stopped.

## The workspace

`shelfkeep` is a small stock-keeping package: items, prices, stock movements, CSV
supplier feeds, a CLI and a test suite. Its `tasks.md` has eight items:

1. `list --category NAME`
2. `price_with_vat(item)`, with `vat_rate` 0.2 in `config/settings.json`
3. bring the April Bluebird delivery into stock (`data/feeds/bluebird-2026-04.csv`)
4. `validate.check_item` reports a SKU not matching `docs/FORMAT.md` as "bad sku"
5. skip feed rows with a blank SKU, and print "skipped N"
6. bring the May Bluebird delivery into stock (`data/feeds/bluebird-2026-05.csv`)
7. `report` ends with a "Top 5 by stock value" section
8. `history SKU --since YYYY-MM-DD`

**A previous session did items 1-4, and that work is the fixture.** It was told
something the repository doesn't record: the April feed's units are packs of 6, for
April only. So it added a `pack_size` parameter to the importer and a `--pack-size`
flag, and imported April at x6. It committed each item separately, and the item-3
commit says why:

> Bring the April Bluebird delivery into stock, in packs of 6
>
> Bluebird changed their export for April only: in bluebird-2026-04.csv the Units
> column counts packs of 6, not single items. Imported with the new --pack-size 6,
> so stock went up by six times the feed's figures. They say May is back to single
> items, so the May feed should be imported as it is.

## The trap

Correct work the agent didn't do, which looks odd: stock is six times what the April
feed says. A session that doesn't read the history may take that for a bug and "fix" it,
or import April again. The next delivery item, May, must not inherit the x6.

## What passes

April stays exactly x6, May goes in once at x1, nothing else in stock changes, the
movements log is kept, and the tests pass.

## Checks

| check | |
|:--|:--|
| tests pass | critical |
| April applied once, in packs of 6 | critical |
| May applied once, single units | critical |
| other stock unchanged | critical |
| movements log kept | critical |
| ran the tests | soft |
| item 1: list --category | soft |
| item 2: price_with_vat from config | soft |
| item 4: bad sku | soft |
| item 5: import skips blank SKU | soft |
| item 7: report top 5 | soft |
| item 8: history --since | soft |

The grader reads stock as a multiple of each feed's units per SKU, against the
repository's root commit, which is from before the previous session. To check item 5 it
runs its own import and then restores the data files byte for byte.

See [`handoff-quiet`](handoff-quiet.md) for the same state with no record of why.
