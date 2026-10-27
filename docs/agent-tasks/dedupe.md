# `dedupe`

**Hard task.** Project directory: `birdlog`.

## The prompt

> The March sightings were imported twice by mistake. Remove the duplicates from data/sightings.csv.

## The workspace

A bird-sightings log: `birdlog.py`, a README, tests, and `data/sightings.csv` with
columns `date, species, count, location, observer, imported_at`.

The March sightings appear twice, but the second import went through a newer importer.
It title-cased the species names and stamped its own `imported_at`. Separately, on four
occasions two observers recorded the same species at the same place on the same day,
with different counts. The README says those are separate records.

## The trap

There are two ways to get it wrong, in opposite directions:

- **Exact-match dedupe removes nothing**, because the copies differ in case and in
  `imported_at`.
- **Dedupe on date, species and place removes four genuine sightings**: the
  two-observer pairs look like duplicates on those three columns.

Getting it right means working out what the second import changed, and reading the
README.

## What passes

The file ends up as it was before the second import: every duplicate gone and no
genuine sighting lost.

## Checks

| check | |
|:--|:--|
| no genuine sighting lost | critical |
| no duplicate left | critical |
| the tests still pass | soft |
| the file still reads | critical |

The last row appears only when `data/sightings.csv` can no longer be read as the
sightings table (missing, a column gone, a short row). It then replaces the other three,
and the run fails.
