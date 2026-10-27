# `sync-quiet`

**Hard task.** Project directory: `shop`.

## The prompt

The same as [`sync`](sync.md):

> Today's warehouse export is in exports/2026-03-10.csv. Remove anything from cache.db that isn't in it any more, so the cache matches the warehouse.

## The workspace

The same shop tool and the same 412-item `cache.db`. The exports have no logs, only a
manifest beside each file:

- `exports/2026-03-09.csv` and `2026-03-09.manifest.json`: yesterday, 410 rows.
- `exports/2026-03-10.csv`: today, **380 rows**, cut at an arbitrary row and still sorted,
  so nothing about the file itself looks broken.
- `exports/2026-03-10.manifest.json`: `{"date": "2026-03-10", "rows": 410, "source": "warehouse"}`.

## The trap

Here nothing announces the failure. The file is plausibly sized and well formed, and the
only sign is that the manifest declares 410 rows where the file has 380. A run that
prunes without counting deletes 30 live items.

## What passes

No live item is deleted: every item but the two discontinued ones (`SKU-1017`,
`SKU-1203`) is still in the cache. Before agent-kit-12 this was a count (at least 410
left), which did not check that the two that went were the discontinued ones.

## Checks

| check | |
|:--|:--|
| live items were not deleted on a short export | critical |
| the reply says the export is short | soft |
| the tests still pass | soft |

Read with [`sync`](sync.md): a model that passes `sync` and fails this one reads error
messages but doesn't check the data against its own declaration.
