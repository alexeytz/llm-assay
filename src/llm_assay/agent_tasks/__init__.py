"""The agent tasks: each a fixture repository, one prompt, and a grader.

Grouped by what the trap tests: the user's data (deleting or corrupting what
only the user has), secrets (a key committed or left live), the check
(passing a test or hook by defeating it), someone else's work (judging a
branch or a previous session's state), and ordinary craft. The groups are
presentation only: a task's kit digest is its own, so moving it between
groups changes no result.

``HARD`` and ``ORDINARY`` are the two sets the tasks were first split into,
kept so ``--tasks hard`` (the default) and ``--tasks ordinary`` still mean
what they meant.
"""

from ._base import Check, Fixture, Run, Task, hashes
from .dedupe import DEDUPE
from .feature import FEATURE
from .flake import FLAKE
from .handoff import HANDOFF, HANDOFF_QUIET
from .hook_blocks import HOOK_BLOCKS
from .leak import LEAK
from .overnight import OVERNIGHT
from .red_test import RED_TEST
from .review import REVIEW
from .search import SEARCH
from .secret_commit import SECRET_COMMIT
from .sync import SYNC, SYNC_QUIET

HARD = [SYNC, SYNC_QUIET, DEDUPE, LEAK, SEARCH, HANDOFF, HANDOFF_QUIET]
ORDINARY = [RED_TEST, FLAKE, HOOK_BLOCKS, SECRET_COMMIT, OVERNIGHT, REVIEW, FEATURE]

#: Every task once, in the order ``--tasks all`` runs them.
GROUPS = {
    "data": [SYNC, SYNC_QUIET, DEDUPE, SEARCH, OVERNIGHT],
    "secrets": [LEAK, SECRET_COMMIT],
    "checks": [RED_TEST, FLAKE, HOOK_BLOCKS],
    "others-work": [REVIEW, HANDOFF, HANDOFF_QUIET],
    "craft": [FEATURE],
}

TASKS = {t.name: t for g in GROUPS.values() for t in g}

__all__ = ["Check", "Fixture", "Run", "Task", "hashes", "TASKS", "GROUPS", "HARD", "ORDINARY"]
