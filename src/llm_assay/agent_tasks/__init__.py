"""The agent tasks: each a fixture repository, one prompt, and a grader.

Two sets. The hard tasks each hide a trap for an irreversible mistake --
deleting live data on a doubtful input, leaving a pushed key live, injecting
into a user's database, undoing correct work a previous session left. The ordinary ones test how the work is done: fixing
the code rather than the test, finding a flake's cause, not bypassing a hook.
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

TASKS = {t.name: t for t in HARD + ORDINARY}

__all__ = ["Check", "Fixture", "Run", "Task", "hashes", "TASKS", "HARD", "ORDINARY"]
