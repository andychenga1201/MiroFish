"""Runtime guard for camel-oasis 0.2.5 TWHIN refresh on an empty post table.

Twitter environments use ``recsys_type="twhin-bert"``. ``OasisEnv.step``
refreshes the recommendation table before performing actions, so the first
step (initial posts) runs while ``post`` is empty. In 0.2.5
``Platform.update_rec_table`` then evaluates ``post_table[-1]`` and logs
``list index out of range``. The exception is caught and the refresh is
skipped, which matches the intended empty-platform behavior, but the
IndexError is reported as a social.twitter error.

The pin stays on camel-oasis 0.2.5. The guard is applied from the simulation
scripts so a reinstall does not depend on an untracked site-packages edit.
It only intercepts a TWHIN refresh when ``post`` has no rows; every other
call is forwarded to the installed method. If a future release no longer
indexes ``post_table[-1]``, installation is a no-op.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger("social.twitter")

_TWHIN_RECSYS = "twhin-bert"
_GUARD_ATTR = "_mirofish_empty_post_guard"
_ORIGINAL_ATTR = "_mirofish_original_update_rec_table"


def install_empty_post_table_guard(platform_cls: Any | None = None) -> bool:
    """Patch ``Platform.update_rec_table`` when the installed method is unsafe.

    Returns True when this process installed the guard. Returns False when the
    method is already guarded or does not contain the 0.2.5 empty-table index.
    """
    if platform_cls is None:
        from oasis.social_platform.platform import Platform

        platform_cls = Platform

    current = platform_cls.update_rec_table
    if getattr(current, _GUARD_ATTR, False):
        return False
    if not _indexes_last_post(current):
        return False

    original = current

    async def update_rec_table(self):
        if _is_twhin(getattr(self, "recsys_type", None)) and _post_table_is_empty(self):
            logger.info(
                "Skipping recommendation cache refresh because the post table is empty."
            )
            return None
        return await original(self)

    setattr(update_rec_table, _GUARD_ATTR, True)
    setattr(update_rec_table, _ORIGINAL_ATTR, original)
    platform_cls.update_rec_table = update_rec_table
    return True


def _indexes_last_post(method: Callable[..., Any]) -> bool:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    return "post_table[-1]" in source


def _is_twhin(recsys_type: Any) -> bool:
    value = getattr(recsys_type, "value", recsys_type)
    return value == _TWHIN_RECSYS


def _post_table_is_empty(platform: Any) -> bool:
    cursor = getattr(platform, "db_cursor", None)
    if cursor is None:
        return False
    try:
        cursor.execute("SELECT 1 FROM post LIMIT 1")
        return cursor.fetchone() is None
    except Exception:
        # A missing table or a closed cursor is not the empty-table case.
        # Let the original refresh report its own error.
        return False
