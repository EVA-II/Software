"""Runtime compatibility helpers for legacy Python environments."""

from __future__ import annotations

import sys
import typing


def patch_typing_extensions_self() -> None:
    """Allow newer torch-geometric builds to import under Python 3.9/3.10.

    Some installed builds annotate types with ``typing_extensions.Self`` in places
    where Python 3.9's ``typing`` module rejects it during runtime evaluation.
    For this project we only need those annotations for import-time compatibility,
    so replacing them with ``typing.Any`` is sufficient.
    """
    if sys.version_info >= (3, 11):
        return

    if not hasattr(typing, "Self"):
        typing.Self = typing.Any  # type: ignore[attr-defined]

    try:
        import typing_extensions
    except Exception:
        return

    typing_extensions.Self = typing.Any
