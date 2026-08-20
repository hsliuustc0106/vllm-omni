# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exception reporting helpers that never disturb lifecycle control flow."""

from __future__ import annotations

import os
import sys


def _safe_exception_type_name(exc: BaseException) -> str:
    try:
        name = type.__getattribute__(type(exc), "__name__")
    except BaseException:
        return "BaseException"
    return name if isinstance(name, str) else "BaseException"


def safe_exception_detail(exc: BaseException) -> str:
    """Return best-effort exception text without trusting ``exc.__str__``."""
    try:
        return str(exc)
    except BaseException:
        return f"<{_safe_exception_type_name(exc)} detail unavailable>"


def safe_exception_summary(exc: BaseException) -> str:
    """Return a stable type-and-detail summary for diagnostics."""
    return f"{_safe_exception_type_name(exc)}: {safe_exception_detail(exc)}"


def safe_add_exception_note(
    primary: BaseException,
    context: str,
    secondary: BaseException,
) -> None:
    """Best-effort annotation that can never replace ``primary``.

    Calling the builtin descriptor directly avoids a hostile override of
    ``add_note`` on an exception subclass.  Annotation remains diagnostic: an
    allocation failure or any other note failure is deliberately ignored.
    """
    try:
        BaseException.add_note(primary, f"{context}: {safe_exception_summary(secondary)}")
    except BaseException:
        pass


def safe_close_fd(fd: int, context: str) -> None:
    """Close ``fd`` without masking an exception active in a ``finally``."""
    primary = sys.exception()
    try:
        os.close(fd)
    except BaseException as close_exc:
        if primary is None:
            raise
        safe_add_exception_note(primary, context, close_exc)


__all__ = [
    "safe_add_exception_note",
    "safe_close_fd",
    "safe_exception_detail",
    "safe_exception_summary",
]
