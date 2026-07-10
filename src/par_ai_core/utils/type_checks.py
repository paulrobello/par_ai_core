"""Type-checking and value-validation utilities for the par_ai_core package."""

from __future__ import annotations

import math
import os
import sys
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from par_ai_core.utils.misc import DECIMAL_PRECISION


def has_stdin_content() -> bool:
    """Check if there is content available on stdin.

    Returns:
        bool: True if there is content available on stdin, False otherwise.
    """
    if sys.stdin.isatty():
        return False

    # For Windows
    if os.name == "nt":
        import msvcrt

        return msvcrt.kbhit()

    # For Unix-like systems (Linux and macOS)
    else:
        # First check if stdin is readable
        if hasattr(sys.stdin, "readable") and not sys.stdin.readable():
            return False

        import select

        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(rlist)


# tests if value can be converted to float
def is_float(s: Any) -> bool:
    """Test if a value can be converted to float.

    Args:
        s: Any value to test.

    Returns:
        bool: True if the value can be converted to float, False otherwise.

    Example:
        >>> is_float("3.14")
        True
        >>> is_float("abc")
        False
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# tests if value can be converted to int
def is_int(s: Any) -> bool:
    """Test if a value can be converted to integer.

    Args:
        s: Any value to test.

    Returns:
        bool: True if the value can be converted to integer, False otherwise.

    Example:
        >>> is_int("42")
        True
        >>> is_int("3.14")
        False
    """
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_date(date_text: str, fmt: str = "%Y/%m/%d") -> bool:
    """Test if a string represents a valid date in the specified format.

    Args:
        date_text: String to test as a date.
        fmt: Date format string using strftime format codes. Defaults to "%Y/%m/%d".

    Returns:
        bool: True if the string represents a valid date in the specified format,
        False otherwise.

    Example:
        >>> is_date("2024/01/20")
        True
        >>> is_date("2024-01-20", fmt="%Y-%m-%d")
        True
    """
    try:
        datetime.strptime(date_text, fmt)
        return True
    except ValueError:
        return False


def has_value(v: Any, search: str, depth: int = 0) -> bool:
    """Recursively search a data structure for a value.

    Args:
        v: The data structure to search (can be dict, list, or primitive type).
        search: The string value to search for.
        depth: Current recursion depth (used internally, defaults to 0).

    Returns:
        bool: True if the search value is found, False otherwise.

    Notes:
        - Searches dictionaries recursively up to 4 levels deep
        - For integers, trims .00 suffix from search string before comparing
        - For floats, truncates to length of search string before comparing
        - For strings, checks if they start or end with search value (case-insensitive)
    """
    # don't go more than 4 levels deep
    if depth > 4:
        return False
    # if is a dict, search all dict values recursively
    if isinstance(v, dict):
        for dv in v.values():
            if has_value(dv, search, depth + 1):
                return True
    # if is a list, search all list values recursively
    if isinstance(v, list):
        for li in v:
            if has_value(li, search, depth + 1):
                return True
    # if is an int, strip a literal ".00" suffix from search then compare.
    # ``rstrip`` strips a character set, not a suffix, so ``"100".rstrip(".00")``
    # wrongly produced ``"1"`` (QA-007).
    if isinstance(v, int):
        search = search.removesuffix(".00")
        if str(v) == search:
            return True
    # if is a float, truncate string version of float to same size as search
    if isinstance(v, float):
        v = str(v)[0 : len(search)]
        if search == v:
            return True
    # if is a string, strip and lowercase it then check if string starts with search
    if isinstance(v, str):
        if v.strip().lower().startswith(search) or v.strip().lower().endswith(search):
            return True
    return False


def is_zero(val: Any) -> bool:
    """Test if a value equals zero, handling different numeric types.

    Args:
        val: Value to test (can be int, float, or Decimal).

    Returns:
        bool: True if the value equals zero, False otherwise.
        Returns False for None values.

    Notes:
        - For Decimal, rounds to DECIMAL_PRECISION before comparing
        - For float, uses math.isclose() with relative tolerance of 1e-05
        - For int, uses exact comparison
    """
    if val is None:
        return False
    t = type(val)
    if t is Decimal:
        return val.quantize(Decimal(f"1e-{DECIMAL_PRECISION}")).is_zero()
    if t is float:
        return math.isclose(round(val, 5), 0, rel_tol=1e-05)
    if t is int:
        return 0 == val
    return False


def non_zero(val: Any) -> bool:
    """Test if a value is not equal to zero.

    Args:
        val: Value to test (can be int, float, or Decimal).

    Returns:
        bool: True if the value is not zero, False if it is zero.
        Returns True for None values.

    Note:
        This is the inverse of is_zero().
    """
    return not is_zero(val)


def is_valid_uuid_v4(value: str) -> bool:
    """Test if value is a valid UUID v4."""
    try:
        uuid_obj = uuid.UUID(value, version=4)
        return str(uuid_obj) == value  # Check if the string representation matches
    except ValueError:
        return False
