"""
Time Display and Conversion Utilities

This module provides a set of functions for formatting timestamps, converting
datetimes between UTC and local timezones, and retrieving local timezone information.
It is designed to simplify common datetime operations in the par_ai_core project.

Functions:
    format_datetime: Convert a datetime object to a formatted string.
    format_timestamp: Convert a Unix timestamp to a formatted string.
    convert_to_local: Convert a UTC datetime to the local timezone.
    get_local_timezone: Retrieve the local timezone.

These utilities are particularly useful for displaying timestamps in a human-readable
format and handling timezone conversions, which are common requirements in many
parts of the par_ai_core project.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo


def format_datetime(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert a datetime object into a string in human-readable format.

    Args:
        dt: The datetime object.
        fmt: The format string. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: The string datetime in the format specified. Or "Never" if dt is None.
    """
    if dt is None:
        return "Never"
    return dt.strftime(fmt)


def format_timestamp(timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert a Unix timestamp into a string in human-readable format.

    Args:
        timestamp: The Unix timestamp.
        fmt: The format string. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: The string timestamp in the format specified.
    """
    utc_dt = datetime.fromtimestamp(timestamp, timezone.utc)
    local_dt = utc_dt.astimezone()
    return local_dt.strftime(fmt)


def convert_to_local(utc_dt: datetime | str | None) -> datetime | None:
    """Convert a UTC datetime to the local timezone.

    Args:
        utc_dt: The UTC datetime, either as a datetime object or an ISO format string.

    Returns:
        datetime | None: A datetime in the local timezone, or None if input is None or an empty string.
    """
    if utc_dt is None:
        return None
    if isinstance(utc_dt, str):
        if utc_dt == "":
            return None
        utc_dt = datetime.fromisoformat(utc_dt)

    local_dt_now = datetime.now(timezone.utc)
    local_tz = local_dt_now.astimezone().tzinfo
    local_dt = utc_dt.astimezone(local_tz)
    return local_dt


def get_local_timezone() -> tzinfo | None:
    """Get the local timezone.

    Returns:
        tzinfo | None: The local timezone.
    """
    return datetime.now(timezone.utc).astimezone().tzinfo
