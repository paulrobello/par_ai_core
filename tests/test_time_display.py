"""Tests for the time_display module."""

from datetime import UTC, datetime

from par_ai_core.time_display import (
    convert_to_local,
    format_datetime,
    format_timestamp,
    get_local_timezone,
)


def test_format_datetime_none():
    """Test format_datetime with None input."""
    assert format_datetime(None) == "Never"


def test_format_datetime_with_date():
    """Test format_datetime with a datetime object."""
    test_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert format_datetime(test_dt) == "2024-01-01 12:00:00"


def test_format_datetime_custom_format():
    """Test format_datetime with custom format string."""
    test_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert format_datetime(test_dt, "%Y/%m/%d") == "2024/01/01"


def test_format_timestamp():
    """Test format_timestamp with a known timestamp."""
    # Using a known timestamp (2024-01-01 12:00:00 UTC)
    timestamp = 1704110400.0
    expected = datetime.fromtimestamp(timestamp, UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert format_timestamp(timestamp) == expected


def test_format_timestamp_custom_format():
    """Test format_timestamp with custom format string."""
    timestamp = 1704110400.0  # 2024-01-01 12:00:00 UTC
    expected = datetime.fromtimestamp(timestamp, UTC).astimezone().strftime("%Y/%m/%d")
    assert format_timestamp(timestamp, "%Y/%m/%d") == expected


def test_convert_to_local_none():
    """Test convert_to_local with None input."""
    assert convert_to_local(None) is None


def test_convert_to_local_empty_string():
    """Test convert_to_local with empty string input."""
    assert convert_to_local("") is None


def test_convert_to_local_datetime():
    """Test convert_to_local with datetime object."""
    utc_dt = datetime.now(UTC)
    local_dt = convert_to_local(utc_dt)

    assert local_dt is not None
    assert local_dt.tzinfo == get_local_timezone()
    # The timestamp should be the same, just in different timezone
    assert utc_dt.timestamp() == local_dt.timestamp()


def test_convert_to_local_string():
    """Test convert_to_local with ISO format string."""
    utc_dt = datetime.now(UTC)
    iso_str = utc_dt.isoformat()
    local_dt = convert_to_local(iso_str)

    assert local_dt is not None
    assert local_dt.tzinfo == get_local_timezone()
    assert utc_dt.timestamp() == local_dt.timestamp()


def test_get_local_timezone():
    """Test get_local_timezone returns a valid timezone."""
    local_tz = get_local_timezone()
    assert local_tz is not None

    # Verify the timezone works for actual conversions
    utc_now = datetime.now(UTC)
    local_now = utc_now.astimezone(local_tz)
    assert local_now.tzinfo == local_tz
