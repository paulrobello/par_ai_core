"""Tests for par_logging module."""

import logging
from unittest.mock import patch

import pytest
from rich.console import Console

from par_ai_core.par_logging import LOG_LEVEL_MAP, console_err, console_out, log


def test_console_initialization():
    """Test console objects are properly initialized."""
    assert isinstance(console_out, Console)
    assert isinstance(console_err, Console)
    import sys

    assert console_err.file is sys.stderr  # Verify stderr is used


def test_log_level_mapping():
    """Test log level string to constant mapping."""
    assert LOG_LEVEL_MAP["DEBUG"] == logging.DEBUG
    assert LOG_LEVEL_MAP["INFO"] == logging.INFO
    assert LOG_LEVEL_MAP["WARNING"] == logging.WARNING
    assert LOG_LEVEL_MAP["ERROR"] == logging.ERROR
    assert LOG_LEVEL_MAP["CRITICAL"] == logging.CRITICAL


@pytest.mark.parametrize(
    "env_level,expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("INVALID", logging.ERROR),  # Default to ERROR for invalid levels
        (None, logging.ERROR),  # Default to ERROR when not set
    ],
)
def test_log_level_configuration(monkeypatch, env_level, expected_level):
    """Test log level is properly configured from environment variable."""
    if env_level is None:
        monkeypatch.delenv("PARAI_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("PARAI_LOG_LEVEL", env_level)

    # Mock basicConfig before importing
    with patch("logging.basicConfig") as mock_basic_config:
        # Import the module to trigger configuration
        import importlib

        import par_ai_core.par_logging

        importlib.reload(par_ai_core.par_logging)

        # Verify basicConfig was called with expected level
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args[1]
        assert call_args["level"] == expected_level


def test_logger_name():
    """Test logger is created with correct name."""
    assert log.name == "par_ai"


@pytest.mark.parametrize(
    "log_method,log_level",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_logging_methods(log_method, log_level, caplog):
    """Test all logging methods work correctly."""
    # Configure logging and capture
    caplog.set_level(log_level)
    logger = logging.getLogger("par_ai")
    logger.setLevel(log_level)

    # Send test message
    test_message = f"Test {log_method} message"
    getattr(logger, log_method)(test_message)

    # Verify log record
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == log_level
    assert test_message in caplog.records[0].message
