"""Tests for par_logging module."""

import logging

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
    """Test init_logging resolves the level from PARAI_LOG_LEVEL at call time.

    Prior to v0.5.9 this module configured the root logger at import time via
    ``logging.basicConfig``. That global side effect was removed (ARC-002);
    level resolution now happens inside the opt-in ``init_logging()`` instead.
    """
    import sys

    from par_ai_core.par_logging import init_logging

    if env_level is None:
        monkeypatch.delenv("PARAI_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("PARAI_LOG_LEVEL", env_level)

    saved_level = log.level
    saved_handlers = list(log.handlers)
    saved_excepthook = sys.excepthook
    try:
        init_logging()  # reads PARAI_LOG_LEVEL at call time
        assert log.level == expected_level
    finally:
        log.setLevel(saved_level)
        log.handlers = saved_handlers
        sys.excepthook = saved_excepthook


def test_init_logging_does_not_touch_root_logger(monkeypatch):
    """init_logging must configure only the par_ai logger, never the root logger."""
    import sys

    from par_ai_core.par_logging import init_logging

    monkeypatch.setenv("PARAI_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    saved_level = log.level
    saved_handlers = list(log.handlers)
    saved_excepthook = sys.excepthook
    try:
        init_logging("INFO")
        # Root logger handlers are unchanged.
        assert root.handlers == saved_root_handlers
        # The par_ai logger has the rich handler and does not propagate.
        assert log.level == logging.INFO
        assert log.propagate is False
        assert any(isinstance(h, logging.Handler) and not isinstance(h, logging.NullHandler) for h in log.handlers)
    finally:
        log.setLevel(saved_level)
        log.handlers = saved_handlers
        log.propagate = False
        sys.excepthook = saved_excepthook


def test_import_does_not_reconfigure_root_logger_or_excepthook():
    """Importing par_ai_core must not mutate the root logger or sys.excepthook."""
    import importlib
    import sys

    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    saved_excepthook = sys.excepthook
    try:
        importlib.reload(__import__("par_ai_core.par_logging", fromlist=["log"]))
        assert logging.getLogger().handlers == saved_root_handlers
        assert sys.excepthook is saved_excepthook
    finally:
        sys.excepthook = saved_excepthook


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
