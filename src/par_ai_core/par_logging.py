"""
Logging handler for Par AI Core using rich.

This module sets up a customized logging configuration using the rich library,
providing enhanced console output with color-coded log levels, rich tracebacks,
and configurable log levels via environment variables.

Features:
- Uses rich for prettier console output
- Configurable log level via PARAI_LOG_LEVEL environment variable
- Rich tracebacks for better error visualization
- Separate console instances for standard output and error streams

Usage:
    from par_ai_core.par_logging import log

    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")

Environment Variables:
    PARAI_LOG_LEVEL: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                     Defaults to ERROR if not set or invalid.
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

console_out = Console()
console_err = Console(stderr=True)

install(max_frames=10, show_locals=True, console=console_err)

# Map string log levels to logging constants
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Get and validate log level from environment
log_level_str = os.environ.get("PARAI_LOG_LEVEL", "ERROR").upper()
log_level = LOG_LEVEL_MAP.get(log_level_str, logging.ERROR)

logging.basicConfig(
    level=log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console_err, markup=True, tracebacks_max_frames=10)],
)

log = logging.getLogger("par_ai")
