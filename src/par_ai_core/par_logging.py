"""
Logging handler for Par AI Core using rich.

This module provides a logger (``log``) named ``"par_ai"`` plus console
instances. As a library, it does **not** reconfigure the host application's
root logger or replace ``sys.excepthook`` at import time — those are global
side effects that silently override a consumer's own logging/traceback setup.

To opt in to the rich console handler, rich tracebacks, and the warning
filters (the behavior this module applied at import prior to v0.5.9), call
``init_logging()`` once from your application entrypoint:

    from par_ai_core.par_logging import log, init_logging

    init_logging()  # optional: install rich handler + tracebacks on "par_ai"
    log.info("Info message")

Features:
- ``log``: a ``logging.getLogger("par_ai")`` logger with a ``NullHandler``
  attached and ``propagate = False`` (library best practice — never emits
  unless configured, and never bubbles to the root logger).
- ``init_logging(level=None)``: opt-in rich handler + rich tracebacks +
  warning suppression. Reads ``PARAI_LOG_LEVEL`` at call time (not import).
- Separate console instances for standard output and error streams.

Environment Variables:
    PARAI_LOG_LEVEL: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                     when ``init_logging()`` is called without an explicit level.
                     Defaults to ERROR if not set or invalid.
"""

from __future__ import annotations

import logging
import os
import warnings

from rich.console import Console
from rich.logging import RichHandler

console_out = Console()
console_err = Console(stderr=True)

# Library best practice: attach a NullHandler so that, if the consuming
# application has not configured logging, nothing is emitted and we never
# raise "No handlers could be found" warnings. propagate=False ensures we
# never bubble par_ai records up to the root logger the host owns.
log = logging.getLogger("par_ai")
log.addHandler(logging.NullHandler())
log.propagate = False

# Map string log levels to logging constants
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def init_logging(level: str | int | None = None) -> None:
    """Install the rich console handler and rich tracebacks on the ``"par_ai"`` logger.

    This is the explicit, opt-in replacement for the import-time global
    configuration this module performed prior to v0.5.9 (it used to call
    ``logging.basicConfig(...)`` on the root logger and
    ``rich.traceback.install(...)`` on ``sys.excepthook`` at import). Call this
    once from your application entrypoint if you want the rich-formatted
    output and rich tracebacks. A library must not mutate global logging or
    excepthook state, so this is now opt-in only.

    Args:
        level: Logging level as a string (e.g. ``"DEBUG"``), an ``int``
            constant, or ``None`` to read ``PARAI_LOG_LEVEL`` from the
            environment at call time (defaulting to ``ERROR``).
    """
    # Resolve the effective level (read at call time, not import time).
    if level is None:
        level = os.environ.get("PARAI_LOG_LEVEL", "ERROR").upper()
    if isinstance(level, str):
        level = LOG_LEVEL_MAP.get(level.upper(), logging.ERROR)
    log.setLevel(level)

    # Attach the rich handler to the par_ai logger only (not the root logger).
    rich_handler = RichHandler(
        rich_tracebacks=True,
        console=console_err,
        markup=True,
        tracebacks_max_frames=10,
    )
    rich_handler.setLevel(level)
    # Replace any previously-installed non-Null handler so repeated calls stay clean.
    log.handlers = [h for h in log.handlers if isinstance(h, logging.NullHandler)]
    log.addHandler(rich_handler)
    log.propagate = False

    # Install rich tracebacks on sys.excepthook (process-wide, opt-in only).
    from rich.traceback import install

    install(max_frames=10, show_locals=True, console=console_err)

    # Suppress noisy langchain/deprecation warnings originating from par_ai_core.
    try:
        from langchain_core._api import LangChainBetaWarning

        warnings.filterwarnings("ignore", category=LangChainBetaWarning, module=r"par_ai_core\..*")
    except Exception:
        pass
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"par_ai_core\..*")
