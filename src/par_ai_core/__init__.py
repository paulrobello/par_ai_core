"""
Par AI Core.
This package provides a simple interface for interacting with various LLM providers.
Created by Paul Robello probello@gmail.com.
"""

from __future__ import annotations

import os

# NOTE: This library intentionally performs NO global side effects at import
# time. It does not reconfigure the root logger, replace sys.excepthook, or
# install global warnings filters. Consumers that want the rich logging
# handler, rich tracebacks, and par_ai_core warning suppression should call
# ``init_logging()`` from par_ai_core.par_logging (see ARC-002 / QA-013).


__author__ = "Paul Robello"
__credits__ = ["Paul Robello"]
__maintainer__ = "Paul Robello"
__email__ = "probello@gmail.com"
__version__ = "0.5.9"
__application_title__ = "Par AI Core"
__application_binary__ = "par_ai_core"
__licence__ = "MIT"


def apply_nest_asyncio() -> bool:
    """Apply nest_asyncio to allow nested event loops.

    Call this explicitly if you need nested asyncio support (e.g., calling
    sync wrappers from within an already-running async event loop).

    Returns:
        True if nest_asyncio was applied successfully, False otherwise.
    """
    try:
        import asyncio

        import nest_asyncio

        try:
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__

            if "uvloop" in loop_type.lower():
                return False

            if hasattr(loop, "_nest_patched"):
                return True

        except RuntimeError:
            pass

        nest_asyncio.apply()
        return True

    except Exception:
        return False


def configure_user_agent(user_agent: str | None = None) -> None:
    """Set the USER_AGENT environment variable.

    Args:
        user_agent: Custom user agent string. If None, uses default
            "{application_title} {version}" format.
    """
    os.environ["USER_AGENT"] = user_agent or f"{__application_title__} {__version__}"


def init_logging(level: str | int | None = None) -> None:
    """Install the rich console handler, rich tracebacks, and warning filters.

    Thin re-export of :func:`par_ai_core.par_logging.init_logging`. Call once
    from your application entrypoint if you want rich-formatted logs and
    tracebacks. This is opt-in; importing ``par_ai_core`` does not configure
    logging globally.

    Args:
        level: Logging level (string, int, or None to read ``PARAI_LOG_LEVEL``).
    """
    from par_ai_core.par_logging import init_logging as _init_logging

    _init_logging(level)


__all__: list[str] = [
    "__author__",
    "__credits__",
    "__maintainer__",
    "__email__",
    "__version__",
    "__application_binary__",
    "__licence__",
    "__application_title__",
    "apply_nest_asyncio",
    "configure_user_agent",
    "init_logging",
]
