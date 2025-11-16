"""
Par AI Core.
This package provides a simple interface for interacting with various LLM providers.
Created by Paul Robello probello@gmail.com.
"""

from __future__ import annotations

import os
import warnings

import nest_asyncio
from langchain_core._api import LangChainBetaWarning  # type: ignore


# Apply nest_asyncio only when it's safe to do so
def _apply_nest_asyncio_safely():
    """Apply nest_asyncio only when it's safe to do so."""
    try:
        import asyncio

        # Check if we're in a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__

            # Don't patch uvloop - it doesn't support nest_asyncio
            if "uvloop" in loop_type.lower():
                return False

            # Don't patch if already patched
            if hasattr(loop, "_nest_patched"):
                return True

        except RuntimeError:
            # No running loop - safe to apply
            pass

        # Apply nest_asyncio
        nest_asyncio.apply()
        return True

    except Exception:
        # If anything fails, don't apply
        return False


# Apply nest_asyncio safely
_applied = _apply_nest_asyncio_safely()


warnings.simplefilter("ignore", category=LangChainBetaWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)


__author__ = "Paul Robello"
__credits__ = ["Paul Robello"]
__maintainer__ = "Paul Robello"
__email__ = "probello@gmail.com"
__version__ = "0.5.3"
__application_title__ = "Par AI Core"
__application_binary__ = "par_ai_core"
__licence__ = "MIT"


os.environ["USER_AGENT"] = f"{__application_title__} {__version__}"


__all__: list[str] = [
    "__author__",
    "__credits__",
    "__maintainer__",
    "__email__",
    "__version__",
    "__application_binary__",
    "__licence__",
    "__application_title__",
]
