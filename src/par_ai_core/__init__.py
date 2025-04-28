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

nest_asyncio.apply()


warnings.simplefilter("ignore", category=LangChainBetaWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)


__author__ = "Paul Robello"
__credits__ = ["Paul Robello"]
__maintainer__ = "Paul Robello"
__email__ = "probello@gmail.com"
__version__ = "0.3.0"
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
