"""Test package initialization."""

import warnings

from langchain_core._api import LangChainBetaWarning

warnings.simplefilter("ignore", category=LangChainBetaWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
