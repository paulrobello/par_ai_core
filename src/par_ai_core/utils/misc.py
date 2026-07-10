"""Miscellaneous utilities, context managers, and package constants."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console

from par_ai_core.par_logging import console_err

DECIMAL_PRECISION = 5
# Backward-compatible alias for the original (misspelled) name. Will be removed
# in a future release (QA-018).
DECIMAL_PRECESSION = DECIMAL_PRECISION


def all_subclasses(cls: type) -> set[type]:
    """Return all subclasses of a given class."""
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


@contextmanager
def catch_to_logger(logger: object, re_throw: bool = False) -> Generator[None, None, None]:
    """Catch exceptions and log them to a logger."""
    try:
        yield
    except Exception as e:
        if logger and hasattr(logger, "exception"):
            logger.exception(e)  # type: ignore[reportAttributeAccessIssue]
            if re_throw:
                raise e
        else:
            raise e


@contextmanager
def timer_block(label: str = "Timer", console: Console | None = None) -> Generator[None, None, None]:
    """Time a block of code."""

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        if not console:
            console = console_err
        console.print(f"{label} took {elapsed_time:.4f} seconds.")


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr.

    Warning: this redirects stdout and stderr to devnull and will silently hide
    errors, tracebacks, and prompts. Never wrap authentication, network fetch, or
    other security-relevant code paths with it. Keep its scope narrow (e.g.,
    quieting a noisy library banner) so real failures are not masked (CWE-778).
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
