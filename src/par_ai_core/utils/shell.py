"""Shell and subprocess command utilities for the par_ai_core package."""

from __future__ import annotations

import logging
import shlex
import subprocess

from rich.console import Console

from par_ai_core.par_logging import console_err

logger = logging.getLogger(__name__)


def run_shell_cmd(cmd: str) -> str | None:
    """Run a shell command and return the output.

    .. deprecated::
        Prefer ``run_cmd`` which accepts a list[str] to avoid argument injection.
    """
    try:
        return subprocess.run(
            shlex.split(cmd), shell=False, capture_output=True, check=True, encoding="utf-8"
        ).stdout.strip()
    except Exception:
        logger.debug("Shell command failed: %s", cmd, exc_info=True)
        return None


def run_cmd(params: list[str], console: Console | None = None, check: bool = True) -> str | None:
    """Run a command and return the output.

    Args:
        params: Command and arguments as list of strings
        console: Optional console for error output
        check: Whether to raise CalledProcessError on command failure

    Returns:
        Command output as string, or None if command failed
    """
    try:
        result = subprocess.run(params, capture_output=True, text=True, check=check)
        if result.returncode != 0:
            if not console:
                console = console_err
            console.print(f"Error running command: {result.stderr}")
            return None

        ret = result.stdout.strip()
        # Split the output into lines
        lines = [line for line in ret.splitlines() if not line.startswith("failed to get console mode")]
        # Get the last two lines
        return "\n".join(lines)
    except FileNotFoundError as e:
        if not console:
            console = console_err
        console.print(f"Error running command: {e}")
        return None
    except subprocess.CalledProcessError as e:
        if not console:
            console = console_err
        console.print(f"Error running command: {e.stderr}")
        return None
