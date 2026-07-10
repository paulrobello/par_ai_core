"""File, path, and context-gathering utilities for the par_ai_core package."""

from __future__ import annotations

import glob
import html
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from os.path import isfile, join
from pathlib import Path

from rich.console import Console

from par_ai_core.par_logging import console_err

logger = logging.getLogger(__name__)


def get_files(path: str | Path | os.PathLike[str], ext: str = "") -> list[str]:
    """Get list of files in a directory, optionally excluding by extension.

    Args:
        path: Directory path to search
        ext: File extension to exclude. Files whose name ends with this suffix are
            omitted from the result. If empty, returns all files. Defaults to "".

    Returns:
        list[str]: Alphabetically sorted list of filenames in the directory,
            excluding files ending with the specified extension if provided.
    """
    ret = [f for f in os.listdir(path) if isfile(join(path, f)) and (not ext or not f.endswith(ext))]
    ret.sort()
    return ret


def read_text_file_to_stringio(file_path: str, encoding: str = "utf-8") -> StringIO:
    """
    Reads in a text file and returns it as a StringIO object.

    Args:
            file_path (str): The path to the file to read.
            encoding (str): The encoding of the file.

    Returns:
            StringIO: The text file as a StringIO object.
    """
    with open(file_path, encoding=encoding) as file:
        return StringIO(file.read())


def read_env_file(filename: str, console: Console | None = None) -> dict[str, str]:
    """
    Read environment variables from a file into a dictionary
    Lines starting with # are ignored

    Args:
        filename (str): The name of the file to read
        console (Console, optional): The console to use for output

    Returns:
        Dict[str, str]: A dictionary containing the environment variables
    """
    env_vars: dict[str, str] = {}
    if not os.path.exists(filename):
        return env_vars
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
            except ValueError:
                if not console:
                    console = console_err
                console.print("Invalid line format")
    return env_vars


@contextmanager
def add_module_path(path: str) -> Generator[None, None, None]:
    """Add a module path to sys.path temporarily."""
    sys.path.append(path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)


code_python_file_globs: list[str | Path] = [
    "./**/*.py",
    "./**/*.ipynb",
]

code_js_file_globs: list[str | Path] = [
    "./**/*.js",
    "./**/*.ts",
]

code_frontend_file_globs: list[str | Path] = [
    "./**/*.jsx",
    "./**/*.tsx",
    "./**/*.vue",
    "./**/*.svelte",
    "./**/*.html",
    "./**/*.css",
]

code_rust_file_globs: list[str | Path] = [
    "./**/*.rs",
    "./**/*.toml",
]

code_java_file_globs: list[str | Path] = [
    "./**/*.java",
    "./**/*.gradle",
    "./**/*.gradle.kts",
    "./**/*.kt",
    "./**/*.kts",
]


def get_file_list_for_context(file_patterns: list[str | Path]) -> list[Path]:
    """
    Gather files for context.

    Args:
        file_patterns (list[str | Path]): List of file glob patterns to match

    Returns:
        list[Path]: List of files matching the patterns
    """
    files = []
    for pattern in file_patterns:
        if isinstance(pattern, Path):
            pattern = pattern.as_posix()
        files += glob.glob(pattern, recursive=True, include_hidden=False)
    result = []
    for file in files:
        f = Path(file)
        if f.is_file():
            f_path = str(f.as_posix())
            if (
                f.name.startswith(".")
                or "/.git/" in f_path
                or "/.venv/" in f_path
                or "/venv/" in f_path
                or "/node_modules/" in f_path
                or "/__pycache__/" in f_path
            ):
                continue
            result.append(f)

    return result


def gather_files_for_context(file_patterns: list[str | Path], max_context_length: int = 0) -> str:
    """
    Gather files for context.

    Args:
        file_patterns (list[str | Path]): List of file glob patterns to match
        max_context_length (int, optional): Maximum context length. Defaults to 0 (no limit).

    Returns:
        str: xml formatted list of files and their contents
    """
    files = get_file_list_for_context(file_patterns)

    if not files:
        return "<files>\n</files>\n"

    if max_context_length < 0:
        max_context_length = 0
    doc = StringIO()
    doc.write("<files>\n")
    i: int = 0
    curr_len = 17
    for file in files:
        try:
            st = f"""<file index="{i}">\n<source>{html.escape(file.as_posix(), quote=True)}</source>\n<file-content>{html.escape(file.read_text(encoding="utf-8"), quote=True)}</file-content>\n</file>\n"""
            if max_context_length and curr_len + len(st) > max_context_length:
                break
            doc.write(st)
            curr_len += len(st)
            i += 1
        except Exception:
            logger.debug("Failed to read file for context: %s", file, exc_info=True)

    doc.write("</files>\n")
    return doc.getvalue()
