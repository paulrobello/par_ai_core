"""
Various types, utility functions and decorators for the par_ai_core package.

This module provides a wide range of utility functions and decorators to support
common operations in AI and data processing tasks. It includes functions for:

- String manipulation (e.g., camel case to snake case conversion)
- Data structure operations (e.g., nested dictionary access)
- File and I/O operations (e.g., CSV parsing, file reading)
- Type checking and conversion
- Hashing and encryption
- Command execution and shell interactions
- Timing and performance measurement
- Exception handling and logging
- UUID validation
- Environment variable management

The module also includes several context managers for temporary modifications
to system state or execution environment.

These utilities are designed to streamline development and improve code
readability across the par_ai_core package and related projects.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import html
import math
import os
import random
import re
import shlex
import string
import subprocess
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from io import StringIO
from os.path import isfile, join
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from rich.console import Console

from par_ai_core.par_logging import console_err

DECIMAL_PRECESSION = 5


def has_stdin_content() -> bool:
    """Check if there is content available on stdin.

    Returns:
        bool: True if there is content available on stdin, False otherwise.
    """
    try:
        if sys.stdin.isatty():
            return False
        data = sys.stdin.buffer.read(1)
        sys.stdin.buffer.seek(0)  # Reset the buffer position
        return data != b""
    except (OSError, AttributeError):
        return False


def md(soup: BeautifulSoup, **options: Any) -> str:
    """Convert BeautifulSoup object to Markdown.

    Args:
        soup: The BeautifulSoup object to convert
        **options: Additional options to pass to the MarkdownConverter

    Returns:
        str: The converted Markdown string
    """
    return MarkdownConverter(**options).convert_soup(soup)


def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    """Generate a random string of uppercase letters and digits.

    Args:
        size: Length of the string to generate. Defaults to 6.
        chars: Characters to use for the string. Defaults to uppercase letters and digits.

    Returns:
        str: The generated random string
    """
    return "".join(random.choice(chars) for _ in range(size))


def json_serial(obj: Any) -> str:
    """
    JSON serializer for objects not serializable by default json code.

    :param obj: The object to serialize.
    :return: The serialized object.
    """

    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def coalesce(*arg: Any) -> Any:
    """Return first non-None item from the provided arguments.

    Args:
        *arg: Variable number of arguments to check.

    Returns:
        Any: The first non-None item in the arguments.
        If all arguments are None, returns None.

    Example:
        >>> coalesce(None, "", 0, "hello")
        ''
        >>> coalesce(None, None, 42)
        42
    """
    return next((a for a in arg if a is not None), None)


def chunks(lst: list[Any], n: int) -> Generator[list[Any], None, None]:
    """Yield successive n-sized chunks from a list.

    Args:
        lst: The list to split into chunks
        n: The size of each chunk

    Returns:
        Generator[list[Any], None, None]: Generator yielding chunks of the list
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def to_camel_case(snake_str: str) -> str:
    """Convert a snake_case string to camelCase.

    Args:
        snake_str: The snake_case string to convert

    Returns:
        str: The converted camelCase string

    Example:
        >>> to_camel_case("hello_world")
        'helloWorld'
    """
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def to_class_case(snake_str: str) -> str:
    """Convert a snake_case string to PascalCase (ClassCase).

    Spaces are converted to underscores before conversion.

    Args:
        snake_str: The snake_case string to convert

    Returns:
        str: The converted PascalCase string

    Example:
        >>> to_class_case("hello_world")
        'HelloWorld'
        >>> to_class_case("hello world")
        'HelloWorld'
    """
    components = snake_str.replace(" ", "_").split("_")
    # We capitalize the first letter of each component
    # with the 'title' method and join them together.
    return "".join(x.title() for x in components[0:])


def get_files(path: str | Path | os.PathLike[str], ext: str = "") -> list[str]:
    """Get list of files in a directory, optionally filtered by extension.

    Args:
        path: Directory path to search
        ext: File extension to filter by. If empty, returns all files. Defaults to "".

    Returns:
        list[str]: Alphabetically sorted list of filenames in the directory,
            excluding files ending with the specified extension if provided.
    """
    ret = [f for f in os.listdir(path) if isfile(join(path, f)) and (not ext or not f.endswith(ext))]
    ret.sort()
    return ret


# tests if value can be converted to float
def is_float(s: Any) -> bool:
    """Test if a value can be converted to float.

    Args:
        s: Any value to test.

    Returns:
        bool: True if the value can be converted to float, False otherwise.

    Example:
        >>> is_float("3.14")
        True
        >>> is_float("abc")
        False
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# tests if value can be converted to int
def is_int(s: Any) -> bool:
    """Test if a value can be converted to integer.

    Args:
        s: Any value to test.

    Returns:
        bool: True if the value can be converted to integer, False otherwise.

    Example:
        >>> is_int("42")
        True
        >>> is_int("3.14")
        False
    """
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_date(date_text: str, fmt: str = "%Y/%m/%d") -> bool:
    """Test if a string represents a valid date in the specified format.

    Args:
        date_text: String to test as a date.
        fmt: Date format string using strftime format codes. Defaults to "%Y/%m/%d".

    Returns:
        bool: True if the string represents a valid date in the specified format,
        False otherwise.

    Example:
        >>> is_date("2024/01/20")
        True
        >>> is_date("2024-01-20", fmt="%Y-%m-%d")
        True
    """
    try:
        datetime.strptime(date_text, fmt)
        return True
    except ValueError:
        return False


def has_value(v: Any, search: str, depth: int = 0) -> bool:
    """Recursively search a data structure for a value.

    Args:
        v: The data structure to search (can be dict, list, or primitive type).
        search: The string value to search for.
        depth: Current recursion depth (used internally, defaults to 0).

    Returns:
        bool: True if the search value is found, False otherwise.

    Notes:
        - Searches dictionaries recursively up to depth of 4
        - For integers, trims .00 from search string before comparing
        - For floats, truncates to length of search string before comparing
        - For strings, checks if they start or end with search value (case-insensitive)
    """
    # don't go more than 3 levels deep
    if depth > 4:
        return False
    # if is a dict, search all dict values recursively
    if isinstance(v, dict):
        for dv in v.values():
            if has_value(dv, search, depth + 1):
                return True
    # if is a list, search all list values recursively
    if isinstance(v, list):
        for li in v:
            if has_value(li, search, depth + 1):
                return True
    # if is an int, trim off .00 for search if it exists then compare
    if isinstance(v, int):
        search = search.rstrip(".00")
        if str(v) == search:
            return True
    # if is a float, truncate string version of float to same size as search
    if isinstance(v, float):
        v = str(v)[0 : len(search)]
        if search == v:
            return True
    # if is a string, strip and lowercase it then check if string starts with search
    if isinstance(v, str):
        if v.strip().lower().startswith(search) or v.strip().lower().endswith(search):
            return True
    return False


def is_zero(val: Any) -> bool:
    """Test if a value equals zero, handling different numeric types.

    Args:
        val: Value to test (can be int, float, or Decimal).

    Returns:
        bool: True if the value equals zero, False otherwise.
        Returns False for None values.

    Notes:
        - For Decimal, rounds to DECIMAL_PRECESSION before comparing
        - For float, uses math.isclose() with relative tolerance of 1e-05
        - For int, uses exact comparison
    """
    if val is None:
        return False
    t = type(val)
    if t is Decimal:
        return val.quantize(Decimal(f"1e-{DECIMAL_PRECESSION}")).is_zero()
    if t is float:
        return math.isclose(round(val, 5), 0, rel_tol=1e-05)
    if t is int:
        return 0 == val
    return False


def non_zero(val: Any) -> bool:
    """Test if a value is not equal to zero.

    Args:
        val: Value to test (can be int, float, or Decimal).

    Returns:
        bool: True if the value is not zero, False if it is zero.
        Returns True for None values.

    Note:
        This is the inverse of is_zero().
    """
    return not is_zero(val)


def dict_keys_to_lower(dictionary: dict) -> dict:
    """
    Return a new dictionary with all keys lowercase
    @param dictionary: dict with keys that you want to lowercase
    @return: new dictionary with lowercase keys
    """
    return {k.lower(): v for k, v in dictionary.items()}


def is_valid_uuid_v4(value: str) -> bool:
    """Test if value is a valid UUID v4."""
    try:
        uuid_obj = uuid.UUID(value, version=4)
        return str(uuid_obj) == value  # Check if the string representation matches
    except ValueError:
        return False


def parse_csv_text(csv_data: StringIO, has_header: bool = True) -> list[dict]:
    """
    Reads in a CSV file as text and returns it as a list of dictionaries.

    Args:
            csv_data (StringIO): The CSV file as text.
            has_header (bool): Whether the CSV has a header row. Defaults to True.

    Returns:
            list[dict]: The CSV data as a list of dictionaries.

    Raises:
            csv.Error: If there's an issue parsing the CSV data.
    """
    try:
        if has_header:
            reader = csv.DictReader(csv_data, strict=True)
            try:
                rows = []
                for row in reader:
                    rows.append(row)
                return rows
            except Exception as e:
                raise csv.Error(f"Error parsing CSV data: {str(e)}")
        else:
            reader = csv.reader(csv_data, strict=True)
            try:
                rows = list(reader)
                if not rows:
                    return []
                # Use column indices as keys when no header
                headers = [str(i) for i in range(len(rows[0]))]
                return [dict(zip(headers, row)) for row in rows]
            except Exception as e:
                raise csv.Error(f"Error parsing CSV data: {str(e)}")
    except Exception as e:
        raise csv.Error(f"Error parsing CSV data: {str(e)}")


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


def md5_hash(data: str) -> str:
    """
    Returns a md5 hash of the input data.

    Args:
            data (str): The input data.

    Returns:
            str: The md5 hash of the input data.
    """
    md5 = hashlib.md5(data.encode("utf-8"))
    return md5.hexdigest()


def sha1_hash(data: str) -> str:
    """
    Returns a SHA1 hash of the input data.

    Args:
            data (str): The input data.

    Returns:
            str: The SHA1 hash of the input data.
    """
    sha1 = hashlib.sha1(data.encode("utf-8"))
    return sha1.hexdigest()


def sha256_hash(data: str) -> str:
    """
    Returns a SHA256 hash of the input data.

    Args:
            data (str): The input data.

    Returns:
            str: The SHA256 hash of the input data.
    """
    sha256 = hashlib.sha256(data.encode("utf-8"))
    return sha256.hexdigest()


def nested_get(dictionary: dict, keys: str | list[str]) -> Any:
    """
    Returns the value for a given key in a nested dictionary.

    Args:
            dictionary (dict): The nested dictionary to search.
            keys (str | list[str]): The key or list of keys to search for.

    Returns:
            Any: The value for the given key or None if the key does not exist.
    """
    if isinstance(keys, str):
        keys = keys.split(".")
    if keys and dictionary:
        element = keys[0]
        if element in dictionary:
            if len(keys) == 1:
                return dictionary[element]
            return nested_get(dictionary[element], keys[1:])
    return None


@contextmanager
def add_module_path(path: str) -> Generator[None, None, None]:
    """Add a module path to sys.path temporarily."""
    sys.path.append(path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)


@contextmanager
def catch_to_logger(logger: object, re_throw: bool = False) -> Generator[None, None, None]:
    """Catch exceptions and log them to a logger."""
    try:
        yield
    except Exception as e:
        if logger and hasattr(logger, "exception"):
            logger.exception(e)  # type: ignore
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


def str_ellipsis(s: str, max_len: int, pad_char: str = " ") -> str:
    """Return a left space padded string exactly max_len with ellipsis if it exceeds max_len."""
    if len(s) <= max_len:
        if pad_char:
            return s.ljust(max_len, pad_char)
        return s
    return s[: max_len - 3] + "..."


def camel_to_snake(name: str) -> str:
    """Convert name from CamelCase to snake_case.

    Args:
        name: A symbol name, such as a class name.

    Returns:
        Name in snake case.

    Examples:
        >>> camel_to_snake("camelCase")
        'camel_case'
        >>> camel_to_snake("ThisIsATest")
        'this_is_a_test'
        >>> camel_to_snake("ABC")
        'abc'
    """
    # Special case for all uppercase strings
    if name.isupper():
        return name.lower()

    pattern = re.compile(r"(?<!^)(?<!_)(?:[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$))")
    return pattern.sub(lambda m: f"_{m.group(0).lower()}", name).lower()


def detect_syntax(text: str) -> str | None:
    """Detect the syntax of the text."""
    lines = text.split("\n")
    if len(lines) > 0:
        line = lines[0]
        if line.startswith("#!"):
            if line.endswith("/bash") or line.endswith("/sh") or line.endswith(" bash") or line.endswith(" sh"):
                return "bash"
    return None


def hash_list_by_key(data: list[dict], id_key: str = "message_id") -> dict:
    """Hash a list of dictionaries by a key."""
    return {item[id_key]: item for item in data}


def run_shell_cmd(cmd: str) -> str | None:
    """Run a command and return the output."""
    try:
        return subprocess.run(
            shlex.split(cmd), shell=False, capture_output=True, check=True, encoding="utf-8"
        ).stdout.strip()
    except Exception as _:
        return None


def output_to_dicts(output: str) -> list[dict[str, Any]]:
    """Convert a tab-delimited output to a list of dicts."""
    if not output:
        return []
    # split string on newline loop over each line and convert
    # Use csv module to parse the tab-delimited output
    reader = csv.DictReader(StringIO(output), delimiter="\t")
    ret = []
    for model in reader:
        mod = {}
        for key, value in model.items():
            mod[key.strip().lower()] = value.strip()
        ret.append(mod)
    return ret


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


def all_subclasses(cls: type) -> set[type]:
    """Return all subclasses of a given class."""
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
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


def gather_files_for_context(file_patterns: list[str | Path], max_context_length: int = 0) -> str:
    """
    Gather files for context.

    Args:
        file_patterns (list[str | Path]): List of file glob patterns to match
        max_context_length (int, optional): Maximum context length. Defaults to 0 (no limit).

    Returns:
        str: xml formatted list of files and their contents
    """
    files = []
    for pattern in file_patterns:
        try:
            if isinstance(pattern, Path):
                pattern = pattern.as_posix()
            if sys.version_info >= (3, 11):
                files += glob.glob(pattern, recursive=True, include_hidden=False)
            else:
                files += glob.glob(pattern, recursive=True)
        except Exception as _:
            raise _

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
                st = f"""<file index="{i}">\n<source>{html.escape(f.as_posix(), quote=True)}</source>\n<file-content>{html.escape(f.read_text(encoding='utf-8'), quote=True)}</file-content>\n</file>\n"""
                if max_context_length and curr_len + len(st) > max_context_length:
                    break
                doc.write(st)
                curr_len += len(st)
                i += 1
        except Exception as _:
            pass

    doc.write("</files>\n")
    return doc.getvalue()
