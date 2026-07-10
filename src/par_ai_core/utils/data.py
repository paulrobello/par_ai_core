"""Data-structure and serialization utilities for the par_ai_core package."""

from __future__ import annotations

import csv
from collections.abc import Generator
from datetime import date, datetime
from io import StringIO
from typing import Any


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


def dict_keys_to_lower(dictionary: dict) -> dict:
    """
    Return a new dictionary with all keys lowercase
    @param dictionary: dict with keys that you want to lowercase
    @return: new dictionary with lowercase keys
    """
    return {k.lower(): v for k, v in dictionary.items()}


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
            return list(reader)
        reader = csv.reader(csv_data, strict=True)
        rows = list(reader)
        if not rows:
            return []
        # Use column indices as keys when no header
        headers = [str(i) for i in range(len(rows[0]))]
        return [dict(zip(headers, row, strict=False)) for row in rows]
    except Exception as e:
        raise csv.Error(f"Error parsing CSV data: {str(e)}") from e
