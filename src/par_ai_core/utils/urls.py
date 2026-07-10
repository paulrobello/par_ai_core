"""URL and path-suffix utilities for the par_ai_core package."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit


def is_url(url: str) -> bool:
    """
    Return True if the given string is a valid URL.

    Args:
        url (str): The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(url)
        matches = re.match(r"^https?://", url) is not None
        return all([result.scheme, result.netloc, matches, " " not in url])
    except ValueError:
        return False


def get_url_file_suffix(url: str, default: str = ".jpg") -> str:
    """
    Get url file suffix

    Args:
        url (str): URL
        default (str): Default file suffix if none found

    Returns:
        str: File suffix in lowercase with leading dot
    """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    suffix = os.path.splitext(filename)[1].lower()
    return suffix or default


def get_file_suffix(path: str, default: str = ".jpg") -> str:
    """
    Get file suffix

    Args:
        path (str): file path or url
        default (str): Default file suffix if none found

    Returns:
        str: File suffix in lowercase with leading dot
    """
    if is_url(path):
        return get_url_file_suffix(path)
    try:
        suffix = os.path.splitext(Path(path).name)[1].lower()
        return suffix or default
    except Exception:
        return default


def extract_url_auth(url: str) -> tuple[str, tuple[str, str] | None]:
    """
    Separate auth info from url if present and return clean url and auth info as tuple.

    url str: url to parse

    Returns:
        tuple[str, tuple[str, str] | None]: clean url and auth info as tuple
    """
    parsed_url = urlsplit(url)
    username = parsed_url.username
    password = parsed_url.password
    new_netloc = parsed_url.hostname
    if parsed_url.port is not None:
        if new_netloc:
            new_netloc = f"{new_netloc}:{parsed_url.port}"
        else:
            new_netloc = f":{parsed_url.port}"
    components = (parsed_url.scheme, new_netloc or "", parsed_url.path, parsed_url.query, parsed_url.fragment)
    clean_host_url = str(urlunsplit(components))
    if username and password:
        auth = (username, password)
    else:
        auth = None
    return clean_host_url, auth
