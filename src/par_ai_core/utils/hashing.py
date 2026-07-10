"""Hashing utilities for the par_ai_core package."""

from __future__ import annotations

import hashlib
import warnings


def md5_hash(data: str) -> str:
    """Returns a md5 hash of the input data.

    .. deprecated::
        MD5 is cryptographically broken. Use ``sha256_hash`` for integrity checks.

    Args:
            data (str): The input data.

    Returns:
            str: The md5 hash of the input data.
    """
    warnings.warn(
        "md5_hash uses a cryptographically broken algorithm. Use sha256_hash instead.", DeprecationWarning, stacklevel=2
    )
    md5 = hashlib.md5(data.encode("utf-8"), usedforsecurity=False)
    return md5.hexdigest()


def sha1_hash(data: str) -> str:
    """Returns a SHA1 hash of the input data.

    .. deprecated::
        SHA1 is cryptographically broken. Use ``sha256_hash`` for integrity checks.

    Args:
            data (str): The input data.

    Returns:
            str: The SHA1 hash of the input data.
    """
    warnings.warn(
        "sha1_hash uses a cryptographically broken algorithm. Use sha256_hash instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    sha1 = hashlib.sha1(data.encode("utf-8"), usedforsecurity=False)
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


def hash_list_by_key(data: list[dict], id_key: str = "message_id") -> dict:
    """Hash a list of dictionaries by a key."""
    return {item[id_key]: item for item in data}
