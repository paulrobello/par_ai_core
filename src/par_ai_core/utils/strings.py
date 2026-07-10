"""String-manipulation utilities for the par_ai_core package."""

from __future__ import annotations

import random
import re
import string


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


def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    """Generate a random string of uppercase letters and digits.

    Args:
        size: Length of the string to generate. Defaults to 6.
        chars: Characters to use for the string. Defaults to uppercase letters and digits.

    Returns:
        str: The generated random string

    Note:
        Not cryptographically secure. This uses the non-security ``random``
        module and is intended only for non-security purposes such as
        user-agent jitter. Do NOT use this for session IDs, nonces, API tokens,
        or any other security-sensitive identifier; use ``secrets.choice``
        instead.
    """
    return "".join(random.choice(chars) for _ in range(size))
