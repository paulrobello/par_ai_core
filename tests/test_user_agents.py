"""Tests for user_agents module."""

import re

import pytest

from par_ai_core.user_agents import get_random_user_agent


@pytest.fixture
def user_agent() -> str:
    """Get a random user agent string for testing.

    Returns:
        str: A randomly generated user agent string.
    """
    return get_random_user_agent()


def test_user_agent_basic_structure(user_agent: str) -> None:
    """Test that the user agent string has the expected basic structure.

    Args:
        user_agent: The generated user agent string to test.
    """
    assert user_agent.startswith("Mozilla/5.0 (")
    assert ")" in user_agent
    assert len(user_agent) > 50  # Reasonable minimum length


def test_operating_system_present(user_agent: str) -> None:
    """Test that the user agent contains valid OS information.

    Args:
        user_agent: The generated user agent string to test.
    """
    valid_os = [
        "Windows NT 10.0; Win64; x64",
        "Windows NT 11.0; Win64; x64",
        "Macintosh; Intel Mac OS X",
        "Macintosh; arm64",
    ]
    assert any(os in user_agent for os in valid_os), f"No valid OS found in: {user_agent}"


def test_platform_info(user_agent: str) -> None:
    """Test that platform information is correctly included.

    Args:
        user_agent: The generated user agent string to test.
    """
    if "Windows" in user_agent:
        assert "Win64; x64" in user_agent
    elif "Macintosh" in user_agent:
        assert any(platform in user_agent for platform in ["Intel Mac OS X", "arm64"])


def test_browser_version_format(user_agent: str) -> None:
    """Test that browser version numbers follow expected patterns.

    Args:
        user_agent: The generated user agent string to test.
    """
    # Chrome version pattern
    chrome_pattern = r"Chrome/\d+\.\d+\.\d+\.\d+"
    # Firefox version pattern
    firefox_pattern = r"Firefox/\d+\.\d+"
    # Safari version pattern
    safari_pattern = r"Version/\d+\.\d+ Safari/\d+"
    # Edge version pattern
    edge_pattern = r"Edg/\d+\.\d+\.\d+\.\d+"

    # At least one browser version pattern should match
    patterns = [chrome_pattern, firefox_pattern, safari_pattern, edge_pattern]
    assert any(re.search(pattern, user_agent) for pattern in patterns)


def test_webkit_version_format(user_agent: str) -> None:
    """Test WebKit version format when present.

    Args:
        user_agent: The generated user agent string to test.
    """
    if "AppleWebKit" in user_agent:
        webkit_pattern = r"AppleWebKit/\d+\.\d+"
        assert re.search(webkit_pattern, user_agent) is not None


def test_multiple_agents_are_different() -> None:
    """Test that multiple calls generate different user agents."""
    agents = {get_random_user_agent() for _ in range(5)}
    assert len(agents) > 1  # At least some should be different


def test_get_random_user_agent_return_type() -> None:
    """Test that the function returns a string."""
    assert isinstance(get_random_user_agent(), str)


def test_user_agent_components_consistency(user_agent: str) -> None:
    """Test that user agent components are consistent with each other.

    Args:
        user_agent: The generated user agent string to test.
    """
    if "Firefox" in user_agent:
        assert "AppleWebKit" not in user_agent
        assert "Gecko/20100101" in user_agent
    elif "Safari" in user_agent and "Chrome" not in user_agent and "Edg" not in user_agent:
        assert "AppleWebKit" in user_agent
        assert "Version/" in user_agent
    elif "Chrome" in user_agent:
        assert "AppleWebKit" in user_agent
        assert "Chrome/" in user_agent
        assert "Safari/" in user_agent
    elif "Edg" in user_agent:
        assert "AppleWebKit" in user_agent
        assert "Edg/" in user_agent
    elif "Chrome" in user_agent:
        assert "AppleWebKit" in user_agent
        assert "Safari/" in user_agent
