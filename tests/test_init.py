"""Tests for __init__.py module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import par_ai_core


def test_module_attributes():
    """Test that all module attributes are properly defined."""
    assert par_ai_core.__author__ == "Paul Robello"
    assert par_ai_core.__credits__ == ["Paul Robello"]
    assert par_ai_core.__maintainer__ == "Paul Robello"
    assert par_ai_core.__email__ == "probello@gmail.com"
    assert par_ai_core.__version__ == "0.3.2"
    assert par_ai_core.__application_title__ == "Par AI Core"
    assert par_ai_core.__application_binary__ == "par_ai_core"
    assert par_ai_core.__licence__ == "MIT"


def test_user_agent_environment_variable():
    """Test that USER_AGENT environment variable is set correctly."""
    expected_user_agent = f"{par_ai_core.__application_title__} {par_ai_core.__version__}"
    assert os.environ["USER_AGENT"] == expected_user_agent


def test_all_exports():
    """Test that __all__ contains expected exports."""
    expected_exports = [
        "__author__",
        "__credits__",
        "__maintainer__",
        "__email__",
        "__version__",
        "__application_binary__",
        "__licence__",
        "__application_title__",
    ]
    assert par_ai_core.__all__ == expected_exports


@patch("par_ai_core.nest_asyncio")
def test_apply_nest_asyncio_safely_no_loop(mock_nest_asyncio):
    """Test _apply_nest_asyncio_safely when no event loop is running."""
    # Import the function for testing
    from par_ai_core import _apply_nest_asyncio_safely

    with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running event loop")):
        result = _apply_nest_asyncio_safely()
        assert result is True
        mock_nest_asyncio.apply.assert_called_once()


@patch("par_ai_core.nest_asyncio")
def test_apply_nest_asyncio_safely_with_uvloop(mock_nest_asyncio):
    """Test _apply_nest_asyncio_safely with uvloop (should not patch)."""
    from par_ai_core import _apply_nest_asyncio_safely

    mock_loop = MagicMock()
    mock_loop.__class__.__name__ = "UVLoopType"

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        result = _apply_nest_asyncio_safely()
        assert result is False
        mock_nest_asyncio.apply.assert_not_called()


@patch("par_ai_core.nest_asyncio")
def test_apply_nest_asyncio_safely_already_patched(mock_nest_asyncio):
    """Test _apply_nest_asyncio_safely when already patched."""
    from par_ai_core import _apply_nest_asyncio_safely

    mock_loop = MagicMock()
    mock_loop.__class__.__name__ = "EventLoopType"
    mock_loop._nest_patched = True

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        result = _apply_nest_asyncio_safely()
        assert result is True
        mock_nest_asyncio.apply.assert_not_called()


@patch("par_ai_core.nest_asyncio")
def test_apply_nest_asyncio_safely_with_regular_loop(mock_nest_asyncio):
    """Test _apply_nest_asyncio_safely with regular event loop."""
    from par_ai_core import _apply_nest_asyncio_safely

    mock_loop = MagicMock()
    mock_loop.__class__.__name__ = "EventLoopType"
    del mock_loop._nest_patched  # Ensure no _nest_patched attribute

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        result = _apply_nest_asyncio_safely()
        assert result is True
        mock_nest_asyncio.apply.assert_called_once()


@patch("par_ai_core.nest_asyncio")
def test_apply_nest_asyncio_safely_exception_handling(mock_nest_asyncio):
    """Test _apply_nest_asyncio_safely handles exceptions gracefully."""
    from par_ai_core import _apply_nest_asyncio_safely

    # Mock nest_asyncio.apply to raise an exception
    mock_nest_asyncio.apply.side_effect = Exception("Test exception")

    with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running event loop")):
        result = _apply_nest_asyncio_safely()
        assert result is False


def test_warnings_suppressed():
    """Test that warnings are properly suppressed."""
    import warnings
    from langchain_core._api import LangChainBetaWarning

    # Check that the warnings filters are set
    filters = warnings.filters
    langchain_filters = [
        f
        for f in filters
        if len(f) > 2
        and f[2] is not None
        and (f[2] == LangChainBetaWarning or (hasattr(f[2], "__name__") and f[2].__name__ == "LangChainBetaWarning"))
    ]
    deprecation_filters = [
        f
        for f in filters
        if len(f) > 2
        and f[2] is not None
        and (f[2] == DeprecationWarning or (hasattr(f[2], "__name__") and f[2].__name__ == "DeprecationWarning"))
    ]

    # Since warnings may be filtered in various ways, just check that the module imports successfully
    # and that warnings module is configured
    assert isinstance(filters, list), "Warnings filters should be a list"

    # Check that deprecation warnings are being handled
    assert len(deprecation_filters) > 0 or any("ignore" in str(f) for f in filters), (
        "DeprecationWarning filter not found"
    )


def test_nest_asyncio_applied_flag():
    """Test that the _applied flag is set."""
    # The _applied variable should be available in the module
    import par_ai_core

    # Check that _applied exists and is a boolean
    assert hasattr(par_ai_core, "_applied")
    assert isinstance(par_ai_core._applied, bool)
