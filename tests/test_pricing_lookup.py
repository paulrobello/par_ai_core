"""Tests for pricing lookup functionality."""

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from par_ai_core.llm_config import LlmConfig
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.pricing_lookup import (
    PricingDisplay,
    accumulate_cost,
    get_api_call_cost,
    get_api_cost_model_name,
    mk_usage_metadata,
    show_llm_cost,
)


def test_mk_usage_metadata():
    """Test creation of usage metadata dictionary."""
    metadata = mk_usage_metadata()

    assert isinstance(metadata, dict)
    assert metadata["input_tokens"] == 0
    assert metadata["output_tokens"] == 0
    assert metadata["total_tokens"] == 0
    assert metadata["cache_write"] == 0
    assert metadata["cache_read"] == 0
    assert metadata["reasoning"] == 0
    assert metadata["successful_requests"] == 0
    assert metadata["tool_call_count"] == 0
    assert metadata["total_cost"] == 0.0


@pytest.mark.parametrize(
    "input_model,expected_model",
    [
        ("gpt-4", "gpt-4"),
        ("claude-3-sonnet-20240229", "claude-3-sonnet-20240229"),
        ("gpt-4-turbo", "gpt-4-turbo"),
        ("unknown-model", "unknown-model"),
    ],
)
def test_get_api_cost_model_name(input_model: str, expected_model: str):
    """Test API cost model name resolution."""
    assert get_api_cost_model_name(input_model) == expected_model


def test_get_api_call_cost():
    """Test API call cost calculation."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read": 20,
        "cache_write": 10,
    }

    cost = get_api_call_cost(config, usage)
    assert isinstance(cost, float)
    assert cost > 0

    # Test free providers
    config.provider = LlmProvider.OLLAMA
    assert get_api_call_cost(config, usage) == 0

    # Test batch pricing
    config.provider = LlmProvider.OPENAI
    regular_cost = get_api_call_cost(config, usage, batch_pricing=False)
    batch_cost = get_api_call_cost(config, usage, batch_pricing=True)
    assert batch_cost == regular_cost * 0.5

    # Test unknown model
    config.model_name = "unknown-model"
    assert get_api_call_cost(config, usage) == 0


def test_accumulate_cost_dict():
    """Test cost accumulation from dictionary response."""
    usage = mk_usage_metadata()
    response = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
    }

    accumulate_cost(response, usage)

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150
    assert usage["cache_write"] == 10
    assert usage["cache_read"] == 20


class MockResponse:
    """Mock response object for testing."""

    def __init__(self):
        self.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {
                "cache_creation": 10,
                "cache_read": 20,
            },
            "output_token_details": {
                "reasoning": 30,
            },
        }


def test_accumulate_cost_object():
    """Test cost accumulation from object response."""
    usage = mk_usage_metadata()
    response = MockResponse()

    accumulate_cost(response, usage)

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["cache_write"] == 10
    assert usage["cache_read"] == 20
    assert usage["reasoning"] == 30


def test_show_llm_cost():
    """Test LLM cost display functionality."""
    usage = {
        "gpt-4": {
            "total_cost": 0.123,
            "input_tokens": 100,
            "output_tokens": 50,
        }
    }

    # Test NONE display
    show_llm_cost(usage, show_pricing=PricingDisplay.NONE)

    # Test PRICE display
    console = Console(file=StringIO(), force_terminal=False)  # Disable colors
    with patch("par_ai_core.pricing_lookup.console_err", console):
        show_llm_cost(usage, show_pricing=PricingDisplay.PRICE)
        output = console.file.getvalue()
        assert "0.1230" in output

    # Test DETAILS display
    console = Console(file=StringIO(), force_terminal=True)
    with patch("par_ai_core.pricing_lookup.console_err", console):
        show_llm_cost(usage, show_pricing=PricingDisplay.DETAILS)
        output = console.file.getvalue()
        assert "gpt-4" in output
        assert "$0.123" in output

    # Test with missing total_cost
    usage_no_cost = {
        "gpt-4": {
            "input_tokens": 100,
            "output_tokens": 50,
        }
    }
    console = Console(file=StringIO(), force_terminal=False)
    with patch("par_ai_core.pricing_lookup.console_err", console):
        show_llm_cost(usage_no_cost, show_pricing=PricingDisplay.PRICE)
        output = console.file.getvalue()
        assert "0.0000" in output

    # Test with empty usage
    empty_usage = {}
    console = Console(file=StringIO(), force_terminal=False)
    with patch("par_ai_core.pricing_lookup.console_err", console):
        show_llm_cost(empty_usage, show_pricing=PricingDisplay.PRICE)
        output = console.file.getvalue()
        assert "$0.0000" in output