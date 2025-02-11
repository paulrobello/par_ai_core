"""Tests for llm_providers module."""

import os
from unittest.mock import patch

import pytest

from par_ai_core.llm_providers import (
    LlmProvider,
    get_provider_name_fuzzy,
    get_providers_with_api_keys,
    get_provider_select_options,
    is_provider_api_key_set,
    provider_name_to_enum,
)


def test_llm_provider_enum():
    """Test LlmProvider enum values."""
    assert LlmProvider.OPENAI.value == "OpenAI"
    assert LlmProvider.ANTHROPIC.value == "Anthropic"
    assert LlmProvider.GEMINI.value == "Gemini"
    assert LlmProvider.OLLAMA.value == "Ollama"
    assert LlmProvider.BEDROCK.value == "Bedrock"
    assert LlmProvider.GITHUB.value == "Github"
    assert LlmProvider.MISTRAL.value == "Mistral"
    assert LlmProvider.OPENROUTER.value == "OpenRouter"
    assert LlmProvider.DEEPSEEK.value == "Deepseek"
    assert LlmProvider.LITELLM.value == "LiteLLM"


def test_get_provider_name_fuzzy():
    """Test fuzzy provider name matching."""
    # Exact matches
    assert get_provider_name_fuzzy("OpenAI") == "OpenAI"
    assert get_provider_name_fuzzy("Anthropic") == "Anthropic"

    # Case insensitive
    assert get_provider_name_fuzzy("openai") == "OpenAI"
    assert get_provider_name_fuzzy("ANTHROPIC") == "Anthropic"

    # Prefix matches
    assert get_provider_name_fuzzy("openr") == "OpenRouter"
    assert get_provider_name_fuzzy("anth") == "Anthropic"

    # No match
    assert get_provider_name_fuzzy("invalid") == ""


def test_provider_name_to_enum():
    """Test converting provider name to enum."""
    assert provider_name_to_enum("OpenAI") == LlmProvider.OPENAI
    assert provider_name_to_enum("Anthropic") == LlmProvider.ANTHROPIC

    with pytest.raises(ValueError):
        provider_name_to_enum("InvalidProvider")


@patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
    },
    clear=True,
)
def test_is_provider_api_key_set():
    """Test checking if provider API key is set."""
    # Providers that don't need API keys
    assert is_provider_api_key_set(LlmProvider.OLLAMA) is True
    assert is_provider_api_key_set(LlmProvider.LLAMACPP) is True

    # Providers with API keys set
    assert is_provider_api_key_set(LlmProvider.OPENAI) is True
    assert is_provider_api_key_set(LlmProvider.ANTHROPIC) is True

    # Provider without API key
    assert is_provider_api_key_set(LlmProvider.GEMINI) is False


@patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
    },
    clear=True,
)
def test_get_providers_with_api_keys():
    """Test getting list of providers with API keys."""
    providers = get_providers_with_api_keys()

    # Should always include providers that don't need API keys
    assert LlmProvider.OLLAMA in providers
    assert LlmProvider.LLAMACPP in providers

    # Should include providers with API keys set
    assert LlmProvider.OPENAI in providers
    assert LlmProvider.ANTHROPIC in providers

    # Should not include providers without API keys
    assert LlmProvider.GEMINI not in providers


@patch.dict(
    os.environ,
    {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
    },
    clear=True,
)
def test_get_provider_select_options():
    """Test getting provider selection options."""
    options = get_provider_select_options()

    # Check format of options
    assert all(isinstance(opt, tuple) and len(opt) == 2 for opt in options)
    assert all(isinstance(opt[0], str) and isinstance(opt[1], LlmProvider) for opt in options)

    # Check content
    provider_names = [opt[0] for opt in options]
    assert "OpenAI" in provider_names
    assert "Anthropic" in provider_names
    assert "Ollama" in provider_names
    assert "LlamaCpp" in provider_names
