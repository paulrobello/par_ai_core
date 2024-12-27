"""Tests for llm_utils module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from par_ai_core.llm_config import LlmConfig
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.llm_utils import llm_config_from_env, summarize_content


@pytest.fixture
def mock_env():
    """Set up test environment variables."""
    env_vars = {
        "PARAI_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "PARAI_MODEL": "gpt-4",
        "PARAI_AI_BASE_URL": "https://test.openai.com",
        "PARAI_TEMPERATURE": "0.5",
        "PARAI_USER_AGENT_APPID": "test-app",
        "PARAI_STREAMING": "true",
        "PARAI_NUM_CTX": "2048",
        "PARAI_TIMEOUT": "30",
        "PARAI_NUM_PREDICT": "100",
        "PARAI_REPEAT_LAST_N": "64",
        "PARAI_REPEAT_PENALTY": "1.1",
        "PARAI_MIROSTAT": "2",
        "PARAI_MIROSTAT_ETA": "0.1",
        "PARAI_MIROSTAT_TAU": "5.0",
        "PARAI_TFS_Z": "0.5",
        "PARAI_TOP_K": "40",
        "PARAI_TOP_P": "0.9",
        "PARAI_SEED": "42",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield


def test_llm_config_from_env_basic(mock_env):
    """Test basic configuration from environment variables."""
    config = llm_config_from_env()
    assert isinstance(config, LlmConfig)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.base_url == "https://test.openai.com"
    assert config.temperature == 0.5
    assert config.user_agent_appid == "test-app"
    assert config.streaming is True


def test_llm_config_from_env_missing_provider():
    """Test error when AI provider is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="PARAI_AI_PROVIDER environment variable not set"):
            llm_config_from_env()


def test_llm_config_from_env_missing_api_key():
    """Test error when API key is not set for non-local providers."""
    with patch.dict(os.environ, {"PARAI_AI_PROVIDER": "OpenAI"}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
            llm_config_from_env()


def test_llm_config_from_env_local_provider():
    """Test configuration with local providers (no API key required)."""
    with patch.dict(os.environ, {"PARAI_AI_PROVIDER": "Ollama", "PARAI_MODEL": "llama2"}, clear=True):
        config = llm_config_from_env()
        assert isinstance(config, LlmConfig)
        assert config.provider == LlmProvider.OLLAMA


def test_llm_config_from_env_custom_prefix():
    """Test configuration with custom environment variable prefix."""
    env_vars = {
        "CUSTOM_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "CUSTOM_MODEL": "gpt-4",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = llm_config_from_env(prefix="CUSTOM")
        assert isinstance(config, LlmConfig)
        assert config.provider == LlmProvider.OPENAI
        assert config.model_name == "gpt-4"


def test_summarize_content():
    """Test content summarization."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.name = "test-model"
    mock_llm.invoke.return_value = AIMessage(content="Test Summary")

    content = "Test content to summarize"
    result = summarize_content(content, mock_llm)

    assert result == "Test Summary"
    mock_llm.invoke.assert_called_once()

    # Verify the messages passed to the LLM
    calls = mock_llm.invoke.call_args
    messages = calls[0][0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == content


def test_summarize_content_with_long_text():
    """Test summarization with longer content."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.name = "test-model"
    mock_llm.invoke.return_value = AIMessage(content="Long Summary")

    long_content = "Long " * 1000
    result = summarize_content(long_content, mock_llm)

    assert result == "Long Summary"
    mock_llm.invoke.assert_called_once()


def test_summarize_content_empty_input():
    """Test summarization with empty content."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.name = "test-model"
    mock_llm.invoke.return_value = AIMessage(content="Empty content summary")

    result = summarize_content("", mock_llm)

    assert result == "Empty content summary"
    mock_llm.invoke.assert_called_once()
