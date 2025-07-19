"""Tests for llm_utils module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from par_ai_core.llm_config import LlmConfig, ReasoningEffort
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.llm_utils import (
    _chunk_text,
    _estimate_tokens,
    _get_model_context_size,
    llm_config_from_env,
    summarize_content,
)


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


def test_llm_config_from_env_missing_model():
    """Test error when model is not set and no default available."""
    with patch.dict(os.environ, {"PARAI_AI_PROVIDER": "OpenAI", "OPENAI_API_KEY": "test-key"}, clear=True):
        with patch("par_ai_core.llm_utils.provider_default_models", {LlmProvider.OPENAI: None}):
            with pytest.raises(ValueError, match="PARAI_MODEL environment variable not set"):
                llm_config_from_env()


def test_llm_config_from_env_invalid_provider():
    """Test error with invalid provider name."""
    with patch.dict(os.environ, {"PARAI_AI_PROVIDER": "InvalidProvider"}, clear=True):
        with pytest.raises(ValueError):
            llm_config_from_env()


def test_llm_config_from_env_reasoning_effort():
    """Test configuration with reasoning effort parameters."""
    env_vars = {
        "PARAI_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "PARAI_MODEL": "gpt-4",
        "PARAI_REASONING_EFFORT": "high",
        "PARAI_REASONING_BUDGET": "1000",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = llm_config_from_env()
        assert config.reasoning_effort == ReasoningEffort.HIGH
        assert config.reasoning_budget == 1000


def test_llm_config_from_env_invalid_reasoning_effort():
    """Test error with invalid reasoning effort."""
    env_vars = {
        "PARAI_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "PARAI_MODEL": "gpt-4",
        "PARAI_REASONING_EFFORT": "invalid",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError, match="PARAI_REASONING_EFFORT must be one of"):
            llm_config_from_env()


def test_llm_config_from_env_zero_reasoning_budget():
    """Test reasoning budget set to zero becomes None."""
    env_vars = {
        "PARAI_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "PARAI_MODEL": "gpt-4",
        "PARAI_REASONING_BUDGET": "0",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = llm_config_from_env()
        assert config.reasoning_budget is None


def test_llm_config_from_env_negative_num_ctx():
    """Test negative num_ctx becomes 0."""
    env_vars = {
        "PARAI_AI_PROVIDER": "OpenAI",
        "OPENAI_API_KEY": "test-key",
        "PARAI_MODEL": "gpt-4",
        "PARAI_NUM_CTX": "-100",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = llm_config_from_env()
        assert config.num_ctx == 0


def test_llm_config_from_env_bedrock_no_key():
    """Test Bedrock provider doesn't require API key."""
    with patch.dict(os.environ, {"PARAI_AI_PROVIDER": "Bedrock", "PARAI_MODEL": "claude-3"}, clear=True):
        config = llm_config_from_env()
        assert config.provider == LlmProvider.BEDROCK


def test_get_model_context_size_known_models():
    """Test context size retrieval for known models."""
    # Test that the function returns reasonable context sizes
    gpt4_size = _get_model_context_size("gpt-4")
    assert gpt4_size > 0, "GPT-4 should have a positive context size"

    claude_size = _get_model_context_size("claude-3-sonnet")
    assert claude_size > 0, "Claude should have a positive context size"

    gemini_size = _get_model_context_size("gemini-1.5-pro")
    assert gemini_size > 0, "Gemini should have a positive context size"


def test_get_model_context_size_partial_match():
    """Test context size with partial model name matches."""
    assert _get_model_context_size("custom-gpt-4-model") == 8192
    assert _get_model_context_size("my-claude-3-sonnet-variant") == 200000


def test_get_model_context_size_unknown_model():
    """Test context size for unknown model returns default."""
    assert _get_model_context_size("unknown-model") == 8192


@patch("par_ai_core.llm_utils.get_model_metadata")
def test_get_model_context_size_from_metadata(mock_get_metadata):
    """Test context size retrieval from model metadata."""
    mock_model_info = MagicMock()
    mock_model_info.max_input_tokens = 16000
    mock_get_metadata.return_value = mock_model_info

    result = _get_model_context_size("custom-model")
    assert result == 16000


@patch("par_ai_core.llm_utils.get_model_metadata")
def test_get_model_context_size_metadata_exception(mock_get_metadata):
    """Test context size fallback when metadata lookup fails."""
    mock_get_metadata.side_effect = Exception("API error")

    result = _get_model_context_size("unknown-model")
    assert result == 8192


def test_estimate_tokens_gpt_model():
    """Test token estimation for GPT models."""
    with patch("tiktoken.encoding_for_model") as mock_encoding:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.return_value = mock_encoder

        result = _estimate_tokens("test text", "gpt-4")
        assert result == 5
        mock_encoding.assert_called_with("gpt-4")


def test_estimate_tokens_claude_model():
    """Test token estimation for Claude models."""
    with patch("tiktoken.encoding_for_model") as mock_encoding:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_encoder

        result = _estimate_tokens("test", "claude-3-sonnet")
        assert result == 3
        mock_encoding.assert_called_with("gpt-4")  # Claude uses GPT-4 tokenizer


def test_estimate_tokens_other_model():
    """Test token estimation for other models."""
    with patch("tiktoken.encoding_for_model") as mock_encoding:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2]
        mock_encoding.return_value = mock_encoder

        result = _estimate_tokens("hi", "gemini-pro")
        assert result == 2
        mock_encoding.assert_called_with("gpt-4")  # Default to GPT-4


def test_estimate_tokens_fallback():
    """Test token estimation fallback when encoding fails."""
    with patch("tiktoken.encoding_for_model", side_effect=Exception("Encoding error")):
        result = _estimate_tokens("test text here", "gpt-4")
        # Fallback: len(text) // 4 = 14 // 4 = 3
        assert result == 3


def test_chunk_text_small_content():
    """Test chunking when content fits in one chunk."""
    text = "Short text"
    chunks = _chunk_text(text, 1000, "gpt-4")
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_by_paragraphs():
    """Test chunking by paragraphs."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    # Mock token estimation to force chunking
    with patch("par_ai_core.llm_utils._estimate_tokens") as mock_estimate:
        # First call (full text): too many tokens
        # Subsequent calls for chunks: within limits
        mock_estimate.side_effect = [100, 30, 50, 30, 30, 30]

        chunks = _chunk_text(text, 50, "gpt-4")
        assert len(chunks) >= 1


def test_chunk_text_by_sentences():
    """Test chunking by sentences when paragraphs are too large."""
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."

    with patch("par_ai_core.llm_utils._estimate_tokens") as mock_estimate:
        # Simulate paragraph too large, but sentences fit
        mock_estimate.side_effect = [100, 100, 25, 50, 25, 25, 25, 25]

        chunks = _chunk_text(text, 50, "gpt-4")
        assert len(chunks) >= 1


def test_summarize_content_chunked():
    """Test summarization with content that needs chunking."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.name = "gpt-4"
    mock_llm.invoke.side_effect = [
        AIMessage(content="Chunk 1 summary"),
        AIMessage(content="Chunk 2 summary"),
        AIMessage(content="Final combined summary"),
    ]

    long_content = "Very long content " * 1000

    with patch("par_ai_core.llm_utils._estimate_tokens") as mock_estimate:
        with patch("par_ai_core.llm_utils._chunk_text") as mock_chunk:
            # Simulate content too large
            mock_estimate.return_value = 50000
            mock_chunk.return_value = ["Chunk 1 content", "Chunk 2 content"]

            result = summarize_content(long_content, mock_llm)

            assert result == "Final combined summary"
            assert mock_llm.invoke.call_count == 3  # 2 chunks + final summary


def test_summarize_content_no_model_name():
    """Test summarization when LLM has no name attribute."""
    mock_llm = MagicMock(spec=BaseChatModel)
    # Use getattr with defaults to avoid AttributeError
    mock_llm.configure_mock(**{"name": None, "model_name": None})
    mock_llm.invoke.return_value = AIMessage(content="Summary without model name")

    with patch("par_ai_core.llm_utils.getattr") as mock_getattr:
        # Mock getattr to return empty string for both name and model_name
        def getattr_side_effect(obj, attr, default=None):
            if attr in ("name", "model_name"):
                return ""
            return default

        mock_getattr.side_effect = getattr_side_effect

        result = summarize_content("Test content", mock_llm)
        assert result == "Summary without model name"


def test_summarize_content_model_name_attribute():
    """Test summarization with model_name attribute instead of name."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.configure_mock(**{"name": None, "model_name": "test-model"})
    mock_llm.invoke.return_value = AIMessage(content="Summary with model_name")

    with patch("par_ai_core.llm_utils.getattr") as mock_getattr:
        # Mock getattr to return model_name but not name
        def getattr_side_effect(obj, attr, default=None):
            if attr == "model_name":
                return "test-model"
            elif attr == "name":
                return ""
            return default

        mock_getattr.side_effect = getattr_side_effect

        result = summarize_content("Test content", mock_llm)
        assert result == "Summary with model_name"
