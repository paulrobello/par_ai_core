"""Tests for llm_config module."""

from __future__ import annotations

import os
from unittest.mock import ANY, MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, BaseLanguageModel

from par_ai_core.llm_config import (
    LlmConfig,
    LlmMode,
    LlmProvider,
    LlmRunManager,
    llm_run_manager,
)


def test_llm_config_init() -> None:
    """Test LlmConfig initialization."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7,
        mode=LlmMode.CHAT,
    )
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.mode == LlmMode.CHAT


def test_llm_config_to_json() -> None:
    """Test LlmConfig to_json method."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
    )
    json_data = config.to_json()
    assert json_data["class_name"] == "LlmConfig"
    assert json_data["provider"] == LlmProvider.OPENAI
    assert json_data["model_name"] == "gpt-4"


def test_llm_config_from_json() -> None:
    """Test LlmConfig from_json method."""
    # Test with enum provider
    json_data = {
        "class_name": "LlmConfig",
        "provider": LlmProvider.OPENAI,
        "model_name": "gpt-4",
        "mode": LlmMode.CHAT,
        "temperature": 0.7,
        "streaming": True,
        "base_url": None,
        "timeout": None,
        "user_agent_appid": None,
        "num_ctx": None,
        "num_predict": None,
        "repeat_last_n": None,
        "repeat_penalty": None,
        "mirostat": None,
        "mirostat_eta": None,
        "mirostat_tau": None,
        "tfs_z": None,
        "top_k": None,
        "top_p": None,
        "seed": None,
        "env_prefix": "PARAI",
    }
    config = LlmConfig.from_json(json_data)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7

    # Test with string provider
    json_data["provider"] = "OpenAI"  # String instead of enum
    config = LlmConfig.from_json(json_data)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7

    # Test with string mode
    json_data["mode"] = "Chat"  # String instead of enum
    config = LlmConfig.from_json(json_data)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.mode == LlmMode.CHAT


def test_llm_config_clone() -> None:
    """Test LlmConfig clone method."""
    original = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7,
    )
    cloned = original.clone()
    assert cloned.provider == original.provider
    assert cloned.model_name == original.model_name
    assert cloned.temperature == original.temperature
    assert cloned is not original


def test_llm_config_gen_runnable_config() -> None:
    """Test LlmConfig gen_runnable_config method."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
    )
    runnable_config = config.gen_runnable_config()
    assert "metadata" in runnable_config
    assert "config_id" in runnable_config["metadata"]
    assert "tags" in runnable_config


@pytest.mark.parametrize(
    "provider,mode,expected_type",
    [
        (LlmProvider.OPENAI, LlmMode.BASE, BaseLanguageModel),
        (LlmProvider.OPENAI, LlmMode.CHAT, BaseChatModel),
        (LlmProvider.OPENAI, LlmMode.EMBEDDINGS, Embeddings),
    ],
)
def test_llm_config_build_models(provider: LlmProvider, mode: LlmMode, expected_type: type) -> None:
    """Test LlmConfig model building methods."""
    config = LlmConfig(provider=provider, model_name="test-model", mode=mode)

    with (
        patch("langchain_openai.OpenAI") as mock_openai,
        patch("langchain_openai.ChatOpenAI") as mock_chat_openai,
        patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings,
    ):
        # Create properly typed mock instances
        if mode == LlmMode.BASE:
            mock_instance = MagicMock(spec=BaseLanguageModel)
            mock_openai.return_value = mock_instance
        elif mode == LlmMode.CHAT:
            mock_instance = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_instance
        else:
            mock_instance = MagicMock(spec=Embeddings)
            mock_embeddings.return_value = mock_instance

        if mode == LlmMode.BASE:
            result = config.build_llm_model()
        elif mode == LlmMode.CHAT:
            result = config.build_chat_model()
        else:
            result = config.build_embeddings()

        assert isinstance(result, expected_type)


def test_llm_config_set_env() -> None:
    """Test LlmConfig set_env method."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7,
        base_url="https://api.example.com",
    )
    config.set_env()
    assert os.environ["PARAI_AI_PROVIDER"] == LlmProvider.OPENAI.value
    assert os.environ["PARAI_MODEL"] == "gpt-4"
    assert os.environ["PARAI_TEMPERATURE"] == "0.7"
    assert os.environ["PARAI_AI_BASE_URL"] == "https://api.example.com"


def test_llm_run_manager() -> None:
    """Test LlmRunManager functionality."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    runnable_config = config.gen_runnable_config()
    config_id = runnable_config["metadata"]["config_id"]

    # Test registration
    llm_run_manager.register_id(runnable_config, config)

    # Test retrieval
    retrieved_config = llm_run_manager.get_config(config_id)
    assert retrieved_config is not None
    assert retrieved_config[0] == runnable_config
    assert retrieved_config[1] == config

    # Test get_runnable_config
    retrieved_runnable = llm_run_manager.get_runnable_config(config_id)
    assert retrieved_runnable == runnable_config

    # Test get_runnable_config_by_model
    model_config = llm_run_manager.get_runnable_config_by_model("gpt-4")
    assert model_config == runnable_config

    # Test get_provider_and_model
    provider_model = llm_run_manager.get_provider_and_model(config_id)
    assert provider_model == (LlmProvider.OPENAI, "gpt-4")


def test_llm_run_manager_invalid_config() -> None:
    """Test LlmRunManager with invalid configs."""
    # Test non-existent config_id
    assert llm_run_manager.get_config("non-existent") is None
    assert llm_run_manager.get_runnable_config("non-existent") is None
    assert llm_run_manager.get_provider_and_model("non-existent") is None

    # Test None inputs
    assert llm_run_manager.get_runnable_config(None) is None
    assert llm_run_manager.get_runnable_config_by_model("") is None
    assert llm_run_manager.get_provider_and_model(None) is None

    # Test invalid runnable config registration
    with pytest.raises(ValueError, match="Runnable config must have a config_id in metadata"):
        llm_run_manager.register_id({"metadata": {}}, LlmConfig(provider=LlmProvider.OPENAI, model_name="test"))


def test_llm_config_invalid_json() -> None:
    """Test LlmConfig with invalid JSON data."""
    with pytest.raises(ValueError, match="Invalid config class"):
        LlmConfig.from_json({"class_name": "InvalidClass"})


def test_llm_config_invalid_provider_mode() -> None:
    """Test LlmConfig with invalid provider/mode combinations."""
    config = LlmConfig(provider=LlmProvider.ANTHROPIC, model_name="test", mode=LlmMode.BASE)
    with pytest.raises(ValueError, match="provider does not support mode"):
        config.build_llm_model()

    config = LlmConfig(provider=LlmProvider.ANTHROPIC, model_name="test", mode=LlmMode.EMBEDDINGS)
    with pytest.raises(ValueError, match="provider does not support mode"):
        config.build_embeddings()


def test_llm_config_get_runnable_config_by_llm_config() -> None:
    """Test get_runnable_config_by_llm_config method."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    runnable_config = config.gen_runnable_config()
    llm_run_manager.register_id(runnable_config, config)

    # Test with matching config
    result = llm_run_manager.get_runnable_config_by_llm_config(config)
    assert result is not None
    assert result["metadata"]["model_name"] == runnable_config["metadata"]["model_name"]
    assert result["metadata"]["provider"] == runnable_config["metadata"]["provider"]

    # Test with non-matching config
    other_config = LlmConfig(provider=LlmProvider.OPENAI, model_name="different-model")
    result = llm_run_manager.get_runnable_config_by_llm_config(other_config)
    assert result is None

    # Test with None input
    assert llm_run_manager.get_runnable_config_by_llm_config(None) is None


@pytest.mark.parametrize(
    "provider,mode,expected_error",
    [
        (LlmProvider.OLLAMA, LlmMode.CHAT, "LLM provider is 'Ollama' but OPENAI requested."),
        (LlmProvider.GROQ, LlmMode.BASE, "Groq provider does not support mode Base"),
        (LlmProvider.XAI, LlmMode.EMBEDDINGS, "XAI provider does not support mode Embeddings"),
        (LlmProvider.OPENROUTER, LlmMode.EMBEDDINGS, "OpenRouter provider does not support mode Embeddings"),
        (LlmProvider.DEEPSEEK, LlmMode.EMBEDDINGS, "Deepseek provider does not support mode Embeddings"),
        (LlmProvider.LITELLM, LlmMode.BASE, "LiteLLM provider does not support mode Base"),
        (LlmProvider.XAI, LlmMode.EMBEDDINGS, "XAI provider does not support mode Embeddings"),
        (LlmProvider.OPENROUTER, LlmMode.EMBEDDINGS, "OpenRouter provider does not support mode Embeddings"),
        (LlmProvider.XAI, LlmMode.EMBEDDINGS, "XAI provider does not support mode Embeddings"),
        (LlmProvider.DEEPSEEK, LlmMode.EMBEDDINGS, "Deepseek provider does not support mode Embeddings"),
        (LlmProvider.LITELLM, LlmMode.EMBEDDINGS, "LiteLLM provider does not support mode Embeddings"),
        (LlmProvider.ANTHROPIC, LlmMode.EMBEDDINGS, "Anthropic provider does not support mode Embeddings"),
    ],
)
def test_llm_config_provider_errors(provider: LlmProvider, mode: LlmMode, expected_error: str) -> None:
    """Test error cases for different providers and modes."""
    config = LlmConfig(provider=provider, model_name="test-model", mode=mode)

    with pytest.raises(ValueError, match=expected_error):
        if mode == LlmMode.BASE:
            config.build_llm_model()
        elif mode == LlmMode.CHAT:
            config._build_openai_compat_llm()
        else:
            config.build_embeddings()


def test_llm_config_o1_model_adjustments() -> None:
    """Test automatic adjustments for O1 models."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="o1-test", temperature=0.7, streaming=True)

    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance

        mock_chat_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_chat_instance
        mock_chat_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_chat_instance
        config.build_chat_model()
        assert config.temperature == 1
        assert config.streaming is False


def test_llm_config_bedrock_setup() -> None:
    """Test Bedrock LLM configuration setup."""
    config = LlmConfig(provider=LlmProvider.BEDROCK, model_name="test-model", timeout=30, user_agent_appid="test-app")

    with patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        config.build_chat_model()

        mock_session.assert_called_once()
        mock_session.return_value.client.assert_called_once_with(
            "bedrock-runtime", config=ANY, endpoint_url=config.base_url
        )


def test_llm_config_google_setup() -> None:
    """Test Google AI configuration setup."""
    config = LlmConfig(provider=LlmProvider.GEMINI, model_name="test-model")

    with (
        patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_chat,
        patch("langchain_google_genai.GoogleGenerativeAI") as mock_base,
        patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as mock_embeddings,
    ):
        # Test chat mode
        config.mode = LlmMode.CHAT
        # Test chat mode
        mock_chat_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_chat_instance
        config.build_chat_model()
        mock_chat.assert_called_once()

        # Test base mode
        config.mode = LlmMode.BASE
        mock_base_instance = MagicMock(spec=BaseLanguageModel)
        mock_base.return_value = mock_base_instance
        config.build_llm_model()
        mock_base.assert_called_once()

        # Test embeddings mode
        config.mode = LlmMode.EMBEDDINGS
        mock_embeddings_instance = MagicMock(spec=Embeddings)
        mock_embeddings.return_value = mock_embeddings_instance
        config.build_embeddings()
        mock_embeddings.assert_called_once()


def test_llm_config_github_api_key() -> None:
    """Test GitHub provider API key handling."""
    config = LlmConfig(
        provider=LlmProvider.GITHUB,
        model_name="github-model",
        mode=LlmMode.CHAT,
    )

    # Mock environment variables and ChatOpenAI
    with (
        patch.dict(
            os.environ,
            {"GITHUB_API_KEY": "test-github-key"},
            clear=True,
        ),
        patch("langchain_openai.ChatOpenAI") as mock_chat,
    ):
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance

        config.build_chat_model()

        # Verify ChatOpenAI was called with GitHub API key
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args[1]
        assert call_args["api_key"].get_secret_value() == "test-github-key"


def test_llm_config_anthropic_setup() -> None:
    """Test Anthropic configuration setup."""
    config = LlmConfig(
        provider=LlmProvider.ANTHROPIC,
        model_name="claude-3",
    )

    with patch("langchain_anthropic.ChatAnthropic") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance
        config.build_chat_model()
        mock_chat.assert_called_once_with(
            model="claude-3",
            temperature=0.8,
            streaming=True,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            timeout=None,
            top_k=None,
            top_p=None,
            max_tokens_to_sample=1024,
            disable_streaming=False,
            max_tokens=None,
        )


def test_llm_config_groq_setup() -> None:
    """Test Groq configuration setup."""
    config = LlmConfig(provider=LlmProvider.GROQ, model_name="mixtral-8x7b", mode=LlmMode.CHAT)

    with patch("langchain_groq.ChatGroq") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance
        config.build_chat_model()
        mock_chat.assert_called_once()


def test_llm_config_xai_setup() -> None:
    """Test XAI configuration setup."""
    config = LlmConfig(provider=LlmProvider.XAI, model_name="xai-model", mode=LlmMode.CHAT)

    with patch("langchain_xai.ChatXAI") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance
        config.build_chat_model()
        mock_chat.assert_called_once()


def test_llm_config_ollama_setup() -> None:
    """Test Ollama configuration setup."""
    config = LlmConfig(provider=LlmProvider.OLLAMA, model_name="llama2")

    with (
        patch("langchain_ollama.ChatOllama") as mock_chat,
        patch("langchain_ollama.OllamaLLM") as mock_base,
        patch("langchain_ollama.OllamaEmbeddings") as mock_embeddings,
    ):
        # Test chat mode
        config.mode = LlmMode.CHAT
        mock_chat_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_chat_instance
        config.build_chat_model()
        mock_chat.assert_called_once()

        # Test base mode
        config.mode = LlmMode.BASE
        mock_base_instance = MagicMock(spec=BaseLanguageModel)
        mock_base.return_value = mock_base_instance
        config.build_llm_model()
        mock_base.assert_called_once()

        # Test embeddings mode
        config.mode = LlmMode.EMBEDDINGS
        mock_embeddings_instance = MagicMock(spec=Embeddings)
        mock_embeddings.return_value = mock_embeddings_instance
        config.build_embeddings()
        mock_embeddings.assert_called_once()


def test_llm_config_invalid_mode_provider_combinations() -> None:
    """Test invalid mode/provider combinations."""
    # Test wrong provider for Ollama build
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="test", mode=LlmMode.CHAT)
    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but OLLAMA requested."):
        config._build_ollama_llm()

    # Test XAI with base mode
    config = LlmConfig(provider=LlmProvider.XAI, model_name="test", mode=LlmMode.BASE)
    with pytest.raises(ValueError, match="provider does not support mode"):
        config.build_llm_model()

    # Test Groq with embeddings mode
    config = LlmConfig(provider=LlmProvider.GROQ, model_name="test", mode=LlmMode.EMBEDDINGS)
    with pytest.raises(ValueError, match="provider does not support mode"):
        config.build_embeddings()

    # Test invalid provider
    config = LlmConfig(provider="INVALID", model_name="test")  # type: ignore
    with pytest.raises(ValueError, match="Invalid LLM provider"):
        config._build_llm()


def test_llm_config_environment_variables() -> None:
    """Test environment variable handling."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.5,
        num_ctx=2048,
        top_p=0.9,
        seed=42,
        timeout=30,
    )

    config.set_env()

    assert os.environ["PARAI_AI_PROVIDER"] == "OpenAI"
    assert os.environ["PARAI_MODEL"] == "gpt-4"
    assert os.environ["PARAI_TEMPERATURE"] == "0.5"
    assert os.environ["PARAI_NUM_CTX"] == "2048"
    assert os.environ["PARAI_TOP_P"] == "0.9"
    assert os.environ["PARAI_SEED"] == "42"
    assert os.environ["PARAI_TIMEOUT"] == "30"


def test_llm_run_manager_thread_safety() -> None:
    """Test thread safety of LlmRunManager."""
    import threading
    import time

    manager = LlmRunManager()
    configs = []
    errors = []

    def worker(idx: int) -> None:
        try:
            config = LlmConfig(provider=LlmProvider.OPENAI, model_name=f"model-{idx}")
            runnable_config = config.gen_runnable_config()
            manager.register_id(runnable_config, config)
            configs.append((runnable_config, config))
            time.sleep(0.01)  # Simulate some work
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads encountered errors: {errors}"
    assert len(configs) == 10, "Not all configs were registered"

    # Verify all configs can be retrieved
    for runnable_config, config in configs:
        config_id = runnable_config["metadata"]["config_id"]
        retrieved = manager.get_config(config_id)
        assert retrieved is not None
        assert retrieved[1].model_name == config.model_name
