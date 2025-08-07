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
from par_ai_core.llm_providers import provider_env_key_names


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
            {provider_env_key_names[LlmProvider.GITHUB]: "test-github-key"},
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


def test_llm_config_openai_api_key() -> None:
    """Test OpenAI provider API key handling."""
    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="openai-model",
        mode=LlmMode.CHAT,
    )

    # Mock environment variables and ChatOpenAI
    with (
        patch.dict(
            os.environ,
            {provider_env_key_names[LlmProvider.OPENAI]: "test-openai-key"},
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
        assert call_args["api_key"].get_secret_value() == "test-openai-key"


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
            # default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            timeout=None,
            top_k=None,
            top_p=None,
            max_tokens_to_sample=2048,
            disable_streaming=False,
            thinking=None,
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


def test_provider_wrong_builder_validation_errors() -> None:
    """Test wrong provider validation for each builder method."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="test")

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but OLLAMA requested"):
        config._build_ollama_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but GROQ requested"):
        config._build_groq_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but XAI requested"):
        config._build_xai_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but OPENROUTER requested"):
        config._build_openrouter_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but DEEPSEEK requested"):
        config._build_deepseek_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but ANTHROPIC requested"):
        config._build_anthropic_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but GOOGLE requested"):
        config._build_google_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but BEDROCK requested"):
        config._build_bedrock_llm()

    with pytest.raises(ValueError, match="LLM provider is 'OpenAI' but LITELLM requested"):
        config._build_litellm_llm()


def test_ollama_url_determination_failure() -> None:
    """Test OLLAMA URL determination failure."""
    config = LlmConfig(provider=LlmProvider.OLLAMA, model_name="test")

    with patch.dict(os.environ, {}, clear=True):
        with patch("par_ai_core.llm_config.OLLAMA_HOST", None):
            with patch("par_ai_core.llm_config.provider_base_urls", {LlmProvider.OLLAMA: None}):
                with pytest.raises(ValueError, match="Could not determine OLLAMA URL"):
                    config._build_ollama_llm()


def test_ollama_with_authentication() -> None:
    """Test OLLAMA authentication handling."""
    config = LlmConfig(provider=LlmProvider.OLLAMA, model_name="test", base_url="http://user:pass@localhost:11434")

    with patch("langchain_ollama.ChatOllama") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance
        config.build_chat_model()

        call_args = mock_chat.call_args[1]
        assert "client_kwargs" in call_args
        assert "auth" in call_args["client_kwargs"]


def test_azure_openai_api_key_fallback() -> None:
    """Test Azure OpenAI API key fallback logic."""
    config = LlmConfig(provider=LlmProvider.AZURE, model_name="gpt-4", mode=LlmMode.CHAT)

    with patch.dict(os.environ, {provider_env_key_names[LlmProvider.OPENAI]: "fallback-key"}, clear=True):
        with patch("langchain_openai.AzureChatOpenAI") as mock_azure:
            mock_instance = MagicMock(spec=BaseChatModel)
            mock_azure.return_value = mock_instance
            config.build_chat_model()
            call_args = mock_azure.call_args[1]
            assert call_args["api_key"].get_secret_value() == "fallback-key"


def test_azure_openai_all_modes() -> None:
    """Test Azure OpenAI BASE and EMBEDDINGS modes."""
    # Test BASE mode
    config = LlmConfig(provider=LlmProvider.AZURE, model_name="gpt-4", mode=LlmMode.BASE)
    with patch("langchain_openai.AzureOpenAI") as mock_azure:
        mock_instance = MagicMock(spec=BaseLanguageModel)
        mock_azure.return_value = mock_instance
        config.build_llm_model()
        mock_azure.assert_called_once()

    # Test EMBEDDINGS mode
    config.mode = LlmMode.EMBEDDINGS
    with patch("langchain_openai.AzureOpenAIEmbeddings") as mock_embeddings:
        mock_instance = MagicMock(spec=Embeddings)
        mock_embeddings.return_value = mock_instance
        config.build_embeddings()
        mock_embeddings.assert_called_once()


def test_litellm_chat_mode() -> None:
    """Test LiteLLM CHAT mode implementation."""
    config = LlmConfig(provider=LlmProvider.LITELLM, model_name="gpt-4", mode=LlmMode.CHAT)
    with patch("langchain_community.chat_models.ChatLiteLLM") as mock_litellm:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_litellm.return_value = mock_instance
        config.build_chat_model()
        mock_litellm.assert_called_once()


def test_anthropic_reasoning_budget_validation() -> None:
    """Test Anthropic reasoning budget validation."""

    # Test budget too low
    config = LlmConfig(
        provider=LlmProvider.ANTHROPIC,
        model_name="claude-3",
        reasoning_budget=500,  # Less than 1024
    )
    with pytest.raises(ValueError, match="Reasoning budget must be at least 1024 tokens"):
        config.build_chat_model()

    # Test valid budget
    config.reasoning_budget = 2048
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_anthropic.return_value = mock_instance
        config.build_chat_model()
        call_args = mock_anthropic.call_args[1]
        assert call_args["thinking"]["budget_tokens"] == 2048


def test_bedrock_base_and_embeddings_modes() -> None:
    """Test Bedrock BASE and EMBEDDINGS modes."""
    with patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        # Test BASE mode
        config = LlmConfig(provider=LlmProvider.BEDROCK, model_name="test", mode=LlmMode.BASE)
        with patch("langchain_aws.BedrockLLM") as mock_bedrock:
            mock_instance = MagicMock(spec=BaseLanguageModel)
            mock_bedrock.return_value = mock_instance
            config.build_llm_model()
            mock_bedrock.assert_called_once()

        # Test EMBEDDINGS mode
        config.mode = LlmMode.EMBEDDINGS
        with patch("langchain_aws.BedrockEmbeddings") as mock_embeddings:
            mock_instance = MagicMock(spec=Embeddings)
            mock_embeddings.return_value = mock_instance
            config.build_embeddings()
            mock_embeddings.assert_called_once()


def test_mistral_base_mode_error_and_embeddings() -> None:
    """Test Mistral BASE mode error and EMBEDDINGS mode."""
    # Test BASE mode error
    config = LlmConfig(provider=LlmProvider.MISTRAL, model_name="mistral-7b", mode=LlmMode.BASE)
    with pytest.raises(ValueError, match="Mistral provider does not support mode Base"):
        config.build_llm_model()

    # Test EMBEDDINGS mode
    config.mode = LlmMode.EMBEDDINGS
    with patch("langchain_mistralai.MistralAIEmbeddings") as mock_embeddings:
        mock_instance = MagicMock(spec=Embeddings)
        mock_embeddings.return_value = mock_instance
        config.build_embeddings()
        mock_embeddings.assert_called_once()


def test_o1_o3_model_temperature_adjustment_base_mode() -> None:
    """Test O1/O3 model temperature adjustment for base models."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="o1-preview", temperature=0.7, mode=LlmMode.BASE)
    with patch("langchain_openai.OpenAI") as mock_openai:
        mock_instance = MagicMock(spec=BaseLanguageModel)
        mock_openai.return_value = mock_instance
        config.build_llm_model()
        assert config.temperature == 1


def test_invalid_llm_type_exceptions() -> None:
    """Test invalid LLM type exceptions."""
    # Create a mock that's not a valid LLM type
    invalid_mock = MagicMock()
    invalid_mock.__class__.__name__ = "InvalidType"

    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="test")

    # Test wrong type for build_llm_model
    with patch.object(config, "_build_llm", return_value=invalid_mock):
        with pytest.raises(ValueError, match="Invalid LLM type returned for base mode"):
            config.build_llm_model()

    # Test wrong type for build_chat_model
    with patch.object(config, "_build_llm", return_value=invalid_mock):
        with pytest.raises(ValueError, match="Invalid LLM type returned for chat mode"):
            config.build_chat_model()

    # Test wrong type for build_embeddings
    with patch.object(config, "_build_llm", return_value=invalid_mock):
        with pytest.raises(ValueError, match="does not support embeddings"):
            config.build_embeddings()


def test_api_key_checking_method() -> None:
    """Test API key checking method."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="test")
    with patch("par_ai_core.llm_config.is_provider_api_key_set", return_value=True) as mock_check:
        result = config.is_api_key_set()
        assert result is True
        mock_check.assert_called_once_with(LlmProvider.OPENAI)


def test_set_env_with_all_optional_parameters() -> None:
    """Test set_env with all optional parameters."""
    from par_ai_core.llm_config import ReasoningEffort

    config = LlmConfig(
        provider=LlmProvider.OPENAI,
        model_name="gpt-4",
        user_agent_appid="test-app",
        num_ctx=2048,
        num_predict=100,
        repeat_last_n=64,
        repeat_penalty=1.1,
        mirostat=1,
        mirostat_eta=0.1,
        mirostat_tau=5.0,
        tfs_z=1.0,
        top_k=40,
        top_p=0.9,
        seed=42,
        timeout=30,
        reasoning_effort=ReasoningEffort.HIGH,
        reasoning_budget=2048,
    )

    config.set_env()

    # Verify all optional environment variables are set
    assert os.environ["PARAI_USER_AGENT_APPID"] == "test-app"
    assert os.environ["PARAI_NUM_CTX"] == "2048"
    assert os.environ["PARAI_NUM_PREDICT"] == "100"
    assert os.environ["PARAI_REPEAT_LAST_N"] == "64"
    assert os.environ["PARAI_REPEAT_PENALTY"] == "1.1"
    assert os.environ["PARAI_MIROSTAT"] == "1"
    assert os.environ["PARAI_MIROSTAT_ETA"] == "0.1"
    assert os.environ["PARAI_MIROSTAT_TAU"] == "5.0"
    assert os.environ["PARAI_TFS_Z"] == "1.0"
    assert os.environ["PARAI_TOP_K"] == "40"
    assert os.environ["PARAI_TOP_P"] == "0.9"
    assert os.environ["PARAI_SEED"] == "42"
    assert os.environ["PARAI_TIMEOUT"] == "30"
    assert os.environ["PARAI_REASONING_EFFORT"] == "high"
    assert os.environ["PARAI_REASONING_BUDGET"] == "2048"


def test_llm_run_manager_no_matches() -> None:
    """Test LlmRunManager when no matches are found."""
    manager = LlmRunManager()

    # Test get_runnable_config_by_model with no matches
    result = manager.get_runnable_config_by_model("non-existent-model")
    assert result is None

    # Test get_runnable_config_by_model with empty string
    result = manager.get_runnable_config_by_model("")
    assert result is None


def test_from_json_with_missing_fields() -> None:
    """Test from_json with missing optional fields."""
    minimal_data = {
        "class_name": "LlmConfig",
        "provider": "OpenAI",
        "model_name": "gpt-4",
        "mode": "Chat",  # Add required mode field
    }
    config = LlmConfig.from_json(minimal_data)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.8  # Default value


def test_from_json_with_extra_fields() -> None:
    """Test from_json ignores extra fields not in dataclass."""
    data_with_extras = {
        "class_name": "LlmConfig",
        "provider": "OpenAI",
        "model_name": "gpt-4",
        "mode": "Chat",  # Add required mode field
        "extra_field": "should_be_ignored",
        "another_extra": 123,
    }
    config = LlmConfig.from_json(data_with_extras)
    assert config.provider == LlmProvider.OPENAI
    assert config.model_name == "gpt-4"
    assert not hasattr(config, "extra_field")


def test_invalid_mode_for_various_providers() -> None:
    """Test invalid mode errors for various providers."""
    # Test Groq with BASE mode
    config = LlmConfig(provider=LlmProvider.GROQ, model_name="test", mode=LlmMode.BASE)
    with pytest.raises(ValueError, match="Groq provider does not support mode Base"):
        config.build_llm_model()

    # Test XAI with BASE mode
    config = LlmConfig(provider=LlmProvider.XAI, model_name="test", mode=LlmMode.BASE)
    with pytest.raises(ValueError, match="XAI provider does not support mode Base"):
        config.build_llm_model()

    # Test XAI with EMBEDDINGS mode
    config = LlmConfig(provider=LlmProvider.XAI, model_name="test", mode=LlmMode.EMBEDDINGS)
    with pytest.raises(ValueError, match="XAI provider does not support mode Embeddings"):
        config.build_embeddings()


def test_mistral_provider_dispatch() -> None:
    """Test Mistral provider dispatch."""
    config = LlmConfig(provider=LlmProvider.MISTRAL, model_name="mistral-7b", mode=LlmMode.CHAT)
    with patch.object(config, "_build_mistral_llm") as mock_mistral:
        mock_mistral.return_value = MagicMock(spec=BaseChatModel)
        config._build_llm()
        mock_mistral.assert_called_once()


def test_deepseek_setup() -> None:
    """Test Deepseek configuration setup."""
    config = LlmConfig(provider=LlmProvider.DEEPSEEK, model_name="deepseek-chat", mode=LlmMode.CHAT)

    # Test that the provider dispatch works
    with patch.object(config, "_build_deepseek_llm") as mock_deepseek:
        mock_deepseek.return_value = MagicMock(spec=BaseChatModel)
        config._build_llm()
        mock_deepseek.assert_called_once()


def test_openrouter_setup() -> None:
    """Test OpenRouter configuration setup."""
    config = LlmConfig(provider=LlmProvider.OPENROUTER, model_name="openrouter-model", mode=LlmMode.CHAT)

    with patch("langchain_openai.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_chat.return_value = mock_instance
        config.build_chat_model()
        mock_chat.assert_called_once()
