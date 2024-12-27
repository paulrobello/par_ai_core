"""Tests for the provider callback info module."""

from uuid import UUID
import pytest
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from par_ai_core.llm_config import LlmConfig
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.provider_cb_info import ParAICallbackHandler, get_parai_callback
from par_ai_core.pricing_lookup import PricingDisplay


def test_parai_callback_handler_init():
    """Test initialization of ParAICallbackHandler."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    handler = ParAICallbackHandler(llm_config=config, show_prompts=True, show_end=True, show_tool_calls=True)

    assert handler.llm_config == config
    assert handler.show_prompts is True
    assert handler.show_end is True
    assert handler.show_tool_calls is True
    assert handler.usage_metadata == {}


def test_usage_metadata_thread_safety():
    """Test thread-safe access to usage metadata."""
    handler = ParAICallbackHandler()

    # Test that we get a deep copy
    metadata = handler.usage_metadata
    metadata["test"] = {"count": 1}
    assert "test" not in handler.usage_metadata


def test_llm_end_chat_completion():
    """Test handling of chat completion results."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    handler = ParAICallbackHandler(llm_config=config)

    # Mock chat completion response with token usage in additional_kwargs
    message = AIMessage(content="Test response")
    message.additional_kwargs["token_usage"] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    message.usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    generation = ChatGeneration(message=message)
    response = LLMResult(generations=[[generation]])

    handler.on_llm_end(response)

    # Verify token usage was properly accumulated
    metadata = handler.usage_metadata["gpt-4"]
    assert metadata["input_tokens"] == 20
    assert metadata["output_tokens"] == 40
    assert metadata["total_tokens"] == 60
    assert metadata["successful_requests"] == 1


def test_parai_callback_context_manager():
    """Test the get_parai_callback context manager."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")

    with get_parai_callback(
        config, show_prompts=True, show_end=True, show_pricing=PricingDisplay.PRICE, show_tool_calls=True
    ) as cb:
        assert isinstance(cb, ParAICallbackHandler)
        assert cb.llm_config == config
        assert cb.show_prompts is True
        assert cb.show_end is True
        assert cb.show_tool_calls is True


def test_copy_operations():
    """Test copy and deepcopy operations."""
    handler = ParAICallbackHandler()

    # Test copy
    copy_handler = handler.__copy__()
    assert copy_handler is handler  # Should return self

    # Test deepcopy
    deepcopy_handler = handler.__deepcopy__({})
    assert deepcopy_handler is handler  # Should return self


def test_on_llm_start():
    """Test the on_llm_start callback."""
    handler = ParAICallbackHandler(show_prompts=True)
    prompts = ["Test prompt"]
    handler.on_llm_start({"name": "test"}, prompts)

    # Test without show_prompts
    handler = ParAICallbackHandler(show_prompts=False)
    handler.on_llm_start({"name": "test"}, prompts)


def test_on_tool_start():
    """Test the on_tool_start callback."""
    handler = ParAICallbackHandler(show_tool_calls=True)
    inputs = {"query": "test"}
    handler.on_tool_start(
        {"name": "test_tool"},
        "test input",
        run_id=UUID("12345678-1234-5678-1234-567812345678"),
        inputs=inputs,
    )

    # Test without show_tool_calls
    handler = ParAICallbackHandler(show_tool_calls=False)
    handler.on_tool_start(
        {"name": "test_tool"},
        "test input",
        run_id=UUID("12345678-1234-5678-1234-567812345678"),
        inputs=inputs,
    )


def test_llm_end_no_config():
    """Test handling LLM end without config."""
    handler = ParAICallbackHandler()
    message = AIMessage(content="Test response")
    generation = ChatGeneration(message=message)
    response = LLMResult(generations=[[generation]])

    handler.on_llm_end(response)


def test_llm_end_empty_response():
    """Test handling empty LLM response."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    handler = ParAICallbackHandler(llm_config=config)

    # Test with empty generations
    response = LLMResult(generations=[[]])
    handler.on_llm_end(response)

    # Test with no generations
    response = LLMResult(generations=[])
    handler.on_llm_end(response)


def test_llm_end_non_chat_completion():
    """Test handling non-chat completion results."""
    config = LlmConfig(provider=LlmProvider.OPENAI, model_name="gpt-4")
    handler = ParAICallbackHandler(llm_config=config)

    # Test with llm_output containing token usage
    response = LLMResult(
        generations=[[]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        },
    )
    handler.on_llm_end(response)

    metadata = handler.usage_metadata["gpt-4"]
    assert metadata["successful_requests"] == 1


def test_always_verbose():
    """Test always_verbose property."""
    handler = ParAICallbackHandler()
    assert handler.always_verbose is True


def test_is_lc_serializable():
    """Test is_lc_serializable class method."""
    assert ParAICallbackHandler.is_lc_serializable() is False


def test_repr():
    """Test string representation."""
    handler = ParAICallbackHandler()
    assert repr(handler) == "{}"
