# -*- coding: utf-8 -*-
"""Tests for the main module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from par_ai_core.__main__ import main
from par_ai_core.llm_providers import LlmProvider


@pytest.fixture
def mock_console():
    """Mock console fixture."""
    return MagicMock()


@pytest.fixture
def mock_chat_model():
    """Mock chat model fixture."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Why did the dog go to the vet? Because he needed a barking lot!")
    mock.name = "test-model"
    return mock


def test_main_success(mock_console, mock_chat_model):
    """Test main function success path."""
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=True),
        patch("par_ai_core.__main__.LlmConfig") as mock_config,
        patch("par_ai_core.__main__.console_out", mock_console),
        patch("par_ai_core.__main__.get_parai_callback") as mock_callback,
        patch("par_ai_core.__main__.sys.argv", ["script_name", "test prompt"]),
    ):
        # Setup mocks
        mock_config.return_value.build_chat_model.return_value = mock_chat_model
        mock_callback.return_value.__enter__.return_value = None
        mock_callback.return_value.__exit__.return_value = None
        mock_chat_model.invoke.return_value.content = "Because he was feeling ruff!"

        # Run main
        main()

        # Verify API key check and model invocation
        mock_chat_model.invoke.assert_called_once()
        assert mock_console.print.call_count == 1
        assert "Because he was feeling ruff!" in str(mock_console.print.call_args_list[0])


def test_main_no_api_key(mock_console):
    """Test main function when API key is not set."""
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=False),
        patch("par_ai_core.__main__.console_out", mock_console),
    ):
        main()
        mock_console.print.assert_called_once_with(
            "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
        )


def test_main_with_provider_check():
    """Test main function provider check."""
    with patch("par_ai_core.__main__.is_provider_api_key_set") as mock_check:
        mock_check.return_value = False
        main()
        mock_check.assert_called_once_with(LlmProvider.OPENAI)


def test_main_with_command_line_arg(mock_console, mock_chat_model):
    """Test main function with command line argument."""
    test_input = "Tell me about cats"
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=True),
        patch("par_ai_core.__main__.LlmConfig") as mock_config,
        patch("par_ai_core.__main__.console_out", mock_console),
        patch("par_ai_core.__main__.get_parai_callback") as mock_callback,
        patch("par_ai_core.__main__.sys.argv", ["script_name", test_input]),
    ):
        # Setup mocks
        mock_config.return_value.build_chat_model.return_value = mock_chat_model
        mock_callback.return_value.__enter__.return_value = None
        mock_callback.return_value.__exit__.return_value = None
        mock_chat_model.invoke.return_value.content = "Cats are wonderful pets!"

        # Run main
        main()

        # Verify the command line argument was used
        mock_chat_model.invoke.assert_called_once()
        assert mock_chat_model.invoke.call_args[0][0][1]["content"] == test_input


def test_main_no_prompt_provided(mock_console):
    """Test main function when no prompt is provided."""
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=True),
        patch("par_ai_core.__main__.console_out", mock_console),
        patch("par_ai_core.__main__.sys.argv", ["script_name"]),  # No prompt argument
    ):
        main()
        mock_console.print.assert_called_once_with("Please provide a prompt as 1st command line argument.")


def test_main_load_dotenv_called():
    """Test that load_dotenv is called."""
    with (
        patch("par_ai_core.__main__.load_dotenv") as mock_load_dotenv,
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=False),
        patch("par_ai_core.__main__.console_out"),
    ):
        main()
        mock_load_dotenv.assert_called_once()


def test_main_llm_config_setup(mock_console, mock_chat_model):
    """Test LLM configuration setup in main function."""
    test_input = "Test prompt"
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=True),
        patch("par_ai_core.__main__.LlmConfig") as mock_config,
        patch("par_ai_core.__main__.console_out", mock_console),
        patch("par_ai_core.__main__.get_parai_callback") as mock_callback,
        patch("par_ai_core.__main__.sys.argv", ["script_name", test_input]),
        patch("par_ai_core.__main__.provider_light_models", {LlmProvider.OPENAI: "gpt-3.5-turbo"}),
    ):
        # Setup mocks
        mock_config.return_value.build_chat_model.return_value = mock_chat_model
        mock_callback.return_value.__enter__.return_value = None
        mock_callback.return_value.__exit__.return_value = None

        # Run main
        main()

        # Verify LlmConfig was called with correct parameters
        mock_config.assert_called_once_with(provider=LlmProvider.OPENAI, model_name="gpt-3.5-turbo")
        mock_config.return_value.build_chat_model.assert_called_once()


def test_main_callback_context_manager(mock_console, mock_chat_model):
    """Test that callback context manager is used correctly."""
    test_input = "Test prompt"
    with (
        patch("par_ai_core.__main__.is_provider_api_key_set", return_value=True),
        patch("par_ai_core.__main__.LlmConfig") as mock_config,
        patch("par_ai_core.__main__.console_out", mock_console),
        patch("par_ai_core.__main__.get_parai_callback") as mock_callback,
        patch("par_ai_core.__main__.sys.argv", ["script_name", test_input]),
        patch("par_ai_core.__main__.PricingDisplay") as mock_pricing_display,
    ):
        # Setup mocks
        mock_config.return_value.build_chat_model.return_value = mock_chat_model
        mock_callback.return_value.__enter__.return_value = None
        mock_callback.return_value.__exit__.return_value = None

        # Run main
        main()

        # Verify callback was called with correct parameters
        mock_callback.assert_called_once_with(
            show_pricing=mock_pricing_display.DETAILS, show_tool_calls=True, show_end=False
        )
