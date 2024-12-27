"""Basic LLM example using Par AI Core."""

import sys

from dotenv import load_dotenv

from par_ai_core.llm_config import LlmConfig, llm_run_manager
from par_ai_core.llm_providers import (
    LlmProvider,
    is_provider_api_key_set,
    provider_light_models,
)
from par_ai_core.par_logging import console_out
from par_ai_core.pricing_lookup import PricingDisplay
from par_ai_core.provider_cb_info import get_parai_callback

load_dotenv()


def main() -> None:
    """
    Use OpenAI lightweight model to answer a question from the command line.

    This function performs the following steps:
    1. Checks if OpenAI API key is set
    2. Validates that a prompt is provided as a command-line argument
    3. Configures an OpenAI chat model
    4. Invokes the model with a system and user message
    5. Prints the model's response

    Requires:
    - OPENAI_API_KEY environment variable to be set
    - A prompt provided as the first command-line argument
    """
    # Validate OpenAI API key is available
    if not is_provider_api_key_set(LlmProvider.OPENAI):
        console_out.print("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")
        return

    # Ensure a prompt is provided via command-line argument
    if len(sys.argv) < 2:
        console_out.print("Please provide a prompt as 1st command line argument.")
        return

    # Configure the LLM using OpenAI's lightweight model
    llm_config = LlmConfig(provider=LlmProvider.OPENAI, model_name=provider_light_models[LlmProvider.OPENAI])
    chat_model = llm_config.build_chat_model()

    # Use context manager to handle callbacks for pricing and tool calls
    with get_parai_callback(show_pricing=PricingDisplay.DETAILS, show_tool_calls=True, show_end=False):
        # Prepare messages with a system context and user prompt
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sys.argv[1]},
        ]

        # Invoke the chat model and get the result
        result = chat_model.invoke(messages, config=llm_run_manager.get_runnable_config(chat_model.name or ""))

        # Print the model's response
        console_out.print(result.content)


if __name__ == "__main__":
    main()
