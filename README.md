# Par AI Core

[![PyPI](https://img.shields.io/pypi/v/par_ai_core)](https://pypi.org/project/par_ai_core/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/par_ai_core.svg)](https://pypi.org/project/par_ai_core/)  
![Runs on Linux | MacOS | Windows](https://img.shields.io/badge/runs%20on-Linux%20%7C%20MacOS%20%7C%20Windows-blue)
![Arch x86-63 | ARM | AppleSilicon](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM%20%7C%20AppleSilicon-blue)  
![PyPI - License](https://img.shields.io/pypi/l/par_ai_core)
[![codecov](https://codecov.io/gh/paulrobello/par_ai_core/branch/main/graph/badge.svg)](https://codecov.io/gh/paulrobello/par_ai_core)

## Description
Par AI Core is a Python library that provides a set of tools, helpers, and wrappers built on top of LangChain. 
It is designed to accelerate the development of AI-powered applications by offering a streamlined and efficient way 
to interact with various Large Language Models (LLMs) and related services. This library serves as the foundation 
for my AI projects, encapsulating common functionalities and best practices for LLM integration.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/probello3)

## Technology
- Python
- LangChain

## Prerequisites

- Python 3.10 or higher
- UV package manager
- API keys for chosen AI provider (except for Ollama and LlamaCpp)
    - See (Environment Variables)[#environment-variables] below for provider-specific variables

## Features

* **Simplified LLM Configuration:** Easily configure and manage different LLM providers (OpenAI, Anthropic, etc.) and models through a unified interface.
* **Asynchronous and Synchronous Support:** Supports both asynchronous and synchronous calls to LLMs.
* **Context Management:** Tools for gathering relevant files as context for LLM prompts.
* **Output Formatting:** Utilities for displaying LLM outputs in various formats (JSON, CSV, tables).
* **Cost Tracking:**  Optional integration to display the cost of LLM calls.
* **Tool Call Handling:** Support for handling tool calls within LLM interactions.

## Documentation
[Library Documentation](https://htmlpreview.github.io/?https://github.com/paulrobello/par_ai_core/blob/main/src/par_ai_core/docs/index.html)

## Installation
```shell
uv add par_ai_core
```

## Update
```shell
uv add par_ai_core -U
```

## Environment Variables

### Create a .env file in the root of your project with the following content adjusted for your needs

```shell
# AI API KEYS
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
XAI_API_KEY=
GOOGLE_API_KEY=
MISTRAL_API_KEY=
GITHUB_TOKEN=
AWS_PROFILE=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Search
GOOGLE_CSE_ID=
GOOGLE_CSE_API_KEY=
SERPER_API_KEY=
SERPER_API_KEY_GOOGLE=
TAVILY_API_KEY=
JINA_API_KEY=
BRAVE_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# Misc API
WEATHERAPI_KEY=
GITHUB_PERSONAL_ACCESS_TOKEN=

### Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=par_ai

# PARAI Related (Not all providers / models support all vars)
PARAI_AI_PROVIDER=
PARAI_MODEL=
PARAI_AI_BASE_URL=
PARAI_TEMPERATURE=
PARAI_TIMEOUT=
PARAI_NUM_CTX=
PARAI_NUM_REDICT=
PARAI_REPEAT_LAST_N=
PARAI_REPEAT_PENALTY=
PARAI_MIROSTAT=
PARAI_MIROSTAT_ETA=
PARAI_MIROSTAT_TAU=
PARAI_TFS_Z=
PARAI_TOP_K=
PARAI_TOP_P=
PARAI_SEED=
```

### AI API KEYS

* ANTHROPIC_API_KEY is required for Anthropic. Get a key from https://console.anthropic.com/
* OPENAI_API_KEY is required for OpenAI. Get a key from https://platform.openai.com/account/api-keys
* GITHUB_TOKEN is required for GitHub Models. Get a free key from https://github.com/marketplace/models
* GOOGLE_API_KEY is required for Google Models. Get a free key from https://console.cloud.google.com
* XAI_API_KEY is required for XAI. Get a free key from https://x.ai/api
* GROQ_API_KEY is required for Groq. Get a free key from https://console.groq.com/
* MISTRAL_API_KEY is required for Mistral. Get a free key from https://console.mistral.ai/
* AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are used for Bedrock authentication. The environment must
  already be authenticated with AWS.
* No key required to use with Ollama or LlamaCpp.

### Search

* TAVILY_API_KEY is required for Tavily AI search. Get a free key from https://tavily.com/. Tavily is much better than
* JINA_API_KEY is required for Jina search. Get a free key from https://jina.ai
* BRAVE_API_KEY is required for Brave search. Get a free key from https://brave.com/search/api/
* SERPER_API_KEY is required for Serper search. Get a free key from https://serper.dev
* SERPER_API_KEY_GOOGLE is required for Google Serper search. Get a free key from https://serpapi.com/
* GOOGLE_CSE_ID and GOOGLE_CSE_API_KEY are required for Google search.
* REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are needed for Reddit search. Get a free key
  from https://www.reddit.com/prefs/apps/

### Misc API

* GITHUB_PERSONAL_ACCESS_TOKEN is required for GitHub related tools. Get a free key
  from https://github.com/settings/tokens
* WEATHERAPI_KEY is required for weather. Get a free key from https://www.weatherapi.com/
* LANGCHAIN_API_KEY is required for Langchain / Langsmith tracing. Get a free key
  from https://smith.langchain.com/settings

### PARAI Related
* PARAI_AI_PROVIDER is one of Ollama|OpenAI|Groq|XAI|Anthropic|Google|Bedrock|Github|LlamaCpp
* PARAI_MODEL is the model to use with the selected provider
* PARAI_AI_BASE_URL can be used to override the base url used to call a provider
* PARAI_TEMPERATURE sets model temperature. Range depends on provider usually 0.0 to 1.0
* PARAI_TIMEOUT length of time to wait in seconds for ai response
* PARAI_NUM_CTX sets the context window size. Max size varies by model
* Other PARAI related params are to tweak model responses not all are supported / used by all providers

### Open AI Compatible Providers

If a specify provider is not listed but has an OpenAI compatible endpoint you can use the following combo of vars:
* PARAI_AI_PROVIDER=OpenAI
* PARAI_MODEL=Your selected model
* PARAI_AI_BASE_URL=The providers OpenAI endpoint URL

## Example

```python
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
```

## Whats New
- Version 0.1.8:
  - Added time display utils
  - Made LlmConfig.from_json more robust
- Version 0.1.7:
  - Fix documentation issues
- Version 0.1.6:
  - Pricing for Deepseek
  - Updated Docs
- Version 0.1.5:
  - Initial release

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Paul Robello - probello@gmail.com
