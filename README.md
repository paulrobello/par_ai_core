# Par AI Core

[![PyPI](https://img.shields.io/pypi/v/par_ai_core)](https://pypi.org/project/par_ai_core/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/par_ai_core.svg)](https://pypi.org/project/par_ai_core/)
[![Build](https://github.com/paulrobello/par_ai_core/actions/workflows/build.yml/badge.svg)](https://github.com/paulrobello/par_ai_core/actions/workflows/build.yml)
![Runs on Linux | MacOS | Windows](https://img.shields.io/badge/runs%20on-Linux%20%7C%20MacOS%20%7C%20Windows-blue)
![Arch x86-64 | ARM | AppleSilicon](https://img.shields.io/badge/arch-x86--64%20%7C%20ARM%20%7C%20AppleSilicon-blue)
![PyPI - Downloads](https://img.shields.io/pypi/dm/par_ai_core)



![PyPI - License](https://img.shields.io/pypi/l/par_ai_core)

[![codecov](https://codecov.io/gh/paulrobello/par_ai_core/branch/main/graph/badge.svg)](https://codecov.io/gh/paulrobello/par_ai_core)

## Table of Contents

- [Description](#description)
- [Technology](#technology)
- [Prerequisites](#prerequisites)
- [Features](#features)
- [Documentation](#documentation)
- [Installation](#installation)
- [Update](#update)
- [Environment Variables](#environment-variables)
- [Example](#example)
- [What's New](#whats-new)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

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

- Python 3.11 or higher
- API keys for chosen AI provider (except for Ollama and LlamaCpp)
    - See [Environment Variables](#environment-variables) below for provider-specific variables
- (Contributors) UV package manager for development — see [Contributing](#contributing)

## Features

* **Simplified LLM Configuration:** Easily configure and manage different LLM providers (OpenAI, Anthropic, etc.) and models through a unified interface.
* **Asynchronous and Synchronous Support:** Supports both asynchronous and synchronous calls to LLMs.
* **Context Management:** Tools for gathering relevant files as context for LLM prompts.
* **Output Formatting:** Utilities for displaying LLM outputs in various formats (JSON, CSV, tables).
* **Cost Tracking:**  Optional integration to display the cost of LLM calls.
* **Tool Call Handling:** Support for handling tool calls within LLM interactions.

## Documentation

- [Architecture Overview](docs/architecture.md) — module structure and dependencies
- [Operations Guide](docs/operations.md) — deployment, driver setup, cloud configuration
- API reference: regenerate the HTML reference locally with `make docs` (output is written to `./docs/build/` and is not shipped in the package)

## Installation

Install the core package with pip:

```shell
pip install par_ai_core
```

Or with uv:

```shell
uv add par_ai_core
```

PAR AI Core uses optional extras so you only install the backends you use. Install a single backend, a feature group, or everything:

```shell
pip install "par_ai_core[openai]"     # one provider backend
pip install "par_ai_core[anthropic]"
pip install "par_ai_core[web]"        # Playwright/Selenium web scraping
pip install "par_ai_core[search]"     # Tavily, Brave, Serper, Google CSE, Reddit, YouTube
pip install "par_ai_core[pricing]"    # LiteLLM cost tracking
pip install "par_ai_core[all]"        # every backend and feature group
```

A feature whose backend is not installed raises an `ImportError` naming the extra to install.

> **Note:** Web scraping (`fetch_url` with Playwright) requires a browser binary. After installing the `[web]` (or `[all]`) extra, run the following once:
>
> ```shell
> playwright install chromium
> ```

## Update

```shell
pip install par_ai_core -U
# or
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
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
AZURE_OPENAI_API_KEY=
# Used by Bedrock
AWS_PROFILE=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
# Local providers (optional)
OLLAMA_HOST=

# Search
GOOGLE_CSE_ID=
GOOGLE_CSE_API_KEY=
SERPER_API_KEY=
TAVILY_API_KEY=
JINA_API_KEY=
BRAVE_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USERNAME=
REDDIT_PASSWORD=

# Tracing (optional)
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
PARAI_STREAMING=
PARAI_USER_AGENT_APPID=
PARAI_NUM_CTX=
PARAI_MAX_OUTPUT_TOKENS=
PARAI_NUM_PREDICT=
PARAI_REPEAT_LAST_N=
PARAI_REPEAT_PENALTY=
PARAI_MIROSTAT=
PARAI_MIROSTAT_ETA=
PARAI_MIROSTAT_TAU=
PARAI_TFS_Z=
PARAI_TOP_K=
PARAI_TOP_P=
PARAI_SEED=
PARAI_REASONING_EFFORT=
PARAI_REASONING_BUDGET=
PARAI_LOG_LEVEL=
```

### AI API KEYS

* ANTHROPIC_API_KEY is required for Anthropic. Get a key from https://console.anthropic.com/
* OPENAI_API_KEY is required for OpenAI. Get a key from https://platform.openai.com/account/api-keys
* GITHUB_TOKEN is required for GitHub Models. Get a free key from https://github.com/marketplace/models
* GOOGLE_API_KEY is required for Google Models. Get a free key from https://console.cloud.google.com
* XAI_API_KEY is required for XAI. Get a free key from https://x.ai/api
* GROQ_API_KEY is required for Groq. Get a free key from https://console.groq.com/
* MISTRAL_API_KEY is required for Mistral. Get a free key from https://console.mistral.ai/
* OPENROUTER_API_KEY is required for OpenRouter. Get a key from https://openrouter.ai/
* DEEPSEEK_API_KEY is required for Deepseek. Get a key from https://platform.deepseek.com/
* AZURE_OPENAI_API_KEY is required for Azure OpenAI.
* AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are used for Bedrock authentication. The environment must
  already be authenticated with AWS.
* OLLAMA_HOST optionally points at an Ollama server (defaults to http://localhost:11434).
* No key required to use with Ollama, LlamaCpp, LiteLLM.

### Open AI Compatible Providers

If a specify provider is not listed but has an OpenAI compatible endpoint you can use the following combo of vars:
* PARAI_AI_PROVIDER=OpenAI
* PARAI_MODEL=Your selected model
* PARAI_AI_BASE_URL=The providers OpenAI endpoint URL

### Search

* TAVILY_API_KEY is required for Tavily AI search. Get a free key from https://tavily.com/
* JINA_API_KEY is required for Jina search. Get a free key from https://jina.ai
* BRAVE_API_KEY is required for Brave search. Get a free key from https://brave.com/search/api/
* SERPER_API_KEY is required for Serper (Google) search. Get a free key from https://serper.dev
* GOOGLE_CSE_ID and GOOGLE_CSE_API_KEY are required for Google Custom Search.
* REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are needed for Reddit search. Get a free key
  from https://www.reddit.com/prefs/apps/
* REDDIT_USERNAME and REDDIT_PASSWORD are optional; provide them to authenticate for
  Reddit search with a higher rate limit.

### Tracing

* LANGCHAIN_API_KEY is required for Langchain / Langsmith tracing. Get a free key
  from https://smith.langchain.com/settings

### PARAI Related
* PARAI_AI_PROVIDER is one of Ollama|LlamaCpp|OpenRouter|OpenAI|Gemini|Github|XAI|Anthropic|Groq|Mistral|Deepseek|LiteLLM|Bedrock|Azure
* PARAI_MODEL is the model to use with the selected provider
* PARAI_AI_BASE_URL can be used to override the base url used to call a provider
* PARAI_TEMPERATURE sets model temperature. Range depends on provider usually 0.0 to 1.0
* PARAI_TIMEOUT length of time to wait in seconds for ai response
* PARAI_STREAMING enables or disables streaming responses (true/false)
* PARAI_USER_AGENT_APPID sets an app identifier added to the User-Agent header for API requests
* PARAI_NUM_CTX sets the Ollama context window size (Ollama only; deprecated for other providers). Max size varies by model
* PARAI_MAX_OUTPUT_TOKENS sets the output token cap (``max_tokens``) for OpenAI, Anthropic, Groq, XAI, OpenRouter, Deepseek, Gemini, Bedrock, Mistral, LiteLLM, and Azure
* PARAI_REASONING_EFFORT sets reasoning effort for OpenAI thinking models (low/medium/high)
* PARAI_REASONING_BUDGET sets the reasoning token budget for Anthropic models
* PARAI_LOG_LEVEL sets the logging level (DEBUG/INFO/WARNING/ERROR)
* Other PARAI related params (NUM_PREDICT, REPEAT_LAST_N, REPEAT_PENALTY, MIROSTAT, MIROSTAT_ETA, MIROSTAT_TAU, TFS_Z, TOP_K, TOP_P, SEED) are to tweak model responses; not all are supported by all providers



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

    load_dotenv()

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

## What's New
- Version 0.5.9:
  - Comprehensive audit remediation: 77 of 78 issues resolved across security, architecture, code quality, and documentation (see `AUDIT-REMEDIATION.md`)
  - Lean core with optional extras — install only the providers you use (`pip install "par_ai_core[openai]"`, `[all]`, etc.)
  - Security hardening: SSRF/local-file guard in `fetch_url`, TLS-bypass + credential safety, FIPS-safe hashing, SHA-pinned CI actions
  - Search functions now return a typed, dict-compatible `SearchResult`; `utils.py` split into a `utils/` package (backward-compatible re-exports)
  - Bug fixes: LiteLLM env configuration, context-size lookup, Playwright text wait, cost double-counting, CSV rendering, and more
- Version 0.5.8:
  - Updated all dependencies to latest versions (langchain, langgraph, openai, litellm, etc.)
  - Migrated `ChatLiteLLM` to the standalone `langchain-litellm` package
- Version 0.5.7:
  - Updated all dependencies to latest versions
  - Refreshed pricing-lookup tests to current LiteLLM model database
  - Resolved all pyright type errors in test suite

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Paul Robello - probello@gmail.com
