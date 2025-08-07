# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PAR AI Core** is a Python library that provides a comprehensive foundation for AI-powered applications. Built on top of LangChain, it offers a streamlined and efficient way to interact with various Large Language Models (LLMs) and related services. The library serves as a core component for AI projects, encapsulating common functionalities and best practices for LLM integration.

Key capabilities include:
- Multi-provider LLM support (OpenAI, Anthropic, Groq, XAI, Google, GitHub, Mistral, Deepseek, AWS Bedrock, Azure, OpenRouter, LiteLLM, Ollama, LlamaCpp)
- Asynchronous and synchronous LLM operations
- Cost tracking and pricing lookup
- Web scraping and search capabilities
- Output formatting utilities
- Context management and file gathering
- Image handling for vision models

## Development Commands

### Setup and Installation
```bash
make setup          # First-time setup with uv (lock + sync)
make resetup        # Recreate virtual environment from scratch
make remove-venv    # Remove the virtual environment
make depsupdate     # Update all dependencies
make depsshow       # Show the dependency graph
make shell          # Start shell inside .venv
```

### Code Quality Commands
```bash
make checkall       # Format, lint, and typecheck (run before commits)
make format         # Format code with ruff
make lint           # Lint code with ruff
make typecheck      # Type check with pyright
make typecheck-stats # Type check with stats
make pre-commit     # Run pre-commit checks on all files
make pre-commit-update # Update pre-commit hooks
```

### Testing
```bash
make test           # Run tests with coverage report
make test-trace     # Run tests with full trace enabled
make coverage       # Generate coverage report and XML output
```

### Build and Documentation
```bash
make package        # Build wheel package
make spackage       # Create source package
make package-all    # Build both wheel and source packages
make docs           # Generate HTML documentation
```

### Running the Application
```bash
make run            # Run the application
make app_help       # Show application help
```

## Architecture and Key Components

### Core Modules Structure
- **`llm_providers.py`**: Provider definitions, configurations, and utilities
- **`llm_config.py`**: LLM configuration management and model building
- **`llm_utils.py`**: Utility functions for LLM operations and environment setup
- **`web_tools.py`**: Web scraping, HTML parsing, and page fetching
- **`search_utils.py`**: Search functionality integration
- **`output_utils.py`**: Output formatting (JSON, CSV, tables, Markdown)
- **`pricing_lookup.py`**: Cost tracking and pricing data
- **`provider_cb_info.py`**: Provider callback management
- **`par_logging.py`**: Logging utilities and console output
- **`utils.py`**: General utility functions
- **`llm_image_utils.py`**: Image handling for vision models
- **`time_display.py`**: Time formatting and display utilities
- **`user_agents.py`**: User agent management for web requests

### Provider System
The library supports multiple LLM providers through a unified interface:
- **Local providers**: Ollama, LlamaCpp
- **Cloud providers**: OpenAI, Anthropic, Google (Gemini), XAI, Groq, Mistral, Deepseek
- **Platform providers**: GitHub, Azure, AWS Bedrock
- **Aggregators**: OpenRouter, LiteLLM

Each provider is configured through:
- Environment variables for API keys
- Base URLs and endpoints
- Default model selections
- Provider-specific parameters

### Configuration Management
- **LlmConfig**: Comprehensive configuration class for model setup
- **LlmMode**: Operating modes (Base, Chat, Embeddings)
- **Environment-based configuration**: Automatic config from environment variables
- **Flexible parameter handling**: Temperature, context size, streaming, etc.

### Web Tools and Search
- **Multi-engine search**: Google Custom Search, Tavily, Brave, Serper
- **Web scraping**: Playwright and Selenium support
- **Content processing**: HTML to Markdown conversion
- **Parallel fetching**: Concurrent web requests
- **Proxy and authentication**: HTTP auth and proxy configuration

### Key Technical Details
- **Type Safety**: Fully typed with Python type annotations
- **Async Support**: Both async and sync operations supported
- **Error Handling**: Comprehensive error handling and logging
- **Performance**: Optimized for concurrent operations
- **Security**: Safe URL handling and content validation
- **Memory Management**: Efficient handling of large content

## Development Guidelines

### Code Style
- Use type annotations for all functions and methods
- Google-style docstrings for all public APIs
- Line length: 120 characters
- Import grouping and sorting enforced by ruff
- Follow Python 3.10+ best practices (use `|` for unions, built-in generics)

### Testing
- Run `make checkall` before committing
- Add unit tests for new functionality
- Use pytest with coverage reporting
- Test both sync and async operations where applicable
- Mock external API calls in tests

### Environment Variables
The library uses extensive environment variable configuration:

#### AI Provider API Keys
```bash
OPENAI_API_KEY=         # OpenAI API key
ANTHROPIC_API_KEY=      # Anthropic API key
GROQ_API_KEY=          # Groq API key
XAI_API_KEY=           # XAI API key
GOOGLE_API_KEY=        # Google/Gemini API key
MISTRAL_API_KEY=       # Mistral API key
DEEPSEEK_API_KEY=      # Deepseek API key
GITHUB_TOKEN=          # GitHub Models API key
OPENROUTER_API_KEY=    # OpenRouter API key
AWS_PROFILE=           # AWS profile for Bedrock
```

#### Search API Keys
```bash
TAVILY_API_KEY=        # Tavily search API key
BRAVE_API_KEY=         # Brave search API key
SERPER_API_KEY=        # Serper search API key
GOOGLE_CSE_ID=         # Google Custom Search Engine ID
GOOGLE_CSE_API_KEY=    # Google Custom Search API key
JINA_API_KEY=          # Jina search API key
```

#### PAR AI Configuration
```bash
PARAI_AI_PROVIDER=     # Provider name (e.g., OpenAI, Anthropic)
PARAI_MODEL=           # Model name
PARAI_AI_BASE_URL=     # Custom base URL
PARAI_TEMPERATURE=     # Model temperature (0.0-1.0)
PARAI_TIMEOUT=         # Request timeout in seconds
PARAI_NUM_CTX=         # Context window size
# ... additional model parameters
```

### Common Development Tasks

**Adding a New LLM Provider:**
1. Add provider to `LlmProvider` enum in `llm_providers.py`
2. Update provider configuration dictionaries:
   - `provider_base_urls`
   - `provider_default_models`
   - `provider_env_key_names`
   - `provider_light_models` (optional)
3. Add provider-specific logic in `llm_config.py` if needed
4. Update documentation and tests

**Adding New Web Tools:**
1. Add functionality to `web_tools.py`
2. Update search capabilities in `search_utils.py` if applicable
3. Add corresponding tests
4. Update documentation

**Modifying Output Formatting:**
1. Update `output_utils.py` with new formatting options
2. Add display format options to `DisplayOutputFormat` enum
3. Test with various data types and sizes

**Adding Pricing Data:**
1. Update pricing information in `pricing_lookup.py`
2. Add provider-specific pricing logic
3. Test cost calculation accuracy

**Working with Configuration:**
- Use `llm_config_from_env()` for environment-based configuration
- Extend `LlmConfig` class for new configuration options
- Maintain backward compatibility when possible

### Version Management
- **Important**: Only bump the project version if explicitly requested
- Version is managed in `src/par_ai_core/__init__.py`
- Update version in `__version__` variable
- Follow semantic versioning (MAJOR.MINOR.PATCH)

### Build and Release Process
- Use `make package-all` to build both wheel and source distributions
- GitHub Actions automatically handle CI/CD on main branch
- Tags are automatically created based on version in `__init__.py`
- Use `uv` for all package management operations

### Documentation
- API documentation is auto-generated using `pdoc3`
- Run `make docs` to generate HTML documentation
- Documentation is included in the package build
- Keep README.md updated with new features and examples
