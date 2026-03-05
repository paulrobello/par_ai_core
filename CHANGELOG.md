# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.5.6]

### Changed
- Bumped version for post-audit release

## [0.5.5]

### Added
- `.env.example` with placeholder values for all API keys
- `apply_nest_asyncio()` and `configure_user_agent()` opt-in functions
- LRU eviction to `LlmRunManager` (max 1000 entries)
- `pytest-timeout` for test reliability
- `tiktoken` as explicit dependency
- `CONTRIBUTING.md` and `CHANGELOG.md`
- Architecture and operations documentation

### Fixed
- Made `ParAICallbackHandler` hashable to prevent LangChain callback merge errors
- `LlmConfig` build methods no longer mutate original config (clone-before-mutate)
- `LlmRunManager` uses instance-level state instead of class-level shared mutable state
- `ignore_ssl` default changed from `True` to `False` in web fetch functions
- Asyncio anti-pattern (`loop.run_until_complete`) removed in `web_tools.py`
- Pinned GitHub Actions to specific versions for supply-chain security
- `JINA_API_KEY` / `BRAVE_API_KEY` now raise descriptive `ValueError` instead of `KeyError`
- Silent exception swallowing replaced with `logger.debug` in file context gathering
- Bare `raise _` fixed to bare `raise` to preserve tracebacks

### Deprecated
- `md5_hash` and `sha1_hash` now emit `DeprecationWarning` (use `sha256_hash`)
- `run_shell_cmd` marked as deprecated in docstring

### Removed
- ~280 lines of dead `pricing_lookup` dictionary
- Commented-out code across multiple modules
- Auto-applied `nest_asyncio` and `USER_AGENT` side effects on import

## [0.5.4]

### Changed
- Updated default OpenAI model from gpt-5 to gpt-5.1
- Updated LiteLLM default model from gpt-5 to gpt-5.1
- Updated vision model from gpt-5 to gpt-5.1

## [0.4.3]

### Added
- Python 3.14 support while maintaining compatibility with Python 3.11-3.13

### Changed
- Minimum required Python version is now 3.11 (dropped Python 3.10)
- Ruff and Pyright now target Python 3.14
- GitHub Actions workflows updated to test against Python 3.11-3.14

### Removed
- `langchain-chroma` and `langchain-qdrant` (unused dependencies)

## [0.4.2]

### Changed
- Updated dependencies

## [0.4.1]

### Fixed
- Compatibility issue between litellm and openai libraries by constraining openai to <1.100.0

## [0.4.0]

### Added
- Python 3.10-3.13 support with 3.12 as development target
- Automated CI/CD workflow chain: Build -> TestPyPI -> GitHub Release -> PyPI

### Changed
- Standardized Python version targeting across all development tools
- Improved `.gitignore` with comprehensive patterns
- Enhanced Makefile with fixed lint target and improved dependency management

### Fixed
- All linting and type checking errors
- Updated test mocks and model references for better compatibility

## [0.3.2]

### Changed
- Improved test coverage to 93%

### Fixed
- `nest_asyncio` safety handling

## [0.3.1]

### Changed
- Updated dependencies

## [0.3.0]

### Added
- Support for Azure OpenAI

## [0.2.0]

### Added
- Support for basic auth in Ollama base URLs

## [0.1.25]

### Fixed
- Suppressed pricing not found warning

## [0.1.24]

### Changed
- Changed default fetch wait from idle to sleep

## [0.1.23]

### Fixed
- Issue caused by providing reasoning effort to models that don't support it

## [0.1.22]

### Fixed
- Asyncio issues with web fetch utils

## [0.1.21]

### Added
- Config options for OpenAI reasoning effort and Anthropic reasoning token budget

### Fixed
- o3 error when temperature is set

## [0.1.20]

### Added
- Parallel fetch support for `fetch_url` related utils

## [0.1.19]

### Added
- Proxy config and HTTP auth support for `fetch_url` related utils

## [0.1.18]

### Changed
- Updated web scraping utils

## [0.1.17]

### Changed
- Use LiteLLM for pricing data
- **BREAKING**: Provider `Google` renamed to `Gemini`

## [0.1.16]

### Added
- More default base URLs for providers

## [0.1.15]

### Added
- Support for Deepseek and LiteLLM
- Mistral pricing
- Better fuzzy model matching for price lookup

## [0.1.14]

### Added
- o3-mini pricing

### Fixed
- Gets actual model used from OpenRouter
- Other pricing issues
- Open router default model name

## [0.1.13]

### Added
- Support for supplying extra body params to OpenAI compatible providers like OpenRouter
- Better handling of model names for pricing lookup

## [0.1.12]

### Added
- Support for OpenRouter

## [0.1.11]

### Changed
- Updated utility functions

### Fixed
- dotenv loading issue
- Updated pricing for o1 and Deepseek R1

## [0.1.10]

### Added
- Format param to `LlmConfig` for Ollama output format

### Fixed
- Bug with util function `has_stdin_content`

## [0.1.9]

### Added
- Mistral support

### Fixed
- dotenv loading bug

## [0.1.8]

### Added
- Time display utils
- Made `LlmConfig.from_json` more robust

## [0.1.7]

### Fixed
- Documentation issues

## [0.1.6]

### Added
- Pricing for Deepseek
- Updated docs

## [0.1.5]

### Added
- Initial release
