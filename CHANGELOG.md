# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release dates are derived from the corresponding git tag timestamps.

## [Unreleased]

## [0.5.9] - 2026-07-10

Comprehensive audit remediation — 77 of 78 issues resolved (1 backlog refactor,
ARC-008, deferred). See `AUDIT-REMEDIATION.md` for the full per-issue record.

### Added
- **Opt-in `par_ai_core.init_logging()` (ARC-002):** restores the Rich console
  handlers and tracebacks that previously ran at import time.
- **`LlmConfig.max_output_tokens` (ARC-006):** the output-token cap for non-Ollama
  providers, wired to `PARAI_MAX_OUTPUT_TOKENS`.
- **`LlmConfig.safety_settings` (SEC-003):** optional Gemini safety-setting
  override (default `None` = provider-standard thresholds).
- **`SearchResult` model (ARC-014):** typed, dict-compatible return type for all
  search functions (see Changed).
- **`fetch_url(raise_on_error=...)` (QA-006):** escape hatch to surface fetch
  errors instead of swallowing them.
- **Standard Makefile targets `build` and `fmt` (ARC-024).**

### Changed
- **Lean core with optional extras (ARC-001):** the package now installs only the
  core by default. Install backends individually (`pip install "par_ai_core[openai]"`,
  `[anthropic]`, etc.) or all at once (`pip install "par_ai_core[all]"`). A feature
  whose backend is missing raises an `ImportError` naming the extra to install.
- **Library no longer mutates global logging (ARC-002):** importing `par_ai_core` no
  longer calls `logging.basicConfig` or installs a global `sys.excepthook`. The
  `par_ai` logger now attaches a `NullHandler` and sets `propagate = False`. Call the
  opt-in `par_ai_core.init_logging()` to restore the Rich handlers and tracebacks.
- **`num_ctx` is Ollama-only (ARC-006):** `LlmConfig.num_ctx` is now the Ollama
  context-window size only. For the output token cap on other providers
  (OpenAI/Anthropic/etc.) use the new `LlmConfig.max_output_tokens` field
  (`PARAI_MAX_OUTPUT_TOKENS`). Setting `num_ctx` on a non-Ollama provider emits a
  `DeprecationWarning` at build time.
- **Provider matching is exact-first (ARC-009):** `get_provider_name_fuzzy` now
  returns an exact case-insensitive match first and raises on ambiguous prefixes.
  `"open"` is now ambiguous (matches both OpenAI and OpenRouter) where it previously
  guessed; use the full provider name.
- **Search functions return typed `SearchResult` (ARC-014):** `tavily_search`,
  `jina_search`, `brave_search`, `serper_search`, `reddit_search`, `youtube_search`,
  and `web_search` now return `list[SearchResult]` instead of `list[dict[str, Any]]`.
  `SearchResult` is a Pydantic v2 model that is **dict-compatible** — it supports
  `result["title"]`, `result.get("url")`, `"title" in result`, `dict(result)`, and
  iteration over field names — so existing dict-style callers continue to work without
  changes. The unified fields are `title`, `url`, `content`, `raw_content`, and
  `score` (optional). `web_search` (Google CSE) has been relocated from `web_tools`
  to `search_utils` and is re-exported from `web_tools` for backward compatibility;
  it now maps `link`→`url` and `snippet`→`content` to align with the other engines.
  `GoogleSearchResult` remains in `web_tools` as a deprecated legacy model.
- **Search subsystem decoupled from LLM config (ARC-014):** `import par_ai_core.search_utils`
  no longer transitively imports `llm_utils`, `llm_config`, or `llm_providers`. The
  top-level `from par_ai_core.llm_utils import summarize_content` import has been
  removed; `youtube_search` accepts a new `summarizer: Callable[[str], str] | None`
  parameter for decoupled transcript summarization. The legacy `summarize_llm`
  parameter still works (the `summarize_content` import is deferred to call time)
  but is deprecated in favor of `summarizer`. The `fetch_url_and_convert_to_markdown`
  import is now lazy inside `brave_search`/`serper_search`, removing the top-level
  `web_tools` dependency from `search_utils`.
- **LLM correlation no longer hijacks `llm.name` (ARC-010):** `build_chat_model`/
  `build_llm_model` no longer overwrite the model's `name` with the config_id UUID.
  Correlation (cost tracking, run config) flows exclusively through `RunnableConfig`
  metadata + tags, which were already in place. Code that read `llm.name` expecting
  the config_id should read the `config_id=` tag or `metadata["config_id"]` instead.
- **`utils.py` is now a package (ARC-013):** the 931-line module is split into a
  `utils/` package of 10 cohesive submodules. `__init__.py` re-exports all 49 public
  names, so `from par_ai_core.utils import X` is unchanged. `markdownify` was dropped;
  `html2text` is now the single HTML→Markdown converter (`md()` reimplemented on it,
  LLM-facing output preserved).
- **Lazy `litellm` + data-driven zero-cost pricing (ARC-012):** `litellm` is imported
  at call time in the pricing layer (faster cold imports). The hardcoded zero-cost
  list is reduced to genuinely-local `[OLLAMA, LLAMACPP]` — Groq and GitHub Models
  are now priced from litellm data.
- **`serper_search` parameter rename (QA-010):** the `type` parameter is now
  `search_type` (no longer shadows the builtin); `days` is wired into a Serper
  freshness filter.
- **`fetch_url_and_convert_to_markdown` forwards fetcher params (ARC-021):** proxy,
  credentials, wait type, headless, ignore_ssl, and max_parallel are now passed
  through to the underlying fetcher.
- **`OLLAMA_HOST` read at build time (ARC-023):** read fresh per build, so a
  post-import `.env` load is honored.
- **Stdlib `enum.StrEnum` (ARC-018):** the `strenum` backport is dropped.

### Security
- **`fetch_url` rejects unsafe URLs (SEC-001):** only public `http(s)` URLs are
  fetched; `file://`, loopback, link-local, and private/reserved ranges are rejected
  with `ValueError` (SSRF / local-file-disclosure guard).
- **`ignore_ssl` + `http_credentials` combination raises (SEC-002):** a visible
  warning is emitted whenever `ignore_ssl=True`, and combining it with
  `http_credentials` raises `ValueError` to prevent credential exposure to a
  man-in-the-middle.
- **Gemini no longer forces `BLOCK_NONE` (SEC-003):** `LlmConfig.safety_settings`
  (default `None`) lets the provider's standard safety thresholds apply; pass an
  explicit mapping only to override.
- **FIPS-safe weak-hash helpers (SEC-006):** `md5_hash`/`sha1_hash` use
  `usedforsecurity=False`; deprecation warnings preserved.
- **Selenium credential-injection logging (SEC-008):** the worker logs the
  pre-injection URL; `inject_credentials` docstring warns of netloc leakage and
  prefers Playwright `http_credentials`.
- **Pinned CI actions (SEC-009):** third-party GitHub Actions in the publish/release
  workflows are pinned to verified commit SHAs.

### Fixed
- **Playwright TEXT wait (QA-001):** `ScraperWaitType.TEXT` now uses
  `page.wait_for_function(...)` (the previous `wait_for_text` method did not exist),
  so text-based waits work.
- **LiteLLM context-size lookup (QA-002):** `getattr` on a TypedDict (always `None`)
  replaced with dict access — restores correct context sizes for modern 128k–2M
  models.
- **LiteLLM env configuration (ARC-005):** LiteLLM can now be configured from the
  environment (previously raised an empty-variable-name `ValueError`).
- **Exception-safe cost callback (ARC-004):** `parai_callback_var` resets and prints
  cost even when the wrapped block raises.
- **Selenium wait semantics (QA-003):** `SLEEP` now sleeps; trailing unconditional
  sleeps removed; `NONE` skips all sleeps.
- **`LlmConfig.from_json` errors (QA-005):** raises a clear `ValueError` naming the
  missing field instead of a bare `KeyError`; `mode` defaults when absent.
- **`has_value` suffix stripping (QA-007):** `rstrip(".00")` (a char set) →
  `removesuffix(".00")`.
- **Reasoning-model detection (QA-009):** consolidated into one helper with an
  explicit prefix tuple (now catches `gpt-5`/`gpt-5-mini`/`o4-mini`).
- **`youtube_search` max comments (QA-011):** forwards `max_results=max_comments`.
- **CSV rendering (QA-012):** `display_formatted_output` reuses `csv_to_table` (no
  more `StopIteration` on empty input / Rich error on ragged rows).
- **Cost double-counting (QA-015):** `accumulate_cost` normalizes the payload to one
  convention before accumulating.
- **Blocking `input()` in async (QA-017):** replaced with
  `await asyncio.to_thread(input)`.
- **Desktop Chrome user agent (QA-021):** no longer includes the "Mobile Safari"
  token.

### Removed
- `markdownify` dependency (ARC-013) — `html2text` is the single converter.
- `strenum` backport dependency (ARC-018) — stdlib `enum.StrEnum` instead.
- Generated HTML/PNG documentation from the wheel (ARC-015) — `make docs` writes to
  the gitignored `./docs/build/` directory.

## [0.5.8] - 2026-06-22

### Changed
- Updated all dependencies to latest versions (langchain 1.3.11, langchain-core 1.4.8, langchain-aws 1.6.0, langchain-openai 1.3.3, langchain-google-community 5.0.0, langgraph 1.2.6, litellm 1.89.3, openai 2.43.0, praw 8.0.1, etc.)

### Fixed
- Migrated `ChatLiteLLM` import from the sunset `langchain_community.chat_models` to the standalone `langchain-litellm` package (new dependency)
- Updated `BeautifulSoup.find_all(attrs=...)` call in `web_tools.py` to pass `name` positionally, fixing a pyright error from beautifulsoup4 4.15 stub overloads

## [0.5.7] - 2026-04-25

### Changed
- Updated all dependencies to latest versions (langchain 1.2.15, langgraph 1.1.9, openai 2.32.0, litellm 1.83.0, pydantic 2.13.3, etc.)
- `LlmRunManager.get_runnable_config_by_llm_config` signature now accepts `LlmConfig | None` to match runtime behavior

### Fixed
- Updated pricing-lookup tests to current LiteLLM model database (claude-sonnet-4-5, gemini-2.5-flash, gemini-2.0-flash-lite) after upstream removed older model IDs
- Resolved all pyright type errors in test suite (TypedDict access, MockConsole inheritance, IO buffer typing)

## [0.5.6] - 2026-03-05

### Changed
- Bumped version for post-audit release

## [0.5.5] - 2026-01-22

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

## [0.5.4] - 2025-11-24

### Changed
- Updated default OpenAI model from gpt-5 to gpt-5.1
- Updated LiteLLM default model from gpt-5 to gpt-5.1
- Updated vision model from gpt-5 to gpt-5.1

## [0.5.3] - 2025-11-16

### Changed
- Updated dependencies

## [0.4.3] - 2025-11-16

### Added
- Python 3.14 support while maintaining compatibility with Python 3.11-3.13

### Changed
- Minimum required Python version is now 3.11 (dropped Python 3.10)
- Ruff and Pyright now target Python 3.14
- GitHub Actions workflows updated to test against Python 3.11-3.14

### Removed
- `langchain-chroma` and `langchain-qdrant` (unused dependencies)

## [0.4.2] - 2025-10-24

### Changed
- Updated dependencies

## [0.4.1] - 2025-08-18

### Fixed
- Compatibility issue between litellm and openai libraries by constraining openai to <1.100.0

## [0.4.0] - 2025-08-08

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

## [0.3.2] - 2025-07-19

### Changed
- Improved test coverage to 93%

### Fixed
- `nest_asyncio` safety handling

## [0.3.1] - 2025-05-17

### Changed
- Updated dependencies

## [0.3.0] - 2025-04-29

### Added
- Support for Azure OpenAI

## [0.2.0] - 2025-04-25

### Added
- Support for basic auth in Ollama base URLs

## [0.1.25] - 2025-04-11

### Fixed
- Suppressed pricing not found warning

## [0.1.24] - 2025-03-20

### Changed
- Changed default fetch wait from idle to sleep

## [0.1.23] - 2025-03-12

### Fixed
- Issue caused by providing reasoning effort to models that don't support it

## [0.1.22] - 2025-03-10

### Fixed
- Asyncio issues with web fetch utils

## [0.1.21] - 2025-03-04

### Added
- Config options for OpenAI reasoning effort and Anthropic reasoning token budget

### Fixed
- o3 error when temperature is set

## [0.1.20] - 2025-02-17

### Added
- Parallel fetch support for `fetch_url` related utils

## [0.1.19] - 2025-02-14

### Added
- Proxy config and HTTP auth support for `fetch_url` related utils

## [0.1.18] - 2025-02-13

### Changed
- Updated web scraping utils

## [0.1.17] - 2025-02-11

### Changed
- Use LiteLLM for pricing data
- **BREAKING**: Provider `Google` renamed to `Gemini`

## [0.1.16] - 2025-02-11

### Added
- More default base URLs for providers

## [0.1.15] - 2025-02-10

### Added
- Support for Deepseek and LiteLLM
- Mistral pricing
- Better fuzzy model matching for price lookup

## [0.1.14] - 2025-02-07

### Added
- o3-mini pricing

### Fixed
- Gets actual model used from OpenRouter
- Other pricing issues
- Open router default model name

## [0.1.13] - 2025-01-31

### Added
- Support for supplying extra body params to OpenAI compatible providers like OpenRouter
- Better handling of model names for pricing lookup

## [0.1.12] - 2025-01-26

### Added
- Support for OpenRouter

## [0.1.11] - 2025-01-23

### Changed
- Updated utility functions

### Fixed
- dotenv loading issue
- Updated pricing for o1 and Deepseek R1

## [0.1.10] - 2025-01-08

### Added
- Format param to `LlmConfig` for Ollama output format

### Fixed
- Bug with util function `has_stdin_content`

## [0.1.9] - 2024-12-31

### Added
- Mistral support

### Fixed
- dotenv loading bug

## [0.1.8] - 2024-12-30

### Added
- Time display utils
- Made `LlmConfig.from_json` more robust

## [0.1.7] - 2024-12-29

### Fixed
- Documentation issues

## [0.1.6] - 2024-12-29

### Added
- Pricing for Deepseek
- Updated docs

## [0.1.5] - 2024-12-27

### Added
- Initial release
