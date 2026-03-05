# Contributing to PAR AI Core

Contributions are welcome! Here's how to get started.

## Development Setup

1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) for package management
3. Run `make setup` to install dependencies
4. Run `make checkall` to verify everything works

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run `make checkall` (format, lint, typecheck, tests must all pass)
4. Commit with conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
5. Submit a Pull Request

## Code Standards

- Python 3.11+ with type annotations on all functions
- Google-style docstrings for public APIs
- Line length: 120 characters
- Use `ruff` for formatting and linting, `pyright` for type checking
- Tests use `pytest` with 30-second timeout per test

## Running Checks

```bash
make format      # Format code
make lint        # Lint code
make typecheck   # Type check
make test        # Run tests
make checkall    # All of the above
```

## Adding a New LLM Provider

1. Add the provider to `LlmProvider` enum in `llm_providers.py`
2. Update provider configuration dictionaries (`provider_base_urls`, `provider_default_models`, `provider_env_key_names`, `provider_light_models`)
3. Add a `_build_<provider>_llm` method in `llm_config.py`
4. Wire it up in `_build_llm` dispatch
5. Add tests and update documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
