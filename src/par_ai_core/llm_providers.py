"""LLM provider types and configurations.

This module defines the supported Large Language Model (LLM) providers and their
configurations, including model names, API endpoints, and environment variables.
It provides utilities for provider management, configuration access, and API key
validation.

Key components:
- LlmProvider: Enum of supported LLM providers (e.g., OpenAI, Anthropic, Google)
- LlmProviderConfig: Dataclass for storing provider-specific configurations
- Provider dictionaries: Mappings of providers to their default models, API URLs, etc.
- Utility functions: Helper methods for provider name matching, API key validation,
  and retrieving available providers

The module supports various LLM providers, including cloud-based services and
local instances, and offers flexibility in configuring model selections for
different use cases (e.g., standard, lightweight, vision tasks).

Usage:
    from par_ai_core.llm_providers import LlmProvider, get_provider_name_fuzzy

    # Get a provider enum from a string
    provider = get_provider_name_fuzzy("openai")

    # Check if API key is set for a provider
    is_configured = is_provider_api_key_set(LlmProvider.OPENAI)

    # Get list of configured providers
    available_providers = get_providers_with_api_keys()

This module is designed to be easily extensible for adding new LLM providers
and updating existing configurations as provider offerings evolve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


@dataclass
class LangChainConfig:
    """Configuration for LangChain integration.

    Attributes:
        tracing: Whether to enable LangChain tracing
        project: Project name for LangChain
        base_url: Base URL for LangChain API
        api_key: API key for LangChain authentication
    """

    tracing: bool = False
    project: str = "parllama"
    base_url: str = "https://api.smith.langchain.com"
    api_key: str = ""


class LlmProvider(str, Enum):
    """Enumeration of supported LLM providers.

    Supported providers:
        OLLAMA: Local Ollama instance
        OPENAI: OpenAI API
        GROQ: Groq API
        XAI: X.AI (formerly Twitter) API
        ANTHROPIC: Anthropic Claude API
        GOOGLE: Google AI (Gemini) API
        BEDROCK: AWS Bedrock API
        GITHUB: GitHub Copilot API
        LLAMACPP: Local LlamaCpp instance
    """

    OLLAMA = "Ollama"
    OPENAI = "OpenAI"
    GROQ = "Groq"
    XAI = "XAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"
    BEDROCK = "Bedrock"
    GITHUB = "Github"
    LLAMACPP = "LlamaCpp"


llm_provider_types: list[LlmProvider] = list(LlmProvider)
llm_provider_names: list[str] = [p.value.lower() for p in llm_provider_types]

provider_base_urls: dict[LlmProvider, str | None] = {
    LlmProvider.OLLAMA: "http://localhost:11434",
    LlmProvider.LLAMACPP: "http://localhost:8080/v1",
    LlmProvider.OPENAI: None,
    LlmProvider.GROQ: None,
    LlmProvider.XAI: None,
    LlmProvider.ANTHROPIC: None,
    LlmProvider.GOOGLE: None,
    LlmProvider.BEDROCK: None,
    LlmProvider.GITHUB: "https://models.inference.ai.azure.com",
}

provider_default_models: dict[LlmProvider, str] = {
    LlmProvider.OLLAMA: "",
    LlmProvider.LLAMACPP: "default",
    LlmProvider.OPENAI: "gpt-4o",
    LlmProvider.GROQ: "llama3-70b-8192",
    LlmProvider.XAI: "grok-beta",
    LlmProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LlmProvider.GOOGLE: "gemini-1.5-pro-002",
    LlmProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    LlmProvider.GITHUB: "gpt-4o",
}

provider_light_models: dict[LlmProvider, str] = {
    LlmProvider.OLLAMA: "",
    LlmProvider.LLAMACPP: "default",
    LlmProvider.OPENAI: "gpt-4o-mini",
    LlmProvider.GROQ: "llama3-70b-8192",
    LlmProvider.XAI: "grok-beta",
    LlmProvider.ANTHROPIC: "claude-3-haiku-20240307",
    LlmProvider.GOOGLE: "gemini-2.0-flash-exp",
    LlmProvider.BEDROCK: "anthropic.claude-3-haiku-20240307-v1:0",
    LlmProvider.GITHUB: "gpt-4o-mini",
}

provider_vision_models: dict[LlmProvider, str] = {
    LlmProvider.OLLAMA: "",
    LlmProvider.LLAMACPP: "default",
    LlmProvider.OPENAI: "gpt-4o",
    LlmProvider.GROQ: "llama-3.2-90b-vision-preview",
    LlmProvider.XAI: "grok-vision-beta",
    LlmProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LlmProvider.GOOGLE: "gemini-1.5-pro-002",
    LlmProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    LlmProvider.GITHUB: "gpt-4o",
}

provider_default_embed_models: dict[LlmProvider, str] = {
    LlmProvider.OLLAMA: "",  # nomic-embed-text:latest
    LlmProvider.LLAMACPP: "default",
    LlmProvider.OPENAI: "text-embedding-3-large",
    LlmProvider.GROQ: "",
    LlmProvider.XAI: "",
    LlmProvider.ANTHROPIC: "",
    LlmProvider.GOOGLE: "text-embedding-005",
    LlmProvider.BEDROCK: "amazon.titan-embed-text-v2:0",
    LlmProvider.GITHUB: "text-embedding-3-large",
}

provider_env_key_names: dict[LlmProvider, str] = {
    LlmProvider.OLLAMA: "",
    LlmProvider.LLAMACPP: "",
    LlmProvider.OPENAI: "OPENAI_API_KEY",
    LlmProvider.GROQ: "GROQ_API_KEY",
    LlmProvider.XAI: "XAI_API_KEY",
    LlmProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LlmProvider.GOOGLE: "GOOGLE_API_KEY",
    LlmProvider.BEDROCK: "BEDROCK_API_KEY",
    LlmProvider.GITHUB: "GITHUB_TOKEN",
}


def get_provider_name_fuzzy(provider: str) -> str:
    """Get provider name using fuzzy matching.

    Attempts to match a provider name string to a valid provider by checking for
    exact matches and prefix matches. Case-insensitive matching is used.

    Args:
        provider: String to match against provider names. Can be full name or prefix
            (e.g. "openai" or "open" for OpenAI)

    Returns:
        str: Matched provider name if found, empty string if no match found.
            Returns exact provider name with proper casing if matched.

    Examples:
        >>> get_provider_name_fuzzy("openai")
        'OpenAI'
        >>> get_provider_name_fuzzy("anth")
        'Anthropic'
        >>> get_provider_name_fuzzy("invalid")
        ''
    """
    provider = provider.lower()
    for p in llm_provider_types:
        if p.value.lower() == provider:
            return p.value
        if p.value.lower().startswith(provider):
            return p.value
    return ""


@dataclass
class LlmProviderConfig:
    """Configuration for an LLM provider.

    Attributes:
        default_model: Default model identifier for standard usage
        default_light_model: Default model for lightweight/fast usage
        default_vision_model: Default model for vision/multimodal tasks
        default_embeddings_model: Default model for text embeddings
        supports_base_url: Whether provider supports custom base URL
        env_key_name: Environment variable name for API key
    """

    default_model: str
    default_light_model: str
    default_vision_model: str
    default_embeddings_model: str
    supports_base_url: bool
    env_key_name: str


provider_config: dict[LlmProvider, LlmProviderConfig] = {
    LlmProvider.OLLAMA: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.OLLAMA],
        default_light_model=provider_light_models[LlmProvider.OLLAMA],
        default_vision_model=provider_vision_models[LlmProvider.OLLAMA],
        default_embeddings_model=provider_default_embed_models[LlmProvider.OLLAMA],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.OLLAMA],
    ),
    LlmProvider.OPENAI: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.OPENAI],
        default_light_model=provider_light_models[LlmProvider.OPENAI],
        default_vision_model=provider_vision_models[LlmProvider.OPENAI],
        default_embeddings_model=provider_default_embed_models[LlmProvider.OPENAI],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.OPENAI],
    ),
    LlmProvider.GROQ: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.GROQ],
        default_light_model=provider_light_models[LlmProvider.GROQ],
        default_vision_model=provider_vision_models[LlmProvider.GROQ],
        default_embeddings_model=provider_default_embed_models[LlmProvider.GROQ],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.GROQ],
    ),
    LlmProvider.XAI: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.XAI],
        default_light_model=provider_light_models[LlmProvider.XAI],
        default_vision_model=provider_vision_models[LlmProvider.XAI],
        default_embeddings_model=provider_default_embed_models[LlmProvider.XAI],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.XAI],
    ),
    LlmProvider.ANTHROPIC: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.ANTHROPIC],
        default_light_model=provider_light_models[LlmProvider.ANTHROPIC],
        default_vision_model=provider_vision_models[LlmProvider.ANTHROPIC],
        default_embeddings_model=provider_default_embed_models[LlmProvider.ANTHROPIC],
        supports_base_url=False,
        env_key_name=provider_env_key_names[LlmProvider.ANTHROPIC],
    ),
    LlmProvider.GOOGLE: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.GOOGLE],
        default_light_model=provider_light_models[LlmProvider.GOOGLE],
        default_vision_model=provider_vision_models[LlmProvider.GOOGLE],
        default_embeddings_model=provider_default_embed_models[LlmProvider.GOOGLE],
        supports_base_url=False,
        env_key_name=provider_env_key_names[LlmProvider.GOOGLE],
    ),
    LlmProvider.BEDROCK: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.BEDROCK],
        default_light_model=provider_light_models[LlmProvider.BEDROCK],
        default_vision_model=provider_vision_models[LlmProvider.BEDROCK],
        default_embeddings_model=provider_default_embed_models[LlmProvider.BEDROCK],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.BEDROCK],
    ),
    LlmProvider.GITHUB: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.GITHUB],
        default_light_model=provider_light_models[LlmProvider.GITHUB],
        default_vision_model=provider_vision_models[LlmProvider.GITHUB],
        default_embeddings_model=provider_default_embed_models[LlmProvider.GITHUB],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.GITHUB],
    ),
    LlmProvider.LLAMACPP: LlmProviderConfig(
        default_model=provider_default_models[LlmProvider.LLAMACPP],
        default_light_model=provider_light_models[LlmProvider.LLAMACPP],
        default_vision_model=provider_vision_models[LlmProvider.LLAMACPP],
        default_embeddings_model=provider_default_embed_models[LlmProvider.LLAMACPP],
        supports_base_url=True,
        env_key_name=provider_env_key_names[LlmProvider.LLAMACPP],
    ),
}


def provider_name_to_enum(name: str) -> LlmProvider:
    """Convert provider name string to LlmProvider enum.

    Args:
        name: Provider name string (case-sensitive)

    Returns:
        LlmProvider: Corresponding enum value

    Raises:
        ValueError: If name doesn't match any provider
    """
    return LlmProvider(name)


def is_provider_api_key_set(provider: LlmProvider) -> bool:
    """Check if API key is set for the given provider.

    Args:
        provider: LLM provider to check

    Returns:
        bool: True if provider doesn't need key (Ollama/LlamaCpp) or
            if required environment variable is set and non-empty
    """
    if provider in [LlmProvider.OLLAMA, LlmProvider.LLAMACPP]:
        return True
    return len(os.environ.get(provider_env_key_names[provider], "")) > 0


def get_providers_with_api_keys() -> list[LlmProvider]:
    """Get list of providers that have valid API keys configured.

    Returns:
        list[LlmProvider]: List of providers that are ready to use
            (either have API key set or don't require one)
    """
    return [p for p in LlmProvider if is_provider_api_key_set(p)]


def get_provider_select_options() -> list[tuple[str, LlmProvider]]:
    """Get provider options for UI selection.

    Returns:
        list[tuple[str, LlmProvider]]: List of tuples containing
            (provider display name, provider enum) for each available provider
    """
    return [
        (
            p.value,
            LlmProvider(p),
        )
        for p in get_providers_with_api_keys()
    ]
