"""
Pricing Lookup and Cost Calculation Module

This module provides functionality for managing and calculating costs associated with
various AI language models across different providers. It includes:

- Functions to calculate API call costs based on usage
- Utilities for accumulating and displaying cost information

Key components:
- PricingDisplay: An enum for controlling the level of cost display detail
- Functions for cost calculation, usage metadata management, and cost display

Usage:
    from par_ai_core.pricing_lookup import get_api_call_cost, show_llm_cost

    # Calculate cost for an API call
    cost = get_api_call_cost(llm_config, usage_metadata)

    # Display cost information
    show_llm_cost(usage_metadata, show_pricing=PricingDisplay.DETAILS)

This module is essential for tracking and managing costs in AI-powered applications,
especially when working with multiple AI providers and models.
"""

from typing import Literal

from litellm.types.utils import ModelInfo
from litellm.utils import get_model_info
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from strenum import StrEnum

from par_ai_core.llm_config import LlmConfig
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.par_logging import console_err


class PricingDisplay(StrEnum):
    NONE = "none"
    PRICE = "price"
    DETAILS = "details"


pricing_lookup = {
    # OpenAI
    "gpt-4o": {
        "input": (2.50 / 1_000_000),
        "output": (10.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4o-latest": {
        "input": (5.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4o-2024-05-13": {
        "input": (5.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4o-mini": {
        "input": (0.15 / 1_000_000),
        "output": (0.6 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "o1": {
        "input": (15.0 / 1_000_000),
        "output": (60.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "o1-preview": {
        "input": (15.0 / 1_000_000),
        "output": (60.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "o1-mini": {
        "input": (3.0 / 1_000_000),
        "output": (12.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "o3-mini": {
        "input": (1.10 / 1_000_000),
        "output": (4.40 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4": {
        "input": (30.0 / 1_000_000),
        "output": (60.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4-32k": {
        "input": (60.0 / 1_000_000),
        "output": (120.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4-turbo": {
        "input": (10.0 / 1_000_000),
        "output": (30.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-4-turbo-2024-04-09": {
        "input": (10.0 / 1_000_000),
        "output": (30.0 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    "gpt-3.5-turbo-0125": {
        "input": (0.5 / 1_000_000),
        "output": (1.50 / 1_000_000),
        "cache_read": 0.5,
        "cache_write": 1,
    },
    # Anthropic
    "claude-3-5-sonnet-20240620": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 3.75,
    },
    "claude-3-5-sonnet-20241022": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 3.75,
    },
    "claude-3-5-sonnet-latest": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 3.75,
    },
    "claude-3-5-haiku-20241022": {
        "input": (1.0 / 1_000_000),
        "output": (5.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    "claude-3-5-haiku-latest": {
        "input": (1.0 / 1_000_000),
        "output": (5.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    "claude-3-haiku-20240307": {
        "input": (0.25 / 1_000_000),
        "output": (1.25 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    "claude-3-sonnet-20240229": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    "claude-3-opus-20240229": {
        "input": (15.0 / 1_000_000),
        "output": (75.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    # AWS
    "amazon.nova-micro-v1:0": {
        "input": (0.035 / 1_000_000),
        "output": (0.14 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    "amazon.nova-lite-v1:0": {
        "input": (0.06 / 1_000_000),
        "output": (0.24 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    "amazon.nova-pro-v1:0": {
        "input": (0.8 / 1_000_000),
        "output": (3.2 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input": (1.0 / 1_000_000),
        "output": (5.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1.25,
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 3.75,
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input": (3.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 3.75,
    },
    # Google
    "flash1.5": {
        "input": (0.075 / 1_000_000),
        "output": (0.30 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    "flash1.5-8b": {
        "input": (0.0375 / 1_000_000),
        "output": (0.15 / 1_000_000),
        "cache_read": 0.27,
        "cache_write": 1,
    },
    "pro1.5": {
        "input": (1.25 / 1_000_000),
        "output": (10.0 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    "pro1.0": {
        "input": (0.5 / 1_000_000),
        "output": (1.5 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    # XAI
    "grok-beta": {
        "input": (5.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "grok-vision-beta": {
        "input": (5.0 / 1_000_000),
        "output": (15.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "grok-2-vision": {
        "input": (2.0 / 1_000_000),
        "output": (10.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "grok-2": {
        "input": (2.0 / 1_000_000),
        "output": (10.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    # Deepseek
    "deepseek-chat": {
        "input": (0.14 / 1_000_000),
        "output": (0.28 / 1_000_000),
        "cache_read": 0.1,
        "cache_write": 1,
    },
    "deepseek-reasoner": {
        "input": (0.55 / 1_000_000),
        "output": (2.19 / 1_000_000),
        "cache_read": 0.25,
        "cache_write": 1,
    },
    # Mistral
    "mistral-large": {
        "input": (2.0 / 1_000_000),
        "output": (6.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "mistral-small": {
        "input": (0.1 / 1_000_000),
        "output": (0.3 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "mistral-8b": {
        "input": (0.1 / 1_000_000),
        "output": (0.1 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "mistral-3b": {
        "input": (0.04 / 1_000_000),
        "output": (0.04 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "mistral-embed": {
        "input": (0.1 / 1_000_000),
        "output": (0.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "mistral-moderation": {
        "input": (0.1 / 1_000_000),
        "output": (0.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "codestral": {
        "input": (0.3 / 1_000_000),
        "output": (0.9 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
    "pixtral-large": {
        "input": (2.0 / 1_000_000),
        "output": (6.0 / 1_000_000),
        "cache_read": 1,
        "cache_write": 1,
    },
}


def mk_usage_metadata() -> dict[str, int | float]:
    """Create a new usage metadata dictionary.

    Initializes a dictionary to track various usage metrics including:
    token counts, cache operations, tool calls, and costs.

    Returns:
        Dictionary with usage tracking fields initialized to zero
    """
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_write": 0,
        "cache_read": 0,
        "reasoning": 0,
        "successful_requests": 0,
        "tool_call_count": 0,
        "total_cost": 0.0,
    }


def get_api_cost_model_name(*, provider_name: str = "", model_name: str) -> str:
    """
    Get API cost model name

    If model name contains 'inference-profile', replace it with 'bedrock/'
    If provider_name is specified and model does not contain a '/', the provider is prepended to the model name with a '/'

    Args:
        provider_name: Provider name (optional, default is "")
        model_name: Model name to use

    Returns:
        API cost model name to use

    """
    if "inference-profile" in model_name:
        model_name = "bedrock/" + model_name.split("/")[-1]
    elif provider_name and "/" not in model_name:
        model_name = f"{provider_name.lower()}/{model_name}"
    model_name = model_name.replace("google/", "gemini/")
    if model_name.startswith("litellm/"):
        model_name = model_name.replace("litellm/", "")
    return model_name


def get_model_metadata(provider_name: str, model_name: str) -> ModelInfo:
    """
    Get model metadata from LiteLLM

    Args:
        provider_name: Provider name
        model_name: Model name

    Returns:
        ModelInfo: Model metadata
    """
    model_name = get_api_cost_model_name(provider_name=provider_name, model_name=model_name)
    return get_model_info(model=model_name)


def get_model_mode(
    provider: LlmProvider, model_name: str
) -> Literal["completion", "embedding", "image_generation", "chat", "audio_transcription", "unknown"]:
    """
    Get model mode

    Args:
        provider (LlmProvider): The provider
        model_name (str): The model name

    Returns:
        str: The model mode ("completion", "embedding", "image_generation", "chat", "audio_transcription", "unknown")
    """
    try:
        if provider == LlmProvider.OLLAMA:
            if "embed" in model_name:
                return "embedding"
            return "chat"
        metadata = get_model_metadata(provider.value.lower(), model_name)
        return metadata.get("mode") or "unknown"  # type: ignore
    except Exception:
        # console_err.print(f"Error getting model metadata {get_api_cost_model_name(provider_name=provider.value.lower(), model_name=model_name)}: {e}", severity="error")
        return "unknown"


def get_api_call_cost(
    *,
    llm_config: LlmConfig,
    usage_metadata: dict[str, int | float],
    batch_pricing: bool = False,
    model_name_override: str | None = None,
) -> float:
    """Calculate the cost of API calls based on usage.

    Computes the total cost of API usage taking into account:
    - Input and output tokens
    - Cache operations
    - Provider-specific pricing
    - Batch pricing discounts if applicable

    Args:
        llm_config: Configuration of the LLM used
        usage_metadata: Dictionary containing usage statistics
        batch_pricing: Whether to apply batch pricing discount
        model_name_override: Override the model name for pricing calculations

    Returns:
        Total cost in USD
    """
    if llm_config.provider in [LlmProvider.OLLAMA, LlmProvider.LLAMACPP, LlmProvider.GROQ, LlmProvider.GITHUB]:
        return 0
    batch_multiplier = 0.5 if batch_pricing else 1

    model_name = get_api_cost_model_name(
        provider_name=llm_config.provider, model_name=model_name_override or llm_config.model_name
    )
    # console_err.print(f"price model name {model_name}")

    try:
        if "deepseek" in model_name and "deepseek/" not in model_name:
            model_name = f"deepseek/{model_name}"
        model_info = get_model_info(model=model_name)
    except Exception as _:
        # console_err.print(f"No pricing data found for model {llm_config.provider.lower()}/{model_name}")
        return 0
    # console_err.print(usage_metadata)
    # console_err.print(model_info)

    total_cost = (
        (
            (usage_metadata["input_tokens"] - usage_metadata["cache_read"] - usage_metadata["cache_write"])
            * (model_info.get("input_cost_per_token") or 0)
        )
        + (
            usage_metadata["cache_read"]
            * (model_info.get("cache_read_input_token_cost") or (model_info.get("input_cost_per_token") or 0))
        )
        + (
            usage_metadata["cache_write"]
            * (model_info.get("cache_creation_input_token_cost") or (model_info.get("input_cost_per_token") or 0))
        )
        + (usage_metadata["output_tokens"] * (model_info.get("output_cost_per_token") or 0))
    )
    return total_cost * batch_multiplier

    # if model_name in pricing_lookup:
    #     model_info = pricing_lookup[model_name]
    #     total_cost = (
    #         (
    #             (usage_metadata["input_tokens"] - usage_metadata["cache_read"] - usage_metadata["cache_write"])
    #             * model_info["input"]
    #         )
    #         + (usage_metadata["cache_read"] * model_info["input"] * model_info["cache_read"])
    #         + (usage_metadata["cache_write"] * model_info["input"] * model_info["cache_write"])
    #         + (usage_metadata["output_tokens"] * model_info["output"])
    #     )
    #     return total_cost * batch_multiplier
    # else:
    #     console_err.print(f"No pricing data found for model {model_name}")

    return 0


def accumulate_cost(response: object | dict, usage_metadata: dict[str, int | float]) -> None:
    """
    Accumulate cost for the given response by finding token metadata in the response.

    Args:
        response: Response object or dictionary containing usage statistics
        usage_metadata: Dictionary to accumulate usage statistics
    """
    if isinstance(response, dict):
        usage_metadata["input_tokens"] += response.get("prompt_tokens", 0)
        usage_metadata["output_tokens"] += response.get("completion_tokens", 0)

        usage_metadata["input_tokens"] += response.get("input_tokens", 0)
        usage_metadata["output_tokens"] += response.get("output_tokens", 0)
        usage_metadata["total_tokens"] += response.get("input_tokens", 0) + response.get("output_tokens", 0)
        usage_metadata["cache_write"] += response.get("cache_creation_input_tokens", 0)
        usage_metadata["cache_read"] += response.get("cache_read_input_tokens", 0)
        return
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:  # type: ignore
        for key, value in response.usage_metadata.items():  # type: ignore
            if key in usage_metadata:
                usage_metadata[key] += value
            if key == "input_token_details":
                usage_metadata["cache_write"] += value.get("cache_creation", 0)
                usage_metadata["cache_read"] += value.get("cache_read", 0)
            if key == "output_token_details":
                usage_metadata["reasoning"] += value.get("reasoning", 0)
        return
    if (
        hasattr(response, "response_metadata")
        and response.response_metadata is not None  # type: ignore
        and "token_usage" in response.response_metadata  # type: ignore
    ):
        if not isinstance(response.response_metadata["token_usage"], dict):  # type: ignore
            response.response_metadata["token_usage"] = response.response_metadata["token_usage"].__dict__  # type: ignore

        for key, value in response.response_metadata["token_usage"].items():  # type: ignore
            if key in usage_metadata:
                usage_metadata[key] += value
            if key == "prompt_tokens":
                usage_metadata["input_tokens"] += value
            if key == "completion_tokens":
                usage_metadata["output_tokens"] += value
        return


def show_llm_cost(
    usage_metadata: dict[str, dict[str, int | float]],
    *,
    show_pricing: PricingDisplay = PricingDisplay.PRICE,
    console: Console | None = None,
) -> None:
    """
    Show LLM costs for all models captured in the usage_metadata dictionary.

    Args:
        usage_metadata: Dictionary containing usage statistics
        show_pricing: Display pricing options
        console: Optional console object to use for printing output

    """
    if show_pricing == PricingDisplay.NONE:
        return
    if not console:
        console = console_err
    grand_total: float = 0.0
    if show_pricing == PricingDisplay.PRICE:
        for m, u in usage_metadata.items():
            if "total_cost" in u:
                grand_total += u["total_cost"]
    else:
        for m, u in usage_metadata.items():
            cost = 0.0
            if "total_cost" in u:
                cost = u["total_cost"]
                grand_total += cost
            console.print(
                Panel.fit(
                    Pretty(u),
                    title=f"Model: [green]{m}[/green] Cost: [yellow]${cost:.5f}",
                    border_style="bold",
                )
            )
    console.print(f"Total Cost [yellow]${grand_total:.5f}")


# if __name__ == "__main__":
#     model_info = get_model_info(model="gpt-4o", custom_llm_provider="openai")
#     console_err.print(model_info)
