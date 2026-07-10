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

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from par_ai_core.llm_config import LlmConfig
from par_ai_core.llm_providers import LlmProvider
from par_ai_core.par_logging import console_err

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # litellm is an optional backend (install with ``par_ai_core[pricing]``).
    # It is imported lazily inside the functions that use it so that merely
    # importing this module (and ``llm_utils``, which imports ``get_model_metadata``)
    # does not pull in the heavy litellm package.
    from litellm.types.utils import ModelInfo


def _import_get_model_info():
    """Import litellm's ``get_model_info`` lazily with an actionable error.

    Raises:
        ImportError: if litellm is not installed, pointing the user at the
            ``par_ai_core[pricing]`` extra.
    """
    try:
        from litellm.utils import get_model_info
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "LiteLLM pricing lookup requires the litellm package: pip install 'par_ai_core[pricing]'"
        ) from e
    return get_model_info


class PricingDisplay(StrEnum):
    """Controls the level of cost display detail.

    Members:
        NONE: Do not display any pricing information.
        PRICE: Display only the total cost.
        DETAILS: Display full usage breakdown and per-model costs.
    """

    NONE = "none"
    PRICE = "price"
    DETAILS = "details"


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
    return _import_get_model_info()(model=model_name)


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
        return metadata.get("mode") or "unknown"  # type: ignore[reportReturnType]
    except Exception:
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
    if llm_config.provider in [LlmProvider.OLLAMA, LlmProvider.LLAMACPP]:
        # Genuinely local backends with no per-token billing. Groq and GitHub
        # Models previously sat in this list too, but both now have paid tiers,
        # so their cost is derived from litellm's pricing data below instead of
        # being hardcoded to zero (was ARC-012).
        return 0
    batch_multiplier = 0.5 if batch_pricing else 1

    model_name = get_api_cost_model_name(
        provider_name=llm_config.provider, model_name=model_name_override or llm_config.model_name
    )

    try:
        if "deepseek" in model_name and "deepseek/" not in model_name:
            model_name = f"deepseek/{model_name}"
        get_model_info = _import_get_model_info()
        model_info = get_model_info(model=model_name)
    except Exception as e:
        # Log once per call so cost under-reporting is at least visible. The
        # default behavior (return 0) is preserved (QA-006).
        logger.warning("Could not look up pricing for model %r: %s", model_name, e)
        return 0

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


def accumulate_cost(response: object | dict, usage_metadata: dict[str, int | float]) -> None:
    """
    Accumulate cost for the given response by finding token metadata in the response.

    Args:
        response: Response object or dictionary containing usage statistics
        usage_metadata: Dictionary to accumulate usage statistics
    """
    if isinstance(response, dict):
        # Normalize the two token-naming conventions providers use so each token
        # is accumulated exactly once. ``prompt_tokens``/``completion_tokens``
        # alias ``input_tokens``/``output_tokens``; the previous code added both
        # (double-counting when a payload carried both) and derived
        # ``total_tokens`` from only one pair (QA-015). ``or`` (not ``+``) picks
        # the populated convention.
        input_tokens = response.get("input_tokens") or response.get("prompt_tokens") or 0
        output_tokens = response.get("output_tokens") or response.get("completion_tokens") or 0
        usage_metadata["input_tokens"] += input_tokens
        usage_metadata["output_tokens"] += output_tokens
        usage_metadata["total_tokens"] += input_tokens + output_tokens
        usage_metadata["cache_write"] += response.get("cache_creation_input_tokens", 0)
        usage_metadata["cache_read"] += response.get("cache_read_input_tokens", 0)
        return
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:  # type: ignore[reportAttributeAccessIssue]
        for key, value in response.usage_metadata.items():  # type: ignore[reportAttributeAccessIssue]
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
        and response.response_metadata is not None  # type: ignore[reportAttributeAccessIssue]
        and "token_usage" in response.response_metadata  # type: ignore[reportAttributeAccessIssue]
    ):
        if not isinstance(response.response_metadata["token_usage"], dict):  # type: ignore[reportAttributeAccessIssue]
            response.response_metadata["token_usage"] = response.response_metadata["token_usage"].__dict__  # type: ignore[reportAttributeAccessIssue]

        for key, value in response.response_metadata["token_usage"].items():  # type: ignore[reportAttributeAccessIssue]
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
        for _, u in usage_metadata.items():
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
