"""Callback Handler for tracking token usage, tool calling, and LLM interactions.

This module provides a custom callback handler for monitoring and managing
interactions with Large Language Models (LLMs). It offers functionality for:

1. Token usage tracking: Monitor input, output, and total token consumption.
2. Cost calculation: Compute and accumulate costs associated with LLM API calls.
3. Tool call tracking: Keep track of tool invocations made by the LLM.
4. Debug information: Optionally display prompts, completions, and tool calls.
5. Thread-safe operations: Ensure proper handling in multi-threaded environments.

Key Components:
- ParAICallbackHandler: The main callback handler class that inherits from
  BaseCallbackHandler and Serializable.
- get_parai_callback: A context manager for easy setup and teardown of the
  callback handler.

This module integrates with the par_ai_core ecosystem, utilizing components like
LlmConfig, llm_run_manager, and pricing_lookup for a cohesive experience in
managing LLM interactions and associated metadata.

Usage:
    with get_parai_callback(llm_config, show_prompts=True) as cb:
        # Your LLM interaction code here
        # The callback handler will automatically track usage and costs

Note: This module is designed to work seamlessly with LangChain and supports
various LLM providers through the LlmConfig system.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from par_ai_core.llm_config import LlmConfig, llm_run_manager
from par_ai_core.par_logging import console_err
from par_ai_core.pricing_lookup import (
    PricingDisplay,
    accumulate_cost,
    get_api_call_cost,
    get_api_cost_model_name,
    mk_usage_metadata,
    show_llm_cost,
)


class ParAICallbackHandler(BaseCallbackHandler, Serializable):
    """Callback Handler that tracks LLM usage and cost information.

    This handler monitors token usage, costs, and other metrics across
    different LLM providers. It supports thread-safe operation and
    provides detailed usage statistics.

    Attributes:
        llm_config: Configuration for the LLM being monitored
        show_prompts: Whether to display input prompts
        show_end: Whether to display completion information
        show_tool_calls: Whether to display tool call information
        verbose: Whether to display verbose output
    """

    llm_config: LlmConfig | None = None
    show_prompts: bool = False
    show_end: bool = False
    show_tool_calls: bool = False
    verbose: bool = False

    def __init__(
        self,
        *,
        llm_config: LlmConfig | None = None,
        show_prompts: bool = False,
        show_end: bool = False,
        show_tool_calls: bool = False,
        verbose: bool = False,
        console: Console | None = None,
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._usage_metadata = {}
        self._console = console or console_err
        self.llm_config = llm_config
        self.show_prompts = show_prompts
        self.show_end = show_end
        self.show_tool_calls = show_tool_calls
        self.verbose = verbose

    def __repr__(self) -> str:
        with self._lock:
            return self._usage_metadata.__repr__()

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def usage_metadata(self) -> dict[str, dict[str, int | float]]:
        """Get thread-safe COPY of usage metadata."""
        with self._lock:
            return deepcopy(self._usage_metadata)

    def _get_usage_metadata(self, provider_name: str, model_name: str) -> dict[str, int | float]:
        """
        Get usage metadata for model_name. Create if not found.

        Args:
            provider_name (str): Name of the LLM provider
            model_name (str): Name of the LLM model

        Returns:
            dict[str, int | float]: Usage metadata for model_name
        """
        model_name = get_api_cost_model_name(provider_name=provider_name, model_name=model_name)
        if model_name not in self._usage_metadata:
            self._usage_metadata[model_name] = mk_usage_metadata()
        return self._usage_metadata[model_name]

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Print out the prompts."""
        if self.show_prompts:
            console = kwargs.get("console", self._console)
            console.print(Panel(f"Prompt: {prompts[0]}", title="Prompt"))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        console = kwargs.get("console", self._console)
        if self.show_end:
            console.print(Panel(Pretty(response), title="LLM END"))
            console.print(Panel(Pretty(kwargs), title="LLM END KWARGS"))

        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        llm_config: LlmConfig | None = self.llm_config
        if "tags" in kwargs:
            for tag in reversed(kwargs["tags"]):
                if tag.startswith("config_id="):
                    config_id = tag[len("config_id=") :]
                    config = llm_run_manager.get_config(config_id)
                    llm_config = config[1] if config else None
                    break

        if not llm_config:
            if self.verbose:
                console.print(
                    "[yellow]Warning: config_id not found in on_llm_end did you forget to set a RunnableConfig?[/yellow]"
                )
        else:
            # update shared state behind lock
            with self._lock:
                if isinstance(generation, ChatGeneration):
                    if "model_name" in generation.message.response_metadata:
                        model_name = generation.message.response_metadata["model_name"]
                        # console.print(f"Overriding model_name from message.response_metadata with: {model_name}")
                    else:
                        model_name = llm_config.model_name
                        # console.print(f"using default model_name: {model_name}")

                    usage_metadata = self._get_usage_metadata(llm_config.provider, model_name)

                    if hasattr(generation.message, "tool_calls"):
                        usage_metadata["tool_call_count"] += len(generation.message.tool_calls)  # type: ignore

                    # Handle token usage from additional_kwargs
                    if "token_usage" in generation.message.additional_kwargs:
                        token_usage = generation.message.additional_kwargs["token_usage"]
                        usage_metadata["input_tokens"] += token_usage.get("prompt_tokens", 0)
                        usage_metadata["output_tokens"] += token_usage.get("completion_tokens", 0)
                        usage_metadata["total_tokens"] += token_usage.get("total_tokens", 0)
                    accumulate_cost(generation.message, usage_metadata)
                else:
                    model_name = llm_config.model_name
                    usage_metadata = self._get_usage_metadata(provider_name=llm_config.provider, model_name=model_name)
                    if response.llm_output and "token_usage" in response.llm_output:
                        accumulate_cost(response.llm_output, usage_metadata)
                usage_metadata["total_cost"] += get_api_call_cost(
                    llm_config=llm_config, usage_metadata=usage_metadata, model_name_override=model_name
                )
                usage_metadata["successful_requests"] += 1

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool starts running."""
        if not self.show_tool_calls:
            return
        console = kwargs.get("console", self._console)
        console.print(Panel(Pretty(inputs), title=f"Tool Call: {serialized['name']}"))

    def __copy__(self) -> "ParAICallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> "ParAICallbackHandler":
        """Return a deep copy of the callback handler."""
        return self


parai_callback_var: ContextVar[ParAICallbackHandler | None] = ContextVar("parai_callback", default=None)

register_configure_hook(parai_callback_var, True)


@contextmanager
def get_parai_callback(
    llm_config: LlmConfig | None = None,
    *,
    show_prompts: bool = False,
    show_end: bool = False,
    show_pricing: PricingDisplay = PricingDisplay.NONE,
    show_tool_calls: bool = False,
    verbose: bool = False,
    console: Console | None = None,
) -> Generator[ParAICallbackHandler, None, None]:
    """Get the llm callback handler in a context manager which exposes token / cost and debug information.

    Args:
        llm_config (LlmConfig): The LLM config.
        show_prompts (bool, optional): Whether to show prompts. Defaults to False.
        show_end (bool, optional): Whether to show end. Defaults to False.
        show_pricing (PricingDisplay, optional): Whether to show pricing. Defaults to PricingDisplay.NONE.
        show_tool_calls (bool, optional): Whether to show tool calls. Defaults to False.
        verbose (bool, optional): Whether to be verbose. Defaults to False.
        console (Console, optional): The console. Defaults to None.

    Returns:
        ParAICallbackHandler: The LLM callback handler.

    Example:
        >>> with get_parai_callback() as cb:
        ...     # All token usage and cost information will be captured
    """
    cb = ParAICallbackHandler(
        llm_config=llm_config,
        show_prompts=show_prompts,
        show_end=show_end,
        show_tool_calls=show_tool_calls,
        verbose=verbose,
        console=console,
    )
    parai_callback_var.set(cb)
    yield cb
    show_llm_cost(cb.usage_metadata, show_pricing=show_pricing, console=console)
    parai_callback_var.set(None)
