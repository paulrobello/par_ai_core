"""
Utilities for LLM (Large Language Model) setup and operations.

This module provides helper functions and utilities for configuring and
interacting with Large Language Models. It includes functionality for:

1. Creating LLM configurations from environment variables
2. Summarizing content using LLMs

The module is designed to work with various LLM providers and offers
flexible configuration options through environment variables.

Key functions:
- llm_config_from_env: Creates an LlmConfig instance from environment variables
- summarize_content: Generates a structured summary of given content using an LLM

This module is part of the par_ai_core package and relies on other
components such as llm_config, llm_providers, and langchain integrations.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from par_ai_core.llm_config import LlmConfig, ReasoningEffort, llm_run_manager
from par_ai_core.llm_providers import LlmProvider, provider_base_urls, provider_default_models, provider_env_key_names


def llm_config_from_env(prefix: str = "PARAI") -> LlmConfig:
    """
    Create instance of LlmConfig from environment variables.
    The following environment variables are used:

    - {prefix}_AI_PROVIDER (required)
    - {prefix}_MODEL (optional - defaults to provider default)
    - {prefix}_AI_BASE_URL (optional - defaults to provider default)
    - {prefix}_TEMPERATURE (optional - defaults to 0.8)
    - {prefix}_USER_AGENT_APPID (optional)
    - {prefix}_STREAMING (optional - defaults to false)
    - {prefix}_MAX_CONTEXT_SIZE (optional - defaults to provider default)

    Args:
        prefix: Prefix to use for environment variables (default: "PARAI")

    Returns:
        LlmConfig
    """
    prefix = prefix.strip("_")
    ai_provider_name = os.environ.get(f"{prefix}_AI_PROVIDER")
    if not ai_provider_name:
        raise ValueError(f"{prefix}_AI_PROVIDER environment variable not set.")

    ai_provider = LlmProvider(ai_provider_name)
    if ai_provider not in [LlmProvider.OLLAMA, LlmProvider.LLAMACPP, LlmProvider.BEDROCK]:
        key_name = provider_env_key_names[ai_provider]
        if not os.environ.get(key_name):
            raise ValueError(f"{key_name} environment variable not set.")

    model_name = os.environ.get(f"{prefix}_MODEL") or provider_default_models[ai_provider]
    if not model_name:
        raise ValueError(f"{prefix}_MODEL environment variable not set.")

    ai_base_url = os.environ.get(f"{prefix}_AI_BASE_URL") or provider_base_urls[ai_provider]
    temperature = float(os.environ.get(f"{prefix}_TEMPERATURE", "0.8"))
    user_agent_appid = os.environ.get(f"{prefix}_USER_AGENT_APPID")
    streaming = os.environ.get(f"{prefix}_STREAMING", "false") == "true"
    num_ctx = os.environ.get(f"{prefix}_NUM_CTX")
    if num_ctx is not None:
        num_ctx = int(num_ctx)
        if num_ctx < 0:
            num_ctx = 0

    timeout = os.environ.get(f"{prefix}_TIMEOUT")
    if timeout is not None:
        timeout = int(timeout)
    num_predict = os.environ.get(f"{prefix}_NUM_PREDICT")
    if num_predict is not None:
        num_predict = int(num_predict)
    repeat_last_n = os.environ.get(f"{prefix}_REPEAT_LAST_N")
    if repeat_last_n is not None:
        repeat_last_n = int(repeat_last_n)
    repeat_penalty = os.environ.get(f"{prefix}_REPEAT_PENALTY")
    if repeat_penalty is not None:
        repeat_penalty = float(repeat_penalty)
    mirostat = os.environ.get(f"{prefix}_MIROSTAT")
    if mirostat is not None:
        mirostat = int(mirostat)
    mirostat_eta = os.environ.get(f"{prefix}_MIROSTAT_ETA")
    if mirostat_eta is not None:
        mirostat_eta = float(mirostat_eta)
    mirostat_tau = os.environ.get(f"{prefix}_MIROSTAT_TAU")
    if mirostat_tau is not None:
        mirostat_tau = float(mirostat_tau)
    tfs_z = os.environ.get(f"{prefix}_TFS_Z")
    if tfs_z is not None:
        tfs_z = float(tfs_z)
    top_k = os.environ.get(f"{prefix}_TOP_K")
    if top_k is not None:
        top_k = int(top_k)
    top_p = os.environ.get(f"{prefix}_TOP_P")
    if top_p is not None:
        top_p = float(top_p)
    seed = os.environ.get(f"{prefix}_SEED")
    if seed is not None:
        seed = int(seed)

    reasoning_effort = os.environ.get(f"{prefix}_REASONING_EFFORT")
    if reasoning_effort not in [None, "low", "medium", "high"]:
        raise ValueError(f"{prefix}_REASONING_EFFORT must be one of 'low', 'medium', or 'high'")
    if reasoning_effort is not None:
        reasoning_effort = ReasoningEffort(reasoning_effort)

    reasoning_budget = os.environ.get(f"{prefix}_REASONING_BUDGET")
    if reasoning_budget is not None:
        reasoning_budget = int(reasoning_budget)
        if not reasoning_budget:
            reasoning_budget = None

    return LlmConfig(
        provider=ai_provider,
        model_name=model_name,
        base_url=ai_base_url,
        temperature=temperature,
        user_agent_appid=user_agent_appid,
        streaming=streaming,
        num_ctx=num_ctx,
        env_prefix=prefix,
        timeout=timeout,
        num_predict=num_predict,
        repeat_last_n=repeat_last_n,
        repeat_penalty=repeat_penalty,
        mirostat=mirostat,
        mirostat_eta=mirostat_eta,
        mirostat_tau=mirostat_tau,
        tfs_z=tfs_z,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        reasoning_effort=reasoning_effort,
        reasoning_budget=reasoning_budget,
    )


def summarize_content(content: str, llm: BaseChatModel) -> str:
    """Summarize content using an LLM.

    Args:
        content: Text content to summarize
        llm: Language model to use for summarization

    Returns:
        A structured summary containing:
        - Title
        - Key points
        - Summary paragraph
    """
    summarize_content_instructions = """Your goal is to generate a summary of the user provided content.

    Your response should include the following:
    - Title
    - List of key points
    - Summary

    Do not include the content itself.
    Do not include a preamble such as "Summary of the content:"
    """
    return str(
        llm.invoke(
            [
                SystemMessage(content=summarize_content_instructions),
                HumanMessage(content=content),
            ],
            config=llm_run_manager.get_runnable_config(llm.name or ""),
        ).content
    )
