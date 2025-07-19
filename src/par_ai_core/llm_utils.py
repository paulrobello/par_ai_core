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
import re

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from par_ai_core.llm_config import LlmConfig, ReasoningEffort, llm_run_manager
from par_ai_core.llm_providers import LlmProvider, provider_base_urls, provider_default_models, provider_env_key_names
from par_ai_core.pricing_lookup import get_model_metadata


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


def _get_model_context_size(model_name: str) -> int:
    """
    Get the maximum context size for a model.

    Args:
        model_name: Name of the model

    Returns:
        Maximum context size in tokens
    """
    try:
        # Try to get model metadata from pricing lookup
        model_info = get_model_metadata("", model_name)
        # Try different fields that might contain context size
        context_size = (
            getattr(model_info, "max_input_tokens", None)
            or getattr(model_info, "max_tokens", None)
            or getattr(model_info, "context_length", None)
        )
        if context_size:
            return int(context_size)
    except Exception:
        pass

    # Fallback to hardcoded values for common models
    context_sizes = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16384,
        "claude-3-sonnet": 200000,
        "claude-3-opus": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-sonnet-4": 200000,
        "gemini-1.5-pro": 2000000,
        "gemini-1.5-flash": 1000000,
    }

    # Check for exact match or partial match
    for model_key, size in context_sizes.items():
        if model_key in model_name.lower():
            return size

    # Conservative default
    return 8192


def _estimate_tokens(text: str, model_name: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Args:
        text: Text to estimate tokens for
        model_name: Name of the model (used to choose appropriate tokenizer)

    Returns:
        Estimated number of tokens
    """
    try:
        # Try to get appropriate tokenizer for the model
        if "gpt" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "claude" in model_name.lower():
            # Claude uses a similar tokenizer to GPT
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Default to GPT-4 tokenizer
            encoding = tiktoken.encoding_for_model("gpt-4")

        return len(encoding.encode(text))
    except Exception:
        # Fallback: estimate 4 characters per token (rough approximation)
        return len(text) // 4


def _chunk_text(text: str, max_tokens: int, model_name: str) -> list[str]:
    """
    Split text into chunks that fit within token limits.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        model_name: Model name for token estimation

    Returns:
        List of text chunks
    """
    if _estimate_tokens(text, model_name) <= max_tokens:
        return [text]

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the limit
        test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
        if _estimate_tokens(test_chunk, model_name) <= max_tokens:
            current_chunk = test_chunk
        else:
            # If current chunk is not empty, save it
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # If single paragraph is too large, split by sentences
            if _estimate_tokens(paragraph, model_name) > max_tokens:
                sentences = re.split(r"[.!?]+", paragraph)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    sentence = sentence.strip() + "."

                    test_chunk = current_chunk + (" " if current_chunk else "") + sentence
                    if _estimate_tokens(test_chunk, model_name) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                current_chunk = paragraph

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_content(content: str, llm: BaseChatModel) -> str:
    """Summarize content using an LLM with context size awareness and chunking.

    Args:
        content: Text content to summarize
        llm: Language model to use for summarization

    Returns:
        A structured summary containing:
        - Title
        - Key points
        - Summary paragraph
    """
    model_name = getattr(llm, "model_name", "") or getattr(llm, "name", "") or "gpt-4"

    # Get model context size
    context_size = _get_model_context_size(model_name)

    # Reserve tokens for system message, response, and safety margin
    system_message_tokens = 100  # Rough estimate
    response_tokens = 500  # Expected response size
    safety_margin = 200  # Safety buffer

    max_content_tokens = context_size - system_message_tokens - response_tokens - safety_margin

    # Check if content fits within context size
    content_tokens = _estimate_tokens(content, model_name)

    if content_tokens <= max_content_tokens:
        # Content fits, use normal summarization
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
    else:
        # Content is too large, chunk it
        chunks = _chunk_text(content, max_content_tokens, model_name)

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_instructions = f"""Your goal is to generate a summary of the user provided content.

            This is part {i + 1} of {len(chunks)} chunks from a larger document that exceeded the context size limit.

            Your response should include the following:
            - Title (if this chunk contains title information)
            - Key points from this section
            - Summary of this section

            Do not include the content itself.
            Do not include a preamble such as "Summary of the content:"
            Focus on the most important information from this section.
            """

            chunk_summary = str(
                llm.invoke(
                    [
                        SystemMessage(content=chunk_instructions),
                        HumanMessage(content=chunk),
                    ],
                    config=llm_run_manager.get_runnable_config(llm.name or ""),
                ).content
            )
            chunk_summaries.append(chunk_summary)

        # Combine chunk summaries into a final summary
        combined_summaries = "\n\n".join(chunk_summaries)

        final_instructions = f"""Your goal is to create a unified summary from the provided chunk summaries.

        The original content was split into {len(chunks)} chunks due to context size limits, and each chunk was summarized separately. Your task is to combine these summaries into a coherent final summary.

        Your response should include the following:
        - Title (synthesized from all chunks)
        - List of key points (consolidated from all chunks)
        - Summary (unified overview of the entire content)

        Do not include the chunk summaries themselves.
        Do not include a preamble such as "Summary of the content:"
        Focus on creating a cohesive summary that captures the main themes and important details from all chunks.
        """

        return str(
            llm.invoke(
                [
                    SystemMessage(content=final_instructions),
                    HumanMessage(content=combined_summaries),
                ],
                config=llm_run_manager.get_runnable_config(llm.name or ""),
            ).content
        )
