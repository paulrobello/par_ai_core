# ENH-006 — Robust provider fallback (retry + circuit breaker)

## Goal

Add an optional resilient-invoke helper that consumes `LlmConfig.fallback_models`, retrying transient
provider errors with exponential backoff and failing over to the next model, with a lightweight circuit
breaker to avoid hammering a down provider.

## Current-State Context

- `LlmConfig` already has a `fallback_models` field (`src/par_ai_core/llm_config.py`), but nothing in the
  library uses it for resilient invocation — it is declared and inert.
- `build_chat_model` returns a LangChain model; callers invoke it directly with no retry/failover.
- Cost tracking flows through `ParAICallbackHandler` / `get_parai_callback` (`provider_cb_info.py`) — the
  helper must preserve that.

## Implementation Steps

1. **Define transient-error classification**: a small predicate `_is_transient(exc) -> bool` matching rate
   limits (HTTP 429), 5xx, timeouts, and connection errors across providers (inspect exception type/status;
   LangChain surfaces provider errors — match on class name/status attributes conservatively).
2. **Add a resilient invoke helper** (new `src/par_ai_core/resilience.py` or in `llm_utils.py`):
   ```python
   def invoke_with_fallback(llm_config, messages, *, max_retries=3, base_delay=1.0, **invoke_kwargs):
       models = [llm_config.model_name, *(llm_config.fallback_models or [])]
       last_exc = None
       for model_name in models:
           cfg = dataclasses.replace(llm_config, model_name=model_name)   # or .clone() pre-ARC-007
           model = cfg.build_chat_model()
           for attempt in range(max_retries):
               try:
                   return model.invoke(messages, **invoke_kwargs)
               except Exception as e:
                   last_exc = e
                   if not _is_transient(e): break            # non-transient → next model
                   time.sleep(base_delay * (2 ** attempt))    # exponential backoff
       raise last_exc
   ```
3. **Circuit breaker** (simple, in-process): keep a module dict `{model_name: (failure_count, opened_monotonic)}`.
   Skip a model whose breaker is "open" (recent repeated failures) until a cooldown elapses; reset on success.
4. **Async variant**: mirror with `ainvoke` + `asyncio.sleep` (pairs with ENH-005).
5. **Preserve cost tracking**: run invocations inside the caller's `get_parai_callback` context; the helper
   must not swallow the usage metadata.
6. **Keep it opt-in**: existing direct-invoke callers are unaffected; this is an additional helper.

## Files to Touch

- `src/par_ai_core/resilience.py` (new) or `src/par_ai_core/llm_utils.py`
- Tests: `tests/test_resilience.py` (new)
- README (resilience section)

## Verification

- Add a test: a model that raises a transient error twice then succeeds returns the result after retries
  (patch `build_chat_model` to return a mock whose `invoke` fails-then-succeeds; assert 3 calls).
- Add a test: a non-transient error on the primary immediately fails over to the fallback model.
- Add a test: an open circuit breaker skips the failing model.
- Cost tracking still records usage when wrapped in `get_parai_callback`.
- `make checkall && uv run pytest tests/test_resilience.py -q`.

## Rollback Considerations

- Fully additive and opt-in. Circuit-breaker state is process-local and self-healing; if it misbehaves,
  disable it via a flag (`use_circuit_breaker=False`) or remove the module.
- Ensure retries respect a total time budget so a helper never hangs indefinitely under sustained failures.
