# ENH-003 — Memoize model metadata, context sizes, and tiktoken encoders

## Goal

Cache the pure-function-of-model-name lookups (`_get_model_context_size`, `get_model_metadata`, tiktoken
encoder construction) so hot chunking/token-counting paths stop recomputing them.

## Current-State Context

- `src/par_ai_core/llm_utils.py:150-...` — `_get_model_context_size(model_name)` calls
  `get_model_metadata("", model_name)` and then a hardcoded fallback table, on every call.
  (Note: fix QA-002 first — the metadata lookup is currently dead due to `getattr` on a TypedDict.)
- `summarize_content` chunking calls context-size lookup repeatedly for the same model.
- tiktoken encoder construction (`tiktoken.encoding_for_model(...)` / `get_encoding(...)`) is expensive
  and is rebuilt per token count. Grep for `tiktoken` and `encoding_for_model` in `src/` to find sites.

## Implementation Steps

1. **Cache the context-size function**:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=256)
   def _get_model_context_size(model_name: str) -> int:
       ...
   ```
   `_get_model_context_size` takes a single hashable `str` arg — safe to `lru_cache` directly.
2. **Cache metadata** (if `get_model_metadata` is called elsewhere): wrap with `@lru_cache(maxsize=256)`
   keyed by `(provider, model_name)`. Ensure both args are hashable strings.
3. **Cache tiktoken encoders**: add a helper
   ```python
   @lru_cache(maxsize=32)
   def _get_encoder(model_name: str):
       import tiktoken
       try:
           return tiktoken.encoding_for_model(model_name)
       except KeyError:
           return tiktoken.get_encoding("cl100k_base")
   ```
   and route existing encoder construction through it.
4. **Interaction with ENH-004**: if pricing/metadata can be refreshed at runtime (TTL), expose a
   `_get_model_context_size.cache_clear()` call from the refresh hook so stale sizes don't persist.
   Document this coupling.

## Files to Touch

- `src/par_ai_core/llm_utils.py` (lru_cache on context size + metadata + encoder helper)
- Any module constructing tiktoken encoders inline (route through `_get_encoder`)

## Verification

- Add a test that calling `_get_model_context_size("gpt-4o")` twice invokes the underlying metadata
  lookup once (patch `get_model_metadata` and assert `call_count == 1`).
- Add a test that `_get_encoder` returns the same object for repeated same-model calls (`is` identity).
- Correctness unchanged: existing context-size tests still pass (run **after** QA-002 lands).
- `make checkall && uv run pytest tests/ -k "context or token or encode" -q`.

## Rollback Considerations

- Pure caching of deterministic functions — low risk. If a model's metadata changes within a process
  and must be observed live, either lower `maxsize` or expose `cache_clear()` (already recommended for
  the ENH-004 coupling).
- Remove the decorators to revert; no signature changes.
