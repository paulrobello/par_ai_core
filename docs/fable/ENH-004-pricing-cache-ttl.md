# ENH-004 — Cached, lazily-loaded pricing map with TTL refresh

## Goal

Load litellm's pricing/model map once, cache it, and refresh on a TTL (or explicit invalidation), so
cost tracking is both correct and cheap and litellm is imported lazily.

## Current-State Context

- `src/par_ai_core/pricing_lookup.py:29-30` — `litellm` imported at module top (heavy; folds with ARC-012
  which makes the import lazy).
- `get_model_metadata` / `get_api_call_cost` re-query litellm per call.
- ARC-012 replaces the hardcoded zero-cost provider list with data-driven detection — that detection also
  benefits from a cached pricing map.

## Implementation Steps

1. **Do ARC-012 first** (lazy litellm import + data-driven zero-cost). This enhancement builds on it.
2. **Add a cached pricing accessor**:
   ```python
   _pricing_cache: dict | None = None
   _pricing_loaded_monotonic: float | None = None
   _PRICING_TTL_SECONDS = 3600

   def _get_pricing_map(force: bool = False) -> dict:
       global _pricing_cache, _pricing_loaded_monotonic
       import time
       now = time.monotonic()
       if (force or _pricing_cache is None
               or (_pricing_loaded_monotonic is not None and now - _pricing_loaded_monotonic > _PRICING_TTL_SECONDS)):
           import litellm
           _pricing_cache = dict(litellm.model_cost)  # snapshot
           _pricing_loaded_monotonic = now
       return _pricing_cache
   ```
   Use `time.monotonic()` (not wall clock) for the TTL — note the Workflow/script constraint does not
   apply here; this is library runtime code where `time` is available.
3. **Route lookups** in `get_model_metadata`/`get_api_call_cost` through `_get_pricing_map()` instead of
   querying litellm directly.
4. **Expose a refresh hook**: `def refresh_pricing() -> None: _get_pricing_map(force=True)` for callers
   that want fresh data; document it.
5. **Coordinate with ENH-003**: if ENH-003 caches context sizes derived from pricing metadata, call its
   `cache_clear()` inside `refresh_pricing()`.

## Files to Touch

- `src/par_ai_core/pricing_lookup.py` (cache + accessor + `refresh_pricing`)
- Docs: note the TTL and refresh hook in `docs/operations.md`

## Verification

- Add a test that `_get_pricing_map()` imports litellm once across multiple calls within the TTL (patch
  the import / `litellm.model_cost` and assert single load).
- Add a test that `refresh_pricing()` forces a reload.
- Cost correctness unchanged: existing pricing tests pass (run after ARC-012).
- `make checkall && uv run pytest tests/ -k pricing -q`.

## Rollback Considerations

- The cache is process-local; a stale price within the TTL window is a minor cost-reporting inaccuracy,
  bounded by `_PRICING_TTL_SECONDS`. Lower the TTL or call `refresh_pricing()` to mitigate.
- Revert by routing lookups back through litellm directly and deleting the cache globals.
