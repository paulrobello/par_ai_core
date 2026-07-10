# ENH-005 — Native async public API (avoid nested `asyncio.run`)

## Goal

Expose first-class `async` fetch/search entry points so async applications and notebooks await directly
instead of relying on `apply_nest_asyncio` monkey-patching, and so calls from within a running event
loop don't raise `RuntimeError`.

## Current-State Context

- `src/par_ai_core/web_tools.py:218-235` — `fetch_url` calls `asyncio.run(fetch_url_playwright(...))`.
  `asyncio.run` raises `RuntimeError: asyncio.run() cannot be called from a running event loop` when a
  loop is already running (common in async apps, FastAPI handlers, Jupyter).
- The project already ships `apply_nest_asyncio` (added 0.5.5) as a workaround — evidence the limitation
  is real and users hit it.
- `fetch_url_playwright` is already an `async def` coroutine; it just isn't public-facing as an awaitable.

## Implementation Steps

1. **Expose async entry points**: add public async wrappers that return the coroutine's result directly:
   ```python
   async def afetch_url(urls, *, fetch_using="playwright", **kwargs) -> list[str]:
       """Async-native fetch. Await this from within an event loop instead of fetch_url()."""
       # apply the same SSRF guard (SEC-001) here, then:
       if fetch_using == "playwright":
           return await fetch_url_playwright(urls, **_playwright_kwargs(kwargs))
       # selenium is sync; offload it:
       return await asyncio.to_thread(fetch_url_selenium, urls, **_selenium_kwargs(kwargs))
   ```
   Keep the existing sync `fetch_url` as a thin wrapper: when no loop is running, `asyncio.run(afetch_url(...))`;
   when a loop *is* running, raise a clear error pointing the caller to `afetch_url` (do not silently
   monkey-patch).
2. **Factor the SSRF guard** (SEC-001) into a shared helper called by both `fetch_url` and `afetch_url`
   so neither path can bypass it.
3. **Async search helpers** (optional, same pattern): where `search_utils` functions do blocking I/O,
   add `async` variants using `asyncio.to_thread` or the backends' native async clients.
4. **Document both entry points**: sync `fetch_url` for scripts, `afetch_url` for async contexts; note
   `apply_nest_asyncio` is no longer required when using the async API.

## Files to Touch

- `src/par_ai_core/web_tools.py` (`afetch_url`, shared guard helper, clearer sync-in-loop error)
- Optionally `src/par_ai_core/search_utils.py` (async search variants)
- README + `docs/operations.md` (sync vs async guidance)

## Verification

- `uv run python -c "import asyncio; from par_ai_core.web_tools import afetch_url; print(type(asyncio.run(afetch_url('https://example.com'))))"` prints `<class 'list'>`.
- Add a test that calling `afetch_url` inside a running loop succeeds (no `RuntimeError`), and that sync
  `fetch_url` inside a running loop raises a clear, actionable error.
- SSRF guard fires on both `fetch_url` and `afetch_url`.
- `make checkall && uv run pytest tests/test_web_tools.py -q`.

## Rollback Considerations

- Additive: sync `fetch_url` remains for existing callers. The only behavior change is a clearer error
  (instead of a raw `RuntimeError`) when sync is called inside a loop — document it in CHANGELOG.
- Revert by removing `afetch_url` and restoring the prior sync-only wrapper.
