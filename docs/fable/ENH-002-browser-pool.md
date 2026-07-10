# ENH-002 — Persistent Playwright browser/context pool

## Goal

Let callers amortize one Chromium launch across many fetches instead of launching and closing a
browser per `fetch_url` call, while keeping per-fetch `BrowserContext` isolation and guaranteed
cleanup.

## Current-State Context

- `src/par_ai_core/web_tools.py:357-...` — `fetch_url_playwright` opens `async with async_playwright() as p`
  and `await p.chromium.launch(...)` on every call, then closes it. Launch (~300–800ms) dominates the
  cost and is paid per invocation.
- Each URL already gets its own `BrowserContext` (`browser.new_context(...)`) — that isolation must be
  preserved.

## Implementation Steps

1. **Add a pool object** in a new `src/par_ai_core/browser_pool.py`:
   ```python
   class PlaywrightBrowserPool:
       """Reusable Chromium instance. Create once, fetch many, close explicitly."""
       def __init__(self, headless: bool = True, proxy: ProxySettings | None = None):
           self._headless = headless
           self._proxy = proxy
           self._pw = None
           self._browser = None
           self._lock = asyncio.Lock()

       async def browser(self):
           async with self._lock:
               if self._browser is None:
                   self._pw = await async_playwright().start()
                   self._browser = await self._pw.chromium.launch(proxy=self._proxy, headless=self._headless)
               return self._browser

       async def aclose(self):
           if self._browser: await self._browser.close()
           if self._pw: await self._pw.stop()
           self._browser = self._pw = None
   ```
2. **Accept an optional pool** in `fetch_url_playwright(..., pool: PlaywrightBrowserPool | None = None)`:
   - When `pool` is provided, use `browser = await pool.browser()` and do **not** close it in `finally`
     (only close the per-URL `context`).
   - When `pool` is None, keep exact current launch-per-call behavior.
3. **Thread the parameter** through `fetch_url` as an optional `pool` argument (default None → unchanged
   behavior).
4. **Cleanup guarantees**: the pool is explicitly owned by the caller. Provide an async context manager
   (`async with PlaywrightBrowserPool() as pool:`) implementing `__aenter__/__aexit__` that calls
   `aclose()`. Document that the pool must be closed to avoid leaking a Chromium process.
5. **Concurrency**: the existing semaphore-limited parallelism inside `fetch_url_playwright` still applies
   per call; the pool only shares the browser process, and each fetch still gets an isolated context.

## Files to Touch

- `src/par_ai_core/browser_pool.py` (new)
- `src/par_ai_core/web_tools.py` (`fetch_url_playwright`, `fetch_url` optional `pool` param)
- `docs/operations.md` (usage + the must-close warning)

## Verification

- Add a test that two fetches through one pool call `chromium.launch` exactly once (mock `async_playwright`
  and assert launch call count == 1 across two `fetch_url_playwright(..., pool=pool)` calls).
- Add a test that `async with PlaywrightBrowserPool() as pool` closes the browser on exit.
- No-pool path unchanged: existing `test_web_tools.py` Playwright tests still pass.
- `make checkall && uv run pytest tests/test_web_tools.py -q`.

## Rollback Considerations

- Additive and opt-in: `pool=None` preserves current behavior exactly.
- Risk: a leaked pool leaves a Chromium process alive — mitigate with the context-manager API and a
  docstring warning. If problematic, delete `browser_pool.py` and the optional parameter.
