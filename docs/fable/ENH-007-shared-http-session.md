# ENH-007 — Shared HTTP session / connection pooling for search + fetch

## Goal

Reuse a single pooled `httpx.Client`/`AsyncClient` (or `requests.Session`) across search-backend calls
and the HTTP fetch fast path (ENH-001), so repeated requests to the same hosts avoid new TCP/TLS
handshakes.

## Current-State Context

- `src/par_ai_core/search_utils.py` — search backends construct requests/clients per call (grep for
  `requests.get`, `httpx`, backend wrapper instantiation).
- ENH-001 introduces `fetch_url_http`, which currently opens a fresh `httpx.Client` per call — the same
  per-call-connection cost this enhancement removes.
- No shared session module exists today.

## Implementation Steps

1. **Add a shared-session module** `src/par_ai_core/http_session.py`:
   ```python
   import httpx
   _sync_client: httpx.Client | None = None
   _async_client: httpx.AsyncClient | None = None
   _DEFAULT_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)

   def get_sync_client(timeout: float = 10.0) -> httpx.Client:
       global _sync_client
       if _sync_client is None:
           _sync_client = httpx.Client(follow_redirects=True, timeout=timeout, limits=_DEFAULT_LIMITS)
       return _sync_client

   def get_async_client(timeout: float = 10.0) -> httpx.AsyncClient:
       global _async_client
       if _async_client is None:
           _async_client = httpx.AsyncClient(follow_redirects=True, timeout=timeout, limits=_DEFAULT_LIMITS)
       return _async_client

   def close_clients() -> None:
       global _sync_client, _async_client
       if _sync_client: _sync_client.close(); _sync_client = None
       # async client requires: await _async_client.aclose() — expose an async closer too
   ```
   Add `async def aclose_clients()` for the async client.
2. **Route `fetch_url_http` (ENH-001)** through `get_sync_client()` instead of constructing a client per
   call. Per-request settings that vary (auth, proxy, verify) are passed to `.get(...)`, not baked into the
   shared client — but note `verify`/`proxy` are client-level in httpx, so when `ignore_ssl=True` or a
   proxy is set, fall back to a per-call client (do not mutate the shared one).
3. **Route search backends** that do raw HTTP through the shared client where the backend allows a custom
   session/transport.
4. **Lifecycle**: document that long-running apps should call `close_clients()`/`await aclose_clients()`
   on shutdown; register nothing at import time (respect ARC-002's no-import-side-effects principle).

## Files to Touch

- `src/par_ai_core/http_session.py` (new)
- `src/par_ai_core/web_tools.py` (`fetch_url_http` uses the shared client)
- `src/par_ai_core/search_utils.py` (where a custom session can be injected)
- Docs: shutdown/close guidance in `docs/operations.md`

## Verification

- Add a test that two `fetch_url_http` calls with default settings reuse one client instance (patch
  `httpx.Client` and assert single construction), and that `ignore_ssl=True` uses a separate per-call client.
- Add a test that `close_clients()` resets the module globals.
- `make checkall && uv run pytest tests/ -k "http or session or search" -q`.

## Rollback Considerations

- The shared client is process-local; a leaked client holds sockets open until `close_clients()`. Mitigate
  with the documented shutdown hook. If pooling causes cross-request state issues (unlikely with httpx),
  revert `fetch_url_http` to per-call clients and delete the module.
- Do not share a client across event loops for the async case — `get_async_client` must be created within
  the running loop; document this constraint.
