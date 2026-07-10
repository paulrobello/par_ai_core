# ENH-001 — HTTP-first fetch fast path with browser fallback

## Goal

Add a lightweight HTTP fetch path (via `httpx`) that returns static-page HTML without launching a
browser, and an `auto` mode that escalates to Playwright/Selenium only when a page requires JS
rendering. Preserve the SSRF guard (SEC-001) and the error contract (QA-006).

## Current-State Context

- `src/par_ai_core/web_tools.py:180-256` — `fetch_url(urls, fetch_using="playwright", ...)` dispatches
  to `fetch_url_playwright` (via `asyncio.run`) or `fetch_url_selenium`. There is no non-browser path.
- Every fetch pays browser launch cost (~300–800ms) even for static HTML.
- `fetch_using` is currently `Literal["playwright", "selenium"]`.
- Downstream `search_utils.brave_search`/`serper_search` with `scrape=True` fetch many URLs in bulk —
  the biggest beneficiary.

## Implementation Steps

1. **Add the dependency** (respect ARC-001 extras): add `httpx` to the `[web]` extra (or core if
   already present). Confirm with `grep -n httpx pyproject.toml`.
2. **Add an HTTP fetcher** in `web_tools.py`:
   ```python
   def fetch_url_http(
       urls: str | list[str],
       *,
       timeout: int = 10,
       proxy_config: ProxySettings | None = None,
       http_credentials: HttpCredentials | None = None,
       verbose: bool = False,
       ignore_ssl: bool = False,
       console: Console | None = None,
   ) -> list[str]:
       """Fetch static HTML via httpx without a browser. No JS execution."""
       import httpx
       if isinstance(urls, str):
           urls = [urls]
       results: list[str] = []
       auth = None
       verify = not ignore_ssl
       headers = {"User-Agent": get_random_user_agent()}
       with httpx.Client(follow_redirects=True, timeout=timeout, verify=verify,
                         proxy=(proxy_config or {}).get("server") if proxy_config else None) as client:
           for url in urls:
               try:
                   if http_credentials and "username" in http_credentials:
                       auth = (http_credentials["username"], http_credentials["password"])
                   resp = client.get(url, auth=auth, headers=headers)
                   results.append(resp.text)
               except Exception as e:
                   logger.warning("HTTP fetch failed for %s: %s", url, e)
                   results.append("")
       return results
   ```
3. **Wire into dispatch**: extend `fetch_using` to `Literal["playwright", "selenium", "http", "auto"]`.
   - `"http"` → call `fetch_url_http`.
   - `"auto"` → call `fetch_url_http` first; for each result that looks JS-gated (empty body, or a
     known SPA marker like a root `<div id="root"></div>` with little text), re-fetch that URL with
     Playwright. Implement the heuristic as a small helper `_looks_js_rendered(html: str) -> bool`
     (e.g. `len(BeautifulSoup(html,"html.parser").get_text(strip=True)) < 200`).
4. **Keep the SSRF guard**: the validation block at the top of `fetch_url` (SEC-001) runs before
   dispatch, so all paths inherit it. Do not add a second bypass in `fetch_url_http`.
5. **Keep the error contract** (QA-006): `fetch_url_http` logs and returns `""` per failed URL, matching
   the other fetchers; honor `raise_on_error` if that parameter was added.
6. **Docs**: document the new modes in the `fetch_url` docstring and `docs/operations.md`; note that
   `http`/`auto` do not execute JavaScript.

## Files to Touch

- `src/par_ai_core/web_tools.py` (new `fetch_url_http`, dispatch, heuristic helper)
- `pyproject.toml` (`httpx` in `[web]` extra)
- `docs/operations.md`, README fetch section

## Verification

- `uv run python -c "from par_ai_core.web_tools import fetch_url; print(bool(fetch_url('https://example.com', fetch_using='http')[0]))"` prints `True`.
- SSRF guard still fires: `fetch_url('file:///etc/passwd', fetch_using='http')` raises `ValueError`.
- Add tests: `http` mode returns HTML for a mocked 200; `auto` escalates to browser when the HTTP body
  is JS-gated (mock `_looks_js_rendered` → True and assert Playwright path is called).
- `make checkall && uv run pytest tests/test_web_tools.py -q`.

## Rollback Considerations

- Fully additive: default `fetch_using="playwright"` is unchanged, so existing callers are unaffected.
- If `httpx` proves problematic, remove the `http`/`auto` branches and the dependency; no other code
  depends on them.
