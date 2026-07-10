# PAR AI Core — Enhancement Ideas

> Opportunities **beyond** the defect findings in `AUDIT.md`. These are performance and
> capability improvements, prioritized by impact-to-effort. Each has a full implementation
> plan under `docs/fable/ENH-XXX-<slug>.md`, written to be executable by a smaller model.
>
> Prerequisite note: several enhancements assume the `AUDIT.md` fixes have landed — in
> particular the optional-extras split (ARC-001), the SSRF guard (SEC-001), and the
> web-fetch error contract (QA-006). Cross-references are called out per idea.

## Priority Order (impact / effort)

| ID | Title | Impact | Effort | Depends on |
|----|-------|:------:|:------:|-----------|
| ENH-001 | HTTP-first fetch fast path with browser fallback | High | Medium | SEC-001, QA-006 |
| ENH-002 | Persistent Playwright browser/context pool | High | Medium | — |
| ENH-003 | Memoize model metadata, context sizes, tiktoken encoders | Medium | Low | QA-002 |
| ENH-004 | Cached, lazily-loaded pricing map with TTL refresh | Medium | Low–Med | ARC-012 |
| ENH-005 | Native async public API (avoid nested `asyncio.run`) | Med–High | Medium | — |
| ENH-006 | Robust provider fallback (retry + circuit breaker) | Medium | Medium | — |
| ENH-007 | Shared HTTP session / connection pooling for search + fetch | Medium | Low–Med | ENH-001 |
| ENH-008 | Structured observability hooks (OpenTelemetry) | Medium | Medium | ARC-002 |

---

## ENH-001 — HTTP-first fetch fast path with browser fallback

Today every `fetch_url` call spins up a full Playwright/Selenium browser even for static HTML that a plain HTTP GET would return in milliseconds. Add a lightweight `httpx`/`requests` fetch path that is tried first (opt-in via `fetch_using="http"` or an `auto` mode that escalates to a browser only when the page needs JS rendering). For the large fraction of scraping targets that are static, this cuts latency from seconds to tens of milliseconds and eliminates the browser's memory/CPU cost entirely.

- **Expected impact**: High — order-of-magnitude latency reduction and near-zero resource use for static pages; makes bulk fetching (search `scrape=True`) dramatically cheaper.
- **Effort**: Medium — new fetch function, an `auto` heuristic, and wiring into the existing dispatch; must preserve the SSRF guard and error contract.

## ENH-002 — Persistent Playwright browser/context pool

`fetch_url_playwright` launches a fresh Chromium via `async_playwright()` on every call and closes it afterward. Browser launch is the dominant cost (~300–800ms) and it is paid per invocation even when fetching many URLs across separate calls. Introduce an optional reusable browser pool (a context manager / singleton with explicit lifecycle) so callers doing repeated fetches amortize one launch across many pages, with per-fetch isolated `BrowserContext`s for safety.

- **Expected impact**: High for workloads that fetch repeatedly (agents, search scraping) — removes per-call launch overhead.
- **Effort**: Medium — lifecycle management, cleanup guarantees, and a fallback to the current launch-per-call behavior when no pool is active.

## ENH-003 — Memoize model metadata, context sizes, and tiktoken encoders

`_get_model_context_size`, `get_model_metadata`, and tiktoken encoder construction are recomputed on every call. These are pure functions of the model name and are hot in chunking/token-counting paths. Wrap them in `functools.lru_cache` (encoders) and a small dict cache (metadata/context size) so repeated lookups are free.

- **Expected impact**: Medium — removes redundant litellm lookups and expensive `tiktoken.encoding_for_model` construction from tight loops (e.g. `summarize_content` chunking).
- **Effort**: Low — decorators plus cache-invalidation consideration for the pricing refresh (ENH-004).

## ENH-004 — Cached, lazily-loaded pricing map with TTL refresh

Pricing/zero-cost detection currently imports heavy `litellm` at module load and re-queries it per call. Load the pricing map once, cache it, and refresh on a TTL (or explicit invalidation). Combined with ARC-012's data-driven zero-cost detection, this makes cost tracking both correct and cheap.

- **Expected impact**: Medium — faster imports (folds with ARC-012), consistent cost data, no repeated litellm calls.
- **Effort**: Low–Medium — cache layer plus a documented refresh hook.

## ENH-005 — Native async public API (avoid nested `asyncio.run`)

`fetch_url` calls `asyncio.run(fetch_url_playwright(...))`, which cannot be called from within an already-running event loop (raises `RuntimeError`) — a real limitation for async applications and notebooks (the project already ships `apply_nest_asyncio` as a workaround). Expose first-class `async def afetch_url(...)` (and async search helpers) so async callers await directly instead of relying on `nest_asyncio` monkey-patching.

- **Expected impact**: Medium–High — makes the library correct and idiomatic inside async apps/agents; removes reliance on `nest_asyncio`.
- **Effort**: Medium — expose the async coroutines publicly, keep the sync wrappers, document the two entry points.

## ENH-006 — Robust provider fallback (retry + circuit breaker)

`LlmConfig` has a `fallback_models` field, but there is no resilient invocation wrapper that retries transient provider errors (rate limits, 5xx, timeouts) with exponential backoff and fails over to the next model, with a circuit breaker to avoid hammering a down provider. Add an optional resilient-invoke helper that consumes `fallback_models`.

- **Expected impact**: Medium — meaningfully improves reliability of downstream apps under provider flakiness.
- **Effort**: Medium — a retry/backoff wrapper plus simple circuit-breaker state; must integrate with the callback/cost tracking.

## ENH-007 — Shared HTTP session / connection pooling for search + fetch

Search backends and the HTTP fetch fast path (ENH-001) create new connections per request. A shared, reused `httpx.Client`/`AsyncClient` (or `requests.Session`) with connection pooling and sane timeouts reduces TCP/TLS handshake overhead for repeated calls to the same hosts (search APIs, scraped domains).

- **Expected impact**: Medium — lower latency and fewer sockets for repeated same-host requests.
- **Effort**: Low–Medium — a shared session module with lifecycle management; depends on ENH-001 for the fetch side.

## ENH-008 — Structured observability hooks (OpenTelemetry)

Cost and latency are currently surfaced via Rich console printing. Add optional OpenTelemetry spans/metrics (or a pluggable callback) around LLM calls and fetches so production consumers can export token usage, cost, and latency to their observability stack. Pairs naturally with the ARC-002 logging refactor (opt-in, no import-time side effects).

- **Expected impact**: Medium — production observability without bolting on custom instrumentation.
- **Effort**: Medium — optional dependency, span/metric emission, and a no-op default when OTel isn't installed.
