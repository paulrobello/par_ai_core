# ENH-008 — Structured observability hooks (OpenTelemetry)

## Goal

Add optional OpenTelemetry spans/metrics around LLM calls and fetches so production consumers can export
token usage, cost, and latency to their observability stack, with a no-op default when OTel isn't
installed. Pairs with ARC-002 (opt-in, no import-time side effects).

## Current-State Context

- Cost/usage are surfaced only via Rich console printing in `provider_cb_info.py` (`ParAICallbackHandler`,
  `show_llm_cost`) — no machine-exportable telemetry.
- ARC-002 makes logging opt-in via `init_logging()`; observability should follow the same opt-in,
  no-side-effects-on-import pattern.
- `ParAICallbackHandler` already sits on every LangChain run and accumulates `usage_metadata` — the
  natural emission point.

## Implementation Steps

1. **Add an optional dependency**: `opentelemetry-api` (and `opentelemetry-sdk` for the example) under a
   new `[otel]` extra in `pyproject.toml` (respect ARC-001's extras pattern).
2. **Guarded import + no-op fallback** in a new `src/par_ai_core/observability.py`:
   ```python
   try:
       from opentelemetry import trace, metrics
       _OTEL = True
   except ImportError:
       _OTEL = False

   def get_tracer():
       return trace.get_tracer("par_ai_core") if _OTEL else _NoopTracer()
   ```
   Provide `_NoopTracer`/`_NoopSpan` context managers so callers write the same code whether or not OTel
   is installed.
3. **Instrument the callback**: in `ParAICallbackHandler`, when a run ends, emit a span with attributes
   (`llm.provider`, `llm.model`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`) and record
   latency; emit metrics counters/histograms (tokens, cost, duration). Guard all of it behind `_OTEL`.
4. **Instrument fetches** (optional): wrap `fetch_url`/`afetch_url` in a span with `http.url` (redacted of
   credentials — see SEC-008) and duration.
5. **Opt-in only**: nothing is emitted unless OTel is installed *and* the consumer has configured a
   provider. No tracer/meter is created at import; follow ARC-002 — expose an `init_observability()` if any
   setup is needed, otherwise rely on the global OTel provider the consumer configures.
6. **Redaction**: never put credentials or full prompt text in span attributes; token counts and model
   names only.

## Files to Touch

- `src/par_ai_core/observability.py` (new — guarded import, no-op fallback)
- `src/par_ai_core/provider_cb_info.py` (emit spans/metrics on run end)
- `src/par_ai_core/web_tools.py` (optional fetch spans)
- `pyproject.toml` (`[otel]` extra)
- README + `docs/operations.md` (observability section)

## Verification

- With OTel **not** installed: `uv run python -c "import par_ai_core.observability as o; import par_ai_core.provider_cb_info"` succeeds and no telemetry is emitted (no import error, no-op tracer).
- With OTel installed and an in-memory span exporter configured in a test: an LLM run produces one span
  with the expected token/cost attributes and no credential/prompt leakage.
- Confirm no tracer is created at import time (`assert 'opentelemetry' not in sys.modules` after importing
  the package when the `[otel]` extra is absent).
- `make checkall && uv run pytest tests/ -k observ -q`.

## Rollback Considerations

- Fully optional and additive: absent the `[otel]` extra, the code is a no-op and imports cleanly.
- Spans/metrics must never raise into the LLM call path — wrap emission in a defensive `try/except` that
  logs at debug and continues, so observability failures never break inference.
- Revert by removing the emission calls and the `observability` module; no core behavior depends on it.
