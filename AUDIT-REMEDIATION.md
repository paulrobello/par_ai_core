# Audit Remediation Report

> **Project**: PAR AI Core (`par_ai_core`) v0.5.8
> **Audit Date**: 2026-07-09 (see `AUDIT.md`)
> **Remediation Date**: 2026-07-10
> **Severity Filter Applied**: `all` (every phase executed)
> **Branch**: `fix/audit-remediation` (base `4cd3b80`; HEAD `0d70bc6`)
> **Follow-up**: backlog refactors ARC-010/013/014 were implemented after the initial phased run (commits `39de403`, `0d70bc6`); only ARC-008 remains deferred.

---

## Execution Summary

| Phase | Status | Agent | Issues Targeted | Resolved | Deferred | Manual |
|-------|--------|-------|----------------:|---------:|---------:|-------:|
| 1 — Critical Security | ✅ | fix-security | 6 | 6 | 0 | 0 |
| 2 — Critical Architecture | ✅ | fix-architecture | 4 (+3 folded) | 7 | 0 | 0 |
| 3a — Security (remaining) | ✅ | fix-security | 4 | 4 ¹ | 0 | 0 |
| 3b — Architecture (remaining) | ✅ | fix-architecture | 17 | 12 | 4 ³ | 0 |
| 3c — Code Quality (all) | ✅ | fix-code-quality | 26 | 26 ² | 0 | 0 |
| 3d — Documentation (all) | ✅ | fix-documentation | 18 | 18 | 0 | 0 |
| 4 — Verification | ✅ | — | — | — | — | — |
| 3e — Backlog refactors (follow-up) | ✅ | fix-architecture | 3 | 3 | 0 | 0 |

¹ SEC-004/005 required no change — `.idea/` and build/coverage artifacts were already untracked and gitignored at the audit base (the audit's tracking claim was stale against current HEAD). SEC-009/010 were resolved by change.
² Includes 8 cross-reference IDs resolved in earlier phases (QA-004/008/013/014/016/023/025 done, QA-026 folded into DOC-001) plus 18 active fixes this phase.
³ Three of Phase 3b's four deferred refactors (ARC-010, ARC-013, ARC-014) were completed in the Phase 3e follow-up; only ARC-008 remains deferred.

**Overall: 77 of 78 issues resolved. 1 deferred (ARC-008, backlog-tier). 0 partial. 0 require human decision-making beyond ARC-008.**

By domain: Security 10/10, Architecture 23/24 (1 deferred — ARC-008), Code Quality 26/26, Documentation 18/18.

### Orchestrator deviation (documented)
Phase 3 prescribes running the four domain agents **in parallel**. This audit's File Conflict Map shows heavy multi-domain overlap on the same core files (`web_tools.py`, `llm_config.py`, `utils.py`, `search_utils.py`, `provider_cb_info.py` each touched by 3–4 domains). Per the orchestrator's own escape clause ("serialize their work rather than reaching for worktrees" when domains heavily overlap on the same core files) and to avoid silent overwrites, the four Phase 3 agents were run **sequentially** in dependency order — Architecture → Code Quality → Security → Documentation — with the authoritative gate (`make checkall`) run and a checkpoint commit after each.

---

## Resolved Issues ✅

### Security
- **[SEC-001]** SSRF / local-file guard in URL fetching — `web_tools.py` — new `is_safe_public_url()` enforces `{http,https}` scheme and rejects private/loopback/link-local/reserved/multicast hosts via DNS resolution; `fetch_url` raises `ValueError` on unsafe URLs (verified: `file:///etc/passwd`, `169.254.169.254`, `localhost`, `127.0.0.1`, `chrome://` all rejected).
- **[SEC-002]** `ignore_ssl` TLS-bypass — `web_tools.py` — emits a visible warning whenever `ignore_ssl=True` (independent of `verbose`) and raises `ValueError` when combined with `http_credentials` (MITM credential exposure).
- **[SEC-003]** Configurable Gemini safety settings — `llm_config.py` — new `safety_settings: dict | None = None` field; Gemini builders no longer force `BLOCK_NONE`; default is provider-standard thresholds. Threaded through serialization.
- **[SEC-004]** `.idea/` in VCS — already untracked + gitignored; verified, no change needed.
- **[SEC-005]** Build/coverage artifacts in VCS — `dist/`, `coverage.xml`, `.coverage`, `htmlcov/` already untracked + gitignored; verified, no change needed.
- **[SEC-006]** Weak-hash helpers — `utils.py` — `usedforsecurity=False` added to `md5_hash`/`sha1_hash` (FIPS-safe); deprecation warnings preserved.
- **[SEC-007]** `id_generator` randomness contract — `utils.py` — documented as non-cryptographic; confirmed no security-sensitive callers exist.
- **[SEC-008]** Basic-auth credential logging — `web_tools.py` — Selenium worker logs the pre-injection URL; `inject_credentials` docstring warns of netloc leakage and prefers Playwright `http_credentials`.
- **[SEC-009]** Mutable third-party CI actions — `.github/workflows/{publish,publish-dev,release}.yml` — `pypa/gh-action-pypi-publish` and `Ilshidur/action-discord` pinned to verified commit SHAs (resolved via `gh api`) with traceability comments.
- **[SEC-010]** `suppress_output` error masking — `utils.py` — added CWE-778 docstring warning; audited call sites (only its own test uses it — no production/auth/fetch masking).

### Architecture
- **[ARC-001]** Optional-extras dependency split — `pyproject.toml` + `search_utils.py`/`web_tools.py`/`pricing_lookup.py` — core deps trimmed 38→15; 17 extras added (`openai`, `anthropic`, `google`, `bedrock`, `azure`, `groq`, `xai`, `mistral`, `deepseek`, `openrouter`, `litellm`, `github`, `ollama`, `web`, `search`, `pricing`, `all`); all backends converted to guarded lazy imports with actionable `ImportError` messages.
- **[ARC-002 (≡QA-013)]** Import-time logging/excepthook mutation — `par_logging.py`/`__init__.py` — removed `basicConfig`/`rich.traceback.install`/global `warnings.filterwarnings` at import; `par_ai` logger now uses `NullHandler` + `propagate=False`; opt-in `init_logging()` added. *Behavioral change — see CHANGELOG.*
- **[ARC-003]** Undeclared `googleapiclient` — declared inside the `[google]` extra (folded into ARC-001).
- **[ARC-004 (≡QA-004)]** Callback context-manager exception safety — `provider_cb_info.py` — `try/finally` so `parai_callback_var` resets and cost prints even on exception.
- **[ARC-005]** Divergent API-key validation — `llm_utils.py` — `llm_config_from_env` now reuses `is_provider_api_key_set`; LiteLLM no longer raises an empty-variable-name `ValueError` (verified). Bedrock exemption retained.
- **[ARC-006 (≡QA-016)]** `num_ctx` vs `max_output_tokens` — `llm_config.py` — split into `num_ctx` (Ollama context window only) and new `max_output_tokens` (the actual `max_tokens` for 10 providers); deprecation shim warns when `num_ctx` is set on a non-Ollama provider; `PARAI_MAX_OUTPUT_TOKENS` env wired.
- **[ARC-007 + ARC-017]** LlmConfig refactor + table-driven env — `llm_config.py`/`llm_utils.py` — `to_json()` via `dataclasses.asdict`, `clone()` via `dataclasses.replace` (+ deep-copy of mutable fields), `_build_llm` if-chain → `_PROVIDER_BUILDERS` dict registry (pure, no caller mutation), `set_env`/`llm_config_from_env` share `_ENV_NUMERIC_FIELDS`.
- **[ARC-009]** Fuzzy provider matching — `llm_providers.py` — exact-match first, then unique-prefix, returns `""` on ambiguity; docstring corrected (`"open"` is now correctly ambiguous, not OpenAI).
- **[ARC-011]** Web-fetch error contract — resolved via QA-001 + QA-006 in Code Quality.
- **[ARC-012]** Lazy litellm + data-driven zero-cost — `pricing_lookup.py` — litellm lazy (Phase 2); hardcoded zero-cost list reduced to genuinely-local `[OLLAMA, LLAMACPP]` (Groq/GitHub now priced from litellm data).
- **[ARC-015]** Generated docs out of wheel — `pyproject.toml`/`Makefile`/`.gitignore` — `make docs` writes to gitignored `./docs/build/`; dropped `**/*.{html,png,gif,jpg,md}` wheel includes; untracked 14 stale HTML files. Wheel now ships 0 HTML (verified).
- **[ARC-016]** Immutable release tags — `.github/workflows/build.yml` — `tag-version` no longer deletes/force-pushes; skips when the tag exists.
- **[ARC-018]** Drop `strenum` — 4 modules + `pyproject.toml` — stdlib `enum.StrEnum`.
- **[ARC-019]** Invalid PEP 621 keys — `pyproject.toml` — removed redundant `url`/`packages` from `[project]`.
- **[ARC-020]** Tooling version skew — `ruff.toml`/`pyrightconfig.json` — both target `py311` (matches `requires-python`).
- **[ARC-021 (≡QA-025)]** Markdown facade params — `web_tools.py` — `fetch_url_and_convert_to_markdown` now forwards proxy, credentials, wait type, headless, ignore_ssl, max_parallel.
- **[ARC-022 (≡QA-012)]** Duplicate CSV rendering — resolved via QA-012.
- **[ARC-023]** Import-time env reads — `llm_providers.py`/`llm_config.py` — `OLLAMA_HOST` read fresh at build time (post-import `load_dotenv()` honored).
- **[ARC-024]** Makefile aliases + dead code — `Makefile`/`__main__.py` — added `build`/`fmt` standard targets; clarified `__main__.py` docstring.
- **[ARC-010]** *(follow-up)* Correlation via `RunnableConfig` metadata — `llm_config.py`/`llm_utils.py`/`__main__.py` — `build_chat_model`/`build_llm_model` no longer overwrite `llm.name` with the config_id UUID; correlation flows exclusively through `RunnableConfig` metadata + tags (already in place). The two readers that used `llm.name` (`summarize_content`, the `__main__` demo) now route through `llm_run_manager` model-based lookups. +2 regression tests.
- **[ARC-013]** *(follow-up)* Split `utils.py` + single HTML→Markdown pipeline — `utils.py` (931 lines) → `utils/` package of 10 cohesive submodules; `__init__.py` re-exports all 49 public names for full backward compat. Consolidated to one converter: dropped `markdownify`, made `html2text` canonical (preserves `web_tools.html_to_markdown`'s LLM-facing output; `md()` reimplemented on html2text).
- **[ARC-014]** *(follow-up)* Typed `SearchResult` + relocate `web_search` — `search_utils.py`/`web_tools.py` — new dict-compatible `SearchResult` Pydantic model; all seven search functions return `list[SearchResult]`; `web_search` relocated to `search_utils` with a backward-compat re-export in `web_tools`; search subsystem decoupled from the LLM config stack (optional `summarizer` callable; `import par_ai_core.search_utils` no longer pulls in `llm_config`/`llm_utils`). *Breaking return-type change — see CHANGELOG `[Unreleased]`.*

### Code Quality
- **[QA-001]** *Critical* — Playwright TEXT wait — `web_tools.py`/`tests` — replaced nonexistent `Locator.wait_for_text` with `page.wait_for_function(...)`; a documented feature was silently dead in production, hidden by an unconstrained mock. Test rewritten with `spec=Page`.
- **[QA-002]** LiteLLM `ModelInfo` lookup — `llm_utils.py`/`tests` — `getattr` on a TypedDict (always `None`) → dict access `.get(...)`; restores correct context sizes for modern 128k–2M models.
- **[QA-003]** Selenium wait semantics — `web_tools.py`/`tests` — SLEEP now sleeps, the two unconditional trailing sleeps removed, `NONE` skips all sleeps.
- **[QA-005]** `from_json` partial dicts — `llm_config.py`/`tests` — `ValueError("...missing required field 'provider'")` instead of bare `KeyError`; `mode` defaults when absent.
- **[QA-006]** Silent-failure error handling — `web_tools.py`/`pricing_lookup.py`/`search_utils.py` — per-module `logger`, `logger.warning` at every swallow site, `fetch_url(raise_on_error=True)` escape hatch.
- **[QA-007 (+QA-023)]** `has_value` suffix removal — `utils.py`/`tests` — `rstrip(".00")` (strips a char set) → `removesuffix(".00")`; depth-limit comment aligned.
- **[QA-009]** Reasoning-model prefix matching — `llm_config.py`/`tests` — single `_is_reasoning_model()` helper with explicit prefix tuple (now catches `gpt-5`/`gpt-5-mini`/`o4-mini`).
- **[QA-010]** `serper_search` `days` — `search_utils.py`/`tests` — `days` wired into a Serper freshness filter; `type`→`search_type` (no longer shadows builtin).
- **[QA-011]** `youtube_search` max comments — `search_utils.py`/`tests` — forwards `max_results=max_comments`.
- **[QA-012]** `display_formatted_output` CSV — `output_utils.py`/`tests` — reuses `csv_to_table` (no more `StopIteration` on empty / Rich error on ragged rows).
- **[QA-015]** `accumulate_cost` double-count — `pricing_lookup.py`/`tests` — payload normalized to one convention before accumulation.
- **[QA-017]** Blocking `input()` in async — `web_tools.py` — `await asyncio.to_thread(input)` for the PAUSE wait.
- **[QA-018]** Naming — `utils.py`/`llm_image_utils.py`/`search_utils.py` — `DECIMAL_PRECISION` (alias kept), `image_bytes`, `search_type`.
- **[QA-019]** Dead code — `utils.py`/`time_display.py`/`web_tools.py`/`provider_cb_info.py` — removed version shims, pointless `try/except: raise`, commented debug prints, simplified triple-nested `csv.Error` wrapping.
- **[QA-020]** Ruff posture — `ruff.toml`/`src/` — added `UP` + `B` rules and fixed all fallout; all 34 bare `# type: ignore` qualified with codes (20 unnecessary ones removed). `C901` complexity deferred (documented in `ruff.toml`).
- **[QA-021]** Desktop Chrome UA — `user_agents.py`/`tests` — removed the "Mobile Safari" token from desktop UAs.
- **[QA-022]** `get_files` ext semantics — `utils.py` — docstring corrected to "exclude".
- **[QA-024]** Fragile markdown fence hack — `web_tools.py` — `html_content.replace("<pre", ...)` replaced with a BeautifulSoup transform.

### Documentation
- **[DOC-001]** *Critical* — Stale API reference — verified `make docs` regenerates an accurate, out-of-tree reference at `./docs/build/` (gitignored per ARC-015); spot-checked that the dead `wait_for_text` is gone and current public functions/fields are present. Nothing to commit (docs no longer shipped).
- **[DOC-002]** `PARAI_NUM_REDICT`→`PARAI_NUM_PREDICT` — `README.md`.
- **[DOC-003]** `.env.example` rebuilt from code-read env vars — removed unused keys, added missing provider keys, fixed `DEEP_AI_API_KEY`→`DEEPSEEK_API_KEY`.
- **[DOC-004]** `llm_config_from_env` docstring — `llm_utils.py` — lists all 21 `{prefix}_*` vars it reads; removed nonexistent `MAX_CONTEXT_SIZE`.
- **[DOC-005]** Removed unused env vars — `README.md` — `WEATHERAPI_KEY`, `GITHUB_PERSONAL_ACCESS_TOKEN`.
- **[DOC-006]** Trimmed "What's New" — `README.md` — last 3 releases + CHANGELOG link.
- **[DOC-007]** `LlmProvider` docstring — `llm_providers.py` — all 14 members accurate.
- **[DOC-008]** `LlmConfig` attributes — `llm_config.py` — all 28 fields documented (incl. `max_output_tokens`, `safety_settings`).
- **[DOC-009]** `fetch_url` wait_type — `web_tools.py` — corrected `ScraperWaitType`/default `SLEEP`; documented `raise_on_error`.
- **[DOC-010]** CHANGELOG — `CHANGELOG.md` — added missing `[0.5.3]`, added dates from tag timestamps, added `[Unreleased]` documenting the remediation's behavioral changes.
- **[DOC-011]** README link + sentence — reversed link fixed; truncated Tavily fragment removed.
- **[DOC-012]** README docs links + Playwright install — linked `docs/architecture.md`/`docs/operations.md`; added `playwright install chromium` note.
- **[DOC-013]** Reddit credentials — `README.md`/`.env.example` — documented `REDDIT_USERNAME`/`REDDIT_PASSWORD` as optional.
- **[DOC-014]** Mermaid diagram — `docs/architecture.md` — ASCII module-dependency diagram converted to Mermaid `graph TD`.
- **[DOC-015]** Badges — `README.md` — fixed "x86-63"→"x86-64"; added CI status badge.
- **[DOC-016]** Install path — `README.md` — added `pip install par_ai_core` (+ extras); scoped UV to development.
- **[DOC-017]** Nonexistent sdist include — `pyproject.toml` — removed `extraction_prompt.md`.
- **[DOC-018]** `ParAICallbackHandler.__init__` docstring — `provider_cb_info.py`.

---

## Requires Manual Intervention 🔧

One architecture refactor remains **deferred**. The audit's own roadmap classifies it as **Long-term Backlog**; it is large enough that bundling it into the phased remediation risked destabilizing working code. It is not a defect — a structural improvement. (ARC-010, ARC-013, and ARC-014 — the other three originally-deferred backlog refactors — were completed in the Phase 3e follow-up; see *Resolved Issues* above.)

### [ARC-008] Single-source provider registry
- **Why deferred**: Largest mechanical restructure in the phase — collapses six parallel dicts + `provider_config` into one `dict[LlmProvider, LlmProviderConfig]` (adding `base_url`), deriving the legacy dicts as comprehensions. It touches the most-edited file and interacts subtly with ARC-023's lazy `OLLAMA_HOST` resolution.
- **Recommended approach**: Add `base_url` to `LlmProviderConfig`; inline all values into `provider_config`; derive `provider_base_urls`/`provider_default_models`/`provider_light_models`/`provider_vision_models`/`provider_default_embed_models`/`provider_env_key_names` as comprehensions; assert `provider_base_urls[p] == provider_config[p].base_url` for all providers in a test.
- **Estimated effort**: Medium.

---

## Verification Results

- **Format (`ruff format`)**: ✅ Pass — 15 files unchanged.
- **Lint (`ruff check`)**: ✅ Pass — All checks passed (after adding `UP` + `B` rules and fixing fallout).
- **Type Check (`pyright`, pinned v1.1.410)**: ✅ Pass — 0 errors, 0 warnings, 0 informations.
- **Tests (`pytest`)**: ✅ Pass — **369 passed**, 4 warnings (all pre-existing third-party deprecations: `google.genai` `_UnionGenericAlias`, `langchain-community` sunset notice, and the two intentional `md5_hash`/`sha1_hash` deprecation tests). No new warnings. 93% coverage.
- **Package build (`uv build`)**: ✅ Pass — wheel + sdist both build; wheel ships **0 HTML / 0 PNG**, includes `py.typed` and the `utils/` package (10 submodules); `markdownify` dropped from deps; sdist has no `extraction_prompt.md` reference.
- **Docs generation (`make docs`)**: ✅ Pass — regenerates accurate reference to `./docs/build/`.

**Test growth**: 330 (audit baseline) → 369 (+39 regression tests across phases: SSRF guards, ignore_ssl refusal, exception-safe callback, LiteLLM env, num_ctx split, OLLAMA_HOST timing, Playwright real-API wait, Selenium timing, from_json partial dicts, reasoning-model helper, csv_to_table, token normalization, UA fingerprint, no-name-hijack correlation, SearchResult dict-compat, summarizer decoupling, etc.).

> ⚠️ **Pyright version watch item (not a failure)**: The host editor runs a newer pyright (v1.1.411+) than the project's pinned v1.1.410. The newer version surfaces stricter findings that the project gate does not — chiefly `StrEnum` member inference (`Literal['OpenAI']` vs `LlmProvider`), `reportCallIssue` on LangChain provider kwargs (`max_tokens`, `model`, `extra_body`), and `reportTypedDictNotRequiredAccess` on `RunnableConfig["metadata"]`. These are **not regressions** (the code is runtime-correct; all 369 tests pass; the pinned gate is clean), but when the project bumps pyright, expect to address StrEnum narrowing and either add `# type: ignore[code]` or adjust the provider constructor call sites.

---

## Files Changed

**Source modules (14 modified + `utils.py` split):** `__init__.py`, `__main__.py`, `llm_config.py`, `llm_image_utils.py`, `llm_providers.py`, `llm_utils.py`, `output_utils.py`, `par_logging.py`, `pricing_lookup.py`, `provider_cb_info.py`, `search_utils.py`, `time_display.py`, `user_agents.py`, `web_tools.py`, and `utils.py` → `utils/` package (10 cohesive submodules, ARC-013)

**Build / config (6 modified):** `pyproject.toml`, `pyrightconfig.json`, `ruff.toml`, `Makefile`, `.gitignore`, `uv.lock`

**CI workflows (4 modified):** `.github/workflows/{build,publish,publish-dev,release}.yml`

**Docs (4 modified):** `README.md`, `CHANGELOG.md`, `.env.example`, `docs/architecture.md`

**Tests (12 modified):** `tests/test_{llm_config,llm_providers,llm_utils,output_utils,par_logging,pricing_lookup,provider_cb_info,search_utils,time_display,user_agents,utils,web_tools}.py`

**Deleted (14):** `src/par_ai_core/docs/*.html` — stale generated API reference, now gitignored and generated out-of-tree (ARC-015/DOC-001).

**Net**: +3,241 / −11,106 lines (the large deletion is the removed generated HTML docs).

### Commits on the branch
```
0d70bc6 refactor(architecture): implement ARC-014
39de403 refactor(architecture): implement ARC-010 and ARC-013
5e1edf5 docs: add audit remediation report (AUDIT-REMEDIATION.md)
697d1fb docs: resolve Phase 3d documentation issues
cabbcde fix(security): resolve Phase 3a remaining security issues
5b751ee fix(quality): resolve Phase 3c code-quality issues
804599c fix(architecture): resolve remaining Phase 3b architecture issues
82c05a9 fix(architecture): resolve Phase 2 critical architecture issues
c8c0286 fix(security): resolve Phase 1 security issues from audit
af48eb0 chore: add audit report, remediation playbook, and enhancement proposals
```
Each phase is an isolated, independently-revertable checkpoint commit.

---

## Next Steps

1. **Security review of the behavioral changes** before release. These are opt-in-safe but change observable behavior (see CHANGELOG `[Unreleased]`): SEC-001 (`fetch_url` now rejects non-public URLs), SEC-002 (`ignore_ssl` + `http_credentials` now raises), SEC-003 (Gemini no longer forces `BLOCK_NONE`), ARC-001 (lean core — existing `pip install par_ai_core` users lose backends unless they switch to `[all]`), ARC-002 (no global logging mutation — callers must call `init_logging()`), ARC-006 (`num_ctx` deprecation on non-Ollama).
2. **Pick up the remaining deferred refactor** (ARC-008) as a dedicated, reviewed change — see *Requires Manual Intervention*. (ARC-010/013/014 were completed in the Phase 3e follow-up.)
3. **Decide on pyright bump strategy** — either stay on v1.1.410 or address the StrEnum/LangChain-kwarg findings when upgrading (see the watch item above).
4. **Re-run `/audit`** to produce an updated `AUDIT.md` reflecting current state (expected to show ARC-008 and the pyright-version items as the remaining open work).
5. **Merge or iterate** — the branch is green and fully verified; ready for review/merge to `main` whenever you want.
