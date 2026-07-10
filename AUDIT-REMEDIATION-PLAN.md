# AUDIT Remediation Playbook

> Companion to `AUDIT.md`. One executable entry per issue, ordered to match the Remediation Plan
> phases. Each entry is written so a smaller model (e.g. sonnet running `/fix-audit`) can execute it
> without re-deriving the analysis.
>
> **Global rules for every fix:**
> - Re-read the target region before editing — a prior phase may have already changed the file
>   (see the File Conflict Map in `AUDIT.md`). Line numbers below are as of audit time (v0.5.8).
> - Never modify auth/token/secret handling silently — flag SEC changes for review (they are here).
> - After each atomic fix: `make checkall` (format + lint + pyright) and the targeted test must pass.
> - Do **not** bump the version (`src/par_ai_core/__init__.py`) unless explicitly asked.
> - Duplicate issues (e.g. QA-004 ≡ ARC-004) are fixed once; the second ID points to the first.
> - Full verification gate: `make checkall && make test`.

---

## Phase 1 — Security (Sequential, land first; they reshape shared files)

### [SEC-001] SSRF and local-file read via unrestricted URL fetching
- **Files**: `src/par_ai_core/web_tools.py:214-217` (`fetch_url` validation), also guards the Playwright path (`fetch_url_playwright` ~312) and Selenium path (`fetch_url_selenium` ~459); call sites `src/par_ai_core/search_utils.py:196-197,242-243`.
- **Steps**:
  1. Add module-level imports near the top of `web_tools.py`: `import ipaddress` and `import socket` (check they are not already imported).
  2. Add a helper above `fetch_url`:
     ```python
     _ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

     def is_safe_public_url(url: str) -> bool:
         """Return True only for http/https URLs that resolve to public IP addresses.

         Blocks file://, chrome://, private/loopback/link-local/reserved ranges to
         mitigate SSRF and local-file disclosure (CWE-918). Note: this narrows but does
         not fully close the DNS-rebinding window; callers handling untrusted URLs should
         also pin the resolved address and block redirects to private IPs.
         """
         parsed = urlparse(url)
         if parsed.scheme not in _ALLOWED_URL_SCHEMES or not parsed.hostname:
             return False
         try:
             infos = socket.getaddrinfo(parsed.hostname, None)
         except socket.gaierror:
             return False
         for info in infos:
             ip = ipaddress.ip_address(info[4][0])
             if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                 return False
         return True
     ```
  3. Replace the validation block at `fetch_url:216-217`:
     ```python
     if isinstance(urls, str):
         urls = [urls]
     bad = [u for u in urls if not is_safe_public_url(u)]
     if bad:
         raise ValueError(
             f"Refusing to fetch unsafe or non-public URL(s): {bad}. "
             "Only public http(s) URLs are allowed (SSRF/local-file protection)."
         )
     ```
  4. Add a docstring `Raises: ValueError` note to `fetch_url` documenting the SSRF guard, and a one-line note to `brave_search`/`serper_search` in `search_utils.py` that `scrape=True` fetches are subject to the guard.
- **Method**: The existing check only verified a truthy scheme, letting `file://`/metadata IPs through. Resolving the host and rejecting private ranges is the standard SSRF mitigation. Pitfalls: (a) do the resolve check on **every** URL in the list, not just the first; (b) keep the guard in `fetch_url` (the single choke point) so both playwright and selenium paths inherit it; (c) do not remove the guard when SEC-002/QA-001/QA-003 rewrite the same function — layer on top of it; (d) `getaddrinfo` can return multiple records — reject if **any** is private.
- **Verify**: `uv run python -c "from par_ai_core.web_tools import fetch_url; fetch_url('file:///etc/passwd')"` must raise `ValueError`; add/keep a test asserting `ValueError` for `file://`, `http://169.254.169.254/`, `http://localhost/`. Then `make test` (targeted: `uv run pytest tests/test_web_tools.py -q`).

### [SEC-002] `ignore_ssl` disables TLS certificate validation
- **Files**: `src/par_ai_core/web_tools.py:305` (Playwright `ignore_https_errors`), `:430-431` (Selenium flags), `:456-457` (credential injection).
- **Steps**:
  1. In `fetch_url` (or both fetcher functions), when `ignore_ssl` is truthy, emit a visible warning once: `console.print("[bold yellow]WARNING: TLS certificate validation is disabled (ignore_ssl=True). Do not use with untrusted networks or credentials.[/bold yellow]")` (use `console_err` if no console).
  2. In the Selenium path at `:456-457`, refuse credential injection when SSL is disabled:
     ```python
     if http_credentials and "username" in http_credentials and "password" in http_credentials:
         if ignore_ssl:
             raise ValueError("Refusing to inject HTTP credentials while ignore_ssl=True (credentials would be exposed to MITM).")
         url = inject_credentials(url, http_credentials["username"], http_credentials["password"])
     ```
  3. Add to the `fetch_url`/`fetch_url_playwright`/`fetch_url_selenium` docstrings: "`ignore_ssl=True` disables certificate validation and must never be combined with `http_credentials`."
- **Method**: The flag is legitimately needed for self-signed hosts, so keep it but make its risk loud and block the dangerous combination (creds + no TLS validation). Pitfall: the warning must fire regardless of `verbose` (this is a security condition, not debug output).
- **Verify**: `uv run python -c "from par_ai_core.web_tools import fetch_url_selenium"` still imports; add a test that passing `ignore_ssl=True` with `http_credentials` raises `ValueError`. `make checkall`.

### [SEC-008] Basic-auth credentials embedded in URL / logging
- **Files**: `src/par_ai_core/web_tools.py:91-110` (`inject_credentials`), used at `:457`; check verbose branches at `:454`, `:477-481`, `:489-490`.
- **Steps**:
  1. Grep for any `console.print` that includes `url` **after** `inject_credentials` runs (the mutated `url` variable in the Selenium loop). If found, print the pre-injection URL or a redacted form instead.
  2. Add a docstring note to `inject_credentials` warning that URL-embedded credentials can leak into logs/history and that Playwright's `http_credentials` is preferred.
- **Method**: Credentials in the netloc leak via logs/referrers. The current Selenium error branch at `:490` prints `url` inside the `except` — after injection this would leak the secret. Redact it. Pitfall: don't break the Playwright path, which already uses `http_credentials` safely (no injection).
- **Verify**: `grep -n "console.print" src/par_ai_core/web_tools.py` — confirm no post-injection `url` is logged. `make checkall`.

### [SEC-003] Google Gemini safety filters hardcoded to BLOCK_NONE
- **Files**: `src/par_ai_core/llm_config.py:692,708` (both Gemini builders); field definition near `:71-146`; serialization (`to_json`/`clone`/`set_env`) if adding a field before ARC-007 lands.
- **Steps**:
  1. Add an optional field to `LlmConfig`: `safety_settings: dict | None = None` (place with the other optional fields; add to the class docstring Attributes — coordinates with DOC-008).
  2. In both Gemini builders replace the hardcoded `safety_settings={HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE}` with `safety_settings=self.safety_settings` and only pass the kwarg when it is not None (so the provider default applies otherwise).
  3. If ARC-007 has **not** yet refactored serialization, add `safety_settings` to `to_json`, `clone`, and (optionally) `set_env`/`llm_config_from_env`. If ARC-007 has landed (asdict/replace), no extra work.
- **Method**: The unconditional BLOCK_NONE silently disables safety for every consumer — make it opt-in via config with the provider default when unset. Pitfall: the field must be threaded through clone/to_json or it silently vanishes on `build_chat_model` (which clones first). This is a **security change** — flag for review; do not commit inside an unrelated change.
- **Verify**: `uv run python -c "from par_ai_core.llm_config import LlmConfig; print('safety_settings' in {f.name for f in __import__('dataclasses').fields(LlmConfig)})"` prints `True`; add a test that the default config passes no BLOCK_NONE. `make checkall`.

### [SEC-006] `usedforsecurity=False` on md5/sha1 helpers
- **Files**: `src/par_ai_core/utils.py:507-548` (`md5_hash`, `sha1_hash`).
- **Steps**: Change `hashlib.md5(data.encode("utf-8"))` → `hashlib.md5(data.encode("utf-8"), usedforsecurity=False)` and likewise for `hashlib.sha1(...)`. Keep the existing `DeprecationWarning`.
- **Method**: Signals non-security intent and unblocks FIPS-mode environments. Pitfall: only add the kwarg; do not remove the deprecation warnings or change the return format.
- **Verify**: `uv run python -c "import warnings; warnings.simplefilter('ignore'); from par_ai_core.utils import md5_hash; print(md5_hash('x'))"` still returns a hex digest. `make checkall`.

### [SEC-007] `id_generator` randomness contract
- **Files**: `src/par_ai_core/utils.py:158-168`.
- **Steps**: Add a docstring line: "Not cryptographically secure; do not use for session IDs, nonces, or tokens. Use `secrets` for security-sensitive identifiers." (No code change unless a security caller is found — grep `id_generator(` across the repo; if any use mints a token, switch that call to a `secrets`-based generator.)
- **Method**: Current use (user-agent randomization) is fine; the risk is future misuse. Documentation is the proportionate fix. Pitfall: don't swap to `secrets.choice` globally — it's slower and unnecessary for UA jitter.
- **Verify**: `grep -rn "id_generator" src/` shows only non-security callers. `make checkall`.

---

## Phase 2 — Critical Architecture (Sequential, Blocking)

### [ARC-001] Optional-extras dependency split + guarded imports
- **Files**: `pyproject.toml:39-79` (deps), `src/par_ai_core/search_utils.py:52-65` (top-level backend imports), `src/par_ai_core/web_tools.py` (playwright/selenium imports), `src/par_ai_core/pricing_lookup.py:29-30` (litellm import).
- **Steps**:
  1. In `pyproject.toml`, trim `[project] dependencies` to the true core: `langchain-core`, `langchain`, `pydantic`, `pydantic-settings`, `rich`, `requests`, `orjson`, `beautifulsoup4`, `python-dotenv`, `tiktoken` (keep anything imported unconditionally at package import time — verify with step 5).
  2. Add `[project.optional-dependencies]`:
     ```toml
     openai = ["langchain-openai"]
     anthropic = ["langchain-anthropic"]
     google = ["langchain-google-genai", "langchain-google-community", "google-api-python-client"]
     bedrock = ["langchain-aws", "boto3", "botocore"]
     groq = ["langchain-groq"]
     mistral = ["langchain-mistralai"]
     ollama = ["langchain-ollama"]
     web = ["playwright", "selenium", "webdriver-manager", "html2text", "markdownify"]
     search = ["tavily-python", "praw", "google-api-python-client", "youtube-transcript-api"]
     pricing = ["litellm"]
     all = ["par_ai_core[openai,anthropic,google,bedrock,groq,mistral,ollama,web,search,pricing]"]
     ```
     (Enumerate the current 38 deps first — `grep -A40 'dependencies = \[' pyproject.toml` — and assign each to exactly one extra or to core. Do not drop any dependency without a home.)
  3. Convert every now-optional top-level import into a lazy import inside the function that uses it, wrapped with an actionable error:
     ```python
     try:
         import praw
     except ImportError as e:
         raise ImportError("Reddit search requires: pip install 'par_ai_core[search]'") from e
     ```
     Apply to `search_utils.py` (praw, googleapiclient, tavily), and confirm `web_tools.py`/`llm_config.py`/`pricing_lookup.py` do the same (ARC-003, ARC-012, QA-014 fold in here).
  4. Update README + `docs/operations.md` install instructions to show `pip install 'par_ai_core[all]'` and per-feature extras.
  5. **Critical guard**: run `uv run python -c "import par_ai_core"` and import each module with the extras *uninstalled* to confirm nothing optional is imported at package-import time. The `__init__.py` must not eagerly import web/search/pricing modules.
- **Method**: The code already lazy-imports provider SDKs; the manifest is what forces the fat install. Pitfall #1: `search_utils.py` imports `llm_utils` → `litellm` at module load — that chain must be broken too (defer `summarize_content`'s litellm dependency). Pitfall #2: do not move a dependency that is genuinely core (e.g. `tiktoken` if imported at import time). Pitfall #3: this is the **blocking** Phase-2 change — QA-014/ARC-003/ARC-012 must be done as part of it, not separately, or they conflict.
- **Verify**: In a scratch venv, `pip install -e .` (core only) then `python -c "import par_ai_core.llm_config, par_ai_core.pricing_lookup"` must succeed; `python -c "import par_ai_core.search_utils; par_ai_core.search_utils.reddit_search"` referencing praw must raise the actionable ImportError only when praw is absent. `make checkall && make test` with `[all]` installed.

### [ARC-007] LlmConfig serialization + builder-registry refactor (folds ARC-017, QA-008)
- **Files**: `src/par_ai_core/llm_config.py:175` (`to_json`), `:235` (`clone`), `:823-855` (`_build_llm` if-chain), `:902-941` (`set_env`); `src/par_ai_core/llm_utils.py:35-147` (`llm_config_from_env`).
- **Steps**:
  1. Replace `clone()` body with `return dataclasses.replace(self, extra_body=copy.deepcopy(self.extra_body), fallback_models=list(self.fallback_models) if self.fallback_models else self.fallback_models)` (deep-copy the mutable fields explicitly; import `dataclasses`, `copy`).
  2. Replace `to_json()`'s hand-listed dict with `{"class_name": "LlmConfig", **dataclasses.asdict(self)}`, then post-process enum/SecretStr fields to their serialized forms exactly as the current code does (provider→value, mode→value, reasoning_effort→value). Verify the output keys match the previous `to_json` output for round-trip compatibility.
  3. Replace the `_build_llm` if-chain with a registry: `_PROVIDER_BUILDERS: dict[LlmProvider, Callable[[LlmConfig], BaseChatModel]] = {LlmProvider.OPENAI: LlmConfig._build_openai_llm, ...}` and dispatch `return _PROVIDER_BUILDERS[self.provider](self)`. Keep each `_build_*` method.
  4. Make `_build_llm` pure: compute effective base_url in a local variable instead of `self.base_url = ...` (line ~831); same for `self.extra_body` (~578) and `self.num_ctx` (~650) — assign to locals passed to the provider constructor.
  5. Table-drive `llm_config_from_env`/`set_env` (ARC-017): define `_ENV_FIELDS = [("MODEL", "model_name", str), ("TEMPERATURE", "temperature", float), ("NUM_CTX", "num_ctx", int), ...]` covering every field both functions handle, and loop over it in both directions so they cannot drift.
- **Method**: This is the largest structural change and **blocks** QA-005, QA-009, ARC-006/QA-016, DOC-008 — do it first so those edit the new structure. Pitfalls: (a) `asdict` recurses into nested dataclasses/dicts — verify `extra_body` and `fallback_models` serialize correctly; (b) builders currently mutate `self` and callers rely on cloning first — making them pure removes a latent footgun but re-test `build_chat_model` which clones then builds; (c) preserve exact `to_json` key names or you break `from_json` round-trips and any persisted configs.
- **Verify**: Add/keep a round-trip test: `c2 = LlmConfig.from_json(c.to_json()); assert c2 == c` for a fully-populated config across several providers; `assert c.clone() == c and c.clone() is not c`. Run `uv run pytest tests/test_llm_config.py -q`. `make checkall`.

### [ARC-002] Remove import-time logging/excepthook mutation (≡ QA-013)
- **Files**: `src/par_ai_core/par_logging.py:38-58`, `src/par_ai_core/__init__.py:14-15`.
- **Steps**:
  1. In `par_logging.py`, delete the module-level `logging.basicConfig(...)` and `rich.traceback.install(...)` calls. Replace with: `log = logging.getLogger("par_ai"); log.addHandler(logging.NullHandler())`.
  2. Wrap the removed setup in an opt-in function: `def init_logging(level: str | None = None) -> None:` that installs the RichHandler on the `"par_ai"` logger (set `log.propagate = False`) and optionally calls `rich.traceback.install(...)`. Export it.
  3. In `__init__.py:14-15`, move the global `warnings.filterwarnings(...)` calls into `init_logging` (or remove if not essential).
  4. Change `PARAI_LOG_LEVEL` handling to be read inside `init_logging` at call time (coordinates with ARC-023), not at import.
  5. Update `docs/operations.md` and the README to tell consumers to call `par_ai_core.init_logging()` if they want the rich logging/traceback (this is the observable contract change — do docs **after** the code per the blocking note).
- **Method**: A library must not reconfigure the root logger or `sys.excepthook` on import. Pitfall: many internal modules call `getLogger`/`log` — keep the module-level `log` object so those keep working; only the global side effects move behind `init_logging`. Note this changes observable behavior — record it in CHANGELOG (Unreleased).
- **Verify**: `uv run python -c "import logging, par_ai_core; assert logging.getLogger().handlers == [] or not any(getattr(h,'_par_ai',False) for h in logging.getLogger().handlers)"`; confirm importing the package no longer replaces `sys.excepthook` (`uv run python -c "import sys; before=sys.excepthook; import par_ai_core; assert sys.excepthook is before"`). `make checkall && make test`.

### [ARC-016] Immutable release tags in CI
- **Files**: `.github/workflows/build.yml:189-209` (`tag-version` job).
- **Steps**: Replace the delete-and-force-push logic with a guard: if `git rev-parse "v${VERSION}"` succeeds (tag exists), `echo "Tag v${VERSION} already exists; skipping" && exit 0`; otherwise create and push the tag once. Remove the `git tag -d` / `git push --delete` / force-push lines.
- **Method**: Mutable release tags break provenance and race the tag-triggered publish/release workflows. This **blocks** SEC-009 (which edits the same workflow family). Pitfall: this is a CI/release change — do not run `make publish` locally; let CI handle it. Confirm the `release`/`publish` triggers still fire on the newly-created tag.
- **Verify**: `grep -n "push --delete\|tag -d\|--force" .github/workflows/build.yml` returns nothing in the tag job; validate YAML with `uv run python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`.

---

## Phase 3a — Security (remaining, parallel-safe)

### [SEC-004] Untrack `.idea/`
- **Files**: `.idea/` (tracked), `.gitignore`.
- **Steps**: `git rm -r --cached .idea` then add `.idea/` to `.gitignore` (append if not present).
- **Method**: JetBrains files can leak local paths/data-source configs and are environment noise. Pitfall: `--cached` only (do not delete the local dir). This is a git-index change — do not commit as part of an unrelated fix.
- **Verify**: `git status --short .idea` shows deletions staged; `git check-ignore .idea/x` matches.

### [SEC-005] Untrack build/coverage artifacts
- **Files**: `dist/`, `coverage.xml`, `.coverage`, `htmlcov/`, `.gitignore`.
- **Steps**: `git rm -r --cached dist htmlcov coverage.xml .coverage` (only those that are tracked — check `git ls-files` first). Ensure all four patterns are in `.gitignore`. Confirm none appear in the sdist/wheel include globs in `pyproject.toml` (they currently don't).
- **Method**: Generated artifacts risk shipping stale/sensitive content (absolute paths in coverage XML). Pitfall: `git ls-files dist htmlcov coverage.xml .coverage` first — only untrack what is actually tracked.
- **Verify**: `git ls-files | grep -E '^(dist/|htmlcov/|coverage.xml|.coverage)'` returns nothing after the change.

### [SEC-009] SHA-pin third-party GitHub Actions (after ARC-016)
- **Files**: `.github/workflows/publish.yml:48,53`, `publish-dev.yml:45,53`, `release.yml:69`.
- **Steps**: For each third-party action (`pypa/gh-action-pypi-publish@release/v1`, `Ilshidur/action-discord@0.4.0`), look up the commit SHA for the intended release and replace the ref with the full 40-char SHA plus a trailing `# vX.Y.Z` comment. Per `guides/git-ci.md`, verify each ref resolves before committing. Leave first-party `actions/*` and `astral-sh/setup-uv@v8.2.0` as-is.
- **Method**: Mutable refs in workflows with `id-token: write` are a supply-chain risk. Pitfall: verify the SHA actually belongs to the tag (`gh api repos/pypa/gh-action-pypi-publish/git/refs/tags/vX`) — do not invent SHAs. Sequence **after** ARC-016 since both touch the workflow family.
- **Verify**: `grep -nE 'uses:.*@[0-9a-f]{40}' .github/workflows/*.yml` shows the pinned third-party actions; YAML still parses.

### [SEC-010] Audit `suppress_output` usage constraints
- **Files**: `src/par_ai_core/utils.py:779-791`.
- **Steps**: `grep -rn "suppress_output" src/` — confirm no auth/fetch/security path is wrapped. Add a docstring warning that it hides stdout/stderr including error output and must not wrap security-relevant code.
- **Method**: Documentation-only unless a misuse is found. Pitfall: if a fetch/auth call is wrapped, unwrap it instead of just documenting.
- **Verify**: `grep -rn "suppress_output" src/` reviewed. `make checkall`.

---

## Phase 3b — Architecture (remaining, parallel-safe)

### [ARC-003] Declare `google-api-python-client`
- **Files**: `pyproject.toml`, `src/par_ai_core/search_utils.py:56`.
- **Steps**: Folded into ARC-001 step 2 (add to the `[search]`/`[google]` extra) and step 3 (lazy-import inside `youtube_search`/`youtube_get_comments`). If ARC-001 is deferred, add `google-api-python-client` to `[project] dependencies` as a stopgap.
- **Method**: The import currently works only transitively via `langchain-google-community` — an implicit contract. Pitfall: prefer the extras home over adding to core deps.
- **Verify**: `uv run python -c "import googleapiclient; print('ok')"` with `[search]` installed. `make checkall`.

### [ARC-004] try/finally in `get_parai_callback` (≡ QA-004)
- **Files**: `src/par_ai_core/provider_cb_info.py:282-285`.
- **Steps**: Replace lines 282-285 with:
  ```python
  parai_callback_var.set(cb)
  try:
      yield cb
  finally:
      show_llm_cost(cb.usage_metadata, show_pricing=show_pricing, console=console)
      parai_callback_var.set(None)
  ```
- **Method**: Without `finally`, an exception in the `with` block leaves the handler installed in the ContextVar (it keeps accumulating for later runs) and skips the cost summary. Putting both cleanup lines in `finally` fixes the leak and still prints costs on failure. Pitfall: keep `show_llm_cost` before the `set(None)` so it reads the handler while still current; confirm `show_llm_cost` tolerates partial usage data (it does — it just formats whatever accumulated).
- **Verify**: Add a test that raising inside `with get_parai_callback() as cb:` still resets `parai_callback_var.get()` to `None` afterward. `uv run pytest tests/ -k callback -q`. `make checkall`.

### [ARC-005] Reuse `is_provider_api_key_set`; fix LiteLLM env config
- **Files**: `src/par_ai_core/llm_utils.py:59-63`.
- **Steps**: Replace lines 60-63 with:
  ```python
  if not is_provider_api_key_set(ai_provider):
      key_name = provider_env_key_names[ai_provider]
      raise ValueError(f"{key_name} environment variable not set.")
  ```
  Add `is_provider_api_key_set` to the import from `llm_providers`.
- **Method**: `is_provider_api_key_set` already handles the keyless providers (OLLAMA/LLAMACPP/BEDROCK **and** LiteLLM's empty `env_key_name`), so the local exclusion list is both duplicate and wrong for LiteLLM. **Verified bug**: `PARAI_AI_PROVIDER=LiteLLM` currently raises `ValueError: " environment variable not set."`. Pitfall: `provider_env_key_names[LiteLLM]` is empty string — only reference it inside the `if` (which won't trigger for LiteLLM once the shared check is used).
- **Verify**: `PARAI_AI_PROVIDER=LiteLLM PARAI_MODEL=gpt-4 uv run python -c "from par_ai_core.llm_utils import llm_config_from_env; print(llm_config_from_env().provider)"` prints `LiteLLM` without error. Add a regression test. `make checkall`.

### [ARC-006] Split `num_ctx` vs `max_output_tokens` (≡ QA-016; after ARC-007)
- **Files**: `src/par_ai_core/llm_config.py:123-125` (field), usages at `:396,413,443,459,500,526,554,594,621,658,691,707,761,772,810`.
- **Steps**:
  1. Add field `max_output_tokens: int | None = None`. Keep `num_ctx` but document it as Ollama context-window only.
  2. For every non-Ollama builder that currently passes `num_ctx` as `max_tokens`/`max_tokens_to_sample`, pass `max_output_tokens` instead. Keep `num_ctx` mapped to Ollama's `num_ctx`.
  3. Add a deprecation shim: if `max_output_tokens is None and num_ctx is not None` for a non-Ollama provider, emit a `DeprecationWarning` and fall back to `num_ctx` so existing callers keep working for one release.
  4. Thread `max_output_tokens` through the (now table-driven, post-ARC-007) serialization and env parsing (`PARAI_MAX_OUTPUT_TOKENS`).
  5. Update docstrings (DOC-008 overlaps).
- **Method**: The field conflates two distinct concepts across 10 of 12 providers. A separate field plus a deprecation fallback avoids breaking existing configs. Pitfall: the Anthropic builder auto-sets `num_ctx = 2 * reasoning_budget` — move that to `max_output_tokens`. Do this **after** ARC-007 so you edit the registry-based builders.
- **Verify**: Add a test asserting OpenAI builder receives `max_tokens=max_output_tokens` and Ollama receives `num_ctx=num_ctx`; a config with only `num_ctx` on OpenAI emits `DeprecationWarning`. `uv run pytest tests/test_llm_config.py -q`. `make checkall`.

### [ARC-008] Single-source provider registry
- **Files**: `src/par_ai_core/llm_providers.py:96-364`.
- **Steps**: Add `base_url` to `LlmProviderConfig`; make `provider_config: dict[LlmProvider, LlmProviderConfig]` the single source of truth (one entry per provider). Derive the six legacy dicts (`provider_base_urls`, `provider_default_models`, `provider_light_models`, `provider_vision_models`, `provider_default_embed_models`, `provider_env_key_names`) as dict comprehensions over `provider_config` for backward compatibility.
- **Method**: Eliminates 7-structure edits per provider and the KeyError risk. Pitfall: keep the legacy dict names exported (downstream code and tests reference them) — just compute them. Verify every provider has a complete `LlmProviderConfig` before deleting the literal dicts.
- **Verify**: `uv run python -c "from par_ai_core.llm_providers import provider_base_urls, provider_config, LlmProvider; assert all(provider_base_urls[p]==provider_config[p].base_url for p in LlmProvider)"`. `uv run pytest tests/ -k provider -q`. `make checkall`.

### [ARC-009] Deterministic fuzzy provider matching
- **Files**: `src/par_ai_core/llm_providers.py:199-227` (`get_provider_name_fuzzy`).
- **Steps**: Rewrite: (1) exact case-insensitive match returns immediately; (2) collect all prefix matches; (3) return the single match if exactly one, else `""` (ambiguous). Fix the docstring to describe the new behavior and correct/remove the `"open" → OpenAI` claim.
- **Method**: **Verified**: current code returns `'OpenRouter'` for `"open"` because it returns the first enum-order prefix match. Deterministic exact-then-unique-prefix removes the enum-order dependency. Pitfall: preserve the return-type contract (`str`), and keep existing exact-match callers working.
- **Verify**: Add parametrized test: `get_provider_name_fuzzy("openai")=="OpenAI"`, `get_provider_name_fuzzy("open")==""` (ambiguous). `make checkall`.

### [ARC-010] Correlation via RunnableConfig metadata, not `llm.name`
- **Files**: `src/par_ai_core/llm_config.py:868,885,1102`; `src/par_ai_core/provider_cb_info.py:167-173`.
- **Steps**: Stop overwriting `model.name` with the UUID in `build_chat_model`. Instead, put the `config_id` only in `RunnableConfig` metadata/tags (already partly done) and have the callback read it from there. Keep `llm_run_manager` keyed by `config_id`; drop the by-`model_name` lookup or document that it returns the first match.
- **Method**: Hijacking the public `name` attribute surprises consumers and makes two configs for one model indistinguishable. Pitfall: this is a larger behavioral change — ensure the callback still correlates runs (the metadata path must be populated for every build). Backlog-tier; land after the higher-priority config work. Add a test that `model.name` is not a UUID after `build_chat_model`.
- **Verify**: `uv run pytest tests/ -k "run_manager or callback" -q`; assert built model's `name` is not a bare UUID. `make checkall`.

### [ARC-011] Web-fetch error contract (with QA-001/QA-006)
- **Files**: `src/par_ai_core/web_tools.py:251-256,339,350-353`.
- **Steps**: Folded into QA-001 (fix the TEXT wait) and QA-006 (log + `raise_on_error` option). No separate edit — implement those two and the silent-failure data-flow concern is resolved.
- **Method**: Same code paths; avoid double-editing. Pitfall: coordinate so the SSRF guard (SEC-001) stays intact.
- **Verify**: Covered by QA-001/QA-006 verification.

### [ARC-012] Lazy litellm import; data-driven zero-cost detection
- **Files**: `src/par_ai_core/pricing_lookup.py:29-30,165-166`.
- **Steps**: (1) Move `from litellm import ...` from module top into the functions that use it (`get_model_metadata`, `get_api_call_cost`), wrapped with an actionable ImportError referencing `par_ai_core[pricing]` (coordinates with ARC-001). (2) Replace the hardcoded `[OLLAMA, LLAMACPP, GROQ, GITHUB]` zero-cost list at `:165-166` with a data check: treat a model as zero-cost only when litellm's pricing data reports cost-per-token 0 or the model is a local provider (OLLAMA/LLAMACPP).
- **Method**: litellm is heavy and pulled in on nearly every use; Groq/GitHub now have paid tiers so the hardcoded list under-reports cost. Pitfall: keep OLLAMA/LLAMACPP as zero-cost (genuinely local); only remove GROQ/GITHUB from the hardcoded set and let pricing data decide.
- **Verify**: `uv run python -c "import par_ai_core.pricing_lookup"` does not import litellm at module load (check with `-X importtime` or assert `'litellm' not in sys.modules` immediately after import). `uv run pytest tests/ -k pricing -q`. `make checkall`.

### [ARC-013] Split `utils.py`; single HTML→Markdown pipeline
- **Files**: `src/par_ai_core/utils.py` (929 lines), `src/par_ai_core/web_tools.py:524-645` (`html_to_markdown`).
- **Steps**: (1) Choose one HTML→Markdown implementation — prefer `markdownify` (used by `utils.md()`) — and reimplement `web_tools.html_to_markdown` to call it, removing the `html2text` dependency and the fragile `replace("<pre", "``` <pre")` string surgery (QA-024 folds in). (2) Optionally split `utils.py` into `text_utils`/`proc_utils`/`context_files` behind re-exports from `utils` for backward compatibility. Keep the module split backward-compatible (re-export names).
- **Method**: Two converters for one job invites drift. Pitfall: the split is backlog-tier and risky (many importers) — do the converter consolidation first (higher value, lower risk) and re-export from `utils` if you split. Add a test comparing converter output on a representative HTML sample.
- **Verify**: `grep -rn "html2text" src/` returns nothing after consolidation; `uv run pytest tests/ -k "markdown or web_tools" -q`. `make checkall`.

### [ARC-014] Typed `SearchResult` contract; relocate `web_search`
- **Files**: `src/par_ai_core/search_utils.py` (all search functions), `src/par_ai_core/web_tools.py:113-156` (`web_search`).
- **Steps**: (1) Define a `SearchResult` Pydantic model (`title`, `url`, `content`, `raw_content: str | None`) in `search_utils.py`. (2) Return `list[SearchResult]` from all six search functions (or keep dict for back-compat and add a typed variant). (3) Move `web_search` (Google CSE) from `web_tools.py` to `search_utils.py`, re-exporting from `web_tools` for compatibility. (4) Replace the direct `summarize_content` import with an optional `summarizer: Callable | None` parameter to decouple search from the LLM layer.
- **Method**: Standardizes the by-convention dict format and removes the search→LLM coupling. Pitfall: returning a new type is a breaking change — either keep dict output and add `.model_dump()` compatibility or bump behavior in CHANGELOG. Backlog-tier.
- **Verify**: `uv run pytest tests/ -k search -q`. `make checkall`.

### [ARC-015] Generated docs out of wheel/package tree
- **Files**: `pyproject.toml:140-148`, `Makefile:126-129`, `src/par_ai_core/docs/`.
- **Steps**: (1) Change `make docs` to output to a top-level `site/` (or `docs/build/`) instead of `src/par_ai_core/docs/`. (2) Remove `**/*.html`, `**/*.png`, `**/*.gif` include globs from the wheel target (keep only what the package needs at runtime). (3) `git rm -r --cached src/par_ai_core/docs src/par_ai_core/html` and gitignore the new output dir. Coordinate with DOC-001 (which regenerates docs) — point DOC-001 at the new location.
- **Method**: Shipping generated HTML in the wheel bloats installs and risks stale docs. Pitfall: this changes where `make docs` writes — update DOC-001's regeneration step and any README doc links to match. Confirm no runtime code reads from `src/par_ai_core/docs/`.
- **Verify**: `uv build` then `unzip -l dist/*.whl | grep -c '\.html'` returns 0. `make checkall`.

### [ARC-017] Table-driven env serialization
- **Files**: `src/par_ai_core/llm_config.py:902-941`, `src/par_ai_core/llm_utils.py:35-147`.
- **Steps**: Folded into ARC-007 step 5. No separate edit.
- **Method**: Same regions as ARC-007. Pitfall: don't implement twice.
- **Verify**: Covered by ARC-007 (round-trip `set_env` → `llm_config_from_env`).

### [ARC-018] Drop `strenum` for stdlib `enum.StrEnum`
- **Files**: `src/par_ai_core/llm_config.py:30`, `web_tools.py:46`, `pricing_lookup.py:34`, `output_utils.py:49`, `pyproject.toml` deps.
- **Steps**: Replace `from strenum import StrEnum` with `from enum import StrEnum` in all four modules; remove `strenum` from `pyproject.toml` dependencies.
- **Method**: `requires-python >= 3.11` guarantees stdlib `StrEnum`. Pitfall: confirm no behavioral difference (stdlib `StrEnum` auto-lowercases only with `auto()`; the code uses explicit string values, so behavior is identical). Grep all four to be sure the import is the only usage.
- **Verify**: `grep -rn "strenum" src/ pyproject.toml` returns nothing; `uv run pytest tests/ -q` passes. `make checkall`.

### [ARC-019] Fix invalid PEP 621 keys
- **Files**: `pyproject.toml:7` (`url =`), `:80-82` (`packages =`).
- **Steps**: Move `url = "..."` into `[project.urls]` (e.g. `Homepage = "..."`); remove the `[project] packages = [...]` key (packages are already declared under `[tool.hatch.build.targets.wheel]`).
- **Method**: These keys are not valid PEP 621 `[project]` fields; hatchling ignores them now but stricter validators fail. Pitfall: verify the hatch wheel target already lists the packages before deleting the `[project]` one.
- **Verify**: `uv build` succeeds and the wheel contains `par_ai_core/`. `uv run python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`.

### [ARC-020] Align ruff/pyright target versions to 3.11 (≡ QA-020 target part)
- **Files**: `ruff.toml` (`target-version`), `pyrightconfig.json:21` (`pythonVersion`).
- **Steps**: Set `target-version = "py311"` in `ruff.toml` and `"pythonVersion": "3.11"` in `pyrightconfig.json`.
- **Method**: Static analysis must check the oldest supported interpreter (`requires-python >= 3.11`). Pitfall: after changing, re-run `make checkall` — lowering the target may surface new lint/type findings that must be fixed (e.g. use of 3.12+ syntax). If found, that's a real portability bug.
- **Verify**: `make checkall` passes at the new target.

### [ARC-021] Forward full parameter set in markdown facade (≡ QA-025)
- **Files**: `src/par_ai_core/web_tools.py:648-691` (`fetch_url_and_convert_to_markdown`).
- **Steps**: Add the missing parameters (`max_parallel`, `proxy_config`, `http_credentials`, `wait_type`, `wait_selector`, `headless`, `ignore_ssl`, `fetch_using`) to the signature with the same defaults as `fetch_url`, and forward them in the `fetch_url(...)` call.
- **Method**: The facade silently drops capabilities the underlying fetcher supports. Pitfall: keep defaults identical to `fetch_url` so existing callers are unaffected; the SSRF guard (SEC-001) still applies through `fetch_url`.
- **Verify**: `uv run python -c "import inspect; from par_ai_core.web_tools import fetch_url_and_convert_to_markdown as f; print(set(inspect.signature(f).parameters) >= {'wait_type','ignore_ssl','proxy_config'})"` prints `True`. `make checkall`.

### [ARC-023] Read env vars at use time
- **Files**: `src/par_ai_core/llm_providers.py:41` (`OLLAMA_HOST`), `src/par_ai_core/par_logging.py:52` (`PARAI_LOG_LEVEL`).
- **Steps**: Move the `os.environ.get("OLLAMA_HOST")` read out of module scope into the function/builder that needs the Ollama base URL. `PARAI_LOG_LEVEL` is handled by ARC-002 (read inside `init_logging`).
- **Method**: Import-time reads ignore `load_dotenv()` calls made after import (the common pattern in `__main__.py:34`). Pitfall: preserve the same default value; check all readers of the affected module-level constant.
- **Verify**: A test that sets `OLLAMA_HOST` after import and builds an Ollama config sees the new value. `make checkall`.

### [ARC-024] Dead code / stale docstrings / Makefile aliases
- **Files**: `src/par_ai_core/llm_providers.py:61-75` (`LlmProvider` docstring ≡ DOC-007), `web_tools.py:463-465` (Selenium SLEEP ≡ QA-003), `src/par_ai_core/__main__.py:1` (console-script docstring), `Makefile`.
- **Steps**: (1) `LlmProvider` docstring → handled by DOC-007. (2) Selenium SLEEP branch → handled by QA-003. (3) Add `build` and `fmt` phony aliases to the Makefile targeting `package` and `format` respectively (CLAUDE.md requires the standard target set). (4) Clarify the `__main__.py` module docstring or point the console script at a clearer entry.
- **Method**: Housekeeping; most sub-items fold into other issues. Pitfall: only add Makefile aliases, don't rename existing targets (other tooling may call them).
- **Verify**: `make build` and `make fmt` run without "No rule to make target". `make checkall`.

---

## Phase 3c — Code Quality (all, parallel-safe)

### [QA-001] Fix Playwright TEXT wait + spec-constrained mocks
- **Files**: `src/par_ai_core/web_tools.py:335-344`, `tests/test_web_tools.py:592-608` (and `:168-170`).
- **Steps**:
  1. Replace line 339 (`await page.locator("body").wait_for_text(...)`) with:
     ```python
     await page.wait_for_function(
         "text => document.body && document.body.innerText.includes(text)",
         arg=wait_selector,
         timeout=timeout * 1000,
     )
     ```
  2. In the tests, replace `mock_page.locator.return_value.wait_for_text = AsyncMock()` with a mock of `mock_page.wait_for_function = AsyncMock()`, and constrain page mocks with `spec` (e.g. build from `playwright.async_api.Page`) so a nonexistent method raises `AttributeError` in tests.
- **Method**: **Verified**: `Locator.wait_for_text` does not exist in Playwright's Python API, so the `AttributeError` was swallowed and TEXT wait always returned `""`. `page.wait_for_function` with an arg is the correct polling primitive. Pitfall: the current test *certifies the bug* by mocking the fake method — the test must be changed or it will keep passing against broken code. Do this together with QA-003 in one pass over the file.
- **Verify**: A test using a `spec`-constrained page mock where TEXT wait calls `wait_for_function` (not `wait_for_text`) passes; an assertion that `wait_for_text` is not called. `uv run pytest tests/test_web_tools.py -q`. `make checkall`.

### [QA-002] Dict access for LiteLLM ModelInfo
- **Files**: `src/par_ai_core/llm_utils.py:160-171`.
- **Steps**: Replace the `getattr(...)` chain with dict access:
  ```python
  context_size = (
      model_info.get("max_input_tokens")
      or model_info.get("max_tokens")
      or model_info.get("context_length")
  )
  ```
  (Guard for `model_info` being falsy/None first.)
- **Method**: **Verified**: `get_model_metadata` returns LiteLLM's `ModelInfo`, a `TypedDict` (plain dict at runtime), so `getattr` always returned `None` and every model fell back to 8192 — massively over-chunking modern long-context models. Pitfall: test with a **real dict** (`{"max_input_tokens": 128000}`), not an attribute-mock, or the test reproduces the original blind spot.
- **Verify**: A test that a `ModelInfo`-shaped dict with `max_input_tokens=128000` yields `_get_model_context_size(...) == 128000`. `uv run pytest tests/ -k "context_size or metadata" -q`. `make checkall`.

### [QA-003] Coherent Selenium wait-type semantics
- **Files**: `src/par_ai_core/web_tools.py:460-485`.
- **Steps**: (1) Delete the dead `pass` / commented sleep at 463-465 and put `time.sleep(sleep_time)` inside the SLEEP branch. (2) Remove the unconditional `time.sleep(1)` at 476 and the unconditional `time.sleep(sleep_time)` at 485 that run for all wait types; move any needed post-load settle sleep inside the branches that want it. (3) Ensure a "no wait" path (if the enum has NONE, or when SLEEP with `sleep_time==0`) actually skips sleeping.
- **Method**: Current semantics are incoherent — SLEEP is a no-op branch while every URL sleeps ~1+`sleep_time`s unconditionally. Pitfall: keep the dynamic-content scroll+settle behavior that pages rely on, but make it opt-in per wait type rather than always-on; verify against the Playwright path's semantics for consistency. Do together with QA-001.
- **Verify**: A timing test that `wait_type=NONE`/`SLEEP,0` returns quickly and `SLEEP,n` waits ~n seconds (mock `time.sleep` and assert call args). `uv run pytest tests/test_web_tools.py -q`. `make checkall`.

### [QA-004] (≡ ARC-004) — see Phase 3b ARC-004. Fix once.

### [QA-005] Graceful `from_json` partial-dict handling
- **Files**: `src/par_ai_core/llm_config.py:224-233`.
- **Steps**: Replace lines 228-231 with:
  ```python
  if "provider" not in allowed_data:
      raise ValueError("Config data is missing required field 'provider'")
  if not isinstance(allowed_data["provider"], LlmProvider):
      allowed_data["provider"] = provider_name_to_enum(allowed_data["provider"])
  if "mode" in allowed_data and not isinstance(allowed_data["mode"], LlmMode):
      allowed_data["mode"] = LlmMode(allowed_data["mode"])
  ```
- **Method**: **Verified**: a dict without `"mode"` raises `KeyError: 'mode'` even though `mode` has a dataclass default; missing `"provider"` raised `KeyError` instead of the documented `ValueError`. Guarding `mode` lets the default apply and converts the provider error into the documented `ValueError`. Pitfall: if ARC-007 refactored `from_json`, apply the same guards to the new code.
- **Verify**: `uv run python -c "from par_ai_core.llm_config import LlmConfig; LlmConfig.from_json({'provider':'OpenAI','model_name':'gpt-4'})"` succeeds (mode defaults); `from_json({'model_name':'x'})` raises `ValueError`. Add both as tests. `make checkall`.

### [QA-006] Log all exception-swallow sites; `raise_on_error` option
- **Files**: `src/par_ai_core/web_tools.py:251-256`, `src/par_ai_core/pricing_lookup.py:173-178`, `src/par_ai_core/search_utils.py:388-389`.
- **Steps**: (1) Add a module-level `logger = logging.getLogger("par_ai")` to each module lacking one. (2) In each `except Exception` swallow site, `logger.warning(...)` with the context (URL, model name, operation) before returning the fallback. (3) Add `raise_on_error: bool = False` to `fetch_url`; when True, re-raise instead of returning `[""]`. (4) For `youtube_get_comments`'s `except: break`, log a warning naming the API error before breaking.
- **Method**: Silent failures make "empty" indistinguishable from "broken" and cause cost under-reporting. Pitfall: preserve the existing default behavior (return fallback) so callers aren't broken — only add logging + an opt-in raise. Sequence **after** SEC-001/SEC-002 (same functions).
- **Verify**: A test that `fetch_url(bad, raise_on_error=True)` raises and `caplog` captures a warning on the default path. `uv run pytest tests/ -k "fetch or pricing or youtube" -q`. `make checkall`.

### [QA-007] `removesuffix` in `has_value`; align depth comment
- **Files**: `src/par_ai_core/utils.py:377`, comment at `:357,362`.
- **Steps**: (1) Change `search = search.rstrip(".00")` to `search = search.removesuffix(".00")`. (2) Change the comment "don't go more than 3 levels deep" to match the code (`depth > 4`), or change the guard to `depth > 3` if 3 was intended — decide by reading the recursion (it increments `depth+1` per level; keep `> 4` and fix the comment to "4 levels" for a minimal change).
- **Method**: **Verified**: `"100".rstrip(".00")` → `"1"`, so `rstrip` strips characters, not the literal suffix, breaking numeric matches ending in 0. `removesuffix` removes only the exact `.00` suffix. Pitfall: `removesuffix` needs Python 3.9+ (satisfied by 3.11 floor). Keep the comment/code consistent to avoid a future "fix" that changes depth behavior.
- **Verify**: `uv run python -c "from par_ai_core.utils import has_value; assert has_value(100,'100') and not has_value(1,'100')"`. Add a regression test. `make checkall`.

### [QA-008] (fold into ARC-007) — LlmConfig field shotgun surgery. See ARC-007.

### [QA-009] Single `_is_reasoning_model` helper
- **Files**: `src/par_ai_core/llm_config.py:860-863,875-879`.
- **Steps**: Add a module helper:
  ```python
  _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
  def _is_reasoning_model(model_name: str) -> bool:
      name = model_name.lower()
      return any(name.startswith(p) for p in _REASONING_MODEL_PREFIXES)
  ```
  Replace both duplicated `startswith(...)` chains with `_is_reasoning_model(self.model_name)`.
- **Method**: The inline check was duplicated and missed `gpt-5`/`gpt-5-mini` (no dot) and `o4-mini`; `provider_light_models` defaults to `gpt-5-mini`, which the old check excluded. A single helper with an explicit prefix tuple fixes both sites at once. Pitfall: matching `gpt-5` as a prefix also matches `gpt-50` hypothetically — acceptable given current naming, but note it; parametrize the test with current model names.
- **Verify**: Parametrized test enumerating `o1`, `o3-mini`, `o4-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-4o` (last must be False). `uv run pytest tests/test_llm_config.py -q`. `make checkall`.

### [QA-010] Wire or remove `days` in `serper_search`
- **Files**: `src/par_ai_core/search_utils.py:211-255`.
- **Steps**: Either (preferred) pass a freshness filter to `GoogleSerperAPIWrapper` via its `tbs` parameter mapped from `days` (e.g. `tbs="qdr:d"` for 1 day, `"qdr:w"`, `"qdr:m"`, `"qdr:y"`), or remove the `days` parameter and its validation entirely if the wrapper can't support it. Also rename the `type` parameter to `search_type` to stop shadowing the builtin.
- **Method**: `days` is validated then ignored, silently misleading users who think they're date-filtering. Pitfall: verify `GoogleSerperAPIWrapper`'s actual parameter for time filtering before wiring (check the installed version's signature); if unavailable, removal is the honest fix.
- **Verify**: A test that `serper_search(..., days=1)` either passes the expected `tbs` or the parameter no longer exists. `make checkall`.

### [QA-011] Pass `max_comments` through `youtube_search`
- **Files**: `src/par_ai_core/search_utils.py:469`.
- **Steps**: Change `youtube_get_comments(youtube, video_id)` to `youtube_get_comments(youtube, video_id, max_results=max_comments)` (confirm the parameter name on `youtube_get_comments`).
- **Method**: The caller's `max_comments` only gated whether to fetch; the fetch always used the default 10. Pitfall: check `youtube_get_comments`'s signature — pass the value as the correct positional/keyword; ensure pagination respects it.
- **Verify**: A test that `youtube_search(..., max_comments=50)` requests 50. `make checkall`.

### [QA-012] Reuse `csv_to_table` in `display_formatted_output` (≡ ARC-022)
- **Files**: `src/par_ai_core/output_utils.py:244-253`.
- **Steps**: Replace the inline CSV table-building block with `console.print(csv_to_table(content))`.
- **Method**: The inline path raises uncaught `StopIteration` on empty CSV and Rich errors on ragged rows, while `csv_to_table` (line 69) already handles both gracefully. Pitfall: confirm `csv_to_table` returns a Rich renderable that `console.print` accepts (it does); preserve any title/formatting the inline path provided.
- **Verify**: A test that `display_formatted_output("", DisplayOutputFormat.CSV)` does not raise. `uv run pytest tests/ -k output -q`. `make checkall`.

### [QA-013] (≡ ARC-002) — see Phase 2 ARC-002. Fix once.

### [QA-014] Lazy per-engine imports in search_utils (after ARC-001)
- **Files**: `src/par_ai_core/search_utils.py:52-65`.
- **Steps**: Folded into ARC-001 step 3 — move `praw`, `googleapiclient`, `tavily`, and the `llm_utils`→`litellm` chain imports inside the functions that use them, with actionable ImportError messages.
- **Method**: Matches the established lazy-import convention in `llm_config.py`/`web_tools.py`. Pitfall: the `llm_utils` import chain (which pulls litellm) is the subtle one — defer `summarize_content`'s import too.
- **Verify**: Covered by ARC-001 (import a single search engine without others installed).

### [QA-015] Normalize token payload before accumulation
- **Files**: `src/par_ai_core/pricing_lookup.py:206-215`.
- **Steps**: Before accumulating, normalize the payload to one convention: `input_tokens = payload.get("input_tokens") or payload.get("prompt_tokens") or 0` and `output_tokens = payload.get("output_tokens") or payload.get("completion_tokens") or 0`; accumulate each once and compute `total_tokens = input_tokens + output_tokens` consistently.
- **Method**: The current branch adds both `prompt_tokens` and `input_tokens` (double-count when both present) and computes `total_tokens` from only one pair. Pitfall: some providers send both conventions with the same value — `or` (not `+`) prevents the double count. Preserve any additional token fields (cache/reasoning tokens) the code already tracks.
- **Verify**: A test with a payload carrying both `prompt_tokens` and `input_tokens` yields the single expected total. `uv run pytest tests/ -k "cost or pricing or accumulate" -q`. `make checkall`.

### [QA-016] (≡ ARC-006) — see Phase 3b ARC-006. Fix once.

### [QA-017] Non-blocking PAUSE wait in async path
- **Files**: `src/par_ai_core/web_tools.py:315-316`.
- **Steps**: Replace `input()` at line 316 with `await asyncio.to_thread(input)`.
- **Method**: Synchronous `input()` in a coroutine blocks the event loop and all parallel fetches. `asyncio.to_thread` offloads the blocking call. Pitfall: PAUSE is an interactive debug feature — ensure `asyncio` is imported (it is, used by `fetch_url`); the Selenium path's `input()` (line 462) is synchronous already, which is fine there.
- **Verify**: `uv run python -c "import ast,sys; ast.parse(open('src/par_ai_core/web_tools.py').read())"` parses; `make checkall`.

### [QA-018] Naming fixes
- **Files**: `src/par_ai_core/utils.py:60` (`DECIMAL_PRECESSION`), `src/par_ai_core/llm_image_utils.py:42` (`image_path: bytes`), `src/par_ai_core/search_utils.py:214` (`type=`).
- **Steps**: (1) Rename `DECIMAL_PRECESSION` → `DECIMAL_PRECISION` (grep for all references first — `grep -rn DECIMAL_PRECESSION src/ tests/`). (2) Rename the `b64_encode_image` parameter `image_path` → `image_bytes` (it holds raw bytes). (3) `type` → `search_type` in `serper_search` (folds with QA-010).
- **Method**: Public-name renames need a full grep (per the "NO SEMANTIC SEARCH" rule): direct refs, string literals, tests, re-exports. Pitfall: `DECIMAL_PRECESSION` may be imported elsewhere — update all sites atomically. If the constant/param is public API, consider keeping an alias for one release.
- **Verify**: `grep -rn "DECIMAL_PRECESSION\|image_path: bytes" src/ tests/` returns nothing. `make checkall && make test`.

### [QA-019] Remove dead code and commented-out debug prints
- **Files**: `src/par_ai_core/utils.py:839-848,467-489`, `src/par_ai_core/time_display.py:22-25`, `src/par_ai_core/web_tools.py:173,293`, `src/par_ai_core/provider_cb_info.py:186-189`.
- **Steps**: (1) Remove `sys.version_info >= (3, 11)` shim branches (always true) in `utils.py` and `time_display.py`, keeping the 3.11+ path; remove the associated `# noqa: UP036`. (2) Remove the pointless `try: ... except Exception: raise` at `utils.py:839-848`. (3) Delete commented-out debug prints. (4) Simplify the triple-nested `csv.Error` double-wrapping in `parse_csv_text` (`utils.py:467-489`).
- **Method**: `requires-python >= 3.11` makes the version shims dead. Pitfall: confirm each shim's "false" branch is genuinely unreachable before deleting; run the full suite since `parse_csv_text` error handling is behavior-bearing.
- **Verify**: `grep -rn "version_info >= (3, 11)" src/` returns nothing; `uv run pytest tests/ -k "csv or utils" -q`. `make checkall`.

### [QA-020] Broaden ruff rule set; qualify type-ignores
- **Files**: `ruff.toml`, `src/` (31 `# type: ignore`).
- **Steps**: (1) Add `B` (bugbear) and `UP` (pyupgrade) to the ruff `select` list; consider `C901` with a sane `max-complexity`. (2) Set `target-version = "py311"` (≡ ARC-020). (3) Run `make lint` and fix new findings (or add scoped `# noqa` with justification). (4) Qualify bare `# type: ignore` → `# type: ignore[specific-code]` where pyright reports the code.
- **Method**: A minimal rule set misses real bugs (bugbear catches mutable-default-arg, etc.). Pitfall: enabling `B`/`UP` will surface findings — budget time to fix them; don't blanket-`noqa`. Qualifying type-ignores prevents them from masking new errors.
- **Verify**: `make lint` passes with the broadened set; `grep -rn "# type: ignore$" src/ | wc -l` decreases. `make checkall`.

### [QA-021] Fix desktop Chrome UA "Mobile Safari" token
- **Files**: `src/par_ai_core/user_agents.py`.
- **Steps**: In the Chrome UA branch, use the desktop token `Safari/537.36` (no "Mobile") for desktop OS strings; reserve "Mobile Safari" for mobile UA variants only.
- **Method**: "Mobile Safari" on a desktop OS string is an inconsistent fingerprint. Pitfall: match real Chrome desktop UA format exactly; check whether any test asserts the current (wrong) string and update it.
- **Verify**: A test that desktop Chrome UAs do not contain "Mobile". `make checkall`.

### [QA-022] Fix `get_files` ext semantics/docs
- **Files**: `src/par_ai_core/utils.py:270`.
- **Steps**: Decide intended behavior. If `ext` should *include* matching files (per the docstring "filter by"), fix the logic to keep matches. If it should *exclude*, fix the docstring to say "exclude files with these extensions". Prefer matching the docstring (include) unless callers depend on exclude — grep `get_files(` usages first.
- **Method**: Name/docstring say filter-by (include) but code excludes — one must change. Pitfall: this is behavior-bearing; check every caller before flipping the logic, or you break them. If callers rely on exclude, keep behavior and fix only the docstring + parameter name.
- **Verify**: Tests covering both an include and exclude case match the chosen contract. `make checkall`.

### [QA-023] (fold into QA-007) — depth-limit comment. See QA-007.

### [QA-024] Soup-based pre-tag handling in html_to_markdown
- **Files**: `src/par_ai_core/web_tools.py:631-632`.
- **Steps**: Folded into ARC-013 (converter consolidation). If ARC-013 is deferred, replace the `html_content.replace("<pre", "``` <pre")` string surgery with a BeautifulSoup transform that wraps `<pre>` content in fenced code blocks properly.
- **Method**: String replacement corrupts any literal `<pre` in text and mishandles nesting. Pitfall: prefer doing this as part of ARC-013's single-pipeline choice rather than patching the `html2text` path that ARC-013 removes.
- **Verify**: A test converting HTML containing a literal `<pre` in text does not corrupt it. `make checkall`.

### [QA-025] (≡ ARC-021) — see Phase 3b ARC-021. Fix once.

### [QA-026] (≡ DOC-001) — regenerate docs last. See DOC-001.

---

## Phase 3d — Documentation (all, parallel-safe; DOC-001 runs LAST)

### [DOC-002] Fix `PARAI_NUM_REDICT` → `PARAI_NUM_PREDICT`
- **Files**: `README.md` (Environment Variables block).
- **Steps**: Replace `PARAI_NUM_REDICT=` with `PARAI_NUM_PREDICT=`.
- **Method**: Typo; the code reads `{prefix}_NUM_PREDICT`. Pitfall: check both the `.env`-style template block and any prose mention.
- **Verify**: `grep -n "NUM_REDICT" README.md` returns nothing.

### [DOC-003] Rebuild `.env.example` from actual code reads
- **Files**: `.env.example`.
- **Steps**: (1) Fix `DEEP_AI_API_KEY` → `DEEPSEEK_API_KEY`. (2) Add missing keys the library reads: `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY`, `OLLAMA_HOST`. (3) Remove keys never read by the library: `DG_API_KEY`, `ELEVENLABS_API_KEY`, `VECTOR_STORE_URL`, `WEATHERAPI_KEY`, `GITHUB_PERSONAL_ACCESS_TOKEN`, `GITHUB_MODEL_ENDPOINT`, `XAI_ENDPOINT`, `PARAI_PRICING`, `PARAI_DISPLAY_OUTPUT`, `PARAI_YES_TO_ALL`, `PARAI_SHOW_TOOL_CALLS`. (4) Fix `LANGCHAIN_PROJECT=par_gpt` → a neutral placeholder. (5) Cross-check against `provider_env_key_names` values, search keys (`TAVILY_API_KEY`, `JINA_API_KEY`, `BRAVE_API_KEY`, `SERPER_API_KEY`, `GOOGLE_CSE_ID`, `GOOGLE_CSE_API_KEY`, `REDDIT_*`), and the `PARAI_*` set from `llm_config_from_env`.
- **Method**: The example was copied from the downstream par_gpt project and drifted. Pitfall: derive the authoritative list by grepping `os.environ.get` / `provider_env_key_names` in source — don't guess. Keep placeholder values (no real secrets).
- **Verify**: Every key in `.env.example` has a matching `os.environ.get`/`getenv` reference in `src/` (`for k in $(grep -oP '^[A-Z_]+' .env.example); do grep -rq "$k" src/ || echo "unused: $k"; done`).

### [DOC-004] Correct `llm_config_from_env` docstring
- **Files**: `src/par_ai_core/llm_utils.py:36-53`.
- **Steps**: Remove `{prefix}_MAX_CONTEXT_SIZE` from the docstring; add `{prefix}_NUM_CTX`, `{prefix}_TIMEOUT`, `{prefix}_NUM_PREDICT`, `{prefix}_REPEAT_LAST_N`, `{prefix}_REPEAT_PENALTY`, `{prefix}_MIROSTAT`, `{prefix}_MIROSTAT_ETA`, `{prefix}_MIROSTAT_TAU`, `{prefix}_TFS_Z`, `{prefix}_TOP_K`, `{prefix}_TOP_P`, `{prefix}_SEED`, `{prefix}_REASONING_EFFORT`, `{prefix}_REASONING_BUDGET`.
- **Method**: The docstring is the canonical env reference and listed a nonexistent variable while omitting 13 real ones (verified against the function body at `:73-121`). Pitfall: match the exact variable names and note which are optional. Do **before** DOC-001 (docs regen).
- **Verify**: Every `os.environ.get(f"{prefix}_...")` in the function appears in the docstring. `make checkall`.

### [DOC-005] Remove unused env vars from README
- **Files**: `README.md` ("Misc API" section + env template).
- **Steps**: Remove `WEATHERAPI_KEY` and `GITHUB_PERSONAL_ACCESS_TOKEN` (no code reads them), or annotate them explicitly as "used by downstream applications, not this library".
- **Method**: They imply weather/GitHub tooling the library doesn't ship. Pitfall: confirm with `grep -rn "WEATHERAPI_KEY\|GITHUB_PERSONAL_ACCESS_TOKEN" src/` (returns nothing) before removing.
- **Verify**: `grep -n "WEATHERAPI\|GITHUB_PERSONAL" README.md` returns nothing (or only the annotated note).

### [DOC-006] Trim "What's New" to recent releases
- **Files**: `README.md` ("What's New").
- **Steps**: Keep entries for the last 2–3 releases (through 0.5.8) and replace the older history with a link to `CHANGELOG.md`.
- **Method**: Maintaining full history in two places already drifted (README stopped at 0.5.7 while version is 0.5.8). Pitfall: add the 0.5.8 entry while trimming so the top is current.
- **Verify**: "What's New" top entry matches `__version__` in `src/par_ai_core/__init__.py`.

### [DOC-007] Complete `LlmProvider` docstring (14 providers)
- **Files**: `src/par_ai_core/llm_providers.py:61-75`.
- **Steps**: List all 14 enum members with accurate one-line descriptions; add the omitted `OPENROUTER`, `DEEPSEEK`, `LITELLM`, `AZURE`; fix `GITHUB` (GitHub Models, not Copilot) and `XAI` (drop "formerly Twitter"). Do **after** ARC-008 if that lands (registry may reorder).
- **Method**: The enum drives provider selection and feeds generated docs. Pitfall: enumerate members from the actual enum (`list(LlmProvider)`) so none are missed.
- **Verify**: Docstring lists exactly `len(list(LlmProvider))` providers. `make checkall`.

### [DOC-008] Add 5 missing `LlmConfig` attributes (after ARC-007)
- **Files**: `src/par_ai_core/llm_config.py` (class docstring Attributes section).
- **Steps**: Add `fallback_models`, `format`, `extra_body`, `reasoning_effort`, `reasoning_budget` (and `safety_settings` if SEC-003 added it, and `max_output_tokens` if ARC-006 added it) to the Attributes list.
- **Method**: The docstring stopped at `env_prefix`. Pitfall: run **after** ARC-007/SEC-003/ARC-006 so the documented field set matches the final dataclass. Cross-check against `dataclasses.fields(LlmConfig)`.
- **Verify**: Every field in `dataclasses.fields(LlmConfig)` appears in the docstring. `make checkall`.

### [DOC-009] Correct `fetch_url` wait_type default/type
- **Files**: `src/par_ai_core/web_tools.py:204` (and the same line in `fetch_url_selenium`'s docstring).
- **Steps**: Change "wait_type (WaitType, optional): ... Defaults to WaitType.IDLE" to "wait_type (ScraperWaitType, optional): ... Defaults to ScraperWaitType.SLEEP". Check `fetch_url_selenium` for the same drift (the playwright docstring at `:283` is already correct).
- **Method**: The signature default is `ScraperWaitType.SLEEP` (verified at `:267`); the docstring named the wrong type and default. Pitfall: fix every fetcher docstring, not just `fetch_url`. Do before DOC-001.
- **Verify**: `grep -n "WaitType.IDLE" src/par_ai_core/web_tools.py` returns nothing in docstrings where the default is SLEEP. `make checkall`.

### [DOC-010] Add 0.5.3 entry + dates to CHANGELOG
- **Files**: `CHANGELOG.md`.
- **Steps**: (1) Insert a `## [0.5.3]` entry ("Changed: updated dependencies") between 0.5.4 and 0.4.3. (2) Add `- YYYY-MM-DD` dates to each version header from git tag timestamps (`git log --tags --simplify-by-decoration --pretty="%ai %d"`).
- **Method**: 0.5.3 was released (git tag exists) but missing from the log; Keep a Changelog format requires dates. Pitfall: use the actual tag dates, not today's date; don't fabricate a changelog body for 0.5.3 beyond what the release did (dependency bump).
- **Verify**: `grep -n "0.5.3" CHANGELOG.md` present; each `##` version header has a date.

### [DOC-011] Fix reversed link + truncated sentence
- **Files**: `README.md` (Prerequisites; Search section).
- **Steps**: (1) Fix `See (Environment Variables)[#environment-variables]` → `See [Environment Variables](#environment-variables)`. (2) Complete or delete the truncated Tavily sentence "Tavily is much better than".
- **Method**: Reversed markdown link syntax renders as literal text on the landing page. Pitfall: check for other reversed `)[` link patterns (`grep -n ")\[" README.md`).
- **Verify**: `grep -n ")\[" README.md` returns nothing; no sentence ends mid-clause.

### [DOC-012] Link architecture/operations docs; Playwright install step
- **Files**: `README.md` (Documentation, Installation sections).
- **Steps**: (1) Add links to `docs/architecture.md` and `docs/operations.md` in the Documentation section. (2) Add a note in Installation that web scraping requires `playwright install chromium` (and the `[web]` extra once ARC-001 lands).
- **Method**: The good architecture/ops docs are undiscoverable and the missing-browser failure has no documented fix. Pitfall: use repo-relative links that work on both GitHub and PyPI.
- **Verify**: Both doc links present; `grep -n "playwright install" README.md` present.

### [DOC-013] Document Reddit credentials
- **Files**: `README.md` (Search section), `.env.example`.
- **Steps**: Add `REDDIT_USERNAME` and `REDDIT_PASSWORD` (marked optional, for authenticated Reddit search) alongside the documented `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`.
- **Method**: `search_utils.py` reads both but they were undocumented. Pitfall: mark them optional so users don't think they're required for all Reddit use.
- **Verify**: `grep -n "REDDIT_USERNAME" README.md .env.example` present.

### [DOC-014] Mermaid module-dependency diagram
- **Files**: `docs/architecture.md`.
- **Steps**: Convert the ASCII-art module dependency diagram to a Mermaid `graph TD` using the dark-mode palette from `docs/DOCUMENTATION_STYLE_GUIDE.md` (and CLAUDE.md: bg `#1E1E1E`, text `#E6E6E6`, etc.). Keep directory-tree text blocks as text (the style guide allows those).
- **Method**: The style guide requires Mermaid for architecture diagrams. Pitfall: only convert the dependency/architecture diagram, not directory trees.
- **Verify**: `grep -n "mermaid" docs/architecture.md` present; diagram renders (mentally check syntax).

### [DOC-015] Fix badge typo; add CI badge
- **Files**: `README.md` (header).
- **Steps**: Fix alt text "Arch x86-63" → "x86-64"; add a GitHub Actions build-status badge pointing at `.github/workflows/build.yml`.
- **Method**: Cosmetic but visible on the landing page. Pitfall: use the correct workflow badge URL format for the repo.
- **Verify**: `grep -n "x86-63" README.md` returns nothing; a build-status badge line is present.

### [DOC-016] Add pip install path
- **Files**: `README.md` (Prerequisites, Installation).
- **Steps**: Add `pip install par_ai_core` (and `pip install 'par_ai_core[all]'` post-ARC-001) as an alternative to `uv add`; scope UV to the development section.
- **Method**: Consumers shouldn't need UV. Pitfall: keep UV instructions for contributors; just don't imply it's required to install.
- **Verify**: `grep -n "pip install par_ai_core" README.md` present.

### [DOC-017] Remove nonexistent sdist include
- **Files**: `pyproject.toml` (`[tool.hatch.build.targets.sdist]`).
- **Steps**: Remove `extraction_prompt.md` from the sdist include list (the file does not exist in the repo), or restore the file if it was meant to ship.
- **Method**: A missing include can fail strict sdist builds. Pitfall: confirm the file truly doesn't exist (`ls extraction_prompt.md`) before removing.
- **Verify**: `uv build --sdist` succeeds; `grep -n "extraction_prompt" pyproject.toml` returns nothing.

### [DOC-018] Docstring for `ParAICallbackHandler.__init__`
- **Files**: `src/par_ai_core/provider_cb_info.py`.
- **Steps**: Add a Google-style docstring to `ParAICallbackHandler.__init__` documenting its parameters (`llm_config`, `show_prompts`, `show_end`, `show_tool_calls`, `verbose`, `console`).
- **Method**: It's the only genuinely public callable lacking a docstring. Pitfall: match the existing docstring style and the actual constructor signature.
- **Verify**: `make checkall` (if docstring coverage is checked); the constructor has a docstring.

### [DOC-001] Regenerate published API docs (**RUN LAST**)
- **Files**: `src/par_ai_core/docs/` (or the new location if ARC-015 moved it).
- **Steps**: After **all** docstring fixes (DOC-004, DOC-007, DOC-008, DOC-009, DOC-018) and web_tools code fixes (QA-001, QA-003) have landed, run `make docs`, review the output for the corrected `ignore_ssl` default and new public functions (`apply_nest_asyncio`, etc.), and commit the regenerated HTML. Add a docs-regeneration step to the release checklist (and ideally CI).
- **Method**: The published reference was 4 releases stale and misstated the `ignore_ssl` security default. Pitfall: **must run last** — regenerating before the docstring/code fixes republishes the wrong content and forces a second regen. If ARC-015 moved docs out of `src/`, regenerate into the new location and update README links. This is the QA-026 fix too.
- **Verify**: `grep -c "apply_nest_asyncio" src/par_ai_core/docs/index.html` (or new path) > 0; the `ignore_ssl` default in the regenerated docs reads `False`. Do not run `make publish` — CI handles release.

---

## Execution Notes for `/fix-audit`

- **Fix once, not twice**: ARC-004≡QA-004, ARC-002≡QA-013, ARC-006≡QA-016, ARC-022≡QA-012, ARC-021≡QA-025, ARC-017⊂ARC-007, QA-008⊂ARC-007, QA-023⊂QA-007, QA-014⊂ARC-001, ARC-003⊂ARC-001, QA-024⊂ARC-013, QA-026≡DOC-001.
- **Highest-contention files** (read current state before every edit): `web_tools.py`, `llm_config.py`, `llm_utils.py`, `search_utils.py`, `utils.py` — see the File Conflict Map in `AUDIT.md`.
- **Hard ordering**: SEC-001/002 → QA-001/003/006 (same functions); ARC-001 → ARC-003/012 + QA-014; ARC-007 → QA-005/009 + ARC-006 + DOC-008; ARC-002 → docs updates; ARC-016 → SEC-009; all docstring + web_tools fixes → DOC-001 (docs regen, last).
- **Security changes need review** (SEC-001, SEC-002, SEC-003, SEC-006, SEC-008): do not commit silently inside a larger change.
- **Final gate** after each phase: `make checkall && make test`. Do not report complete until both pass.
