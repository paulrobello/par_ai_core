# Operations Guide

This document covers deployment, driver installation, and cloud provider configuration for PAR AI Core.

## Web Driver Setup

### Playwright

Playwright is used for headless browser-based web scraping. Install the browser drivers:

```bash
# Install Chromium (recommended)
playwright install chromium

# Or install all browsers
playwright install
```

Playwright is the preferred fetch backend. Use `fetch_using="playwright"` (the default) in `fetch_url`.

### Selenium

Selenium uses `webdriver-manager` to automatically download and manage ChromeDriver. No manual driver installation is required — it downloads the correct version on first use.

If you need a specific Chrome binary, set the `CHROME_BIN` environment variable.

## Cloud Provider Configuration

### AWS Bedrock

Bedrock requires AWS credentials. Configure using one of:

```bash
# Option 1: Named profile
AWS_PROFILE=your-profile-name

# Option 2: Access keys
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional
AWS_REGION=us-east-1          # Defaults to us-east-1
AWS_SESSION_TOKEN=your-token  # For temporary credentials
```

IAM permissions required:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream` (for streaming)

The `user_agent_appid` config parameter is passed to the boto3 client config for request identification.

### Azure OpenAI

```bash
# Required
AZURE_OPENAI_API_KEY=your-azure-key   # Falls back to OPENAI_API_KEY if not set
PARAI_AI_BASE_URL=https://your-resource.openai.azure.com/

# Provider config
PARAI_AI_PROVIDER=Azure
PARAI_MODEL=your-deployment-name       # Azure deployment name, not model name
```

The API version is set to `2025-03-01-preview` via the `AZURE_API_VERSION` constant in `llm_config.py`.

### Google Gemini

```bash
GOOGLE_API_KEY=your-google-api-key
PARAI_AI_PROVIDER=Gemini
```

Safety settings are disabled by default (`HARM_CATEGORY_UNSPECIFIED: BLOCK_NONE`).

## SSL Verification

By default, SSL certificates are verified for all web fetch operations (`ignore_ssl=False`). To disable verification for development/testing with self-signed certificates:

```python
from par_ai_core.web_tools import fetch_url

results = fetch_url("https://self-signed.example.com", ignore_ssl=True)
```

## Proxy Configuration

Web tools support HTTP proxies:

```python
from par_ai_core.web_tools import fetch_url

results = fetch_url(
    "https://example.com",
    http_proxy="http://proxy.example.com:8080",
)
```

## Logging

Set the log level via environment variable:

```bash
PARAI_LOG_LEVEL=DEBUG   # DEBUG, INFO, WARNING, ERROR
```

The library uses Python's standard `logging` module. Internal debug messages (e.g., failed file reads in context gathering) are logged at `DEBUG` level.

For TUI applications, avoid writing logs to the screen — use file-based logging instead:

```python
import logging
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    encoding="utf-8",
)
```

## Related Documentation

- [Architecture Overview](architecture.md) — module structure and provider support matrix
- [README](../README.md) — quickstart and environment variable reference
