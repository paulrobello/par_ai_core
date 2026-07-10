"""
Web Tools Module

This module provides a set of utilities for web-related tasks, including web searching,
HTML parsing, and web page fetching. It offers functionality to interact with search
engines, extract information from web pages, and handle various web-related operations.

Key Features:
- Web searching using Google Custom Search API
- HTML element extraction
- Web page fetching using either Playwright or Selenium
- URL content fetching and conversion to Markdown

The module includes tools for:
1. Performing Google web searches
2. Extracting specific HTML elements from web pages
3. Fetching web page content using different methods (Playwright or Selenium)
4. Converting fetched web content to Markdown format

Dependencies:
- BeautifulSoup for HTML parsing
- Pydantic for data modeling
- Rich for console output formatting
- Playwright or Selenium for web page interaction (configurable)
- html2text for HTML to Markdown conversion

This module is part of the par_ai_core package and is designed to be used in
conjunction with other AI and web scraping related tasks.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import threading
import time
from enum import StrEnum
from queue import Queue
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from html2text import HTML2Text
from pydantic import BaseModel
from rich.console import Console
from rich.repr import rich_repr

from par_ai_core.par_logging import console_err
from par_ai_core.user_agents import get_random_user_agent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Playwright is an optional backend (install with ``par_ai_core[web]``).
    # These are type-only imports; annotations are stringified via
    # ``from __future__ import annotations``. The runtime ``async_playwright``
    # entry point is imported lazily inside ``fetch_url_playwright``.
    from playwright.async_api import HttpCredentials, ProxySettings


class ScraperChoice(StrEnum):
    """Enum for scraper choices."""

    SELENIUM = "selenium"
    PLAYWRIGHT = "playwright"


class ScraperWaitType(StrEnum):
    """Enum for scraper wait type choices."""

    NONE = "none"
    PAUSE = "pause"
    SLEEP = "sleep"
    IDLE = "idle"
    SELECTOR = "selector"
    TEXT = "text"


def normalize_url(url: str, strip_fragment: bool = True, strip_query: bool = True, strip_slash: bool = True) -> str:
    """
    Normalize URL by removing trailing slashes, fragments and query params.
    Args:
        url (str): The URL to normalize.
        strip_fragment (bool): Whether to remove the fragment from the URL.
        strip_query (bool): Whether to remove the query parameters from the URL.
        strip_slash (bool): Whether to remove the trailing slash from the URL.
    Returns:
        str: The normalized URL.

    """
    if strip_fragment:
        url = url.split("#", 1)[0]  # Remove fragment
    if strip_query:
        url = url.split("?", 1)[0]  # Remove query
    if strip_slash:
        return url.rstrip("/")
    return url


def inject_credentials(url: str, username: str, password: str) -> str:
    """
    Injects the given username and password into the URL, handling special characters.

    Args:
        url (str): The original URL.
        username (str): The username to inject.
        password (str): The password to inject.

    Returns:
        str: The URL with the injected credentials.

    Warning:
        Credentials embedded in the URL netloc can leak into logs, browser history,
        and HTTP ``Referer`` headers. Prefer passing ``http_credentials`` to the
        Playwright browser context (which does not place credentials in the URL) over
        this helper. Never log the returned value; log the original URL instead.
    """
    parsed_url = urlparse(url)
    encoded_username = quote(username)
    encoded_password = quote(password)
    netloc = f"{encoded_username}:{encoded_password}@{parsed_url.hostname}"
    if parsed_url.port:
        netloc += f":{parsed_url.port}"
    new_url = parsed_url._replace(netloc=netloc)
    return urlunparse(new_url)


@rich_repr
class GoogleSearchResult(BaseModel):
    """Google search result."""

    title: str
    link: str
    snippet: str


def web_search(
    query: str, *, num_results: int = 3, verbose: bool = False, console: Console | None = None
) -> list[GoogleSearchResult]:
    """Perform a Google web search using the Google Custom Search API.

    Args:
        query: The search query to execute
        num_results: Maximum number of results to return. Defaults to 3.
        verbose: Whether to print verbose output. Defaults to False.
        console: Console to use for output. Defaults to console_err.

    Returns:
        list[GoogleSearchResult]: List of search results containing title, link and snippet

    Raises:
        ValueError: If GOOGLE_CSE_ID or GOOGLE_CSE_API_KEY environment variables are not set
    """
    from langchain_google_community import GoogleSearchAPIWrapper

    if verbose:
        if not console:
            console = console_err
        console.print(f"[bold green]Web search:[bold yellow] {query}")

    google_cse_id = os.environ.get("GOOGLE_CSE_ID")
    google_api_key = os.environ.get("GOOGLE_CSE_API_KEY")

    if not google_cse_id or not google_api_key:
        raise ValueError("Missing required environment variables: GOOGLE_CSE_ID and GOOGLE_CSE_API_KEY must be set")

    search = GoogleSearchAPIWrapper(
        google_cse_id=google_cse_id,
        google_api_key=google_api_key,
    )
    return [GoogleSearchResult(**result) for result in search.results(query, num_results=num_results)]


def get_html_element(element: str, soup: BeautifulSoup) -> str:
    """Search for and return text of first matching HTML element.

    Args:
        element: The tag name of the HTML element to search for (e.g., 'h1', 'div')
        soup: BeautifulSoup object containing the parsed HTML document

    Returns:
        str: Text content of first matching element, or empty string if not found
    """
    result = soup.find(element)
    if result:
        return result.text

    return ""


_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def is_safe_public_url(url: str) -> bool:
    """Return True only for http/https URLs whose host resolves to public IP addresses.

    Blocks ``file://``, ``chrome://``, and other non-http(s) schemes, and rejects hosts
    that resolve to private, loopback, link-local, reserved, or multicast ranges. This
    mitigates SSRF and local-file disclosure (CWE-918) when URLs may originate from
    model-generated or user input.

    Note:
        This narrows but does not fully close the DNS-rebinding window. Callers handling
        untrusted URLs should additionally pin the resolved address and reject redirects
        to private IPs. The check performs a live DNS resolution, so it requires network
        access for hostname URLs (literal-IP URLs need no resolution).

    Args:
        url: The URL to check.

    Returns:
        True if the URL uses http/https and resolves only to public IPs; False otherwise.
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


def fetch_url(
    urls: str | list[str],
    *,
    fetch_using: Literal["playwright", "selenium"] = "playwright",
    max_parallel: int = 1,
    sleep_time: int = 1,
    timeout: int = 10,
    proxy_config: ProxySettings | None = None,
    http_credentials: HttpCredentials | None = None,
    wait_type: ScraperWaitType = ScraperWaitType.SLEEP,
    wait_selector: str | None = None,
    headless: bool = True,
    verbose: bool = False,
    ignore_ssl: bool = False,
    console: Console | None = None,
    raise_on_error: bool = False,
) -> list[str]:
    """
    Fetch the contents of a webpage using either Playwright or Selenium.

    Args:
        urls (str | list[str]): The URL(s) to fetch.
        fetch_using (Literal["playwright", "selenium"]): The library to use for fetching the webpage.
        max_parallel (int): The maximum number of parallel requests.
        sleep_time (int): The number of seconds to sleep between requests.
        timeout (int): The number of seconds to wait for a response.
        proxy_config (ProxySettings | None): Proxy configuration. Defaults to None.
        http_credentials (HttpCredentials | None): HTTP credentials for authentication. Defaults to None.
        wait_type (ScraperWaitType, optional): The type of wait to use. Defaults to ScraperWaitType.SLEEP.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless (bool): Whether to run the browser in headless mode.
        verbose (bool): Whether to print verbose output.
        ignore_ssl (bool): Whether to ignore SSL errors. ``ignore_ssl=True`` disables certificate
            validation and must never be combined with ``http_credentials`` (credentials would be
            exposed to a man-in-the-middle). A visible warning is emitted whenever this is set.
        console (Console | None): The console to use for output. Defaults to console_err.
        raise_on_error (bool): When True, re-raise fetch exceptions instead of returning a list of
            empty strings. Defaults to False (preserve the silent-fallback behavior callers rely on;
            the failure is logged at warning level either way).

    Returns:
        list[str]: A list of HTML contents of the fetched webpages.

    Raises:
        ValueError: If any URL is not a public http(s) URL (SSRF/local-file guard), or if
            ``ignore_ssl=True`` is combined with ``http_credentials``.
    """
    if isinstance(urls, str):
        urls = [urls]
    bad = [u for u in urls if not is_safe_public_url(u)]
    if bad:
        raise ValueError(
            f"Refusing to fetch unsafe or non-public URL(s): {bad}. "
            "Only public http(s) URLs are allowed (SSRF/local-file protection)."
        )
    if ignore_ssl:
        if not console:
            console = console_err
        console.print(
            "[bold yellow]WARNING: TLS certificate validation is disabled (ignore_ssl=True). "
            "Do not use with untrusted networks or credentials.[/bold yellow]"
        )
        if http_credentials:
            raise ValueError(
                "Refusing to inject HTTP credentials while ignore_ssl=True "
                "(credentials would be exposed to a man-in-the-middle)."
            )
    try:
        if fetch_using == "playwright":
            return asyncio.run(
                fetch_url_playwright(
                    urls,
                    max_parallel=max_parallel,
                    sleep_time=sleep_time,
                    timeout=timeout,
                    proxy_config=proxy_config,
                    http_credentials=http_credentials,
                    wait_type=wait_type,
                    wait_selector=wait_selector,
                    headless=headless,
                    verbose=verbose,
                    ignore_ssl=ignore_ssl,
                    console=console,
                )
            )

        return fetch_url_selenium(
            urls,
            max_parallel=max_parallel,
            sleep_time=sleep_time,
            timeout=timeout,
            proxy_config=proxy_config,
            http_credentials=http_credentials,
            wait_type=wait_type,
            wait_selector=wait_selector,
            headless=headless,
            verbose=verbose,
            ignore_ssl=ignore_ssl,
            console=console,
        )
    except Exception as e:
        if verbose:
            if not console:
                console = console_err
            console.print(f"[bold red]Error fetching URL: {str(e)}[/bold red]")
        logger.warning("fetch_url failed for %s: %s", urls, e)
        if raise_on_error:
            raise
        return [""] * (len(urls) if isinstance(urls, list) else 1)


async def fetch_url_playwright(
    urls: str | list[str],
    *,
    max_parallel: int = 1,
    sleep_time: int = 1,
    timeout: int = 10,
    proxy_config: ProxySettings | None = None,
    http_credentials: HttpCredentials | None = None,
    wait_type: ScraperWaitType = ScraperWaitType.SLEEP,
    wait_selector: str | None = None,
    headless: bool = True,
    ignore_ssl: bool = False,
    verbose: bool = False,
    console: Console | None = None,
) -> list[str]:
    """
    Fetch the contents of a webpage using Playwright.
    Args:
        urls (str | list[str]): The URL(s) to fetch.
        max_parallel (int): The maximum number of parallel requests.
        sleep_time (int): The number of seconds to sleep between requests.
        timeout (int): The number of seconds to wait for a response.
        proxy_config (ProxySettings | None): Proxy configuration. Defaults to None.
        http_credentials (HttpCredentials | None): HTTP credentials for authentication. Defaults to None.
        wait_type (ScraperWaitType, optional): The type of wait to use. Defaults to ScraperWaitType.SLEEP.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless (bool): Whether to run the browser in headless mode.
        ignore_ssl (bool): Whether to ignore SSL errors. ``ignore_ssl=True`` disables certificate
            validation and must never be combined with ``http_credentials``.
        verbose (bool): Whether to print verbose output.
        console (Console | None): The console to use for output. Defaults to console_err.

    Returns:
        list[str]: A list of HTML contents of the fetched webpages.
    """
    if isinstance(urls, str):
        urls = [urls]

    if not console:
        console = console_err

    async def fetch_page(url: str, browser) -> str:
        context = await browser.new_context(
            viewport={"width": 1280, "height": 1024},
            user_agent=get_random_user_agent(),
            ignore_https_errors=ignore_ssl,
            http_credentials=http_credentials,
        )
        page = await context.new_page()
        try:
            if verbose:
                console.print(f"[bold blue]Playwright fetching content from {url}...[/bold blue]")
            await page.goto(url, timeout=timeout * 1000)

            if wait_type == ScraperWaitType.PAUSE:
                console.print("[yellow]Press Enter to continue...[/yellow]")
                # input() blocks the event loop and every parallel fetch with it;
                # offload the blocking call to a worker thread.
                await asyncio.to_thread(input)
            elif wait_type == ScraperWaitType.SLEEP:
                if verbose:
                    console.print(f"[yellow]Waiting {sleep_time} seconds...[/yellow]")
                await page.wait_for_timeout(sleep_time * 1000)
            elif wait_type == ScraperWaitType.IDLE:
                if verbose:
                    console.print("[yellow]Waiting for networkidle...[/yellow]")
                await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            elif wait_type == ScraperWaitType.SELECTOR:
                if wait_selector:
                    if verbose:
                        console.print(f"[yellow]Waiting for selector {wait_selector}...[/yellow]")
                    await page.wait_for_selector(wait_selector, timeout=timeout * 1000)
                else:
                    if verbose:
                        console.print(
                            "[bold yellow]Warning:[/bold yellow] Please specify a selector when using wait_type=selector."
                        )
            elif wait_type == ScraperWaitType.TEXT:
                if wait_selector:
                    if verbose:
                        console.print(f"[yellow]Waiting for text {wait_selector}...[/yellow]")
                    # ``Locator.wait_for_text`` does not exist in Playwright's
                    # Python API; calling it raised AttributeError which was
                    # swallowed by the broad except below, so TEXT wait silently
                    # returned empty content. ``wait_for_function`` with an arg
                    # is the supported polling primitive (QA-001).
                    await page.wait_for_function(
                        "text => document.body && document.body.innerText.includes(text)",
                        arg=wait_selector,
                        timeout=timeout * 1000,
                    )
                else:
                    if verbose:
                        console.print(
                            "[bold yellow]Warning:[/bold yellow] Please specify a selector when using wait_type=text."
                        )

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            html = await page.content()
            return html
        except Exception as e:
            if verbose:
                console.print(f"[bold red]Error fetching content from {url}[/bold red]: {str(e)}")
            return ""
        finally:
            await context.close()

    try:
        from playwright.async_api import async_playwright
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Playwright-based fetching requires the playwright package: pip install 'par_ai_core[web]'"
        ) from e

    async with async_playwright() as p:
        browser = await p.chromium.launch(proxy=proxy_config, headless=headless)
        try:
            semaphore = asyncio.Semaphore(max_parallel)

            async def fetch_with_semaphore(url):
                async with semaphore:
                    return await fetch_page(url, browser)

            results = await asyncio.gather(*[fetch_with_semaphore(url) for url in urls])
        finally:
            await browser.close()
    return results


def fetch_url_selenium(
    urls: str | list[str],
    *,
    max_parallel: int = 1,
    sleep_time: int = 1,
    timeout: int = 10,
    proxy_config: ProxySettings | None = None,
    http_credentials: HttpCredentials | None = None,
    wait_type: ScraperWaitType = ScraperWaitType.SLEEP,
    wait_selector: str | None = None,
    headless: bool = True,
    ignore_ssl: bool = False,
    verbose: bool = False,
    console: Console | None = None,
) -> list[str]:
    """Fetch the contents of a webpage using Selenium with parallel requests using the same driver.

    Args:
        urls: The URL(s) to fetch
        max_parallel: The maximum number of parallel requests
        sleep_time: The number of seconds to sleep between requests
        timeout: The number of seconds to wait for a response
        proxy_config (ProxySettings, optional): Proxy configuration. Defaults to None.
        http_credentials (HttpCredentials, optional): HTTP credentials for authentication. Defaults to None.
        wait_type (ScraperWaitType, optional): The type of wait to use. Defaults to ScraperWaitType.SLEEP.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless: Whether to run the browser in headless mode
        ignore_ssl: Whether to ignore SSL errors. ``ignore_ssl=True`` disables certificate
            validation and must never be combined with ``http_credentials``.
        verbose: Whether to print verbose output
        console: The console to use for printing verbose output

    Returns:
        A list of HTML contents of the fetched webpages
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.wait import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager

    if not console:
        console = console_err

    if isinstance(urls, str):
        urls = [urls]

    os.environ["WDM_LOG_LEVEL"] = "0"
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1024")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])  # Disable logging
    options.add_argument("--log-level=3")  # Suppress console logging
    options.add_argument("--silent")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    if ignore_ssl:
        options.add_argument("--ignore-certificate-errors")
    # Randomize user-agent to mimic different users
    options.add_argument("user-agent=" + get_random_user_agent())
    if headless:
        options.add_argument("--window-position=-2400,-2400")
        options.add_argument("--headless=new")
    if proxy_config and "server" in proxy_config:
        options.add_argument(f"--proxy-server={proxy_config['server']}")

    results: list[str] = [""] * len(urls)
    queue = Queue()

    def worker():
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)  # type: ignore[operator]
            driver.set_page_load_timeout(timeout)

            while not queue.empty():
                index, url = queue.get()
                original_url = url
                try:
                    if verbose:
                        console.print(f"[bold blue]Selenium fetching content from {original_url}...[/bold blue]")

                    if http_credentials and "username" in http_credentials and "password" in http_credentials:
                        url = inject_credentials(url, http_credentials["username"], http_credentials["password"])

                    driver.get(url)
                    if wait_type == ScraperWaitType.PAUSE:
                        console.print("[yellow]Press Enter to continue...[/yellow]")
                        input()
                    elif wait_type == ScraperWaitType.SLEEP:
                        # The actual sleep (was a commented-out no-op while two
                        # unconditional sleeps ran for every wait type below).
                        if sleep_time > 0:
                            if verbose:
                                console.print(f"[bold yellow]Sleeping for {sleep_time} seconds...[/bold yellow]")
                            time.sleep(sleep_time)
                    elif wait_type == ScraperWaitType.IDLE:
                        time.sleep(1)
                    elif wait_type == ScraperWaitType.SELECTOR:
                        if wait_selector:
                            wait = WebDriverWait(driver, 10)
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
                    elif wait_type == ScraperWaitType.TEXT:
                        if wait_selector:
                            wait = WebDriverWait(driver, 10)
                            wait.until(EC.text_to_be_present_in_element((By.TAG_NAME, "body"), wait_selector))
                    # ScraperWaitType.NONE falls through with no waiting, so a
                    # "no wait" request no longer sleeps at all (QA-003).
                    if verbose:
                        console.print(
                            "[bold green]Page loaded. Scrolling and waiting for dynamic content...[/bold green]"
                        )
                    # Scroll to the bottom of the page to trigger lazy-loaded content
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                    results[index] = driver.page_source
                except Exception as e:
                    if verbose:
                        console.print(f"[bold red]Error fetching content from {original_url}: {str(e)}[/bold red]")
                    results[index] = ""
                finally:
                    queue.task_done()
        except Exception as e:
            if verbose:
                console.print(f"[bold red]Error initializing Selenium driver: {e}[/bold red]")
            while not queue.empty():
                queue.get()
                queue.task_done()
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    for i, url in enumerate(urls):
        queue.put((i, url))

    threads = []
    for _ in range(max_parallel):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)

    queue.join()

    for thread in threads:
        thread.join()

    return results


def html_to_markdown(
    html_content: str,
    *,
    url: str | None = None,
    include_links: bool = True,
    include_images: bool = False,
    include_metadata: bool = False,
    tags: list[str] | None = None,
    meta: list[str] | None = None,
) -> str:
    """
    Fetch the contents of a webpage and convert it to markdown.

    Args:
        html_content (str): The raw html.
        url (str, optional): The URL of the webpage. Used for converting relative links. Defaults to None.
        include_links (bool, optional): Whether to include links in the markdown. Defaults to True.
        include_images (bool, optional): Whether to include images in the markdown. Defaults to False.
        include_metadata (bool, optional): Whether to include a metadata section in the markdown. Defaults to False.
        tags (list[str], optional): A list of tags to include in the markdown metadata. Defaults to None.
        meta (list[str], optional): A list of metadata attributes to include in the markdown. Defaults to None.

    Returns:
        str: The converted markdown content
    """
    if tags is None:
        tags = []
    if meta is None:
        meta = []

    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.text if soup.title else None

    if include_links:
        url_attributes = [
            "href",
            "src",
            "action",
            "data",
            "poster",
            "background",
            "cite",
            "codebase",
            "formaction",
            "icon",
        ]

        # Convert relative links to fully qualified URLs
        for tag in soup.find_all(True):
            for attribute in url_attributes:
                if tag.has_attr(attribute):
                    attr_value = tag[attribute]
                    if attr_value.startswith("//"):  # type: ignore[reportAttributeAccessIssue]
                        tag[attribute] = f"https:{attr_value}"
                    elif url and not attr_value.startswith(("http://", "https://")):  # type: ignore[reportAttributeAccessIssue]
                        tag[attribute] = urljoin(url, attr_value)  # type: ignore[reportArgumentType]

    metadata = {
        "source": url,
        "title": title or "",
        "tags": (" ".join(tags)).strip(),
    }
    for m in soup.find_all("meta"):
        n = m.get("name", "").strip()  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        if not n:
            continue
        v = m.get("content", "").strip()  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        if not v:
            continue
        if n in meta:
            metadata[n] = v

    elements_to_remove = [
        "head",
        "header",
        "footer",
        "script",
        "source",
        "style",
        "svg",
        "iframe",
    ]
    if not include_links:
        elements_to_remove.append("a")
        elements_to_remove.append("link")

    if not include_images:
        elements_to_remove.append("img")

    for element in elements_to_remove:
        for tag in soup.find_all(element):
            tag.decompose()

    ### text separators
    # Convert separator elements to <hr>
    for element in soup.find_all(None, attrs={"role": "separator"}):
        hr = soup.new_tag("hr")
        element.replace_with(hr)
        # Add extra newlines around hr to ensure proper markdown rendering
        hr.insert_before(soup.new_string("\n"))
        hr.insert_after(soup.new_string("\n"))

    ### code blocks
    # Wrap each <pre> block in markdown fenced-code markers. Doing this on the
    # parsed tree (not via ``str(soup).replace("<pre", ...)``) avoids corrupting
    # any literal ``<pre`` text that html2text would otherwise emit verbatim and
    # handles arbitrary attributes on the tag (QA-024).
    for pre in soup.find_all("pre"):
        pre.insert_before(soup.new_string("\n```\n"))
        pre.insert_after(soup.new_string("\n```\n"))

    html_content = str(soup)

    ### convert to markdown
    converter = HTML2Text()
    converter.ignore_links = not include_links
    converter.ignore_images = not include_images
    markdown = converter.handle(html_content)

    if include_metadata:
        meta_markdown = "# Metadata\n\n"
        for k, v in metadata.items():
            meta_markdown += f"- {k}: {v}\n"
        markdown = meta_markdown + markdown
    return markdown.strip()


def fetch_url_and_convert_to_markdown(
    urls: str | list[str],
    *,
    fetch_using: Literal["playwright", "selenium"] = "playwright",
    max_parallel: int = 1,
    proxy_config: ProxySettings | None = None,
    http_credentials: HttpCredentials | None = None,
    wait_type: ScraperWaitType = ScraperWaitType.SLEEP,
    wait_selector: str | None = None,
    headless: bool = True,
    ignore_ssl: bool = False,
    include_links: bool = True,
    include_images: bool = False,
    include_metadata: bool = False,
    tags: list[str] | None = None,
    meta: list[str] | None = None,
    sleep_time: int = 1,
    timeout: int = 10,
    verbose: bool = False,
    console: Console | None = None,
) -> list[str]:
    """
    Fetch the contents of a webpage and convert it to markdown.

    This is a thin facade over ``fetch_url`` followed by ``html_to_markdown``;
    the fetch-related parameters mirror ``fetch_url`` and are forwarded as-is
    (was ARC-021 — the facade previously dropped proxy/credentials/wait/headless/
    ignore_ssl/max_parallel).

    Args:
        urls (Union[str, list[str]]): The URL(s) to fetch.
        fetch_using (Literal["playwright", "selenium"], optional): The method to use for fetching the content. Defaults to "playwright".
        max_parallel (int): Maximum number of parallel requests. Defaults to 1.
        proxy_config (ProxySettings | None): Proxy configuration. Defaults to None.
        http_credentials (HttpCredentials | None): HTTP credentials for authentication. Defaults to None.
        wait_type (ScraperWaitType): The type of wait to use. Defaults to ScraperWaitType.SLEEP.
        wait_selector (str | None): CSS selector / text to wait for. Defaults to None.
        headless (bool): Whether to run the browser headless. Defaults to True.
        ignore_ssl (bool): Whether to ignore SSL errors. Defaults to False.
        include_links (bool, optional): Whether to include links in the markdown. Defaults to True.
        include_images (bool, optional): Whether to include images in the markdown. Defaults to False.
        include_metadata (bool, optional): Whether to include a metadata section in the markdown. Defaults to False.
        tags (list[str], optional): A list of tags to include in the markdown metadata. Defaults to None.
        meta (list[str], optional): A list of metadata attributes to include in the markdown. Defaults to None.
        sleep_time (int, optional): The number of seconds to sleep between requests. Defaults to 1.
        timeout (int, optional): The timeout in seconds for the request. Defaults to 10.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        console (Console, optional): The console to use for printing verbose output.

    Returns:
        list[str]: The converted markdown content as a list of strings.
    """
    if not console:
        console = console_err

    if tags is None:
        tags = []
    if meta is None:
        meta = []

    if isinstance(urls, str):
        urls = [urls]
    pages = fetch_url(
        urls,
        fetch_using=fetch_using,
        max_parallel=max_parallel,
        sleep_time=sleep_time,
        timeout=timeout,
        proxy_config=proxy_config,
        http_credentials=http_credentials,
        wait_type=wait_type,
        wait_selector=wait_selector,
        headless=headless,
        verbose=verbose,
        ignore_ssl=ignore_ssl,
        console=console,
    )
    sources = list(zip(urls, pages, strict=False))
    if verbose:
        console.print("[bold green]Converting fetched content to markdown...[/bold green]")
    results: list[str] = []
    for url, page_html in sources:
        markdown = html_to_markdown(
            page_html,
            url=url,
            include_links=include_links,
            include_images=include_images,
            include_metadata=include_metadata,
            tags=tags,
            meta=meta,
        )
        results.append(markdown)
    if verbose:
        console.print("[bold green]Conversion to markdown complete.[/bold green]")
    return results
