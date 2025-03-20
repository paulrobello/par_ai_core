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
import os
import threading
import time
from queue import Queue
from typing import Literal
from urllib.parse import quote, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.async_api import HttpCredentials, ProxySettings, async_playwright
from pydantic import BaseModel
from rich.console import Console
from rich.repr import rich_repr
from strenum import StrEnum

from par_ai_core.par_logging import console_err
from par_ai_core.user_agents import get_random_user_agent


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

    # print(f"No element ${element} found.")
    return ""


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
    ignore_ssl: bool = True,
    console: Console | None = None,
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
        wait_type (WaitType, optional): The type of wait to use. Defaults to WaitType.IDLE.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless (bool): Whether to run the browser in headless mode.
        verbose (bool): Whether to print verbose output.
        ignore_ssl (bool): Whether to ignore SSL errors.
        console (Console | None): The console to use for output. Defaults to console_err.

    Returns:
        list[str]: A list of HTML contents of the fetched webpages.
    """
    if isinstance(urls, str):
        urls = [urls]
    if not all(urlparse(url).scheme for url in urls):
        raise ValueError("All URLs must be absolute URLs with a scheme (e.g. http:// or https://)")
    try:
        if fetch_using == "playwright":
            try:
                loop = asyncio.get_running_loop()
            except Exception as _:
                loop = None
            if loop:
                return loop.run_until_complete(
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
            else:
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
    ignore_ssl: bool = True,
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
        wait_type (WaitType, optional): The type of wait to use. Defaults to WaitType.SLEEP.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless (bool): Whether to run the browser in headless mode.
        ignore_ssl (bool): Whether to ignore SSL errors.
        verbose (bool): Whether to print verbose output.
        console (Console | None): The console to use for output. Defaults to console_err.

    Returns:
        list[str]: A list of HTML contents of the fetched webpages.
    """
    # from playwright.async_api import async_playwright

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
                input()
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
                        console.print(f"[yellow]Waiting for selector {wait_selector}...[/yellow]")
                    await page.locator("body").wait_for_text(wait_selector, timeout=timeout * 1000)
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
    ignore_ssl: bool = True,
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
        wait_type (WaitType, optional): The type of wait to use. Defaults to WaitType.SLEEP.
        wait_selector (str, optional): The CSS selector to wait for. Defaults to None.
        headless: Whether to run the browser in headless mode
        ignore_ssl: Whether to ignore SSL errors
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
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(timeout)

            while not queue.empty():
                index, url = queue.get()
                try:
                    if verbose:
                        console.print(f"[bold blue]Selenium fetching content from {url}...[/bold blue]")

                    if http_credentials and "username" in http_credentials and "password" in http_credentials:
                        url = inject_credentials(url, http_credentials["username"], http_credentials["password"])

                    driver.get(url)
                    if wait_type == ScraperWaitType.PAUSE:
                        console.print("[yellow]Press Enter to continue...[/yellow]")
                        input()
                    elif wait_type == ScraperWaitType.SLEEP and sleep_time > 0:
                        pass
                        # time.sleep(sleep_time)
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
                    time.sleep(1)  # Wait a bit for any dynamic content to load
                    if verbose:
                        console.print(
                            "[bold green]Page loaded. Scrolling and waiting for dynamic content...[/bold green]"
                        )
                        console.print(f"[bold yellow]Sleeping for {sleep_time} seconds...[/bold yellow]")
                    # Scroll to the bottom of the page
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                    time.sleep(sleep_time)  # Sleep for the specified time

                    results[index] = driver.page_source
                except Exception as e:
                    if verbose:
                        console.print(f"[bold red]Error fetching content from {url}: {str(e)}[/bold red]")
                    results[index] = ""
                finally:
                    queue.task_done()
        except Exception as _:
            if verbose:
                console.print("[bold red]Error initializing Selenium driver[/bold red]")
            while not queue.empty():
                queue.get()
                queue.task_done()
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as _:
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
    import html2text

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
                if tag.has_attr(attribute):  # type: ignore
                    attr_value = tag[attribute]  # type: ignore
                    if attr_value.startswith("//"):  # type: ignore
                        tag[attribute] = f"https:{attr_value}"  # type: ignore
                    elif url and not attr_value.startswith(("http://", "https://")):  # type: ignore
                        tag[attribute] = urljoin(url, attr_value)  # type: ignore

    metadata = {
        "source": url,
        "title": title or "",
        "tags": (" ".join(tags)).strip(),
    }
    for m in soup.find_all("meta"):
        n = m.get("name", "").strip()  # type: ignore
        if not n:
            continue
        v = m.get("content", "").strip()  # type: ignore
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
    for element in soup.find_all(attrs={"role": "separator"}):
        hr = soup.new_tag("hr")
        element.replace_with(hr)
        # Add extra newlines around hr to ensure proper markdown rendering
        hr.insert_before(soup.new_string("\n"))
        hr.insert_after(soup.new_string("\n"))

    html_content = str(soup)

    ### code blocks
    html_content = html_content.replace("<pre", "```<pre")
    html_content = html_content.replace("</pre>", "</pre>```")

    ### convert to markdown
    converter = html2text.HTML2Text()
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

    Args:
        urls (Union[str, list[str]]): The URL(s) to fetch.
        fetch_using (Literal["playwright", "selenium"], optional): The method to use for fetching the content. Defaults to "playwright".
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
    import html2text

    if not console:
        console = console_err

    if tags is None:
        tags = []
    if meta is None:
        meta = []

    if isinstance(urls, str):
        urls = [urls]
    pages = fetch_url(urls, fetch_using=fetch_using, sleep_time=sleep_time, timeout=timeout, verbose=verbose)
    sources = list(zip(urls, pages))
    if verbose:
        console.print("[bold green]Converting fetched content to markdown...[/bold green]")
    results: list[str] = []
    for url, html_content in sources:
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
                    if tag.has_attr(attribute):  # type: ignore
                        attr_value = tag[attribute]  # type: ignore
                        if attr_value.startswith("//"):  # type: ignore
                            tag[attribute] = f"https:{attr_value}"  # type: ignore
                        elif not attr_value.startswith(("http://", "https://")):  # type: ignore
                            tag[attribute] = urljoin(url, attr_value)  # type: ignore

        metadata = {
            "source": url,
            "title": title or "",
            "tags": (" ".join(tags)).strip(),
        }
        for m in soup.find_all("meta"):
            n = m.get("name", "").strip()  # type: ignore
            if not n:
                continue
            v = m.get("content", "").strip()  # type: ignore
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
        for element in soup.find_all(attrs={"role": "separator"}):
            hr = soup.new_tag("hr")
            element.replace_with(hr)
            # Add extra newlines around hr to ensure proper markdown rendering
            hr.insert_before(soup.new_string("\n"))
            hr.insert_after(soup.new_string("\n"))

        html_content = str(soup)

        ### code blocks
        html_content = html_content.replace("<pre", "```<pre")
        html_content = html_content.replace("</pre>", "</pre>```")

        ### convert to markdown
        converter = html2text.HTML2Text()
        converter.ignore_links = not include_links
        converter.ignore_images = not include_images
        markdown = converter.handle(html_content)

        if include_metadata:
            meta_markdown = "# Metadata\n\n"
            for k, v in metadata.items():
                meta_markdown += f"- {k}: {v}\n"
            markdown = meta_markdown + markdown
        results.append(markdown)
    if verbose:
        console.print("[bold green]Conversion to markdown complete.[/bold green]")
    return results
