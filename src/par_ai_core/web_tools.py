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

import os
import time
from typing import Literal
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel
from rich.console import Console
from rich.repr import rich_repr

from par_ai_core.par_logging import console_err
from par_ai_core.user_agents import get_random_user_agent


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
    sleep_time: int = 1,
    timeout: int = 10,
    verbose: bool = False,
    ignore_ssl: bool = True,
    console: Console | None = None,
) -> list[str]:
    """
    Fetch the contents of a webpage using either Playwright or Selenium.

    Args:
        urls (str | list[str]): The URL(s) to fetch.
        fetch_using (Literal["playwright", "selenium"]): The library to use for fetching the webpage.
        sleep_time (int): The number of seconds to sleep between requests.
        timeout (int): The number of seconds to wait for a response.
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
            return fetch_url_playwright(
                urls, sleep_time=sleep_time, timeout=timeout, verbose=verbose, ignore_ssl=ignore_ssl, console=console
            )
        return fetch_url_selenium(
            urls, sleep_time=sleep_time, timeout=timeout, verbose=verbose, ignore_ssl=ignore_ssl, console=console
        )
    except Exception as e:
        if verbose:
            if not console:
                console = console_err
            console.print(f"[bold red]Error fetching URL: {str(e)}[/bold red]")
        return [""] * (len(urls) if isinstance(urls, list) else 1)


def fetch_url_selenium(
    urls: str | list[str],
    *,
    sleep_time: int = 1,
    timeout: int = 10,
    ignore_ssl: bool = True,
    verbose: bool = False,
    console: Console | None = None,
) -> list[str]:
    """Fetch the contents of a webpage using Selenium.

    Args:
        urls: The URL(s) to fetch
        sleep_time: The number of seconds to sleep between requests
        timeout: The number of seconds to wait for a response
        ignore_ssl: Whether to ignore SSL errors
        verbose: Whether to print verbose output
        console: The console to use for printing verbose output

    Returns:
        A list of HTML contents of the fetched webpages
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
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
    options.add_argument("--window-position=-2400,-2400")
    options.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(timeout)

    results: list[str] = []
    for url in urls:
        if verbose:
            console.print(f"[bold blue]Selenium fetching content from {url}...[/bold blue]")
        try:
            driver.get(url)
            if verbose:
                console.print("[bold green]Page loaded. Scrolling and waiting for dynamic content...[/bold green]")
                console.print(f"[bold yellow]Sleeping for {sleep_time} seconds...[/bold yellow]")
            time.sleep(sleep_time)  # Sleep for the specified time
            # Scroll to the bottom of the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)  # Wait a bit for any dynamic content to load
            results.append(driver.page_source)
        except Exception as e:
            if verbose:
                console.print(f"[bold red]Error fetching content from {url}: {str(e)}[/bold red]")
            results.append("")
    try:
        driver.quit()
    except Exception as _:
        pass

    return results


def fetch_url_playwright(
    urls: str | list[str],
    *,
    sleep_time: int = 1,
    timeout: int = 10,
    ignore_ssl: bool = True,
    verbose: bool = False,
    console: Console | None = None,
) -> list[str]:
    """
    Fetch HTML content from a URL using Playwright.

    Args:
        urls (Union[str, list[str]]): The URL(s) to fetch.
        sleep_time (int, optional): The number of seconds to sleep between requests. Defaults to 1.
        timeout (int, optional): The timeout in seconds for the request. Defaults to 10.
        ignore_ssl (bool, optional): Whether to ignore SSL errors. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        console (Console, optional): The console to use for printing verbose output.

    Returns:
        list[str]: The fetched HTML content as a list of strings.
    """
    from playwright.sync_api import sync_playwright

    if not console:
        console = console_err

    if isinstance(urls, str):
        urls = [urls]

    results: list[str] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            console.print(
                "[bold red]Error launching playwright browser:[/bold red] Make sure you install playwright: `uv tool install playwright` then run `playwright install chromium`."
            )
            raise e
            # return ["" * len(urls)]
        context = browser.new_context(
            viewport={"width": 1280, "height": 1024}, user_agent=get_random_user_agent(), ignore_https_errors=ignore_ssl
        )

        page = context.new_page()
        for url in urls:
            if verbose:
                console.print(f"[bold blue]Playwright fetching content from {url}...[/bold blue]")
            try:
                page.goto(url, timeout=timeout * 1000)

                # Add delays to mimic human behavior
                if sleep_time > 0:
                    page.wait_for_timeout(sleep_time * 1000)  # Use the specified sleep time

                # Add more realistic actions like scrolling
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)  # Simulate time taken to scroll and read
                html = page.content()
                results.append(html)
                # if verbose:
                #     console.print(
                #         Panel(
                #             html[0:500] + "...",
                #             title="[bold green]Snippet[/bold green]",
                #         )
                #     )
            except Exception as e:
                if verbose:
                    console.print(f"[bold red]Error fetching content from {url}[/bold red]: {str(e)}")
                results.append("")
        try:
            browser.close()
        except Exception as _:
            pass

    return results


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

    if not tags:
        tags = []
    if not meta:
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
                    if tag.has_attr(attribute):
                        attr_value = tag[attribute]
                        if attr_value.startswith("//"):
                            tag[attribute] = f"https:{attr_value}"
                        elif not attr_value.startswith(("http://", "https://")):
                            tag[attribute] = urljoin(url, attr_value)

        metadata = {
            "source": url,
            "title": title or "",
            "tags": (" ".join(tags)).strip(),
        }
        for m in soup.find_all("meta"):
            n = m.get("name", "").strip()
            if not n:
                continue
            v = m.get("content", "").strip()
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
