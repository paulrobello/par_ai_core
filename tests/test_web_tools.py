"""Tests for web tools module."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from bs4 import BeautifulSoup
from rich.console import Console

from par_ai_core.web_tools import (
    GoogleSearchResult,
    fetch_url,
    fetch_url_and_convert_to_markdown,
    fetch_url_playwright,
    fetch_url_selenium,
    get_html_element,
    normalize_url,
    web_search,
)


def test_normalize_url():
    """Test URL normalization with various parameters."""
    test_cases = [
        # (input_url, strip_fragment, strip_query, strip_slash, expected)
        ("https://example.com/path/#frag", True, False, False, "https://example.com/path/"),
        ("https://example.com/path/?q=1", False, True, False, "https://example.com/path/"),
        ("https://example.com/path/", False, False, True, "https://example.com/path"),
        ("https://example.com/path//", False, False, True, "https://example.com/path//".rstrip("/")),
        ("https://example.com/path?q=1#frag", True, True, True, "https://example.com/path"),
        ("https://example.com/path/?q=1#frag", True, True, True, "https://example.com/path"),
        ("https://example.com:8080/path?q=1", False, False, False, "https://example.com:8080/path?q=1"),
        ("http://example.com/path/", False, False, False, "http://example.com/path/"),
    ]

    for url, strip_frag, strip_q, strip_slash, expected in test_cases:
        assert normalize_url(url, strip_frag, strip_q, strip_slash) == expected

    # Test default parameters (all stripping enabled)
    assert normalize_url("https://example.com/path/?q=1#frag") == "https://example.com/path"
    assert normalize_url("https://example.com//") == "https://example.com"


def test_google_search_result_model():
    """Test GoogleSearchResult model creation and validation."""
    result = GoogleSearchResult(title="Test Title", link="https://example.com", snippet="Test snippet text")
    assert result.title == "Test Title"
    assert result.link == "https://example.com"
    assert result.snippet == "Test snippet text"


def test_get_html_element():
    """Test extracting text from HTML elements."""
    html = """
    <html>
        <body>
            <h1>Test Header</h1>
            <div class="content">Test Content</div>
            <p>Test Paragraph</p>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")

    assert get_html_element("h1", soup) == "Test Header"
    assert get_html_element("div", soup) == "Test Content"
    assert get_html_element("p", soup) == "Test Paragraph"
    assert get_html_element("nonexistent", soup) == ""


def test_fetch_url_invalid_url():
    """Test fetch_url with invalid URLs."""
    with pytest.raises(ValueError, match="All URLs must be absolute URLs with a scheme"):
        fetch_url("example.com")


@pytest.mark.parametrize("fetch_using", ["playwright", "selenium"])
def test_fetch_url_parameters(fetch_using, monkeypatch):
    """Test fetch_url parameter validation."""
    url = "https://example.com"

    # Mock the browser functions to raise exceptions
    if fetch_using == "playwright":
        monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: exec('raise Exception("No playwright")'))
    else:
        monkeypatch.setattr("selenium.webdriver.Chrome", lambda *args, **kwargs: exec('raise Exception("No selenium")'))

    # Test that fetch_url handles exceptions by returning empty string
    result = fetch_url(url, fetch_using=fetch_using, sleep_time=1, timeout=5, verbose=True)
    assert result == [""]  # Should return list with empty string when error occurs


def test_fetch_url_and_convert_to_markdown_parameters(monkeypatch):
    """Test markdown conversion parameter validation."""
    url = "https://example.com"

    # Mock fetch_url to raise an exception
    def mock_fetch(*args, **kwargs):
        raise Exception("Mock fetch error")

    monkeypatch.setattr("par_ai_core.web_tools.fetch_url", mock_fetch)

    with pytest.raises(Exception):
        fetch_url_and_convert_to_markdown(
            url,
            include_links=True,
            include_images=True,
            include_metadata=True,
            tags=["test", "example"],
            meta=["description", "keywords"],
            verbose=True,
            console=Console(),
        )


@pytest.mark.asyncio
async def test_web_search_missing_env_vars(monkeypatch):
    """Test web search fails appropriately with missing env vars."""
    # Clear environment variables
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)

    with pytest.raises(ValueError, match=".*Missing required environment variables.*"):
        web_search("test query", num_results=1)


def test_web_search_default_console():
    """Test web_search uses console_err when no console provided."""
    with patch("langchain_google_community.GoogleSearchAPIWrapper") as mock_search:
        mock_search.return_value.results.return_value = [
            {"title": "Test", "link": "https://test.com", "snippet": "Test snippet"}
        ]
        with patch("par_ai_core.web_tools.console_err") as mock_console:
            # Set required env vars
            with patch.dict(
                "os.environ",
                {"GOOGLE_CSE_ID": "test_id", "GOOGLE_CSE_API_KEY": "test_key"},
            ):
                results = web_search("test query", verbose=True)

                # Verify console_err was used
                mock_console.print.assert_called_once_with("[bold green]Web search:[bold yellow] test query")

                # Verify search results
                assert len(results) == 1
                assert results[0].title == "Test"
                assert results[0].link == "https://test.com"
                assert results[0].snippet == "Test snippet"


def test_fetch_url_playwright_success(monkeypatch):
    """Test successful URL fetch using playwright."""
    mock_page = MagicMock()
    mock_page.content.return_value = "<html><body>Test content</body></html>"

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        result = fetch_url("https://example.com", fetch_using="playwright")
        assert result[0] == "<html><body>Test content</body></html>"


def test_fetch_url_playwright_direct_success(monkeypatch):
    """Test successful URL fetch using playwright."""
    mock_page = MagicMock()
    mock_page.content.return_value = "<html><body>Test content</body></html>"

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        result = fetch_url_playwright("https://example.com")
        assert result[0] == "<html><body>Test content</body></html>"


def test_fetch_url_selenium_success(monkeypatch):
    """Test successful URL fetch using selenium."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Test content</body></html>"

    with patch("selenium.webdriver.Chrome", return_value=mock_driver):
        # Test with URL as string
        result = fetch_url("https://example.com", fetch_using="selenium")
        assert result[0] == "<html><body>Test content</body></html>"
        assert isinstance(result, list)
        assert len(result) == 1

        # Test with URL as list
        result = fetch_url(["https://example.com"], fetch_using="selenium")
        assert result[0] == "<html><body>Test content</body></html>"
        assert isinstance(result, list)
        assert len(result) == 1


def test_fetch_url_selenium_verbose():
    """Test fetch_url_selenium with verbose output."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Test content</body></html>"
    mock_console = MagicMock()

    with patch("selenium.webdriver.Chrome", return_value=mock_driver):
        result = fetch_url("https://example.com", fetch_using="selenium", verbose=True, console=mock_console)

        # Verify console output messages
        mock_console.print.assert_any_call(
            "[bold blue]Selenium fetching content from https://example.com...[/bold blue]"
        )
        mock_console.print.assert_any_call(
            "[bold green]Page loaded. Scrolling and waiting for dynamic content...[/bold green]"
        )
        mock_console.print.assert_any_call("[bold yellow]Sleeping for 1 seconds...[/bold yellow]")

        # Verify the result
        assert result[0] == "<html><body>Test content</body></html>"
        assert isinstance(result, list)
        assert len(result) == 1


def test_fetch_url_list():
    """Test fetching multiple URLs."""
    urls = ["https://example1.com", "https://example2.com"]
    with patch("par_ai_core.web_tools.fetch_url_playwright", return_value=["<html>1</html>", "<html>2</html>"]):
        results = fetch_url(urls, fetch_using="playwright")
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)


def test_fetch_url_timeout():
    """Test URL fetch with timeout."""
    with patch("par_ai_core.web_tools.fetch_url_playwright", side_effect=TimeoutError("Timeout")):
        result = fetch_url("https://example.com", timeout=1)
        assert result == [""]


def test_fetch_url_selenium_options():
    """Test Selenium options configuration."""
    mock_options = MagicMock()
    mock_driver = MagicMock()

    with patch("selenium.webdriver.chrome.options.Options", return_value=mock_options) as mock_opts:
        with patch("selenium.webdriver.Chrome", return_value=mock_driver):
            fetch_url("https://example.com", fetch_using="selenium", ignore_ssl=True)

            # Verify options were set correctly
            assert any("--ignore-certificate-errors" in call[0][0] for call in mock_opts().add_argument.call_args_list)
            assert any("--headless=new" in call[0][0] for call in mock_opts().add_argument.call_args_list)


def test_fetch_url_and_convert_to_markdown_success():
    """Test successful markdown conversion."""
    test_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Header</h1>
            <p>Test paragraph</p>
            <a href="https://example.com">Test link</a>
            <img src="test.jpg" alt="Test image"/>
        </body>
    </html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown(
            "https://example.com",
            include_links=True,
            include_images=True,
            include_metadata=True,
            tags=["test"],
            meta=["description"],
        )

        assert "Test Header" in result[0]
        assert "Test paragraph" in result[0]
        assert "Test link" in result[0]
        assert "test.jpg" in result[0]
        assert "Metadata" in result[0]
        assert "source: https://example.com" in result[0]


def test_fetch_url_error_handling():
    """Test error handling in URL fetching."""
    mock_console = MagicMock()
    with patch("par_ai_core.web_tools.fetch_url_playwright", side_effect=Exception("Test error")):
        result = fetch_url("https://example.com", fetch_using="playwright", verbose=True, console=mock_console)
        assert result == [""]
        mock_console.print.assert_called_with("[bold red]Error fetching URL: Test error[/bold red]")


def test_fetch_url_and_convert_to_markdown_no_metadata():
    """Test markdown conversion without metadata."""
    test_html = "<html><body><h1>Test</h1></body></html>"

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com", include_metadata=False)
        assert "Metadata" not in result[0]
        assert "Test" in result[0]


def test_fetch_url_and_convert_to_markdown_no_links():
    """Test markdown conversion with links disabled."""
    test_html = """
    <html><body>
        <a href="https://example.com">Link</a>
        <p>Text</p>
    </body></html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com", include_links=False)
        assert "https://example.com" not in result[0]
        assert "Text" in result[0]


def test_html_to_markdown_special_elements():
    """Test conversion of special HTML elements to markdown."""
    test_html = """
    <html>
        <body>
            <hr class="separator"/>
            <pre><code>Test code</code></pre>
            <div class="content">Test content</div>
        </body>
    </html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com")
        assert "* * *" in result[0]  # Check for separator (html2text format)
        assert "```" in result[0]  # Check for code block
        assert "Test content" in result[0]


def test_playwright_browser_cleanup():
    """Test that playwright browser is cleaned up after use."""
    mock_browser = MagicMock()
    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        fetch_url("https://example.com", fetch_using="playwright")
        mock_browser.close.assert_called_once()


def test_playwright_launch_error():
    """Test handling of playwright launch errors."""
    with patch("playwright.sync_api.sync_playwright", side_effect=Exception("Failed to launch")):
        result = fetch_url("https://example.com", fetch_using="playwright", verbose=True)
        assert result == [""]


def test_fetch_url_playwright_verbose_error():
    """Test fetch_url_playwright error handling with verbose output."""
    mock_page = MagicMock()
    mock_page.goto.side_effect = Exception("Test playwright error")

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_console = MagicMock()

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        result = fetch_url("https://example.com", fetch_using="playwright", verbose=True, console=mock_console)

        # Verify error message was printed to console
        mock_console.print.assert_any_call(
            "[bold blue]Playwright fetching content from https://example.com...[/bold blue]"
        )
        mock_console.print.assert_any_call(
            "[bold red]Error fetching content from https://example.com[/bold red]: Test playwright error"
        )

        # Verify empty string is returned on error
        assert result == [""]


def test_fetch_url_playwright_wait_types():
    """Test fetch_url_playwright with different wait types."""
    mock_page = MagicMock()
    mock_page.content.return_value = "<html><body>Test content</body></html>"

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_console = MagicMock()

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        # Test SLEEP wait type
        result = fetch_url(
            "https://example.com",
            fetch_using="playwright",
            wait_type="sleep",
            sleep_time=2,
            verbose=True,
            console=mock_console,
        )
        # Verify the last call was with 2000ms (2 seconds)
        assert mock_page.wait_for_timeout.call_args_list[-1] == call(2000)
        assert result[0] == "<html><body>Test content</body></html>"

        # Test IDLE wait type
        result = fetch_url(
            "https://example.com",
            fetch_using="playwright",
            wait_type="idle",
            timeout=5,
            verbose=True,
            console=mock_console,
        )
        mock_page.wait_for_load_state.assert_called_with("networkidle", timeout=5000)
        assert result[0] == "<html><body>Test content</body></html>"


def test_fetch_url_selenium_wait_types():
    """Test fetch_url_selenium with different wait types."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Test content</body></html>"
    mock_console = MagicMock()

    with patch("selenium.webdriver.Chrome", return_value=mock_driver), \
         patch("time.sleep") as mock_sleep:

        # Test SLEEP wait type
        result = fetch_url(
            "https://example.com",
            fetch_using="selenium",
            wait_type="sleep",
            sleep_time=2,
            verbose=True,
            console=mock_console,
        )
        mock_sleep.assert_called_with(2)
        assert result[0] == "<html><body>Test content</body></html>"

        # Test IDLE wait type
        result = fetch_url(
            "https://example.com",
            fetch_using="selenium",
            wait_type="idle",
            timeout=5,
            verbose=True,
            console=mock_console,
        )
        # Selenium uses a simple sleep for IDLE
        mock_sleep.assert_called_with(1)
        assert result[0] == "<html><body>Test content</body></html>"


def test_fetch_url_playwright_verbose():
    """Test fetch_url_playwright with verbose output."""
    mock_page = MagicMock()
    mock_page.content.return_value = "<html><body>Test content</body></html>"

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_console = MagicMock()

    with patch("playwright.sync_api.sync_playwright", return_value=MagicMock(__enter__=lambda x: mock_playwright)):
        result = fetch_url("https://example.com", fetch_using="playwright", verbose=True, console=mock_console)

        # Verify console output messages
        mock_console.print.assert_any_call(
            "[bold blue]Playwright fetching content from https://example.com...[/bold blue]"
        )

        # Verify the result
        assert result[0] == "<html><body>Test content</body></html>"
        assert isinstance(result, list)
        assert len(result) == 1


def test_fetch_url_selenium_direct():
    """Test fetch_url_selenium with a string URL parameter."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Test content</body></html>"
    mock_console = MagicMock()

    with patch("selenium.webdriver.Chrome", return_value=mock_driver):
        result = fetch_url_selenium("https://example.com", verbose=True, console=mock_console)

        # Verify the URL was processed as a string
        mock_driver.get.assert_called_once_with("https://example.com")

        # Verify the result is a list with one item
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "<html><body>Test content</body></html>"


def test_fetch_url_selenium_browser_launch_error():
    """Test handling of selenium browser launch errors."""
    mock_console = MagicMock()

    with patch("selenium.webdriver.Chrome", side_effect=Exception("Failed to launch browser")):
        result = fetch_url("https://example.com", fetch_using="selenium", verbose=True, console=mock_console)

        # Verify error message was printed
        mock_console.print.assert_called_with("[bold red]Error fetching URL: Failed to launch browser[/bold red]")

        # Verify empty string is returned
        assert result == [""]


def test_fetch_url_selenium_verbose_error():
    """Test fetch_url_selenium error handling with verbose output."""
    mock_driver = MagicMock()
    mock_driver.get.side_effect = Exception("Test selenium error")
    mock_console = MagicMock()

    with patch("selenium.webdriver.Chrome", return_value=mock_driver):
        result = fetch_url("https://example.com", fetch_using="selenium", verbose=True, console=mock_console)

        # Verify error message was printed to console
        mock_console.print.assert_any_call(
            "[bold blue]Selenium fetching content from https://example.com...[/bold blue]"
        )
        mock_console.print.assert_any_call(
            "[bold red]Error fetching content from https://example.com: Test selenium error[/bold red]"
        )

        # Verify empty string is returned on error
        assert result == [""]


def test_markdown_element_removal():
    """Test removal of unwanted HTML elements during markdown conversion."""
    test_html = """
    <html>
        <head><script>alert('test')</script></head>
        <body>
            <header>Header content</header>
            <div>Main content</div>
            <footer>Footer content</footer>
            <iframe src="test.html"></iframe>
        </body>
    </html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com")
        assert "Header content" not in result[0]
        assert "Main content" in result[0]
        assert "Footer content" not in result[0]
        assert "test.html" not in result[0]


def test_relative_url_conversion():
    """Test conversion of relative URLs to absolute URLs."""
    test_html = """
    <html>
        <body>
            <a href="/relative/path">Link</a>
            <img src="//example.com/image.jpg" alt="Test Image">
            <a href="form.php">Form</a>
        </body>
    </html>
    """

    base_url = "https://example.com"
    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown(base_url, include_links=True, include_images=True)
        assert f"{base_url}/relative/path" in result[0]
        # html2text adds newlines around images, so we need to handle that
        assert "![Test\nImage](https://example.com/image.jpg)" in result[0]
        assert f"{base_url}/form.php" in result[0]


def test_fetch_url_and_convert_to_markdown_verbose():
    """Test fetch_url_and_convert_to_markdown with verbose output."""
    test_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Header</h1>
            <p>Test paragraph</p>
        </body>
    </html>
    """

    mock_console = MagicMock()

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com", verbose=True, console=mock_console)

        # Verify console output messages
        mock_console.print.assert_any_call("[bold green]Converting fetched content to markdown...[/bold green]")
        mock_console.print.assert_any_call("[bold green]Conversion to markdown complete.[/bold green]")

        # Verify the result contains converted content
        assert "Test Header" in result[0]
        assert "Test paragraph" in result[0]


def test_separator_elements_conversion():
    """Test conversion of separator elements to horizontal rules."""
    test_html = """
    <html>
        <body>
            <div>Before separator</div>
            <div role="separator"></div>
            <span role="separator" class="custom-separator"></span>
            <div>After separator</div>
        </body>
    </html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown("https://example.com")

        # html2text converts <hr> tags to "* * *"
        assert "Before separator" in result[0]
        assert "* * *" in result[0]
        # Should have two separators
        assert result[0].count("* * *") == 2
        assert "After separator" in result[0]


def test_metadata_extraction():
    """Test extraction of metadata from HTML."""
    test_html = """
    <html>
        <head>
            <title>Test Title</title>
            <meta name="description" content="Test Description">
            <meta name="keywords" content="test,keywords">
            <meta name="author" content="Test Author">
        </head>
        <body>
            <div>Content</div>
        </body>
    </html>
    """

    with patch("par_ai_core.web_tools.fetch_url", return_value=[test_html]):
        result = fetch_url_and_convert_to_markdown(
            "https://example.com", include_metadata=True, meta=["description", "keywords", "author"]
        )
        assert "Test Title" in result[0]
        assert "Test Description" in result[0]
        assert "test,keywords" in result[0]
        assert "Test Author" in result[0]
