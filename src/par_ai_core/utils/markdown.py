"""HTML-to-Markdown conversion utilities.

This module uses ``html2text`` as the single HTML→Markdown converter for the
package (ARC-013), the same converter backing ``par_ai_core.web_tools.html_to_markdown``.
The previous implementation used ``markdownify``; that dependency has been dropped.
"""

from __future__ import annotations

from typing import Any

import html2text
from bs4 import BeautifulSoup


def md(soup: BeautifulSoup, **options: Any) -> str:
    """Convert BeautifulSoup object to Markdown.

    Uses ``html2text`` so the package has a single HTML→Markdown pipeline
    (ARC-013). Previously this wrapped ``markdownify.MarkdownConverter``; that
    dependency has been removed in favor of the ``html2text`` converter already
    used by ``par_ai_core.web_tools.html_to_markdown``.

    Args:
        soup: The BeautifulSoup object to convert.
        **options: Accepted for backward compatibility but ignored. The previous
            markdownify-based implementation honored options such as ``strip``;
            html2text uses a different configuration model. For link/image/
            metadata control, call ``web_tools.html_to_markdown`` directly.

    Returns:
        str: The converted Markdown string.
    """
    converter = html2text.HTML2Text()
    return converter.handle(str(soup)).strip()
