"""Utilities for performing web searches across various platforms.

This module provides a set of functions to search different web platforms including
Tavily, Jina, Brave Search, Google Serper, Reddit, and YouTube. Each search function
returns results in a standardized format, making it easy to integrate and compare
results from multiple sources.

Features:
- Standardized result format across all search functions
- Support for various search parameters like date range and result limit
- Optional content scraping for more detailed results
- Special handling for Reddit and YouTube searches, including comment retrieval
- YouTube transcript fetching and summarization capabilities

Typical usage examples:

1. Perform a Tavily search:
    results = tavily_search("artificial intelligence", days=7, max_results=5)

2. Search using Brave Search with content scraping:
    results = brave_search("machine learning", days=30, max_results=3, scrape=True)

3. Search Reddit for recent posts:
    results = reddit_search("python tips", subreddit="learnpython", max_comments=5, max_results=3)

4. Search YouTube with transcript fetching:
    from par_ai_core.llm_config import LlmConfig
    from par_ai_core.llm_providers import LlmProvider, provider_light_models
    llm_config = LlmConfig(provider=LlmProvider.OPENAI, model_name=provider_light_models[LlmProvider.OPENAI])
    llm = llm_config.build_chat_model()
    results = youtube_search("OpenAI GPT-4", fetch_transcript=True, summarize_llm=llm)

5. Perform a Google search using Serper:
    results = serper_search("climate change", days=7, max_results=5)

Each search function returns a list of :class:`SearchResult` objects, where each result
has standardized fields: ``title``, ``url``, ``content``, and ``raw_content``.
``SearchResult`` is dict-compatible, so ``result["title"]`` and ``dict(result)`` still
work for callers that have not migrated to attribute access.

Note: Proper API keys and environment variables must be set up for each search
service before use. Refer to the individual function docstrings for specific
requirements and usage details.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from typing import Any, Literal
from urllib.parse import quote

import orjson as json
import requests
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, SecretStr
from rich.console import Console

from par_ai_core.par_logging import console_err

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Standardized search result across all search engines.

    Dict-compatible: supports ``result["title"]``, ``result.get("url")``,
    ``"title" in result``, ``dict(result)``, and iteration over field names,
    so callers using dict-style access continue to work after the migration
    from ``list[dict]`` to ``list[SearchResult]`` return types.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    content: str
    raw_content: str = ""
    score: float | None = None

    def __getitem__(self, key: str) -> Any:
        if key in type(self).model_fields:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in type(self).model_fields

    def get(self, key: str, default: Any = None) -> Any:
        if key in type(self).model_fields:
            return getattr(self, key)
        return default

    def keys(self) -> tuple[str, ...]:
        return tuple(type(self).model_fields)

    def values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, k) for k in type(self).model_fields)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((k, getattr(self, k)) for k in type(self).model_fields)

    def __iter__(self) -> Iterator[str]:
        return iter(type(self).model_fields)


def web_search(
    query: str, *, num_results: int = 3, verbose: bool = False, console: Console | None = None
) -> list[SearchResult]:
    """Perform a Google web search using the Google Custom Search API.

    Args:
        query: The search query to execute
        num_results: Maximum number of results to return. Defaults to 3.
        verbose: Whether to print verbose output. Defaults to False.
        console: Console to use for output. Defaults to console_err.

    Returns:
        list[SearchResult]: List of search results.

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
    return [
        SearchResult(title=result["title"], url=result["link"], content=result.get("snippet", ""))
        for result in search.results(query, num_results=num_results)
    ]


def tavily_search(
    query: str,
    *,
    include_raw_content: bool = True,
    topic: Literal["general", "news"] = "general",
    days: int = 3,
    max_results: int = 3,
) -> list[SearchResult]:
    """Search the web using the Tavily API.

    Performs a web search using Tavily's AI-powered search engine. Can search for
    either general web content or recent news articles.

    Args:
        query (str): The search query to execute.
        include_raw_content (bool, optional): Whether to include raw content from Tavily.
            Defaults to True.
        topic (Literal["general", "news"], optional): Topic of search, either "general"
            or "news". Defaults to "general".
        days (int, optional): Number of days to search back when topic is "news".
            Defaults to 3.
        max_results (int, optional): Maximum number of results to return.
            Defaults to 3.

    Returns:
        list[SearchResult]: List of search results, where each result contains:
            title (str): Title of the search result
            url (str): URL of the search result
            content (str): Snippet/summary of the content
            raw_content (str): Full content if available and requested
            score (float | None): Relevance score if provided by Tavily

    Raises:
        TavilyError: If the Tavily API request fails or returns an error response.
    """
    try:
        from tavily import TavilyClient
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("Tavily search requires: pip install 'par_ai_core[search]'") from e
    tavily_client = TavilyClient()
    raw_results = tavily_client.search(
        query, max_results=max_results, topic=topic, days=days, include_raw_content=include_raw_content
    )["results"]
    return [SearchResult(**r) for r in raw_results]


def jina_search(query: str, *, max_results: int = 3) -> list[SearchResult]:
    """Search the web using the Jina AI search API.

    Performs a web search using Jina's neural search engine that combines
    traditional search with AI-powered relevance ranking.

    Args:
        query (str): The search query to execute.
        max_results (int): Maximum number of results to return.

    Returns:
        list[SearchResult]: List of search results, where each result contains:
            title (str): Title of the search result
            url (str): URL of the search result
            content (str): Snippet/summary of the content
            raw_content (str): Full content of the page if available

    Raises:
        Exception: If the Jina API request fails or returns an error status code.
    """
    jina_api_key = os.environ.get("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY environment variable is not set")

    response = requests.get(
        f"https://s.jina.ai/{quote(query)}",
        headers={
            "Authorization": f"Bearer {jina_api_key}",
            "X-Retain-Images": "none",
            "Accept": "application/json",
        },
        timeout=10,
    )

    if response.status_code == 200:
        res = response.json()
        return [
            SearchResult(title=r["title"], url=r["url"], content=r["description"], raw_content=r["content"])
            for r in res["data"][:max_results]
            if "warning" not in r
        ]

    response.raise_for_status()
    return []


def brave_search(query: str, *, days: int = 0, max_results: int = 3, scrape: bool = False) -> list[SearchResult]:
    """Search the web using the Brave Search API.

    Performs a web search using Brave's privacy-focused search engine. Can optionally
    scrape the content of returned URLs for more detailed results.

    Args:
        query (str): The search query to execute.
        days (int, optional): Number of days to search back. Must be >= 0.
            Defaults to 0 meaning all time.
        max_results (int, optional): Maximum number of results to return.
            Defaults to 3.
        scrape (bool, optional): Whether to scrape the content of the search result
            URLs. Defaults to False. When True, each result URL is subject to the
            public-http(s) SSRF guard enforced by ``fetch_url``; non-public URLs are rejected.

    Returns:
        list[SearchResult]: List of search results, where each result contains:
            title (str): Title of the search result
            url (str): URL of the search result
            content (str): Snippet/summary of the content
            raw_content (str): Full content of the page if scraped

    Raises:
        ValueError: If days parameter is negative.
    """
    if days < 0:
        raise ValueError("days parameter must be >= 0")
    if days > 0:
        start_date = date.today() - timedelta(days=days)
        end_date = date.today()
        date_range = f"{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"
    else:
        date_range = "false"
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        raise ValueError("BRAVE_API_KEY environment variable is not set")
    try:
        from langchain_community.utilities.brave_search import BraveSearchWrapper
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("Brave search requires: pip install 'par_ai_core[search]'") from e
    wrapper = BraveSearchWrapper(
        api_key=SecretStr(brave_api_key),
        search_kwargs={"count": max_results, "summary": True, "freshness": date_range},
    )
    res = json.loads(wrapper.run(query))
    if scrape:
        from par_ai_core.web_tools import fetch_url_and_convert_to_markdown

        urls = [r["link"] for r in res[:max_results]]
        content = fetch_url_and_convert_to_markdown(urls)
        for r, c in zip(res, content, strict=False):
            r["raw_content"] = c
    return [
        SearchResult(
            title=r["title"],
            url=r["link"],
            content=r["snippet"],
            raw_content=r.get("raw_content", r["snippet"]),
        )
        for r in res[:max_results]
    ]


def serper_search(
    query: str,
    *,
    search_type: Literal["news", "search", "places", "images"] = "search",
    days: int = 0,
    max_results: int = 3,
    scrape: bool = False,
    include_images: bool = False,
) -> list[SearchResult]:
    """Search the web using Google Serper.

    Args:
        query (str): The search query to execute.
        search_type (Literal["news", "search", "places", "images"], optional): Type of search.
            Defaults to "search".
        days (int, optional): Number of days to search back. Must be >= 0. Defaults to 0 meaning all
            time. Serper/Google only buckets freshness, so this maps to the nearest ``tbs`` bucket:
            1 -> past day, 2-7 -> past week, 8-30 -> past month, >30 -> past year.
        max_results (int, optional): Maximum number of results to return. Defaults to 3.
        scrape (bool, optional): Whether to scrape the search result urls. Defaults to False.
            When True, each result URL is subject to the public-http(s) SSRF guard enforced
            by ``fetch_url``; non-public URLs are rejected.

    Returns:
        list[SearchResult]: List of search results.

    Raises:
        ValueError: If days is negative.
    """
    if days < 0:
        raise ValueError("days parameter must be >= 0")
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("Serper (Google) search requires: pip install 'par_ai_core[search]'") from e
    # ``days`` was previously validated then ignored. Serper/Google don't accept
    # an arbitrary day count, so map it to the closest freshness bucket (QA-010).
    tbs: str | None
    if days == 0:
        tbs = None
    elif days == 1:
        tbs = "qdr:d"
    elif days <= 7:
        tbs = "qdr:w"
    elif days <= 30:
        tbs = "qdr:m"
    else:
        tbs = "qdr:y"
    search = GoogleSerperAPIWrapper(type=search_type, tbs=tbs)
    res = search.results(query)
    results_list = res.get(search_type, [])[:max_results]

    if scrape:
        from par_ai_core.web_tools import fetch_url_and_convert_to_markdown

        urls = [r["link"] for r in results_list]
        content = fetch_url_and_convert_to_markdown(urls, include_images=include_images)
        for r, c in zip(results_list, content, strict=False):
            r["raw_content"] = c

    return [
        SearchResult(
            title=r["title"],
            url=r["link"],
            content=r.get("snippet", r.get("section")) or "",
            raw_content=r.get("raw_content") or "",
        )
        for r in results_list
    ]


def reddit_search(
    query: str, subreddit: str = "all", max_comments: int = 0, max_results: int = 3
) -> list[SearchResult]:
    """Search Reddit for posts and comments.

    Searches Reddit for posts matching a query, optionally within a specific subreddit.
    Special query words 'hot', 'new', or 'controversial' can be used to fetch posts
    sorted by those criteria instead of performing a text search.

    Args:
        query (str): The search query. Special values: 'hot', 'new', 'controversial'
            will fetch posts sorted by those criteria instead of searching.
        subreddit (str, optional): The subreddit to search. Defaults to 'all'.
        max_comments (int, optional): Maximum number of comments to return per post.
            Defaults to 0 (no comments).
        max_results (int, optional): Maximum number of posts to return.
            Defaults to 3.

    Returns:
        list[SearchResult]: List of search results, where each result contains:
            title (str): Title of the Reddit post
            url (str): URL of the post
            content (str): Post text content
            raw_content (str): Formatted post content including metadata and comments

    Note:
        If the specified subreddit is not found, falls back to searching 'all'.
    """
    try:
        import praw
        import praw.models
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("Reddit search requires: pip install 'par_ai_core[search]'") from e
    reddit = praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        username=os.environ.get("REDDIT_USERNAME"),
        password=os.environ.get("REDDIT_PASSWORD"),
        user_agent="parai",
    )
    try:
        sub_reddit = reddit.subreddit(subreddit)
    except Exception:
        subreddit = "all"
        sub_reddit = reddit.subreddit(subreddit)
    if query == "hot":
        sub_reddit = sub_reddit.hot(limit=max_results)
    elif query == "new":
        sub_reddit = sub_reddit.new(limit=max_results)
    elif query == "controversial":
        sub_reddit = sub_reddit.controversial(limit=max_results)
    else:
        sub_reddit = sub_reddit.search(query, limit=max_results)
    results: list[SearchResult] = []
    for sub in sub_reddit:
        comments_res = []

        if max_comments > 0:
            sub.comments.replace_more(limit=3)
            for comment in sub.comments.list():
                if isinstance(comment, praw.models.MoreComments):
                    continue
                if not comment.author:  # skip deleted comments
                    continue
                comments_res.append(
                    f"* Author: {comment.author.name if comment.author else 'Unknown'} Score: {comment.score} Content: {comment.body}"
                )
                if len(comments_res) >= max_comments:
                    break

        raw_content = [
            "# " + sub.title,
            "*Author*: " + (sub.author.name if sub.author else "Unknown"),
            "*Score*: " + str(sub.score),
            "*URL*: " + sub.url,
            "*Content*: ",
            sub.selftext,
            "*Comments*: ",
            "\n".join(comments_res),
        ]
        rec = SearchResult(title=sub.title, url=sub.url, content=sub.selftext, raw_content="\n".join(raw_content))
        results.append(rec)
    return results


def youtube_get_video_id(url: str) -> str | None:
    """Extract the video ID from a YouTube URL.

    Supports various YouTube URL formats including standard watch URLs,
    shortened youtu.be URLs, and embed URLs.

    Args:
        url (str): The YouTube URL to parse.

    Returns:
        str | None: The 11-character video ID if found, None if no match.
    """
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"  # pylint: disable=line-too-long
    match = re.search(pattern, url)
    return match.group(1) if match else None


def youtube_get_comments(youtube, video_id: str, max_results: int = 10) -> list[str]:
    """Fetch comments for a YouTube video."""
    comments = []

    # Fetch top-level comments
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        textFormat="plainText",
        maxResults=max_results,  # Adjust based on needs
    )

    while request:
        try:
            response = request.execute()

            for item in response["items"]:
                # Top-level comment
                top_level_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(top_level_comment)

                # Check if there are replies in the thread
                if "replies" in item and "comments" in item["replies"]:
                    for reply in item["replies"]["comments"][:max_results]:
                        reply_text = reply["snippet"]["textDisplay"]
                        # Add incremental spacing and a dash for replies
                        comments.append("    - " + reply_text)

            # Prepare the next page of comments, if available
            if "nextPageToken" in response:
                request = youtube.commentThreads().list_next(previous_request=request, previous_response=response)
            else:
                request = None
        except Exception as e:
            # Was a bare ``except: break`` that silently truncated comment
            # fetching on any API error; log so the truncation is visible (QA-006).
            logger.warning("YouTube comment fetch stopped early for video %s: %s", video_id, e)
            break

    return comments


def youtube_get_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Fetch transcript for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("YouTube transcript fetch requires: pip install 'par_ai_core[search]'") from e
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages or ["en"])  # type: ignore[reportAttributeAccessIssue]
    transcript_text = " ".join([item["text"] for item in transcript_list])
    return transcript_text.replace("\n", " ")


def youtube_search(
    query: str,
    *,
    days: int = 0,
    max_comments: int = 0,
    max_results: int = 3,
    fetch_transcript: bool = False,
    summarize_llm: BaseChatModel | None = None,
    summarizer: Callable[[str], str] | None = None,
) -> list[SearchResult]:
    """Search YouTube for videos with optional transcript and comment fetching.

    Performs a YouTube search and can optionally fetch video transcripts,
    comments, and generate transcript summaries.

    Args:
        query (str): The search query to execute.
        days (int, optional): Number of days to search back. Must be >= 0.
            Defaults to 0 meaning all time.
        max_comments (int, optional): Maximum number of comments to fetch per video.
            Defaults to 0 meaning no comments.
        max_results (int, optional): Maximum number of results to return.
            Defaults to 3.
        fetch_transcript (bool, optional): Whether to fetch video transcripts.
            Defaults to False.
        summarize_llm (BaseChatModel | None, optional): LLM to use for summarizing
            transcripts (uses ``llm_utils.summarize_content`` internally). Defaults
            to None meaning no summarization. Deprecated in favor of ``summarizer``;
            this parameter couples the call to the LLM config stack.
        summarizer (Callable[[str], str] | None, optional): Callable that takes the
            transcript text and returns a summary. When provided, takes precedence
            over ``summarize_llm`` and avoids importing the LLM config stack entirely.
            Defaults to None meaning no summarization.

    Returns:
        list[SearchResult]: List of search results, where each result contains:
            title (str): Title of the video
            url (str): URL of the video
            content (str): Description, metadata, and optionally comments and
                transcript summary
            raw_content (str): Full transcript text if fetched

    Raises:
        ValueError: If days parameter is negative.
        googleapiclient.errors.HttpError: If the YouTube API request fails.
    """
    if days < 0:
        raise ValueError("days parameter must be >= 0")
    api_key = os.environ.get("GOOGLE_API_KEY")
    try:
        from googleapiclient.discovery import build
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("YouTube search requires: pip install 'par_ai_core[search]'") from e
    youtube = build("youtube", "v3", developerKey=api_key)

    start_date = date.today() - timedelta(days=days)

    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        videoCaption="closedCaption" if fetch_transcript else "any",
        # order="date", # broken
        publishedAfter=start_date.strftime("%Y-%m-%dT%H:%M:%SZ") if days > 0 else None,
    )
    response = request.execute()

    results: list[SearchResult] = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        video_title = item["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        content = (
            f"PublishTime: {item['snippet']['publishedAt']}\n"
            + f"ChannelId: {item['snippet']['channelId']}\n"
            + f"Description: {item['snippet']['description']}"
        )
        if max_comments > 0:
            # Pass max_comments through so the requested count is actually
            # fetched (was hardcoded to the default 10) (QA-011).
            comments = youtube_get_comments(youtube, video_id, max_results=max_comments)
        else:
            comments = []

        if comments:
            content += "\n\nComments:\n" + "\n".join(comments)

        if fetch_transcript:
            transcript_text = youtube_get_transcript(video_id, languages=["en"])
            transcript_summary = ""
            if transcript_text and summarizer is not None:
                transcript_summary = summarizer(transcript_text)
                content += "\n\nTranscript Summary:\n" + transcript_summary
            elif transcript_text and summarize_llm is not None:
                from par_ai_core.llm_utils import summarize_content

                transcript_summary = summarize_content(transcript_text, summarize_llm)
                content += "\n\nTranscript Summary:\n" + transcript_summary
            else:
                content += "\n\nTranscript:\n" + transcript_text
        else:
            transcript_text = ""
            transcript_summary = ""

        results.append(SearchResult(title=video_title, url=video_url, content=content, raw_content=transcript_text))

    return results
