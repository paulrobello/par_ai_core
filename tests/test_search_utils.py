"""Tests for search_utils module"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from par_ai_core.search_utils import (
    brave_search,
    jina_search,
    reddit_search,
    serper_search,
    tavily_search,
    youtube_get_comments,
    youtube_get_transcript,
    youtube_get_video_id,
    youtube_search,
)


def test_youtube_get_video_id():
    """Test extracting video IDs from various YouTube URL formats."""
    test_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("invalid_url", None),
    ]

    for url, expected in test_cases:
        assert youtube_get_video_id(url) == expected


@pytest.mark.parametrize("days", [-1, 0, 7])
def test_youtube_search_days_validation(days):
    """Test days parameter validation in youtube_search."""
    if days < 0:
        with pytest.raises(ValueError, match="days parameter must be >= 0"):
            youtube_search("test query", days=days)
    else:
        with patch("par_ai_core.search_utils.build") as mock_build:
            mock_youtube = MagicMock()
            mock_build.return_value = mock_youtube
            mock_youtube.search().list().execute.return_value = {"items": []}

            result = youtube_search("test query", days=days)
            assert isinstance(result, list)

            # Verify date filtering
            if days > 0:
                expected_date = date.today() - timedelta(days=days)
                # Get the actual call arguments
                assert mock_youtube.search.called
                assert mock_youtube.search().list.called
                call_kwargs = mock_youtube.search().list.call_args[1]
                assert "publishedAfter" in call_kwargs
                assert expected_date.strftime("%Y-%m-%dT%H:%M:%SZ") in call_kwargs["publishedAfter"]


@patch("par_ai_core.search_utils.TavilyClient")
def test_tavily_search(mock_tavily):
    """Test Tavily search functionality."""
    mock_results = [
        {
            "title": "Test Title",
            "url": "https://example.com",
            "content": "Test content",
        }
    ]
    mock_tavily.return_value.search.return_value = {"results": mock_results}

    results = tavily_search("test query", max_results=1)
    assert results == mock_results
    mock_tavily.return_value.search.assert_called_once_with(
        "test query",
        max_results=1,
        topic="general",
        days=3,
        include_raw_content=True,
    )


@patch("par_ai_core.search_utils.BraveSearchWrapper")
def test_brave_search(mock_brave):
    """Test Brave search functionality."""
    mock_results = [
        {
            "title": "Test Title",
            "link": "https://example.com",
            "snippet": "Test snippet",
        }
    ]
    mock_brave.return_value.run.return_value = json.dumps(mock_results)

    with patch.dict(os.environ, {"BRAVE_API_KEY": "test_key"}):
        # Test basic search
        results = brave_search("test query", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["content"] == "Test snippet"

        # Test with days parameter
        results = brave_search("test query", days=7, max_results=1)
        assert len(results) == 1

        # Test with scraping enabled
        with patch("par_ai_core.search_utils.fetch_url_and_convert_to_markdown") as mock_fetch:
            mock_fetch.return_value = ["Scraped content"]
            results = brave_search("test query", max_results=1, scrape=True)
            assert results[0]["raw_content"] == "Scraped content"

        # Test with negative days
        with pytest.raises(ValueError, match="days parameter must be >= 0"):
            brave_search("test query", days=-1)


@patch("par_ai_core.search_utils.GoogleSerperAPIWrapper")
def test_serper_search(mock_serper):
    """Test Google Serper search functionality."""
    # Test organic search
    mock_results = [
        {
            "title": "Test Title",
            "link": "https://example.com",
            "snippet": "Test snippet",
        }
    ]
    mock_serper.return_value.results.return_value = {"organic": mock_results}

    results = serper_search("test query", max_results=1)
    assert len(results) == 1
    assert results[0]["title"] == "Test Title"
    assert results[0]["url"] == "https://example.com"
    assert results[0]["content"] == "Test snippet"
    assert results[0]["raw_content"] == ""

    # Test news search
    mock_news_results = {
        "news": [
            {
                "title": "News Title",
                "link": "https://news.com",
                "snippet": "News snippet",
                "section": "News section",
            }
        ],
        "organic": [
            {
                "title": "News Title",
                "link": "https://news.com",
                "snippet": "News snippet",
                "section": "News section",
            }
        ],
    }
    mock_serper.return_value.results.return_value = mock_news_results

    results = serper_search("test query", days=7, max_results=1)
    assert len(results) == 1
    assert results[0]["title"] == "News Title"
    assert results[0]["content"] == "News snippet"

    # Test with scraping enabled
    with patch("par_ai_core.search_utils.fetch_url_and_convert_to_markdown") as mock_fetch:
        mock_fetch.return_value = ["Scraped content"]
        results = serper_search("test query", max_results=1, scrape=True)
        assert len(results) == 1
        assert results[0]["raw_content"] == "Scraped content"

    # Test with negative days
    with pytest.raises(ValueError, match="days parameter must be >= 0"):
        serper_search("test query", days=-1)


@patch("par_ai_core.search_utils.YouTubeTranscriptApi")
def test_youtube_get_transcript(mock_transcript_api):
    """Test fetching YouTube video transcripts."""
    # Mock transcript data
    mock_transcript = [{"text": "Hello"}, {"text": "World"}, {"text": "Test\nNewline"}]
    mock_transcript_api.get_transcript.return_value = mock_transcript

    result = youtube_get_transcript("test_video_id", languages=["en"])
    assert result == "Hello World Test Newline"
    mock_transcript_api.get_transcript.assert_called_once_with("test_video_id", languages=["en"])


@patch("par_ai_core.search_utils.build")
def test_youtube_get_comments(mock_build):
    """Test fetching YouTube video comments."""
    mock_youtube = MagicMock()
    mock_build.return_value = mock_youtube

    # Test basic comments
    mock_response = {
        "items": [
            {
                "snippet": {"topLevelComment": {"snippet": {"textDisplay": "Top comment"}}},
                "replies": {"comments": [{"snippet": {"textDisplay": "Reply comment"}}]},
            }
        ],
        "nextPageToken": None,
    }
    mock_youtube.commentThreads().list().execute.return_value = mock_response

    comments = youtube_get_comments(mock_youtube, "test_video_id")
    assert len(comments) == 2
    assert comments[0] == "Top comment"
    assert comments[1] == "    - Reply comment"

    # Test pagination
    mock_response_with_next = {
        "items": [
            {
                "snippet": {"topLevelComment": {"snippet": {"textDisplay": "Comment 1"}}},
                "replies": {},
            }
        ],
        "nextPageToken": "next_token",
    }
    mock_response_next_page = {
        "items": [
            {
                "snippet": {"topLevelComment": {"snippet": {"textDisplay": "Comment 2"}}},
                "replies": {},
            }
        ],
    }

    # Set up the mock for pagination
    mock_list = mock_youtube.commentThreads().list
    mock_list.return_value.execute.side_effect = [mock_response_with_next]
    mock_youtube.commentThreads().list_next.return_value.execute.side_effect = [mock_response_next_page]
    comments = youtube_get_comments(mock_youtube, "test_video_id")
    assert len(comments) == 2
    assert comments[0] == "Comment 1"
    assert comments[1] == "Comment 2"

    # Test error handling
    mock_youtube.commentThreads().list().execute.side_effect = Exception("API Error")
    comments = youtube_get_comments(mock_youtube, "test_video_id")
    assert comments == []


@patch("par_ai_core.search_utils.requests.get")
def test_jina_search(mock_get):
    """Test Jina search functionality."""
    mock_response = {
        "data": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "description": "Test description",
                "content": "Test content",
            }
        ]
    }
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    with patch.dict(os.environ, {"JINA_API_KEY": "test_key"}):
        results = jina_search("test query", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["content"] == "Test description"
        assert results[0]["raw_content"] == "Test content"

        # Verify API call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "https://s.jina.ai/" in args[0]
        assert kwargs["headers"]["Authorization"] == "Bearer test_key"


@patch("par_ai_core.search_utils.requests.get")
def test_jina_search_error(mock_get):
    """Test Jina search error handling."""
    mock_get.return_value.status_code = 400

    with (
        patch.dict(os.environ, {"JINA_API_KEY": "test_key"}),
        pytest.raises(Exception, match="Jina API request failed with status code 400"),
    ):
        jina_search("test query")


@patch("par_ai_core.search_utils.build")
def test_youtube_search_with_transcript_and_summary(mock_build):
    """Test YouTube search with transcript and summary functionality."""
    mock_youtube = MagicMock()
    mock_build.return_value = mock_youtube

    # Mock search response
    mock_response = {
        "items": [
            {
                "id": {"videoId": "test_id"},
                "snippet": {
                    "title": "Test Title",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "channelId": "test_channel",
                    "description": "Test description",
                },
            }
        ]
    }
    mock_youtube.search().list().execute.return_value = mock_response

    # Mock transcript
    with patch("par_ai_core.search_utils.youtube_get_transcript", return_value="Test transcript"):
        # Mock LLM for summary
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.name = "test_llm"
        mock_llm.invoke.return_value.content = "Summary of transcript"

        results = youtube_search(
            "test query",
            max_results=1,
            fetch_transcript=True,
            summarize_llm=mock_llm,
        )

        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert "Summary of transcript" in results[0]["content"]
        assert results[0]["raw_content"] == "Test transcript"


@patch("par_ai_core.search_utils.build")
def test_youtube_search_with_comments(mock_build):
    """Test YouTube search with comments functionality."""
    mock_youtube = MagicMock()
    mock_build.return_value = mock_youtube

    # Mock search response
    mock_response = {
        "items": [
            {
                "id": {"videoId": "test_id"},
                "snippet": {
                    "title": "Test Title",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "channelId": "test_channel",
                    "description": "Test description",
                },
            }
        ]
    }
    mock_youtube.search().list().execute.return_value = mock_response

    with patch("par_ai_core.search_utils.youtube_get_comments", return_value=["Comment 1", "Comment 2"]):
        results = youtube_search("test query", max_results=1, max_comments=2)

        assert len(results) == 1
        assert "Comment 1" in results[0]["content"]
        assert "Comment 2" in results[0]["content"]


def test_youtube_get_transcript_error():
    """Test YouTube transcript fetching error handling."""
    with patch(
        "par_ai_core.search_utils.YouTubeTranscriptApi.get_transcript", side_effect=Exception("Transcript error")
    ):
        with pytest.raises(Exception, match="Transcript error"):
            youtube_get_transcript("test_video_id")


@patch("par_ai_core.search_utils.praw.Reddit")
def test_reddit_search_subreddit_fallback(mock_reddit):
    """Test Reddit search with subreddit fallback."""
    mock_subreddit = MagicMock()
    # Set up the mock submission
    mock_submission = MagicMock()
    mock_submission.title = "Test Title"
    mock_submission.url = "https://example.com"
    mock_submission.selftext = "Test content"
    mock_author = MagicMock()
    mock_author.name = "TestAuthor"
    mock_submission.author = mock_author
    mock_submission.score = 100

    # Configure the mock subreddit's behavior
    mock_subreddit.search.return_value = [mock_submission]
    mock_subreddit.hot.return_value = [mock_submission]
    mock_subreddit.new.return_value = [mock_submission]
    mock_subreddit.controversial.return_value = [mock_submission]

    # Configure the Reddit client mock
    mock_reddit.return_value.subreddit.return_value = mock_subreddit

    with patch.dict(
        os.environ,
        {
            "REDDIT_CLIENT_ID": "test_id",
            "REDDIT_CLIENT_SECRET": "test_secret",
            "REDDIT_USERNAME": "test_user",
            "REDDIT_PASSWORD": "test_pass",
        },
    ):
        # Test subreddit fallback
        results = reddit_search("test query", subreddit="nonexistent")
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"

        # Test hot posts
        results = reddit_search("hot", subreddit="all")
        assert len(results) == 1

        # Test new posts
        results = reddit_search("new", subreddit="all")
        assert len(results) == 1

        # Test controversial posts
        results = reddit_search("controversial", subreddit="all")
        assert len(results) == 1

        # Test with deleted author
        mock_submission.author = None
        results = reddit_search("test query")
        assert len(results) == 1
        assert "Unknown" in results[0]["raw_content"]


@patch("par_ai_core.search_utils.praw.Reddit")
def test_reddit_search(mock_reddit):
    """Test Reddit search functionality."""
    mock_submission = MagicMock()
    mock_submission.title = "Test Title"
    mock_submission.url = "https://example.com"
    mock_submission.selftext = "Test content"
    mock_submission.author = MagicMock()
    mock_submission.author.name = "TestAuthor"
    mock_submission.score = 100

    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [mock_submission]
    mock_reddit.return_value.subreddit.return_value = mock_subreddit

    with patch.dict(
        os.environ,
        {
            "REDDIT_CLIENT_ID": "test_id",
            "REDDIT_CLIENT_SECRET": "test_secret",
            "REDDIT_USERNAME": "test_user",
            "REDDIT_PASSWORD": "test_pass",
        },
    ):
        results = reddit_search("test query", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["content"] == "Test content"
