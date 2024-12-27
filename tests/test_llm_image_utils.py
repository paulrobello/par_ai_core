"""Tests for llm_image_utils module."""

import base64
from pathlib import Path
import pytest

from src.par_ai_core.llm_image_utils import (
    UnsupportedImageTypeError,
    b64_encode_image,
    try_get_image_type,
    image_to_base64,
    image_to_chat_message,
)


def test_b64_encode_image():
    """Test b64_encode_image function."""
    test_bytes = b"test image data"
    expected = base64.b64encode(test_bytes).decode("utf-8")
    result = b64_encode_image(test_bytes)
    assert result == expected


@pytest.mark.parametrize(
    "image_path,expected",
    [
        ("test.jpg", "jpeg"),
        ("test.jpeg", "jpeg"),
        ("test.png", "png"),
        ("test.gif", "gif"),
        (Path("test.jpg"), "jpeg"),
        ("data:image/jpeg;base64,abc123", "jpeg"),
        ("data:image/png;base64,abc123", "png"),
        ("data:image/gif;base64,abc123", "gif"),
    ],
)
def test_try_get_image_type_valid(image_path, expected):
    """Test try_get_image_type function with valid inputs."""
    assert try_get_image_type(image_path) == expected


@pytest.mark.parametrize(
    "image_path",
    [
        "test.bmp",
        "test.webp",
        "test.tiff",
        "data:image/webp;base64,abc123",
    ],
)
def test_try_get_image_type_invalid(image_path):
    """Test try_get_image_type function with invalid inputs."""
    with pytest.raises(UnsupportedImageTypeError):
        try_get_image_type(image_path)


@pytest.mark.parametrize(
    "image_type,expected_prefix",
    [
        ("jpeg", "data:image/jpeg;base64,"),
        ("png", "data:image/png;base64,"),
        ("gif", "data:image/gif;base64,"),
    ],
)
def test_image_to_base64(image_type, expected_prefix):
    """Test image_to_base64 function."""
    test_bytes = b"test image data"
    expected_b64 = base64.b64encode(test_bytes).decode("utf-8")
    result = image_to_base64(test_bytes, image_type)

    assert result.startswith(expected_prefix)
    assert result.endswith(expected_b64)


def test_image_to_chat_message():
    """Test image_to_chat_message function."""
    test_url = "data:image/jpeg;base64,abc123"
    expected = {
        "type": "image_url",
        "image_url": {"url": test_url},
    }
    result = image_to_chat_message(test_url)
    assert result == expected
