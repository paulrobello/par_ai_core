"""Utilities for handling images with LLMs.

This module provides functions for encoding, decoding and processing images
for use with language learning models.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal


class UnsupportedImageTypeError(ValueError):
    """Unsupported image type error."""


def b64_encode_image(image_path: bytes) -> str:
    """Encode an image as base64.

    Args:
        image_path: Raw bytes of the image to encode

    Returns:
        Base64 encoded string representation of the image
    """
    return base64.b64encode(image_path).decode("utf-8")


def try_get_image_type(image_path: str | Path) -> Literal["jpeg", "png", "gif"]:
    """Get image type from image path.

    Args:
        image_path: Path to the image file or data URL

    Returns:
        Image type as one of: "jpeg", "png", "gif"

    Raises:
        UnsupportedImageTypeError: If image type is not supported
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    if image_path.startswith("data:"):
        ext = image_path.split(";")[0].split("/")[-1].lower()
    else:
        ext = image_path.split(".")[-1].lower()
    if ext in ["jpg", "jpeg"]:
        return "jpeg"
    if ext in ["png"]:
        return "png"
    if ext in ["gif"]:
        return "gif"
    raise UnsupportedImageTypeError(f"Unsupported image type: {ext}")


def image_to_base64(image_bytes: bytes, image_type: Literal["jpeg", "png", "gif"] = "jpeg") -> str:
    """Convert an image to a base64 data URL.

    Args:
        image_bytes: Raw bytes of the image
        image_type: Type of image (jpeg, png, or gif). Defaults to "jpeg".

    Returns:
        Base64 data URL string representation of the image
    """
    return f"data:image/{image_type};base64,{b64_encode_image(image_bytes)}"


def image_to_chat_message(image_url_str: str) -> dict[str, Any]:
    """Convert an image URL to a chat message format.

    Args:
        image_url_str: URL or base64 data URL of the image

    Returns:
        Dictionary containing the image URL in chat message format
    """
    return {
        "type": "image_url",
        "image_url": {"url": image_url_str},
    }
