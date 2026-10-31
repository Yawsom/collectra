"""Text handling and processing classes for Collectra workflows.

This module provides basic text data structures and utilities for handling
textual content within Collectra workflows. It supports text content with
optional path associations for file-based text processing tasks.

Classes:
    Text: Basic text container with content and optional path metadata
"""

from pathlib import Path

from pydantic import BaseModel


class Text(BaseModel):
    """Basic text container for handling textual content in workflows.

    Provides a simple data structure for storing text content along with
    optional path metadata for file-based text processing operations.

    Attributes:
        content (str): The actual text content.
        path (Path): Optional path to the source file.
    """

    content: str
    path: Path | None = None

    def __init__(self, content: str | None = None, **kwargs):
        """Initialize a Text instance with content.

        Args:
            content (str): The text content to store.
        """
        if content is not None:
            kwargs["content"] = content
        super().__init__(**kwargs)

    def __str__(self):
        """Return the text content as a string.

        Returns:
            str: The stored text content.
        """
        return self.content
