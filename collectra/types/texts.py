import re
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Any

from markdown import Markdown as MDown
from pydantic import PrivateAttr, field_validator

from .base import Artefact

__all__ = ["Text"]


def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
MDown.output_formats["plain"] = unmark_element
__md = MDown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)


class Text(Artefact):

    data: str | Path | None = None
    path: str | Path | None = None

    # Set by a task that was asked to save its output; consumed by ``finalise``.
    _file_extension: str | None = PrivateAttr(default=None)

    @field_validator("data", mode="before")
    @classmethod
    def _numbers_are_text(cls, value):
        # YAML reads ``book: 1`` as a number, but a Text always holds text.
        if isinstance(value, (bool, int, float)):
            return str(value)
        return value

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.path is not None and self.data is not None:
            raise ValueError("Text accepts either data or path, not both")
        # Continue accepting legacy file references in data.
        if self.path is None and self.data:
            try:
                if Path(self.data).is_file():
                    self.path, self.data = Path(self.data), None
            except (OSError, ValueError):
                pass
        if self.path is not None:
            self.path = Path(self.path)
            self.data = self.path.read_text(encoding="utf-8")
        elif self.data is None:
            self.data = ""

    @staticmethod
    def _file_safe(name: str) -> str:
        """Keep a name inside its folder: no separators or leading dots."""
        return re.sub(r"[^\w.\-]", "_", name).lstrip(".")

    def stored_files(self, folder: Path) -> list[Path]:
        return [Path(folder) / self.path] if self.path is not None else []

    @classmethod
    def output_directory(cls, node: str) -> str:
        """Name of the results subfolder that holds a node's saved text files."""
        return cls._file_safe(node) or "text"

    def finalise(self, folder: Path, node: str) -> None:
        """Write the text to ``{node}/{id}.{extension}`` and refer to that file.

        Only texts that a task marked for saving are written, and only once.
        Empty text stays inline: an empty file would say nothing.
        The content stays available in memory for later tasks.
        """
        extension, self._file_extension = self._file_extension, None
        if not extension or self.path is not None or not str(self.data):
            return
        directory = self.output_directory(node)
        relative = f"{directory}/{self._file_safe(self.id) or directory}.{extension}"
        target = Path(folder) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(self.data), encoding="utf-8")
        self.path = relative

    def serialize(self) -> dict:
        serialized = super().serialize()
        if self.path is not None:
            serialized.pop("data", None)
            serialized["path"] = str(self.path)
        else:
            serialized.pop("path", None)
        return serialized

    def __call__(self) -> str:
        return str(self.data)

    def display(self, context) -> dict:
        return context.text(self)

    def evaluate(self, gold: "Text") -> float:
        if not isinstance(gold, Text):
            raise ValueError("Reference data must be an instance of Text.")
        ratio = SequenceMatcher(None, str(self.data), str(gold.data)).ratio()
        return ratio
