from dataclasses import dataclass, field
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

from markdown import Markdown as MDown

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


@dataclass
class Text(Artefact):

    data: str | Path | None = field(default=None)
    path: str | Path | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.path is not None and self.data is not None:
            raise ValueError("Text accepts either data or path, not both")
        # YAML reads ``book: 1`` as a number, but a Text always holds text.
        if isinstance(self.data, (bool, int, float)):
            self.data = str(self.data)
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
