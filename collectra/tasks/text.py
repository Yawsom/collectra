"""Tasks for combining and transforming text artefacts."""

from __future__ import annotations

from typing import Any

from collectra.tasks.text_output import TextOutputTask
from collectra.types.texts import Text

__all__ = ["ConcatenateText"]


class ConcatenateText(TextOutputTask):
    """Join every input Text artefact into one Text artefact."""

    collective: bool = True

    def __init__(self, name: str, separator: str = "\n", **kwargs: Any) -> None:
        super().__init__(name, separator=separator, **kwargs)

    def input_type(self) -> type:
        return Text

    def output_type(self) -> type:
        return Text

    def run(self, *texts: Text) -> Text:
        for text in texts:
            if not isinstance(text, Text):
                raise TypeError(f"text must be Text, got {type(text)}")
        return self.text_output(self.separator.join(str(text()) for text in texts))
