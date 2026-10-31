"""Parent class for tasks whose output is text."""

from __future__ import annotations

import re

from pydantic import field_validator

from collectra.tasks.base import Task
from collectra.types.texts import Text

__all__ = ["TextOutputTask"]


class TextOutputTask(Task):
    """A task that produces Text artefacts, optionally saved as files.

    With ``output_extension`` set (for example ``md``, ``txt`` or ``xml``), each output
    is written to ``{node}/{artefact_id}.{extension}`` in the results folder and
    recorded there by path instead of inline. Later tasks still read the text as
    usual. Leave it unset to keep the text inline in ``results.yaml``.

    Subclasses build their outputs with ``self.text_output(...)``.
    """

    output_extension: str | None = None

    @field_validator("output_extension", mode="before")
    @classmethod
    def _plain_suffix(cls, value):
        if value is None:
            return None
        suffix = str(value).strip()
        suffix = suffix[1:] if suffix.startswith(".") else suffix
        if not suffix:
            return None
        if not re.fullmatch(r"[A-Za-z0-9]+", suffix):
            raise ValueError(
                f"output_extension must be a plain file suffix such as 'md', got {value!r}"
            )
        return suffix.lower()

    def text_output(self, data: str, name: str | None = None) -> Text:
        """Make this task's output Text, marked for saving if ``output_extension`` is set.

        ``name`` is the output node the text belongs to; it defaults to the
        task's first (or only) output.
        """
        text = Text(name=name or self.get_output_name(), data=data)
        if self.output_extension:
            text._file_extension = self.output_extension
        return text
