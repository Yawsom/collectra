"""The base class for pipeline resources."""

import re

from pydantic import BaseModel, ConfigDict, field_validator

from collectra.commons import PositionalFields

__all__ = ["Resource"]

_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


class Resource(PositionalFields, BaseModel):
    """A named helper declared in ``pipeline.yaml`` and used by tasks.

    Unlike tasks and artefacts, a resource is not a node in the pipeline graph:
    the pipeline just keeps it by name, and a task refers to it by that name
    (for example ``tools: [genera_lookup]``). The name is also what an LLM sees
    when the resource is offered as a tool, so it must be a plain identifier.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_a_plain_identifier(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(
                f"name {value!r} must be 1-64 letters, digits, underscores or hyphens"
            )
        return value
