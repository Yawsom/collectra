from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from .base import Artefact

__all__ = ["Link"]


class Link(Artefact):
    """A soft reference to the artefact identified by its first parent."""

    target: Artefact | None = Field(default=None, repr=False, exclude=True)

    _not_compared: ClassVar[frozenset[str]] = frozenset({"source_file", "target"})

    @property
    def parent_id(self) -> str:
        return self.parents[0] if self.parents else ""

    def _stand_in_type(self) -> type:
        """The class of the artefact this link resolves to, for ``isinstance``."""
        target = self.__dict__.get("target")
        seen = {id(self)}
        while type(target) is Link:
            if id(target) in seen:
                return Link
            seen.add(id(target))
            target = target.__dict__.get("target")
        return type(target) if target is not None else Link

    def bind(self, artefacts: dict[str, Artefact]) -> Artefact | None:
        """Resolve the first parent ID and retain it as the runtime target."""
        self.target = artefacts.get(self.parent_id)
        return self.target

    def resolve(self) -> Artefact:
        """Follow any number of Links and return the concrete artefact."""
        current: Artefact = self
        chain: list[str] = []
        seen: set[int] = set()
        while type(current) is Link:
            if id(current) in seen:
                chain.append(current.id)
                raise RuntimeError(f"Cyclic Link chain: {' -> '.join(chain)}")
            seen.add(id(current))
            chain.append(current.id)
            target = current.__dict__.get("target")
            if target is None:
                raise RuntimeError(
                    f"Link chain {' -> '.join(chain)!r} is unresolved at "
                    f"parent {current.parent_id!r}"
                )
            current = target
        return current

    def __call__(self) -> Any:
        return self.resolve()()

    def display(self, context) -> dict:
        return self.resolve().display(context)

    def __getattr__(self, name: str) -> Any:
        # Pydantic's own lookup (private and extra attributes) comes first.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        target = self.__dict__.get("target")
        if target is None:
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("target")
        return attributes

    def serialize(self) -> dict:
        """Serialize only the soft-link identity and parent reference."""
        serialized = {
            "type": self.get_class_path(),
            "id": self.id,
        }
        if self.parents:
            serialized["parents"] = (
                self.parents
                if isinstance(self.parents, str)
                else self.parents[0] if len(self.parents) == 1 else self.parents
            )
        return serialized

    def extract(self, path: Path | str) -> None:
        self.resolve().extract(path)

    def evaluate(self, gold: Artefact) -> float:
        if type(gold) is Link:
            if self.target is not None and gold.target is not None:
                return self.resolve().evaluate(gold.resolve())
            return float(bool(self.parent_id) and self.parent_id == gold.parent_id)
        return self.resolve().evaluate(gold)
