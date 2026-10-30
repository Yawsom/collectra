import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

__all__ = ["T", "EntityMixin", "BaseEntity", "Node", "NodeStatus", "TaskContext"]

T = TypeVar("T")


@dataclass
class TaskContext:
    """Runtime context for task execution.

    This provides tasks with runtime information without polluting
    their function signatures or requiring global state.

    Attributes:
        usage_file: Optional path to write LLM usage statistics.
        render: Whether to render workflow visualizations.
        file_path: Current file being processed.
        single_run: Whether to stop after first task execution.
    """

    usage_file: Path | None = None
    render: bool = False
    file_path: Path | None = None
    single_run: bool = False


class EntityMixin:
    """Naming and serialisation shared by artefacts and pydantic task models."""

    def __str__(self) -> str:
        return f"{self.name}\n{self.get_class_path()}"

    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: The name of the task.
        """
        return self.name

    def serialize(self) -> dict:
        serialized = dict(type=self.get_class_path())
        for key, value in self.attributes.items():
            if isinstance(value, Path):
                value = value.name
            serialized[key] = value
        return serialized

    @property
    def attributes(self) -> dict:
        ignore = self.attributes_to_ignore()
        values = dict(self.__dict__)
        # Pydantic keeps undeclared (extra) attributes outside ``__dict__``.
        values.update(getattr(self, "__pydantic_extra__", None) or {})
        return {k: v for k, v in values.items() if k not in ignore}

    def attributes_to_ignore(self) -> set:
        return set()

    @classmethod
    def get_class_path(cls: type) -> str:
        """Return the fully qualified path for a class.

        Example:
            >>> from collectra.images import Image
            >>> get_class_path(Image)
            'collectra.images.Image'
        """
        return f"{cls.__module__.split('.')[0]}.{cls.__name__}"


class BaseEntity(EntityMixin, ABC, Generic[T]):

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self) -> T:
        pass


class NodeStatus(enum.Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


@dataclass
class Node:

    name: str
    status: NodeStatus = field(init=False, default=NodeStatus.NOT_READY)

    def __post_init__(self) -> None:
        pass

    def process(self) -> list:
        return []
