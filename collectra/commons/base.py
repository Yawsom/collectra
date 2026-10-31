import enum
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = [
    "T",
    "EntityMixin",
    "PositionalFields",
    "LenientAttributes",
    "BaseEntity",
    "Node",
    "NodeStatus",
    "TaskContext",
]

T = TypeVar("T")


class TaskContext(BaseModel):
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


class LenientAttributes:
    """Keeps pydantic models usable like plain Python objects when assigning.

    - A method or property can be replaced on an instance (tests and callers do
      this), without that being treated as a field assignment.
    - Validated assignment rebuilds ``__dict__`` from the declared fields; state
      held there outside the fields (cached properties, ad-hoc attributes) is
      carried over rather than lost.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        cls = type(self)
        declared = name in cls.model_fields or name in cls.__private_attributes__
        if not declared and hasattr(cls, name):
            object.__setattr__(self, name, value)
            return
        lazy = {
            key: held
            for key, held in self.__dict__.items()
            if key not in cls.model_fields
        }
        super().__setattr__(name, value)
        for key, held in lazy.items():
            self.__dict__.setdefault(key, held)


class PositionalFields:
    """Lets a model's fields be given positionally, in declaration order.

    Fields named in ``_runtime_fields`` are set or measured at runtime and are
    skipped, as they were when these classes were dataclasses.
    """

    _runtime_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            names = [
                field
                for field in type(self).model_fields
                if field not in self._runtime_fields
            ]
            if len(args) > len(names):
                raise TypeError(
                    f"{type(self).__name__} takes at most {len(names)} positional "
                    f"arguments but {len(args)} were given"
                )
            for field_name, value in zip(names, args):
                if field_name in kwargs:
                    raise TypeError(
                        f"{type(self).__name__} got multiple values for "
                        f"argument {field_name!r}"
                    )
                kwargs[field_name] = value
        super().__init__(**kwargs)


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


class Node(PositionalFields, BaseModel):
    """A node of the workflow graph."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
    )
    _runtime_fields: ClassVar[frozenset[str]] = frozenset({"status"})

    name: str
    status: NodeStatus = NodeStatus.NOT_READY

    def process(self) -> list:
        return []
