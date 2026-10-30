"""Base task class for Collectra workflows.

This module defines the fundamental task interface used throughout the Collectra system.
It provides the base classes that all tasks must inherit from and implements the core task execution lifecycle.

Classes:
    Task: Base class for all workflow tasks
"""

__all__ = ["Task", "TaskNode"]

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Generic

from pydantic import BaseModel, ConfigDict, Field

from collectra.commons import EntityMixin, Node, NodeStatus, T, TaskContext
from collectra.cli import command


class Task(EntityMixin, BaseModel, Generic[T]):
    """Abstract base class for all tasks in Collectra workflows.

    Defines the common interface and behavior that all tasks must implement.

    Attributes:
        name (str): Unique identifier for the task within the workflow.
        input (list[str]): List of input parameter names.
        output (list[str]): List of output names.
        context (TaskContext): Runtime context for task execution.
        fusion (bool): Opt-in feature fusion for task types that support it.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    name: str
    collective: bool = False
    fusion: bool = False
    batch_size: int = Field(default=1, ge=1)
    # Runtime wiring, set by the pipeline: neither validated, printed nor dumped.
    pipeline: Any = Field(default=None, exclude=True, repr=False)
    context: TaskContext = Field(default_factory=TaskContext, exclude=True, repr=False)

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Validated assignment rebuilds ``__dict__`` from the declared fields,
        # which would drop cached properties and private state held there.
        lazy = {
            key: held
            for key, held in self.__dict__.items()
            if key not in type(self).model_fields
        }
        super().__setattr__(name, value)
        for key, held in lazy.items():
            self.__dict__.setdefault(key, held)

    def attributes_to_ignore(self) -> set:
        return super().attributes_to_ignore() | {"pipeline", "context"}

    # Tasks are stateful, long-lived objects held by graph nodes: compare by identity.
    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = object.__hash__

    @property
    def input_dict(self) -> dict[str, list]:
        input_dict = dict()
        if hasattr(self, "input"):
            if not isinstance(self.input, list):
                input_list = [self.input]
            else:
                input_list = self.input.copy()
            input_dict = {k: [] for k in input_list}
        return input_dict

    def prepare_inputs(self, parents: list) -> list:
        input_dict = self.input_dict
        input_keys = list(input_dict.keys())
        for parent in parents:
            if parent.name in input_dict:
                input_dict[parent.name].extend(parent.items.values())

        if self.collective:
            collective_entries = [
                item for values in input_dict.values() for item in values
            ]
            return [collective_entries] if collective_entries else []

        # Filter out empty inputs (READY but no items)
        non_empty_dict = {k: v for k, v in input_dict.items() if v}
        if not non_empty_dict:
            return []
        non_empty_keys = list(non_empty_dict.keys())
        value_lists = list(non_empty_dict.values())
        entries = [list(combination) for combination in list(product(*value_lists))]

        entries_to_remove = []
        for entry_id, entry in enumerate(entries):
            entry_items_by_key = {f"{item.name}": item for item in entry}
            is_valid = True
            for item in entry:
                for parent_id in item.parents:
                    parent_key = re.sub("-.*", "", parent_id)
                    if (
                        parent_key in non_empty_keys
                    ):  # Only validate against non-empty inputs
                        if (
                            parent_key not in entry_items_by_key
                            or entry_items_by_key[parent_key].id != parent_id
                        ):
                            is_valid = False
                            break
            if not is_valid:
                entries_to_remove.append(entry_id)

        entries = [
            entry for id, entry in enumerate(entries) if id not in entries_to_remove
        ]
        return entries

    def run(self, *args) -> T | None:
        """Run the task execution logic.

        This method contains the core task execution logic and must be
        implemented by all subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def prefetch(self, entries: list[list]) -> None:
        """Optionally compute results for many input entries in one batch.

        The pipeline calls this with the entries of several files (each entry is
        the list of artefacts a later ``run`` call will receive) when
        ``batch_size`` is above 1. Tasks that can batch their work override this
        to cache results for ``run`` to reuse; the default does nothing.
        """

    def clear_prefetch(self) -> None:
        """Discard anything cached by ``prefetch``."""

    def get_output_name(self) -> str:
        name = (
            f"{self.get_name()}_output"
            if not hasattr(self, "output")
            else self.output[0] if isinstance(self.output, list) else self.output
        )
        return name

    def __call__(self, *args) -> T | None:
        """Run the task with the provided arguments.

        Args:
            **kwargs: Keyword arguments for task execution.

        Returns:

        """
        return self.run(*args)

    @command(name="run")
    def cli_run(
        self,
        inputs: list[str],
        output: Path | None = None,
        verbose: bool = False,
        usage: bool = False,
        render: bool = False,
    ) -> None:
        """Run this task for the supplied input files."""
        if self.pipeline is None:
            raise RuntimeError(f"Task {self.name!r} is not attached to a pipeline")
        self.pipeline.cli_run(
            inputs,
            task=self.name,
            output=output,
            single=True,
            verbose=verbose,
            usage=usage,
            render=render,
        )


@dataclass
class TaskNode(Node):

    task: Task

    def get_task(self) -> Task:
        return self.task
