"""Tasks are pydantic models: typed fields with class-level defaults."""

from functools import cached_property
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from collectra.commons import TaskContext
from collectra.tasks.base import Task
from collectra.types.texts import Text


class Greeter(Task):
    greeting: str = "hello"
    repeat: int = 1
    destination: Path | None = None

    def run(self, text: Text) -> Text:
        return Text(name="out", data=(self.greeting + " ") * self.repeat)


def test_task_is_a_pydantic_model():
    assert issubclass(Task, BaseModel)
    assert Task("t").name == "t"
    assert Task(name="t").name == "t"


def test_base_defaults_are_class_level_fields():
    fields = Task.model_fields
    assert fields["collective"].default is False
    assert fields["fusion"].default is False
    assert fields["batch_size"].default == 1


def test_subclass_defaults_and_keyword_overrides():
    task = Greeter("g")
    assert (task.greeting, task.repeat, task.destination) == ("hello", 1, None)
    task = Greeter("g", greeting="hi", repeat=3, destination="out")
    assert (task.greeting, task.repeat) == ("hi", 3)
    assert task.destination == Path("out")


def test_values_are_validated_on_construction_and_on_assignment():
    with pytest.raises(ValidationError):
        Greeter("g", repeat="many")
    task = Greeter("g")
    task.repeat = "4"  # what a command-line override would supply
    assert task.repeat == 4
    with pytest.raises(ValidationError):
        task.repeat = "lots"
    assert task.repeat == 4


def test_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        Task("t", batch_size=0)
    task = Task("t", batch_size=2)
    with pytest.raises(ValidationError):
        task.batch_size = -1


def test_undeclared_attributes_are_still_accepted():
    task = Task("t", legacy_option=5)
    assert task.legacy_option == 5
    task.another = "x"
    assert task.another == "x"
    task._private = [1]
    assert task._private == [1]


def test_runtime_wiring_is_not_validated_or_printed():
    task = Task("t")
    assert isinstance(task.context, TaskContext)
    assert task.pipeline is None
    task.pipeline = object()  # a Collectra pipeline holds its tasks: no recursion
    task.context = TaskContext(render=True)
    assert "pipeline" not in repr(task)
    assert "pipeline" not in task.model_dump()


def test_tasks_compare_and_hash_by_identity():
    first, second = Greeter("g"), Greeter("g")
    assert first == first
    assert first != second
    assert len({first, second}) == 2


def test_generic_tasks_can_be_parametrised():
    class Joiner(Task[Text]):
        def run(self, text: Text) -> Text:
            return text

    assert isinstance(Joiner("j"), Task)


def test_cached_properties_work_on_task_models():
    calls = []

    class Lazy(Task):
        @cached_property
        def expensive(self) -> int:
            calls.append(1)
            return 42

    task = Lazy("l")
    assert (task.expensive, task.expensive, len(calls)) == (42, 42, 1)


def test_attributes_list_declared_and_extra_values_but_not_wiring():
    task = Greeter("g", repeat=2, legacy_option=5)
    assert task.attributes["repeat"] == 2
    assert task.attributes["legacy_option"] == 5
    assert task.attributes["name"] == "g"
    assert not {"pipeline", "context"} & set(task.attributes)
    assert task.serialize()["type"] == "tests.Greeter"


def test_calling_a_task_runs_it():
    assert Greeter("g", greeting="yo", repeat=2)(Text(name="t", data="")).data == (
        "yo yo "
    )


def test_lazy_state_survives_assignment_to_a_field():
    """Pydantic rebuilds ``__dict__`` on validated assignment; tasks keep their extras."""
    calls = []

    class Lazy(Task):
        label: str = "a"

        @cached_property
        def expensive(self) -> int:
            calls.append(1)
            return 42

    task = Lazy("l")
    task._private = ["kept"]
    assert task.expensive == 42
    task.label = "b"
    task.undeclared = "also kept"
    assert (task.expensive, len(calls)) == (42, 1)
    assert task._private == ["kept"]
    assert task.undeclared == "also kept"
    assert task.label == "b"
