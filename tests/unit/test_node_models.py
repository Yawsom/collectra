"""Graph nodes and the task context are pydantic models."""

import copy
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from collectra.commons import Node, NodeStatus, TaskContext
from collectra.tasks.base import Task, TaskNode
from collectra.types.base import ArtefactNode
from collectra.types.texts import Text


def test_nodes_and_the_context_are_pydantic_models():
    for cls in (Node, ArtefactNode, TaskNode, TaskContext):
        assert issubclass(cls, BaseModel), cls


# --- TaskContext ---------------------------------------------------------------


def test_task_context_defaults_are_class_level_fields():
    fields = TaskContext.model_fields
    assert fields["usage_file"].default is None
    assert fields["render"].default is False
    assert fields["file_path"].default is None
    assert fields["single_run"].default is False


def test_task_context_validates_and_converts_paths():
    context = TaskContext(
        usage_file="usage.yaml", file_path=Path("results"), render=True
    )
    assert context.usage_file == Path("usage.yaml")
    assert context.render is True
    with pytest.raises(ValidationError):
        TaskContext(render="maybe")


# --- Node --------------------------------------------------------------------


def test_a_new_node_is_not_ready_and_takes_its_name_positionally():
    node = Node("n")
    assert (node.name, node.status) == ("n", NodeStatus.NOT_READY)
    assert Node(name="m").name == "m"
    assert Node("n").process() == []


def test_status_is_validated_when_assigned():
    node = Node("n")
    node.status = NodeStatus.READY
    node.status = "running"
    assert node.status is NodeStatus.RUNNING
    with pytest.raises(ValidationError):
        node.status = "bogus"
    assert node.status is NodeStatus.RUNNING


def test_status_cannot_be_passed_positionally():
    with pytest.raises(TypeError, match="positional"):
        Node("n", NodeStatus.READY)


# --- ArtefactNode ------------------------------------------------------------


def test_artefact_node_defaults_are_declared_fields():
    fields = ArtefactNode.model_fields
    assert fields["skip_type_check"].default is False
    assert fields["ensemble"].default is False
    node = ArtefactNode("n")
    assert (node.items, node.ensemble_items, node.types) == ({}, {}, set())


def test_artefact_nodes_do_not_share_their_collections():
    first, second = ArtefactNode("a"), ArtefactNode("b")
    first.add_item(Text(name="a", id="x", data="1"))
    first.add_type(Text)
    assert second.items == {} and second.types == set()


def test_status_follows_whether_the_node_starts_with_items():
    assert ArtefactNode("n").status is NodeStatus.NOT_READY
    item = Text(name="n", id="n", data="x")
    assert ArtefactNode("n", items={"n": item}).status is NodeStatus.READY
    assert ArtefactNode("n", items={}).status is NodeStatus.NOT_READY


def test_adding_an_item_makes_the_node_ready_and_keys_it_by_id():
    node = ArtefactNode("n")
    item = Text(name="n", id="n-1", data="x")
    node.add_item(item)
    assert node.status is NodeStatus.READY
    assert node.get_item("n-1") is item
    assert node.get_item("missing") is None


def test_items_are_stored_as_the_very_same_artefacts():
    item = Text(name="n", id="n", data="x")
    node = ArtefactNode("n", items={"n": item})
    assert node.items["n"] is item


def test_items_can_be_changed_in_place():
    node = ArtefactNode("n")
    node.add_item(Text(name="n", id="a", data="1"))
    node.items.clear()
    node.items.update({"b": Text(name="n", id="b", data="2")})
    assert list(node.items) == ["b"]


def test_only_artefacts_are_accepted_as_items():
    with pytest.raises(ValidationError):
        ArtefactNode("n", items={"x": 5})
    node = ArtefactNode("n")
    with pytest.raises(ValidationError):
        node.items = {"x": 5}


def test_types_accept_a_list_and_become_a_set_of_classes():
    assert ArtefactNode("n", types=[Text, Text]).types == {Text}
    assert ArtefactNode("n", types={Text}).check_type(Text)
    with pytest.raises(ValidationError):
        ArtefactNode("n", types={"not a class"})


def test_check_type_follows_subclassing():
    node = ArtefactNode("n", types={Text})

    class Note(Text):
        pass

    assert node.check_type(Note)
    assert not node.check_type(int)


def test_artefact_nodes_describe_themselves():
    assert str(ArtefactNode("n", types={Text})) == "n\ncollectra.Text"


def test_artefact_nodes_compare_by_content_and_copy_deeply():
    item = Text(name="n", id="n", data="x")
    assert ArtefactNode("n") == ArtefactNode("n")
    assert ArtefactNode("n") != ArtefactNode("other")
    assert ArtefactNode("n", items={"n": item}) != ArtefactNode("n")
    node = ArtefactNode("n", items={"n": item}, types={Text})
    clone = copy.deepcopy(node)
    assert clone == node
    clone.items.clear()
    assert list(node.items) == ["n"]


def test_the_other_positional_fields_keep_their_order():
    item = Text(name="n", id="n", data="x")
    node = ArtefactNode("n", {"n": item}, {}, {Text}, True, True)
    assert (node.skip_type_check, node.ensemble, node.types) == (True, True, {Text})


# --- TaskNode ---------------------------------------------------------------


class Plain(Task):
    def run(self, text: Text) -> Text:
        return text


def test_a_task_node_holds_the_very_same_task():
    task = Plain("p")
    node = TaskNode("p", task)
    assert node.get_task() is task
    assert node.status is NodeStatus.NOT_READY
    assert TaskNode(name="q", task=task).name == "q"


def test_a_task_node_needs_a_task():
    with pytest.raises(ValidationError):
        TaskNode("p")
    with pytest.raises(ValidationError):
        TaskNode("p", "not a task")


def test_task_nodes_compare_their_tasks_by_identity():
    task = Plain("p")
    assert TaskNode("p", task) == TaskNode("p", task)
    assert TaskNode("p", task) != TaskNode("p", Plain("p"))
