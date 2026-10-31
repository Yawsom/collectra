from types import SimpleNamespace

import pytest

from collectra import Pipeline, ConcatenateText, TaskNode, Text


def text(value: str, identifier: str) -> Text:
    return Text(name="lines", id=identifier, data=value)


def test_collective_task_prepares_all_node_items_as_one_entry():
    first = text("first", "first-id")
    second = text("second", "second-id")
    parent = SimpleNamespace(
        name="lines",
        items={first.id: first, second.id: second},
    )
    task = ConcatenateText("join", input="lines", output="joined")

    assert task.prepare_inputs([parent]) == [[first, second]]


def test_concatenates_text_in_node_order_with_configured_separator():
    task = ConcatenateText("join", output="joined", separator=" | ")

    result = task.run(text("first", "1"), text("second", "2"))

    assert result.name == "joined"
    assert result.data == "first | second"


def test_pipeline_assigns_every_collective_input_as_a_parent(tmp_path, monkeypatch):
    first = text("first", "first-id")
    second = text("second", "second-id")
    parent = SimpleNamespace(
        name="lines",
        items={first.id: first, second.id: second},
    )
    task = ConcatenateText("join", input="lines", output="joined")
    pipeline = Pipeline("collective", "collectra", "1", path=tmp_path)
    task_node = TaskNode("join", task)
    monkeypatch.setattr(
        pipeline.node_manager,
        "get_parents_artefact",
        lambda node: [parent],
    )

    results = pipeline._run_task(task_node)

    assert len(results) == 1
    assert results[0].data == "first\nsecond"
    assert results[0].parents == ["first-id", "second-id"]


def test_empty_collective_input_does_not_run():
    parent = SimpleNamespace(name="lines", items={})
    task = ConcatenateText("join", input="lines")

    assert task.prepare_inputs([parent]) == []


def test_rejects_non_text_input():
    with pytest.raises(TypeError, match="text must be Text"):
        ConcatenateText("join").run("not-text")
