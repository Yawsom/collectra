"""Resources are kept by name on the pipeline and are not nodes of its graph."""

import pytest

from collectra import Pipeline
from collectra.resources.base import Resource
from collectra.resources.entity_lookup import EntityLookup
from collectra.tasks.llms import LLM


def make(tmp_path, **entries):
    (tmp_path / "genera.txt").write_text("Didymograptus\nMonograptus\n")
    config = {
        "genera_lookup": {
            "type": "collectra.EntityLookup",
            "file": "genera.txt",
            "description": "Graptolite genera",
        },
        **entries,
    }
    return Pipeline("Res", "res", "0.1.0", path=str(tmp_path), **config)


def test_a_resource_entry_is_kept_by_name_and_is_not_a_graph_node(tmp_path):
    pipeline = make(tmp_path)
    pipeline.connect()
    lookup = pipeline.resources["genera_lookup"]
    assert isinstance(lookup, EntityLookup) and isinstance(lookup, Resource)
    assert lookup.description == "Graptolite genera"
    assert not pipeline.node_manager.has_node("genera_lookup")
    assert pipeline.node_manager.get_artefact_nodes() == []
    assert pipeline.node_manager.get_task_nodes() == []


def test_the_file_is_found_relative_to_the_pipeline_folder(tmp_path, monkeypatch):
    pipeline = make(tmp_path)
    monkeypatch.chdir("/")
    pipeline.connect()
    assert pipeline.resources["genera_lookup"].search("didymograptus")[0].exact


def test_connecting_again_keeps_the_same_resource(tmp_path):
    pipeline = make(tmp_path)
    pipeline.connect()
    first = pipeline.resources["genera_lookup"]
    pipeline.connect()
    assert pipeline.resources["genera_lookup"] is first


def test_resource_looks_up_by_name_and_names_what_exists_when_it_is_missing(tmp_path):
    pipeline = make(tmp_path)
    pipeline.connect()
    assert pipeline.resource("genera_lookup") is pipeline.resources["genera_lookup"]
    with pytest.raises(KeyError, match="genera_lookup"):
        pipeline.resource("nope")


def test_a_mistake_in_a_resource_entry_is_reported_with_its_name(tmp_path):
    pipeline = make(tmp_path)
    pipeline.data["genera_lookup"]["descripton"] = "typo"
    with pytest.raises(Exception, match="descripton"):
        pipeline.connect()


def test_resources_do_not_appear_among_the_runnable_tasks(tmp_path):
    from collectra.cli import command_for_object

    pipeline = make(tmp_path)
    task_group = pipeline.cli_tasks()
    assert "genera_lookup" not in task_group


def test_entries_that_are_neither_tasks_nor_resources_are_still_artefacts(tmp_path):
    pipeline = make(tmp_path, note={"type": "collectra.Text", "data": "x"})
    pipeline.connect()
    assert [n.name for n in pipeline.node_manager.get_artefact_nodes()] == ["note"]


# --- tasks that name tools are checked when the pipeline connects ----------------------


def llm(tools):
    return {
        "type": "collectra.LLM",
        "model": "m",
        "template": "{note}",
        "input": "note",
        "output": "answer",
        "tools": tools,
    }


def test_a_task_naming_an_unknown_tool_fails_at_connect_with_the_choices(tmp_path):
    pipeline = make(tmp_path, reader=llm(["genera_lookup", "genra_lookup"]))
    with pytest.raises(ValueError) as error:
        pipeline.connect()
    message = str(error.value)
    assert "reader" in message and "genra_lookup" in message
    assert "genera_lookup" in message.split("Available tools:")[1]


def test_a_task_naming_known_tools_connects(tmp_path):
    pipeline = make(tmp_path, reader=llm(["genera_lookup"]))
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("reader").get_task()
    assert isinstance(task, LLM) and task.tools == ["genera_lookup"]


def test_tasks_with_no_tools_are_unaffected(tmp_path):
    pipeline = make(tmp_path, reader={**llm([]), "tools": []})
    pipeline.connect()
