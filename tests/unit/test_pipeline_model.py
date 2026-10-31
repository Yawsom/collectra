"""The pipeline is a pydantic model."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from collectra import Pipeline
from collectra.pipelines.node_graph_manager import NodeGraphManager


def test_the_pipeline_is_a_pydantic_model():
    assert issubclass(Pipeline, BaseModel)


def test_name_ext_and_version_can_be_given_by_position_or_name(tmp_path):
    first = Pipeline("Demo", "dmo", "0.1.0", path=tmp_path)
    second = Pipeline(name="Demo", ext="dmo", version="0.1.0", path=tmp_path)
    assert (first.name, first.ext, first.version) == ("Demo", "dmo", "0.1.0")
    assert (second.name, second.ext, second.version) == ("Demo", "dmo", "0.1.0")


def test_the_path_defaults_to_a_folder_named_after_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Pipeline("Demo", "dmo", "1").path == tmp_path / "Demo"
    assert Pipeline("Demo", "dmo", "1", path="elsewhere").path == Path("elsewhere")
    assert Pipeline("Demo", "dmo", "1", path="").path == tmp_path / "Demo"


def test_numeric_versions_from_yaml_are_kept_as_text():
    assert Pipeline("Demo", "dmo", 1).version == "1"
    assert Pipeline("Demo", "dmo", 1.5).version == "1.5"


def test_values_are_validated():
    with pytest.raises(ValidationError):
        Pipeline(["not", "a", "name"], "dmo", "1")


def test_the_remaining_keywords_are_the_pipeline_definition(tmp_path):
    entry = {"type": "collectra.Text", "data": "x"}
    pipeline = Pipeline("Demo", "dmo", "1", path=tmp_path, note=entry)
    assert pipeline.data == {"note": entry}
    pipeline.data["other"] = {"type": "collectra.Text"}
    assert list(pipeline.data) == ["note", "other"]


def test_each_pipeline_has_its_own_definition_and_graph(tmp_path):
    first = Pipeline("Demo", "dmo", "1", path=tmp_path)
    second = Pipeline("Demo", "dmo", "1", path=tmp_path)
    first.data["x"] = 1
    assert second.data == {}
    assert isinstance(first.node_manager, NodeGraphManager)
    assert first.node_manager is not second.node_manager


def test_pipeline_metadata_always_carries_the_identity(tmp_path):
    pipeline = Pipeline(
        "Demo",
        "dmo",
        "1",
        path=tmp_path,
        pipeline_metadata={"icon": "x.png", "name": "ignored"},
    )
    assert pipeline.pipeline_metadata == {
        "icon": "x.png",
        "name": "Demo",
        "ext": "dmo",
        "version": "1",
    }


def test_the_constants_are_class_attributes_not_fields():
    assert Pipeline.RESULTS_FILE == "results.yaml"
    assert Pipeline.PIPELINE_FILE == "pipeline.yaml"
    assert Pipeline.METADATA_KEY == "collectra_results_metadata"
    assert Pipeline.COLOR_SUCCESS == "green"
    assert not {"RESULTS_FILE", "COLOR_FAILURE", "PIPELINE_FILE"} & set(
        Pipeline.model_fields
    )


def test_pipelines_compare_and_hash_by_identity(tmp_path):
    first = Pipeline("Demo", "dmo", "1", path=tmp_path)
    second = Pipeline("Demo", "dmo", "1", path=tmp_path)
    assert first == first and first != second
    assert len({first, second}) == 2


def test_the_repr_stays_short(tmp_path):
    pipeline = Pipeline("Demo", "dmo", "1", path=tmp_path, big={"x": "y" * 500})
    assert "yyyy" not in repr(pipeline)
    assert "Demo" in repr(pipeline)


def test_runtime_state_is_private_and_starts_empty(tmp_path):
    pipeline = Pipeline("Demo", "dmo", "1", path=tmp_path)
    assert pipeline._stage_gate is None and pipeline._run_progress is None
    pipeline._stage_gate = "held"
    pipeline.name = "Renamed"  # a validated assignment must not lose private state
    assert pipeline._stage_gate == "held"


def test_methods_can_be_replaced_on_an_instance(tmp_path):
    pipeline = Pipeline("Demo", "dmo", "1", path=tmp_path)
    pipeline.run = lambda *a, **k: "patched"
    pipeline._process_single_file = lambda *a, **k: "also patched"
    assert pipeline.run() == "patched"
    assert pipeline._process_single_file() == "also patched"
    assert Pipeline("Demo", "dmo", "1", path=tmp_path).run != pipeline.run


def test_the_pipeline_can_still_be_connected_and_listed(tmp_path):
    pipeline = Pipeline(
        "Demo",
        "dmo",
        "1",
        path=tmp_path,
        page={"type": "collectra.Text", "data": "x"},
    )
    pipeline.connect()
    assert [n.name for n in pipeline.node_manager.get_artefact_nodes()] == ["page"]
