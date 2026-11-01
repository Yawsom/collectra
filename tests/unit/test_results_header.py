"""A results file declares its own header as an entry, like any other artefact."""

import warnings

import pytest
import yaml

import collectra
from collectra import Results
from collectra.commons import results as header_module
from collectra.commons.files import CollectraFile, CollectraResultsMetadata
from collectra.commons.results import (
    HEADER_KEY,
    LEGACY_KEY,
    is_header,
    peek_header,
    pop_header,
    put_header,
)

NEW = {
    "type": "collectra.Results",
    "partition": "validation",
    "timestamp": "2026-02-13T17:36:26",
}
IMAGE = {"type": "collectra.Image", "id": "page", "data": "page.png"}


class MyResults(Results):
    """A header someone extended with their own field."""

    batch: str | None = None


@pytest.fixture(autouse=True)
def warn_again(monkeypatch):
    monkeypatch.setattr(header_module, "_warned", False)


# --- the class --------------------------------------------------------------------------


def test_results_is_exported_and_the_old_name_is_an_alias():
    assert collectra.Results is Results
    assert CollectraResultsMetadata is Results


def test_the_entry_names_its_class_and_drops_what_is_unset():
    entry = Results(partition="train").entry()
    assert entry["type"] == "collectra.Results"
    assert entry["partition"] == "train" and "timestamp" in entry
    assert "workflow" not in entry and "version" not in entry


def test_a_subclass_writes_its_own_type_and_keeps_extra_keys():
    entry = MyResults(batch="b1", note="kept").entry()
    assert entry["type"] == "tests.unit.test_results_header.MyResults"
    assert entry["batch"] == "b1" and entry["note"] == "kept"


def test_a_numeric_version_is_text():
    assert Results(version=1.0).version == "1.0"


# --- finding the header -----------------------------------------------------------------


def test_is_header_recognises_the_class_and_its_subclasses():
    assert is_header(NEW)
    assert is_header({"type": "tests.unit.test_results_header.MyResults"})


@pytest.mark.parametrize(
    "value",
    [
        IMAGE,
        "text",
        [NEW],
        {"id": "x"},
        {"type": 3},
        {"type": "no_such_module.Results"},
        {"type": "tests.unit.test_results_header.NoSuchClass"},
        {"type": "tests.unit.test_results_header.IMAGE"},
        {"type": "nodots"},
    ],
)
def test_is_header_ignores_everything_else(value):
    assert not is_header(value)


def test_the_header_is_found_whatever_its_key_and_position():
    data = {"page": dict(IMAGE), "meta": dict(NEW)}
    header = pop_header(data)
    assert header.partition == "validation" and list(data) == ["page"]


def test_a_file_without_a_header_gets_an_empty_one():
    data = {"page": dict(IMAGE)}
    assert pop_header(data) == Results()
    assert list(data) == ["page"]


def test_a_subclass_header_comes_back_as_that_class():
    data = {
        HEADER_KEY: {"type": "tests.unit.test_results_header.MyResults", "batch": "b"}
    }
    header = pop_header(data)
    assert type(header) is MyResults and header.batch == "b"


def test_peek_leaves_the_data_alone():
    data = {HEADER_KEY: dict(NEW), "page": dict(IMAGE)}
    assert peek_header(data).partition == "validation"
    assert list(data) == [HEADER_KEY, "page"] and data[HEADER_KEY] == NEW


# --- the old form -----------------------------------------------------------------------


def test_the_old_header_still_reads_with_a_warning_naming_the_file():
    data = {LEGACY_KEY: {"partition": "train", "workflow": "W"}, "page": dict(IMAGE)}
    with pytest.warns(
        FutureWarning, match=r"deprecated.*a\.yaml.*type: collectra.Results"
    ):
        header = pop_header(data, source="a.yaml")
    assert (header.partition, header.workflow) == ("train", "W")
    assert list(data) == ["page"]


def test_an_empty_old_header_reads_as_empty():
    data = {LEGACY_KEY: None}
    with pytest.warns(FutureWarning):
        assert pop_header(data) == Results()


def test_the_warning_appears_once_not_for_every_file():
    with pytest.warns(FutureWarning) as caught:
        for _ in range(3):
            pop_header({LEGACY_KEY: {}})
    assert len(caught) == 1


def test_the_warning_works_without_a_source():
    with pytest.warns(FutureWarning, match="deprecated"):
        pop_header({LEGACY_KEY: {}})


# --- writing ----------------------------------------------------------------------------


def test_put_header_puts_the_new_entry_first():
    result = put_header(Results(partition="p"), {"page": dict(IMAGE)})
    assert list(result) == [HEADER_KEY, "page"]
    assert result[HEADER_KEY]["type"] == "collectra.Results"


def test_a_node_called_results_does_not_collide_with_the_header():
    result = put_header(Results(), {"results": dict(IMAGE)})
    assert list(result) == ["collectra_results", "results"]
    assert result["results"] == IMAGE
    assert pop_header(dict(result)) == Results(
        timestamp=result["collectra_results"]["timestamp"]
    )


# --- CollectraFile ----------------------------------------------------------------------


def write(folder, data):
    folder.mkdir(exist_ok=True)
    (folder / "results.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def test_a_file_with_the_new_header_loads(tmp_path):
    folder = write(tmp_path / "a.x", {HEADER_KEY: dict(NEW), "page": dict(IMAGE)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        file = CollectraFile.from_data(folder)
    assert file.collectra_results_metadata.partition == "validation"
    assert list(file.data) == ["page"]


def test_a_file_with_the_old_header_loads_and_warns_naming_it(tmp_path):
    folder = write(
        tmp_path / "a.x", {LEGACY_KEY: {"partition": "train"}, "page": dict(IMAGE)}
    )
    with pytest.warns(FutureWarning, match="a.x"):
        file = CollectraFile.from_data(folder)
    assert file.collectra_results_metadata.partition == "train"


def test_saving_writes_the_new_header_first_even_for_an_old_file(tmp_path):
    folder = write(
        tmp_path / "a.x", {LEGACY_KEY: {"partition": "train"}, "page": dict(IMAGE)}
    )
    with pytest.warns(FutureWarning):
        file = CollectraFile.from_data(folder)
    file.save()
    saved = yaml.safe_load((folder / "results.yaml").read_text())
    assert list(saved) == [HEADER_KEY, "page"]
    assert saved[HEADER_KEY]["type"] == "collectra.Results"
    assert saved[HEADER_KEY]["partition"] == "train"
    assert LEGACY_KEY not in saved
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            CollectraFile.from_data(folder).collectra_results_metadata.partition
            == "train"
        )


def test_a_subclass_header_survives_a_load_and_save(tmp_path):
    entry = {"type": "tests.unit.test_results_header.MyResults", "batch": "b7"}
    folder = write(tmp_path / "a.x", {HEADER_KEY: entry, "page": dict(IMAGE)})
    file = CollectraFile.from_data(folder)
    assert type(file.collectra_results_metadata) is MyResults
    file.save()
    saved = yaml.safe_load((folder / "results.yaml").read_text())
    assert saved[HEADER_KEY]["type"] == entry["type"]
    assert saved[HEADER_KEY]["batch"] == "b7"


def test_a_new_file_gets_the_new_header(tmp_path):
    source = tmp_path / "page.png"
    source.write_bytes(b"x")
    file = CollectraFile.from_file(source, "page", "x", partition="p")
    file.save()
    saved = yaml.safe_load((tmp_path / "page.x" / "results.yaml").read_text())
    assert list(saved)[0] == HEADER_KEY and saved[HEADER_KEY]["partition"] == "p"


# --- the GUI's graph --------------------------------------------------------------------


def graph_data(header):
    return {
        **header,
        "page": {"type": "collectra.Image", "id": "page", "data": "page.png"},
    }


@pytest.mark.parametrize(
    "header", [{HEADER_KEY: dict(NEW)}, {LEGACY_KEY: {"partition": "validation"}}]
)
def test_the_graph_reads_either_header_and_does_not_show_it_as_a_node(header):
    from collectra.gui.data_display import CollectraGraph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = CollectraGraph.from_yaml_data(graph_data(header))
    assert graph.metadata.partition == "validation"
    assert graph.nodes == ["page"]
    written = graph.to_yaml_data()
    assert list(written) == [HEADER_KEY, "page"]
    assert written[HEADER_KEY]["type"] == "collectra.Results"
    assert written[HEADER_KEY]["partition"] == "validation"


def test_the_graph_built_from_a_loaded_file_keeps_the_header(tmp_path):
    from collectra.gui.data_display import CollectraGraph

    folder = write(tmp_path / "a.x", graph_data({HEADER_KEY: dict(NEW)}))
    file = CollectraFile.from_data(folder)
    graph = CollectraGraph.from_collectra_file(file)
    assert graph.nodes == ["page"] and graph.metadata.partition == "validation"


# --- what the pipeline writes and reads ----------------------------------------------------


def test_a_new_results_file_made_by_the_pipeline_has_the_new_header(tmp_path):
    from pathlib import Path

    from collectra import Pipeline

    pipeline = Pipeline("Demo", "dmo", "1.0", path=tmp_path)
    folder = tmp_path / "a.dmo"
    folder.mkdir()
    pipeline._create_results_file(folder, "page", Path("page.png"))
    saved = yaml.safe_load((folder / "results.yaml").read_text())
    assert list(saved) == [HEADER_KEY, "page"]
    assert saved[HEADER_KEY]["type"] == "collectra.Results"
    assert (saved[HEADER_KEY]["workflow"], saved[HEADER_KEY]["version"]) == (
        "Demo",
        "1.0",
    )


@pytest.mark.parametrize(
    "header",
    [
        {LEGACY_KEY: {"partition": "train", "timestamp": "old"}},
        {
            HEADER_KEY: {
                "type": "collectra.Results",
                "partition": "train",
                "timestamp": "old",
            }
        },
    ],
)
def test_rewriting_a_results_file_keeps_the_header_current(tmp_path, header):
    from collectra import Pipeline
    from collectra.utils import change_dir

    pipeline = Pipeline("Demo", "dmo", "1.0", path=tmp_path)
    folder = write(tmp_path / "a.dmo", {**header, "page": dict(IMAGE)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with change_dir(folder):
            results = pipeline._load_or_create_results("page", "page.png")
    assert list(results) == [HEADER_KEY, "page"]
    assert results[HEADER_KEY]["partition"] == "train"
    assert results[HEADER_KEY]["timestamp"] != "old"


def test_a_missing_results_file_is_created_with_the_new_header(tmp_path):
    from collectra import Pipeline
    from collectra.utils import change_dir

    pipeline = Pipeline("Demo", "dmo", "1.0", path=tmp_path)
    folder = tmp_path / "empty.dmo"
    folder.mkdir()
    with change_dir(folder):
        results = pipeline._load_or_create_results("page", "page.png")
    assert list(results) == [HEADER_KEY, "page"]
    assert results["page"]["data"] == "page.png"


@pytest.mark.parametrize(
    "header",
    [
        {HEADER_KEY: {"type": "collectra.Results", "partition": "validation"}},
        {LEGACY_KEY: {"partition": "validation"}},
        {},
    ],
)
def test_training_finds_a_files_partition_in_either_header(tmp_path, header):
    from collectra.tasks.machine_learning.training import _file_partition

    folder = write(tmp_path / "a.x", {**header, "page": dict(IMAGE)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = _file_partition(folder)
    assert found == ("validation" if header else "")
    assert _file_partition(tmp_path / "missing") == ""


def test_the_evaluator_does_not_count_the_header_as_a_node(tmp_path):
    from collectra.evaluator.base import Evaluator

    folder = write(tmp_path / "a.x", {HEADER_KEY: dict(NEW), "page": dict(IMAGE)})
    labels = Evaluator(tmp_path, tmp_path, "x")._load_collectra_file(folder)
    assert list(labels) == ["page"]
