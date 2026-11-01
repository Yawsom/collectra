"""A pipeline.yaml declares its pipeline as an entry, like any other class."""

import warnings

import pytest
import yaml

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.types.texts import Text

THIS = "tests.unit.test_pipeline_declaration"

TASK = f"""
reader:
  type: {THIS}.Describe
  input: page
  output: note
"""


class Describe(Task):
    def run(self, page: Text) -> Text:
        return Text(name="note", data="x")


class MyPipeline(Pipeline):
    """A pipeline someone wrote by inheriting from the standard one."""


def write(tmp_path, text):
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    return path


def load(tmp_path, text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Pipeline.from_file(write(tmp_path, text))


NEW = (
    "Grapto:\n  type: collectra.Pipeline\n  ext: grapto\n  version: 1.0\n"
    "  logo: Grapto.png\n  theme: Grapto.css\n" + TASK
)
LEGACY = (
    "collectra_pipeline_metadata:\n  name: Grapto\n  ext: grapto\n  version: 1.0\n"
    "  logo: Grapto.png\n" + TASK
)


# --- reading ----------------------------------------------------------------------------


def test_the_pipeline_entry_gives_the_name_and_attributes(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the new format raises no warning
        pipeline = Pipeline.from_file(write(tmp_path, NEW))
    assert type(pipeline) is Pipeline
    assert (pipeline.name, pipeline.ext, pipeline.version) == (
        "Grapto",
        "grapto",
        "1.0",
    )
    assert pipeline.path == tmp_path.resolve()
    assert pipeline.pipeline_metadata == {
        "name": "Grapto",
        "ext": "grapto",
        "version": "1.0",
        "logo": "Grapto.png",
        "theme": "Grapto.css",
    }


def test_the_pipeline_entry_is_not_a_task_or_artefact(tmp_path):
    pipeline = load(tmp_path, NEW)
    assert list(pipeline.data) == ["reader"]
    pipeline.connect()
    assert "Grapto" not in pipeline.node_manager.get_node_names()


def test_the_pipeline_entry_need_not_be_first(tmp_path):
    pipeline = load(tmp_path, TASK + NEW)
    assert pipeline.name == "Grapto" and list(pipeline.data) == ["reader"]


def test_a_subclass_can_be_the_pipeline(tmp_path):
    text = NEW.replace("collectra.Pipeline", f"{THIS}.MyPipeline")
    assert type(load(tmp_path, text)) is MyPipeline


def test_a_file_without_a_pipeline_says_so(tmp_path):
    with pytest.raises(ValueError, match="type: collectra.Pipeline"):
        load(tmp_path, TASK)


def test_an_empty_file_says_so(tmp_path):
    with pytest.raises(ValueError, match="type: collectra.Pipeline"):
        load(tmp_path, "")


@pytest.mark.parametrize("missing", ["ext", "version"])
def test_ext_and_version_are_required(tmp_path, missing):
    text = "\n".join(
        line for line in NEW.splitlines() if not line.strip().startswith(f"{missing}:")
    )
    with pytest.raises(ValueError, match=f"'Grapto'.*{missing}"):
        load(tmp_path, text)


# --- the old form ----------------------------------------------------------------------


def test_the_old_metadata_block_still_loads(tmp_path):
    pipeline = load(tmp_path, LEGACY)
    assert (pipeline.name, pipeline.ext, pipeline.version) == (
        "Grapto",
        "grapto",
        "1.0",
    )
    assert pipeline.pipeline_metadata["logo"] == "Grapto.png"
    assert list(pipeline.data) == ["reader"]


def test_the_old_metadata_block_warns_that_it_is_deprecated(tmp_path):
    path = write(tmp_path, LEGACY)
    with pytest.warns(FutureWarning, match="deprecated.*type: collectra.Pipeline"):
        Pipeline.from_file(path)


# --- saving ----------------------------------------------------------------------------


def saved(tmp_path):
    return yaml.safe_load((tmp_path / "pipeline.yaml").read_text())


def test_saving_keeps_the_new_form_with_the_pipeline_first(tmp_path):
    pipeline = load(tmp_path, NEW)
    pipeline.save()
    data = saved(tmp_path)
    assert list(data) == ["Grapto", "reader"]
    assert data["Grapto"] == {
        "type": "collectra.Pipeline",
        "ext": "grapto",
        "version": "1.0",
        "logo": "Grapto.png",
        "theme": "Grapto.css",
    }
    assert load(tmp_path, (tmp_path / "pipeline.yaml").read_text()).name == "Grapto"


def test_saving_does_not_turn_the_pipeline_entry_into_data(tmp_path):
    pipeline = load(tmp_path, NEW)
    pipeline.save("reader")
    pipeline.save()
    assert list(saved(tmp_path)) == ["Grapto", "reader"]


def test_saving_keeps_a_subclass_type(tmp_path):
    pipeline = load(tmp_path, NEW.replace("collectra.Pipeline", f"{THIS}.MyPipeline"))
    pipeline.save()
    assert saved(tmp_path)["Grapto"]["type"] == f"{THIS}.MyPipeline"


def test_saving_does_not_rewrite_the_old_form_behind_your_back(tmp_path):
    pipeline = load(tmp_path, LEGACY)
    pipeline.save()
    data = saved(tmp_path)
    assert list(data) == ["collectra_pipeline_metadata", "reader"]
    assert data["collectra_pipeline_metadata"]["name"] == "Grapto"


def test_a_pipeline_made_in_code_saves_in_the_new_form(tmp_path):
    pipeline = Pipeline("Demo", "dmo", "1", path=tmp_path)
    pipeline.save()
    assert saved(tmp_path) == {
        "Demo": {"type": "collectra.Pipeline", "ext": "dmo", "version": "1"}
    }


def test_a_task_whose_class_cannot_be_loaded_does_not_hide_the_pipeline(tmp_path):
    broken = "broken:\n  type: no_such_module.Task\n  input: a\n  output: b\n"
    assert load(tmp_path, broken + NEW).name == "Grapto"
    missing_class = f"broken:\n  type: {THIS}.NoSuchClass\n  input: a\n  output: b\n"
    assert load(tmp_path, missing_class + NEW).name == "Grapto"


def test_plain_values_before_the_pipeline_are_skipped(tmp_path):
    assert load(tmp_path, "llm: some-model\nnotes: [a, b]\n" + NEW).name == "Grapto"
