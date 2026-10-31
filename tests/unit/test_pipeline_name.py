"""The pipeline class is called Pipeline; the old name still works."""

import collectra
from collectra import Pipeline
from collectra.pipelines.base import Pipeline as FromModule


def test_pipeline_is_exported_from_the_package_and_its_module():
    assert Pipeline is FromModule
    assert Pipeline.__name__ == "Pipeline"
    assert "Pipeline" in dir(collectra)


def test_the_old_name_is_an_alias_for_the_same_class():
    assert collectra.Collectra is Pipeline
    from collectra.pipelines.base import Collectra

    assert Collectra is Pipeline
    assert Collectra("Demo", "dmo", "1").name == "Demo"


def test_from_file_is_available_under_the_new_name(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        "collectra_pipeline_metadata:\n  name: Demo\n  ext: dmo\n  version: 0.1.0\n"
    )
    pipeline = Pipeline.from_file(tmp_path / "pipeline.yaml")
    assert isinstance(pipeline, Pipeline)
    assert (pipeline.name, pipeline.ext) == ("Demo", "dmo")
