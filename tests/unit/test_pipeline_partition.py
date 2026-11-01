from collections import Counter

import cappa
import pytest
import yaml

from collectra.cli import invoke
from collectra.commons.results import peek_header
from collectra.pipelines.base import Pipeline


@pytest.fixture
def dataset(tmp_path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        'collectra_pipeline_metadata: {name: HTR, ext: htr, version: "1.0"}\n'
    )
    parent = tmp_path / "results"
    parent.mkdir()
    for index in range(10):
        folder = parent / f"page-{index:02}.htr"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.safe_dump(
                {
                    "collectra_results_metadata": {"partition": "original"},
                    "text": {
                        "type": "collectra.Text",
                        "id": f"text-{index}",
                        "data": "Preserve this",
                    },
                }
            )
        )
    return Pipeline.from_file(pipeline_path), parent


def assignments(parent):
    return {
        path.parent.name: peek_header(yaml.safe_load(path.read_text())).partition
        for path in parent.glob("*.htr/results.yaml")
    }


def test_partition_command_is_in_pipeline_help(dataset, capsys):
    pipeline, parent = dataset
    with pytest.raises(cappa.HelpExit):
        invoke(pipeline, ["--help"], name="htr")
    assert "partition" in capsys.readouterr().out


def test_percentage_partition_parent_directory_and_reproducible_seed(dataset):
    pipeline, parent = dataset
    command = [
        "partition",
        str(parent),
        "--train",
        "80%",
        "--validation",
        "20%",
        "--seed",
        "17",
        "--force",
    ]
    invoke(pipeline, command, name="htr")
    first = assignments(parent)
    assert Counter(first.values()) == {"train": 8, "validation": 2}
    invoke(pipeline, command, name="htr")
    assert assignments(parent) == first
    for path in parent.glob("*.htr/results.yaml"):
        assert yaml.safe_load(path.read_text())["text"]["data"] == "Preserve this"


def test_fractional_partitions_and_val_label(dataset):
    pipeline, parent = dataset
    invoke(
        pipeline,
        [
            "partition",
            str(parent),
            "--train",
            "0.6",
            "--val",
            "0.2",
            "--test",
            "0.2",
            "--force",
        ],
    )
    assert Counter(assignments(parent).values()) == {"train": 6, "val": 2, "test": 2}


def test_custom_count_partitions_and_leftover_results(dataset):
    pipeline, parent = dataset
    invoke(
        pipeline,
        [
            "partition",
            str(parent),
            "--partition",
            "training=3",
            "--partition",
            "heldout=2",
            "--force",
        ],
    )
    assert Counter(assignments(parent).values()) == {
        "training": 3,
        "heldout": 2,
        "original": 5,
    }


def test_individual_results_yaml_and_duplicate_inputs(dataset):
    pipeline, parent = dataset
    folder = parent / "page-00.htr"
    invoke(
        pipeline,
        [
            "partition",
            str(folder),
            str(folder / "results.yaml"),
            "--test",
            "1",
            "--force",
        ],
    )
    assert Counter(assignments(parent).values()) == {"test": 1, "original": 9}


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--train", "110%"],
        ["--partition", "bad"],
        ["--train", "1", "--partition", "train=2"],
        ["--train", "abc"],
        ["--train", "80%"],
    ],
)
def test_invalid_partitions_leave_files_unchanged(dataset, options):
    pipeline, parent = dataset
    before = {path: path.read_bytes() for path in parent.glob("*.htr/results.yaml")}
    with pytest.raises(ValueError):
        invoke(pipeline, ["partition", str(parent), *options])
    assert all(path.read_bytes() == value for path, value in before.items())
