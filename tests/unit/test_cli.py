from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cappa
import collectra
import pytest

from collectra.cli import command, group, invoke
from collectra.main import find_pipeline
from collectra.main import main
from collectra.pipelines.base import Pipeline


class Detector:
    def __init__(self, calls: list):
        self.calls = calls

    @command
    def run(self, inputs: list[str]):
        """Run object detection."""
        self.calls.append(("detector-run", inputs))

    @command
    def train(
        self,
        inputs: list[str],
        dropout: float = 0.5,
        batch_size: int = 32,
    ):
        """Train object detection."""
        self.calls.append(("detector-train", inputs, dropout, batch_size))


class Classifier:
    @command
    def run(self, inputs: list[str]):
        """Run classification."""

    @command
    def train(self, inputs: list[str], epochs: int = 100):
        """Train classification."""


class Reader:
    @command
    def run(self, inputs: list[str]):
        """Run the reader."""


class RuntimeApplication:
    def __init__(self, tasks: dict[str, object], calls: list | None = None):
        self.tasks = tasks
        self.calls = calls if calls is not None else []

    @command
    def run(self, inputs: list[str]):
        """Run the complete pipeline."""
        self.calls.append(("run", inputs))

    @command
    def install(self, name: str):
        """Install this pipeline."""

    @group
    def task(self):
        """Operate on a task."""
        return self.tasks


def help_for(app, argv, capsys) -> str:
    with pytest.raises(cappa.HelpExit):
        invoke(app, argv, name="collectra")
    return capsys.readouterr().out


def test_rfdetr_training_help_and_annotated_options(capsys, monkeypatch):
    from collectra import ObjectDetectionRFDETR

    captured = {}
    monkeypatch.setattr(
        "collectra.tasks.object_detection.rfdetr.run_training_command",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    app = RuntimeApplication({"detector": ObjectDetectionRFDETR("detector")})
    help_text = help_for(app, ["task", "detector", "train", "--help"], capsys)
    normalized_help = " ".join(help_text.split())
    assert "Checkpoint path or pretrained variant" in normalized_help
    assert "rfdetr[plus]" in normalized_help
    assert "Number of training epochs" in normalized_help
    assert "--no-use-ema" in normalized_help

    invoke(
        app,
        [
            "task",
            "detector",
            "train",
            "data",
            "--model",
            "small",
            "--epochs",
            "3",
            "--no-use-ema",
            "--early-stopping",
        ],
        name="collectra",
    )
    assert captured["model"] == "small"
    assert captured["epochs"] == 3
    assert captured["use_ema"] is False
    assert captured["early_stopping"] is True


def test_dynamic_recursive_help(capsys):
    calls = []
    app = RuntimeApplication(
        {"label_classifier": Classifier(), "label_detector": Detector(calls)}
    )

    root_help = help_for(app, ["--help"], capsys)
    assert all(name in root_help for name in ("run", "install", "task"))

    task_help = help_for(app, ["task", "--help"], capsys)
    assert "label_classifier" in task_help
    assert "label_detector" in task_help

    detector_help = help_for(app, ["task", "label_detector", "--help"], capsys)
    assert "run" in detector_help
    assert "train" in detector_help


def test_capabilities_and_signature_help_are_task_specific(capsys):
    app = RuntimeApplication({"classifier": Classifier(), "reader": Reader()})

    reader_help = help_for(app, ["task", "reader", "--help"], capsys)
    assert "run" in reader_help
    assert "train" not in reader_help

    classifier_help = help_for(app, ["task", "classifier", "train", "--help"], capsys)
    assert "--epochs" in classifier_help


def test_parsed_arguments_reach_bound_method():
    calls = []
    app = RuntimeApplication({"detector": Detector(calls)})
    invoke(
        app,
        [
            "task",
            "detector",
            "train",
            "one",
            "two",
            "--dropout",
            "0.25",
            "--batch-size",
            "16",
        ],
        name="collectra",
    )
    assert calls == [("detector-train", ["one", "two"], 0.25, 16)]


@pytest.mark.parametrize(
    "argv",
    [
        ["task", "missing", "--help"],
        ["task", "reader", "train", "--help"],
    ],
)
def test_unknown_runtime_commands_fail_cleanly(argv):
    app = RuntimeApplication({"reader": Reader()})
    with pytest.raises(cappa.Exit) as error:
        invoke(app, argv, name="collectra")
    assert error.value.code == 2


def test_two_pipelines_produce_different_trees(capsys):
    first = RuntimeApplication({"detector": Detector([])})
    second = RuntimeApplication({"reader": Reader()})
    first_help = help_for(first, ["task", "--help"], capsys)
    second_help = help_for(second, ["task", "--help"], capsys)
    assert "detector" in first_help and "reader" not in first_help
    assert "reader" in second_help and "detector" not in second_help


def test_find_pipeline_accepts_both_option_forms():
    assert find_pipeline(["--pipeline", "a.yaml", "task"])[0] == Path("a.yaml")
    assert find_pipeline(["task", "--pipeline=b.yaml"])[0] == Path("b.yaml")


def write_empty_pipeline(path: Path) -> Path:
    pipeline_file = path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Test\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
    )
    return pipeline_file


def test_actual_entry_point_root_help(tmp_path, capsys):
    pipeline_file = write_empty_pipeline(tmp_path)
    with pytest.raises(cappa.HelpExit):
        main(["--pipeline", str(pipeline_file), "--help"])
    output = capsys.readouterr().out
    assert all(name in output for name in ("run", "install", "task", "artefact"))


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".TIF"])
@pytest.mark.parametrize("directory_input", [False, True])
def test_pipeline_run_accepts_tiff_inputs(
    tmp_path, monkeypatch, suffix, directory_input
):
    pipeline = Pipeline.from_file(write_empty_pipeline(tmp_path))
    image = tmp_path / f"multiview{suffix}"
    image.touch()
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append(kwargs))

    pipeline.cli_run([str(tmp_path if directory_input else image)])

    assert calls[0]["files"] == [image]


def test_pipeline_run_warns_when_no_inputs_match(tmp_path, monkeypatch, capsys):
    pipeline = Pipeline.from_file(write_empty_pipeline(tmp_path))
    unsupported = tmp_path / "unsupported.txt"
    unsupported.touch()
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append(kwargs))

    pipeline.cli_run([str(unsupported)])

    assert calls == []
    assert "No supported input files found" in capsys.readouterr().out


@pytest.mark.parametrize("spaced", [False, True])
def test_installed_pipeline_launcher_delegates_to_main(tmp_path, monkeypatch, spaced):
    if spaced:
        import shlex
        import sys

        directory = tmp_path / "Pipeline with 'quotes' and $symbols"
        directory.mkdir()
        interpreter = directory / "Python Environment" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        interpreter.chmod(0o755)
        monkeypatch.setattr(sys, "executable", str(interpreter))
        tmp_path = directory
    pipeline_file = write_empty_pipeline(tmp_path)
    pipeline = Pipeline.from_file(pipeline_file)
    bin_dir = tmp_path / "bin"
    pipeline.cli_install("test-pipeline", bin_dir=bin_dir)
    launcher = bin_dir / "test-pipeline"

    result = subprocess.run(
        [str(launcher), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert all(name in result.stdout for name in ("run", "install", "task", "artefact"))


def test_command_discovery_never_evaluates_model_properties(capsys):
    from functools import cached_property

    class ModelTask:
        @property
        def chain(self):
            raise AssertionError("Command discovery must not load the LLM chain")

        @cached_property
        def model(self):
            raise AssertionError("Command discovery must not load model weights")

        @command
        def run(self, text: str):
            """Run this task."""

    help_text = help_for(RuntimeApplication({"model": ModelTask()}), ["--help"], capsys)
    assert "task" in help_text
    assert "TEXT" in help_for(ModelTask(), ["run", "--help"], capsys)


def test_pipeline_exposes_commands_for_artefact_types(tmp_path, capsys):
    image_path = tmp_path / "source.jpg"
    from PIL import Image as PillowImage

    PillowImage.new("RGB", (10, 10)).save(image_path)
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Artefacts\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
        "Palynomorph:\n"
        "  type: collectra.ImageCrop\n"
        "  data: source.jpg\n"
    )
    pipeline = Pipeline.from_file(pipeline_file)

    artefact_help = help_for(pipeline, ["artefact", "--help"], capsys)
    image_help = help_for(pipeline, ["artefact", "Palynomorph", "--help"], capsys)

    assert "Palynomorph" in artefact_help
    assert "extract" in image_help
    assert "evaluate" in image_help

    evaluate_help = help_for(
        pipeline,
        ["artefact", "Palynomorph", "evaluate", "--help"],
        capsys,
    )
    assert "--match-by-order" in evaluate_help


def test_image_artefact_extract_command(tmp_path):
    image_path = tmp_path / "source.jpg"
    from PIL import Image as PillowImage

    PillowImage.new("RGB", (10, 10), "red").save(image_path)
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Artefacts\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
        "Palynomorph:\n"
        "  type: collectra.Image\n"
        "  data: source.jpg\n"
    )
    pipeline = Pipeline.from_file(pipeline_file)
    output = tmp_path / "output"

    invoke(
        pipeline,
        ["artefact", "Palynomorph", "extract", "--output", str(output)],
        name="collectra",
    )

    assert len(list(output.glob("*.jpg"))) == 1


def test_artefact_evaluate_command_uses_selected_node(tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text("configured value")
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        "collectra_pipeline_metadata:\n"
        "  name: Artefacts\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
        "Palynomorph:\n"
        "  type: collectra.Text\n"
        "  data: source.txt\n"
    )
    predictions = tmp_path / "predictions" / "sample.collectra"
    ground_truth = tmp_path / "ground_truth" / "differently-named.collectra"
    predictions.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    result = (
        "collectra_results_metadata: {}\n"
        "Palynomorph:\n"
        "  type: collectra.Text\n"
        "  id: palynomorph-1\n"
        "  data: matching text\n"
    )
    (predictions / "results.yaml").write_text(result)
    (ground_truth / "results.yaml").write_text(result)
    pipeline = Pipeline.from_file(pipeline_file)
    csv_output = tmp_path / "reports" / "evaluation.csv"

    invoke(
        pipeline,
        [
            "artefact",
            "Palynomorph",
            "evaluate",
            str(predictions),
            str(ground_truth),
            "--output",
            str(csv_output),
        ],
        name="palynomorph",
    )

    output = capsys.readouterr().out
    assert "Aggregate Evaluation Metrics" in output
    assert "for Text" in output
    assert "Palynomorph" in output
    csv_text = csv_output.read_text()
    assert "filename,label,artefact_type,precision" in csv_text
    assert "sample,Palynomorph,Text,1.0" in csv_text


def test_cli_import_does_not_import_ml_backends():
    script = (
        "import sys; import collectra.main; "
        "print(*(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False False"
    assert result.stderr == ""


def test_model_task_class_discovery_does_not_import_backends():
    script = (
        "import sys; "
        "from collectra.tasks.object_detection.yolo import ObjectDetectionYOLO; "
        "from collectra.tasks.image_classifier.yolo import ImageClassifierYOLO; "
        "from collectra.tasks.object_detection.detr import ObjectDetectionDETR; "
        "from collectra.tasks.object_detection.rfdetr import ObjectDetectionRFDETR; "
        "from collectra.tasks.machine_learning.orienters import ImageOrienter; "
        "print(*(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False False"
    assert result.stderr == ""


def test_pipeline_help_with_model_tasks_does_not_import_backends(tmp_path):
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        "Lazy:\n"
        "  type: collectra.Pipeline\n"
        "  ext: collectra\n"
        "  version: 1.0.0\n"
        "detector:\n"
        "  type: collectra.ObjectDetectionYOLO\n"
        "  model: model.pt\n"
        "rfdetr_detector:\n"
        "  type: collectra.ObjectDetectionRFDETR\n"
        "  model: model.pth\n"
    )
    script = (
        "import sys; import cappa; from collectra.main import main; "
        f"path = {str(pipeline_file)!r}; "
        "\ntry: main(['--pipeline', path, '--help'])"
        "\nexcept cappa.HelpExit: pass"
        "\nprint('BACKENDS', *(name in sys.modules for name in "
        "('torch', 'torchvision', 'ultralytics', 'rfdetr', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "BACKENDS False False False False False" in result.stdout
    assert result.stderr == ""


def test_training_is_owned_by_concrete_tasks():
    assert not hasattr(Pipeline, "train")
    assert not hasattr(Pipeline, "save_train")
    assert "MachineLearningTask" not in collectra.__all__
    assert not hasattr(collectra, "MachineLearningTask")


def test_model_task_classes_have_domain_specific_module_paths():
    from collectra import (
        ImageClassifierYOLO,
        ObjectDetectionDETR,
        ObjectDetectionRFDETR,
        ObjectDetectionYOLO,
    )

    assert ObjectDetectionYOLO.__module__ == "collectra.tasks.object_detection.yolo"
    assert ObjectDetectionDETR.__module__ == "collectra.tasks.object_detection.detr"
    assert ObjectDetectionRFDETR.__module__ == "collectra.tasks.object_detection.rfdetr"
    assert ImageClassifierYOLO.__module__ == "collectra.tasks.image_classifier.yolo"


def test_yolo_task_types_are_siblings():
    from collectra import ImageClassifierYOLO, ObjectDetectionYOLO, YOLOTask

    assert issubclass(ObjectDetectionYOLO, YOLOTask)
    assert issubclass(ImageClassifierYOLO, YOLOTask)
    assert not issubclass(ImageClassifierYOLO, ObjectDetectionYOLO)


def test_concrete_model_tasks_own_typed_train_commands(capsys):
    from collectra import ImageClassifierYOLO, ObjectDetectionDETR

    app = RuntimeApplication(
        {
            "classifier": ImageClassifierYOLO("classifier", model="model.pt"),
            "detector": ObjectDetectionDETR("detector", model="default"),
        }
    )
    classifier_help = help_for(app, ["task", "classifier", "train", "--help"], capsys)
    detector_help = help_for(app, ["task", "detector", "train", "--help"], capsys)

    assert "--imgsz" in classifier_help
    assert "--model" in classifier_help
    assert "--output" in classifier_help
    assert "--min-size" in classifier_help
    assert "--learning-rate" in classifier_help
    assert "--model-name" not in classifier_help
    assert "--model-name" in detector_help
    assert "--output" in detector_help
    assert "--weight-decay" in detector_help


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("with_inputs", [False, True])
@pytest.mark.parametrize("max_pixels", [None, 2048])
def test_pipeline_gui_opens_selected_workflow(
    tmp_path, monkeypatch, debug, with_inputs, max_pixels
):
    from unittest.mock import MagicMock

    from collectra.gui import backend

    pipeline_dir = tmp_path / "example.collectra"
    pipeline_dir.mkdir()
    (pipeline_dir / "pipeline.yaml").write_text(
        "collectra_pipeline_metadata:\n"
        "  name: example\n"
        "  ext: example\n"
        "  version: '1.0'\n"
        "  theme: custom.css\n"
    )
    create_window = MagicMock()
    start_webview = MagicMock()
    monkeypatch.setattr(backend.webview, "create_window", create_window)
    monkeypatch.setattr(backend.webview, "start", start_webview)

    inputs = []
    if with_inputs:
        for name in ("first.example", "second.example"):
            folder = tmp_path / name
            folder.mkdir()
            (folder / "results.yaml").write_text(
                "text: {type: collectra.Text, id: text, data: hello}"
            )
            inputs.append(str(folder / "results.yaml"))
    main(
        [
            "--pipeline",
            str(pipeline_dir),
            "gui",
            *inputs,
            *(["--debug"] if debug else []),
            *(["--max-pixels", str(max_pixels)] if max_pixels else []),
        ]
    )

    create_window.assert_called_once()
    api = create_window.call_args.kwargs["js_api"]
    assert api._max_pixels == (max_pixels or 4096)
    assert api._pipeline.path == pipeline_dir.resolve()
    assert api._extension == "example"
    initial = api.get_initial_items()
    assert initial["provided"] is with_inputs
    if with_inputs:
        assert [folder["name"] for folder in initial["folders"]] == [
            "first.example",
            "second.example",
        ]
    assert api._pipeline.pipeline_metadata["theme"] == "custom.css"
    assert api.get_pipeline_graph() == {
        "success": True,
        "nodes": [],
        "edges": [],
        "ext": "example",
    }
    start_webview.assert_called_once()
    assert start_webview.call_args.kwargs["debug"] is debug
