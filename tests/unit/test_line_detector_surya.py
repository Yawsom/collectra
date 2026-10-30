import sys
from types import ModuleType, SimpleNamespace

import cappa
import pytest

from collectra.cli import invoke
from collectra.tasks.ocr import LineDetectorSurya


class FakeDetectionPredictor:
    def __init__(self, boxes=()):
        self.boxes = boxes

    def __call__(self, images):
        assert len(images) == 1
        bboxes = [SimpleNamespace(bbox=box) for box in self.boxes]
        return [SimpleNamespace(bboxes=bboxes)]


def detector(boxes=(), **kwargs):
    task = LineDetectorSurya("lines")
    task.merge_horizontal = kwargs.get("merge_horizontal", False)
    task.min_height = kwargs.get("min_height", 0)
    task.detection_predictor = FakeDetectionPredictor(boxes)
    return task


def test_detector_uses_current_positional_task_contract(image):
    task = detector(([1, 20, 30, 28], [2, 5, 25, 12]))

    crops = task.run(image)

    assert len(crops) == 2
    assert all(crop.name == image.name for crop in crops)


def test_detector_returns_whole_image_crop_when_no_lines(image):
    crops = detector().run(image)

    assert len(crops) == 1
    assert crops[0].width == image.width
    assert crops[0].height == image.height


def test_detector_rejects_non_image_input():
    with pytest.raises(TypeError, match="image must be Image"):
        detector().run("not-an-image")


def test_detector_accepts_pipeline_io_configuration_without_loading_model(monkeypatch):
    detection = ModuleType("surya.detection")
    detection.DetectionPredictor = type("DetectionPredictor", (), {})
    monkeypatch.setitem(sys.modules, "surya.detection", detection)

    task = LineDetectorSurya(
        "detect_lines",
        input="apparatus_image",
        output="apparatus_lines",
        merge_horizontal=True,
    )

    assert task.input == "apparatus_image"
    assert task.output == "apparatus_lines"
    assert task.merge_horizontal is True
    assert task.detection_predictor is None


def test_detector_exposes_inherited_run_command(capsys):
    with pytest.raises(cappa.HelpExit):
        invoke(detector(), ["--help"], name="lines")

    output = capsys.readouterr().out
    assert "run" in output
