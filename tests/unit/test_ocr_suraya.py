import sys
from types import ModuleType, SimpleNamespace

import cappa
import pytest

from collectra.cli import invoke
from collectra.tasks.ocr import OCRSuraya
from collectra.types.texts import Text


class FakeRecognitionPredictor:
    def __init__(self, foundation_predictor=None, lines=("first", "second")):
        self.foundation_predictor = foundation_predictor
        self.lines = lines
        self.calls = []

    def __call__(self, images, det_predictor):
        self.calls.append((images, det_predictor))
        text_lines = [SimpleNamespace(text=line) for line in self.lines]
        return [SimpleNamespace(text_lines=text_lines)]


def ocr(lines=("first", "second")):
    task = OCRSuraya("ocr")
    task.detection_predictor = object()
    task.recognition_predictor = FakeRecognitionPredictor(lines=lines)
    return task


def test_ocr_uses_current_positional_task_contract(image):
    task = ocr()

    results = task.run(image)

    assert len(results) == 1
    assert isinstance(results[0], Text)
    assert results[0].name == "ocr_output"
    assert results[0].data == "first\nsecond"
    assert task.recognition_predictor.calls[0][1] is task.detection_predictor


def test_ocr_returns_one_text_artefact_per_image(image):
    results = ocr(("text",)).run(image, image)

    assert [result.data for result in results] == ["text", "text"]


def test_ocr_rejects_non_image_input():
    with pytest.raises(TypeError, match="image must be Image"):
        ocr().run("not-an-image")


def test_ocr_initializes_surya_predictors_lazily(monkeypatch, image):
    detection = ModuleType("surya.detection")
    foundation = ModuleType("surya.foundation")
    recognition = ModuleType("surya.recognition")
    detection.DetectionPredictor = type("DetectionPredictor", (), {})
    foundation.FoundationPredictor = type("FoundationPredictor", (), {})
    recognition.RecognitionPredictor = FakeRecognitionPredictor
    monkeypatch.setitem(sys.modules, "surya.detection", detection)
    monkeypatch.setitem(sys.modules, "surya.foundation", foundation)
    monkeypatch.setitem(sys.modules, "surya.recognition", recognition)

    task = OCRSuraya("ocr", input="page", output="transcription")

    assert task.input == "page"
    assert task.output == "transcription"
    assert task.detection_predictor is None
    assert task.recognition_predictor is None

    task.run(image)

    assert isinstance(task.detection_predictor, detection.DetectionPredictor)
    assert isinstance(task.recognition_predictor, FakeRecognitionPredictor)
    assert isinstance(
        task.recognition_predictor.foundation_predictor,
        foundation.FoundationPredictor,
    )


def test_ocr_exposes_inherited_run_command(capsys):
    with pytest.raises(cappa.HelpExit):
        invoke(ocr(), ["--help"], name="ocr")

    assert "run" in capsys.readouterr().out
