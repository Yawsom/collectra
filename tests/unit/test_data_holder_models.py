"""Small result and sample holders are pydantic models too."""

import pickle
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from collectra.editor.base import Editor
from collectra.evaluator.base import EvaluationReport, FileEvaluationResult
from collectra.pipelines.handlers import DataHandler, DirectoryHandler, ZipHandler
from collectra.tasks.image_classifier.huggingface import _ProcessorTransform
from collectra.tasks.image_classifier.torchvision import _NormalizeClassifierPixels
from collectra.tasks.image_classifier.training import ClassificationTrainResult
from collectra.tasks.image_segmentation.data import SegmentationSample
from collectra.tasks.object_detection.detr import DetectionTrainResult
from collectra.tasks.ocr_huggingface import OCRSample
from collectra.text.base import Text as LegacyText
from collectra.types.images import Image, ImageSegmentation


def test_every_holder_is_a_pydantic_model():
    for cls in (
        FileEvaluationResult, EvaluationReport, OCRSample, SegmentationSample,
        DetectionTrainResult, ClassificationTrainResult, _NormalizeClassifierPixels,
        _ProcessorTransform, DataHandler, DirectoryHandler, ZipHandler, Editor,
        LegacyText,
    ):  # fmt: skip
        assert issubclass(cls, BaseModel), cls


# --- training results ----------------------------------------------------------


@pytest.mark.parametrize("cls", [DetectionTrainResult, ClassificationTrainResult])
def test_train_results_take_their_fields_by_position_or_name(cls):
    by_position = cls("logs", {"status": "completed"})
    by_name = cls(save_dir="logs", results_dict={"status": "completed"})
    assert by_position == by_name
    assert by_position.save_dir == Path("logs")
    with pytest.raises(ValidationError):
        cls("logs", "not a dict")


# --- evaluation ----------------------------------------------------------------


def test_file_results_keep_their_metrics_mutable():
    result = FileEvaluationResult(filename="a", label_metrics={})
    result.label_metrics["human"] = {"precision": 1.0}
    assert result.label_metrics == {"human": {"precision": 1.0}}
    assert FileEvaluationResult("a", {}).filename == "a"


def test_a_report_builds_its_tables_from_the_aggregate():
    aggregate = {
        "collectra.ImageCrop": {
            "overall": {
                "precision": 0.5,
                "recall": 0.25,
                "f1_micro": 0.33,
                "total_files": 2,
            },
            "per_label": {"human": {"precision": 1.0, "support": 3}},
        },
        "collectra.Text": {"overall": {}},
    }
    report = EvaluationReport([FileEvaluationResult("a", {})], aggregate, 0.5)
    assert len(report.aggregate_tables) == 2
    assert len(report.tables) == 1
    assert report.threshold == 0.5
    assert "aggregate_tables" not in report.model_dump()
    assert "Table" not in repr(report)


def test_report_values_are_validated():
    with pytest.raises(ValidationError):
        EvaluationReport([], {}, "high")


# --- samples ---------------------------------------------------------------------


def test_samples_hold_the_very_same_artefacts(tmp_path, monkeypatch):
    from PIL import Image as PilImage

    monkeypatch.chdir(tmp_path)
    PilImage.new("RGB", (4, 4)).save("p.png")
    image = Image(name="p", id="p", data="p.png")
    sample = SegmentationSample(image, [])
    assert sample.image is image and sample.instances == []
    ocr = OCRSample(image, "text", "train")
    assert (ocr.text, ocr.partition) == ("text", "train")
    assert OCRSample(image, "text").partition == ""
    with pytest.raises(ValidationError):
        SegmentationSample("not an image", [])


# --- transforms used in DataLoader workers must stay picklable ----------------------


def test_the_normalising_transform_pickles_and_keeps_its_fields():
    transform = _NormalizeClassifierPixels([0.5, 0.5], [0.2, 0.2])
    clone = pickle.loads(pickle.dumps(transform))
    assert (clone.mean, clone.std) == ([0.5, 0.5], [0.2, 0.2])


def test_the_processor_transform_pickles_and_defaults_to_no_flip():
    transform = _ProcessorTransform(("a", "processor"), 0.25)
    clone = pickle.loads(pickle.dumps(transform))
    assert (clone.processor, clone.fliplr) == (("a", "processor"), 0.25)
    assert _ProcessorTransform(object).fliplr == 0.0


# --- handlers, editor and the legacy text container ---------------------------------


def test_handlers_take_their_pipeline_by_keyword_only():
    pipeline = object()
    for cls in (DirectoryHandler, ZipHandler):
        assert cls(pipeline=pipeline).pipeline is pipeline
        with pytest.raises(TypeError):
            cls(pipeline)
    with pytest.raises(NotImplementedError):
        DataHandler(pipeline=pipeline).save()


def test_the_editor_prepares_its_templates(tmp_path):
    editor = Editor(file=tmp_path)
    assert editor.file == tmp_path
    assert editor.env.get_template("imagebb.html") is not None


def test_the_legacy_text_container_still_works():
    text = LegacyText("hello")
    assert text.content == "hello"
    assert str(text) == "hello"
