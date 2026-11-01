"""The three object detectors share their common settings through one parent."""

import warnings
from types import ModuleType
import sys

import pytest
from pydantic import ValidationError

import collectra
from collectra import ObjectDetectionDETR, ObjectDetectionRFDETR, ObjectDetectionYOLO
from collectra.tasks.object_detection.base import ObjectDetectionTask

DETECTORS = [ObjectDetectionYOLO, ObjectDetectionRFDETR, ObjectDetectionDETR]
MODULES = {
    ObjectDetectionYOLO: "yolo",
    ObjectDetectionRFDETR: "rfdetr",
    ObjectDetectionDETR: "detr",
}


# --- the parent -------------------------------------------------------------------------


def test_the_parent_is_exported_and_all_three_detectors_inherit_from_it():
    assert collectra.ObjectDetectionTask is ObjectDetectionTask
    for detector in DETECTORS:
        assert issubclass(detector, ObjectDetectionTask)


@pytest.mark.parametrize("detector", DETECTORS)
def test_the_shared_defaults(detector):
    task = detector("d")
    assert task.resolution == 1280
    assert task.threshold == 0.25
    assert task.epochs == 100


def test_backends_keep_the_defaults_that_only_make_sense_for_them():
    assert (ObjectDetectionYOLO("d").batch, ObjectDetectionYOLO("d").learning_rate) == (
        16,
        0.01,
    )
    assert ObjectDetectionYOLO("d").weight_decay == 0.0005
    assert ObjectDetectionYOLO("d").iou == 0.7
    assert (
        ObjectDetectionRFDETR("d").batch,
        ObjectDetectionRFDETR("d").learning_rate,
    ) == (
        4,
        1e-4,
    )
    assert (ObjectDetectionDETR("d").batch, ObjectDetectionDETR("d").learning_rate) == (
        2,
        1e-4,
    )


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "bad",
    [
        {"threshold": 1.5},
        {"threshold": -0.1},
        {"resolution": 0},
        {"epochs": 0},
        {"batch": 0},
    ],
)
def test_settings_are_validated(detector, bad):
    with pytest.raises(ValidationError):
        detector("d", **bad)


@pytest.mark.parametrize("detector", DETECTORS)
def test_settings_are_validated_when_assigned_too(detector):
    task = detector("d")
    task.resolution = 640
    assert task.resolution == 640
    with pytest.raises(ValidationError):
        task.threshold = 2


# --- the resolution used when running ----------------------------------------------------


@pytest.mark.parametrize("detector", DETECTORS)
def test_running_uses_the_models_own_resolution_unless_one_was_chosen(detector):
    assert detector("d").inference_resolution is None
    assert detector("d", resolution=640).inference_resolution == 640
    task = detector("d")
    task.resolution = 960
    assert task.inference_resolution == 960


# --- old names ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector", DETECTORS)
def test_imgsz_is_still_accepted_as_resolution_with_a_warning(detector):
    with pytest.warns(FutureWarning, match="imgsz.*deprecated.*resolution"):
        task = detector("d", imgsz=640)
    assert task.resolution == 640
    assert task.inference_resolution == 640
    assert not hasattr(task, "imgsz") or "imgsz" not in (task.model_extra or {})


@pytest.mark.parametrize("detector", DETECTORS)
def test_giving_both_names_is_an_error(detector):
    with pytest.raises(ValueError, match="imgsz.*resolution"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector("d", imgsz=640, resolution=320)


def test_lr0_is_still_accepted_by_yolo_as_learning_rate():
    with pytest.warns(FutureWarning, match="lr0.*deprecated.*learning_rate"):
        task = ObjectDetectionYOLO("d", lr0=0.02)
    assert task.learning_rate == 0.02


# --- unknown settings --------------------------------------------------------------------


@pytest.mark.parametrize("detector", DETECTORS)
def test_an_unknown_setting_is_reported_not_silently_ignored(detector):
    with pytest.warns(UserWarning, match="epohcs") as caught:
        detector("labels", epohcs=3)
    message = str(caught[0].message)
    assert "'labels'" in message and "epochs" in message and "resolution" in message


@pytest.mark.parametrize("detector", DETECTORS)
def test_known_settings_and_pipeline_wiring_raise_no_warning(detector):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector(
            "d",
            input="page",
            output=["a", "b"],
            resolution=1280,
            threshold=0.3,
            epochs=5,
            batch_size=2,
        )


# --- running: YOLO -----------------------------------------------------------------------


def test_yolo_inference_uses_the_shared_threshold_and_its_own_iou(image, monkeypatch):
    from tests.unit.test_yolo_task import FakeInferenceModel

    fake_ops = ModuleType("torchvision.ops")
    fake_ops.batched_nms = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))
    monkeypatch.setitem(sys.modules, "torchvision.ops", fake_ops)
    calls = []

    class Recording(FakeInferenceModel):
        def __init__(self, path):
            pass

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return super().__call__(*args, **kwargs)

    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = Recording
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)
    task = ObjectDetectionYOLO(
        "d", model="m.pt", threshold=0.4, iou=0.5, resolution=960
    )
    task.run(image)
    assert calls == [{"iou": 0.5, "conf": 0.4, "imgsz": 960}]


# --- training: each backend resolves the shared settings from the task --------------------


def capture(monkeypatch, detector):
    seen = {}

    def fake(task, inputs, trainer, **kwargs):
        seen.update(kwargs)
        return "trained"

    monkeypatch.setattr(
        f"collectra.tasks.object_detection.{MODULES[detector]}.run_training_command",
        fake,
    )
    return seen


NAMES = {
    ObjectDetectionYOLO: dict(
        resolution="imgsz", learning_rate="lr0", weight_decay="weight_decay"
    ),
    ObjectDetectionRFDETR: dict(
        resolution="resolution", learning_rate="lr", weight_decay="weight_decay"
    ),
    ObjectDetectionDETR: dict(
        resolution=None, learning_rate="lr", weight_decay="weight_decay"
    ),
}


@pytest.mark.parametrize("detector", DETECTORS)
def test_training_takes_its_settings_from_the_task(detector, monkeypatch):
    seen = capture(monkeypatch, detector)
    task = detector(
        "d", epochs=7, batch=3, learning_rate=0.5, weight_decay=0.25, resolution=640
    )
    assert task.train(["x"]) == "trained"
    names = NAMES[detector]
    assert seen["epochs"] == 7 and seen["batch"] == 3
    assert seen[names["learning_rate"]] == 0.5
    assert seen[names["weight_decay"]] == 0.25
    if names["resolution"]:
        assert seen[names["resolution"]] == 640


@pytest.mark.parametrize("detector", DETECTORS)
def test_training_uses_the_default_when_nothing_was_set(detector, monkeypatch):
    seen = capture(monkeypatch, detector)
    detector("d").train(["x"])
    assert seen["epochs"] == 100
    resolution = NAMES[detector]["resolution"]
    if resolution:
        assert seen[resolution] == 1280


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_value_given_to_train_beats_the_task(detector, monkeypatch):
    seen = capture(monkeypatch, detector)
    task = detector("d", epochs=7, learning_rate=0.5)
    task.train(["x"], epochs=2, learning_rate=0.1, resolution=320)
    names = NAMES[detector]
    assert seen["epochs"] == 2 and seen[names["learning_rate"]] == 0.1
    if names["resolution"]:
        assert seen[names["resolution"]] == 320


def test_train_still_accepts_the_old_names_with_a_warning(monkeypatch):
    seen = capture(monkeypatch, ObjectDetectionYOLO)
    task = ObjectDetectionYOLO("d")
    with pytest.warns(FutureWarning, match="imgsz.*deprecated"):
        task.train(["x"], imgsz=256)
    assert seen["imgsz"] == 256
    with pytest.warns(FutureWarning, match="lr0.*deprecated"):
        task.train(["x"], lr0=0.3)
    assert seen["lr0"] == 0.3


def test_train_rejects_the_old_and_new_name_together(monkeypatch):
    capture(monkeypatch, ObjectDetectionYOLO)
    with pytest.raises(ValueError, match="imgsz.*resolution"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ObjectDetectionYOLO("d").train(["x"], imgsz=1, resolution=2)


# --- DETR: the resolution sets the size the image processor resizes to ---------------------


def test_detr_processor_size_follows_the_resolution(monkeypatch):
    seen = []

    class Processor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            seen.append((name, kwargs))
            return "processor"

    fake_transformers = ModuleType("transformers")
    fake_transformers.DetrImageProcessor = Processor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    task = ObjectDetectionDETR("d")
    assert task._load_processor("model", None) == "processor"
    task._load_processor("model", 640)
    assert seen == [
        ("model", {}),
        ("model", {"size": {"shortest_edge": 640, "longest_edge": 640}}),
    ]


# --- the shared code ------------------------------------------------------------------------


class Counting(ObjectDetectionTask):
    """A detector that predicts by returning the image's name."""

    def _init_model(self):
        pass

    def _predict_batch(self, images):
        self.batches.append([image.name for image in images])
        return [f"result-{image.name}" for image in images]


@pytest.fixture
def pictures(tmp_path):
    from PIL import Image as PILImage

    from collectra.types.images import Image

    def make(*names):
        found = []
        for name in names:
            path = tmp_path / f"{name}.png"
            PILImage.new("RGB", (4, 4)).save(path)
            found.append(Image(name=name, data=path))
        return found

    return make


def test_prefetch_predicts_in_batches_and_run_takes_the_cached_result(pictures):
    task = Counting("c", batch_size=2)
    task.__dict__["batches"] = []
    first, second, third = pictures("a", "b", "c")
    task.prefetch([[first], [second], [third], [first, second]])
    assert task.batches == [["a", "b"], ["c"]]
    assert task._take_prefetched(first) == "result-a"
    assert task._take_prefetched(first) is None  # taken once
    task.clear_prefetch()
    assert task._take_prefetched(second) is None


def test_the_cache_is_per_resolution(pictures):
    task = Counting("c", batch_size=2)
    task.__dict__["batches"] = []
    (image,) = pictures("a")
    task.prefetch([[image]])
    task.resolution = 640  # chosen after prefetching: the cached result no longer fits
    assert task._take_prefetched(image) is None


def test_a_backend_without_batching_ignores_prefetch(pictures):
    task = ObjectDetectionDETR("d", batch_size=4)
    task.prefetch([pictures("a")])  # does nothing, and does not load a model


def test_the_parent_leaves_prediction_and_loading_to_the_backend():
    parent = ObjectDetectionTask("p")
    with pytest.raises(NotImplementedError):
        parent._predict_batch([])
    with pytest.raises(NotImplementedError):
        parent._init_model()


def test_a_device_is_chosen_for_the_machine(monkeypatch):
    task = ObjectDetectionYOLO("d")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert task._select_device() == "mps"
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert task._select_device() == "cuda"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert task._select_device() == "cpu"


def test_something_that_is_not_a_mapping_is_left_for_pydantic_to_reject():
    with pytest.raises(ValidationError):
        ObjectDetectionYOLO.model_validate("not a task")
