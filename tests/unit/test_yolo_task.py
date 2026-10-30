import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from collectra import ImageCrop, ObjectDetectionYOLO
from collectra.cli import invoke


class FakeYOLO:
    model_name = "fake.pt"


def install_fake_yolo(monkeypatch):
    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)


def test_train_yolo_temp_dir(classes, image, tmp_path, monkeypatch):
    """Exercise YOLO training orchestration without weights or a backend."""
    install_fake_yolo(monkeypatch)
    task = ObjectDetectionYOLO(name="label-detector", model=FakeYOLO())
    crop = image.make_crop(0.5, 0.5, 0.25, 0.25, name="human")
    crop.add_source_parent(image)
    monkeypatch.setattr(task, "_init_model", lambda: None)
    monkeypatch.setattr(task, "_prepare_assets", lambda *args, **kwargs: (["train"], ["val"]))

    def fake_train_fold(train, validation, output_classes, log, kwargs):
        weights = log / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").touch()
        return SimpleNamespace(save_dir=log, results_dict={"mock": True})

    monkeypatch.setattr(task, "_train_fold", fake_train_fold)
    result = task._train(
        crop,
        classes=classes,
        log="run",
        base_folder=tmp_path,
    )

    assert result.results_dict == {"mock": True}
    assert (result.save_dir / "weights" / "best.pt").exists()


class Scalar:
    def __init__(self, value):
        self.value = value

    def int(self):
        return self

    def item(self):
        return self.value


class Coordinates(list):
    def clone(self):
        return self


class Boxes:
    xywhn = Coordinates([[0.5, 0.5, 0.25, 0.25]])
    cls = [Scalar(0)]
    conf = [Scalar(0.9)]

    def __len__(self):
        return 1


class FakeInferenceModel:
    model = SimpleNamespace(end2end=False)

    def __call__(self, *args, **kwargs):
        return [SimpleNamespace(boxes=Boxes(), names={0: "human"})]


@pytest.mark.parametrize("model", [None, "/models/override.pt", Path("override.pt")])
@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    ("configured_imgsz", "imgsz", "expected_imgsz"),
    [(None, None, None), (None, 1280, 1280), (960, None, 960), (960, 1280, 1280)],
)
def test_run_yolo(
    image, monkeypatch, model, cropped, configured_imgsz, imgsz, expected_imgsz
):
    """Exercise YOLO result conversion without importing Ultralytics."""
    fake_torchvision = ModuleType("torchvision")
    fake_ops = ModuleType("torchvision.ops")
    fake_ops.batched_nms = lambda *args, **kwargs: []
    fake_torchvision.ops = fake_ops
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.ops", fake_ops)

    loaded_paths = []
    inference_calls = []

    class LoadingModel(FakeInferenceModel):
        def __init__(self, path):
            loaded_paths.append(path)

        def __call__(self, *args, **kwargs):
            inference_calls.append(kwargs)
            return super().__call__(*args, **kwargs)

    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = LoadingModel
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)
    task = ObjectDetectionYOLO(name="label-detector", model="configured.pt")
    task.imgsz = configured_imgsz
    if cropped:
        image = image.make_crop(0.5, 0.5, 0.5, 0.5)
    detections = task.run(image, model=model, imgsz=imgsz)

    expected_kwargs = {"iou": 0.7, "conf": 0.25}
    if expected_imgsz is not None:
        expected_kwargs["imgsz"] = expected_imgsz
    assert inference_calls == [expected_kwargs]
    assert loaded_paths == [Path(model or "configured.pt")]
    assert len(detections) == 1
    assert isinstance(detections[0], ImageCrop)
    assert detections[0].name == "human"
    assert detections[0].confidence == 0.9


@pytest.mark.parametrize("model", [None, "/models/override.pt"])
@pytest.mark.parametrize("imgsz", [None, 1280])
def test_run_cli_model_override(model, imgsz):
    task = ObjectDetectionYOLO("detector", model="configured.pt")
    calls = []
    task.pipeline = SimpleNamespace(
        cli_run=lambda inputs, **kwargs: calls.append(
            (inputs, kwargs, task.model, task.imgsz)
        )
    )
    argv = ["run", "image.jpg", "--output", "results", "--render"]
    if model is not None:
        argv.extend(["--model", model])
    if imgsz is not None:
        argv.extend(["--imgsz", str(imgsz)])

    invoke(task, argv)

    assert calls == [
        (
            ["image.jpg"],
            dict(
                task="detector",
                output=Path("results"),
                single=True,
                verbose=False,
                usage=False,
                render=True,
                force=False,
            ),
            Path(model) if model is not None else "configured.pt",
            imgsz,
        )
    ]


def test_prefetch_batches_the_model_and_run_reuses_results(image, monkeypatch):
    fake_torchvision = ModuleType("torchvision")
    fake_ops = ModuleType("torchvision.ops")
    fake_ops.batched_nms = lambda *args, **kwargs: []
    fake_torchvision.ops = fake_ops
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.ops", fake_ops)

    calls = []

    class BatchModel(FakeInferenceModel):
        def __init__(self, path):
            pass

        def __call__(self, sources, **kwargs):
            calls.append((len(sources), kwargs.get("batch")))
            return [
                SimpleNamespace(boxes=Boxes(), names={0: "human"}) for _ in sources
            ]

    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = BatchModel
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)
    task = ObjectDetectionYOLO(name="label-detector", model="configured.pt")
    task.batch_size = 2
    crops = [image.make_crop(0.5, 0.5, w, w) for w in (0.2, 0.3, 0.4)]

    task.prefetch([[crop] for crop in crops])
    assert calls == [(2, 2), (1, None)]  # 3 images -> batches of 2 and 1

    for crop in crops:
        assert len(task.run(crop)) == 1
    assert calls == [(2, 2), (1, None)]  # run() used the cache, no new model calls

    task.clear_prefetch()
    task.run(crops[0])
    assert len(calls) == 3  # cache cleared, so the model is called again
