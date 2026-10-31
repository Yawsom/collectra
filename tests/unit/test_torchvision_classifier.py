"""Torchvision classification tests without pretrained-weight downloads."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cappa
import pytest
from PIL import Image as PILImage

from collectra import Image, ImageClassifierTorchvision, Link
from collectra.cli import invoke
from collectra.tasks.image_classifier.torchvision import (
    _disable_auxiliary_heads,
    _replace_head,
)


@pytest.fixture
def torch_backend():
    import torch
    from torchvision import models

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch, models
    torch.set_num_threads(threads)


@pytest.fixture
def tiny_backend(torch_backend, monkeypatch):
    torch, models = torch_backend
    from torchvision.transforms import InterpolationMode

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU()
            )
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 1000)

        def forward(self, x):
            return self.fc(self.pool(self.features(x)).flatten(1))

    weights = SimpleNamespace(
        meta={"categories": [str(index) for index in range(1000)]},
        transforms=lambda: SimpleNamespace(
            crop_size=[8],
            resize_size=[10],
            mean=[0.5] * 3,
            std=[0.5] * 3,
            interpolation=InterpolationMode.BILINEAR,
        ),
    )
    calls = []

    def get_model(name, **kwargs):
        assert name == "resnet18"
        calls.append(kwargs)
        return TinyClassifier()

    monkeypatch.setattr(models, "get_model", get_model)
    monkeypatch.setattr(
        models, "get_model_weights", lambda name: SimpleNamespace(DEFAULT=weights)
    )
    return calls, weights


def make_inputs(tmp_path):
    samples = []
    for index, (name, partition, color) in enumerate(
        [
            ("Pollen", "train", "red"),
            ("Spore", "train", "blue"),
            ("Pollen", "validation", "red"),
            ("Spore", "validation", "blue"),
            ("Pollen", "excluded", "green"),
        ]
    ):
        path = tmp_path / f"image{index}.png"
        PILImage.new("RGB", (16, 12), color).save(path)
        image = Image(
            name="specimen", id=f"image{index}", data=path, partition=partition
        )
        crop = image.make_crop(0.5, 0.5, 0.5, 0.5, name="specimen")
        samples.append(
            Link(
                name=name,
                id=f"label{index}",
                target=crop,
                parents=[crop.id],
                partition=partition,
            )
        )
    return samples


@pytest.mark.parametrize(
    "name",
    [
        "alexnet",
        "convnext_tiny",
        "densenet121",
        "efficientnet_b0",
        "efficientnet_v2_s",
        "googlenet",
        "inception_v3",
        "maxvit_t",
        "mnasnet0_5",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "regnet_x_400mf",
        "resnet18",
        "resnext50_32x4d",
        "shufflenet_v2_x0_5",
        "squeezenet1_0",
        "swin_t",
        "vgg11",
        "vit_b_16",
        "wide_resnet50_2",
    ],
)
def test_replace_heads_for_classification_families(name, torch_backend):
    torch, models = torch_backend
    kwargs = {"init_weights": False} if name in {"googlenet", "inception_v3"} else {}
    # Meta tensors exercise the real architecture without allocating large backbones.
    # RegNet computes its block widths from tensor values during construction.
    with torch.device("cpu" if name.startswith("regnet_") else "meta"):
        model = models.get_model(name, weights=None, **kwargs)
        _disable_auxiliary_heads(model)
        head = _replace_head(model, 3)
    assert getattr(head, "out_features", getattr(head, "out_channels", None)) == 3
    assert not getattr(model, "aux_logits", False)


def test_extracts_crops_into_separate_splits(tmp_path):
    samples = make_inputs(tmp_path)
    task = ImageClassifierTorchvision("classifier")
    train, val = task._prepare_assets(
        ["Pollen", "Spore"],
        tmp_path / "export",
        "validation",
        "excluded",
        *samples,
    )
    assert len(list(train.rglob("*.png"))) == 2
    assert len(list(val.rglob("*.png"))) == 2
    with PILImage.open(next((train / "Pollen").glob("*.png"))) as crop:
        assert crop.size == (8, 6)
        assert crop.getpixel((0, 0)) == (255, 0, 0)
    with pytest.raises(ValueError, match="not empty"):
        task._prepare_assets(
            ["Pollen", "Spore"], tmp_path / "export", "validation", "excluded", *samples
        )


def test_training_checkpoint_reload_and_prediction(
    tmp_path, tiny_backend, torch_backend
):
    torch, _ = torch_backend
    samples = make_inputs(tmp_path)
    task = ImageClassifierTorchvision("classifier", output=["Pollen", "Spore"])
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="run",
        classes=["Spore", "Pollen"],
        validation="validation",
        exclude="excluded",
        epochs=2,
        batch=2,
        # This tiny dataset does not need worker processes or external logging.
        workers=0,
        wandb=False,
        device="cpu",
        fliplr=0,
        freeze_backbone=True,
    )
    calls, weights = tiny_backend
    assert calls[0]["weights"] is weights
    assert calls[-1]["weights"] is None
    assert result.results_dict["classes"] == ["Pollen", "Spore"]
    assert "val_accuracy" in result.results_dict
    checkpoint_path = result.save_dir / "weights" / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["state_dict"]["fc.weight"].shape[0] == 2
    assert (result.save_dir / "weights" / "last.pt").exists()
    assert (
        json.loads((result.save_dir / "metrics.json").read_text())
        == result.results_dict
    )
    assert len((result.save_dir / "history.csv").read_text().splitlines()) == 3
    reloaded = ImageClassifierTorchvision(
        "classifier", model=checkpoint_path, output=["Pollen", "Spore"], device="cpu"
    )
    image = samples[0].resolve()
    prediction = reloaded.run(image)
    assert type(prediction) is Link
    assert prediction.target is image
    assert prediction.name in {"Pollen", "Spore"}
    for key, value in checkpoint["state_dict"].items():
        assert torch.equal(reloaded.model.state_dict()[key], value)


def test_checkpoint_retraining_replaces_head_for_new_classes(
    tmp_path, tiny_backend, torch_backend
):
    torch, _ = torch_backend
    samples = make_inputs(tmp_path)
    task = ImageClassifierTorchvision("classifier")
    first = task._train(
        *samples,
        base_folder=tmp_path,
        log="first",
        exclude="excluded",
        epochs=1,
        batch=2,
        workers=0,
        wandb=False,
        device="cpu",
    )
    for sample in samples:
        sample.name = "NewClass"
    second = task._train(
        *samples,
        model=str(first.save_dir / "weights" / "best.pt"),
        base_folder=tmp_path,
        log="second",
        exclude="excluded",
        epochs=1,
        batch=2,
        workers=0,
        wandb=False,
        device="cpu",
    )
    assert second.results_dict["classes"] == ["NewClass"]
    assert task.model.fc.out_features == 1


def test_real_resnet_training_roundtrip(tmp_path, torch_backend):
    torch, _ = torch_backend
    samples = make_inputs(tmp_path)
    task = ImageClassifierTorchvision("classifier", model="resnet18")
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="real",
        validation="validation",
        exclude="excluded",
        epochs=1,
        batch=2,
        workers=0,
        wandb=False,
        pretrained=False,
        freeze_backbone=True,
        device="cpu",
        fliplr=0,
    )
    assert 0 <= result.results_dict["val_accuracy"] <= 1
    assert task.model.fc.out_features == 2
    assert task.run(samples[0].resolve()).name in {"Pollen", "Spore"}


def test_checkpoint_requires_architecture_and_classes(tmp_path, torch_backend):
    torch, _ = torch_backend
    path = tmp_path / "bare.pt"
    torch.save({"fc.weight": torch.ones(2, 3)}, path)
    task = ImageClassifierTorchvision("classifier", model=path)
    with pytest.raises(ValueError, match="bare state_dict"):
        task._load()


def test_unknown_model_is_rejected(torch_backend):
    task = ImageClassifierTorchvision("classifier", model="not-a-model.pt")
    with pytest.raises(ValueError, match="Unknown torchvision classification model"):
        task._load()


def test_training_help_and_forwarding(capsys, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "collectra.tasks.image_classifier.torchvision.run_training_command",
        lambda *args, **kwargs: calls.update(kwargs),
    )
    task = ImageClassifierTorchvision("classifier")
    with pytest.raises(cappa.HelpExit):
        invoke(task, ["train", "--help"], name="classifier")
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Torchvision classification model name" in help_text
    invoke(
        task,
        [
            "train",
            "data",
            "--model",
            "resnet50",
            "--no-pretrained",
            "--freeze-backbone",
        ],
        name="classifier",
    )
    assert calls["model"] == "resnet50"
    assert calls["pretrained"] is False
    assert calls["freeze_backbone"] is True


def test_import_and_help_do_not_load_ml_backends():
    code = (
        "import sys, cappa; from collectra import ImageClassifierTorchvision; "
        "from collectra.cli import invoke; "
        "task = ImageClassifierTorchvision('classifier')\n"
        "try: invoke(task, ['train', '--help'])\n"
        "except cappa.HelpExit: pass\n"
        "assert not any(name in sys.modules for name in ('torch', 'torchvision', 'ultralytics'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_training_persists_model_and_link_outputs(tmp_path, tiny_backend):
    import yaml
    from collectra import Pipeline
    from collectra.types.base import ArtefactNode

    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    pipeline_file = pipeline_dir / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "collectra_pipeline_metadata": {
                    "name": "test",
                    "ext": "sample",
                    "version": "1.0",
                },
                "image": {
                    "type": "collectra.Image",
                    "data": str(tmp_path / "data" / "0.png"),
                },
                "classifier": {
                    "type": "collectra.ImageClassifierTorchvision",
                    "model": "resnet18",
                    "input": ["image"],
                    "output": ["Pollen", "Spore"],
                },
            }
        )
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index, (name, partition) in enumerate(
        [
            ("Pollen", "train"),
            ("Spore", "train"),
            ("Pollen", "validation"),
            ("Spore", "validation"),
        ]
    ):
        path = data_dir / f"{index}.png"
        PILImage.new("RGB", (16, 12), "red" if name == "Pollen" else "blue").save(path)
        sample = data_dir / f"{index}.sample"
        sample.mkdir()
        (sample / "results.yaml").write_text(
            yaml.safe_dump(
                {
                    "collectra_results_metadata": {"partition": partition},
                    "image": {
                        "type": "collectra.Image",
                        "id": f"image{index}",
                        "data": str(path),
                    },
                    name: {
                        "type": "collectra.Link",
                        "id": f"label{index}",
                        "parents": f"image{index}",
                    },
                }
            )
        )
    pipeline = Pipeline.from_file(pipeline_file)
    pipeline.connect()
    for name in ("Pollen", "Spore"):
        node = pipeline.node_manager.resolve_node(name)
        assert isinstance(node, ArtefactNode)
        assert node.types == {Link}
    task = pipeline.node_manager.resolve_node("classifier").get_task()
    result = task.train(
        [str(data_dir)],
        output=tmp_path / "run",
        validation="validation",
        epochs=1,
        batch=2,
        workers=0,
        wandb=False,
        device="cpu",
        keep_log=False,
    )
    assert not result.save_dir.exists()
    saved = yaml.safe_load(pipeline_file.read_text())["classifier"]["model"]
    assert (pipeline_dir / saved).is_file()
    task._reload()
    assert task.original_model_path == (pipeline_dir / saved).resolve()
    assert (pipeline_dir / Path(saved).with_suffix(".classes.json")).is_file()
    reloaded = Pipeline.from_file(pipeline_file)
    reloaded.connect()
    classifier = reloaded.node_manager.resolve_node("classifier").get_task()
    image = Image(name="image", data=data_dir / "0.png")
    # Pipeline execution normally supplies the pipeline working directory.
    from collectra.utils import change_dir

    with change_dir(pipeline_dir):
        assert type(classifier.run(image)) is Link


def test_best_checkpoint_and_early_stopping(
    tmp_path, tiny_backend, torch_backend, monkeypatch
):
    torch, _ = torch_backend
    task = ImageClassifierTorchvision("classifier")
    train_epoch = 0

    def epoch(loader, optimizer=None):
        nonlocal train_epoch
        if optimizer is not None:
            train_epoch += 1
            with torch.no_grad():
                task.model.fc.bias.fill_(train_epoch)
        return {"loss": float(train_epoch), "accuracy": 0.5, "top5_accuracy": 1.0}

    monkeypatch.setattr(task, "_epoch", epoch)
    result = task._train(
        *make_inputs(tmp_path),
        base_folder=tmp_path,
        log="run",
        validation="validation",
        exclude="excluded",
        epochs=5,
        early_stop=1,
        batch=2,
        workers=0,
        wandb=False,
        device="cpu",
    )
    assert result.results_dict["epoch"] == 1
    assert result.results_dict["epochs_completed"] == 2
    assert torch.all(task.model.fc.bias == 1)
    last = torch.load(result.save_dir / "weights" / "last.pt", weights_only=True)
    assert torch.all(last["state_dict"]["fc.bias"] == 2)


def test_missing_validation_partition_is_reported(tmp_path, tiny_backend):
    task = ImageClassifierTorchvision("classifier")
    with pytest.raises(ValueError, match="No validation images match partition"):
        task._train(
            *make_inputs(tmp_path),
            base_folder=tmp_path,
            log="run",
            epochs=1,
            validation="typo",
            exclude="excluded",
            workers=0,
            wandb=False,
            device="cpu",
        )


def test_validation_can_omit_a_training_class(tmp_path, tiny_backend):
    samples = [
        sample
        for sample in make_inputs(tmp_path)
        if not (sample.name == "Pollen" and sample.partition == "validation")
    ]
    task = ImageClassifierTorchvision("classifier")
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="run",
        epochs=1,
        validation="validation",
        exclude="excluded",
        device="cpu",
        workers=0,
        wandb=False,
        batch=2,
    )
    assert result.results_dict["classes"] == ["Pollen", "Spore"]
    assert "val_loss" in result.results_dict


@pytest.mark.parametrize("enabled", [None, False, True])
def test_wandb_logging_requires_opt_in(tmp_path, tiny_backend, monkeypatch, enabled):
    # Test lifecycle wiring without importing the W&B SDK or creating a run.
    started, metrics, finished = [], [], []
    run = SimpleNamespace(log=metrics.append, finish=lambda: finished.append(True))

    def init(**kwargs):
        started.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    task = ImageClassifierTorchvision("classifier")
    options = {} if enabled is None else {"wandb": enabled}
    task._train(
        *make_inputs(tmp_path),
        base_folder=tmp_path,
        log="run",
        epochs=1,
        batch=2,
        workers=0,
        device="cpu",
        **options,
    )
    expected = 1 if enabled else 0
    assert len(started) == len(metrics) == len(finished) == expected
    if enabled:
        assert started[0]["name"] == "run"
        assert "train_loss" in metrics[0]


@pytest.mark.parametrize("mapping_batch", [False, True])
def test_epoch_metrics_cover_all_batches(torch_backend, monkeypatch, mapping_batch):
    from collections import UserDict

    torch, _ = torch_backend
    task = ImageClassifierTorchvision("classifier")
    task._device = torch.device("cpu")
    task._categories = ["a", "b"]
    logits = torch.tensor([[4., 0.], [4., 0.], [4., 0.], [0., 4.]])
    labels = torch.tensor([0, 0, 1, 1])
    batches = []
    for start, end in [(0, 3), (3, 4)]:
        pixels = logits[start:end]
        if mapping_batch:
            pixels = UserDict(pixel_values=pixels)
        batches.append((pixels, labels[start:end]))
    monkeypatch.setattr(
        task, "_forward",
        lambda pixels: pixels["pixel_values"] if mapping_batch else pixels,
    )

    metrics = task._epoch(batches)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(5 / 6)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(11 / 15)
    assert metrics["loss"] == pytest.approx(
        torch.nn.functional.cross_entropy(logits, labels).item()
    )


def test_zero_validation_accuracy_still_saves_best_checkpoint(
    tmp_path, tiny_backend, monkeypatch
):
    task = ImageClassifierTorchvision("classifier")
    monkeypatch.setattr(task, "_epoch", lambda *args: {
        "loss": 1.0, "accuracy": 0.0, "top5_accuracy": 1.0,
    })
    result = task._train(
        *make_inputs(tmp_path), base_folder=tmp_path, log="run", epochs=1,
        validation="validation", exclude="excluded", device="cpu", workers=0,
        wandb=False, batch=2,
    )
    assert (result.save_dir / "weights" / "best.pt").is_file()
    assert result.results_dict["val_accuracy"] == 0.0
