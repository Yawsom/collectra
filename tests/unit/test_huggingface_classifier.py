"""Hugging Face classifier tests using local, tiny real Transformers models."""

import json
import subprocess
import sys
from pathlib import Path

import cappa
import pytest
from PIL import Image as PILImage

from collectra import Image, ImageClassifierHuggingFace, Link
from collectra.cli import invoke


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(threads)


@pytest.fixture
def local_model(tmp_path, backend):
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    config = ViTConfig(
        image_size=16,
        patch_size=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        num_labels=3,
        id2label={0: "old-a", 1: "old-b", 2: "old-c"},
        label2id={"old-a": 0, "old-b": 1, "old-c": 2},
    )
    directory = tmp_path / "original"
    ViTForImageClassification(config).save_pretrained(directory)
    ViTImageProcessor(size={"height": 16, "width": 16}).save_pretrained(directory)
    return directory


def make_samples(tmp_path):
    samples = []
    for index, (name, partition, color) in enumerate(
        [
            ("Pollen", "train", "red"),
            ("Spore", "train", "blue"),
            ("Pollen", "validation", "red"),
            ("Spore", "validation", "blue"),
            ("Spore", "excluded", "green"),
        ]
    ):
        path = tmp_path / f"image{index}.png"
        PILImage.new("RGB", (24, 20), color).save(path)
        image = Image(
            name="specimen", id=f"image{index}", data=path, partition=partition
        )
        crop = image.make_crop(0.5, 0.5, 0.5, 0.5, name="specimen")
        samples.append(
            Link(
                name=name,
                id=f"label{index}",
                parents=[crop.id],
                target=crop,
                partition=partition,
            )
        )
    return samples


def test_epoch_reports_forward_error_for_processor_batch(backend, monkeypatch):
    task = ImageClassifierHuggingFace("classifier")
    pixels = {"pixel_values": backend.zeros(1, 3, 16, 16)}
    labels = backend.zeros(1, dtype=backend.long)

    def fail_forward(inputs):
        raise RuntimeError("backend failure")

    monkeypatch.setattr(task, "_forward", fail_forward)
    with pytest.raises(RuntimeError, match="pixel_values.*backend failure") as error:
        task._epoch([(pixels, labels)])
    assert str(error.value.__cause__) == "backend failure"


def test_train_save_reload_and_native_export(tmp_path, local_model, backend):
    torch = backend
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    task = ImageClassifierHuggingFace(
        "classifier", model=local_model, output=["Pollen", "Spore"]
    )
    samples = make_samples(tmp_path)
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="run",
        validation="validation",
        exclude="excluded",
        epochs=2,
        batch=2,
        # Worker startup costs more than processing this tiny dataset.
        workers=0,
        wandb=False,
        device="cpu",
        freeze_backbone=True,
        fliplr=0,
    )
    assert result.results_dict["classes"] == ["Pollen", "Spore"]
    assert result.results_dict["epochs_completed"] == 2
    assert 0 <= result.results_dict["val_accuracy"] <= 1
    assert len(list((result.save_dir / "train").rglob("*.png"))) == 2
    assert len(list((result.save_dir / "val").rglob("*.png"))) == 2
    with PILImage.open(next((result.save_dir / "train").rglob("*.png"))) as crop:
        assert crop.size == (12, 10)
    assert (
        json.loads((result.save_dir / "metrics.json").read_text())
        == result.results_dict
    )
    assert len((result.save_dir / "history.csv").read_text().splitlines()) == 3
    path = result.save_dir / "weights" / "best.pt"
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["backend"] == "huggingface"
    assert checkpoint["classes"] == ["Pollen", "Spore"]
    assert checkpoint["state_dict"]["classifier.weight"].shape == (2, 16)
    assert (result.save_dir / "weights" / "last.pt").exists()

    reloaded = ImageClassifierHuggingFace(
        "classifier", model=path, output=["Pollen", "Spore"], device="cpu"
    )
    image = samples[0].resolve()
    prediction = reloaded.run(image)
    assert type(prediction) is Link
    assert prediction.target is image
    assert prediction.name in {"Pollen", "Spore"}
    for name, value in task.model.state_dict().items():
        assert torch.equal(reloaded.model.state_dict()[name], value)

    native_dir = result.save_dir / "weights" / "huggingface"
    native = AutoModelForImageClassification.from_pretrained(native_dir).eval()
    processor = AutoImageProcessor.from_pretrained(native_dir)
    with image.pil() as pixels, torch.inference_mode():
        native_logits = native(**processor(images=pixels, return_tensors="pt")).logits
        inputs = {
            key: tensor.unsqueeze(0)
            for key, tensor in reloaded._transforms()(pixels).items()
        }
        assert torch.allclose(reloaded._forward(inputs), native_logits)

    original = AutoModelForImageClassification.from_pretrained(local_model)
    for name, value in original.base_model.state_dict().items():
        assert torch.equal(task.model.base_model.state_dict()[name], value)


@pytest.mark.parametrize("classes", [["new-a", "new-b"], ["new-a", "new-b", "new-c"]])
def test_adapts_different_and_same_size_heads(local_model, backend, classes):
    torch = backend
    from transformers import AutoModelForImageClassification

    original = AutoModelForImageClassification.from_pretrained(local_model)
    task = ImageClassifierHuggingFace("classifier", model=local_model)
    task._prepare_training_model(classes, pretrained=True, freeze=False)
    assert task.model.config.id2label == dict(enumerate(classes))
    assert task.model.classifier.out_features == len(classes)
    assert all(parameter.requires_grad for parameter in task.model.parameters())
    for name, value in original.base_model.state_dict().items():
        assert torch.equal(task.model.base_model.state_dict()[name], value)
    assert not torch.equal(task.model.classifier.weight, original.classifier.weight)


def test_checkpoint_relabeling_preserves_backbone(tmp_path, local_model, backend):
    torch = backend
    task = ImageClassifierHuggingFace("classifier", model=local_model)
    task._load()
    original = {name: value.clone() for name, value in task.model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    torch.save(task._checkpoint(0, {}), path)
    reloaded = ImageClassifierHuggingFace("classifier", model=path)
    reloaded._prepare_training_model(["a", "b", "c"], pretrained=True, freeze=True)
    assert reloaded._categories == ["a", "b", "c"]
    assert not torch.equal(
        reloaded.model.classifier.weight, original["classifier.weight"]
    )
    for name, value in reloaded.model.state_dict().items():
        if name.startswith("vit."):
            assert torch.equal(value, original[name])
    assert not any(
        parameter.requires_grad for parameter in reloaded.model.base_model.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in reloaded.model.classifier.parameters()
    )
    # Reconfiguring an already loaded model also preserves the trained backbone.
    reloaded._prepare_training_model(["only"], pretrained=True, freeze=False)
    assert reloaded.model.classifier.out_features == 1
    for name, value in reloaded.model.state_dict().items():
        if name.startswith("vit."):
            assert torch.equal(value, original[name])


def test_hub_name_passes_pretrained_label_configuration(
    tmp_path, local_model, backend, monkeypatch
):
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    calls = {}
    real_config = AutoConfig.from_pretrained
    real_processor = AutoImageProcessor.from_pretrained
    real_model = AutoModelForImageClassification.from_pretrained

    def config(source, **kwargs):
        calls["config_source"] = source
        return real_config(local_model, **kwargs)

    def processor(source, **kwargs):
        calls["processor_source"] = source
        return real_processor(local_model, **kwargs)

    def model(source, **kwargs):
        calls["model_source"] = source
        calls["model_kwargs"] = kwargs
        return real_model(local_model, **kwargs)

    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", processor)
    monkeypatch.setattr(AutoModelForImageClassification, "from_pretrained", model)
    task = ImageClassifierHuggingFace("classifier", model="example/test-vit")
    task._prepare_training_model(["Pollen", "Spore"], pretrained=True, freeze=False)
    assert (
        calls["config_source"]
        == calls["processor_source"]
        == calls["model_source"]
        == "example/test-vit"
    )
    assert calls["model_kwargs"]["ignore_mismatched_sizes"] is True
    assert calls["model_kwargs"]["config"].id2label == {0: "Pollen", 1: "Spore"}
    calls.clear()
    task = ImageClassifierHuggingFace("classifier", model="example/test-vit")
    task._prepare_training_model(["Pollen", "Spore"], pretrained=False, freeze=False)
    assert "model_source" not in calls


def test_bare_weights_and_missing_paths_are_rejected(tmp_path, backend):
    path = tmp_path / "bare.pt"
    backend.save({"classifier.weight": backend.ones(2, 3)}, path)
    with pytest.raises(ValueError, match="bare weight file"):
        ImageClassifierHuggingFace("classifier", model=path)._load()
    with pytest.raises(ValueError, match="does not exist"):
        ImageClassifierHuggingFace("classifier", model=tmp_path / "missing")._load()


def test_help_and_forwarding(capsys, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "collectra.tasks.image_classifier.huggingface.run_training_command",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    task = ImageClassifierHuggingFace("classifier")
    with pytest.raises(cappa.HelpExit):
        invoke(task, ["train", "--help"])
    assert "Hugging Face model ID" in " ".join(capsys.readouterr().out.split())
    invoke(
        task,
        [
            "train",
            "data",
            "--model",
            "example/test-vit",
            "--no-pretrained",
            "--freeze-backbone",
        ],
    )
    assert captured["model"] == "example/test-vit"
    assert captured["pretrained"] is False
    assert captured["freeze_backbone"] is True


def test_import_and_help_are_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, cappa; from collectra import ImageClassifierHuggingFace; "
                "from collectra.cli import invoke; task = ImageClassifierHuggingFace('classifier')\n"
                "try: invoke(task, ['train', '--help'])\n"
                "except cappa.HelpExit: pass\n"
                "assert not any(name in sys.modules for name in ('torch', 'torchvision', 'transformers', 'ultralytics'))\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_save_and_reload_after_log_cleanup(tmp_path, local_model):
    import yaml
    from collectra import Pipeline
    from collectra.types.base import ArtefactNode
    from collectra.utils import change_dir

    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
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
        PILImage.new("RGB", (20, 20), "red" if name == "Pollen" else "blue").save(path)
        sample_dir = data_dir / f"{index}.sample"
        sample_dir.mkdir()
        (sample_dir / "results.yaml").write_text(
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
    pipeline_file = pipeline_dir / "pipeline.yaml"
    pipeline_file.write_text(
        yaml.safe_dump(
            {
                "collectra_pipeline_metadata": {
                    "name": "test",
                    "ext": "sample",
                    "version": "1.0",
                },
                "image": {"type": "collectra.Image", "data": str(data_dir / "0.png")},
                "classifier": {
                    "type": "collectra.ImageClassifierHuggingFace",
                    "model": str(local_model),
                    "input": ["image"],
                    "output": ["Pollen", "Spore"],
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
    assert (pipeline_dir / Path(saved).with_suffix(".classes.json")).is_file()
    task._reload()
    reloaded = Pipeline.from_file(pipeline_file)
    reloaded.connect()
    classifier = reloaded.node_manager.resolve_node("classifier").get_task()
    image = Image(name="image", data=data_dir / "0.png")
    with change_dir(pipeline_dir):
        prediction = classifier.run(image)
    assert type(prediction) is Link
    assert prediction.target is image
    assert prediction.name in {"Pollen", "Spore"}
