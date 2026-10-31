from copy import deepcopy

import numpy as np
import pytest
import tifffile

from collectra import Image, ImageClassifierTorchvision, Link
from collectra.tasks.image_classifier.torchvision_fusion import RGBViewFusionClassifier


@pytest.fixture(autouse=True)
def single_thread():
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("architecture", ["resnet18", "convnext_tiny"])
def test_real_rgb_encoder_preserved_and_shared_across_views(architecture):
    import torch
    from torchvision import models

    original = models.get_model(architecture, weights=None).eval()
    reference = deepcopy(original)
    model = RGBViewFusionClassifier(original, 9).eval()
    pixels = torch.randn(2, 9, 32, 32)
    observed = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].shape)
    )
    with torch.no_grad():
        torch.testing.assert_close(
            model(pixels), reference(pixels[:, :3]), rtol=1e-4, atol=1e-5
        )
    handle.remove()
    assert observed == [torch.Size([6, 3, 32, 32])]
    first = next(
        layer for layer in model.encoder.modules() if isinstance(layer, torch.nn.Conv2d)
    )
    assert first.in_channels == 3
    assert sum(p.numel() for p in model.encoder.parameters()) + sum(
        p.numel() for p in model.head.parameters()
    ) == sum(p.numel() for p in reference.parameters())


def test_attention_learns_from_every_view_and_freezing_preserves_backbone():
    import torch
    from torchvision import models

    task = ImageClassifierTorchvision(
        "classifier", model=models.resnet18(weights=None), fusion=True
    )
    task._categories = [str(index) for index in range(1000)]
    task._input_channels = 9
    task._prepare_training_model(["a", "b"], pretrained=False, freeze=True)
    model = task.model.eval()
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.attention.parameters())
    assert model.gate.requires_grad and model.view_bias.requires_grad
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    with torch.no_grad():
        model.gate.fill_(0.5)
    pixels = torch.randn(2, 9, 32, 32, requires_grad=True)
    model(pixels).square().mean().backward()
    for view in range(3):
        assert pixels.grad[:, view * 3 : (view + 1) * 3].abs().sum() > 0
    assert model.attention[-1].weight.grad.abs().sum() > 0
    assert model.view_bias.grad.abs().sum() > 0
    assert model.gate.grad.abs() > 0


def test_fusion_checkpoint_roundtrip_and_new_class_head(tmp_path):
    import torch
    from torchvision import models

    task = ImageClassifierTorchvision(
        "classifier", model=models.resnet18(weights=None), fusion=True
    )
    task._architecture = "resnet18"
    task._input_channels = 6
    task._categories = [str(index) for index in range(1000)]
    task._preprocessing = {
        "resize_size": [32],
        "crop_size": [32],
        "mean": [0.5] * 6,
        "std": [0.5] * 6,
        "interpolation": "bilinear",
    }
    task._prepare_training_model(["a", "b"], pretrained=False, freeze=False)
    task.model.eval()
    with torch.no_grad():
        task.model.gate.fill_(0.7)
        task.model.view_bias[1] = 0.6
    pixels = torch.randn(2, 6, 32, 32)
    expected = task.model(pixels).detach()
    checkpoint = tmp_path / "fusion.pt"
    torch.save(task._checkpoint(1, {}), checkpoint)
    loaded = ImageClassifierTorchvision("classifier", model=checkpoint)
    loaded._load()
    assert loaded.fusion is True
    assert loaded.model.fusion_spec == task.model.fusion_spec
    torch.testing.assert_close(loaded.model(pixels), expected)
    loaded._input_channels = 9
    with pytest.raises(ValueError, match="channel count differs"):
        loaded._prepare_training_model(["a", "b"], pretrained=False, freeze=False)
    loaded._input_channels = 6
    before = loaded.model.attention[-1].weight.detach().clone()
    loaded._prepare_training_model(["c", "b", "a"], pretrained=False, freeze=False)
    assert loaded.model.head.out_features == 3
    torch.testing.assert_close(loaded.model.attention[-1].weight, before)


@pytest.mark.parametrize("channels", [1, 4, 5])
def test_fusion_rejects_incomplete_rgb_stacks(channels):
    from torchvision import models

    with pytest.raises(ValueError, match="multiple of 3"):
        RGBViewFusionClassifier(models.resnet18(weights=None), channels)


def test_fusion_rejects_expanded_stem_and_wrong_inference_channels():
    import torch
    from torchvision import models

    expanded = models.resnet18(weights=None)
    expanded.conv1 = torch.nn.Conv2d(6, 64, 7)
    with pytest.raises(ValueError, match="RGB backbone"):
        RGBViewFusionClassifier(expanded, 6)
    model = RGBViewFusionClassifier(models.resnet18(weights=None), 6)
    with pytest.raises(ValueError, match="6 channels"):
        model(torch.randn(1, 3, 32, 32))


def test_yaml_fusion_training_tiff_inference_and_reload(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from torchvision import models, transforms
    from collectra import Pipeline

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 1)
            self.fc = torch.nn.Linear(4, 1000)

        def forward(self, pixels):
            return self.fc(self.conv1(pixels).mean((2, 3)))

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: TinyClassifier())
    weights = SimpleNamespace(
        meta={"categories": [str(index) for index in range(1000)]},
        transforms=lambda: SimpleNamespace(
            crop_size=[8],
            resize_size=[8],
            mean=[0.5] * 3,
            std=[0.5] * 3,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    )
    monkeypatch.setattr(
        models, "get_model_weights", lambda name: SimpleNamespace(DEFAULT=weights)
    )
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        'collectra_pipeline_metadata:\n  name: Fusion\n  ext: collectra\n  version: "1"\nclassifier:\n  type: collectra.ImageClassifierTorchvision\n  input: image\n  output: [a, b]\n  model: resnet18\n  fusion: true\n'
    )
    workflow = Pipeline.from_file(pipeline)
    workflow.connect()
    task = workflow.node_manager.resolve_node("classifier").get_task()
    assert task.fusion is True
    samples = []
    for index, (name, partition) in enumerate(
        [("a", "train"), ("b", "train"), ("a", "val"), ("b", "val")]
    ):
        image = tmp_path / f"{index}.tif"
        tifffile.imwrite(
            image,
            np.full((6, 8, 8), index + 1, dtype="uint8"),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        samples.append(Image(name=name, data=image, partition=partition))
    result = task._train(
        *samples,
        base_folder=tmp_path,
        log="run",
        validation="val",
        epochs=1,
        batch=2,
        workers=0,
        device="cpu",
        wandb=False,
    )
    saved = ImageClassifierTorchvision(
        "classifier",
        model=result.save_dir / "weights/best.pt",
        output=["a", "b"],
        device="cpu",
    )
    prediction = saved.run(samples[0])
    assert isinstance(prediction, Link) and prediction.target is samples[0]
    assert saved.fusion is True and saved._input_channels == 6
    assert saved.model.encoder.conv1.in_channels == 3
    assert saved._preprocessing["mean"] == [0.5] * 6
