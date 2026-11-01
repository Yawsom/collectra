"""Exercise native YOLO graph skips, checkpoint reload and training reconstruction."""

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tifffile

pytest.importorskip("ultralytics")
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from collectra import Image, ObjectDetectionYOLO
from collectra.tasks.object_detection.yolo_feature_fusion import (
    FeatureFusionDetectionModel,
    FeatureFusionDetectionTrainer,
    FeatureFusionYOLO,
    ViewAttention,
    adapt_model,
    load_adapted_model,
    save_adapted_model,
)


@pytest.fixture
def rgb_yolo():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = YOLO("yolo11n.yaml")
    model.model.eval()
    yield model
    torch.set_num_threads(threads)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def assert_outputs_equal(first, second, atol=1e-5, rtol=1e-5):
    a, b = list(tensors(first)), list(tensors(second))
    assert len(a) == len(b) and a
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_attention_reference_and_gradients():
    attention = ViewAttention(8, 3, 1, 4)
    features = torch.randn(2, 3, 8, 4, 4, requires_grad=True)
    torch.testing.assert_close(attention(features), features[:, 1], atol=0, rtol=0)
    attention(features).square().mean().backward()
    assert attention.gate.grad.abs() > 0
    attention.zero_grad()
    features.grad = None
    with torch.no_grad():
        attention.gate.fill_(0.5)
    attention(features).square().mean().backward()
    assert all(features.grad[:, view].abs().sum() > 0 for view in range(3))
    assert attention.score[-1].weight.grad.abs().sum() > 0


def test_real_graph_preserves_rgb_and_fuses_all_neck_skips(rgb_yolo):
    rgb = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        original = rgb_yolo.model(rgb)
    conv_weights = rgb_yolo.model.model[0].conv.weight.detach().clone()
    adapt_model(rgb_yolo, 3, reference_view=1, hidden_dim=4)
    model = rgb_yolo.model
    assert isinstance(rgb_yolo, FeatureFusionYOLO)
    assert set(model.view_fusion) == {"4", "6", "10"}
    torch.testing.assert_close(model.model[0].conv.weight, conv_weights, atol=0, rtol=0)
    with torch.no_grad():
        actual = model(
            torch.cat([torch.rand_like(rgb), rgb, torch.rand_like(rgb)], dim=1)
        )
    assert_outputs_equal(actual, original)
    with torch.no_grad():
        for attention in model.view_fusion.values():
            attention.gate.fill_(0.6)
        repeated = model(rgb.repeat(1, 3, 1, 1))
    assert_outputs_equal(repeated, original)
    with pytest.raises(ValueError, match="Expected BCHW"):
        model(rgb)
    with pytest.raises(KeyError, match="CollectraFeatureFusionRGBConv"):
        DetectionModel(deepcopy(model.yaml), ch=9, verbose=False)


def test_checkpoint_command_reload_in_fresh_process(rgb_yolo, tmp_path):
    import os
    import subprocess
    import sys

    rgb_yolo.ckpt = {}
    source = tmp_path / "rgb.pt"
    rgb_yolo.save(source)
    path = tmp_path / "fusion.pt"
    from collectra import ObjectDetectionYOLO
    from collectra.cli import invoke

    invoke(
        ObjectDetectionYOLO("detector"),
        [
            "adapt-feature-fusion",
            str(source),
            "2",
            "--output",
            str(path),
            "--reference-view",
            "1",
            "--hidden-dim",
            "4",
        ],
    )
    loaded = load_adapted_model(path)
    assert loaded.model.yaml["channels"] == 6
    assert loaded.model.model[0].conv.in_channels == 3
    assert loaded.model.yaml["feature_fusion"]["reference_view"] == 1
    code = """import sys, torch
from collectra.tasks.object_detection.yolo_feature_fusion import load_adapted_model
model = load_adapted_model(sys.argv[1]).model.eval()
torch.set_num_threads(1)
with torch.no_grad():
    model(torch.zeros(1, 6, 64, 64))
print('RELOAD_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1"),
    )
    assert result.returncode == 0, result.stderr
    assert "RELOAD_OK" in result.stdout


def test_training_rebuild_keeps_attention_and_handles_class_change(rgb_yolo):
    adapt_model(rgb_yolo, 2, hidden_dim=4)
    with torch.no_grad():
        rgb_yolo.model.view_fusion["4"].gate.fill_(0.7)
    trainer = object.__new__(FeatureFusionDetectionTrainer)
    trainer.args = SimpleNamespace(cls_remap=False)
    trainer.data = {"channels": 6, "nc": 2, "names": {0: "a", 1: "b"}}
    model = trainer.get_model(
        cfg=rgb_yolo.model.yaml, weights=rgb_yolo.model, verbose=False
    )
    assert isinstance(model, FeatureFusionDetectionModel)
    assert model.model[-1].nc == 2
    assert model.model[0].conv.in_channels == 3
    for name, value in rgb_yolo.model.state_dict().items():
        if name.startswith("view_fusion."):
            torch.testing.assert_close(model.state_dict()[name], value)
    trainer.data["channels"] = 9
    with pytest.raises(ValueError, match="Dataset has 9 channels"):
        trainer.get_model(
            cfg=rgb_yolo.model.yaml, weights=rgb_yolo.model, verbose=False
        )


def test_collectra_tiff_inference_and_fusion_channel_check(
    rgb_yolo, tmp_path, monkeypatch
):
    from ultralytics.models.yolo.detect import DetectionPredictor

    adapt_model(rgb_yolo, 2, hidden_dim=4)
    path = tmp_path / "fusion.pt"
    save_adapted_model(rgb_yolo, path)
    pixels = np.broadcast_to(np.arange(6, dtype=np.uint8), (48, 64, 6)).copy()
    image_path = tmp_path / "views.tiff"
    tifffile.imwrite(
        image_path, pixels, photometric="minisblack", metadata={"axes": "YXC"}
    )
    observed = []
    original = DetectionPredictor.preprocess

    def capture(predictor, values):
        result = original(predictor, values)
        observed.append(result.shape)
        torch.testing.assert_close(result[0, :, 32, 32].cpu(), torch.arange(6) / 255)
        return result

    monkeypatch.setattr(DetectionPredictor, "preprocess", capture)
    task = ObjectDetectionYOLO("fusion", model=path)
    assert isinstance(task.run(Image(name="views", data=image_path), resolution=64), list)
    assert isinstance(task.model, FeatureFusionYOLO)
    assert observed == [torch.Size([1, 6, 64, 64])]
    gray = tmp_path / "gray.tif"
    tifffile.imwrite(gray, np.zeros((48, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="1 image channels.*expects 6"):
        task.run(Image(name="gray", data=gray), resolution=64)


@pytest.mark.parametrize("views, reference, hidden", [(0, 0, 4), (2, 2, 4), (2, 0, 0)])
def test_invalid_fusion_options(rgb_yolo, views, reference, hidden):
    with pytest.raises(ValueError, match="Require"):
        adapt_model(rgb_yolo, views, reference, hidden)


@pytest.mark.parametrize("architecture", ["yolov8n.yaml", "yolo26n.yaml"])
def test_other_standard_detection_graphs(rgb_yolo, architecture):
    yolo = YOLO(architecture)
    yolo.model.eval()
    rgb = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = yolo.model(rgb)
    adapt_model(yolo, 2, reference_view=1, hidden_dim=4)
    with torch.no_grad():
        actual = yolo.model(torch.cat([torch.rand_like(rgb), rgb], dim=1))
    assert_outputs_equal(actual, expected)
