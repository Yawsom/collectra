import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cappa
import numpy as np
import pytest
import yaml
from PIL import Image as PILImage

from collectra import (
    Pipeline,
    Image,
    ImageCrop,
    ImageSegmentation,
    ImageSegmentationSam3,
)
from collectra.cli import invoke
from collectra.tasks.image_segmentation.sam3 import (
    SegmentationSample,
    encode_rle,
    export_dataset,
    mask_in_view,
    mask_to_source,
    prepare_segmentation_inputs,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "source.png"
    PILImage.new("RGB", (12, 8), "white").save(path)
    return Image(name="image", id="image", data=path)


def instance(image, tmp_path, name="region", partition="", pixels=None):
    if pixels is None:
        pixels = np.zeros((8, 12), dtype="uint8")
        pixels[2:6, 3:9] = 1
        pixels[3, 4] = 0  # a hole must survive RLE encoding
    # Unique annotation paths even when their category labels are identical.
    import uuid

    path = tmp_path / f"mask-{uuid.uuid4().hex}.png"
    PILImage.fromarray(pixels).save(path)
    return ImageSegmentation(
        name=name,
        data=image.get_path(),
        mask=path,
        parents=[image.id],
        partition=partition,
    )


def decode_rle(rle):
    flat = []
    for value, count in enumerate(rle["counts"]):
        flat.extend([value % 2] * count)
    return np.asarray(flat, dtype="uint8").reshape(rle["size"], order="F")


@pytest.mark.parametrize("orientation", ["north", "east", "south", "west"])
@pytest.mark.parametrize("cropped", [False, True])
def test_view_masks_map_back_to_source_pixels(image, orientation, cropped):
    if cropped:
        image = ImageCrop(
            name="crop",
            data=image.get_path(),
            x_center=0.5,
            y_center=0.5,
            width_relative=0.5,
            height_relative=0.5,
            orientation=orientation,
        )
    else:
        image = Image(name="image", data=image.get_path(), orientation=orientation)
    local = np.zeros((image.pil().height, image.pil().width), dtype="uint8")
    local[1:3, 1:3] = 1
    source = mask_to_source(image, local)
    assert source.shape == (8, 12)
    restored = mask_in_view(image, PILImage.fromarray(source))
    assert np.array_equal(np.asarray(restored), local)


def test_segmentation_input_clips_predictions_to_its_region(image, tmp_path):
    region = instance(image, tmp_path)
    local = np.ones((region.pil().height, region.pil().width), dtype="uint8")
    projected = mask_to_source(region, local)
    assert np.array_equal(projected, np.asarray(PILImage.open(region.mask)))


@pytest.mark.parametrize("pixels", [np.zeros((2, 3)), np.ones((2, 3)), np.eye(5)])
def test_coco_rle_roundtrip(pixels):
    assert np.array_equal(decode_rle(encode_rle(pixels)), pixels)


def test_inference_returns_one_artifact_per_instance(image, tmp_path):
    task = ImageSegmentationSam3("segment", prompts={"region": "printed region"})
    task.context.file_path = tmp_path / "output.collectra"
    mask = np.zeros((2, 1, 8, 12), dtype=bool)
    mask[0, 0, 1:3, 2:5] = True
    mask[1, 0, 4:7, 7:10] = True
    task._processor = Mock()
    task._processor.set_text_prompt.return_value = {"masks": mask}
    results = task.run(image)
    assert len(results) == 2
    assert all(isinstance(item, ImageSegmentation) for item in results)
    assert all(item.name == "region" and item.parents == ["image"] for item in results)
    assert np.array_equal(np.asarray(PILImage.open(results[0].mask)), mask[0, 0])
    task._processor.set_text_prompt.assert_called_once_with(
        prompt="printed region", state=task._processor.set_image.return_value
    )


def test_inference_runs_under_bfloat16_autocast(image, tmp_path):
    import torch

    task = ImageSegmentationSam3("segment", prompt="region", device="cpu")
    task.context.file_path = tmp_path
    seen = []

    def prompt(**kwargs):
        seen.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
                     torch.is_grad_enabled()))
        masks = torch.zeros((1, 1, 8, 12), dtype=torch.bfloat16)
        masks[0, 0, 1:3, 2:5] = 1
        return {"masks": masks}

    task._processor = Mock()
    task._processor.set_text_prompt.side_effect = prompt
    results = task.run(image)
    assert seen == [(True, torch.bfloat16, False)]
    # bfloat16 masks are converted before NumPy, which has no bfloat16 type.
    assert np.asarray(PILImage.open(results[0].mask))[1:3, 2:5].all()


def test_empty_predictions_and_bad_inputs(image, tmp_path):
    task = ImageSegmentationSam3("segment")
    task.context.file_path = tmp_path
    task._processor = Mock()
    task._processor.set_text_prompt.return_value = {"masks": np.empty((0, 1, 8, 12))}
    assert task.run(image) == []
    with pytest.raises(TypeError):
        task.run("not an image")


@pytest.mark.parametrize("full_checkpoint", [False, True])
def test_model_loading_handles_official_and_finetuned_checkpoints(
    tmp_path, monkeypatch, full_checkpoint
):
    import sys
    from types import ModuleType

    for name in [
        "sam3",
        "sam3.model",
        "sam3.model_builder",
        "sam3.model.sam3_image_processor",
        "torch",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    state = {"detector.layer" if full_checkpoint else "layer": "weights"}
    sys.modules["torch"].load = Mock(return_value={"model": state})
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    builder = sys.modules["sam3.model_builder"].build_sam3_image_model = Mock(
        return_value=model
    )
    processor = sys.modules["sam3.model.sam3_image_processor"].Sam3Processor = Mock()
    task = ImageSegmentationSam3("segment", model="model.pt", threshold=0.7)
    task.pipeline = SimpleNamespace(name="mypipeline", path=tmp_path)
    task._init_model()
    assert builder.call_args.kwargs["load_from_HF"] is False
    assert builder.call_args.kwargs["checkpoint_path"] is None
    model.load_state_dict.assert_called_once_with({"layer": "weights"}, strict=True)
    processor.assert_called_once_with(model, device="cuda", confidence_threshold=0.7)


@pytest.mark.parametrize("full_checkpoint", [False, True])
def test_sam31_weights_preserve_image_features_and_roundtrip(monkeypatch, full_checkpoint):
    torch = pytest.importorskip("torch")
    from collectra.tasks.image_segmentation.sam3_training import load_image_state

    def image_model():
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.scalp = 1
        neck = model.backbone.vision_backbone = torch.nn.Module()
        neck.scale_factors = (4.0, 2.0, 1.0, 0.5)
        neck.convs = torch.nn.ModuleList()
        for _ in neck.scale_factors:
            layer = torch.nn.Sequential()
            layer.add_module("conv_1x1", torch.nn.Conv2d(2, 2, 1))
            layer.add_module("conv_3x3", torch.nn.Conv2d(2, 2, 3, padding=1))
            neck.convs.append(layer)
        return model

    source = image_model()
    prefix = "backbone.vision_backbone."
    state = {k: v for k, v in source.state_dict().items() if not k.startswith(prefix + "convs.3.")}
    for branch in ("interactive_convs", "propagation_convs"):
        state.update({k.replace("convs.", branch + "."): v for k, v in list(state.items())
                      if k.startswith(prefix + "convs.")})
    if full_checkpoint:
        state = {"detector." + k: v for k, v in state.items()}
        state["tracker.unused"] = torch.zeros(1)
    target = image_model()
    load_image_state(target, state)
    pixels = torch.randn(1, 2, 8, 8)
    for original, loaded in zip(source.backbone.vision_backbone.convs[:3],
                                target.backbone.vision_backbone.convs[:3]):
        torch.testing.assert_close(original(pixels), loaded(pixels))
    # Fine-tuned image-only checkpoints remain strictly loadable.
    restored = image_model()
    load_image_state(restored, target.state_dict())
    broken = dict(state)
    first = ("detector." if full_checkpoint else "") + prefix + "convs.0.conv_1x1.weight"
    del broken[first]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_image_state(image_model(), broken)
    unsupported = image_model()
    unsupported.backbone.scalp = 0
    with pytest.raises(ValueError, match="Unsupported"):
        load_image_state(unsupported, state)


def test_training_model_mlps_support_backprop_without_changing_inference(monkeypatch):
    import sys
    from types import ModuleType
    torch = pytest.importorskip("torch")
    from collectra.tasks.image_segmentation.sam3_training import build_training_model

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 8)
            self.act = torch.nn.GELU()
            self.drop1 = torch.nn.Dropout(0.1)
            self.norm = torch.nn.LayerNorm(8)
            self.fc2 = torch.nn.Linear(8, 4)
            self.drop2 = torch.nn.Dropout(0.1)

        def forward(self, x):
            raise ValueError("Expected grad to be disabled.")

    for name in ("sam3", "sam3.model", "sam3.model.vitdet", "sam3.model_builder"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["sam3.model.vitdet"].Mlp = Mlp
    model = torch.nn.Sequential(Mlp())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    builder = Mock(return_value=model)
    sys.modules["sam3.model_builder"].build_sam3_image_model = builder
    monkeypatch.setattr(torch, "load", Mock(return_value={"model": state}))
    pixels = torch.randn(2, 3, 4, requires_grad=True)
    with pytest.raises(ValueError, match="Expected grad"):
        model(pixels)
    trained = build_training_model("checkpoint.pt", "vocabulary.gz")
    assert builder.call_args.kwargs["eval_mode"] is False
    trained(pixels).square().mean().backward()
    assert pixels.grad is not None and pixels.grad.abs().sum() > 0
    for parameter in trained.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert set(trained.state_dict()) == set(state)
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(value, state[key])
    # Validation uses the same activation, normalization and dropout modules.
    trained.eval()
    mlp = trained[0]
    expected = mlp.fc2(mlp.norm(mlp.act(mlp.fc1(pixels))))
    torch.testing.assert_close(trained(pixels), expected)
    with pytest.raises(ValueError, match="Expected grad"):
        Mlp()(pixels)


@pytest.mark.parametrize("freeze_semantic", [False, True])
def test_instance_training_ddp_handles_unsupervised_semantic_output(tmp_path, freeze_semantic):
    from contextlib import nullcontext
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    from collectra.tasks.image_segmentation.sam3_training import freeze_unused_semantic_head

    if not dist.is_available() or not dist.is_gloo_available() or dist.is_initialized():
        pytest.skip("Requires an independent CPU/Gloo process group")

    class StageOutputs(list):
        # SAM3Output iterates over separate storage. Unlike a plain list/dict,
        # this container is opaque to DDP's output pytree reconstruction.
        def __init__(self, outputs):
            self.outputs = outputs

        def __iter__(self):
            return iter(self.outputs)

        def __len__(self):
            return len(self.outputs)

        def __getitem__(self, index):
            return self.outputs[index]

    class InstanceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.segmentation_head = torch.nn.Module()
            self.segmentation_head.instance_seg_head = torch.nn.Linear(4, 1)
            self.segmentation_head.semantic_seg_head = torch.nn.Linear(4, 1)

        def forward(self, x):
            features = self.backbone(x)
            return StageOutputs([{
                "pred_masks": self.segmentation_head.instance_seg_head(features),
                "semantic_seg": self.segmentation_head.semantic_seg_head(features),
            }])

    model = InstanceModel()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    if freeze_semantic:
        freeze_unused_semantic_head(model)
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(),
                            rank=0, world_size=1)
    try:
        parallel = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        expectation = nullcontext() if freeze_semantic else pytest.raises(RuntimeError, match="Expected to have finished reduction")
        with expectation:
            for _ in range(3):
                optimizer.zero_grad()
                outputs = parallel(torch.ones(2, 4))
                outputs[0]["pred_masks"].sum().backward()
                optimizer.step()
        if freeze_semantic:
            assert model.backbone.weight.grad is not None
            assert model.segmentation_head.instance_seg_head.weight.grad is not None
            assert set(model.state_dict()) == set(before)
            for name, value in model.state_dict().items():
                if "semantic_seg_head" in name:
                    torch.testing.assert_close(value, before[name])
                else:
                    assert not torch.equal(value, before[name])
    finally:
        dist.destroy_process_group()


def test_export_preserves_instances_holes_labels_and_partitions(image, tmp_path):
    train = instance(image, tmp_path)
    validation = instance(image, tmp_path, name="other", partition="val")
    dataset = export_dataset(
        [
            SegmentationSample(image, [train, instance(image, tmp_path)]),
            SegmentationSample(image, [validation]),
        ],
        tmp_path / "dataset",
        prompts={"region": "printed region"},
        validation="val",
    )
    train_data = json.loads((dataset / "train/_annotations.coco.json").read_text())
    valid_data = json.loads((dataset / "valid/_annotations.coco.json").read_text())
    assert len(train_data["images"]) == 1
    assert len(train_data["annotations"]) == 2
    assert len(valid_data["annotations"]) == 1
    assert train_data["categories"] == valid_data["categories"]
    annotation = train_data["annotations"][0]
    assert annotation["bbox"] == [3, 2, 6, 4]
    assert annotation["area"] == 23
    assert np.array_equal(
        decode_rle(annotation["segmentation"]), np.asarray(PILImage.open(train.mask))
    )


def test_export_skips_masks_outside_their_input_image(image, tmp_path):
    # The left half of the 12x8 source; masks on the right lie outside it.
    crop = ImageCrop(name="crop", id="crop", data=image.get_path(), x_center=0.25,
                     y_center=0.5, width_relative=0.5, height_relative=1.0)
    inside = np.zeros((8, 12), dtype="uint8")
    inside[2:6, 1:5] = 1
    outside = np.zeros((8, 12), dtype="uint8")
    outside[2:6, 8:11] = 1
    kept = instance(crop, tmp_path, pixels=inside)
    messages = []
    dataset = export_dataset(
        [
            SegmentationSample(crop, [kept, instance(crop, tmp_path, pixels=outside)]),
            SegmentationSample(crop, [instance(crop, tmp_path, pixels=outside)]),
        ],
        tmp_path / "dataset",
        progress=messages.append,
    )
    train = json.loads((dataset / "train/_annotations.coco.json").read_text())
    # The image whose only mask is outside it is left out, not exported as empty.
    assert [item["id"] for item in train["images"]] == [1]
    assert [item["image_id"] for item in train["annotations"]] == [1]
    assert train["annotations"][0]["bbox"] == [1, 2, 4, 4]
    assert sorted(path.name for path in (dataset / "train").glob("*.png")) == ["00000001.png"]
    assert sum("WARNING: skipping mask" in message for message in messages) == 2
    assert any("skipped 2 masks" in message and "1 images" in message for message in messages)


def test_export_fails_when_every_training_mask_is_outside(image, tmp_path):
    crop = ImageCrop(name="crop", id="crop", data=image.get_path(), x_center=0.25,
                     y_center=0.5, width_relative=0.5, height_relative=1.0)
    outside = np.zeros((8, 12), dtype="uint8")
    outside[2:6, 8:11] = 1
    with pytest.raises(ValueError, match="No labelled training instances remain"):
        export_dataset([SegmentationSample(crop, [instance(crop, tmp_path, pixels=outside)])],
                       tmp_path / "dataset", progress=lambda message: None)


@pytest.mark.parametrize("exclude,expected_train", [("test", 2), ("", 3)])
def test_export_excludes_requested_partitions(image, tmp_path, exclude, expected_train):
    samples = [SegmentationSample(image, [instance(image, tmp_path, partition=partition)])
               for partition in ("", "validation", "test", "holdout")]
    dataset = export_dataset(samples, tmp_path / "dataset", validation="validation", exclude=exclude)
    train = json.loads((dataset / "train/_annotations.coco.json").read_text())
    valid = json.loads((dataset / "valid/_annotations.coco.json").read_text())
    assert len(train["images"]) == expected_train
    assert len(valid["images"]) == 1


def test_max_items_caps_combined_splits_after_exclusions(image, tmp_path):
    samples = [SegmentationSample(image, [instance(image, tmp_path, partition=partition)])
               for partition in ("test", "", "validation", "", "validation")]
    dataset = export_dataset(samples, tmp_path / "dataset", validation="validation", exclude="test", max_items=2)
    train = json.loads((dataset / "train/_annotations.coco.json").read_text())
    valid = json.loads((dataset / "valid/_annotations.coco.json").read_text())
    assert len(train["images"]) == len(valid["images"]) == 1


def test_preparation_requires_parent_and_matching_image(image, tmp_path):
    target = instance(image, tmp_path)
    args = ([target], [image], {"page": [target]}, {"page": [image]})
    assert prepare_segmentation_inputs(*args)[0].image is image
    with pytest.raises(ValueError, match="exactly one"):
        prepare_segmentation_inputs([target], [], {"page": [target]}, {})


def test_cli_and_public_exports_need_no_sam_backend(capsys):
    task = ImageSegmentationSam3("segment")
    assert task.input_type() is Image
    assert task.output_type() is ImageSegmentation
    with pytest.raises(cappa.HelpExit):
        invoke(task, ["train", "--help"])
    assert "epochs" in capsys.readouterr().out


@pytest.mark.parametrize("input_kind", ["parent", "page", "yaml"])
@pytest.mark.parametrize("max_items", [None, 1])
def test_pipeline_training_export_without_model_installation(tmp_path, input_kind, capsys, max_items):
    page = tmp_path / "page.collectra"
    page.mkdir()
    image_path = page / "source.png"
    PILImage.new("RGB", (12, 8), "white").save(image_path)
    pixels = np.zeros((8, 12), dtype="uint8")
    pixels[2:6, 3:9] = 1
    PILImage.fromarray(pixels).save(page / "mask.png")
    (page / "results.yaml").write_text(
        yaml.safe_dump(
            {
                "image": {
                    "type": "collectra.Image",
                    "id": "image",
                    "data": "source.png",
                },
                "region": {
                    "type": "collectra.ImageSegmentation",
                    "id": "region",
                    "data": "source.png",
                    "mask": "mask.png",
                    "parents": "image",
                },
            }
        )
    )
    pipeline = Pipeline(
        "SAM",
        "collectra",
        "1",
        path=tmp_path,
        segment={
            "type": "collectra.ImageSegmentationSam3",
            "input": "image",
            "output": "region",
        },
    )
    pipeline.connect()
    task = pipeline.node_manager.resolve_node("segment").get_task()
    source = {"parent": tmp_path, "page": page, "yaml": page / "results.yaml"}[
        input_kind
    ]
    inputs = [str(source)]
    if max_items:
        bad = tmp_path / "unread.collectra"
        bad.mkdir()
        (bad / "results.yaml").write_text("This must never be loaded")
        inputs.append(str(bad))
    directory = task.prepare_training_data(inputs, tmp_path / "dataset", max_items=max_items)
    progress = capsys.readouterr().out
    assert f"Load 1/{2 if max_items else 1}:" in progress
    if max_items:
        assert "Reached max_items=1" in progress
    assert "Loaded 1 labelled images and 1 masks" in progress
    assert "Export 1/1 [train]" in progress
    assert "Dataset export complete" in progress
    assert (
        len(
            json.loads((directory / "train/_annotations.coco.json").read_text())[
                "annotations"
            ]
        )
        == 1
    )


@pytest.fixture
def sam_package(tmp_path, monkeypatch):
    from collectra.tasks.image_segmentation import sam3

    package = tmp_path / "sam3"
    template = (
        package / "train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml"
    )
    template.parent.mkdir(parents=True)
    # Shape of the installed upstream detection template that our adapter patches.
    config = {
        "paths": {},
        "scratch": {"collate_fn": {"with_seg_masks": "${scratch.enable_segmentation}"}},
        "roboflow_train": {
            "loss": {"loss_fns_find": []},
            "val_transforms": [{"transforms": []}],
        },
        "trainer": {
            "model": {},
            "data": {"train": {"dataset": {}}, "val": {"dataset": {}}},
            "meters": {
                "val": {"roboflow100": {"detection": {"pred_file_evaluators": [{}]}}}
            },
        },
        "launcher": {},
        "submitit": {},
    }
    template.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sam3, "_sam3", lambda: SimpleNamespace(__file__=str(package / "__init__.py"))
    )
    return package


def test_training_configuration_enables_real_mask_losses(image, tmp_path, sam_package):
    from collectra.tasks.image_segmentation.sam3_training import training_config

    dataset = export_dataset(
        [SegmentationSample(image, [instance(image, tmp_path)])], tmp_path / "dataset"
    )
    config = training_config(
        dataset, tmp_path, tmp_path / "checkpoint.pt", epochs=2, batch=3, num_gpus=1
    )
    assert config["scratch"]["enable_segmentation"]
    assert config["trainer"]["data"]["train"]["dataset"]["load_segmentation"]
    assert config["roboflow_train"]["loss"]["loss_fns_find"][-1]["weight_dict"] == {
        "loss_mask": 200.0,
        "loss_dice": 10.0,
    }
    assert config["trainer"]["mode"] == "train_only"
    assert "val" not in config["trainer"]["data"]
    assert config["trainer"]["skip_saving_ckpts"]
    assert config["trainer"]["max_epochs"] == 2


def test_validation_configuration_installs_iou_and_reporting(image, tmp_path, sam_package):
    from collectra.tasks.image_segmentation.sam3_training import training_config

    dataset = export_dataset(
        [SegmentationSample(image, [instance(image, tmp_path)]),
         SegmentationSample(image, [instance(image, tmp_path, partition="validation")])],
        tmp_path / "dataset", validation="validation",
    )
    config = training_config(dataset, tmp_path, tmp_path / "initial.pt",
                             epochs=3, batch=1, num_gpus=2, wandb=True,
                             run_name="region_segmentor", confidence_threshold=0.7)
    trainer = config["trainer"]
    assert trainer["_target_"].endswith("sam3_metrics.create_trainer")
    assert trainer["skip_saving_ckpts"] is True
    assert trainer["skip_first_val"] is False
    assert trainer["val_epoch_freq"] == 1
    assert trainer["collectra_wandb"] is True
    assert trainer["collectra_run_name"] == "region_segmentor"
    assert "val" in trainer["data"]
    assert trainer["meters"]["val"]["roboflow100"]["mask_iou"] == {
        "_target_": "collectra.tasks.image_segmentation.sam3_metrics.MaskIoU",
        "confidence_threshold": 0.7,
    }
    # Training images are rescored like validation for train/mask_iou.
    train_eval = trainer["collectra_train_eval"]
    assert train_eval["dataset"]["img_folder"] == str(dataset / "train")
    assert train_eval["dataset"]["ann_file"] == str(dataset / "train/_annotations.coco.json")
    # Otherwise identical to validation (transforms, loader, collate function).
    paths = ("img_folder", "ann_file")
    assert {k: v for k, v in train_eval["dataset"].items() if k not in paths} == {
        k: v for k, v in trainer["data"]["val"]["dataset"].items() if k not in paths
    }
    assert {k: v for k, v in train_eval.items() if k != "dataset"} == {
        k: v for k, v in trainer["data"]["val"].items() if k != "dataset"
    }
    assert trainer["collectra_mask_threshold"] == 0.7


@pytest.mark.parametrize("explicit_output", [True, False])
@pytest.mark.parametrize("keep_log", [True, False])
@pytest.mark.parametrize("excluded", ["", "test"])
def test_train_adopts_checkpoint_and_updates_pipeline_config(
    image, tmp_path, sam_package, monkeypatch, explicit_output, keep_log, capsys, excluded
):
    from collectra.tasks.image_segmentation import sam3

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sam3, "datetime", SimpleNamespace(now=lambda: datetime(2026, 10, 16, 14))
    )
    task = ImageSegmentationSam3("segment", model="initial.pt")
    task.pipeline = SimpleNamespace(name="mypipeline", path=tmp_path, data={"segment": {}}, save=Mock())
    dataset = export_dataset(
        [SegmentationSample(image, [instance(image, tmp_path)])], tmp_path / "dataset"
    )
    prepare = Mock(return_value=dataset)
    monkeypatch.setattr(task, "prepare_training_data", prepare)
    monkeypatch.setattr(task, "_checkpoint", lambda: tmp_path / "initial.pt")

    def train(command, **kwargs):
        assert kwargs["check"]
        assert "-u" in command
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        config = yaml.safe_load(Path(command[-1]).read_text())
        assert config["trainer"]["collectra_wandb"] is True
        assert config["trainer"]["collectra_project"] == f"mypipeline-segment"
        assert config["trainer"]["collectra_run_name"] == output.name
        checkpoint = (
            Path(config["paths"]["experiment_log_dir"]) / "checkpoints/best.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake weights")

    monkeypatch.setattr(sam3.subprocess, "run", train)
    arguments = ["train", "--validation", "validation", "results", "--epochs", "1", "--wandb"]
    output = tmp_path / ("output" if explicit_output else "20261016_140000")
    if explicit_output:
        arguments.extend(["--output", str(output), "--max-items", "2", "--reuse-data"])
        if excluded:
            partial = output / "dataset/train/_annotations.coco.json"
            partial.parent.mkdir(parents=True)
            partial.write_text("{}")
    if not keep_log:
        arguments.append("--no-keep-log")
    if excluded:
        arguments.extend(["--exclude", excluded])
    trained = invoke(task, arguments)
    prepare.assert_called_once_with(["results"], output / "dataset", "validation", excluded, max_items=2 if explicit_output else None)
    assert trained.read_bytes() == b"fake weights"
    assert task.pipeline.data["segment"]["model"] == trained.name
    task.pipeline.save.assert_called_once_with("segment")
    assert output.exists() == keep_log
    progress = capsys.readouterr().out
    if explicit_output:
        assert "falling back to preparation from inputs" in progress
    assert f"output={output}" in progress
    assert "Resolving the SAM checkpoint" in progress
    assert "Configuration saved:" in progress
    assert "Starting SAM trainer" in progress
    assert "Saved trained checkpoint:" in progress
    assert "Training complete" in progress


@pytest.mark.parametrize('inputs', [[], ['nonexistent-source-directory']])
@pytest.mark.parametrize('max_items', [None, 1, 0])
def test_train_reuses_output_dataset_without_source_checks(
    image, tmp_path, sam_package, monkeypatch, capsys, inputs, max_items
):
    from collectra.tasks.image_segmentation import sam3

    output = tmp_path / 'existing-training'
    dataset = export_dataset(
        [SegmentationSample(image, [instance(image, tmp_path)])], output / 'dataset'
    )
    before = {path: path.read_bytes() for path in dataset.rglob('*') if path.is_file()}
    task = ImageSegmentationSam3('segment', model='initial.pt')
    task.pipeline = SimpleNamespace(name="mypipeline", path=tmp_path, data={'segment': {}}, save=Mock())
    prepare = Mock(side_effect=AssertionError('Must not prepare or check source data'))
    monkeypatch.setattr(task, 'prepare_training_data', prepare)
    monkeypatch.setattr(task, '_checkpoint', lambda: tmp_path / 'initial.pt')

    def train(command, **kwargs):
        config = yaml.safe_load(Path(command[-1]).read_text())
        assert config['trainer']['data']['train']['dataset']['img_folder'] == str(dataset / 'train')
        checkpoint = output / 'training/checkpoints/best.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'reused-data-model')

    monkeypatch.setattr(sam3.subprocess, 'run', train)
    options = ['train', *inputs, '--output', str(output), '--reuse-data',
               '--validation', 'ignored-new-partition', '--exclude', 'ignored']
    if max_items is not None:
        options.extend(['--max-items', str(max_items)])
    trained = invoke(task, options)
    prepare.assert_not_called()
    assert trained.read_bytes() == b'reused-data-model'
    assert all(path.read_bytes() == content for path, content in before.items())
    progress = capsys.readouterr().out
    assert 'Skipping source checks and export' in progress
    if max_items is not None:
        assert 'Ignoring --max-items' in progress


def test_reuse_data_requires_output_and_normal_training_requires_inputs():
    task = ImageSegmentationSam3('segment')
    with pytest.raises(ValueError, match='--reuse-data requires --output'):
        invoke(task, ['train', '--reuse-data'])
    with pytest.raises(ValueError, match='Supply training inputs'):
        invoke(task, ['train'])


def test_reuse_missing_dataset_requires_inputs(tmp_path):
    with pytest.raises(ValueError, match="requires an existing export"):
        invoke(ImageSegmentationSam3("segment"), ["train", "--reuse-data", "--output", str(tmp_path)])


@pytest.mark.parametrize("options, message", [
    (["--max-items", "0"], "max_items must be positive"),
    (["--max-items", "-1"], "max_items must be positive"),
])
def test_max_items_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        invoke(ImageSegmentationSam3("segment"), ["train", "results", *options])
