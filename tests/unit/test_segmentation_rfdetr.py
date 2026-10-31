import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cappa
import numpy as np
import pytest
from PIL import Image as PILImage

from collectra import Pipeline, Image, ImageCrop, ImageSegmentation, ImageSegmentationRFDETR
from collectra.cli import invoke
from collectra.tasks.image_segmentation import rfdetr
from collectra.tasks.image_segmentation.data import SegmentationSample


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "page.png"
    PILImage.new("RGB", (12, 8), "white").save(path)
    image = Image(name="page", id="page", data=path)
    pixels = np.zeros((8, 12), dtype="uint8")
    pixels[1:6, 2:10] = 1
    pixels[3, 4] = 0
    mask = tmp_path / "mask.png"
    PILImage.fromarray(pixels).save(mask)
    instance = ImageSegmentation(name="text_region", data=path, mask=mask, parents=[image.id])
    return SegmentationSample(image, [instance])


class Prediction(SimpleNamespace):
    def __len__(self):
        return len(self.class_id)


def predictor(masks, names=("text_region",)):
    return SimpleNamespace(
        model_config=SimpleNamespace(segmentation_head=True), class_names=list(names),
        predict=Mock(return_value=Prediction(
            mask=np.asarray(masks), class_id=np.zeros(len(masks), dtype=int),
            data={"class_name": np.array([names[0]] * len(masks))},
        )),
    )


def test_prediction_inverts_crop_and_preserves_parent(sample, tmp_path):
    crop = ImageCrop(name="crop", id="crop", data=sample.image.data,
                     x_center=.5, y_center=3.5/8, width_relative=8/12, height_relative=5/8, partition="test")
    local = np.ones((5, 8), dtype=bool)
    local[2, 2] = False
    task = ImageSegmentationRFDETR("segment", output="text_region", threshold=.7)
    task.context.file_path = tmp_path / "results"
    task._predictor = predictor([local])
    result, = task.run(crop)
    assert isinstance(result, ImageSegmentation)
    assert result.parents == [crop.id]
    assert result.name == "text_region" and result.partition == "test"
    np.testing.assert_array_equal(np.asarray(result._mask_image()) != 0, np.asarray(sample.instances[0]._mask_image()) != 0)
    task._predictor.predict.assert_called_once()
    assert task._predictor.predict.call_args.kwargs == {"threshold": .7}


def test_pipeline_run_stores_mask_paths_relative_to_results(pipeline, tmp_path, monkeypatch):
    import yaml

    page = tmp_path / "pages" / "page.png"
    page.parent.mkdir()
    PILImage.new("RGB", (12, 8), "white").save(page)
    local = np.zeros((8, 12), dtype=bool)
    local[1:6, 2:10] = True

    def init_model(task):
        task._predictor = predictor([local])

    monkeypatch.setattr(ImageSegmentationRFDETR, "_init_model", init_model)
    pipeline.cli_run([str(page)], output=tmp_path / "out")
    folder = tmp_path / "out" / "page.htr"
    record = yaml.safe_load((folder / "results.yaml").read_text())["text_region"]
    assert not Path(record["mask"]).is_absolute()
    assert record["mask"].startswith("masks/")
    assert (folder / record["mask"]).is_file()


def test_segmentation_input_constrains_prediction(sample, tmp_path):
    task = ImageSegmentationRFDETR("segment", output="text_region")
    task.context.file_path = tmp_path
    image = sample.instances[0]
    view = image.pil()
    task._predictor = predictor([np.ones((view.height, view.width), bool)])
    result, = task.run(image)
    np.testing.assert_array_equal(result._mask_image(), image._mask_image())


def test_empty_predictions_and_unconfigured_classes(sample, tmp_path):
    task = ImageSegmentationRFDETR("segment", output="text_region")
    task.context.file_path = tmp_path
    task._predictor = predictor(np.empty((0, 8, 12)))
    assert task.run(sample.image) == []
    task._predictor = predictor([np.ones((8, 12))], names=("cat", "dog"))
    assert task.run(sample.image) == []


@pytest.mark.parametrize("model,classname", [("small", "RFDETRSegSmall"), ("RFDETRSegNano", "RFDETRSegNano")])
def test_lazy_variant_backend(monkeypatch, model, classname):
    backend = SimpleNamespace(**{classname: Mock(return_value=predictor([]))})
    monkeypatch.setattr(rfdetr, "_rfdetr", lambda: backend)
    task = ImageSegmentationRFDETR("segment", model=model, device="cpu", resolution=384)
    task._init_model()
    task._init_model()
    getattr(backend, classname).assert_called_once_with(device="cpu", resolution=384)


def test_checkpoint_resolves_relative_to_pipeline_and_rejects_detection(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    loaded = predictor([])
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(rfdetr, "_rfdetr", lambda: SimpleNamespace(RFDETR=SimpleNamespace(from_checkpoint=loader)))
    task = ImageSegmentationRFDETR("segment", model="weights.pt")
    task.pipeline = SimpleNamespace(name="mypipeline", path=tmp_path)
    task._init_model()
    loader.assert_called_once_with(str(checkpoint), device="cuda")
    task._predictor = None
    loaded.model_config.segmentation_head = False
    with pytest.raises(ValueError, match="segmentation checkpoint"):
        task._init_model()


def test_export_keeps_class_names_and_native_loader_decodes_holes(sample, tmp_path, monkeypatch):
    task = ImageSegmentationRFDETR("segment", output="text_region", prompt="printed page")
    monkeypatch.setattr(task, "_samples", lambda *a, **kw: [sample])
    dataset = task.prepare_training_data([], tmp_path / "dataset")
    annotations = json.loads((dataset / "train/_annotations.coco.json").read_text())
    assert annotations["categories"] == [{"id": 1, "name": "text_region"}]
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from rfdetr.datasets.coco import convert_coco_poly_to_mask
    decoded = convert_coco_poly_to_mask([annotations["annotations"][0]["segmentation"]], 8, 12)
    np.testing.assert_array_equal(decoded[0].numpy(), np.asarray(sample.instances[0]._mask_image()) != 0)


@pytest.fixture
def pipeline(tmp_path):
    import yaml

    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({
        "collectra_pipeline_metadata": {"name": "Segments", "ext": "htr", "version": "1"},
        "segment": {"type": "collectra.ImageSegmentationRFDETR", "input": "page", "output": "text_region"},
    }))
    return Pipeline.from_file(path)


def fake_training(monkeypatch):
    def run(command, **kwargs):
        assert command[2:4] == ["-m", "collectra.tasks.image_segmentation.rfdetr_training"]
        config = json.loads(Path(command[-1]).read_text())
        (Path(config["params"]["output_dir"]) / "best.pt").write_bytes(b"best weights")
    monkeypatch.setattr(rfdetr.subprocess, "run", run)


def test_training_cli_flags_and_model_adoption(pipeline, sample, tmp_path, monkeypatch):
    task = pipeline.cli_tasks()["segment"]
    monkeypatch.setattr(task, "_samples", Mock(return_value=[sample]))
    fake_training(monkeypatch)
    output = tmp_path / "nested" / "run"
    invoke(pipeline, ["task", "segment", "train", "--output", str(output), "--wandb", "--max-items", "3", "--exclude", "test", str(tmp_path)])
    config = json.loads((output / "rfdetr-training.json").read_text())
    assert config["params"]["wandb"]
    assert config["params"]["project"] == "Segments-segment"
    assert config["params"]["run"] == "run"
    assert config["params"]["use_ema"] is False
    assert config["has_validation"] is False
    task._samples.assert_called_once_with([str(tmp_path)], max_items=3, exclude="test")
    assert task.model.read_bytes() == b"best weights"
    assert pipeline.data["segment"]["model"] == task.model.name


def test_training_model_override(pipeline, sample, tmp_path, monkeypatch):
    task = pipeline.cli_tasks()["segment"]
    monkeypatch.setattr(task, "_samples", Mock(return_value=[sample]))
    fake_training(monkeypatch)
    output = tmp_path / "run"
    invoke(pipeline, ["task", "segment", "train", "--output", str(output), "--model", "medium", str(tmp_path)])
    assert json.loads((output / "rfdetr-training.json").read_text())["model"] == "medium"
    # The trained weights still replace the configured model.
    assert pipeline.data["segment"]["model"] == task.model.name

    (tmp_path / "start.pt").write_bytes(b"weights")
    task.train([str(tmp_path)], output=tmp_path / "run2", model="start.pt")
    config = json.loads((tmp_path / "run2" / "rfdetr-training.json").read_text())
    assert config["model"] == str((tmp_path / "start.pt").resolve())


def test_missing_model_override_fails_before_preparation(pipeline, sample, tmp_path, monkeypatch):
    task = pipeline.cli_tasks()["segment"]
    monkeypatch.setattr(task, "_samples", Mock(return_value=[sample]))
    with pytest.raises(FileNotFoundError, match="nano, small"):
        task.train([str(tmp_path)], output=tmp_path / "run", model="mediun")
    task._samples.assert_not_called()


def test_reuse_overrides_loading_and_max_items(pipeline, sample, tmp_path, monkeypatch):
    task = pipeline.cli_tasks()["segment"]
    monkeypatch.setattr(task, "_samples", Mock(return_value=[sample]))
    output = tmp_path / "run"
    task.prepare_training_data([], output / "dataset")
    task._samples.reset_mock()
    fake_training(monkeypatch)
    task.train([], output=output, reuse_data=True, max_items=0)
    task._samples.assert_not_called()


def test_missing_reuse_falls_back_and_no_keep_log(pipeline, sample, tmp_path, monkeypatch):
    task = pipeline.cli_tasks()["segment"]
    monkeypatch.setattr(task, "_samples", Mock(return_value=[sample]))
    fake_training(monkeypatch)
    output = tmp_path / "run"
    destination = task.train([str(tmp_path)], output=output, reuse_data=True, keep_log=False)
    task._samples.assert_called_once()
    assert destination.is_file() and not output.exists()


def test_help_has_matching_training_flags(pipeline):
    with pytest.raises(cappa.HelpExit):
        invoke(pipeline, ["task", "segment", "train", "--help"])


def test_mask_iou_is_class_aware_and_resizes():
    torch = pytest.importorskip("torch")
    from collectra.tasks.image_segmentation.rfdetr_training import mask_overlap

    prediction = {"masks": torch.ones(3, 1, 4, 4, dtype=torch.bool),
                  "labels": torch.tensor([0, 1, 2]), "scores": torch.tensor([.9, .9, .1])}
    target = {"masks": torch.ones(1, 2, 2, dtype=torch.bool), "labels": torch.tensor([0])}
    intersection, union = mask_overlap(prediction, target, .5)
    assert intersection == 4 and union == 8  # absent-class false positive counts


def test_callbacks_keep_only_best_weights_and_write_csv(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from collectra.tasks.image_segmentation.rfdetr_training import make_callback

    model = torch.nn.Linear(2, 1)
    config = {"model_name": "RFDETRSegSmall", "segmentation_head": True}
    module = SimpleNamespace(model=model, train_config=SimpleNamespace(model_dump=lambda: {}),
                             model_config=SimpleNamespace(model_dump=lambda: config))
    trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False, current_epoch=0,
                              callback_metrics={"val/mask_iou": torch.tensor(.7), "train/loss": torch.tensor(2.)},
                              datamodule=SimpleNamespace(class_names=["text_region"]))
    callback = make_callback(tmp_path, .5, True)
    callback.on_validation_end(trainer, module)
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    assert saved["args"]["class_names"] == ["text_region"]
    assert set(saved) == {"model", "args", "model_config", "model_name", "epoch"}
    trainer.current_epoch = 1
    trainer.callback_metrics["val/mask_iou"] = torch.tensor(.6)
    callback.on_validation_end(trainer, module)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 0
    assert len((tmp_path / "metrics.csv").read_text().splitlines()) == 3
    trainer.callback_metrics["val/mask_iou"] = torch.tensor(.8)
    callback.on_validation_end(trainer, module)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 1


def test_native_trainer_hooks_disable_optimizer_checkpoints(tmp_path):
    pytest.importorskip("rfdetr")
    pytest.importorskip("pytorch_lightning")
    import rfdetr.training as training
    from rfdetr.config import RFDETRSegSmallConfig, SegmentationTrainConfig
    from pytorch_lightning.callbacks import ModelCheckpoint
    from collectra.tasks.image_segmentation.rfdetr_training import training_hooks

    original = training.build_trainer
    config = SegmentationTrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), epochs=1, use_ema=False,
                                     tensorboard=False, wandb=False, progress_bar=None)
    with training_hooks(tmp_path, .5, False):
        trainer = training.build_trainer(config, RFDETRSegSmallConfig(), accelerator="cpu", devices=1)
        assert not any(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
        assert any(type(cb).__name__ == "SegmentationMetrics" for cb in trainer.callbacks)
        assert trainer.limit_val_batches == 0
    assert training.build_trainer is original


def test_native_csv_logger_does_not_share_collectra_metrics_csv(tmp_path):
    pytest.importorskip("rfdetr")
    pytest.importorskip("pytorch_lightning")
    import rfdetr.training as training
    import rfdetr.training.trainer as native
    from pytorch_lightning.loggers import CSVLogger
    from rfdetr.config import RFDETRSegSmallConfig, SegmentationTrainConfig
    from collectra.tasks.image_segmentation.rfdetr_training import training_hooks

    original = native.CSVLogger
    config = SegmentationTrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), epochs=1, use_ema=False,
                                     tensorboard=False, wandb=False, progress_bar=None)
    with training_hooks(tmp_path, .5, True):
        trainer = training.build_trainer(config, RFDETRSegSmallConfig(), accelerator="cpu", devices=1)
    assert native.CSVLogger is original
    (logger,) = [item for item in trainer.loggers if isinstance(item, CSVLogger)]
    assert Path(logger.log_dir) == tmp_path / "lightning"
    # Collectra's own per-epoch file has columns the native logger never logged.
    (tmp_path / "metrics.csv").write_text("epoch,loss_cls,mask_dice\n1,0.5,0.2\n")
    for metrics in ({"train/loss": 1.0}, {"train/loss": 0.9, "train/lr": 0.001}):
        logger.log_metrics(metrics)
        logger.save()
    assert (tmp_path / "metrics.csv").read_text().startswith("epoch,loss_cls,mask_dice")


@pytest.mark.parametrize("has_validation", [True, False])
def test_cpu_epoch_loop_reports_metrics_and_saves_reloadable_weights(tmp_path, has_validation):
    import csv

    torch = pytest.importorskip("torch")
    lightning = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader, TensorDataset
    from collectra.tasks.image_segmentation.rfdetr_training import make_callback

    class Data(lightning.LightningDataModule):
        class_names = ["text_region"]

        def train_dataloader(self):
            return DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=1)

        def val_dataloader(self):
            return self.train_dataloader()

    class Model(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 1)
            self.train_config = SimpleNamespace(model_dump=lambda: {})
            self.model_config = SimpleNamespace(model_dump=lambda: {
                "model_name": "RFDETRSegSmall", "segmentation_head": True,
            })

        def training_step(self, batch, batch_idx):
            loss = self.model(batch[0]).square().mean()
            self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=1)
            return loss

        def validation_step(self, batch, batch_idx):
            return {
                "results": [{"masks": torch.full((1, 1, 2, 2), self.current_epoch == 0),
                             "labels": torch.tensor([0]), "scores": torch.tensor([.9])}],
                "targets": [{"masks": torch.ones(1, 2, 2, dtype=torch.bool), "labels": torch.tensor([0])}],
            }

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.01)

    trainer = lightning.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
        callbacks=[make_callback(tmp_path, .5, has_validation)],
        limit_val_batches=1 if has_validation else 0, num_sanity_val_steps=0,
    )
    trainer.fit(Model(), datamodule=Data())
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert all(row["train/loss"] for row in rows)
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=True)
    assert "optimizer_states" not in checkpoint
    assert checkpoint["args"]["class_names"] == ["text_region"]
    if has_validation:
        assert [float(row["val/mask_iou"]) for row in rows] == [1., 0.]
        assert checkpoint["epoch"] == 0
    else:
        assert checkpoint["epoch"] == 1  # decreasing training loss
