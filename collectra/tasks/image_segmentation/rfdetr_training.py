"""Optional RF-DETR training integration; no backend imports during discovery."""

from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import time


from .mask_metrics import mask_overlap


def rfdetr_train_predictions(output):
    """Predict the training images like validation: eval mode, no augmentation.

    Returns a function yielding (results, targets) batches for make_callback.
    RF-DETR builds non-augmented loaders only for its fixed split names, so a
    view directory OUTPUT/train-eval links both "train" (label mapping) and
    "valid" (images) to the exported training split.
    """
    cache = {}

    def datamodule(pl_module):
        if "module" not in cache:
            from rfdetr.training.module_data import RFDETRDataModule

            images = Path(pl_module.train_config.dataset_dir).resolve() / "train"
            view = Path(output) / "train-eval"
            view.mkdir(exist_ok=True)
            for name in ("train", "valid"):
                try:
                    (view / name).symlink_to(images, target_is_directory=True)
                except FileExistsError:
                    if (view / name).resolve() != images:
                        raise
            config = pl_module.train_config.model_copy(update={"dataset_dir": str(view)})
            cache["module"] = RFDETRDataModule(pl_module.model_config, config)
            cache["module"].setup("validate")
        return cache["module"]

    def predict(trainer, pl_module):
        import torch

        module = datamodule(pl_module)
        model = pl_module._resolve_eval_model()
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode(), trainer.precision_plugin.val_step_context():
                for batch in module.val_dataloader():
                    samples, targets = module.transfer_batch_to_device(batch, pl_module.device, 0)
                    sizes = torch.stack([target["orig_size"] for target in targets])
                    yield pl_module.postprocess(model(samples), sizes), targets
        finally:
            model.train(was_training)

    return predict


def make_callback(output, threshold, has_validation, train_predictions=None):
    """Build the optional Lightning callback after training dependencies load.

    train_predictions(trainer, pl_module), when given, yields (results, targets)
    for the training images; their mask IoU is logged as train/mask_iou.
    """
    import torch
    from pytorch_lightning.callbacks import Callback

    class SegmentationMetrics(Callback):
        def __init__(self):
            self.best = -math.inf
            self.rows = {}

        def on_fit_start(self, trainer, pl_module):
            if trainer.is_global_zero:
                for name in ("best.pt", "best.json"):
                    (Path(output) / name).unlink(missing_ok=True)

        def on_validation_epoch_start(self, trainer, pl_module):
            self.overlap = torch.zeros(2, dtype=torch.int64, device=pl_module.device)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if trainer.sanity_checking:
                return
            for prediction, target in zip(outputs["results"], outputs["targets"]):
                self.overlap += mask_overlap(prediction, target, threshold)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(self.overlap)
            score = self.overlap[0] / self.overlap[1].clamp_min(1)
            pl_module.log("val/mask_iou", score.float(), prog_bar=True, sync_dist=False)
            self._log_train_iou(trainer, pl_module)

        def _log_train_iou(self, trainer, pl_module):
            """Score the training images like validation and log train/mask_iou.

            Every rank scores all images (no sampler sharding), so no reduction is needed.
            """
            if train_predictions is None:
                return
            started = time.monotonic()
            overlap = torch.zeros(2, dtype=torch.int64, device=pl_module.device)
            for results, targets in train_predictions(trainer, pl_module):
                for prediction, target in zip(results, targets):
                    overlap += mask_overlap(prediction, target, threshold)
            score = overlap[0] / overlap[1].clamp_min(1)
            pl_module.log("train/mask_iou", score.float(), sync_dist=False)
            print(
                f"Train mask IoU (eval pass): epoch={trainer.current_epoch + 1}, "
                f"iou={float(score):.4f}, {time.monotonic() - started:.0f}s",
                flush=True,
            )

        def _report(self, trainer):
            if not trainer.is_global_zero or trainer.sanity_checking:
                return
            row = self.rows.setdefault(trainer.current_epoch + 1, {"epoch": trainer.current_epoch + 1})
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, "numel") and value.numel() != 1:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                row[key] = value
            fields = ["epoch"] + sorted({key for row in self.rows.values() for key in row if key != "epoch"})
            with (Path(output) / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows.values())

        def _save(self, trainer, pl_module, metric, score):
            if not trainer.is_global_zero or not math.isfinite(score) or score <= self.best:
                return
            self.best = score
            args = pl_module.train_config.model_dump()
            args["class_names"] = list(trainer.datamodule.class_names)
            config = pl_module.model_config.model_dump()
            checkpoint = {
                "model": pl_module.model.state_dict(), "args": args,
                "model_config": config, "model_name": config["model_name"],
                "epoch": trainer.current_epoch,
            }
            temporary = Path(output) / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(Path(output) / "best.pt")
            (Path(output) / "best.json").write_text(json.dumps({
                "epoch": trainer.current_epoch + 1, "metric": metric,
                "value": -score if metric == "train/loss" else score,
            }), encoding="utf-8")

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            self._report(trainer)
            score = trainer.callback_metrics.get("val/mask_iou")
            if score is not None:
                self._save(trainer, pl_module, "val/mask_iou", float(score))

        def on_train_epoch_end(self, trainer, pl_module):
            if not has_validation:
                self._log_train_iou(trainer, pl_module)
            self._report(trainer)
            if not has_validation:
                score = trainer.callback_metrics.get("train/loss")
                if score is not None:
                    self._save(trainer, pl_module, "train/loss", -float(score))

    return SegmentationMetrics()


@contextmanager
def training_hooks(output, threshold, has_validation):
    """Replace native optimizer snapshots with one best, reloadable weights file.

    RF-DETR's train() constructs its own callbacks. Scope the extension to this
    training process, leaving its COCO evaluation and W&B/CSV loggers intact.
    RF-DETR's CSV logger is moved to OUTPUT/lightning: it also writes
    metrics.csv, and rereads that file whenever new metric keys appear, so
    sharing Collectra's per-epoch OUTPUT/metrics.csv crashes training.
    """
    import rfdetr.training.trainer as native
    from pytorch_lightning.callbacks import ModelCheckpoint

    original = native._append_training_callbacks

    def append(callbacks, loggers, *args, **kwargs):
        original(callbacks, loggers, *args, **kwargs)
        callbacks[:] = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        callbacks.append(make_callback(
            output, threshold, has_validation, train_predictions=rfdetr_train_predictions(output),
        ))

    # Also disable Lightning's implicit default checkpoint when none remains.
    import rfdetr.training as training
    original_builder = training.build_trainer

    def build(*args, **kwargs):
        kwargs["enable_checkpointing"] = False
        if not has_validation:
            kwargs["limit_val_batches"] = 0
        return original_builder(*args, **kwargs)

    original_csv_logger = native.CSVLogger

    def csv_logger(save_dir, *args, **kwargs):
        return original_csv_logger(Path(save_dir) / "lightning", *args, **kwargs)

    native._append_training_callbacks = append
    training.build_trainer = build
    native.CSVLogger = csv_logger
    try:
        yield
    finally:
        native._append_training_callbacks = original
        training.build_trainer = original_builder
        native.CSVLogger = original_csv_logger


def launch(config_path):
    from .rfdetr import ImageSegmentationRFDETR

    config = json.loads(Path(config_path).read_text())
    task = ImageSegmentationRFDETR(
        "train", model=config["model"], device=config["device"], resolution=config["resolution"],
    )
    with training_hooks(config["params"]["output_dir"], config["threshold"], config["has_validation"]):
        task._init_model()
        task._predictor.train(**config["params"])


if __name__ == "__main__":
    import sys

    launch(sys.argv[1])
