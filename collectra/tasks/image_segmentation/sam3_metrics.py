"""Optional SAM3 training metrics, reporting and weights-only checkpoint hooks."""

import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path


class MaskIoU:
    """Pixel-weighted foreground IoU across validation queries and all ranks."""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.reset()

    def reset(self):
        self.intersection = self.union = self.queries = 0

    def update(self, *, find_stages, batch, **kwargs):
        import torch
        from torch.nn import functional as F

        with torch.no_grad():
            for outputs, target in zip(find_stages, batch.find_targets):
                masks = outputs["pred_masks"].detach()
                scores = (
                    outputs["pred_logits"].detach().sigmoid().reshape(masks.shape[:2])
                )
                presence = outputs.get("presence_logit_dec")
                if presence is not None:
                    scores = scores * presence.detach().sigmoid().reshape(-1, 1)
                offset = 0
                for query, count in enumerate(target.num_boxes.tolist()):
                    segments = target.segments[offset : offset + count]
                    validity = target.is_valid_segment
                    valid = validity is None or bool(
                        validity[offset : offset + count].all()
                    )
                    offset += count
                    if not valid:
                        continue
                    truth = (
                        segments.bool().any(dim=0)
                        if count
                        else torch.zeros(
                            batch.img_batch.shape[-2:],
                            device=masks.device,
                            dtype=torch.bool,
                        )
                    )
                    predicted = torch.zeros_like(truth)
                    selected = masks[query, scores[query] >= self.confidence_threshold]
                    # Bound temporary memory when many instances are predicted.
                    for chunk in selected.split(8):
                        if len(chunk):
                            pixels = (
                                F.interpolate(
                                    chunk[:, None].float(),
                                    size=truth.shape,
                                    mode="bilinear",
                                    align_corners=False,
                                )[:, 0]
                                > 0
                            )
                            predicted |= pixels.any(dim=0)
                    self.intersection += int((predicted & truth).sum())
                    self.union += int((predicted | truth).sum())
                    self.queries += 1

    def compute(self):
        """Local progress value; never synchronize from a progress callback."""
        return {
            "iou": self.intersection / self.union
            if self.union
            else (1.0 if self.queries else float("nan"))
        }

    def compute_synced(self):
        import torch
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if distributed and dist.get_backend() == "nccl"
            else "cpu"
        )
        totals = torch.tensor(
            [self.intersection, self.union, self.queries],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(totals)
        intersection, union, queries = totals.tolist()
        return {
            "iou": intersection / union if union else (1.0 if queries else float("nan"))
        }


class EpochReporter:
    """Keep one CSV row per epoch, merging train and validation reports."""

    def __init__(self, output, *, wandb=False, run_name=None, project=None, config=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.run = None
        if wandb:
            import wandb as sdk

            self.run = sdk.init(
                project=project or "collectra",
                name=run_name,
                dir=str(self.output),
                config=config,
            )
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")

    def record(self, epoch, phase, metrics):
        values = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[f"{phase}/{key}" if phase else key] = value
        row = self.rows.setdefault(epoch, {"epoch": epoch})
        row.update(values)
        columns = ["epoch"] + sorted(
            {key for row in self.rows.values() for key in row if key != "epoch"}
        )
        temporary = self.output / "metrics.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows[key] for key in sorted(self.rows))
        os.replace(temporary, self.output / "metrics.csv")
        if self.run is not None:
            self.run.log({"epoch": epoch, **values})

    def close(self):
        if self.run is not None:
            self.run.finish()


class TrainingHooks:
    """Mixin for Meta's trainer; model/backend imports remain lazy."""

    def __init__(
        self,
        *,
        collectra_output,
        collectra_wandb=False,
        collectra_run_name=None,
        collectra_project=None,
        collectra_train_eval=None,
        collectra_mask_threshold=0.5,
        **kwargs,
    ):
        self._collectra_best = None
        self._collectra_output = Path(collectra_output)
        self._collectra_has_validation = bool(kwargs["data"].get("val"))
        self._collectra_reporter = None
        self._collectra_mask_threshold = collectra_mask_threshold
        super().__init__(**kwargs)
        self._collectra_train_eval = None
        if collectra_train_eval:
            from hydra.utils import instantiate

            self._collectra_train_eval = instantiate(collectra_train_eval)
        if self.distributed_rank == 0:
            self._collectra_reporter = EpochReporter(
                collectra_output,
                wandb=collectra_wandb,
                run_name=collectra_run_name,
                project=collectra_project,
                config={
                    "epochs": self.max_epochs,
                    "selection_metric": (
                        "val/mask_iou"
                        if self._collectra_has_validation
                        else "train/loss"
                    ),
                },
            )

    def load_checkpoint(self):
        # Each invocation fine-tunes the explicitly configured model weights.
        # Old optimizer checkpoints in a reused output directory must not resume.
        self._init_model_state()

    def save_checkpoint(self, epoch, checkpoint_names=None):
        # Suppress native periodic, last-epoch and per-meter optimizer snapshots.
        # Selection happens only after the epoch's metric has been computed.
        return

    def _save_best(self, score, metric, *, maximize):
        import torch

        if not math.isfinite(score):
            raise ValueError(f"Cannot select a best SAM3 model: {metric} is not finite")
        improved = self._collectra_best is None or (
            score > self._collectra_best if maximize else score < self._collectra_best
        )
        if not improved:
            return
        self._collectra_best = score
        if self.distributed_rank != 0:
            return
        directory = self._collectra_output / "training/checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        descriptor, name = tempfile.mkstemp(
            prefix=".best-", suffix=".pt", dir=directory
        )
        os.close(descriptor)
        try:
            torch.save(model.state_dict(), name)
            os.replace(name, directory / "best.pt")
        finally:
            Path(name).unlink(missing_ok=True)
        metadata = {"epoch": self.epoch + 1, "metric": metric, "value": score}
        (directory / "best.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logging.info(
            "Best SAM3 weights: epoch=%s, %s=%s", self.epoch + 1, metric, score
        )

    def train_epoch(self, train_loader):
        metrics = super().train_epoch(train_loader)
        loss = metrics.get("Losses/train_all_loss")
        if loss is not None:
            metrics["loss"] = float(loss)
        if not self._collectra_has_validation:
            if loss is None:
                raise ValueError("SAM3 training did not report its training loss")
            self._save_best(float(loss), "train/loss", maximize=False)
        if self._collectra_reporter is not None:
            self._collectra_reporter.record(self.epoch + 1, "train", metrics)
        if not self._collectra_has_validation:
            self._collectra_record_train_iou()
        return metrics

    def _collectra_record_train_iou(self):
        """Score the training images like validation and report train/mask_iou."""
        if self._collectra_train_eval is None:
            return
        import time

        import torch
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.utils.train_utils import get_amp_type

        started = time.monotonic()
        meter = MaskIoU(self._collectra_mask_threshold)
        amp = self.optim_conf.amp if self.optim_conf else None
        self.model.eval()
        loader = self._collectra_train_eval.get_loader(epoch=int(self.epoch))
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda",
            enabled=bool(amp and amp.enabled),
            dtype=get_amp_type(amp.amp_dtype) if amp else None,
        ):
            for batch in loader:
                _, batch = batch.popitem()
                batch = copy_data_to_device(batch, self.device, non_blocking=True)
                meter.update(find_stages=self.model(batch), batch=batch)
        score = meter.compute_synced()["iou"]
        logging.info(
            "Train mask IoU (eval pass): epoch=%s, iou=%s, %.0fs",
            self.epoch + 1, score, time.monotonic() - started,
        )
        if self._collectra_reporter is not None:
            self._collectra_reporter.record(self.epoch + 1, "train", {"mask_iou": score})

    def val_epoch(self, val_loader, phase):
        metrics = super().val_epoch(val_loader, phase)
        values = [
            value
            for key, value in metrics.items()
            if key.replace("\\", "/").endswith("/mask_iou/iou")
        ]
        if len(values) != 1:
            raise ValueError("SAM3 validation did not report its mask IoU")
        metrics["mask_iou"] = float(values[0])
        loss = metrics.get("Losses/val_roboflow100_loss")
        if loss is not None:
            metrics["loss"] = float(loss)
        self._save_best(metrics["mask_iou"], "val/mask_iou", maximize=True)
        if self._collectra_reporter is not None:
            self._collectra_reporter.record(self.epoch + 1, "val", metrics)
        self._collectra_record_train_iou()
        return metrics

    def run(self):
        try:
            return super().run()
        finally:
            if self._collectra_reporter is not None:
                self._collectra_reporter.close()


def create_trainer(**kwargs):
    from sam3.train.trainer import Trainer

    class CollectraTrainer(TrainingHooks, Trainer):
        pass

    return CollectraTrainer(**kwargs)
