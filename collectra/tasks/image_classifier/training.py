"""Shared preparation of labelled classifier inputs."""

import csv
import json
import math
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

from collectra.commons import PositionalFields
from pathlib import Path
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from collectra.logger import get_logger
from ..base import Task
from ..machine_learning.training import print_distribution_table
from collectra.types.images import Image, ImageCrop
from collectra.types.links import Link


def prepare_classification_inputs(
    processed_inputs: list,
    processed_parents: list,
    input_maps: dict[str, list],
    parent_input_maps: dict[str, list],
) -> list:
    """Use parent images labelled by their child Link node names."""
    labeled_parents = []
    for file_name, children in input_maps.items():
        parents = parent_input_maps.get(file_name, [])
        artefacts = {item.id: item for item in [*parents, *children]}
        for item in artefacts.values():
            if type(item) is Link:
                item.bind(artefacts)
        for child in children:
            if type(child) is not Link:
                raise TypeError(
                    f"Classifier label {child.id!r} in {file_name!r} must be "
                    f"a Link, got {type(child).__name__}"
                )
            if not child.parents:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} has no parent"
                )
            parent_id = (
                child.parents if isinstance(child.parents, str) else child.parents[0]
            )
            parent = artefacts.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} points to "
                    f"missing parent {parent_id!r}; available parent IDs: "
                    f"{[item.id for item in parents]}"
                )
            child.target = parent
            try:
                target = child.resolve()
            except RuntimeError as error:
                raise ValueError(
                    f"Classifier Link {child.id!r} in {file_name!r} cannot "
                    f"resolve parent chain: {error}"
                ) from error
            if not isinstance(target, Image):
                raise TypeError(
                    f"Classifier Link {child.id!r} resolves to "
                    f"{type(target).__name__}, expected Image or ImageCrop"
                )
            child.partition = target.partition
            labeled_parents.append(child)
    return labeled_parents


logger = get_logger(__name__)
console = Console()


class ClassificationTrainResult(PositionalFields, BaseModel):
    save_dir: Path
    results_dict: dict


class TorchClassifierTask(Task):
    """Shared export, optimization, and metrics for PyTorch classification backends."""

    @staticmethod
    def _select_device(device: str = "") -> str:
        import torch

        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _reload(self) -> None:
        if self.original_model_path is None:
            raise ValueError("No checkpoint is available to reload")
        self.model = self.original_model_path
        self._load()

    def _prepare_assets(
        self,
        classes,
        log,
        validation,
        exclude,
        *images,
        min_size=0,
        workers=0,
        use_existing=False,
    ):
        """Materialize labelled images and crops in train/<class> and val/<class>."""
        if min_size < 0:
            raise ValueError("min_size cannot be negative")
        for name in classes:
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(f"Class name must be a directory name: {name!r}")
        train_dir, val_dir = log / "train", log / "val"
        if use_existing and train_dir.exists() and val_dir.exists():
            logger.info(
                "Reusing existing classifier assets in %s and %s",
                train_dir,
                val_dir,
            )
            counts = {split: {name: 0 for name in classes} for split in ("train", "val")}
            for split, directory in (("train", train_dir), ("val", val_dir)):
                for name in classes:
                    class_dir = directory / name
                    if class_dir.exists():
                        counts[split][name] = sum(
                            1 for path in class_dir.iterdir() if path.is_file()
                        )
            print_distribution_table(
                "Class Distribution", classes, counts["train"], counts["val"]
            )
            missing = [name for name in classes if not counts["train"][name]]
            if missing:
                raise ValueError(
                    "No training images for classes in existing dataset: "
                    f"{', '.join(missing)}"
                )
            return train_dir, val_dir
        for directory in (train_dir, val_dir):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(
                    f"Dataset directory is not empty: {directory}. Use a new output directory."
                )
        counts = {split: {name: 0 for name in classes} for split in ("train", "val")}
        for directory in (train_dir, val_dir):
            for name in classes:
                (directory / name).mkdir(parents=True, exist_ok=True)
        jobs = []
        for index, image in enumerate(images):
            if exclude and image.partition == exclude:
                continue
            if image.name not in classes:
                raise ValueError(
                    f"Training label {image.name!r} is not a configured class"
                )
            target = image.resolve() if type(image) is Link else image
            if not isinstance(target, Image):
                raise TypeError(
                    f"Classifier input {image.id!r} does not resolve to an Image"
                )
            if isinstance(target, ImageCrop):
                left, top, right, bottom = target.coordinates()
                if right <= left or bottom <= top:
                    logger.warning("Skipping empty classifier crop %s", image.id)
                    continue
            split = "val" if validation and image.partition == validation else "train"
            destination = log / split / image.name / f"{index:08d}"
            jobs.append((image.id, image.name, split, target, destination))

        export_workers = max(0, int(workers))
        tiff_decode_workers = 1 if export_workers > 1 else None

        def export_job(target, destination):
            return self._export_image(
                target,
                destination,
                min_size,
                tiff_maxworkers=tiff_decode_workers,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                "Exporting classifier assets...", total=len(jobs)
            )
            if export_workers > 0:
                with ThreadPoolExecutor(max_workers=export_workers) as executor:
                    futures = {
                        executor.submit(export_job, target, destination):
                        (image_id, class_name, split)
                        for image_id, class_name, split, target, destination in jobs
                    }
                    for future in as_completed(futures):
                        image_id, class_name, split = futures[future]
                        progress.update(task_id, description=f"Exporting {image_id}")
                        if future.result():
                            counts[split][class_name] += 1
                        progress.advance(task_id)
            else:
                for image_id, class_name, split, target, destination in jobs:
                    progress.update(task_id, description=f"Exporting {image_id}")
                    if export_job(target, destination):
                        counts[split][class_name] += 1
                    progress.advance(task_id)
        print_distribution_table(
            "Class Distribution", classes, counts["train"], counts["val"]
        )
        missing = [name for name in classes if not counts["train"][name]]
        if missing:
            raise ValueError(f"No training images for classes: {', '.join(missing)}")
        return train_dir, val_dir

    def _export_image(
        self, target, destination, min_size, tiff_maxworkers: int | None = None
    ):
        with target.pil() as pixels:
            if min(pixels.size) < max(1, min_size):
                logger.warning(
                    "Skipping classifier image %s below minimum size %d",
                    target.id,
                    min_size,
                )
                return False
            pixels.convert("RGB").save(destination.with_suffix(".png"))
        return True

    def _make_datasets(self, train_dir, val_dir, augmentation):
        from torchvision.datasets import ImageFolder

        return (
            ImageFolder(
                train_dir, transform=self._transforms(training=True, **augmentation)
            ),
            ImageFolder(val_dir, transform=self._transforms(), allow_empty=True),
        )

    def _train(self, *images: Image, **kwargs) -> ClassificationTrainResult:
        import torch
        from torch.utils.data import DataLoader

        epochs, batch = int(kwargs.get("epochs", 10)), int(kwargs.get("batch", 16))
        workers, patience = int(kwargs.get("workers", 8)), int(
            kwargs.get("early_stop", 10)
        )
        fliplr = float(kwargs.get("fliplr", 0.5))
        augmentation = {"fliplr": fliplr}
        if "flipud" in kwargs:
            flipud = float(kwargs["flipud"])
            if not 0 <= flipud <= 1:
                raise ValueError("flipud must be in [0, 1]")
            augmentation["flipud"] = flipud
        if (
            epochs < 1
            or batch < 1
            or workers < 0
            or patience < 0
            or not 0 <= fliplr <= 1
        ):
            raise ValueError(
                "epochs and batch must be positive; workers/early_stop nonnegative; fliplr in [0, 1]"
            )
        if "log" not in kwargs or "base_folder" not in kwargs:
            raise ValueError("Training requires log and base_folder directories")
        available_workers = max(1, os.cpu_count() or 1)
        if workers > 0 and workers > available_workers:
            logger.info(
                "Capping workers from %d to available CPU count %d",
                workers,
                available_workers,
            )
            workers = available_workers
        torch.manual_seed(int(kwargs.get("seed", 0)))
        self._device = self._select_device(kwargs.get("device", ""))
        if kwargs.get("model"):
            self.model = kwargs["model"]
        configured = kwargs.get("classes") or getattr(self, "output", [])
        if isinstance(configured, str):
            configured = [configured]
        classes = sorted(
            set(
                configured
                or [
                    image.name
                    for image in images
                    if not kwargs.get("exclude") or image.partition != kwargs["exclude"]
                ]
            )
        )
        if not classes:
            raise ValueError("No classes provided/found for classifier training")
        log = (Path(kwargs["base_folder"]) / kwargs["log"]).resolve()
        logger.info(
            "Preparing classifier assets at %s (use_existing=%s, workers=%d)",
            log,
            bool(kwargs.get("use_existing", False)),
            workers,
        )
        train_dir, val_dir = self._prepare_assets(
            classes,
            log,
            kwargs.get("validation", ""),
            kwargs.get("exclude", ""),
            *images,
            min_size=kwargs.get("min_size", 0),
            workers=workers,
            use_existing=bool(kwargs.get("use_existing", False)),
        )
        logger.info("Asset preparation complete: train=%s val=%s", train_dir, val_dir)
        # Load preprocessing config before dataset construction so channel
        # detection and normalization logic can run on real sample files.
        logger.info(
            "Loading model for preprocessing (model=%s, pretrained=%s)",
            self.model,
            kwargs.get("pretrained", True),
        )
        self._load(pretrained=kwargs.get("pretrained", True))
        logger.info("Building datasets and detecting channel count")
        train_data, val_data = self._make_datasets(train_dir, val_dir, augmentation)
        # Torchvision stores normalization here; other backends (e.g. Hugging
        # Face) own their preprocessing elsewhere. Logging must not require it.
        preprocessing = getattr(self, "_preprocessing", {})
        logger.info(
            "Datasets ready: train=%d, val=%d, detected_channels=%s, mean_len=%d, std_len=%d",
            len(train_data),
            len(val_data),
            getattr(self, "_input_channels", "unknown"),
            len(preprocessing.get("mean", [])),
            len(preprocessing.get("std", [])),
        )
        freeze = kwargs.get("freeze_backbone", False)
        logger.info("Preparing training model (freeze_backbone=%s)", freeze)
        self._prepare_training_model(classes, kwargs.get("pretrained", True), freeze)
        if hasattr(self.model, "conv1") and hasattr(self.model.conv1, "in_channels"):
            logger.info(
                "Model conv1 channels after preparation: %s",
                self.model.conv1.in_channels,
            )
        train_loader = DataLoader(
            train_data, batch_size=batch, shuffle=True, num_workers=workers
        )
        val_loader = (
            DataLoader(val_data, batch_size=batch, num_workers=workers)
            if len(val_data)
            else None
        )
        logger.info(
            "DataLoaders ready: train_batches=%d, val_batches=%d, batch=%d, workers=%d",
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
            batch,
            workers,
        )
        if val_loader is None and kwargs.get("validation"):
            raise ValueError(
                f"No validation images match partition {kwargs['validation']!r} "
                "after exclusions and size filtering"
            )
        if val_loader is None:
            logger.warning(
                "No validation images; best checkpoint will use training loss, with early stopping disabled"
            )
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=float(kwargs.get("learning_rate", 0.0001)),
            weight_decay=float(kwargs.get("weight_decay", 0.0001)),
        )
        weights_dir = log / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "classes.json").write_text(
            json.dumps({"classes": classes}, indent=2)
        )
        wandb_run = None
        if bool(kwargs.get("wandb", False)):
            try:
                import wandb

                wandb_config = {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in kwargs.items()
                }
                wandb_config["classes"] = classes
                wandb_run = wandb.init(
                    project=str(kwargs.get("project", "collectra-classifier")),
                    name=log.name,
                    dir=str(kwargs["base_folder"]),
                    config=wandb_config,
                    reinit=True,
                )
            except Exception as error:
                logger.warning("W&B init skipped: %s", error)
        history, best_accuracy, best_metrics, stale = [], -math.inf, {}, 0
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                epoch_task_id = progress.add_task("Training epochs...", total=epochs)
                for epoch in range(1, epochs + 1):
                    progress.update(epoch_task_id, description=f"Training epoch {epoch}/{epochs}")
                    logger.info("Starting epoch %d/%d", epoch, epochs)
                    self.model.train(
                        not freeze
                    )  # Frozen BatchNorm statistics must remain frozen too.
                    train_metrics = self._epoch(train_loader, optimizer)
                    self.model.eval()
                    with torch.inference_mode():
                        val_metrics = (
                            self._epoch(val_loader) if val_loader is not None else None
                        )
                    metrics = {
                        "epoch": epoch,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                    }
                    metrics.update({f"val_{k}": v for k, v in (val_metrics or {}).items()})
                    history.append(metrics)
                    checkpoint = self._checkpoint(epoch, metrics)
                    torch.save(checkpoint, weights_dir / "last.pt")
                    score = val_metrics["accuracy"] if val_metrics else train_metrics["accuracy"]
                    if not math.isfinite(score):
                        raise ValueError("Training produced a non-finite accuracy")
                    if score > best_accuracy:
                        best_accuracy, best_metrics, stale = score, dict(metrics), 0
                        torch.save(checkpoint, weights_dir / "best.pt")
                        best_print = " best"
                    else:
                        stale += 1
                        best_print = ""
                    if wandb_run is not None:
                        wandb_run.log(metrics)
                    logger.info("Epoch %d/%d: %s", epoch, epochs, metrics)
                    metrics_line = (
                        f"Epoch {epoch}/{epochs} "
                        f"train_loss={metrics['train_loss']:.4f} "
                        f"train_acc={metrics['train_accuracy']:.4f} "
                        f"train_top5={metrics['train_top5_accuracy']:.4f}"
                    )
                    if val_metrics is not None:
                        metrics_line += (
                            f" val_loss={metrics['val_loss']:.4f} "
                            f"val_acc={metrics['val_accuracy']:.4f} "
                            f"val_top5={metrics['val_top5_accuracy']:.4f}"
                        )
                    console.print(metrics_line + best_print)
                    with (log / "history.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(metrics))
                        writer.writeheader()
                        writer.writerows(history)
                    progress.advance(epoch_task_id)
                    if val_loader is not None and patience and stale >= patience:
                        console.print(
                            f"Early stopping at epoch {epoch}: no val-accuracy improvement in {patience} epoch(s)."
                        )
                        break
        finally:
            if wandb_run is not None:
                wandb_run.finish()
        self.original_model_path = weights_dir / "best.pt"
        self._reload()
        best_metrics.update(classes=classes, epochs_completed=len(history))
        (log / "metrics.json").write_text(json.dumps(best_metrics, indent=2))
        return ClassificationTrainResult(save_dir=log, results_dict=best_metrics)

    def _epoch(self, loader, optimizer=None) -> dict:
        import torch
        from torch.nn import functional as F
        from sklearn.metrics import precision_recall_fscore_support

        total, correct, correct5, total_loss = 0, 0, 0, 0.0
        all_preds, all_labels = [], []
        first_batch = True
        phase = "train" if optimizer is not None else "val"
        for pixels, labels in loader:
            # Torchvision batches are tensors; Hugging Face batches are dicts
            # of processor outputs keyed by model input name.
            tensors = pixels if isinstance(pixels, Mapping) else {"pixels": pixels}
            shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
            if first_batch:
                approximate_mib = sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors.values()
                ) / (1024 * 1024)
                logger.info(
                    "%s first batch: input_shapes=%s, labels_shape=%s, dtypes=%s, approx_input_mib=%.1f",
                    phase,
                    shapes,
                    tuple(labels.shape),
                    {name: tensor.dtype for name, tensor in tensors.items()},
                    approximate_mib,
                )
                first_batch = False
            labels = labels.to(self._device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            try:
                logits = self._forward(pixels)
                loss = F.cross_entropy(logits, labels)
            except RuntimeError as error:
                raise RuntimeError(
                    f"{phase} step failed for input shapes {shapes} "
                    f"on device {self._device}: {error}"
                ) from error
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            correct5 += (
                (
                    logits.topk(min(5, len(self._categories)), dim=1).indices
                    == labels[:, None]
                )
                .any(1)
                .sum()
                .item()
            )
            all_preds.extend(logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "loss": total_loss / total,
            "accuracy": correct / total,
            "top5_accuracy": correct5 / total,
        }
