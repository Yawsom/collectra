from __future__ import annotations

from pydantic import BaseModel, Field

from collectra.commons import PositionalFields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from collectra.cli import command
from collectra.types.images import Image, ImageCrop, open_oriented
from collectra.utils import change_dir

from ...logger import get_logger
from ..base import Task
from .base import (
    Batch,
    Epochs,
    LearningRate,
    ObjectDetectionTask,
    TrainResolution,
    WeightDecay,
)
from collectra.utils import threading_locked
from ..machine_learning.training import (
    prepare_object_detection_inputs,
    print_distribution_table,
    run_training_command,
)

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)

__all__ = ["ObjectDetectionDETR", "DetectionTrainResult"]


class DetectionTrainResult(PositionalFields, BaseModel):
    save_dir: Path
    results_dict: dict


class ObjectDetectionDETR(ObjectDetectionTask):
    """DETR task backed by Hugging Face Transformers."""

    model: Any = None  # a path or a loaded torch module
    original_model_path: str | Path | None = None
    batch: int = Field(default=2, ge=1)  # DETR is memory hungry

    def __init__(
        self,
        name: str,
        model: str | Path | nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(name, model=model, **kwargs)
        self._device = None
        self._processor = None
        self._categories: list[str] = []
        self._model_name = "facebook/detr-resnet-50"

    def _load_processor(self, model_name: str, resolution: int | None):
        """The image processor, resizing to ``resolution`` when one is given."""
        from transformers import DetrImageProcessor

        kwargs = {}
        if resolution is not None:
            kwargs["size"] = {"shortest_edge": resolution, "longest_edge": resolution}
        return DetrImageProcessor.from_pretrained(model_name, **kwargs)

    def _init_model(self) -> None:
        from torch import nn

        self._load()
        if not isinstance(self.model, nn.Module):
            raise ValueError("Model must be a torch.nn.Module instance")

    def _reload(self) -> None:
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
            return
        self.model = self.original_model_path
        self._load()

    def _load_model_from_hf(self, model_name: str) -> None:
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self._model_name = model_name
        self._processor = self._load_processor(model_name, self.inference_resolution)
        model = DetrForObjectDetection.from_pretrained(model_name)
        self._categories = [
            model.config.id2label[i] for i in sorted(model.config.id2label.keys())
        ]
        self.model = model.to(self._device)
        self.model.eval()

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> None:
        import torch
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            classes = checkpoint.get("classes", [])
            model_name = checkpoint.get("model_name", self._model_name)
        else:
            state_dict = checkpoint
            classes = []
            model_name = self._model_name

        self._model_name = model_name
        self._processor = self._load_processor(
            self._model_name, self.inference_resolution
        )

        if classes:
            id2label = {idx: name for idx, name in enumerate(classes)}
            label2id = {name: idx for idx, name in enumerate(classes)}
            config = DetrConfig.from_pretrained(
                self._model_name,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
            )
            model = DetrForObjectDetection.from_pretrained(
                self._model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
            self._categories = classes
        else:
            model = DetrForObjectDetection.from_pretrained(self._model_name)
            self._categories = [
                model.config.id2label[i] for i in sorted(model.config.id2label.keys())
            ]

        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        self.model = model

    def _load(self) -> None:
        from torch import nn

        if isinstance(self.model, nn.Module):
            if self._device is None:
                self._device = self._select_device()
            self.model.to(self._device)
            self.model.eval()
            return

        self._device = self._select_device()

        if self.model is None or (
            isinstance(self.model, str) and self.model == "default"
        ):
            self._load_model_from_hf("facebook/detr-resnet-50")
            return

        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be None, 'default', a path, a model id, or nn.Module"
            )

        model_value = str(self.model)
        checkpoint_path = Path(model_value)
        if checkpoint_path.exists() and checkpoint_path.is_file():
            self.original_model_path = checkpoint_path
            self._load_model_from_checkpoint(checkpoint_path)
            return

        # Treat as Hugging Face model id if local file path does not exist.
        self._load_model_from_hf(model_value)

    @threading_locked()
    def run(self, *args: Image) -> list[Image]:
        import torch

        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")

        image = args[0]
        self._init_model()
        from torch import nn

        if not isinstance(self.model, nn.Module) or self._processor is None:
            raise ValueError("DETR model is not initialised properly.")

        threshold = self.threshold
        pil_image = image.pil().convert("RGB")
        width, height = pil_image.size

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self._device)
        processed = self._processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        detections: list[Image] = []
        for box, label in zip(processed["boxes"], processed["labels"]):
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            x1 = max(0.0, min(float(width), float(x1)))
            x2 = max(0.0, min(float(width), float(x2)))
            y1 = max(0.0, min(float(height), float(y1)))
            y2 = max(0.0, min(float(height), float(y2)))

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            width_relative = bbox_width / width
            height_relative = bbox_height / height

            class_idx = int(label.detach().cpu().item())
            class_name = (
                self._categories[class_idx]
                if 0 <= class_idx < len(self._categories)
                else str(class_idx)
            )

            detections.append(
                image.make_crop(
                    x_center=x_center,
                    y_center=y_center,
                    width_relative=width_relative,
                    height_relative=height_relative,
                    orientation=image.orientation,
                    name=class_name,
                )
            )

        print(f"Found {len(detections)} objects in the image.")
        return detections

    def _prepare_params(self, **kwargs) -> dict:
        model_name = kwargs.get("model_name", self._model_name)
        if isinstance(self.model, str) and self.model not in ("", "default"):
            if not Path(self.model).exists():
                model_name = self.model
        return {
            "name": kwargs["log"],
            "project": kwargs.get("project", "runs/detr"),
            "device": self._select_device(),
            "epochs": int(kwargs.get("epochs", 1)),
            "batch": int(kwargs.get("batch", 2)),
            "workers": int(kwargs.get("workers", 0)),
            "lr": float(kwargs.get("lr", 1e-4)),
            "weight_decay": float(kwargs.get("weight_decay", 1e-4)),
            "pretrained": bool(kwargs.get("pretrained", True)),
            "model_name": model_name,
        }

    def _check_distribution(
        self,
        classes: list[str],
        train_samples: list[dict],
        val_samples: list[dict],
    ) -> None:
        train_classes = {name: 0 for name in classes}
        val_classes = {name: 0 for name in classes}
        label_to_name = {index: name for index, name in enumerate(classes)}

        for sample in train_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    train_classes[class_name] += 1

        for sample in val_samples:
            for label in sample["labels"]:
                class_name = label_to_name.get(label, "")
                if class_name:
                    val_classes[class_name] += 1

        print_distribution_table(
            "Class Distribution", classes, train_classes, val_classes
        )

    def _prepare_assets(
        self,
        classes: list[str],
        validation_flag: str,
        exclude_flag: str,
        *images: ImageCrop,
    ) -> tuple[list[dict], list[dict]]:
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        grouped: dict[Path, dict] = {}

        for img in images:
            if exclude_flag and img.partition == exclude_flag:
                continue

            base_img: Image | ImageCrop = img
            if isinstance(img, ImageCrop):
                if img.source_parent:
                    base_img = img.source_parent
                    if isinstance(img.source_parent, ImageCrop):
                        img.set_rel_to_src_parent()

            image_path = base_img.get_path()
            if image_path not in grouped:
                grouped[image_path] = {
                    "image_path": image_path,
                    "partition": img.partition,
                    "boxes": [],
                    "labels": [],
                }

            if not isinstance(img, ImageCrop):
                continue
            if img.name not in class_to_idx:
                continue

            width, height = base_img.width, base_img.height
            box_w = float(img.width_relative) * width
            box_h = float(img.height_relative) * height
            cx = float(img.x_center) * width
            cy = float(img.y_center) * height

            x1 = max(0.0, cx - box_w / 2.0)
            y1 = max(0.0, cy - box_h / 2.0)
            x2 = min(float(width), cx + box_w / 2.0)
            y2 = min(float(height), cy + box_h / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue

            grouped[image_path]["boxes"].append([x1, y1, x2, y2])
            grouped[image_path]["labels"].append(class_to_idx[img.name])

        train_samples: list[dict] = []
        val_samples: list[dict] = []
        for sample in grouped.values():
            if validation_flag and sample["partition"] == validation_flag:
                val_samples.append(sample)
            elif not exclude_flag or sample["partition"] != exclude_flag:
                train_samples.append(sample)
        return train_samples, val_samples

    def _train_fold(
        self,
        train_samples: list[dict],
        val_samples: list[dict],
        classes: list[str],
        log: Path,
        kwargs: dict,
    ) -> DetectionTrainResult:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

        params_kwargs = dict(kwargs)
        params_kwargs["log"] = f"{log.name}"
        params = self._prepare_params(**params_kwargs)
        self._device = params["device"]
        self._model_name = params["model_name"]

        self._processor = self._load_processor(self._model_name, self.resolution)

        class DetrDataset(Dataset):
            def __init__(self, samples: list[dict], processor: DetrImageProcessor):
                self.samples = samples
                self.processor = processor

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx: int):
                sample = self.samples[idx]
                with open_oriented(sample["image_path"]).convert("RGB") as pil_image:
                    annotations = {
                        "image_id": idx,
                        "annotations": [],
                    }
                    for box, label in zip(sample["boxes"], sample["labels"]):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        annotations["annotations"].append(
                            {
                                "bbox": [x1, y1, w, h],
                                "category_id": int(label),
                                "area": float(w * h),
                                "iscrowd": 0,
                            }
                        )

                    encoding = self.processor(
                        images=pil_image,
                        annotations=annotations,
                        return_tensors="pt",
                    )

                pixel_values = encoding["pixel_values"].squeeze(0)
                labels = encoding["labels"][0]
                return pixel_values, labels

        def collate_fn(batch):
            import torch

            pixel_values = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            max_h = max(pv.shape[1] for pv in pixel_values)
            max_w = max(pv.shape[2] for pv in pixel_values)
            batch_size = len(pixel_values)

            pixel_values_padded = torch.zeros(
                (batch_size, 3, max_h, max_w),
                dtype=pixel_values[0].dtype,
            )
            pixel_mask = torch.zeros((batch_size, max_h, max_w), dtype=torch.long)

            for i, pv in enumerate(pixel_values):
                h, w = pv.shape[1], pv.shape[2]
                pixel_values_padded[i, :, :h, :w] = pv
                pixel_mask[i, :h, :w] = 1

            return {
                "pixel_values": pixel_values_padded,
                "pixel_mask": pixel_mask,
                "labels": labels,
            }

        train_loader = DataLoader(
            DetrDataset(train_samples, self._processor),
            batch_size=params["batch"],
            shuffle=True,
            num_workers=params["workers"],
            collate_fn=collate_fn,
        )

        val_loader = None
        if len(val_samples) > 0:
            val_loader = DataLoader(
                DetrDataset(val_samples, self._processor),
                batch_size=params["batch"],
                shuffle=False,
                num_workers=params["workers"],
                collate_fn=collate_fn,
            )

        id2label = {idx: name for idx, name in enumerate(classes)}
        label2id = {name: idx for idx, name in enumerate(classes)}

        from torch import nn

        if isinstance(self.model, nn.Module):
            model = self.model
        else:
            config = DetrConfig.from_pretrained(
                self._model_name,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
            )
            model = DetrForObjectDetection.from_pretrained(
                self._model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )

        model.to(self._device)
        self.model = model
        self._categories = classes

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        best_score = float("inf")
        best_metrics: dict = {}
        weights_dir = log / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        wandb_run = None
        if bool(kwargs.get("wandb", False)):
            try:
                import wandb

                wandb_run = wandb.init(
                    project=params["project"],
                    name=params["name"],
                    dir=str(kwargs["base_folder"]),
                    config=params,
                    reinit=True,
                )
            except Exception as error:
                logger.warning("W&B init skipped: %s", error)

        def move_labels_to_device(batch_labels: list[dict]) -> list[dict]:
            out = []
            for label_dict in batch_labels:
                out.append({k: v.to(self._device) for k, v in label_dict.items()})
            return out

        def run_epoch(data_loader: DataLoader, train_mode: bool) -> float:
            if train_mode:
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_count = 0

            for batch in data_loader:
                pixel_values = batch["pixel_values"].to(self._device)
                pixel_mask = batch["pixel_mask"].to(self._device)
                labels = move_labels_to_device(batch["labels"])

                if train_mode:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(train_mode):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    if train_mode:
                        loss.backward()
                        optimizer.step()

                running_loss += float(loss.detach().cpu().item())
                running_count += 1

            return running_loss / max(running_count, 1)

        for epoch in range(params["epochs"]):
            train_loss = run_epoch(train_loader, train_mode=True)
            val_loss = run_epoch(val_loader, train_mode=False) if val_loader else None
            metric_to_track = val_loss if val_loss is not None else train_loss

            epoch_metrics = {"epoch": epoch + 1, "train/loss": train_loss}
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss

            if metric_to_track < best_score:
                best_score = metric_to_track
                best_metrics = epoch_metrics.copy()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classes": classes,
                        "epoch": epoch + 1,
                        "metrics": epoch_metrics,
                        "model_name": self._model_name,
                    },
                    weights_dir / "best.pt",
                )

            if wandb_run is not None:
                wandb_run.log(epoch_metrics)

            logger.info("DETR epoch %s metrics: %s", epoch + 1, epoch_metrics)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "epoch": params["epochs"],
                "metrics": best_metrics,
                "model_name": self._model_name,
            },
            weights_dir / "last.pt",
        )

        if wandb_run is not None:
            wandb_run.finish()

        model.eval()
        self.original_model_path = weights_dir / "best.pt"
        self._reload()
        return DetectionTrainResult(save_dir=log, results_dict=best_metrics)

    @command
    def train(
        self,
        inputs: list[str],
        output: Path | None = None,
        keep_log: bool = True,
        validation: str = "",
        exclude: str = "",
        epochs: Epochs = None,
        batch: Batch = None,
        workers: int = 0,
        learning_rate: LearningRate = None,
        weight_decay: WeightDecay = None,
        pretrained: bool = True,
        model_name: str = "facebook/detr-resnet-50",
        wandb: bool = False,
        resolution: TrainResolution = None,
    ):
        """Train this DETR object detector."""
        settings = self._settings_for_run(
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            resolution=resolution,
        )
        return run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=prepare_object_detection_inputs,
            include_unlabelled=True,
            epochs=settings["epochs"],
            batch=settings["batch"],
            workers=workers,
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            resolution=settings["resolution"],
            pretrained=pretrained,
            model_name=model_name,
            wandb=wandb,
        )

    def _train(self, *images: ImageCrop, **kwargs) -> DetectionTrainResult:
        log = kwargs.get("log", None)
        if log is None:
            raise ValueError(
                "Log directory must be specified in kwargs with key 'log'."
            )
        if "base_folder" not in kwargs:
            raise ValueError(
                "Base folder must be specified in kwargs with key 'base_folder'."
            )

        log = kwargs["base_folder"] / log
        log.mkdir(parents=True, exist_ok=True)
        print(f"Training files will be saved to: {log}")

        classes = kwargs.get("classes", [])
        if not classes:
            classes = sorted({img.name for img in images if isinstance(img, ImageCrop)})
        if not classes:
            raise ValueError("No classes provided/found for DETR training.")

        print("Training with model: DETR")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train_samples, val_samples = self._prepare_assets(
            classes,
            validation,
            exclude,
            *images,
        )
        self._check_distribution(classes, train_samples, val_samples)

        with change_dir(kwargs["base_folder"]):
            results = self._train_fold(train_samples, val_samples, classes, log, kwargs)

        print(f"Best DETR model saved to: {results.save_dir / 'weights' / 'best.pt'}")
        return results
