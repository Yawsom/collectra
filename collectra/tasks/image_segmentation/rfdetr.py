"""RF-DETR instance segmentation with Collectra's shared mask training format."""

from __future__ import annotations

import json
import shutil
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from cappa import Arg
import numpy as np
from PIL import Image as PILImage

from collectra.cli import command
from collectra.tasks.base import Task
from collectra.types.images import Image, ImageSegmentation
from collectra.utils import threading_locked

from .data import SegmentationTrainingData, mask_to_source


VARIANTS = {
    size: f"RFDETRSeg{suffix}"
    for size, suffix in (
        ("nano", "Nano"), ("small", "Small"), ("medium", "Medium"),
        ("large", "Large"), ("xlarge", "XLarge"), ("2xlarge", "2XLarge"),
    )
}


def _rfdetr():
    try:
        import rfdetr
    except ModuleNotFoundError as error:
        if error.name != "rfdetr":
            raise
        raise ImportError(
            'RF-DETR segmentation requires optional dependencies: pip install "collectra[ml]"'
        ) from error
    return rfdetr


class ImageSegmentationRFDETR(SegmentationTrainingData, Task):
    """Predict one source-aligned ImageSegmentation for each detected instance.

    ``model`` is a segmentation variant (small by default) or a local RF-DETR
    checkpoint. Training learns the annotation names as classes. SAM prompt and
    version settings are accepted for pipeline migration, but are not used.
    """

    def __init__(
        self, name: str, model: str | Path = "small", threshold: float = 0.5,
        device: str = "cuda", resolution: int | None = None, **kwargs,
    ):
        super().__init__(name, **kwargs)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if resolution is not None and resolution < 1:
            raise ValueError("resolution must be positive")
        self.model, self.threshold, self.device = model or "small", threshold, device
        self.resolution = resolution
        self._predictor = None

    def attributes_to_ignore(self):
        return super().attributes_to_ignore() | {"_predictor"}

    def input_type(self) -> type:
        return Image

    def output_type(self) -> type:
        return ImageSegmentation

    @staticmethod
    def _progress(message):
        print(f"[RF-DETR Seg] {message}", flush=True)

    def _label_prompts(self, samples):
        # Preserve artefact names verbatim rather than SAM's prompt descriptions.
        return {item.name: item.name for sample in samples for item in sample.instances}

    def _init_model(self):
        if self._predictor is not None:
            return
        backend = _rfdetr()
        options = {"device": self.device}
        if self.resolution is not None:
            options["resolution"] = self.resolution
        variant = str(self.model)
        # Accept the package's class names as well as short variant names.
        variant = next((k for k, v in VARIANTS.items() if v == variant), variant)
        if variant in VARIANTS:
            predictor = getattr(backend, VARIANTS[variant])(**options)
        else:
            checkpoint = Path(self.model).expanduser()
            if not checkpoint.is_absolute() and self.pipeline is not None:
                checkpoint = self.pipeline.path / checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            predictor = backend.RFDETR.from_checkpoint(str(checkpoint), **options)
        if not predictor.model_config.segmentation_head:
            raise ValueError("ImageSegmentationRFDETR requires an RF-DETR segmentation checkpoint")
        self._predictor = predictor

    @threading_locked()
    def run(self, *images: Image) -> list[ImageSegmentation]:
        if not images or any(not isinstance(image, Image) for image in images):
            raise TypeError("RF-DETR segmentation requires one or more Image artefacts")
        self._init_model()
        directory = Path(self.context.file_path or Path.cwd()) / "masks"
        directory.mkdir(parents=True, exist_ok=True)
        outputs = getattr(self, "output", None)
        outputs = [outputs] if isinstance(outputs, str) else outputs
        results = []
        for image in images:
            prediction = self._predictor.predict(image.pil().convert("RGB"), threshold=self.threshold)
            if not len(prediction):
                continue
            masks = prediction.mask
            if masks is None or np.asarray(masks).ndim != 3:
                raise ValueError("RF-DETR did not return instance segmentation masks")
            names = prediction.data.get("class_name")
            for index, mask in enumerate(masks):
                if names is not None:
                    label = str(names[index])
                else:
                    label = self._predictor.class_names[int(prediction.class_id[index])]
                if outputs:
                    aliases = {item.replace("_", " "): item for item in outputs}
                    aliases.update({prompt: item for item, prompt in getattr(self, "prompts", {}).items() if item in outputs})
                    label = aliases.get(label, label)
                    if len(outputs) == 1 and len(self._predictor.class_names) == 1:
                        label = outputs[0]
                if not label or label == "__background__" or (outputs and label not in outputs):
                    continue
                pixels = mask_to_source(image, np.asarray(mask) > 0.5)
                if not pixels.any():
                    continue
                path = directory / f"rfdetr-{uuid.uuid4().hex}.png"
                PILImage.fromarray(pixels).save(path)
                results.append(ImageSegmentation(
                    name=label, data=image.get_path(), mask=path, parents=[image.id],
                    orientation=image.orientation, partition=image.partition,
                ))
        return results

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                required=False,
                default=[],
                help="Results YAML files, result directories, or parent directories containing annotated images and masks. May be omitted when reusing an existing export.",
            ),
        ],
        output: Annotated[
            Path | None,
            Arg(help="Directory for prepared data, metrics.csv, logs and best model weights. Defaults to a timestamped directory; required with --reuse-data."),
        ] = None,
        model: Annotated[
            str,
            Arg(help="Model to fine-tune: a variant (nano, small, medium, large, xlarge or 2xlarge), its class name such as RFDETRSegMedium, or an RF-DETR segmentation checkpoint relative to the pipeline directory. Omit to use the configured model. The trained weights replace the configured model either way."),
        ] = "",
        validation: Annotated[
            str,
            Arg(help="Partition used for validation and best-model selection by mask IoU. Omit to train without validation. Ignored when reusing an existing export."),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition excluded from both training and validation during data preparation. Ignored when reusing an existing export."),
        ] = "",
        epochs: Annotated[
            int, Arg(help="Number of training epochs; validation runs after every epoch when available."),
        ] = 20,
        batch: Annotated[
            int, Arg(help="Number of training images per batch on each GPU."),
        ] = 1,
        num_gpus: Annotated[
            int, Arg(help="Number of GPUs used for training on this node."),
        ] = 1,
        keep_log: Annotated[
            bool,
            Arg(help="Keep prepared data, metrics, logs and checkpoints after copying the best weights into the pipeline. --no-keep-log removes the output directory."),
        ] = True,
        reuse_data: Annotated[
            bool,
            Arg(help="Reuse OUTPUT/dataset when both split annotation files exist; otherwise prepare it from the input paths. Requires --output. Reuse ignores input paths, partitions and --max-items. Does not resume optimizer state."),
        ] = False,
        wandb: Annotated[
            bool,
            Arg(help="Log epoch metrics to Weights & Biases. Project: <pipeline-directory>-<task>; run: output directory name; entity: W&B settings or WANDB_ENTITY."),
        ] = False,
        max_items: Annotated[
            int | None,
            Arg(help="Maximum total images across training and validation, after exclusions, in input order. Omit for all images. Ignored when reusing an existing export."),
        ] = None,
        eval_masks_head_resolution: Annotated[
            bool,
            Arg(help="Score validation (and train/mask_iou) at the mask head's resolution instead of upsampling every prediction to the full image. Avoids GPU out-of-memory on large pages; changes only these metrics and best-epoch selection, not training or the saved model's inference."),
        ] = False,
    ) -> Path:
        """Fine-tune RF-DETR on mask annotations and adopt its best weights."""
        if min(epochs, batch, num_gpus) < 1:
            raise ValueError("epochs, batch and num_gpus must be positive")
        if self.pipeline is None:
            raise RuntimeError("Attach the task to a pipeline before training")
        if reuse_data and output is None:
            raise ValueError("--reuse-data requires --output")
        # Resolve the starting weights before any data preparation so typos fail fast.
        model = str(model or self.model)
        if model not in VARIANTS and model not in VARIANTS.values():
            path = Path(model).expanduser()
            path = path.resolve() if path.is_absolute() else (self.pipeline.path / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"RF-DETR model must be one of {', '.join(VARIANTS)} or a checkpoint file: {path}"
                )
            model = str(path)
        output = (Path(output).expanduser().resolve() if output is not None
                  else Path.cwd() / datetime.now().strftime("%Y%m%d_%H%M%S"))
        dataset = output / "dataset"
        reuse_prepared = reuse_data and all(
            (dataset / split / "_annotations.coco.json").is_file()
            for split in ("train", "valid")
        )
        if not reuse_prepared:
            if max_items is not None and max_items < 1:
                raise ValueError("max_items must be positive")
            if not inputs:
                raise ValueError("Supply training inputs or reuse an existing OUTPUT/dataset export")
        output.mkdir(parents=True, exist_ok=True)
        self._progress(f"Training {self.name}: output={output}, epochs={epochs}, batch={batch}, GPUs={num_gpus}")
        if reuse_prepared:
            self._progress(f"Reusing {dataset}; ignoring inputs, partitions and --max-items")
        else:
            self._progress("Preparing masks and images (missing exports are rebuilt)")
            dataset = self.prepare_training_data(inputs, dataset, validation, exclude, max_items=max_items)
        annotations = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
        if not annotations["images"] or not annotations["categories"]:
            raise ValueError("No labelled training images were found in the prepared dataset")
        has_validation = bool(json.loads(
            (dataset / "valid" / "_annotations.coco.json").read_text()
        )["images"])
        config = {
            "model": model, "device": self.device, "resolution": self.resolution,
            "threshold": self.threshold, "has_validation": has_validation,
            "params": {
                "dataset_dir": str(dataset), "output_dir": str(output),
                "epochs": epochs, "batch_size": batch, "devices": num_gpus,
                "grad_accum_steps": 1, "device": self.device,
                "wandb": wandb, "project": f"{self.pipeline.name}-{self.name}",
                "run": output.name, "use_ema": False, "run_test": False,
                "eval_interval": 1, "eval_masks_head_resolution": eval_masks_head_resolution,
            },
        }
        config_path = output / "rfdetr-training.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._progress("Starting RF-DETR trainer (initial weights may download); trainer output follows")
        subprocess.run(
            [sys.executable, "-u", "-m", "collectra.tasks.image_segmentation.rfdetr_training", str(config_path)],
            check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        checkpoint = output / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RF-DETR training produced no checkpoint at {checkpoint}")
        destination = self.pipeline.path / f"{self.name}-{uuid.uuid4().hex[:8]}.pt"
        shutil.copy2(checkpoint, destination)
        self.model, self._predictor = destination, None
        self.pipeline.data[self.name]["model"] = destination.name
        self.pipeline.save(self.name)
        self._progress(f"Saved trained checkpoint: {destination}")
        if not keep_log:
            shutil.rmtree(output)
        return destination
