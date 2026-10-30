from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

from cappa import Arg

from PIL import Image as PillowImage

from collectra.cli import command
from collectra.types.images import (
    Image,
    ImageCrop,
    image_channel_count,
    open_oriented,
)
from collectra.types.links import Link
from collectra.utils import change_dir

from collectra.logger import get_logger
from collectra.utils import threading_locked
from ..machine_learning.training import print_distribution_table, run_training_command
from ..machine_learning.yolo import YOLOTask, get_yolo_class
from .training import prepare_classification_inputs

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO
    from ultralytics.utils.metrics import ClassifyMetrics

__all__ = ["ImageClassifierYOLO"]

logger = get_logger(__name__)


def _classification_channels(*roots):
    """Inspect every prepared image using metadata, before starting training."""
    detected = {}
    for root in roots:
        if root is None:
            continue
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and path.suffix.lower() in Image.image_types():
                try:
                    detected[path] = image_channel_count(path)
                except Exception as error:
                    raise ValueError(
                        f"Cannot determine channels for {path}: {error}"
                    ) from error
    if not detected:
        raise ValueError("No prepared classification images found")
    if len(set(detected.values())) != 1:
        details = ", ".join(
            f"{path}: {count} channels" for path, count in detected.items()
        )
        raise ValueError(
            f"Inconsistent input channel counts in classifier dataset: {details}"
        )
    channels = next(iter(detected.values()))
    tensor_loader = channels != 3 or any(
        p.suffix.lower() in {".tif", ".tiff"} for p in detected
    )
    return channels, tensor_loader


class ImageClassifierYOLO(YOLOTask):
    prepare_training_inputs = staticmethod(prepare_classification_inputs)

    @threading_locked()
    def run(self, *args: Image, **kwargs) -> Link:
        """Run image classification inference on the provided Image.

        Args:
            *args: A single Image (or ImageCrop) to classify.

        Returns:
            Link: A link from the predicted class node to the input image.
        """
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")
        image: Image = args[0]
        self._init_model()

        if (
            getattr(
                getattr(self.model, "model", None),
                "collectra_tensor_classification",
                False,
            )
            or image.get_path().suffix.lower() in {".tif", ".tiff"}
            or image_channel_count(image.get_path()) != 3
        ):
            from .yolo_multichannel import classify_multichannel

            predicted_class = classify_multichannel(self.model, image)
        else:
            predicted_class = self._predict_rgb(image)

        outputs = self.output if isinstance(self.output, list) else [self.output]
        if predicted_class not in outputs:
            raise ValueError(
                f"Predicted class {predicted_class!r} is not a configured output: "
                f"{outputs}"
            )
        print(f"Classified as: {predicted_class}")

        return Link(name=predicted_class, target=image)

    def _predict_rgb(self, image):
        if isinstance(image, ImageCrop):
            with tempfile.TemporaryDirectory() as temp_dir:
                img = image.pil()
                img_path = Path(temp_dir) / Path(image.get_path()).name
                img.save(img_path)
                results: Results = (self.model(img_path))[0]
        else:
            results: Results = (self.model(image.get_path()))[0]

        if results.probs is None:
            raise ValueError("No classification probabilities returned.")

        top1_index = results.probs.top1
        return results.names[top1_index]

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing image classification labels."
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="YOLO classification weights name or checkpoint path (for example yolo26n-cls.pt), or a model YAML file. Omit to keep the configured model."
            ),
        ] = "",
        output: Annotated[
            Path | None,
            Arg(help="Directory for training logs, dataset files, and checkpoints."),
        ] = None,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep the training directory after saving the model to the pipeline."
            ),
        ] = True,
        validation: Annotated[
            str,
            Arg(help="Partition value identifying validation examples."),
        ] = "",
        exclude: Annotated[
            str,
            Arg(help="Partition value identifying examples to exclude from training."),
        ] = "",
        epochs: Annotated[
            int,
            Arg(help="Number of training epochs."),
        ] = 100,
        batch: Annotated[
            int,
            Arg(help="Training batch size."),
        ] = 16,
        imgsz: Annotated[
            int,
            Arg(help="Target training image size in pixels."),
        ] = 1280,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without validation improvement; 0 disables early stopping."
            ),
        ] = 50,
        min_size: Annotated[
            int,
            Arg(
                help="Skip images or crops with either dimension below this many pixels; 0 disables the size threshold."
            ),
        ] = 0,
        learning_rate: Annotated[
            float,
            Arg(help="Initial learning rate."),
        ] = 0.01,
        weight_decay: Annotated[
            float,
            Arg(help="Weight decay used for regularization."),
        ] = 0.0005,
        dropout: Annotated[
            float,
            Arg(help="Dropout probability for the classification head."),
        ] = 0.0,
        erasing: Annotated[
            float,
            Arg(help="Probability of random erasing augmentation."),
        ] = 0.4,
        auto_augment: Annotated[
            str,
            Arg(
                help="Automatic augmentation policy: randaugment, autoaugment, or augmix. Skipped for TIFF and non-RGB datasets."
            ),
        ] = "randaugment",
        fliplr: Annotated[
            float,
            Arg(help="Probability of horizontally flipping an image."),
        ] = 0.5,
        flipud: Annotated[
            float,
            Arg(help="Probability of vertically flipping an image."),
        ] = 0.0,
        hsv_h: Annotated[
            float,
            Arg(
                help="Hue augmentation amount as a fraction; skipped for TIFF and non-RGB datasets."
            ),
        ] = 0.015,
        hsv_s: Annotated[
            float,
            Arg(
                help="Saturation augmentation amount as a fraction; skipped for TIFF and non-RGB datasets."
            ),
        ] = 0.7,
        hsv_v: Annotated[
            float,
            Arg(
                help="Brightness augmentation amount as a fraction; skipped for TIFF and non-RGB datasets."
            ),
        ] = 0.4,
        cls_pw: Annotated[
            float,
            Arg(
                help="Class-imbalance weighting power; 0 disables weighting, 1 uses inverse class frequency."
            ),
        ] = 0.0,
        freeze: Annotated[
            int | None,
            Arg(help="Freeze the first N model layers during training."),
        ] = None,
        wandb: Annotated[
            bool, Arg(help="Log training metrics to Weights & Biases.")
        ] = False,
    ):
        """Train this YOLO image classifier."""
        return run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            validation=validation,
            exclude=exclude,
            prepare_inputs=self.prepare_training_inputs,
            model=model,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            early_stop=early_stop,
            min_size=min_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            erasing=erasing,
            auto_augment=auto_augment,
            fliplr=fliplr,
            flipud=flipud,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            cls_pw=cls_pw,
            freeze=freeze,
            wandb=wandb,
        )

    def _train(self, *images: Image, **kwargs) -> ClassifyMetrics | None:
        YOLO = get_yolo_class()

        if kwargs.get("model"):
            self.model = kwargs["model"]
        self._init_model()
        log = kwargs.get("log", None)
        if log is None:
            raise ValueError(
                "Log directory must be specified in kwargs with key 'log'."
            )
        log = kwargs["base_folder"] / log
        log.mkdir(parents=True, exist_ok=True)
        print(f"Training files will be saved to: {log}")
        # Derive classes from training data, not from pipeline children nodes
        classes = sorted(set(img.name for img in images))
        kwargs.pop("classes", None)
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance for training.")

        print(f"Training with model: {self.model.model_name}")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train_dir, val_dir = self._prepare_assets(
            classes,
            log,
            validation,
            exclude,
            *images,
            min_size=int(kwargs.get("min_size", 0)),
        )
        return self._train_fold(train_dir, val_dir, classes, log, kwargs)

    def _train_fold(
        self,
        train: list[str] | Path,
        val: list[str] | Path,
        classes: list[str],
        log: Path,
        kwargs: dict,
        fold_count: int | None = None,
    ) -> ClassifyMetrics:
        YOLO = get_yolo_class()

        if not isinstance(self.model, YOLO):
            raise ValueError("Expected model to be a YOLO instance for training.")
        _, tensor_loader = _classification_channels(train, val)
        kwargs["config_file"] = self._prepare_yolo_config(log, classes, train, val)
        kwargs["log"] = f"{log.name}"
        if fold_count:
            kwargs["log"] += f"_fold_{fold_count}"
        params = self._prepare_params(**kwargs)
        if tensor_loader:
            from .yolo_multichannel import MultichannelClassificationTrainer

            params["trainer"] = MultichannelClassificationTrainer
        with (
            change_dir(kwargs["base_folder"]),
            self._wandb_logging(kwargs.get("wandb", False)),
        ):
            results = self.model.train(**params)
        if results is None:
            raise Exception("[red]Training failed, no results returned.[/red]")
        self._reload()
        return results

    def _prepare_assets(
        self,
        classes: list[str],
        log: Path,
        validation_flag: str,
        exclude_flag: str,
        *images: Image,
        min_size: int = 0,
    ) -> tuple[Path, Path]:
        """Prepare assets for YOLO classification training.

        Creates directory structure: log/train/{class}/ and log/val/{class}/

        Args:
            classes: List of class names.
            log: Path to store the assets for training.
            validation_flag: The value of the validation flag to identify validation images.
            exclude_flag: The value of the exclude flag to identify images to be excluded.
            *images: Image instances to be prepared.

        Returns:
            Tuple of (train_dir, val_dir) paths.
        """
        train_dir = log / "train"
        val_dir = log / "val"

        if min_size < 0:
            raise ValueError("min_size cannot be negative")

        for cls in classes:
            (train_dir / cls).mkdir(parents=True, exist_ok=True)
            (val_dir / cls).mkdir(parents=True, exist_ok=True)

        source_images: dict[Path, PillowImage.Image] = {}
        for img in images:
            if exclude_flag and img.partition == exclude_flag:
                continue

            cls_name = img.name
            if cls_name not in classes:
                continue

            if img.partition == validation_flag:
                target_dir = val_dir / cls_name
            else:
                target_dir = train_dir / cls_name

            target = img.resolve() if type(img) is Link else img
            src = target.get_path()
            dst = target_dir / src.name
            pixel_box = None
            if isinstance(target, ImageCrop):
                dst = target_dir / f"{src.stem}-{img.id}{src.suffix}"
                left, upper, right, bottom = target.coordinates()
                pixel_box = (left, upper, right, bottom)
                width = right - left
                height = bottom - upper
            elif min_size:
                width = int(target.width)
                height = int(target.height)

            if type(img) is Link:
                dst = target_dir / f"{src.stem}-{img.id}{src.suffix}"

            empty_crop = isinstance(target, ImageCrop) and (width <= 0 or height <= 0)
            below_minimum = min_size and (width < min_size or height < min_size)
            if empty_crop or below_minimum:
                effective_minimum = min_size or 1
                parent_id = img.parent_id if type(img) is Link else ""
                logger.warning(
                    "Skipping classifier image %s%s from %s: image is %dx%d "
                    "pixels (minimum %d)%s",
                    img.id,
                    f" -> {parent_id}" if parent_id else "",
                    img.source_file or src,
                    width,
                    height,
                    effective_minimum,
                    f", pixel_box={pixel_box}" if pixel_box else "",
                )
                continue

            if not dst.exists():
                if isinstance(target, ImageCrop) and src.suffix.lower() in {
                    ".tif",
                    ".tiff",
                }:
                    import numpy as np
                    import tifffile
                    from .torchvision import _target_tiff_pixels

                    pixels = _target_tiff_pixels(target)
                    tifffile.imwrite(
                        dst,
                        np.ascontiguousarray(pixels.transpose(2, 0, 1)),
                        photometric="minisblack",
                        metadata={"axes": "CYX"},
                    )
                elif isinstance(target, ImageCrop):
                    source_path = src.resolve()
                    if source_path not in source_images:
                        source_images[source_path] = open_oriented(source_path)
                    with source_images[source_path].crop(pixel_box) as cropped:
                        rotated = cropped.rotate(
                            target.orientation.to_degree(), expand=True
                        )
                        try:
                            rotated.save(dst)
                        finally:
                            rotated.close()
                else:
                    shutil.copy(src, dst)

        for source in source_images.values():
            source.close()

        return train_dir, val_dir

    def _prepare_yolo_config(
        self,
        log: Path,
        classes: list[str],
        train: list[str] | Path,
        val: list[str] | Path,
    ) -> Path:
        """For classification, data param is the parent directory containing train/ and val/.

        Also prints distribution info.
        """
        self._check_distribution(log, classes)
        return log

    def _check_distribution(
        self,
        log: Path,
        metadata: dict | list[str],
    ) -> None:
        classes = (
            metadata
            if isinstance(metadata, list)
            else list(metadata.get("classes", []))
        )
        train_dir = log / "train"
        val_dir = log / "val"

        train_counts = {}
        validation_counts = {}
        for cls_name in classes:
            train_counts[cls_name] = sum(
                1
                for ext in Image.image_types()
                for _ in (train_dir / cls_name).glob(f"*{ext}")
            )
            validation_counts[cls_name] = sum(
                1
                for ext in Image.image_types()
                for _ in (val_dir / cls_name).glob(f"*{ext}")
            )
        print_distribution_table(
            "Classification Distribution", classes, train_counts, validation_counts
        )

    def _prepare_params(self, **kwargs) -> dict:
        import platform

        import torch

        params = {
            "name": kwargs["log"],
            "data": kwargs["config_file"],
            "project": kwargs["project"],
            "device": (
                "mps"
                if platform.system() == "Darwin"
                else "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "epochs": kwargs.get("epochs", 100),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("early_stop", 50),
            "batch": kwargs.get("batch", 16),
            "lr0": kwargs.get("learning_rate", 0.01),
            "weight_decay": kwargs.get("weight_decay", 0.0005),
            "dropout": kwargs.get("dropout", 0.0),
            "erasing": kwargs.get("erasing", 0.4),
            "auto_augment": kwargs.get("auto_augment", "randaugment"),
            "fliplr": kwargs.get("fliplr", 0.5),
            "flipud": kwargs.get("flipud", 0.0),
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "cls_pw": kwargs.get("cls_pw", 0.0),
        }
        if kwargs.get("freeze") is not None:
            params["freeze"] = kwargs["freeze"]
        return params
