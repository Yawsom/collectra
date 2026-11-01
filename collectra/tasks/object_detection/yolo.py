from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated, TYPE_CHECKING

from pydantic import Field

from cappa import Arg

from collectra.cli import command
from .adaptation_commands import YOLOAdaptationCommands
from .base import (
    Batch,
    Epochs,
    LearningRate,
    ObjectDetectionTask,
    RunResolution,
    Threshold,
    TrainResolution,
    WeightDecay,
)
from collectra.types.images import (
    Image,
    ImageCrop,
    convert_image_to_rgb,
    image_channel_count,
    read_tiff_channels,
)
from collectra.types.texts import Text
from collectra.utils import change_dir

from collectra.utils import threading_locked
from ..machine_learning.training import (
    prepare_object_detection_inputs,
    print_distribution_table,
    run_training_command,
)
from ..machine_learning.yolo import YOLOTask, get_yolo_class

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO
    from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics

__all__ = ["ObjectDetectionYOLO"]


class DetectionResult:
    metrics: dict
    best_result: DetMetrics

    def __init__(self, result: DetMetrics) -> None:
        self.best_result = result
        self.metrics = result.results_dict if result else {}

    @staticmethod
    def get_best_result(
        results: list["DetectionResult"], metric: str = "metrics/mAP50-95(B)"
    ) -> DetMetrics:
        best_result: DetMetrics | None = None
        best_metric_result = 0.0
        for result in results:
            if result is None:
                continue
            metric_result = result.metrics.get(metric, 0.0)
            if best_result is None:
                best_metric_result = metric_result
                best_result = result.best_result
                continue
            if metric_result > best_metric_result:
                best_metric_result = metric_result
                best_result = result.best_result
        if best_result is None:
            raise ValueError("No valid results found to determine the best result.")
        return best_result


class ObjectDetectionYOLO(YOLOAdaptationCommands, YOLOTask, ObjectDetectionTask):
    singletons: bool = False
    batch_size: int = Field(default=1, ge=1)
    # Ultralytics' own defaults, where they differ from the shared ones.
    batch: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0005, ge=0)
    # Overlap above which two boxes of one class count as the same object.
    iou: float = Field(default=0.7, gt=0, le=1)

    RENAMED_SETTINGS = {**ObjectDetectionTask.RENAMED_SETTINGS, "lr0": "learning_rate"}

    @command(name="run")
    def cli_run(
        self,
        inputs: list[str],
        output: Path | None = None,
        verbose: bool = False,
        usage: bool = False,
        render: bool = False,
        model: Annotated[
            Path | None,
            Arg(help="Model checkpoint path. Omit to keep the configured model."),
        ] = None,
        resolution: RunResolution = None,
        threshold: Threshold = None,
        imgsz: Annotated[int | None, Arg(help="Deprecated: use --resolution.")] = None,
        batch_size: Annotated[
            int,
            Arg(
                help="Number of files to run through the model at once; results are saved after each batch."
            ),
        ] = 16,
        force: Annotated[
            bool,
            Arg(
                help="Rerun tasks whose outputs already exist, replacing those outputs. By default such tasks are skipped."
            ),
        ] = False,
    ) -> None:
        """Run this object detector for the supplied input files."""
        if batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if model is not None:
            self.model = model
        resolution = self._current_name("imgsz", "resolution", imgsz, resolution)
        if resolution is not None:
            self.resolution = resolution
        if threshold is not None:
            self.threshold = threshold
        self.batch_size = batch_size
        super().cli_run(
            inputs,
            output=output,
            verbose=verbose,
            usage=usage,
            render=render,
            force=force,
        )

    @threading_locked()
    def run(
        self,
        *args: Image,
        model: str | Path | None = None,
        resolution: int | None = None,
        imgsz: int | None = None,
        **kwargs,
    ) -> list[ImageCrop]:
        """Run object detection inference on the provided Image.

        This method performs object detection on the provided input image
        using the trained YOLO model.

        Results include bounding boxes, confidence scores, and class
        predictions for each detected object.

        Args:
            input (Image): The input image on which to perform object detection.
            model: Optional checkpoint path overriding the task's configured model.
            resolution: Optional image size in pixels. Defaults to the size chosen
                for the task, or the model's own setting when none was chosen.
            imgsz: Deprecated name for ``resolution``.

        Returns:
            list[ImageCrop]: A list of ImageCrop objects representing the detected
                              objects in the input image.
        """
        if len(args) != 1:
            raise ValueError("This task only supports a single Image input.")
        if not isinstance(args[0], Image):
            raise TypeError("Input must be an instance of Image.")
        image: Image = args[0]
        if model is not None:
            self.model = model
        self._init_model()
        from torchvision.ops import batched_nms

        resolution = self._current_name("imgsz", "resolution", imgsz, resolution)
        if resolution is None:
            resolution = self.inference_resolution
        results: Results | None = (
            self._take_prefetched(image)
            if resolution == self.inference_resolution
            else None
        )
        if results is None:
            results = self._predict([image], resolution)[0]

        if getattr(self.model.model, "end2end", False) and len(results.boxes):
            keep = batched_nms(
                results.boxes.xyxy,
                results.boxes.conf,
                results.boxes.cls,
                iou_threshold=self.iou,
            )
            results = results[keep]

        detections: list[ImageCrop] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        coordinates = results.boxes.xywhn.clone()
        names = [
            results.names[class_name.int().item()] for class_name in results.boxes.cls
        ]

        counts = Counter()
        if len(names) == 0:
            # No objects detected, return empty list
            return detections
        for index in range(len(coordinates)):
            name = names[index]

            if self.singletons and counts[name]:
                continue

            x, y, w, h = coordinates[index]
            confidence = results.boxes.conf[index]

            image_crop = image.make_crop(
                x_center=float(x),
                y_center=float(y),
                width_relative=float(w),
                height_relative=float(h),
                confidence=confidence.item(),
                orientation=image.orientation,
                name=name,
            )
            counts[name] += 1
            detections.append(image_crop)

        print(f"Found {len(detections)} objects in the image:")
        for name, count in counts.items():
            print(f"\t{name}: {count}")
        return detections

    def _tiff_pixels(self, image: Image):
        """Decode a TIFF (cropped and rotated as needed) for the model's channels."""
        import numpy as np
        import torch

        pixels = read_tiff_channels(image.get_path())
        if isinstance(image, ImageCrop):
            left, top, right, bottom = image.coordinates()
            if right <= left or bottom <= top:
                raise ValueError(f"ImageCrop {image.id!r} produces an empty pixel crop")
            pixels = pixels[top:bottom, left:right, :]
        pixels = np.rot90(pixels, k=image.orientation.to_degree() // 90)
        first_conv = next(
            layer
            for layer in self.model.model.modules()
            if isinstance(layer, torch.nn.Conv2d)
        )
        expected_channels = first_conv.in_channels
        model_config = getattr(self.model.model, "yaml", {})
        if "feature_fusion" in model_config:
            expected_channels = model_config["channels"]
        if pixels.shape[2] != expected_channels:
            raise ValueError(
                f"{image.get_path()}: {pixels.shape[2]} image channels, "
                f"but the detection model expects {expected_channels}"
            )
        # Ultralytics expects BGR for three-channel arrays, but preserves
        # channel order for multispectral arrays. Keep its usual resizing,
        # normalization, and box scaling relative to the original image.
        if pixels.shape[2] == 3:
            pixels = pixels[..., ::-1]
        return np.ascontiguousarray(pixels)

    def _predict(self, images: list[Image], imgsz: int | None) -> list[Results]:
        """Run the model over ``images`` together, returning one result each."""
        inference_kwargs = {"iou": self.iou, "conf": self.threshold}
        if imgsz is not None:
            inference_kwargs["imgsz"] = imgsz

        # Ultralytics takes a list of one source type at a time.
        arrays: dict[int, object] = {}
        paths: dict[int, Path] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, image in enumerate(images):
                if image.get_path().suffix.lower() in {".tif", ".tiff"}:
                    arrays[index] = self._tiff_pixels(image)
                elif isinstance(image, ImageCrop):
                    img_path = (
                        Path(temp_dir) / f"{index}-{Path(image.get_path()).stem}.png"
                    )
                    image.pil().save(img_path, format="PNG")
                    paths[index] = img_path
                else:
                    paths[index] = image.get_path()

            results: list[Results | None] = [None] * len(images)
            for sources in (arrays, paths):
                if not sources:
                    continue
                indices = list(sources)
                call_kwargs = dict(inference_kwargs)
                if len(indices) > 1:
                    call_kwargs["batch"] = len(indices)
                predicted = self.model([sources[i] for i in indices], **call_kwargs)
                for index, result in zip(indices, predicted):
                    results[index] = result
        return results

    def _predict_batch(self, images: list[Image]) -> list[Results]:
        return self._predict(images, self.inference_resolution)

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing training annotations."
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="YOLO weights name or checkpoint path (for example yolo26n.pt), or a model YAML file. Omit to keep the configured model."
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
        epochs: Epochs = None,
        batch: Batch = None,
        resolution: TrainResolution = None,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without validation improvement; 0 disables early stopping."
            ),
        ] = 50,
        preview: Annotated[
            bool,
            Arg(
                help="Preview the prepared training images and annotations before training."
            ),
        ] = False,
        rgb: Annotated[
            bool,
            Arg(
                help="Convert all prepared training images to 3-channel RGB, so images with differing channel counts (grayscale, RGBA) can be trained together."
            ),
        ] = False,
        workers: Annotated[
            int,
            Arg(help="Number of data-loading workers per distributed process."),
        ] = 8,
        pretrained: Annotated[
            bool,
            Arg(help="Use pretrained weights when supported by the selected model."),
        ] = True,
        optimizer: Annotated[
            str,
            Arg(
                help="Optimizer: auto, SGD, MuSGD, Adam, Adamax, AdamW, NAdam, RAdam, or RMSProp."
            ),
        ] = "auto",
        seed: Annotated[
            int,
            Arg(help="Random seed for training."),
        ] = 0,
        deterministic: Annotated[
            bool,
            Arg(
                help="Use deterministic operations for reproducibility; may reduce speed."
            ),
        ] = True,
        single_cls: Annotated[
            bool,
            Arg(help="Treat all object classes as one class."),
        ] = False,
        rect: Annotated[
            bool,
            Arg(help="Use rectangular training batches to reduce padding."),
        ] = False,
        cos_lr: Annotated[
            bool,
            Arg(help="Use a cosine learning-rate schedule."),
        ] = False,
        close_mosaic: Annotated[
            int,
            Arg(
                help="Disable mosaic augmentation for the final N epochs; 0 keeps it enabled."
            ),
        ] = 10,
        amp: Annotated[
            bool,
            Arg(help="Enable automatic mixed precision training."),
        ] = True,
        fraction: Annotated[
            float,
            Arg(help="Fraction of the training dataset to use; 1.0 uses all examples."),
        ] = 1.0,
        freeze: Annotated[
            int | None,
            Arg(help="Freeze the first N model layers during training."),
        ] = None,
        multi_scale: Annotated[
            float,
            Arg(
                help="Vary image size by this fraction of the resolution; 0 disables multi-scale training."
            ),
        ] = 0.0,
        learning_rate: LearningRate = None,
        lrf: Annotated[
            float,
            Arg(
                help="Final learning-rate multiplier; final rate is the learning rate times lrf."
            ),
        ] = 0.01,
        momentum: Annotated[
            float,
            Arg(help="SGD momentum or the first Adam beta coefficient."),
        ] = 0.937,
        weight_decay: WeightDecay = None,
        warmup_epochs: Annotated[
            float,
            Arg(
                help="Number of learning-rate warmup epochs; fractional values are allowed."
            ),
        ] = 3.0,
        warmup_momentum: Annotated[
            float,
            Arg(help="Initial momentum during warmup."),
        ] = 0.8,
        warmup_bias_lr: Annotated[
            float,
            Arg(help="Initial learning rate for bias parameters during warmup."),
        ] = 0.1,
        box: Annotated[
            float,
            Arg(help="Weight applied to bounding-box loss."),
        ] = 7.5,
        cls: Annotated[
            float,
            Arg(help="Weight applied to classification loss."),
        ] = 0.5,
        cls_pw: Annotated[
            float,
            Arg(
                help="Class-imbalance weighting power; 0 disables weighting, 1 uses inverse class frequency."
            ),
        ] = 0.0,
        dfl: Annotated[
            float,
            Arg(help="Weight applied to distribution focal loss."),
        ] = 1.5,
        nbs: Annotated[
            int,
            Arg(help="Nominal batch size used for loss normalization."),
        ] = 64,
        hsv_h: Annotated[
            float,
            Arg(help="Hue augmentation amount as a fraction."),
        ] = 0.015,
        hsv_s: Annotated[
            float,
            Arg(help="Saturation augmentation amount as a fraction."),
        ] = 0.7,
        hsv_v: Annotated[
            float,
            Arg(help="Brightness augmentation amount as a fraction."),
        ] = 0.4,
        degrees: Annotated[
            float,
            Arg(help="Maximum random rotation in either direction, in degrees."),
        ] = 0.0,
        translate: Annotated[
            float,
            Arg(help="Maximum random translation as a fraction of image size."),
        ] = 0.1,
        scale: Annotated[
            float,
            Arg(help="Random scale variation around 1.0."),
        ] = 0.5,
        shear: Annotated[
            float,
            Arg(help="Maximum random shear in either direction, in degrees."),
        ] = 0.0,
        perspective: Annotated[
            float,
            Arg(help="Strength of random perspective augmentation."),
        ] = 0.0,
        flipud: Annotated[
            float,
            Arg(help="Probability of vertically flipping an image."),
        ] = 0.0,
        fliplr: Annotated[
            float,
            Arg(help="Probability of horizontally flipping an image."),
        ] = 0.5,
        bgr: Annotated[
            float,
            Arg(help="Probability of swapping RGB and BGR color channels."),
        ] = 0.0,
        mosaic: Annotated[
            float,
            Arg(help="Probability of mosaic augmentation."),
        ] = 1.0,
        mixup: Annotated[
            float,
            Arg(help="Probability of MixUp augmentation."),
        ] = 0.0,
        cutmix: Annotated[
            float,
            Arg(help="Probability of CutMix augmentation."),
        ] = 0.0,
        time: Annotated[
            float,
            Arg(
                help="Training time limit in hours; a positive value overrides epochs, 0 leaves it unset."
            ),
        ] = 0.0,
        save: Annotated[
            bool,
            Arg(help="Save training checkpoints."),
        ] = True,
        save_period: Annotated[
            int,
            Arg(
                help="Save an additional checkpoint every N epochs; -1 disables periodic saves."
            ),
        ] = -1,
        cache: Annotated[
            str,
            Arg(help="Cache images in ram or on disk; omit to leave caching disabled."),
        ] = "",
        device: Annotated[
            str,
            Arg(
                help="Training device, such as cpu, mps, 0, or 0,1; omit for automatic selection."
            ),
        ] = "",
        verbose: Annotated[
            bool,
            Arg(help="Print detailed training logs."),
        ] = True,
        resume: Annotated[
            bool,
            Arg(help="Resume training from the selected checkpoint."),
        ] = False,
        compile: Annotated[
            str,
            Arg(
                help="Torch compilation mode: default, reduce-overhead, or max-autotune-no-cudagraphs; omit to disable."
            ),
        ] = "",
        distill_model: Annotated[
            str,
            Arg(
                help="Teacher model checkpoint for knowledge distillation; omit to disable."
            ),
        ] = "",
        dis: Annotated[
            float,
            Arg(help="Weight applied to knowledge-distillation loss."),
        ] = 6.0,
        plots: Annotated[
            bool,
            Arg(help="Save training and validation plots and images."),
        ] = True,
        wandb: Annotated[
            bool, Arg(help="Log training metrics to Weights & Biases.")
        ] = False,
        imgsz: Annotated[int | None, Arg(help="Deprecated: use --resolution.")] = None,
        lr0: Annotated[
            float | None, Arg(help="Deprecated: use --learning-rate.")
        ] = None,
    ):
        """Train this YOLO object detector."""
        resolution = self._current_name("imgsz", "resolution", imgsz, resolution)
        learning_rate = self._current_name("lr0", "learning_rate", lr0, learning_rate)
        settings = self._settings_for_run(
            epochs=epochs,
            batch=batch,
            resolution=resolution,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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
            model=model,
            epochs=settings["epochs"],
            batch=settings["batch"],
            imgsz=settings["resolution"],
            early_stop=early_stop,
            preview=preview,
            rgb=rgb,
            workers=workers,
            pretrained=pretrained,
            optimizer=optimizer,
            seed=seed,
            deterministic=deterministic,
            single_cls=single_cls,
            rect=rect,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            amp=amp,
            fraction=fraction,
            freeze=freeze,
            multi_scale=multi_scale,
            lr0=settings["learning_rate"],
            lrf=lrf,
            momentum=momentum,
            weight_decay=settings["weight_decay"],
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box,
            cls=cls,
            cls_pw=cls_pw,
            dfl=dfl,
            nbs=nbs,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            bgr=bgr,
            mosaic=mosaic,
            mixup=mixup,
            cutmix=cutmix,
            time=time,
            save=save,
            save_period=save_period,
            cache=cache,
            device=device,
            verbose=verbose,
            resume=resume,
            compile=compile,
            distill_model=distill_model,
            dis=dis,
            plots=plots,
            wandb=wandb,
        )

    def _train(
        self, *images: ImageCrop, **kwargs
    ) -> DetMetrics | ClassifyMetrics | None:
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
        classes = kwargs.pop("classes", [])

        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance for training.")

        print(f"Training with model: {self.model.model_name}")

        validation = kwargs.get("validation", "")
        exclude = kwargs.get("exclude", "")
        train, val = self._prepare_assets(
            classes, log, validation, exclude, *images, rgb=kwargs.get("rgb", False)
        )
        return self._train_fold(train, val, classes, log, kwargs)

    def _train_fold(
        self,
        train: list[str] | Path,
        val: list[str] | Path,
        classes: list[str],
        log: Path,
        kwargs: dict,
        fold_count: int | None = None,
    ) -> DetMetrics | ClassifyMetrics:
        YOLO = get_yolo_class()

        if not isinstance(self.model, YOLO):
            raise ValueError("Expected model to be a YOLO instance for training.")
        kwargs["config_file"] = self._prepare_yolo_config(
            log,
            classes,
            train,
            val,
        )
        kwargs["log"] = f"{log.name}"
        if fold_count:
            kwargs["log"] += f"_fold_{fold_count}"
        params = self._prepare_params(**kwargs)
        if kwargs.get("preview", False):
            self._preview_assets(log, classes)
        with (
            change_dir(kwargs["base_folder"]),
            self._wandb_logging(kwargs.get("wandb", False)),
        ):
            results: DetMetrics | None = self.model.train(**params)
        if results is None:
            raise Exception("[red]Training failed, no results returned.[/red]")
        self._reload()
        return results

    def _preview_assets(self, log: Path, classes: list[str]) -> None:
        try:
            from drawyolo.draw import draw_box_on_image_with_labels
        except ModuleNotFoundError as error:
            if error.name != "drawyolo":
                raise
            raise ImportError(
                "YOLO previews require drawyolo. Install with: "
                'pip install "collectra[yolo]"'
            ) from error

        preview_dir = log / "preview"
        preview_dir.mkdir(exist_ok=True)

        for image_type in Image.image_types():
            for image_file in log.glob(f"*{image_type}"):
                dest = preview_dir / image_file.name
                shutil.copy(image_file, dest)
                output_path = preview_dir / image_file.name
                draw_box_on_image_with_labels(
                    image=image_file,
                    output=output_path,
                    labels=image_file.with_suffix(".txt"),
                    classes=classes,
                )

    def _prepare_yolo_config(
        self,
        log: Path,
        classes: list[str],
        train: list[str] | Path,
        val: list[str] | Path,
    ) -> Path:

        # Inspect exactly the prepared files referenced by the generated manifests.
        # The dataset, not the selected checkpoint, determines the model input size.
        train = [Path(path).name for path in train]
        val = [Path(path).name for path in val]
        detected = {}
        for name in dict.fromkeys([*train, *val]):
            path = log / name
            try:
                detected[name] = image_channel_count(path)
            except Exception as error:
                raise ValueError(
                    f"Cannot determine input channels for {path}: {error}"
                ) from error
        if not detected:
            raise ValueError(
                "Cannot infer input channels: no prepared training or validation images"
            )
        if len(set(detected.values())) != 1:
            details = ", ".join(
                f"{name}: {count} channels" for name, count in detected.items()
            )
            raise ValueError(
                f"Inconsistent input channel counts in prepared dataset: {details}"
            )
        channels = next(iter(detected.values()))

        metadata = {
            "classes": classes,
            "train": train,
            "val": val,
        }

        Path(log / "train.txt").write_text(
            "\n".join([f"./{Path(p).name}" for p in train])
        )
        Path(log / "val.txt").write_text("\n".join([f"./{Path(p).name}" for p in val]))

        print(
            f"Saved {len(metadata['train'])} training images and {len(metadata['val'])} validation images to {log}"
        )

        self._check_distribution(log, metadata)

        train_txt = "train.txt"
        val_txt = "val.txt"
        num_classes = len(metadata["classes"])
        names = metadata["classes"]

        config = (
            f"train: {train_txt}\nval: {val_txt}\nnc: {num_classes}\nnames: {names}\n"
            f"channels: {channels}\n"
        )
        config_file = log / "config.yml"
        Path(config_file).write_text(config)

        return config_file

    def _check_distribution(
        self,
        log: Path,
        metadata: dict | list[str],
    ) -> None:
        if not isinstance(metadata, dict):
            return

        train_classes = {name: 0 for name in metadata["classes"]}
        val_classes = {name: 0 for name in metadata["classes"]}
        # (with objects, background) image counts for train and validation
        train_images = [0, 0]
        val_images = [0, 0]

        for image_type in Image.image_types():
            for image_file in log.glob(f"*{image_type}"):
                label_file = image_file.with_suffix(".txt")
                with open(label_file, "r") as f:
                    label_lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                if image_file.name in metadata["train"]:
                    train_images[0 if label_lines else 1] += 1
                elif image_file.name in metadata["val"]:
                    val_images[0 if label_lines else 1] += 1
                for line in label_lines:
                    class_index = int(line.split()[0])
                    class_name = metadata["classes"][class_index]
                    if image_file.name in metadata["train"]:
                        train_classes[class_name] += 1
                    elif image_file.name in metadata["val"]:
                        val_classes[class_name] += 1

        print_distribution_table(
            "Class Distribution",
            metadata["classes"],
            train_classes,
            val_classes,
            train_images=tuple(train_images),
            validation_images=tuple(val_images),
        )

    def _write_yolo_label(self, name_index: int, img: ImageCrop, f):
        if name_index == -1:
            f.write("")
        else:
            f.write(
                f"{name_index} {img.x_center:.6f} {img.y_center:.6f} {img.width_relative:.6f} {img.height_relative:.6f}\n"
            )

    def _prepare_assets(
        self,
        classes: list[str],
        log: Path,
        validation_flag: str,
        exclude_flag: str,
        *images: ImageCrop,
        rgb: bool = False,
    ) -> tuple[list[str], list[str]] | tuple[Path, Path]:
        """Prepare assets for YOLO training.

        Args:
            classes (list[str]): List of class names.
            log (Path): Path to store the assets for training.
            validation_flag (str): The value of the validation flag to identify validation images.
            exclude_flag (str): The value of the exclude flag to identify images to be excluded from training.
            images (ImageCrop): Variable number of ImageCrop instances to be prepared.
            rgb (bool): Convert each prepared image to 3-channel RGB.

        Returns:
            tuple[list[list[int]], list[str], list[str]]: A tuple containing the label matrix for iterative stratification,
            training image filenames, and validation image filenames.

        """
        train: list[str] = list()
        val: list[str] = list()
        for img in images:
            src = img.get_path()
            dst = log / src.name
            if isinstance(img, ImageCrop) and img.source_parent:
                dst = log / f"{src.stem}-{str(img.source_parent.id)}{src.suffix}"
            if not dst.exists():
                if hasattr(img, "source_parent") and img.source_parent is not None:
                    img.source_parent.save(dst)
                else:
                    img.save(dst)
                if rgb:
                    convert_image_to_rgb(dst)
                if img.partition == validation_flag:
                    val.append(dst.name)
                elif not exclude_flag or img.partition != exclude_flag:
                    train.append(dst.name)
            if isinstance(img, ImageCrop) and isinstance(img.source_parent, ImageCrop):
                img.set_rel_to_src_parent()
            name_index = classes.index(img.name) if img.name in classes else -1
            text_dst = log / f"{dst.stem}.txt"
            write_mode = "a" if text_dst.exists() else "w"
            with open(text_dst, write_mode) as f:
                self._write_yolo_label(name_index, img, f)
        return train, val

    def _prepare_params(self, **kwargs) -> dict:
        params = {
            "name": kwargs["log"],
            "data": kwargs["config_file"],
            "project": kwargs["project"],
            "device": kwargs.get("device") or self._select_device(),
            "epochs": kwargs.get("epochs", 100),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("early_stop", 50),
            "batch": kwargs.get("batch", 16),
            "workers": kwargs.get("workers", 8),
            "pretrained": kwargs.get("pretrained", True),
            "optimizer": kwargs.get("optimizer", "auto"),
            "seed": kwargs.get("seed", 0),
            "deterministic": kwargs.get("deterministic", True),
            "single_cls": kwargs.get("single_cls", False),
            "rect": kwargs.get("rect", False),
            "cos_lr": kwargs.get("cos_lr", False),
            "close_mosaic": kwargs.get("close_mosaic", 10),
            "amp": kwargs.get("amp", True),
            "fraction": kwargs.get("fraction", 1.0),
            "multi_scale": kwargs.get("multi_scale", 0.0),
            "lr0": kwargs.get("lr0", 0.01),
            "lrf": kwargs.get("lrf", 0.01),
            "momentum": kwargs.get("momentum", 0.937),
            "weight_decay": kwargs.get("weight_decay", 0.0005),
            "warmup_epochs": kwargs.get("warmup_epochs", 3.0),
            "warmup_momentum": kwargs.get("warmup_momentum", 0.8),
            "warmup_bias_lr": kwargs.get("warmup_bias_lr", 0.1),
            "box": kwargs.get("box", 7.5),
            "cls": kwargs.get("cls", 0.5),
            "cls_pw": kwargs.get("cls_pw", 0.0),
            "dfl": kwargs.get("dfl", 1.5),
            "nbs": kwargs.get("nbs", 64),
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "degrees": kwargs.get("degrees", 0.0),
            "translate": kwargs.get("translate", 0.1),
            "scale": kwargs.get("scale", 0.5),
            "shear": kwargs.get("shear", 0.0),
            "perspective": kwargs.get("perspective", 0.0),
            "flipud": kwargs.get("flipud", 0.0),
            "fliplr": kwargs.get("fliplr", 0.5),
            "bgr": kwargs.get("bgr", 0.0),
            "mosaic": kwargs.get("mosaic", 1.0),
            "mixup": kwargs.get("mixup", 0.0),
            "cutmix": kwargs.get("cutmix", 0.0),
            "save": kwargs.get("save", True),
            "save_period": kwargs.get("save_period", -1),
            "verbose": kwargs.get("verbose", True),
            "resume": kwargs.get("resume", False),
            "dis": kwargs.get("dis", 6.0),
            "plots": kwargs.get("plots", True),
        }
        if kwargs.get("freeze") is not None:
            params["freeze"] = kwargs["freeze"]
        for optional in ("time", "cache", "compile", "distill_model"):
            if kwargs.get(optional):
                params[optional] = kwargs[optional]
        return params
