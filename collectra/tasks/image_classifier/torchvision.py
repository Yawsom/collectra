"""Torchvision image classification with portable, metadata-bearing checkpoints."""

from __future__ import annotations

from pydantic import BaseModel

from collectra.commons import PositionalFields
from pathlib import Path
from typing import Annotated

from cappa import Arg

from collectra.cli import command
from collectra.logger import get_logger
from collectra.types.images import (
    Image,
    ImageCrop,
    image_channel_count,
    read_tiff_channels,
)
from collectra.types.links import Link
from collectra.utils import threading_locked
from ..machine_learning.training import run_training_command
from .training import (
    ClassificationTrainResult,
    TorchClassifierTask,
    prepare_classification_inputs,
)

__all__ = ["ImageClassifierTorchvision"]
logger = get_logger(__name__)


def _tiff_tensor(pixels):
    """Convert HWC TIFF samples to float32 CHW, preserving every channel."""
    import numpy as np
    import torch

    values = pixels.astype(np.float32)
    if np.issubdtype(pixels.dtype, np.unsignedinteger):
        values /= np.iinfo(pixels.dtype).max
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def _load_classifier_image(path):
    """ImageFolder loader that bypasses Pillow for TIFF images."""
    from torchvision.datasets.folder import pil_loader

    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        return _tiff_tensor(read_tiff_channels(path))
    return pil_loader(path)


def _target_tiff_pixels(image, *, tiff_maxworkers: int | None = None):
    import numpy as np

    target = image.resolve() if type(image) is Link else image
    pixels = read_tiff_channels(target.get_path(), maxworkers=tiff_maxworkers)
    if isinstance(target, ImageCrop):
        left, top, right, bottom = target.coordinates()
        if right <= left or bottom <= top:
            raise ValueError(f"ImageCrop {target.id!r} produces an empty pixel crop")
        pixels = pixels[top:bottom, left:right, :]
    return np.rot90(pixels, k=target.orientation.to_degree() // 90, axes=(0, 1))


class _NormalizeClassifierPixels(PositionalFields, BaseModel):
    mean: list[float]
    std: list[float]

    def __call__(self, pixels):
        import torch
        from torchvision.transforms import functional as F

        if not isinstance(pixels, torch.Tensor):
            pixels = F.to_tensor(pixels)
        channels = pixels.shape[0]
        if len(self.mean) == len(self.std) == 3 and channels != 3:
            # RGB statistics have no defined meaning for spectral bands.
            return pixels
        if len(self.mean) != len(self.std) or len(self.mean) not in (1, channels):
            raise ValueError(
                f"Normalization statistics do not match {channels} image channels"
            )
        return F.normalize(pixels, self.mean, self.std)


def _replace_head(model, num_classes: int):
    """Replace the output projection for torchvision's classification families."""
    from torch import nn

    for name in ("fc", "classifier", "heads", "head"):
        head = getattr(model, name, None)
        if head is None:
            continue
        if isinstance(head, nn.Linear):
            replacement = nn.Linear(
                head.in_features, num_classes, bias=head.bias is not None
            )
            setattr(model, name, replacement)
            return replacement
        if isinstance(head, nn.Sequential):
            projections = [
                (path, module)
                for path, module in head.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]
            if not projections:
                continue
            path, layer = projections[-1]
            if isinstance(layer, nn.Linear):
                replacement = nn.Linear(
                    layer.in_features, num_classes, bias=layer.bias is not None
                )
            else:  # SqueezeNet's classifier is a 1x1 convolution.
                replacement = nn.Conv2d(
                    layer.in_channels,
                    num_classes,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias is not None,
                )
                model.num_classes = num_classes
            parent_path, _, child = path.rpartition(".")
            parent = head.get_submodule(parent_path) if parent_path else head
            setattr(parent, child, replacement)
            return replacement
    raise ValueError(f"Unsupported classification head in {type(model).__name__}")


def _disable_auxiliary_heads(model) -> None:
    # GoogLeNet and Inception should return a single logits tensor in training too.
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    for name in ("AuxLogits", "aux1", "aux2"):
        if hasattr(model, name):
            setattr(model, name, None)


def _ensure_supported_multichannel_architecture(
    architecture: str, input_channels: int
) -> None:
    if input_channels == 3:
        return
    #supported = {"resnet18", "resnet34", "resnet50"}
    supported = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                 "convnext_base", "convnext_tiny", "convnext_small", "convnext_large"}
    if architecture not in supported:
        raise ValueError(
            f"{input_channels}-channel input adaptation is currently supported only for "
            "resnet18, resnet34, resnet50, resnet101, resnet152, convnext_base, convnext_tiny, convnext_small, convnext_large"
        )


class ImageClassifierTorchvision(TorchClassifierTask):
    """Train torchvision classifiers and return predicted class Links."""

    prepare_training_inputs = staticmethod(prepare_classification_inputs)

    def __init__(
        self, name: str, model: str | Path = "resnet18", fusion: bool = False, **kwargs
    ):
        if not isinstance(fusion, bool):
            raise ValueError("fusion must be a YAML boolean (true or false)")
        super().__init__(name, model=model, fusion=fusion, **kwargs)
        self.original_model_path: Path | None = None
        self._categories: list[str] = []
        self._architecture = ""
        self._model_kwargs: dict = {}
        self._preprocessing: dict = {}
        self._device = "cpu"
        self._input_channels = 3

    def _transforms(
        self, *, training: bool = False, fliplr: float = 0.0, flipud: float = 0.0
    ):
        from torchvision import transforms as T

        config = self._preprocessing
        interpolation = T.InterpolationMode(config["interpolation"])
        steps = [
            T.Resize(
                config["resize_size"], interpolation=interpolation, antialias=True
            ),
            T.CenterCrop(config["crop_size"]),
        ]
        if training and fliplr:
            steps.append(T.RandomHorizontalFlip(fliplr))
        if training and flipud:
            steps.append(T.RandomVerticalFlip(flipud))
        steps.append(_NormalizeClassifierPixels(config["mean"], config["std"]))
        return T.Compose(steps)

    def _export_image(
        self, target, destination, min_size, tiff_maxworkers: int | None = None
    ):
        if target.get_path().suffix.lower() not in {".tif", ".tiff"}:
            return super()._export_image(
                target,
                destination,
                min_size,
                tiff_maxworkers=tiff_maxworkers,
            )
        import numpy as np
        import tifffile

        pixels = _target_tiff_pixels(target, tiff_maxworkers=tiff_maxworkers)
        if min(pixels.shape[:2]) < max(1, min_size):
            logger.warning(
                "Skipping classifier image %s below minimum size %d",
                target.id,
                min_size,
            )
            return False
        tifffile.imwrite(
            destination.with_suffix(".tif"),
            np.ascontiguousarray(pixels.transpose(2, 0, 1)),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
        return True

    def _make_datasets(self, train_dir, val_dir, augmentation):
        from torchvision.datasets import ImageFolder

        train = ImageFolder(train_dir, loader=_load_classifier_image)
        val = ImageFolder(val_dir, loader=_load_classifier_image, allow_empty=True)
        detected = {
            path: (
                image_channel_count(path)
                if Path(path).suffix.lower() in {".tif", ".tiff"}
                else 3
            )
            for path, _ in [*train.samples, *val.samples]
        }
        if len(set(detected.values())) != 1:
            details = ", ".join(
                f"{path}: {count} channels" for path, count in detected.items()
            )
            raise ValueError(
                f"Inconsistent input channel counts in classifier dataset: {details}"
            )
        channels = next(iter(detected.values()))
        if self.fusion:
            if channels < 3 or channels % 3:
                raise ValueError(
                    "fusion requires stacked RGB views with a channel count divisible by 3"
                )
            from .torchvision_fusion import RGBViewFusionClassifier

            if (
                isinstance(self.model, RGBViewFusionClassifier)
                and self.model.input_channels != channels
            ):
                raise ValueError(
                    f"Fusion checkpoint expects {self.model.input_channels} channels; dataset has {channels}"
                )
        self._preprocessing["channels"] = channels
        self._input_channels = channels
        if (
            channels != 3
            and len(self._preprocessing["mean"]) == len(self._preprocessing["std"]) == 3
        ):
            if channels % 3 == 0:
                num_views = channels // 3
                logger.info(
                    "Detected %d RGB views (%d channels); repeating pretrained RGB normalization",
                    num_views,
                    channels,
                )
                self._preprocessing["mean"] = self._preprocessing["mean"] * num_views
                self._preprocessing["std"] = self._preprocessing["std"] * num_views
            else:
                logger.info(
                    "Using identity normalization for %d channels; channel count is not divisible by 3",
                    channels,
                )
                self._preprocessing["mean"] = [0.0] * channels
                self._preprocessing["std"] = [1.0] * channels
        train.transform = self._transforms(training=True, **augmentation)
        val.transform = self._transforms()
        return train, val

    def _load(self, *, pretrained: bool = True) -> None:
        import torch
        from torchvision import models

        if isinstance(self.model, torch.nn.Module):
            return
        if not isinstance(self.model, (str, Path)):
            raise ValueError(
                "Model must be a torchvision model name or a checkpoint path"
            )
        path = Path(self.model).expanduser()
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = {
                "architecture",
                "state_dict",
                "classes",
                "preprocessing",
                "model_kwargs",
            }
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError(
                    "Expected a Collectra torchvision checkpoint containing architecture, "
                    "state_dict, classes, preprocessing, and model_kwargs. "
                    "A bare state_dict does not identify its architecture or class labels."
                )
            self._architecture = checkpoint["architecture"]
            self._model_kwargs = checkpoint["model_kwargs"]
            self._categories = checkpoint["classes"]
            self._preprocessing = checkpoint["preprocessing"]
            model = models.get_model(
                self._architecture, weights=None, **self._model_kwargs
            )
            self._input_channels = checkpoint.get("input_channels", 3)
            fusion_spec = checkpoint.get("fusion")

            if self._input_channels != 3 and not fusion_spec:
                if self.fusion:
                    raise ValueError(
                        "Cannot enable fusion on a first-layer-adapted checkpoint; "
                        "start from an RGB model or a fusion checkpoint"
                    )
                _ensure_supported_multichannel_architecture(
                    self._architecture, self._input_channels
                )
                _adapt_rgb_input_channels(
                    model,
                    self._input_channels,
                )

            _disable_auxiliary_heads(model)
            _replace_head(model, len(self._categories))
            if fusion_spec:
                from .torchvision_fusion import restore_fusion

                model = restore_fusion(model, self._input_channels, fusion_spec)
                self.fusion = True
            model.load_state_dict(checkpoint["state_dict"])
            self.original_model_path = path.resolve()
        else:
            name = str(self.model)
            available = models.list_models(module=models)
            if name not in available:
                raise ValueError(
                    f"Unknown torchvision classification model or missing checkpoint: {name!r}. "
                    f"Available models: {', '.join(available)}"
                )
            weights = models.get_model_weights(name).DEFAULT
            preset = weights.transforms()
            self._preprocessing = {
                "crop_size": preset.crop_size,
                "resize_size": preset.resize_size,
                "mean": preset.mean,
                "std": preset.std,
                "interpolation": preset.interpolation.value,
            }
            builder_kwargs = {}
            if name.startswith("vit_"):
                builder_kwargs["image_size"] = preset.crop_size[0]
            if name in {"inception_v3", "googlenet"}:
                builder_kwargs["init_weights"] = False
            model = models.get_model(
                name, weights=weights if pretrained else None, **builder_kwargs
            )
            _disable_auxiliary_heads(model)
            self._architecture = name
            self._model_kwargs = dict(builder_kwargs)
            if hasattr(model, "transform_input"):
                self._model_kwargs["transform_input"] = model.transform_input
            if hasattr(model, "aux_logits"):
                self._model_kwargs["aux_logits"] = False
            if hasattr(model, "image_size"):
                self._model_kwargs["image_size"] = model.image_size
            self._categories = list(weights.meta["categories"])
            self.original_model_path = None
        self.model = model.to(self._device)
        self.model.eval()

    @threading_locked()
    def run(self, *args: Image) -> Link:
        """Classify one image or crop and link the predicted class to it."""
        import torch

        if len(args) != 1 or not isinstance(args[0], Image):
            raise ValueError("This task requires a single Image or ImageCrop input")
        self._device = self._select_device(getattr(self, "device", ""))
        self._load()
        self.model.to(self._device).eval()
        image = args[0]
        if image.get_path().suffix.lower() in {".tif", ".tiff"}:
            pixels = _tiff_tensor(_target_tiff_pixels(image))
            batch = self._transforms()(pixels).unsqueeze(0).to(self._device)
        else:
            with image.pil() as pixels:
                batch = (
                    self._transforms()(pixels.convert("RGB"))
                    .unsqueeze(0)
                    .to(self._device)
                )
        with torch.inference_mode():
            class_index = self.model(batch).argmax(dim=1).item()
        class_name = self._categories[class_index]
        outputs = getattr(self, "output", [])
        outputs = [outputs] if isinstance(outputs, str) else outputs
        if outputs and class_name not in outputs:
            raise ValueError(
                f"Predicted class {class_name!r} is not a configured output: {outputs}"
            )
        return Link(name=class_name, target=image)

    @command
    def train(
        self,
        inputs: Annotated[
            list[str],
            Arg(
                help="Pipeline data files or directories containing classification labels."
            ),
        ],
        model: Annotated[
            str,
            Arg(
                help="Torchvision classification model name (e.g. resnet18, efficientnet_b0, vit_b_16) or a Collectra torchvision checkpoint. Omit to keep the configured model."
            ),
        ] = "",
        output: Annotated[
            Path | None,
            Arg(
                help="Directory for extracted train/val images, metrics, and checkpoints."
            ),
        ] = None,
        use_existing: Annotated[
            bool,
            Arg(
                help="Reuse an existing output train/ and val/ dataset if present; skip image export."
            ),
        ] = False,
        keep_log: Annotated[
            bool,
            Arg(
                help="Keep extracted images and training logs after saving the model to the pipeline."
            ),
        ] = True,
        validation: Annotated[
            str,
            Arg(
                help="Partition value identifying validation images; omit to train without validation."
            ),
        ] = "",
        exclude: Annotated[
            str, Arg(help="Partition value identifying images to exclude.")
        ] = "",
        epochs: Annotated[int, Arg(help="Number of training epochs.")] = 10,
        batch: Annotated[int, Arg(help="Number of images per training batch.")] = 16,
        learning_rate: Annotated[
            float, Arg(help="Initial learning rate for AdamW.")
        ] = 0.0001,
        weight_decay: Annotated[
            float, Arg(help="Weight decay used for regularization.")
        ] = 0.0001,
        pretrained: Annotated[
            bool,
            Arg(
                help="Use DEFAULT pretrained weights for a named model; checkpoints always load their saved weights."
            ),
        ] = True,
        freeze_backbone: Annotated[
            bool,
            Arg(
                help="Train only the final classification layer; freeze all other parameters."
            ),
        ] = False,
        early_stop: Annotated[
            int,
            Arg(
                help="Stop after this many epochs without validation-loss improvement; 0 disables."
            ),
        ] = 10,
        min_size: Annotated[
            int,
            Arg(
                help="Skip images or crops with either dimension below this pixel threshold; 0 disables."
            ),
        ] = 0,
        fliplr: Annotated[
            float,
            Arg(
                help="Probability of horizontal flipping during training, from 0 to 1."
            ),
        ] = 0.5,
        workers: Annotated[
            int,
            Arg(
                help="Number of data-loading worker processes (default 8, capped by available CPUs)."
            ),
        ] = 8,
        wandb: Annotated[
            bool,
            Arg(help="Log training metrics to Weights & Biases."),
        ] = False,
        device: Annotated[
            str,
            Arg(
                help="Torch device such as cpu, cuda, cuda:0, or mps; omit for automatic selection."
            ),
        ] = "",
        seed: Annotated[
            int, Arg(help="Random seed for model initialization and data loading.")
        ] = 0,
        flipud: Annotated[
            float,
            Arg(help="Probability of vertical flipping during training, from 0 to 1."),
        ] = 0.0,
    ):
        """Fine-tune a torchvision image classifier using its weights' prescribed image preprocessing."""
        result = run_training_command(
            self,
            inputs,
            self._train,
            keep_log=keep_log,
            output=output,
            use_existing=use_existing,
            validation=validation,
            exclude=exclude,
            prepare_inputs=self.prepare_training_inputs,
            model=model,
            epochs=epochs,
            batch=batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            early_stop=early_stop,
            min_size=min_size,
            fliplr=fliplr,
            flipud=flipud,
            workers=workers,
            wandb=wandb,
            device=device,
            seed=seed,
        )
        if self.pipeline is not None:
            saved_model = self.pipeline.data[self.name]["model"]
            self.original_model_path = (self.pipeline.path / saved_model).resolve()
        return result

    def _checkpoint(self, epoch: int, metrics: dict) -> dict:
        return {
            "format_version": 1,
            "architecture": self._architecture,
            "model_kwargs": self._model_kwargs,
            "input_channels": self._input_channels,
            "fusion": self.model.fusion_spec if self.fusion else None,
            "classes": self._categories,
            "preprocessing": self._preprocessing,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "epoch": epoch,
            "metrics": metrics,
        }

    def _prepare_training_model(self, classes, pretrained, freeze):
        import torch

        self._load(pretrained=pretrained)

        if self.fusion:
            from .torchvision_fusion import RGBViewFusionClassifier

            if not isinstance(self.model, RGBViewFusionClassifier):
                self.model = RGBViewFusionClassifier(self.model, self._input_channels)
            elif self.model.input_channels != self._input_channels:
                raise ValueError("Fusion checkpoint channel count differs from training inputs")
        elif self._input_channels != 3:
            _ensure_supported_multichannel_architecture(
                self._architecture,
                self._input_channels,
            )
            _adapt_rgb_input_channels(
                self.model,
                self._input_channels,
            )

        # Retain a fine-tuned head only when its class order still matches exactly.
        if classes != self._categories:
            _replace_head(self.model, len(classes))
        self._categories = classes
        self.model.to(self._device)
        self.model.requires_grad_(not freeze)
        if freeze and self.fusion:
            self.model.attention.requires_grad_(True)
            self.model.view_bias.requires_grad_(True)
            self.model.gate.requires_grad_(True)
        if freeze:
            for name in ("fc", "classifier", "heads", "head"):
                head = getattr(self.model, name, None)
                if head is not None:
                    layers = [
                        m
                        for m in head.modules()
                        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
                    ]
                    if layers:
                        layers[-1].requires_grad_(True)
                        break

    def _forward(self, pixels):
        return self.model(pixels.to(self._device))


def _adapt_rgb_input_channels(model, input_channels: int):
    import torch

    # original RT coding
    # if not hasattr(model, "conv1"):
    #     raise ValueError(
    #         f"{type(model).__name__} does not expose a ResNet-style conv1"
    #     )
    # old_conv = model.conv1
    # if not isinstance(old_conv, torch.nn.Conv2d):
    #     raise ValueError(
    #         f"Expected model.conv1 to be Conv2d, got {type(old_conv).__name__}"
    #     )
    #  ---
    # KT addition to allow for convnext models
    # Locate the model's first convolution.
    if hasattr(model, "conv1"):
        # ResNet-style models
        old_conv = model.conv1
        conv_location = "conv1"

    elif (
        hasattr(model, "features")
        and len(model.features) > 0
        and isinstance(model.features[0][0], torch.nn.Conv2d)
    ):
        # TorchVision ConvNeXt-style models
        old_conv = model.features[0][0]
        conv_location = "convnext"

    else:
        raise ValueError(
            f"Could not find first Conv2d for model type "
            f"{type(model).__name__}"
        )

    if not isinstance(old_conv, torch.nn.Conv2d):
        raise ValueError(
            f"Expected first layer to be Conv2d, "
            f"got {type(old_conv).__name__}"
        )
    # end KT addition

    if old_conv.in_channels == input_channels:
        return

    if old_conv.in_channels != 3:
        raise ValueError(
            f"Expected source model to have 3 input channels, "
            f"got {old_conv.in_channels}"
        )

    if old_conv.groups != 1:
        raise ValueError(
            f"Cannot adapt grouped conv1 with groups={old_conv.groups}; expected 1"
            # KT add
            f"groups={old_conv.groups}; expected 1"
        )

    if input_channels % 3 != 0:
        raise ValueError(
            f"Cannot adapt RGB pretrained weights to "
            f"{input_channels} channels: channel count must be "
            f"divisible by 3."
        )

    num_views = input_channels // 3
    # KT additional
    logger.info(
        "Adapting first convolution: %d -> %d input channels (%d RGB stacks)",
        old_conv.in_channels,
        input_channels,
        num_views,
    )
    # end KT additional

    new_conv = torch.nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    ).to(
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )

    with torch.no_grad():
        new_conv.weight.copy_(
            old_conv.weight.repeat(1, num_views, 1, 1)
            / num_views
        )

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    # original RT coding
    # model.conv1 = new_conv
    # ---
    # KT addition
    # Put the adapted convolution back into the model.
    if conv_location == "conv1":
        model.conv1 = new_conv
    else:
        model.features[0][0] = new_conv
    # end KT addition
