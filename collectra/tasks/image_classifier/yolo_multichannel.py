"""Tensor-based Ultralytics classification for TIFF and non-RGB images.

Imported only when needed so CLI task discovery does not load ML backends.
"""

from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PillowImage
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import LOGGER

from collectra.types.images import open_oriented, read_tiff_channels
from .torchvision import _target_tiff_pixels, _tiff_tensor
from .yolo import _classification_channels


def load_channels(path):
    """Load CHW float32; unsigned integers scale by their dtype's full range."""
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        pixels = read_tiff_channels(path)
    else:
        image = open_oriented(path)
        if image.mode == "P":
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        pixels = np.array(image)
        if pixels.ndim == 2:
            pixels = pixels[..., None]
    return _tiff_tensor(pixels)


def channel_transforms(size, args=None):
    """Use shared spatial transforms on all bands, with identity normalization."""
    if args is None:
        return T.Compose([T.Resize(size), T.CenterCrop(size)])
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(1.0 - args.scale, 1.0)),
            T.RandomHorizontalFlip(args.fliplr),
            T.RandomVerticalFlip(args.flipud),
            T.RandomErasing(p=args.erasing),
        ]
    )


class MultichannelClassificationDataset:
    """Ultralytics batch contract backed by ImageFolder and lossless TIFF reading."""

    def __init__(self, root, args, names, augment=False):
        self.base = ImageFolder(root, allow_empty=True)
        mapping = {name: index for index, name in names.items()}
        unknown = set(self.base.classes) - set(mapping)
        if unknown:
            raise ValueError(
                f"Unknown classification directories in {root}: {sorted(unknown)}"
            )
        self.samples = [
            (path, mapping[self.base.classes[index]])
            for path, index in self.base.samples
        ]
        if augment and args.fraction < 1.0:
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.torch_transforms = channel_transforms(
            args.imgsz, args if augment else None
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return {"img": self.torch_transforms(load_channels(path)), "cls": label}


class MultichannelClassificationValidator(ClassificationValidator):
    def __call__(self, trainer=None, model=None):
        # Standalone/final validation rechecks classification metadata (upstream
        # defaults to RGB). Rebuild the loader so channels are corrected before
        # AutoBackend's warmup, including on GPU.
        if trainer is None:
            self.dataloader = None
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path):
        self.data["channels"], _ = _classification_channels(
            self.data.get("train"), self.data.get("val"), self.data.get("test")
        )
        return MultichannelClassificationDataset(
            img_path, self.args, self.data["names"]
        )

    def plot_val_samples(self, batch, ni):
        """RGB sample plots cannot represent spectral bands; keep metric plots."""

    def plot_predictions(self, batch, preds, ni):
        """Do not pass spectral tensors to RGB visualization routines."""


class MultichannelClassificationTrainer(ClassificationTrainer):
    def get_dataset(self):
        data = super().get_dataset()
        data["channels"], _ = _classification_channels(
            data.get("train"), data.get("val"), data.get("test")
        )
        LOGGER.info(
            f"Collectra: preserving {data['channels']} image channels; using spatial "
            "augmentation and erasing, identity normalization, and no RGB colour/auto augmentation."
        )
        return data

    def get_model(self, cfg=None, weights=None, verbose=True):
        # ClassificationModel otherwise lets an old YAML's channels override ch.
        # Dataset metadata is authoritative, including when transferring weights.
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        cfg["channels"] = self.data["channels"]
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.collectra_tensor_classification = True
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return MultichannelClassificationDataset(
            img_path, self.args, self.data["names"], augment=mode == "train"
        )

    def get_validator(self):
        self.loss_names = ["loss"]
        return MultichannelClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni):
        """RGB sample plots cannot represent spectral bands; keep metric plots."""


@torch.inference_mode()
def classify_multichannel(yolo, image):
    """Classify directly from tensors, bypassing Ultralytics' RGB predictor."""
    if image.get_path().suffix.lower() in {".tif", ".tiff"}:
        pixels = _tiff_tensor(_target_tiff_pixels(image))
    else:
        # Pillow supports the ordinary grayscale/RGBA crop and orientation path.
        with image.pil() as source:
            if source.mode == "P":
                source = source.convert(
                    "RGBA" if "transparency" in source.info else "RGB"
                )
            values = np.array(source)
        pixels = _tiff_tensor(values[..., None] if values.ndim == 2 else values)
    model = yolo.model
    first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    if pixels.shape[0] != first_conv.in_channels:
        raise ValueError(
            f"{image.get_path()}: {pixels.shape[0]} image channels, "
            f"but the classification model expects {first_conv.in_channels}"
        )
    transform = getattr(model, "transforms", None)
    # Models trained via Collectra store tensor transforms in the checkpoint.
    # A separately prepared model may still carry upstream PIL/RGB transforms.
    if transform is None or any(
        isinstance(t, T.ToTensor) for t in getattr(transform, "transforms", [])
    ):
        transform = channel_transforms(yolo.overrides.get("imgsz", 224))
    parameter = next(model.parameters())
    batch = (
        transform(pixels)
        .unsqueeze(0)
        .to(device=parameter.device, dtype=parameter.dtype)
    )
    model.eval()
    predictions = model(batch)
    predictions = (
        predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    )
    return model.names[predictions.argmax(dim=1).item()]
