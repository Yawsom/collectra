"""What every object detector has in common.

The detectors (YOLO, RF-DETR, DETR) share their settings, the way a setting
given for one run beats the task's own, how old option names are still read,
how a device is chosen, and how predictions for many images are made in batches
and kept for ``run``. A backend adds only what is really its own.
"""

from __future__ import annotations

import platform
import warnings
from typing import Annotated, Any, ClassVar

from cappa import Arg
from pydantic import Field, PrivateAttr, model_validator

from collectra.types.images import Image
from collectra.utils import threading_locked

from ..base import Task
from .batching import image_cache_key

# Command-line options the detectors share. Leaving one out keeps the task's
# own setting (from pipeline.yaml, or the default).
RunResolution = Annotated[
    int | None,
    Arg(
        help="Image size in pixels to run at. Omit to use the one chosen for the task, or else the model's own."
    ),
]
TrainResolution = Annotated[
    int | None,
    Arg(help="Image size in pixels to train at. Omit to use the task's setting."),
]
Threshold = Annotated[
    float | None,
    Arg(
        help="Minimum confidence for a detection, from 0 to 1. Omit to use the task's setting."
    ),
]
Epochs = Annotated[
    int | None, Arg(help="Number of training epochs. Omit to use the task's setting.")
]
Batch = Annotated[
    int | None,
    Arg(help="Training batch size. Omit to use the task's setting."),
]
LearningRate = Annotated[
    float | None, Arg(help="Learning rate. Omit to use the task's setting.")
]
WeightDecay = Annotated[
    float | None,
    Arg(help="Weight decay used for regularization. Omit to use the task's setting."),
]


class ObjectDetectionTask(Task):
    """Settings and behaviour shared by all object detectors.

    Attributes:
        resolution: Image size in pixels. Training uses it (1280 unless set).
            Running uses it only when it was set explicitly, so a model keeps
            the size it was trained with unless you ask for another.
        threshold: Minimum confidence for a detection, from 0 to 1.
        epochs: Number of training epochs.
        batch: Training batch size.
        learning_rate: Training learning rate.
        weight_decay: Training weight decay.

    A backend overrides a default where its own library needs another value.
    Names that used to be accepted (``imgsz``) are still read, with a warning.
    """

    resolution: int = Field(default=1280, gt=0)
    threshold: float = Field(default=0.25, ge=0, le=1)
    epochs: int = Field(default=100, ge=1)
    batch: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)

    # Old name -> current name. A backend adds its own.
    RENAMED_SETTINGS: ClassVar[dict[str, str]] = {"imgsz": "resolution"}
    # Keys a pipeline entry has besides settings: they say what the task reads
    # and writes, and are not mistakes.
    WIRING: ClassVar[frozenset[str]] = frozenset({"input", "output"})

    _prefetched: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _read_old_setting_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for old, new in cls.RENAMED_SETTINGS.items():
            if old not in data:
                continue
            if new in data:
                raise ValueError(
                    f"{cls.__name__} was given both {old!r} and {new!r}; "
                    f"use only {new!r}"
                )
            warnings.warn(
                f"{old!r} is deprecated; use {new!r} instead",
                FutureWarning,
                stacklevel=2,
            )
            data[new] = data.pop(old)
        return data

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        unknown = sorted(set(self.model_extra or {}) - self.WIRING)
        if unknown:
            known = sorted(set(type(self).model_fields) - {"pipeline", "context"})
            warnings.warn(
                f"{type(self).__name__} {self.name!r} ignores unknown "
                f"setting(s): {', '.join(unknown)}. Settings it understands: "
                f"{', '.join(known)}",
                UserWarning,
                stacklevel=2,
            )

    # --- settings for one run -------------------------------------------------------

    @property
    def inference_resolution(self) -> int | None:
        """The size to run at, or None to let the model use its own."""
        return self.resolution if "resolution" in self.model_fields_set else None

    def _settings_for_run(self, **given: Any) -> dict[str, Any]:
        """Each value given for this run, else the task's own setting."""
        return {
            name: getattr(self, name) if value is None else value
            for name, value in given.items()
        }

    @staticmethod
    def _current_name(old: str, new: str, old_value: Any, new_value: Any) -> Any:
        """The value of an option that has been renamed, from either name."""
        if old_value is None:
            return new_value
        if new_value is not None:
            raise ValueError(f"Give {new!r}, not both {old!r} and {new!r}")
        warnings.warn(
            f"{old!r} is deprecated; use {new!r} instead", FutureWarning, stacklevel=3
        )
        return old_value

    def _select_device(self) -> str:
        import torch

        return (
            "mps"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    # --- predicting many images at once ---------------------------------------------

    def _predict_batch(self, images: list[Image]) -> list:
        """Run the model over ``images`` together, one prediction each."""
        raise NotImplementedError

    def _before_prefetch(self) -> None:
        """Called once before a prefetch; a backend can report its settings."""

    def _prefetch_key(self, image: Image) -> tuple:
        return image_cache_key(image, self.inference_resolution)

    @threading_locked()
    def prefetch(self, entries: list[list]) -> None:
        """Run the model over many images in batches, caching results for ``run``.

        A backend that does not predict in batches (it has no ``_predict_batch``)
        does nothing here, and ``run`` predicts each image itself.
        """
        if type(self)._predict_batch is ObjectDetectionTask._predict_batch:
            return
        self._init_model()
        images = [
            entry[0]
            for entry in entries
            if len(entry) == 1 and isinstance(entry[0], Image)
        ]
        self._prefetched = {}
        self._before_prefetch()
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            for image, predicted in zip(batch, self._predict_batch(batch)):
                self._prefetched[self._prefetch_key(image)] = predicted

    def clear_prefetch(self) -> None:
        self._prefetched = {}

    def _take_prefetched(self, image: Image):
        """The cached prediction for ``image``, if prefetch made one."""
        return self._prefetched.pop(self._prefetch_key(image), None)

    def _init_model(self) -> None:
        raise NotImplementedError
