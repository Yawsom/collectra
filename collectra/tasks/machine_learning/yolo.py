"""Shared model lifecycle for YOLO-backed tasks."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...logger import get_logger
from ..base import Task

if TYPE_CHECKING:
    from ultralytics.models import YOLO

__all__ = ["YOLOTask"]

logger = get_logger(__name__)


def get_yolo_class():
    """Import the optional backend when a YOLO task executes."""
    try:
        from ultralytics.models import YOLO
    except ModuleNotFoundError as error:
        if error.name != "ultralytics":
            raise
        raise ImportError(
            'YOLO tasks require optional dependencies. Install with: '
            'pip install "collectra[yolo]"'
        ) from error
    return YOLO


class YOLOTask(Task):
    """Base class for tasks backed by an Ultralytics YOLO model."""

    model: Any = ""  # a path or a loaded Ultralytics YOLO model
    original_model_path: str | Path = ""

    def __init__(self, name: str, model: str | Path | YOLO = "", **kwargs) -> None:
        super().__init__(name, model=model, **kwargs)

    @contextmanager
    def _wandb_logging(self, enabled: bool = False):
        """Select Ultralytics' native W&B integration for this training run."""
        from ultralytics import settings

        previous = settings["wandb"]
        module_name = "ultralytics.utils.callbacks.wb"
        callbacks = getattr(self.model, "callbacks", {})

        def remove_wandb_callbacks():
            # Reused models can retain callbacks from an earlier enabled run.
            for handlers in callbacks.values():
                handlers[:] = [
                    handler
                    for handler in handlers
                    if getattr(handler, "__module__", None) != module_name
                ]

        integration = None
        try:
            if previous != enabled:
                settings.update({"wandb": enabled})
            # The native integration checks settings at import time. Reload it
            # so enabling/disabling logging also works on subsequent runs.
            integration = importlib.import_module(module_name)
            importlib.reload(integration)
            remove_wandb_callbacks()
            yield
        finally:
            remove_wandb_callbacks()
            if settings["wandb"] != previous:
                settings.update({"wandb": previous})
            if integration is not None:
                importlib.reload(integration)

    def _init_model(self) -> None:
        """Load and validate the configured YOLO model."""
        YOLO = get_yolo_class()

        self._load()
        if not isinstance(self.model, YOLO):
            raise ValueError("Model must be a YOLO instance")

    def _reload(self) -> None:
        """Reload the model from its original path."""
        if not self.original_model_path:
            logger.warning("Original model path is not set. Skipping...")
            return
        self.model = self.original_model_path
        self._load()

    def _load(self) -> None:
        """Load a YOLO model only when execution or training requires it."""
        YOLO = get_yolo_class()

        if self.model and isinstance(self.model, (str, Path)):
            self.original_model_path = self.model
            self.model = YOLO(Path(self.model))
        config = getattr(getattr(self.model, "model", None), "yaml", None)
        if isinstance(config, dict) and "feature_fusion" in config:
            from ..object_detection.yolo_feature_fusion import enable_feature_fusion

            enable_feature_fusion(self.model)
        if not isinstance(self.model, YOLO):
            logger.warning("Invalid YOLO model value: %r", self.model)
