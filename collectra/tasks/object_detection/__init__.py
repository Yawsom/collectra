"""Object-detection task implementations."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ObjectDetectionTask": "collectra.tasks.object_detection.base",
    "ObjectDetectionYOLO": "collectra.tasks.object_detection.yolo",
    "ObjectDetectionDETR": "collectra.tasks.object_detection.detr",
    "DetectionTrainResult": "collectra.tasks.object_detection.detr",
    "ObjectDetectionRFDETR": "collectra.tasks.object_detection.rfdetr",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
