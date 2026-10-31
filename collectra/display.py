"""Artefact display protocol, independent of the desktop GUI.

``display(context)`` returns a JSON-compatible dict with ``kind`` set to
``image``, ``text``, or ``properties``. Image sources are PNG data URLs; image
annotations use viewer-relative normalized coordinates. Text formats are
``plain``, ``markdown``, or ``xml``. Custom classes may reuse these views.
"""

import base64
import io
from pathlib import Path

from collectra.utils import load_class_from_string


class DisplayContext:
    def __init__(self, records: dict, directory: Path, rgb_view: int = 0, max_pixels: int | None = None):
        from collectra.displays.preview import MAX_PREVIEW_EDGE

        if max_pixels is None:
            max_pixels = MAX_PREVIEW_EDGE
        if type(max_pixels) is not int or max_pixels < 1:
            raise ValueError("max_pixels must be a positive integer")
        if type(rgb_view) is not int or rgb_view < 0:
            raise ValueError("RGB view index must be a non-negative integer")
        self.records = records
        self.directory = Path(directory).resolve()
        self.instances = {}
        self.rgb_view = rgb_view
        self.max_pixels = max_pixels

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.directory / path

    def artefact(self, node_id):
        from collectra.types.images import Image, ImageSegmentation
        from collectra.types.links import Link
        from collectra.types.texts import Text

        if node_id in self.instances:
            return self.instances[node_id]
        record = dict(self.records[node_id])
        cls = load_class_from_string(record.pop("type"))
        record["name"] = record.pop("label", record.get("name", ""))
        if isinstance(record.get("parents"), str):
            record["parents"] = [record["parents"]]
        if issubclass(cls, Image):
            record["data"] = self.path(record.get("data", record.get("path", "")))
        if issubclass(cls, ImageSegmentation):
            record["mask"] = self.path(record.get("mask", ""))
        elif issubclass(cls, Text):
            source = self.text_path(node_id)
            if source:
                if "path" not in record:
                    record.pop("data", None)
                record["path"] = source
        accepted = set(cls.model_fields) - cls._runtime_fields
        item = cls(**{k: v for k, v in record.items() if k in accepted})
        self.instances[node_id] = item
        if type(item) is Link:
            item.target = self.artefact(item.parent_id)
        return item

    def text_path(self, node_id):
        record = self.records[node_id]
        value = record.get("path", record.get("data", ""))
        if not isinstance(value, (str, Path)) or not value:
            return None
        try:
            path = self.path(value)
            return path if path.is_file() else None
        except (OSError, ValueError):
            return None

    def text(self, item):
        from collectra.displays.texts import text

        return text(self, item)

    @staticmethod
    def publish_image(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def renderer(module_path):
        """Publish a local JavaScript ES module supplied by an artefact package.

        Return this URL as ``renderer`` in a display descriptor. The module
        exports a default render function; see docs/artefacts/display.rst.
        Use a path relative to the artefact's Python module, not results data.
        """
        source = Path(module_path).read_bytes()
        return "data:text/javascript;base64," + base64.b64encode(source).decode()

    def region(self, image, region, inverse=False):
        from collectra.displays.images import region as transform_region

        return transform_region(self, image, region, inverse=inverse)

    def annotations(self, image):
        from collectra.displays.images import annotations

        return annotations(self, image)

    def image(self, item):
        from collectra.displays.images import image

        return image(self, item)
