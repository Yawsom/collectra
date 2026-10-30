"""Bound browser image/canvas memory independently of source resolution."""

from pathlib import Path

from PIL import Image, ImageOps

MAX_PREVIEW_EDGE = 4096


def bounded(image, *, mask=False, max_pixels=None):
    limit = max_pixels if max_pixels is not None else MAX_PREVIEW_EDGE
    if max(image.size) <= limit:
        return image
    size = tuple(max(1, round(value * limit / max(image.size))) for value in image.size)
    return image.resize(size, Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)


def open_preview(path, bounds, mode, *, max_pixels=None):
    """Decode JPEGs at reduced resolution where the displayed crop permits it."""
    limit = max_pixels if max_pixels is not None else MAX_PREVIEW_EDGE
    with Image.open(path) as source:
        # ``bounds`` are in displayed (EXIF-oriented) pixels, like every other
        # coordinate in Collectra, so orient before cropping.
        raw_width, raw_height = source.size
        oriented = Path(path).suffix.lower() not in {".tif", ".tiff"}
        turned = oriented and source.getexif().get(0x0112) in {5, 6, 7, 8}
        width, height = (raw_height, raw_width) if turned else (raw_width, raw_height)
        left, top, right, bottom = bounds
        scale = min(1, limit / max(right - left, bottom - top))
        source.draft(
            mode, (max(1, round(raw_width * scale)), max(1, round(raw_height * scale)))
        )
        source = ImageOps.exif_transpose(source) if oriented else source.copy()
        scaled = (left * source.width / width, top * source.height / height,
                  right * source.width / width, bottom * source.height / height)
        if bounds != (0, 0, width, height):
            source = source.crop(tuple(round(value) for value in scaled))
        source.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        return source.convert(mode)
