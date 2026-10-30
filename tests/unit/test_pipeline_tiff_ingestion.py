"""Pipeline ingestion must preserve scientific TIFF data and original inputs."""

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from collectra import Collectra
from collectra.types.images import read_tiff_channels


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".TIF"])
@pytest.mark.parametrize("axes", ["CYX", "YXC", "QYX"])
def test_ingestion_preserves_tiff_source_and_copy(tmp_path, suffix, axes):
    pixels = np.arange(75 * 9 * 11, dtype=np.uint16).reshape(9, 11, 75)
    source = tmp_path / f"views{suffix}"
    tifffile.imwrite(
        source,
        pixels if axes == "YXC" else pixels.transpose(2, 0, 1),
        photometric="minisblack",
        metadata={"axes": axes},
    )
    original = source.read_bytes()
    pipeline = Collectra("test", "collectra", "1", path=tmp_path)

    folder = pipeline._handle_image_file(source, tmp_path / "output")
    copied = folder / source.name

    assert source.read_bytes() == original
    assert copied.read_bytes() == original
    np.testing.assert_array_equal(read_tiff_channels(copied), pixels)

    # Existing result folders also go through EXIF cleanup on subsequent runs.
    pipeline._handle_existing_folder(folder, None)
    assert copied.read_bytes() == original
    new_folder = pipeline._handle_existing_folder(folder, tmp_path / "rerun")
    assert (new_folder / source.name).read_bytes() == original


def test_ingestion_keeps_raster_copy_identical_including_exif(tmp_path):
    source = tmp_path / "rgb.jpg"
    exif = PILImage.Exif()
    exif[270] = "Original metadata"
    PILImage.new("RGB", (16, 16)).save(source, exif=exif)
    original = source.read_bytes()
    pipeline = Collectra("test", "collectra", "1", path=tmp_path)

    folder = pipeline._handle_image_file(source, tmp_path / "output")

    assert source.read_bytes() == original
    assert (folder / source.name).read_bytes() == original
    with PILImage.open(folder / source.name) as copied:
        assert copied.getexif()[270] == "Original metadata"
