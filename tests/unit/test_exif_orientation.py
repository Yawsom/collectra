"""Images keep their EXIF orientation tag; Collectra reads pixels as displayed."""

import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image as PilImage

from collectra import Collectra
from collectra.displays.preview import open_preview
from collectra.types.images import Image, ImageCrop, image_size

RED, BLUE = (255, 0, 0), (0, 0, 255)


def make_image(path: Path, orientation: int | None, fmt: str = "PNG") -> Path:
    """A 100x50 image: left half red, right half blue, tagged with ``orientation``.

    Orientation 6 means "rotate 90 degrees clockwise to display", so it shows as
    50x100 with the red half on top and the blue half at the bottom.
    """
    pixels = PilImage.new("RGB", (100, 50), RED)
    pixels.paste(BLUE, (50, 0, 100, 50))
    exif = pixels.getexif()
    if orientation is not None:
        exif[0x0112] = orientation
    pixels.save(path, format=fmt, exif=exif)
    return path


def near(pixel, expected, tolerance=40):
    return all(abs(a - b) <= tolerance for a, b in zip(pixel[:3], expected))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def rotated(tmp_path):
    return make_image(tmp_path / "rotated.png", orientation=6)


@pytest.fixture
def upright(tmp_path):
    return make_image(tmp_path / "upright.png", orientation=None)


def test_image_size_is_the_displayed_size(rotated, upright):
    assert image_size(upright) == (100, 50)
    assert image_size(rotated) == (50, 100)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4])
def test_orientations_that_keep_the_aspect_do_not_swap_the_size(tmp_path, orientation):
    assert image_size(make_image(tmp_path / "i.png", orientation)) == (100, 50)


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_orientations_that_turn_the_image_swap_the_size(tmp_path, orientation):
    assert image_size(make_image(tmp_path / "i.png", orientation)) == (50, 100)


def test_image_artefact_reports_displayed_dimensions(rotated):
    image = Image(name="page", data=rotated)
    assert (image.raw_width, image.raw_height) == (50, 100)
    assert (image.width, image.height) == (50, 100)


def test_image_pil_returns_displayed_pixels(rotated):
    pixels = Image(name="page", data=rotated).pil()
    assert pixels.size == (50, 100)
    assert near(pixels.getpixel((25, 10)), RED)
    assert near(pixels.getpixel((25, 90)), BLUE)


def test_untagged_images_are_unchanged(upright):
    pixels = Image(name="page", data=upright).pil()
    assert pixels.size == (100, 50)
    assert near(pixels.getpixel((10, 25)), RED)
    assert near(pixels.getpixel((90, 25)), BLUE)


def test_artefact_rotation_applies_on_top_of_the_exif_orientation(rotated):
    image = Image(name="page", data=rotated, orientation="east")
    assert image.pil().size == (100, 50)


def test_crops_are_taken_from_the_displayed_image(rotated):
    top_half = ImageCrop(
        name="top",
        data=rotated,
        x_center=0.5,
        y_center=0.25,
        width_relative=1.0,
        height_relative=0.5,
    )
    crop = top_half.pil()
    assert crop.size == (50, 50)
    assert near(crop.getpixel((25, 25)), RED)


def test_encoded_image_is_displayed_pixels(rotated):
    import base64
    import io

    decoded = PilImage.open(
        io.BytesIO(base64.b64decode(Image(name="p", data=rotated).get_encoding()))
    )
    assert decoded.size == (50, 100)
    assert near(decoded.convert("RGB").getpixel((25, 10)), RED)


def test_previews_use_the_displayed_orientation(rotated):
    preview = open_preview(rotated, (0, 0, 50, 100), "RGB")
    assert preview.size == (50, 100)
    assert near(preview.getpixel((25, 10)), RED)
    assert near(preview.getpixel((25, 90)), BLUE)
    top = open_preview(rotated, (0, 0, 50, 50), "RGB")
    assert top.size == (50, 50)
    assert near(top.getpixel((25, 25)), RED)


def test_previews_decode_reduced_jpegs_in_the_displayed_orientation(tmp_path):
    path = make_image(tmp_path / "big.jpg", orientation=6, fmt="JPEG")
    preview = open_preview(path, (0, 0, 50, 100), "RGB", max_pixels=25)
    assert preview.size[0] < preview.size[1]
    assert near(preview.getpixel((preview.width // 2, 2)), RED, tolerance=60)


# --- the pipeline must leave images alone ---------------------------------


def build(tmp_path):
    from tests.unit.test_pipeline_stages import MODULE

    return Collectra(
        "Exif",
        "exf",
        "0.1.0",
        path=str(tmp_path),
        first={
            "type": f"{MODULE}.Describe",
            "input": ["page"],
            "output": ["described"],
        },
    )


def test_running_a_task_on_an_existing_folder_does_not_touch_its_images(tmp_path):
    folder = tmp_path / "Page.exf"
    folder.mkdir()
    jpg = make_image(folder / "Page.jpg", orientation=6, fmt="JPEG")
    crop = make_image(folder / "Page-crop.jpg", orientation=None, fmt="JPEG")
    (folder / "results.yaml").write_text(
        "collectra_results_metadata: {workflow: Exif, version: 0.1.0, timestamp: now}\n"
        "file: {type: collectra.Image, id: file, data: Page.jpg}\n"
    )
    before = (digest(jpg), digest(crop))
    build(tmp_path).run(files=[str(folder)])
    assert (digest(jpg), digest(crop)) == before


def test_new_images_are_copied_byte_for_byte_with_their_exif_tag(tmp_path):
    source = (
        make_image(tmp_path / "in" / "Page.jpg", 6, "JPEG")
        if (tmp_path / "in").mkdir() is None
        else None
    )
    build(tmp_path).run(files=[str(source)])
    copied = tmp_path / "in" / "Page.exf" / "Page.jpg"
    assert digest(copied) == digest(source)
    assert PilImage.open(copied).getexif()[0x0112] == 6


# --- training exports and other readers -----------------------------------


def test_baking_orientation_rewrites_only_tagged_images(tmp_path):
    from collectra.types.images import bake_orientation

    rotated = make_image(tmp_path / "r.png", orientation=6)
    bake_orientation(rotated)
    with PilImage.open(rotated) as baked:
        assert baked.size == (50, 100)
        assert 0x0112 not in baked.getexif()
        assert near(baked.convert("RGB").getpixel((25, 10)), RED)
        assert near(baked.convert("RGB").getpixel((25, 90)), BLUE)

    upright = make_image(tmp_path / "u.png", orientation=None)
    before = digest(upright)
    bake_orientation(upright)
    assert digest(upright) == before

    normal = make_image(tmp_path / "n.png", orientation=1)
    before = digest(normal)
    bake_orientation(normal)
    assert digest(normal) == before


def test_baking_leaves_tiffs_alone(tmp_path):
    from collectra.types.images import bake_orientation

    tiff = tmp_path / "s.tif"
    PilImage.new("L", (8, 4)).save(tiff)
    before = digest(tiff)
    bake_orientation(tiff)
    assert digest(tiff) == before


def test_multichannel_loader_reads_displayed_pixels(rotated):
    from collectra.tasks.image_classifier.yolo_multichannel import load_channels

    channels = load_channels(rotated)
    assert tuple(channels.shape[1:]) == (100, 50)


def test_open_oriented_returns_a_usable_independent_image(rotated):
    from collectra.types.images import open_oriented

    image = open_oriented(rotated)
    assert image.size == (50, 100)
    rotated.unlink()  # no lingering file handle needed
    assert image.getpixel((25, 10))[:3] is not None
