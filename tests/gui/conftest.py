"""
Shared pytest fixtures for collectra.gui tests.
"""

import pytest

from collectra.gui.data_display import CollectraGraph


@pytest.fixture(autouse=True)
def isolated_gui_session_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "collectra.gui.session_cache.user_cache_dir",
        lambda *args, **kwargs: str(tmp_path / "user-cache"),
    )


@pytest.fixture(autouse=True)
def isolated_gui_arrangement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "collectra.gui.arrangement.user_cache_dir",
        lambda *args, **kwargs: str(tmp_path / "user-arrangement"),
    )


@pytest.fixture
def sample_yaml_data():
    """Minimal valid YAML structure for testing."""
    return {
        "collectra_results_metadata": {"version": "1.0.0"},
        "image_label": [
            {
                "type": "collectra.Image",
                "id": "img_001",
                "data": "test_image.jpg",
            }
        ],
        "crop_label": [
            {
                "type": "collectra.ImageCrop",
                "id": "crop_001",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.1,
            }
        ],
        "text_label": [
            {
                "type": "collectra.Text",
                "id": "text_001",
                "parents": "crop_001",
                "data": "Hello World",
            }
        ],
    }


@pytest.fixture
def complex_yaml_data():
    """Complex YAML with nested crops, containers, and multiple text children."""
    return {
        "collectra_results_metadata": {"version": "1.0.0"},
        "image_label": [
            {
                "type": "collectra.Image",
                "id": "img_001",
                "data": "test_image.jpg",
            }
        ],
        "container_crop": [
            {
                "type": "collectra.ImageCrop",
                "id": "container_crop_001",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.8,
                "height_relative": 0.8,
            }
        ],
        "leaf_crop": [
            {
                "type": "collectra.ImageCrop",
                "id": "leaf_crop_001",
                "parents": "container_crop_001",
                "data": "test_image.jpg",
                "x_center": 0.3,
                "y_center": 0.3,
                "width_relative": 0.2,
                "height_relative": 0.1,
            },
            {
                "type": "collectra.ImageCrop",
                "id": "leaf_crop_no_text",
                "parents": "container_crop_001",
                "data": "test_image.jpg",
                "x_center": 0.7,
                "y_center": 0.7,
                "width_relative": 0.2,
                "height_relative": 0.1,
            },
        ],
        "text_label": [
            {
                "type": "collectra.Text",
                "id": "text_001",
                "parents": "leaf_crop_001",
                "data": "First text",
            },
            {
                "type": "collectra.Text",
                "id": "text_002",
                "parents": "text_001",
                "data": "Deepest text",
            },
        ],
    }


@pytest.fixture
def sample_graph(sample_yaml_data):
    """Pre-built CollectraGraph from sample_yaml_data."""
    return CollectraGraph.from_yaml_data(sample_yaml_data)


@pytest.fixture
def complex_graph(complex_yaml_data):
    """Pre-built CollectraGraph from complex_yaml_data."""
    return CollectraGraph.from_yaml_data(complex_yaml_data)


@pytest.fixture
def empty_graph():
    """Empty CollectraGraph."""
    return CollectraGraph()


@pytest.fixture
def temp_yaml_file(tmp_path, sample_yaml_data):
    """Create a temporary Collectra file for file I/O tests."""
    import yaml

    collectra_file = tmp_path / "test.collectra"
    collectra_file.mkdir()
    yaml_file = collectra_file / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(sample_yaml_data, f)
    return yaml_file


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file for image tests."""
    # Create a minimal valid PNG (1x1 pixel, red)
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_file = tmp_path / "test_image.png"
    image_file.write_bytes(png_data)
    return image_file


@pytest.fixture
def temp_folder_with_files(tmp_path, sample_yaml_data):
    """Create a temp folder containing both a YAML and an image file."""
    import yaml

    # Create YAML file
    yaml_file = tmp_path / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(sample_yaml_data, f)

    # Create minimal PNG
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_file = tmp_path / "image.png"
    image_file.write_bytes(png_data)

    return tmp_path


@pytest.fixture
def multi_label_yaml_data():
    """YAML data with multiple unique labels for label statistics tests."""
    return {
        "collectra_results_metadata": {"version": "1.0.0"},
        "image_label": [
            {
                "type": "collectra.Image",
                "id": "img_001",
                "data": "test_image.jpg",
            }
        ],
        "header_crop": [
            {
                "type": "collectra.ImageCrop",
                "id": "header_001",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.1,
                "width_relative": 0.8,
                "height_relative": 0.1,
            },
            {
                "type": "collectra.ImageCrop",
                "id": "header_002",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.2,
                "width_relative": 0.8,
                "height_relative": 0.1,
            },
        ],
        "body_crop": [
            {
                "type": "collectra.ImageCrop",
                "id": "body_001",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.8,
                "height_relative": 0.3,
            },
        ],
        "footer_crop": [
            {
                "type": "collectra.ImageCrop",
                "id": "footer_001",
                "parents": "img_001",
                "data": "test_image.jpg",
                "x_center": 0.5,
                "y_center": 0.9,
                "width_relative": 0.8,
                "height_relative": 0.1,
            },
        ],
    }


@pytest.fixture
def temp_parent_folder_with_subfolders(tmp_path, sample_yaml_data):
    """Parent folder with multiple .collectra subdirectories."""
    import yaml

    # Create minimal PNG for each subfolder
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    # Create 3 .collectra subfolders
    for i in range(3):
        subfolder = tmp_path / f"image_{i:03d}.collectra"
        subfolder.mkdir()

        # Create YAML file
        yaml_file = subfolder / "results.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sample_yaml_data, f)

        # Create image file
        image_file = subfolder / f"image_{i:03d}.png"
        image_file.write_bytes(png_data)

    return tmp_path


@pytest.fixture
def backend_with_extension(tmp_path):
    """Pre-configured GUIBackend with extension set."""
    from collectra.gui.backend import GUIBackend
    from collectra.pipelines.base import Pipeline

    return GUIBackend(Pipeline("test", "collectra", "1.0", path=tmp_path))


@pytest.fixture
def temp_collectra_folder_with_png(tmp_path, sample_yaml_data):
    """Create a .collectra folder containing a PNG image (not JPG)."""
    import yaml

    # Create .collectra subfolder
    subfolder = tmp_path / "test_image.collectra"
    subfolder.mkdir()

    # Create YAML file with image data referencing PNG
    yaml_data = sample_yaml_data.copy()
    yaml_data["image_label"] = [
        {
            "type": "collectra.Image",
            "id": "img_001",
            "data": "test_image.png",
        }
    ]

    yaml_file = subfolder / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f)

    # Create minimal PNG (1x1 pixel)
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_file = subfolder / "test_image.png"
    image_file.write_bytes(png_data)

    return subfolder


@pytest.fixture
def temp_collectra_folder_with_tiff(tmp_path, sample_yaml_data):
    """Create a .collectra folder containing a TIFF image."""
    import yaml

    subfolder = tmp_path / "test_image.collectra"
    subfolder.mkdir()

    yaml_data = sample_yaml_data.copy()
    yaml_data["image_label"] = [
        {
            "type": "collectra.Image",
            "id": "img_001",
            "data": "test_image.tiff",
        }
    ]

    yaml_file = subfolder / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f)

    # Create minimal TIFF (little-endian header)
    tiff_data = b"II*\x00\x08\x00\x00\x00" + b"\x00" * 100
    image_file = subfolder / "test_image.tiff"
    image_file.write_bytes(tiff_data)

    return subfolder


@pytest.fixture
def temp_collectra_folder_with_gif(tmp_path, sample_yaml_data):
    """Create a .collectra folder containing a GIF image."""
    import yaml

    subfolder = tmp_path / "test_image.collectra"
    subfolder.mkdir()

    yaml_data = sample_yaml_data.copy()
    yaml_data["image_label"] = [
        {
            "type": "collectra.Image",
            "id": "img_001",
            "data": "test_image.gif",
        }
    ]

    yaml_file = subfolder / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f)

    # Create minimal GIF (GIF89a header)
    gif_data = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;\x00"
    image_file = subfolder / "test_image.gif"
    image_file.write_bytes(gif_data)

    return subfolder


@pytest.fixture
def temp_grapto_folder_with_png(tmp_path, sample_yaml_data):
    """Create a .grapto folder containing a PNG image to test workflow extension handling."""
    import yaml

    # Create .grapto subfolder (simulates the bug scenario)
    subfolder = tmp_path / "test_image.grapto"
    subfolder.mkdir()

    yaml_data = sample_yaml_data.copy()
    yaml_data["image_label"] = [
        {
            "type": "collectra.Image",
            "id": "img_001",
            "data": "test_image.png",
        }
    ]

    yaml_file = subfolder / "results.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f)

    # Create minimal PNG
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_file = subfolder / "test_image.png"
    image_file.write_bytes(png_data)

    return subfolder


@pytest.fixture
def backend_with_grapto_extension(tmp_path):
    """Pre-configured GUIBackend with grapto extension set."""
    from collectra.gui.backend import GUIBackend
    from collectra.pipelines.base import Pipeline

    return GUIBackend(Pipeline("test", "grapto", "1.0", path=tmp_path))


@pytest.fixture
def temp_project_with_collectra(tmp_path):
    """Project root holding a .collectra folder with a PNG logo and CSS theme."""

    collectra = tmp_path / "proj.collectra"
    collectra.mkdir()

    # Arbitrary, deliberately non-"Grapto" sample values — proves the readers are
    # generic: they pull whatever the metadata holds, not any hardcoded project name.
    # Minimal 1x1 PNG
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (collectra / "BrandMark.png").write_bytes(png_data)
    (collectra / "Palette.css").write_text("#dag-root { --dag-bg: #002b36; }")

    return tmp_path
