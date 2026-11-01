"""
Tests for collectra.commons.files module (CollectraResultsMetadata and CollectraFile).
"""

import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from collectra.commons.files import CollectraFile, CollectraResultsMetadata


class TestCollectraResultsMetadata:
    """Tests for CollectraResultsMetadata model."""

    def test_model_dump_adds_timestamp(self):
        """model_dump should inject a current ISO-format timestamp."""
        metadata = CollectraResultsMetadata()
        before = datetime.now()
        data = metadata.model_dump()
        after = datetime.now()

        assert "timestamp" in data
        ts = datetime.fromisoformat(data["timestamp"])
        assert before <= ts <= after

    def test_model_dump_strips_none_fields(self):
        """None-valued optional fields should be removed from the dump."""
        metadata = CollectraResultsMetadata()
        data = metadata.model_dump()

        assert "workflow" not in data
        assert "version" not in data
        assert "partition" not in data

    def test_model_dump_retains_present_fields(self):
        """Fields with non-None values should remain in the dump."""
        metadata = CollectraResultsMetadata(
            workflow="detect", version="1.0", partition="train"
        )
        data = metadata.model_dump()

        assert data["workflow"] == "detect"
        assert data["version"] == "1.0"
        assert data["partition"] == "train"
        assert "timestamp" in data

    def test_model_dump_partial_fields(self):
        """Only the explicitly set non-None fields should be present."""
        metadata = CollectraResultsMetadata(partition="val")
        data = metadata.model_dump()

        assert data["partition"] == "val"
        assert "workflow" not in data
        assert "version" not in data

    def test_default_values(self):
        """All optional fields default to None."""
        metadata = CollectraResultsMetadata()
        assert metadata.workflow is None
        assert metadata.version is None
        assert metadata.partition is None
        assert metadata.timestamp is None

    def test_accepts_numeric_version_from_yaml(self):
        """Some results.yaml files were written with an unquoted version
        (e.g. `version: 1.0`), which PyYAML parses as a float, not a str.
        Loading must not reject that — it should be treated as "1.0"."""
        metadata = CollectraResultsMetadata(version=1.0)
        assert metadata.version == "1.0"

    def test_accepts_integer_version_from_yaml(self):
        metadata = CollectraResultsMetadata(version=2)
        assert metadata.version == "2"


class TestCollectraFileModelDump:
    """Tests for CollectraFile.model_dump flattening behaviour."""

    def _make_file(self, data: dict, **kwargs) -> CollectraFile:
        return CollectraFile(
            collectra_file_path=Path("/tmp/fake.arb"),
            collectra_results_metadata=CollectraResultsMetadata(**kwargs),
            data=data,
        )

    def test_data_keys_flattened_to_top_level(self):
        """Keys from 'data' dict should appear as top-level keys in the dump."""
        cf = self._make_file({"image": {"type": "collectra.Image", "data": "test.jpg"}})
        dumped = cf.model_dump()

        assert "image" in dumped
        assert "data" not in dumped  # internal 'data' key removed

    def test_single_element_list_unwrapped(self):
        """A single-element list value should be unwrapped to the element itself."""
        cf = self._make_file({"items": [{"type": "collectra.Image", "data": "a.jpg"}]})
        dumped = cf.model_dump()

        assert dumped["items"] == {"type": "collectra.Image", "data": "a.jpg"}

    def test_multi_element_list_kept(self):
        """A multi-element list should be kept as-is."""
        items = [
            {"type": "collectra.ImageCrop", "data": "a.jpg"},
            {"type": "collectra.ImageCrop", "data": "b.jpg"},
        ]
        cf = self._make_file({"crops": items})
        dumped = cf.model_dump()

        assert dumped["crops"] == items

    def test_non_list_value_kept(self):
        """A non-list value should be kept unchanged."""
        cf = self._make_file({"image": {"type": "collectra.Image", "data": "test.jpg"}})
        dumped = cf.model_dump()

        assert dumped["image"] == {"type": "collectra.Image", "data": "test.jpg"}

    def test_internal_keys_removed(self):
        """Internal keys (data, assets, collectra_file_path) should not appear in dump."""
        cf = self._make_file({"key": "value"})
        dumped = cf.model_dump()

        assert "collectra_file_path" not in dumped
        assert "assets" not in dumped
        # 'data' as internal dict key is removed; only the flattened keys remain
        assert "data" not in dumped

    def test_metadata_included_with_timestamp(self):
        """The results header should be written first, with a fresh timestamp."""
        cf = self._make_file({"k": "v"}, partition="train")
        dumped = cf.model_dump()

        assert list(dumped)[0] == "results"
        meta = dumped["results"]
        assert meta["type"] == "collectra.Results"
        assert "timestamp" in meta
        assert meta["partition"] == "train"


class TestCollectraFileFromData:
    """Tests for CollectraFile.from_data class method."""

    def test_loads_from_results_yaml(self, tmp_path):
        """Should load a CollectraFile from a directory with results.yaml."""
        collectra_dir = tmp_path / "test.arb"
        collectra_dir.mkdir()

        results = {
            "collectra_results_metadata": {
                "timestamp": "2025-01-01T00:00:00",
                "partition": "train",
            },
            "image_sheet": {"type": "collectra.Image", "data": "test.jpg"},
        }
        with open(collectra_dir / "results.yaml", "w") as f:
            yaml.dump(results, f)

        cf = CollectraFile.from_data(collectra_dir)

        assert cf.collectra_file_path == collectra_dir
        assert cf.collectra_results_metadata.partition == "train"
        assert "image_sheet" in cf.data

    def test_raises_if_results_yaml_missing(self, tmp_path):
        """Should raise FileNotFoundError when results.yaml does not exist."""
        empty_dir = tmp_path / "empty.arb"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="results.yaml not found"):
            CollectraFile.from_data(empty_dir)


class TestCollectraFileFromFile:
    """Tests for CollectraFile.from_file class method."""

    def test_creates_collectra_dir(self, tmp_path):
        """Should create a .ext collectra directory for the input file."""
        img = tmp_path / "photo.jpg"
        img.touch()

        cf = CollectraFile.from_file(
            file=img, label="image", ext="arb", output=tmp_path / "out"
        )

        assert cf.collectra_file_path.exists()
        assert cf.collectra_file_path.suffix == ".arb"

    def test_raises_if_exists_and_no_force(self, tmp_path):
        """Should raise FileExistsError if the collectra dir already exists and force=False."""
        img = tmp_path / "photo.jpg"
        img.touch()
        existing = tmp_path / "out" / "photo.arb"
        existing.mkdir(parents=True)

        with pytest.raises(FileExistsError, match="already exists"):
            CollectraFile.from_file(
                file=img, label="image", ext="arb", output=tmp_path / "out"
            )

    def test_force_overwrites_existing(self, tmp_path):
        """Should succeed when force=True even if the directory already exists."""
        img = tmp_path / "photo.jpg"
        img.touch()
        existing = tmp_path / "out" / "photo.arb"
        existing.mkdir(parents=True)

        cf = CollectraFile.from_file(
            file=img, label="image", ext="arb", output=tmp_path / "out", force=True
        )

        assert cf.collectra_file_path == existing

    def test_partition_metadata_propagated(self, tmp_path):
        """Partition kwarg should be stored in metadata."""
        img = tmp_path / "photo.jpg"
        img.touch()

        cf = CollectraFile.from_file(
            file=img,
            label="image",
            ext="arb",
            output=tmp_path / "out",
            partition="val",
        )

        assert cf.collectra_results_metadata.partition == "val"

    def test_asset_registered(self, tmp_path):
        """The original file should be registered as an asset."""
        img = tmp_path / "photo.jpg"
        img.touch()

        cf = CollectraFile.from_file(
            file=img, label="image", ext="arb", output=tmp_path / "out"
        )

        assert img.name in cf.assets
        assert cf.assets[img.name] == img

    def test_data_contains_label_entry(self, tmp_path):
        """The data dict should contain an entry under the provided label."""
        img = tmp_path / "photo.jpg"
        img.touch()

        cf = CollectraFile.from_file(
            file=img, label="specimen", ext="arb", output=tmp_path / "out"
        )

        assert "specimen" in cf.data
        entry = cf.data["specimen"]
        assert entry["type"] == "collectra.Image"
        assert entry["data"] == img.name

    def test_no_output_uses_cwd_relative(self, tmp_path):
        """When output is None the collectra dir is created next to the file."""
        img = tmp_path / "photo.jpg"
        img.touch()

        cf = CollectraFile.from_file(file=img, label="image", ext="arb")

        assert cf.collectra_file_path == img.with_suffix(".arb")


class TestCollectraFileSave:
    """Tests for CollectraFile.save method."""

    def test_save_copies_assets_and_writes_yaml(self, tmp_path):
        """Assets should be copied into the collectra dir and results.yaml should be written."""
        # Create a source image
        src_img = tmp_path / "source" / "photo.jpg"
        src_img.parent.mkdir()
        src_img.write_bytes(b"\xff\xd8fake-jpeg-content")

        # Create the collectra dir
        collectra_dir = tmp_path / "output" / "photo.arb"
        collectra_dir.mkdir(parents=True)

        cf = CollectraFile(
            collectra_file_path=collectra_dir,
            collectra_results_metadata=CollectraResultsMetadata(partition="train"),
            data={"image": {"type": "collectra.Image", "data": "photo.jpg"}},
            assets={"photo.jpg": src_img},
        )
        cf.save()

        # Asset was copied
        assert (collectra_dir / "photo.jpg").exists()
        assert (
            collectra_dir / "photo.jpg"
        ).read_bytes() == b"\xff\xd8fake-jpeg-content"

        # results.yaml was written
        results_file = collectra_dir / "results.yaml"
        assert results_file.exists()
        with open(results_file, "r") as f:
            content = yaml.safe_load(f)
        assert content["results"]["type"] == "collectra.Results"
        assert "collectra_results_metadata" not in content

    def test_save_preserves_existing_directory_on_asset_error(self, tmp_path):
        """A failed update must not delete an existing Collectra directory."""
        collectra_dir = tmp_path / "output" / "photo.arb"
        collectra_dir.mkdir(parents=True)

        cf = CollectraFile(
            collectra_file_path=collectra_dir,
            collectra_results_metadata=CollectraResultsMetadata(),
            data={"image": {"type": "collectra.Image", "data": "photo.jpg"}},
            assets={"photo.jpg": tmp_path / "nonexistent.jpg"},
        )

        with pytest.raises(RuntimeError, match="Failed to save"):
            cf.save()

        assert collectra_dir.exists()

    def test_save_cleans_up_new_directory_on_asset_error(self, tmp_path):
        source = tmp_path / "missing.jpg"
        collectra_file = CollectraFile.from_file(
            file=source,
            label="image",
            ext="arb",
            output=tmp_path / "output",
        )

        with pytest.raises(RuntimeError, match="Failed to save"):
            collectra_file.save()

        assert not collectra_file.collectra_file_path.exists()

    def test_save_no_assets(self, tmp_path):
        """save with no assets should still write results.yaml."""
        collectra_dir = tmp_path / "output" / "photo.arb"
        collectra_dir.mkdir(parents=True)

        cf = CollectraFile(
            collectra_file_path=collectra_dir,
            collectra_results_metadata=CollectraResultsMetadata(),
            data={"image": {"type": "collectra.Image", "data": "photo.jpg"}},
        )
        cf.save()

        assert (collectra_dir / "results.yaml").exists()
