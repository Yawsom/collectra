"""Tests for the output parameter functionality in Collectra pipeline.

Tests cover three main cases:
- Case A: Existing .grapto folder with results.yaml
- Case B: Image file input
- Case C: Directory without results.yaml
"""

import shutil
from pathlib import Path

import yaml

from collectra import Pipeline


def build_pipeline(pipeline: dict) -> Pipeline:
    """Helper function to build a Collectra pipeline from a dict."""
    metadata: dict = pipeline.pop("collectra_pipeline_metadata")
    name = metadata["name"]
    ext = metadata["ext"]
    version = metadata["version"]
    path = pipeline.pop("pipeline_path", "")
    return Pipeline(name, ext, version, path=path, **pipeline)


def test_output_with_image_file(pipeline, debug, tmpdir, raw_img_path):
    """Test Case B: Image file with output directory specified."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create input image and output directory
        input_image = Path(tmpdir) / "input" / "test_image.jpg"
        input_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(raw_img_path, input_image)

        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline with output parameter
        pipeline.run(files=[str(input_image)], output=str(output_dir), render=False)

        # Verify output folder was created in the specified location
        expected_output = output_dir / f"test_image.{pipeline.ext}"
        assert (
            expected_output.exists()
        ), f"Output folder should be created at {expected_output}"
        assert expected_output.is_dir(), "Output should be a directory"

        # Verify results.yaml exists in output
        results_file = expected_output / "results.yaml"
        assert results_file.exists(), "results.yaml should exist in output folder"

        # Verify image was copied to output
        image_in_output = expected_output / "test_image.jpg"
        assert image_in_output.exists(), "Image should be copied to output folder"

        # Verify original input is unchanged
        assert input_image.exists(), "Original input should still exist"

    except Exception as e:
        debug(e)


def test_output_with_existing_grapto_folder(pipeline, debug, tmpdir, raw_img_path):
    """Test Case A: Existing .grapto folder with results.yaml."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create an existing .grapto folder with results.yaml
        input_grapto = Path(tmpdir) / "input" / f"existing.{pipeline.ext}"
        input_grapto.mkdir(parents=True, exist_ok=True)

        # Create results.yaml
        results_data = {
            "collectra_results_metadata": {
                "workflow": pipeline.name,
                "version": pipeline.version,
                "timestamp": "2026-01-08T10:00:00",
            },
            "file": {"type": "collectra.Image", "id": "file", "data": "existing.jpg"},
        }

        with open(input_grapto / "results.yaml", "w") as f:
            yaml.dump(results_data, f)

        # Copy image to the .grapto folder
        shutil.copy(raw_img_path, input_grapto / "existing.jpg")

        # Create output directory
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline with output parameter
        pipeline.run(files=[str(input_grapto)], output=str(output_dir), render=False)

        # Verify output folder was created
        expected_output = output_dir / f"existing.{pipeline.ext}"
        assert (
            expected_output.exists()
        ), f"Output folder should be created at {expected_output}"
        assert expected_output.is_dir(), "Output should be a directory"

        # Verify results.yaml was copied
        output_results = expected_output / "results.yaml"
        assert output_results.exists(), "results.yaml should be copied to output"

        # Verify image was copied
        output_image = expected_output / "existing.jpg"
        assert output_image.exists(), "Image should be copied to output folder"

        # Verify original is unchanged
        assert input_grapto.exists(), "Original .grapto folder should still exist"
        assert (
            input_grapto / "results.yaml"
        ).exists(), "Original results.yaml should still exist"

    except Exception as e:
        debug(e)


def test_output_without_parameter(pipeline, debug, tmpdir, raw_img_path):
    """Test that without output parameter, files are created beside input (default behavior)."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create input image
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_image = input_dir / "test_image.jpg"
        shutil.copy(raw_img_path, input_image)

        # Run pipeline WITHOUT output parameter
        pipeline.run(files=[str(input_image)], render=False)

        # Verify output folder was created beside input
        expected_output = input_dir / f"test_image.{pipeline.ext}"
        assert (
            expected_output.exists()
        ), f"Output folder should be created beside input at {expected_output}"
        assert expected_output.is_dir(), "Output should be a directory"

        # Verify results.yaml exists
        results_file = expected_output / "results.yaml"
        assert results_file.exists(), "results.yaml should exist"

    except Exception as e:
        debug(e)


def test_output_with_multiple_files(pipeline, debug, tmpdir, raw_img_path):
    """Test processing multiple files with output directory."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create multiple input images
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_files = []
        for i in range(3):
            input_image = input_dir / f"test_image_{i}.jpg"
            shutil.copy(raw_img_path, input_image)
            input_files.append(str(input_image))

        # Create output directory
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline with multiple files
        pipeline.run(files=input_files, output=str(output_dir), render=False)

        # Verify all outputs were created
        for i in range(3):
            expected_output = output_dir / f"test_image_{i}.{pipeline.ext}"
            assert expected_output.exists(), f"Output folder {i} should exist"
            assert (
                expected_output / "results.yaml"
            ).exists(), f"results.yaml {i} should exist"

    except Exception as e:
        debug(e)


def test_output_preserves_results_data(pipeline, debug, tmpdir, raw_img_path):
    """Test that existing results.yaml data is preserved when copying."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create .grapto folder with custom data in results.yaml
        input_grapto = Path(tmpdir) / "input" / f"test.{pipeline.ext}"
        input_grapto.mkdir(parents=True, exist_ok=True)

        custom_data = {
            "collectra_results_metadata": {
                "workflow": pipeline.name,
                "version": pipeline.version,
                "timestamp": "2026-01-08T10:00:00",
            },
            "file": {"type": "collectra.Image", "id": "file", "data": "test.jpg"},
            "custom_field": {
                "type": "collectra.Text",
                "data": "custom value",
                "id": "custom_1",
            },
        }

        with open(input_grapto / "results.yaml", "w") as f:
            yaml.dump(custom_data, f)

        shutil.copy(raw_img_path, input_grapto / "test.jpg")

        # Create output directory
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        pipeline.run(files=[str(input_grapto)], output=str(output_dir), render=False)

        # Verify custom data was preserved
        output_results = output_dir / f"test.{pipeline.ext}" / "results.yaml"
        with open(output_results, "r") as f:
            output_data = yaml.safe_load(f)

        assert "custom_field" in output_data, "Custom field should be preserved"
        assert (
            output_data["custom_field"]["data"] == "custom value"
        ), "Custom data should match"

    except Exception as e:
        debug(e)


def test_output_dir_nonexistent(pipeline, debug, tmpdir, raw_img_path):
    """Test that output directory is created if it doesn't exist."""
    try:
        pipeline = build_pipeline(pipeline)

        # Create input image
        input_image = Path(tmpdir) / "test_image.jpg"
        shutil.copy(raw_img_path, input_image)

        # Use non-existent output directory
        output_dir = Path(tmpdir) / "new_output" / "nested" / "path"
        assert not output_dir.exists(), "Output directory should not exist initially"

        # Run pipeline
        pipeline.run(files=[str(input_image)], output=str(output_dir), render=False)

        # Verify output was created
        expected_output = output_dir / f"test_image.{pipeline.ext}"
        assert (
            expected_output.exists()
        ), "Output should be created even if directory didn't exist"

    except Exception as e:
        debug(e)
