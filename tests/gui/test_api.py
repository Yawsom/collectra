"""
Tests for collectra.gui.backend module.

Tests cover:
- GUIBackend class methods for graph operations
- File I/O operations (load_yaml, get_image_base64)
- CRUD operations (create, update, delete annotations)
- get_resource_path helper function
- Multi-folder workflows (select_parent_folder, load_collectra_folder)
- Label statistics (get_available_labels, get_label_statistics, get_global_label_statistics)
"""

import base64
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from collectra.commons.files import CollectraFile
from collectra.gui.backend import GUIBackend, get_resource_path


class TestApiInit:
    """Tests for GUIBackend initialization."""

    def test_init_sets_none_values(self, backend_with_extension):
        assert backend_with_extension._graph is None
        assert backend_with_extension._yaml_path is None
        assert backend_with_extension._window is None

    def test_set_window_stores_reference(self, backend_with_extension):
        mock_window = MagicMock()
        backend_with_extension.set_window(mock_window)
        assert backend_with_extension._window is mock_window

    def test_pipeline_graph_comes_from_live_pipeline(self):
        pipeline_path = (
            Path(__file__).parents[1] / "data" / "collectrapipeline" / "pipeline.yaml"
        )
        from collectra.pipelines.base import Collectra

        pipeline = Collectra.from_file(pipeline_path)
        backend = GUIBackend(pipeline)

        result = backend.get_pipeline_graph()

        assert backend._pipeline is pipeline
        assert result["success"] is True
        assert {node["id"] for node in result["nodes"]} == set(
            pipeline.node_manager.get_node_names()
        )
        assert {(edge["source"], edge["target"]) for edge in result["edges"]} == set(
            pipeline.node_manager.flow.edges
        )
        assert result["ext"] == pipeline.ext


class TestApiLoadYaml:
    """Tests for GUIBackend.load_yaml method."""

    def test_load_valid_yaml(self, backend_with_extension, temp_yaml_file):
        result = backend_with_extension.load_yaml(str(temp_yaml_file))

        assert result["success"] is True
        assert result["path"] == str(temp_yaml_file)
        assert backend_with_extension._graph is not None
        assert backend_with_extension._yaml_path == str(temp_yaml_file)
        assert isinstance(backend_with_extension._collectra_file, CollectraFile)
        assert (
            backend_with_extension._collectra_file.collectra_file_path
            == temp_yaml_file.parent
        )
        assert (
            backend_with_extension._graph.metadata
            is backend_with_extension._collectra_file.collectra_results_metadata
        )

    def test_load_nonexistent_file(self, backend_with_extension):
        result = backend_with_extension.load_yaml("/nonexistent/path.yaml")

        assert result["success"] is False
        assert "error" in result

    def test_load_invalid_yaml(self, backend_with_extension, tmp_path):
        # Create invalid YAML
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("{{invalid yaml content")

        result = backend_with_extension.load_yaml(str(invalid_file))

        assert result["success"] is False
        assert "error" in result


class TestApiGetDisplayValue:
    """Tests for GUIBackend.get_display_value method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_display_value("any_node")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_valid_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_display_value("text_001")

        assert result["success"] is True
        assert "value" in result
        assert "source_id" in result
        assert "crop_region" in result
        assert "reason" in result

    def test_nonexistent_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_display_value("nonexistent")

        assert result["success"] is True
        assert result["value"] is None
        assert "not found" in result["reason"]

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        # The DAG only ever knows the label ("text_label"), never the real
        # internal id ("text_001") — this must still find the node's real value.
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_display_value("text_label")

        assert result["success"] is True
        assert result["value"] == "Hello World"


class TestApiGetAllNodes:
    """Tests for GUIBackend.get_all_nodes method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_all_nodes()

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_returns_all_node_ids(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_all_nodes()

        assert result["success"] is True
        assert "nodes" in result
        assert "img_001" in result["nodes"]
        assert "crop_001" in result["nodes"]
        assert "text_001" in result["nodes"]


class TestApiGetNodeInfo:
    """Tests for GUIBackend.get_node_info method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_node_info("any_node")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_valid_node_info(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_node_info("crop_001")

        assert result["success"] is True
        assert result["id"] == "crop_001"
        assert "ImageCrop" in result["type"]
        assert "data" in result
        assert "children" in result
        assert "parents" in result

    def test_nonexistent_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_node_info("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_node_info("crop_label")

        assert result["success"] is True
        assert result["id"] == "crop_001"
        assert "ImageCrop" in result["type"]


class TestApiGetNodesByType:
    """Tests for GUIBackend.get_nodes_by_type method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_nodes_by_type("ImageCrop")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_filter_by_type(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_nodes_by_type("ImageCrop")
        assert result["success"] is True
        assert "crop_001" in result["nodes"]
        assert result["count"] == 1

        result = backend_with_extension.get_nodes_by_type("Text")
        assert "text_001" in result["nodes"]
        assert result["count"] == 1

    def test_filter_no_matches(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_nodes_by_type("NonExistent")

        assert result["success"] is True
        assert result["nodes"] == []
        assert result["count"] == 0


class TestApiGetAllNodesForGrid:
    """Tests for GUIBackend.get_all_nodes_for_grid method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_all_nodes_for_grid()

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_returns_grid_formatted_rows(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        result = backend_with_extension.get_all_nodes_for_grid()

        assert result["success"] is True
        assert "rows" in result
        assert len(result["rows"]) == 3

        # Check row structure
        row_ids = {row["id"] for row in result["rows"]}
        assert row_ids == {"img_001", "crop_001", "text_001"}

        # Check row has required fields
        for row in result["rows"]:
            assert "id" in row
            assert "type" in row
            assert "data" in row
            assert "displayValue" in row
            assert "crop_region" in row
            assert "displaySourceId" in row
            assert "reason" in row
            assert "parents" in row
            assert "children" in row
            assert "locked" in row
            assert "name" in row
            assert "label" in row

    def test_row_label_matches_node_label(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_all_nodes_for_grid()

        row = next(r for r in result["rows"] if r["id"] == "crop_001")
        assert row["label"] == "crop_label"

    def test_row_reflects_set_name(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        backend_with_extension.rename_annotation("crop_001", "My Sandglass")

        result = backend_with_extension.get_all_nodes_for_grid()

        row = next(r for r in result["rows"] if r["id"] == "crop_001")
        assert row["name"] == "My Sandglass"


class TestApiUpdateNodeData:
    """Tests for GUIBackend.update_node_data method."""

    def test_updates_node_data(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.update_node_data(
            "text_001", "Updated text content"
        )

        assert backend_with_extension._graph is not None

        assert result["success"] is True
        # Verify the update in graph
        assert (
            backend_with_extension._graph.get_data("text_001") == "Updated text content"
        )

    def test_update_nonexistent_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.update_node_data("nonexistent", "value")

        assert result["success"] is False
        assert "error" in result

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.update_node_data(
            "text_label", "Edited via label"
        )

        assert result["success"] is not False
        assert backend_with_extension._graph.get_data("text_001") == "Edited via label"


class TestApiUpdateNodeCoordinates:
    """Tests for GUIBackend.update_node_coordinates method."""

    def test_updates_crop_region(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        assert backend_with_extension._graph is not None

        new_region = {
            "x_center": 0.8,
            "y_center": 0.9,
            "width_relative": 0.3,
            "height_relative": 0.4,
        }
        result = backend_with_extension.update_node_coordinates("crop_001", new_region)

        assert result["success"] is True
        # Verify the update
        stored_region = backend_with_extension._graph.get_crop_region("crop_001")
        assert stored_region == new_region

    def test_update_with_missing_fields(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        incomplete_region = {"x_center": 0.5}  # Missing other fields
        result = backend_with_extension.update_node_coordinates(
            "crop_001", incomplete_region
        )

        assert result["success"] is False
        assert "error" in result

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        new_region = {
            "x_center": 0.1,
            "y_center": 0.2,
            "width_relative": 0.3,
            "height_relative": 0.4,
        }

        result = backend_with_extension.update_node_coordinates(
            "crop_label", new_region
        )

        assert result["success"] is True
        assert backend_with_extension._graph.get_crop_region("crop_001") == new_region


class TestApiCreateAnnotation:
    """Tests for GUIBackend.create_annotation method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.create_annotation(
            {"x_center": 0.5}, "user_crop", "", "/path/to/image.png"
        )

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_creates_annotation_imagecrop_only(
        self, backend_with_extension, temp_folder_with_files
    ):
        """create_annotation() creates only ImageCrop, no Text child."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        assert backend_with_extension._graph is not None

        initial_count = len(backend_with_extension._graph.nodes)

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert result["success"] is True
        # Should have added 1 node (ImageCrop only, no Text child)
        assert len(backend_with_extension._graph.nodes) == initial_count + 1

    def test_created_annotation_has_correct_parent(
        self, backend_with_extension, temp_folder_with_files
    ):
        """Created ImageCrop has correct parent relationship."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        assert backend_with_extension._graph is not None

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        # Find the newly created crop (starts with user_crop-)
        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_crop")
        ]
        assert len(new_crops) == 1

        new_crop = new_crops[0]
        parents = backend_with_extension._graph.parents(new_crop)
        assert "img_001" in parents

        # Verify no text node was created
        new_texts = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_text_")
        ]
        assert len(new_texts) == 0

    def test_first_annotation_of_a_label_is_numbered_1(
        self, backend_with_extension, temp_folder_with_files
    ):
        """Matches the pipeline's own id scheme (e.g. sandglass1), not a uuid."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert result["success"] is True
        assert "user_crop1" in backend_with_extension._graph.nodes

    def test_second_annotation_of_same_label_is_numbered_2(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert "user_crop1" in backend_with_extension._graph.nodes
        assert "user_crop2" in backend_with_extension._graph.nodes

    def test_numbering_reuses_the_smallest_id_a_delete_freed_up(
        self, backend_with_extension, temp_folder_with_files
    ):
        """Deleting user_crop1 means the next one created becomes user_crop1
        again — gap-filling, not always growing past the highest used."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        backend_with_extension.delete_annotation("user_crop1")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert "user_crop1" in backend_with_extension._graph.nodes
        assert "user_crop2" in backend_with_extension._graph.nodes
        assert "user_crop3" not in backend_with_extension._graph.nodes

    def test_numbering_fills_a_middle_gap_not_just_the_last_deleted(
        self, backend_with_extension, temp_folder_with_files
    ):
        """1,2,3,4 with 3 deleted -> next is 3, not 5 (smallest free number,
        not "one past the highest still in use")."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        for _ in range(4):
            backend_with_extension.create_annotation(
                crop_region, "user_crop", "", image_path
            )
        backend_with_extension.delete_annotation("user_crop3")

        preview = backend_with_extension.preview_annotation_id("user_crop")

        assert preview["id"] == "user_crop3"

        result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        assert result.get("success") is True, result.get("error")
        assert "user_crop3" in backend_with_extension._graph.nodes

    def test_preview_matches_the_id_actually_created(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        preview = backend_with_extension.preview_annotation_id("user_crop")
        assert preview["success"] is True
        assert preview["id"] == "user_crop1"

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert "user_crop1" in backend_with_extension._graph.nodes

        preview_2 = backend_with_extension.preview_annotation_id("user_crop")
        assert preview_2["id"] == "user_crop2"

    def test_preview_does_not_create_a_node(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        initial_count = len(backend_with_extension._graph.nodes)

        backend_with_extension.preview_annotation_id("user_crop")

        assert len(backend_with_extension._graph.nodes) == initial_count

    def test_preview_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.preview_annotation_id("user_crop")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_create_with_name_sets_name(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path, name="My Box"
        )

        assert backend_with_extension._graph.get_node("user_crop1").name == "My Box"

    def test_create_without_name_leaves_it_unset(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        assert backend_with_extension._graph.get_node("user_crop1").name == ""

    def test_explicit_crop_id_is_used_verbatim(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path, crop_id="  hand_picked_id  "
        )

        assert result["success"] is True
        assert "hand_picked_id" in backend_with_extension._graph.nodes
        assert "user_crop1" not in backend_with_extension._graph.nodes

    def test_blank_crop_id_falls_back_to_auto_numbering(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path, crop_id="   "
        )

        assert "user_crop1" in backend_with_extension._graph.nodes

    def test_duplicate_crop_id_is_rejected(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path, crop_id="taken"
        )
        count = len(backend_with_extension._graph.nodes)

        result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path, crop_id="taken"
        )

        assert result["success"] is False
        assert "taken" in result["error"]
        assert len(backend_with_extension._graph.nodes) == count


class TestApiDeleteAnnotation:
    """Tests for GUIBackend.delete_annotation method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.delete_annotation("any_node")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_deletes_crop_and_text_children(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        assert backend_with_extension._graph is not None
        # Delete crop_001 which has text_001 as child
        result = backend_with_extension.delete_annotation("crop_001")

        assert result["success"] is True
        assert "crop_001" not in backend_with_extension._graph.nodes
        assert "text_001" not in backend_with_extension._graph.nodes

    def test_deletes_nested_child_crops_too(
        self, backend_with_extension, temp_folder_with_files
    ):
        """A crop containing its own child ImageCrop (a sub-crop) cascades:
        deleting the parent must not leave the child behind pointing at a
        now-nonexistent parent id."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))

        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "parent_crop", "", image_path, "", "parent_crop1"
        )
        backend_with_extension.create_annotation(
            crop_region, "child_crop", "parent_crop1", image_path, "", "child_crop1"
        )
        backend_with_extension.create_annotation(
            crop_region,
            "grandchild_crop",
            "child_crop1",
            image_path,
            "",
            "grandchild_crop1",
        )

        result = backend_with_extension.delete_annotation("parent_crop1")

        assert result["success"] is True
        assert "parent_crop1" not in backend_with_extension._graph.nodes
        assert "child_crop1" not in backend_with_extension._graph.nodes
        assert "grandchild_crop1" not in backend_with_extension._graph.nodes

        # The freed id is safe to reuse afterwards — no dangling reference
        # left behind blocking it.
        preview = backend_with_extension.preview_annotation_id("parent_crop")
        assert preview["id"] == "parent_crop1"
        create_result = backend_with_extension.create_annotation(
            crop_region, "parent_crop", "", image_path
        )
        assert create_result.get("success") is True, create_result.get("error")
        assert "parent_crop1" in backend_with_extension._graph.nodes

    def test_delete_nonexistent_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.delete_annotation("nonexistent")

        assert result["success"] is False
        assert "error" in result

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.delete_annotation("crop_label")

        assert result["success"] is True
        assert "crop_001" not in backend_with_extension._graph.nodes


class TestApiDeletePreview:
    """Tests for GUIBackend.get_delete_preview method (read-only)."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_delete_preview("any_node")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_unknown_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_delete_preview("nonexistent")

        assert result["success"] is False
        assert "error" in result

    def test_lists_children_without_deleting(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_delete_preview("crop_001")

        assert result["success"] is True
        assert result["id"] == "crop_001"
        assert [c["id"] for c in result["children"]] == ["text_001"]
        child = result["children"][0]
        assert {"id", "label", "type"} <= set(child)
        # Read-only: nothing was removed.
        assert "crop_001" in backend_with_extension._graph.nodes
        assert "text_001" in backend_with_extension._graph.nodes

    def test_leaf_has_no_children(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_delete_preview("text_001")

        assert result["success"] is True
        assert result["children"] == []

    def test_nested_children_match_what_delete_removes(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        crop_region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "parent_crop", "", image_path, "", "parent_crop1"
        )
        backend_with_extension.create_annotation(
            crop_region, "child_crop", "parent_crop1", image_path, "", "child_crop1"
        )
        backend_with_extension.create_annotation(
            crop_region,
            "grandchild_crop",
            "child_crop1",
            image_path,
            "",
            "grandchild_crop1",
        )

        preview = backend_with_extension.get_delete_preview("parent_crop1")
        deleted = backend_with_extension.delete_annotation("parent_crop1")

        previewed_ids = [c["id"] for c in preview["children"]]
        assert previewed_ids == ["child_crop1", "grandchild_crop1"]
        # The preview lists exactly what the real deletion removes, minus the
        # requested node itself.
        assert sorted(deleted["deleted_ids"]) == sorted(
            ["parent_crop1", *previewed_ids]
        )

    def test_children_carry_their_depth_and_sit_under_their_parent(
        self, backend_with_extension, temp_folder_with_files
    ):
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        region = {
            "x_center": 0.6,
            "y_center": 0.7,
            "width_relative": 0.1,
            "height_relative": 0.05,
        }
        image = str(temp_folder_with_files / "image.png")
        make = lambda label, parent, crop_id: backend_with_extension.create_annotation(
            region, label, parent, image, "", crop_id
        )
        make("parent_crop", "", "p1")
        make("child_crop", "p1", "c1")
        make("child_crop", "p1", "c2")
        make("grandchild_crop", "c1", "g1")
        make("grandchild_crop", "c2", "g2")

        preview = backend_with_extension.get_delete_preview("p1")

        # Each branch is listed together, directly under its own parent.
        assert [(c["id"], c["depth"]) for c in preview["children"]] == [
            ("c1", 1),
            ("g1", 2),
            ("c2", 1),
            ("g2", 2),
        ]

    def test_root_is_flagged_so_its_own_node_can_be_kept(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        root = backend_with_extension.get_delete_preview("img_001")
        crop = backend_with_extension.get_delete_preview("crop_001")

        assert root["keeps_self"] is True
        assert crop["keeps_self"] is False


class TestApiDeleteNested:
    """delete_nested removes everything under a node but keeps the node."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.delete_nested("any_node")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_keeps_the_node_and_removes_all_below_it(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))
        graph = backend_with_extension._graph

        result = backend_with_extension.delete_nested("img_001")

        assert result["success"] is True
        assert "img_001" in graph.nodes
        assert "crop_001" not in graph.nodes
        assert "text_001" not in graph.nodes
        assert "crop_001" in result["deleted_ids"]
        assert "img_001" not in result["deleted_ids"]

    def test_unknown_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.delete_nested("nonexistent")

        assert result["success"] is False


class TestApiRenameAnnotation:
    """Tests for GUIBackend.rename_annotation method.

    Always sets the node's name. label and new_id are optional — omitted,
    the node's label/id are left as-is (the default rename_annotation calls
    below only pass name); provided, they're changed too.
    """

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.rename_annotation("any_node", "My Box")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_renames_sets_name(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation("crop_001", "My Box")

        assert result["success"] is True
        node = backend_with_extension._graph.get_node("crop_001")
        assert node.name == "My Box"

    def test_rename_does_not_change_label_or_id(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        backend_with_extension.rename_annotation("crop_001", "My Box")

        assert "crop_001" in backend_with_extension._graph.nodes
        node = backend_with_extension._graph.get_node("crop_001")
        assert node.label == "crop_label"

    def test_rename_nonexistent_node(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation("nonexistent", "x")

        assert result["success"] is False
        assert "error" in result

    def test_resolves_dag_label_to_real_node(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation("crop_label", "My Box")

        assert result["success"] is True
        node = backend_with_extension._graph.get_node("crop_001")
        assert node.name == "My Box"

    def test_rename_updates_label_when_provided(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation(
            "crop_001", "My Box", label="new_label"
        )

        assert result["success"] is True
        node = backend_with_extension._graph.get_node("crop_001")
        assert node.label == "new_label"
        assert node.name == "My Box"

    def test_rename_updates_id_when_new_id_provided(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation(
            "crop_001", "My Box", new_id="crop_999"
        )

        assert result["success"] is True
        assert "crop_001" not in backend_with_extension._graph.nodes
        node = backend_with_extension._graph.get_node("crop_999")
        assert node.id == "crop_999"
        assert node.name == "My Box"

    def test_rename_id_collision_returns_error(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation(
            "crop_001", "My Box", new_id="text_001"
        )

        assert result["success"] is False
        assert "error" in result
        # Nothing should have been changed by the failed attempt
        assert "crop_001" in backend_with_extension._graph.nodes

    def test_rename_updates_label_and_id_together(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.rename_annotation(
            "crop_001", "My Box", label="new_label", new_id="crop_999"
        )

        assert result["success"] is True
        node = backend_with_extension._graph.get_node("crop_999")
        assert node.label == "new_label"
        assert node.name == "My Box"
        assert node.id == "crop_999"


class TestApiGetImageBase64:
    """Tests for GUIBackend.get_image_base64 method."""

    def test_reads_png_file(self, backend_with_extension, temp_image_file):
        result = backend_with_extension.get_image_base64(str(temp_image_file))

        assert result["success"] is True
        assert result["data"].startswith("data:image/png;base64,")

    def test_nonexistent_file(self, backend_with_extension):
        result = backend_with_extension.get_image_base64("/nonexistent/image.png")

        assert result["success"] is False
        assert "error" in result

    def test_different_mime_types(self, backend_with_extension, tmp_path):
        # Test JPEG
        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"fake jpg data")
        result = backend_with_extension.get_image_base64(str(jpg_file))
        assert "image/jpg" in result["data"]

        # Test GIF
        gif_file = tmp_path / "test.gif"
        gif_file.write_bytes(b"fake gif data")
        result = backend_with_extension.get_image_base64(str(gif_file))
        assert "image/gif" in result["data"]

        # Test TIFF
        tif_file = tmp_path / "test.tif"
        tif_file.write_bytes(b"fake tif data")
        result = backend_with_extension.get_image_base64(str(tif_file))
        assert "image/tiff" in result["data"]

    def test_unknown_extension_defaults_to_png(self, backend_with_extension, tmp_path):
        unknown_file = tmp_path / "test.xyz"
        unknown_file.write_bytes(b"fake data")
        result = backend_with_extension.get_image_base64(str(unknown_file))
        # Unknown extensions raise KeyError and return error
        assert result["success"] is False
        assert "error" in result


class TestApiRevealInFolder:
    """Tests for GUIBackend.reveal_in_folder method."""

    def test_returns_error_for_nonexistent_path(self, backend_with_extension):
        with patch("subprocess.run") as mock_run:
            result = backend_with_extension.reveal_in_folder("/nonexistent/folder")

        assert result["success"] is False
        assert "error" in result
        mock_run.assert_not_called()

    def test_calls_open_dash_r_on_macos(self, backend_with_extension, tmp_path):
        """`open -R` reveals + selects the results.yaml file within its
        specimen folder — the actual "Reveal in Finder" behavior — rather
        than opening the specimen folder's own contents."""
        yaml_path = tmp_path / "results.yaml"
        yaml_path.touch()
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.reveal_in_folder(str(yaml_path))

        assert result["success"] is True
        mock_run.assert_called_once_with(["open", "-R", str(yaml_path)])

    def test_calls_explorer_select_on_windows(self, backend_with_extension, tmp_path):
        """Explorer's equivalent of reveal+select is a single combined
        `/select,<path>` argument, not two separate ones."""
        yaml_path = tmp_path / "results.yaml"
        yaml_path.touch()
        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.reveal_in_folder(str(yaml_path))

        assert result["success"] is True
        mock_run.assert_called_once_with(["explorer", f"/select,{yaml_path}"])

    def test_calls_xdg_open_on_parent_on_linux(self, backend_with_extension, tmp_path):
        """No universal reveal+select convention across Linux desktop
        environments — opening the specimen folder (the file's parent) is
        the closest approximation available via a single xdg-open call."""
        yaml_path = tmp_path / "results.yaml"
        yaml_path.touch()
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.reveal_in_folder(str(yaml_path))

        assert result["success"] is True
        mock_run.assert_called_once_with(["xdg-open", str(yaml_path.parent)])


class TestApiCopyToClipboard:
    """Tests for GUIBackend.copy_to_clipboard (used by the 'Copy Path' menu item)."""

    def test_pipes_the_text_to_pbcopy_on_macos(self, backend_with_extension):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.copy_to_clipboard("/data/Page022.livy")

        assert result == {"success": True}
        mock_run.assert_called_once_with(
            ["pbcopy"], input="/data/Page022.livy", text=True, check=True
        )

    def test_pipes_the_text_to_clip_on_windows(self, backend_with_extension):
        with (
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.copy_to_clipboard("C:\\data\\Page022.livy")

        assert result["success"] is True
        mock_run.assert_called_once_with(
            ["clip"], input="C:\\data\\Page022.livy", text=True, check=True
        )

    def test_uses_xclip_on_linux(self, backend_with_extension):
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run") as mock_run,
        ):
            result = backend_with_extension.copy_to_clipboard("/data/Page022.livy")

        assert result["success"] is True
        mock_run.assert_called_once_with(
            ["xclip", "-selection", "clipboard"],
            input="/data/Page022.livy",
            text=True,
            check=True,
        )

    def test_falls_back_to_wl_copy_on_linux_without_xclip(self, backend_with_extension):
        def run(command, **kwargs):
            if command[0] == "xclip":
                raise FileNotFoundError(command[0])

        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=run) as mock_run,
        ):
            result = backend_with_extension.copy_to_clipboard("/data/x")

        assert result["success"] is True
        assert [call.args[0][0] for call in mock_run.call_args_list] == [
            "xclip",
            "wl-copy",
        ]

    def test_reports_a_missing_clipboard_tool(self, backend_with_extension):
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=FileNotFoundError("none")),
        ):
            result = backend_with_extension.copy_to_clipboard("/data/x")

        assert result["success"] is False
        assert "clipboard" in result["error"].lower()

    def test_reports_a_failing_clipboard_command(self, backend_with_extension):
        import subprocess

        with (
            patch("platform.system", return_value="Darwin"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "pbcopy"),
            ),
        ):
            result = backend_with_extension.copy_to_clipboard("/data/x")

        assert result["success"] is False
        assert result["error"]


class TestApiSelectFolder:
    """Tests for GUIBackend.select_folder method."""

    def test_no_window_initialized(self, backend_with_extension):
        result = backend_with_extension.select_folder()

        assert result["success"] is False
        assert "Window not initialized" in result["error"]

    def test_dialog_cancelled(self, backend_with_extension):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = None
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_folder()

        assert result["success"] is False
        assert "No folder selected" in result["error"]

    def test_dialog_returns_empty_list(self, backend_with_extension):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = []
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_folder()

        assert result["success"] is False

    def test_finds_results_and_image_files(
        self, backend_with_extension, temp_folder_with_files
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [str(temp_folder_with_files)]
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_folder()

        assert result["success"] is True
        assert result["folder_path"] == str(temp_folder_with_files)
        assert result["yaml_path"].endswith("results.yaml")
        assert result["image_path"] is not None
        assert result["image_path"].endswith(".png")

    def test_rejects_non_results_yaml(self, backend_with_extension, tmp_path):
        (tmp_path / "other.yaml").write_text("data: value\n")
        (tmp_path / "image.png").write_bytes(b"PNG")
        window = MagicMock()
        window.create_file_dialog.return_value = [str(tmp_path)]
        backend_with_extension.set_window(window)

        result = backend_with_extension.select_folder()

        assert result["success"] is False
        assert "results.yaml" in result["error"]

    def test_handles_dialog_exception(self, backend_with_extension):
        mock_window = MagicMock()
        mock_window.create_file_dialog.side_effect = Exception("Dialog error")
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_folder()

        assert result["success"] is False
        assert "Dialog error" in result["error"]


class TestGetResourcePath:
    """Tests for get_resource_path helper function."""

    def test_development_mode_path(self):
        # In normal (non-frozen) mode, should use __file__ parent
        result = get_resource_path("test.html")
        assert result.endswith("test.html")
        assert os.path.normpath("collectra/gui") in result

    def test_frozen_mode_path(self):
        """Test PyInstaller frozen mode by patching sys module directly."""
        import sys

        # Store original values
        original_frozen = getattr(sys, "frozen", None)
        original_meipass = getattr(sys, "_MEIPASS", None)

        try:
            # Simulate PyInstaller frozen mode
            sys.frozen = True
            sys._MEIPASS = "/tmp/pyinstaller_bundle"

            result = get_resource_path("test.html")

            assert result == "/tmp/pyinstaller_bundle/test.html"
        finally:
            # Restore original state
            if original_frozen is None:
                if hasattr(sys, "frozen"):
                    delattr(sys, "frozen")
            else:
                sys.frozen = original_frozen
            if original_meipass is None:
                if hasattr(sys, "_MEIPASS"):
                    delattr(sys, "_MEIPASS")
            else:
                sys._MEIPASS = original_meipass


class TestApiSaveCollectraFile:
    """Tests for GUIBackend._save_collectra_file."""

    def test_save_preserves_data(self, backend_with_extension, temp_yaml_file):
        import yaml

        backend_with_extension.load_yaml(str(temp_yaml_file))

        # Modify data
        backend_with_extension._graph.set_data("text_001", "Modified text")
        backend_with_extension._save_collectra_file()

        # Reload and verify
        with open(temp_yaml_file, "r") as f:
            saved_data = yaml.safe_load(f)

        # Find the text node in saved data
        text_items = saved_data.get("text_label", [])
        if not isinstance(text_items, list):
            text_items = [text_items]
        text_node = next((t for t in text_items if t.get("id") == "text_001"), None)

        assert text_node is not None
        assert text_node["data"] == "Modified text"

    def test_save_uses_shared_collectra_writer(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        with patch.object(CollectraFile, "save", autospec=True) as save:
            backend_with_extension._save_collectra_file()

        save.assert_called_once_with(backend_with_extension._collectra_file)

    def test_save_raises_when_no_yaml_loaded(self, backend_with_extension):
        backend_with_extension._graph = MagicMock()  # Set graph but no yaml_path

        with pytest.raises(ValueError, match="No Collectra file loaded"):
            backend_with_extension._save_collectra_file()

    def test_save_raises_when_no_graph(self, backend_with_extension, temp_yaml_file):
        backend_with_extension._collectra_file = MagicMock()
        # No graph set

        with pytest.raises(ValueError, match="No Collectra file loaded"):
            backend_with_extension._save_collectra_file()


class TestApiIntegration:
    """Integration tests for GUIBackend class workflows."""

    def test_full_crud_workflow(self, backend_with_extension, temp_folder_with_files):
        """Test create, read, update, delete workflow."""
        # Load
        load_result = backend_with_extension.load_yaml(
            str(temp_folder_with_files / "results.yaml")
        )
        assert load_result["success"] is True

        assert backend_with_extension._graph is not None

        initial_count = len(backend_with_extension._graph.nodes)

        # Create (now only creates ImageCrop, no Text child)
        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_folder_with_files / "image.png")
        create_result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        assert create_result["success"] is True
        assert len(backend_with_extension._graph.nodes) == initial_count + 1

        # Find new crop id
        new_crop_id = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_crop")
        ][0]

        # Create text node by calling update_node_data with empty node_id and crop_id
        create_text_result = backend_with_extension.update_node_data(
            "", "New annotation text", crop_id=new_crop_id
        )
        assert create_text_result["success"] is True
        assert len(backend_with_extension._graph.nodes) == initial_count + 2

        # Find new text id
        new_text_id = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_text_")
        ][0]

        # Update text
        update_result = backend_with_extension.update_node_data(
            new_text_id, "Updated annotation text"
        )
        assert update_result["success"] is True

        # Read to verify
        node_info = backend_with_extension.get_node_info(new_text_id)
        assert node_info["success"] is True
        assert (
            backend_with_extension._graph.get_data(new_text_id)
            == "Updated annotation text"
        )

        # Delete
        delete_result = backend_with_extension.delete_annotation(new_crop_id)
        assert delete_result["success"] is True
        assert new_crop_id not in backend_with_extension._graph.nodes
        assert new_text_id not in backend_with_extension._graph.nodes


class TestApiCreateAnnotationBehavior:
    """Tests for create_annotation new behavior (ImageCrop only)."""

    def test_create_annotation_no_text_child_created(
        self, backend_with_extension, temp_folder_with_files
    ):
        """create_annotation() creates only ImageCrop, no Text node."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        assert backend_with_extension._graph is not None
        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_folder_with_files / "image.png")
        backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )

        # Should only find crop, not text
        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_crop")
        ]
        new_texts = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_text_")
        ]
        assert len(new_crops) == 1
        assert len(new_texts) == 0


class TestApiGridLockedField:
    """Tests for locked field in grid data."""

    def test_get_all_nodes_for_grid_includes_locked_field(
        self, backend_with_extension, temp_yaml_file
    ):
        """Grid rows include 'locked' field from NodeDisplayValue."""
        backend_with_extension.load_yaml(str(temp_yaml_file))
        assert backend_with_extension._graph is not None
        result = backend_with_extension.get_all_nodes_for_grid()

        for row in result["rows"]:
            assert "locked" in row
            assert isinstance(row["locked"], bool)

    def test_grid_locked_field_correct_for_node_types(
        self, backend_with_extension, temp_yaml_file
    ):
        """Image nodes are locked, others are not."""
        backend_with_extension.load_yaml(str(temp_yaml_file))
        assert backend_with_extension._graph is not None
        result = backend_with_extension.get_all_nodes_for_grid()

        for row in result["rows"]:
            if "Image" in row["type"] and "ImageCrop" not in row["type"]:
                assert row["locked"] is True
            # Leaf crops and text can be edited


class TestApiUpdateNodeDataWithCropId:
    """Tests for update_node_data with crop_id parameter."""

    def test_update_node_data_with_crop_id_creates_text(
        self, backend_with_extension, temp_yaml_file
    ):
        """update_node_data() with crop_id creates new Text node."""
        backend_with_extension.load_yaml(str(temp_yaml_file))
        assert backend_with_extension._graph is not None
        initial_count = len(backend_with_extension._graph.nodes)

        result = backend_with_extension.update_node_data(
            "", "New text", crop_id="crop_001"
        )

        assert result["success"] is True
        assert len(backend_with_extension._graph.nodes) == initial_count + 1
        new_texts = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_text_")
        ]
        assert len(new_texts) == 1

    def test_update_node_data_node_id_takes_precedence(
        self, backend_with_extension, temp_yaml_file
    ):
        """When both node_id and crop_id provided, updates existing node."""
        backend_with_extension.load_yaml(str(temp_yaml_file))
        assert backend_with_extension._graph is not None
        initial_count = len(backend_with_extension._graph.nodes)

        result = backend_with_extension.update_node_data(
            "text_001", "Updated", crop_id="crop_001"
        )

        assert result["success"] is True
        assert len(backend_with_extension._graph.nodes) == initial_count  # No new nodes
        assert backend_with_extension._graph.get_data("text_001") == "Updated"


class TestApiIntegrationCreateCropThenAddText:
    """Integration test for creating crop then adding text."""

    def test_integration_create_crop_then_add_text(
        self, backend_with_extension, temp_folder_with_files
    ):
        """Workflow: create ImageCrop, then add Text via update_node_data."""
        backend_with_extension.load_yaml(str(temp_folder_with_files / "results.yaml"))
        assert backend_with_extension._graph is not None

        # Create crop
        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_folder_with_files / "image.png")
        create_result = backend_with_extension.create_annotation(
            crop_region, "user_crop", "", image_path
        )
        assert create_result["success"] is True

        # Get new crop ID
        new_crop_id = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_crop")
        ][0]

        # Add text to crop
        update_result = backend_with_extension.update_node_data(
            "", "Annotation text", crop_id=new_crop_id
        )
        assert update_result["success"] is True

        # Verify structure
        new_text_id = [
            n for n in backend_with_extension._graph.nodes if n.startswith("user_text_")
        ][0]
        assert new_text_id in backend_with_extension._graph.children(new_crop_id)
        assert backend_with_extension._graph.get_data(new_text_id) == "Annotation text"


class TestGUIBackendSelectParentFolder:
    """Tests for GUIBackend.select_parent_folder method."""

    def test_no_window_initialized(self, backend_with_extension):
        result = backend_with_extension.select_parent_folder()
        assert result["success"] is False
        assert "Window not initialized" in result["error"]

    def test_dialog_cancelled(self, backend_with_extension):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = None
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_parent_folder()
        assert result["success"] is False
        assert "No folder selected" in result["error"]

    def test_dialog_returns_empty_list(self, backend_with_extension):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = []
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_parent_folder()
        assert result["success"] is False

    def test_finds_collectra_subfolders(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_parent_folder()

        assert result["success"] is True
        assert result["parent_path"] == str(temp_parent_folder_with_subfolders)
        assert "folders" in result
        assert len(result["folders"]) == 3
        # Check folder structure
        for folder in result["folders"]:
            assert "name" in folder
            assert "index" in folder
            assert "path" in folder
            assert "yaml_path" in folder
            assert folder["name"].endswith(".collectra")

    def test_computes_global_label_statistics(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)

        result = backend_with_extension.select_parent_folder()

        assert result["success"] is True
        assert "global_label_counts" in result
        assert "global_total" in result


class TestSelectCollectraPathResetsStaleYamlPath:
    """Tests for GUIBackend.select_collectra_path clearing _yaml_path from a
    previously-viewed page in a DIFFERENT project.

    _collectra_dirs() searches near _yaml_path (in addition to _parent_folder
    and the pipeline path) so that opening a page's own results.yaml still finds
    its sibling .collectra folder. But if _yaml_path is never cleared when the
    user picks a brand new top-level project, that stale path keeps pointing
    _collectra_dirs() at the OLD project too — so get_logo()/get_theme_css()
    can silently return the old project's file instead of falling back to the
    new project's own (or Collectra's bundled default), because they return on
    the first match found across every directory in the list.
    """

    def test_switching_project_clears_stale_yaml_path(
        self, backend_with_extension, tmp_path
    ):
        # Project A: has a page whose results.yaml gets "opened" (setting
        # _yaml_path), and its own .collectra folder has a real logo.
        project_a = tmp_path / "ProjectA"
        project_a.mkdir()
        a_collectra = project_a / "A.collectra"
        a_collectra.mkdir()
        (a_collectra / "logo-a.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        )
        a_page = project_a / "page000.demo"
        a_page.mkdir()
        (a_page / "results.yaml").write_text(
            "field:\n  type: collectra.Text\n  id: field\n"
        )

        # Project B: a totally separate project with no logo of its own.
        project_b = tmp_path / "ProjectB"
        project_b.mkdir()
        b_page = project_b / "page000.collectra"
        b_page.mkdir()
        (b_page / "results.yaml").write_text(
            "field:\n  type: collectra.Text\n  id: field\n"
        )
        (b_page / "image.png").write_bytes(b"PNG")

        backend = backend_with_extension
        mock_window = MagicMock()
        backend.set_window(mock_window)

        # Simulate having viewed a page in Project A (sets _yaml_path).
        backend._parent_folder = str(project_a)
        backend.load_yaml(str(a_page / "results.yaml"))
        assert backend._yaml_path is not None

        # Now switch to Project B entirely via the normal folder-select flow.
        mock_window.create_file_dialog.return_value = [str(b_page)]
        result = backend.select_collectra_path()

        assert result["success"] is True
        assert backend._yaml_path is None
        assert a_collectra not in backend._collectra_dirs()


class TestGUIBackendLoadCollectraFolder:
    """Tests for GUIBackend.load_collectra_folder method."""

    def test_no_folders_loaded(self, backend_with_extension):
        result = backend_with_extension.load_collectra_folder(0)
        assert result["success"] is False
        assert "No folders loaded" in result["error"]

    def test_invalid_index_negative(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)
        backend_with_extension.select_parent_folder()

        result = backend_with_extension.load_collectra_folder(-1)
        assert result["success"] is False
        assert "Invalid folder index" in result["error"]

    def test_invalid_index_too_large(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)
        backend_with_extension.select_parent_folder()

        result = backend_with_extension.load_collectra_folder(999)
        assert result["success"] is False
        assert "Invalid folder index" in result["error"]

    def test_loads_folder_by_index(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)
        backend_with_extension.select_parent_folder()

        result = backend_with_extension.load_collectra_folder(0)

        assert result["success"] is True
        assert "folder_name" in result
        assert "folder_path" in result
        assert "yaml_path" in result
        assert "image_path" in result
        assert result["folder_name"].endswith(".collectra")


class TestGUIBackendGetAvailableLabels:
    """Tests for GUIBackend.get_available_labels method."""

    def test_returns_empty_labels_initially(self, backend_with_extension):
        result = backend_with_extension.get_available_labels()
        assert result["success"] is True
        assert result["labels"] == []

    def test_returns_labels_after_loading_yaml(
        self, backend_with_extension, temp_yaml_file
    ):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_available_labels()

        assert result["success"] is True
        assert "labels" in result
        assert isinstance(result["labels"], list)
        # sample_yaml_data has image_label, crop_label, text_label
        assert "image_label" in result["labels"]
        assert "crop_label" in result["labels"]
        assert "text_label" in result["labels"]

    def test_labels_are_sorted(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_available_labels()

        assert result["success"] is True
        labels = result["labels"]
        assert labels == sorted(labels)

    def test_accumulates_labels_across_files(
        self, backend_with_extension, tmp_path, sample_yaml_data, multi_label_yaml_data
    ):
        import yaml

        # Create two Collectra files with different labels
        file1 = tmp_path / "file1.collectra"
        file2 = tmp_path / "file2.collectra"
        file1.mkdir()
        file2.mkdir()
        yaml1 = file1 / "results.yaml"
        yaml2 = file2 / "results.yaml"

        with open(yaml1, "w") as f:
            yaml.dump(sample_yaml_data, f)
        with open(yaml2, "w") as f:
            yaml.dump(multi_label_yaml_data, f)

        backend_with_extension.load_yaml(str(yaml1))
        backend_with_extension.load_yaml(str(yaml2))

        result = backend_with_extension.get_available_labels()

        assert result["success"] is True
        # Should have labels from both files
        assert "image_label" in result["labels"]
        assert "header_crop" in result["labels"]
        assert "body_crop" in result["labels"]
        assert "footer_crop" in result["labels"]


class TestGUIBackendGetLabelStatistics:
    """Tests for GUIBackend.get_label_statistics method."""

    def test_no_graph_loaded(self, backend_with_extension):
        result = backend_with_extension.get_label_statistics()
        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_returns_label_counts_for_current_graph(
        self, backend_with_extension, tmp_path, multi_label_yaml_data
    ):
        import yaml

        collectra_file = tmp_path / "multi.collectra"
        collectra_file.mkdir()
        yaml_file = collectra_file / "results.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(multi_label_yaml_data, f)

        backend_with_extension.load_yaml(str(yaml_file))

        result = backend_with_extension.get_label_statistics()

        assert result["success"] is True
        assert "label_counts" in result
        assert "total" in result
        # multi_label_yaml_data has: 2 header_crop, 1 body_crop, 1 footer_crop
        assert result["label_counts"].get("header_crop", 0) == 2
        assert result["label_counts"].get("body_crop", 0) == 1
        assert result["label_counts"].get("footer_crop", 0) == 1
        assert result["total"] == 4

    def test_counts_only_imagecrop_types(self, backend_with_extension, temp_yaml_file):
        backend_with_extension.load_yaml(str(temp_yaml_file))

        result = backend_with_extension.get_label_statistics()

        assert result["success"] is True
        # sample_yaml_data has 1 ImageCrop (crop_label), should not count Image or Text
        assert "crop_label" in result["label_counts"]
        # Image and Text types should not be counted
        assert result["label_counts"].get("image_label", 0) == 0
        assert result["label_counts"].get("text_label", 0) == 0


class TestGUIBackendGetGlobalLabelStatistics:
    """Tests for GUIBackend.get_global_label_statistics method."""

    def test_returns_empty_counts_initially(self, backend_with_extension):
        result = backend_with_extension.get_global_label_statistics()
        assert result["success"] is True
        assert result["label_counts"] == {}
        assert result["total"] == 0

    def test_returns_aggregated_counts_after_parent_folder_scan(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)
        backend_with_extension.select_parent_folder()

        result = backend_with_extension.get_global_label_statistics()

        assert result["success"] is True
        assert "label_counts" in result
        assert "total" in result
        # Each subfolder has sample_yaml_data with 1 crop_label ImageCrop
        # 3 folders * 1 crop = 3 total
        assert result["label_counts"].get("crop_label", 0) == 3
        assert result["total"] == 3

    def test_updates_after_creating_annotation(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        mock_window = MagicMock()
        mock_window.create_file_dialog.return_value = [
            str(temp_parent_folder_with_subfolders)
        ]
        backend_with_extension.set_window(mock_window)
        backend_with_extension.select_parent_folder()

        # Load first folder
        folder_result = backend_with_extension.load_collectra_folder(0)
        backend_with_extension.load_yaml(folder_result["yaml_path"])

        initial_stats = backend_with_extension.get_global_label_statistics()
        initial_total = initial_stats["total"]

        # Create new annotation
        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        backend_with_extension.create_annotation(
            crop_region, "new_label", "", folder_result["image_path"]
        )

        updated_stats = backend_with_extension.get_global_label_statistics()

        assert updated_stats["label_counts"].get("new_label", 0) == 1
        assert updated_stats["total"] == initial_total + 1


class TestApiCreateAnnotationImageExtension:
    """Tests for create_annotation correctly handling image file extensions.

    Regression tests to prevent the bug where annotation data fields get
    workflow extensions (.grapto, .collectra) instead of image extensions
    (.png, .jpg, .gif, etc.).
    """

    def test_create_annotation_preserves_png_extension(
        self, backend_with_extension, temp_collectra_folder_with_png
    ):
        """When image is PNG, created annotation data field should end with .png."""
        yaml_path = temp_collectra_folder_with_png / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_png / "test_image.png")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        # Find the newly created annotation
        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        assert len(new_crops) == 1

        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        assert data_field.endswith(
            ".png"
        ), f"Annotation data field should end with .png, got: {data_field}"

    def test_create_annotation_preserves_tiff_extension(
        self, backend_with_extension, temp_collectra_folder_with_tiff
    ):
        """When image is TIFF, created annotation data field should end with .tiff."""
        yaml_path = temp_collectra_folder_with_tiff / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_tiff / "test_image.tiff")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        assert len(new_crops) == 1

        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        assert data_field.endswith(".tiff") or data_field.endswith(
            ".tif"
        ), f"Annotation data field should end with .tiff or .tif, got: {data_field}"

    def test_create_annotation_preserves_gif_extension(
        self, backend_with_extension, temp_collectra_folder_with_gif
    ):
        """When image is GIF, created annotation data field should end with .gif."""
        yaml_path = temp_collectra_folder_with_gif / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_gif / "test_image.gif")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        assert len(new_crops) == 1

        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        assert data_field.endswith(
            ".gif"
        ), f"Annotation data field should end with .gif, got: {data_field}"

    def test_create_annotation_does_not_use_workflow_extension_collectra(
        self, backend_with_extension, temp_collectra_folder_with_png
    ):
        """Created annotation data field should NOT contain .collectra extension."""
        yaml_path = temp_collectra_folder_with_png / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_png / "test_image.png")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        assert (
            ".collectra" not in data_field
        ), f"Annotation data field should not contain workflow extension .collectra, got: {data_field}"

    def test_create_annotation_does_not_use_workflow_extension_grapto(
        self, backend_with_grapto_extension, temp_grapto_folder_with_png
    ):
        """Created annotation data field should NOT contain .grapto extension."""
        yaml_path = temp_grapto_folder_with_png / "results.yaml"
        backend_with_grapto_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_grapto_folder_with_png / "test_image.png")
        backend_with_grapto_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n
            for n in backend_with_grapto_extension._graph.nodes
            if n.startswith("test_label")
        ]
        new_crop_id = new_crops[0]
        data_field = backend_with_grapto_extension._graph.get_data(new_crop_id)

        assert (
            ".grapto" not in data_field
        ), f"Annotation data field should not contain workflow extension .grapto, got: {data_field}"


class TestImageExtensionValidation:
    """Tests to validate image extensions are correctly handled throughout the system."""

    def test_image_format_enum_covers_common_extensions(self):
        """Verify ImageFormat enum includes all expected formats."""
        from collectra.gui.backend import ImageFormat

        expected = ["PNG", "JPEG", "JPG", "GIF", "BMP", "WEBP", "TIFF", "TIF"]
        for ext in expected:
            assert (
                ext in ImageFormat.__members__
            ), f"ImageFormat enum should include {ext}"

    def test_scan_collectra_folder_detects_png(
        self, backend_with_extension, temp_collectra_folder_with_png
    ):
        """_scan_collectra_folder should correctly identify PNG image."""
        from pathlib import Path

        result = backend_with_extension._scan_collectra_folder(
            Path(temp_collectra_folder_with_png)
        )

        assert result is not None
        assert result["image_path"].endswith(".png")

    def test_scan_collectra_folder_detects_tiff(
        self, backend_with_extension, temp_collectra_folder_with_tiff
    ):
        """_scan_collectra_folder should correctly identify TIFF image."""
        from pathlib import Path

        result = backend_with_extension._scan_collectra_folder(
            Path(temp_collectra_folder_with_tiff)
        )

        assert result is not None
        assert result["image_path"].endswith(".tiff")

    def test_scan_collectra_folder_detects_gif(
        self, backend_with_extension, temp_collectra_folder_with_gif
    ):
        """_scan_collectra_folder should correctly identify GIF image."""
        from pathlib import Path

        result = backend_with_extension._scan_collectra_folder(
            Path(temp_collectra_folder_with_gif)
        )

        assert result is not None
        assert result["image_path"].endswith(".gif")

    def test_data_field_has_valid_image_extension(
        self, backend_with_extension, temp_collectra_folder_with_png
    ):
        """After creating annotation, verify data field extension is a valid image extension."""
        from pathlib import Path

        from collectra.gui.backend import ImageFormat

        yaml_path = temp_collectra_folder_with_png / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_png / "test_image.png")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        # Extract extension and validate it's a known image format
        extension = Path(data_field).suffix[1:].upper()  # Remove leading dot
        valid_extensions = list(ImageFormat.__members__.keys())

        assert extension in valid_extensions, (
            f"Data field extension '{extension}' should be a valid image format. "
            f"Valid formats: {valid_extensions}. Got data field: {data_field}"
        )

    def test_annotation_data_does_not_contain_double_extension(
        self, backend_with_extension, temp_collectra_folder_with_png
    ):
        """Annotation data field should not have double extensions like .collectra.jpg."""
        yaml_path = temp_collectra_folder_with_png / "results.yaml"
        backend_with_extension.load_yaml(str(yaml_path))

        crop_region = {
            "x_center": 0.5,
            "y_center": 0.5,
            "width_relative": 0.1,
            "height_relative": 0.1,
        }
        image_path = str(temp_collectra_folder_with_png / "test_image.png")
        backend_with_extension.create_annotation(
            crop_region, "test_label", "", image_path
        )

        new_crops = [
            n for n in backend_with_extension._graph.nodes if n.startswith("test_label")
        ]
        new_crop_id = new_crops[0]
        data_field = backend_with_extension._graph.get_data(new_crop_id)

        # Count number of dots in filename (should be exactly 1 for single extension)
        dot_count = data_field.count(".")
        assert (
            dot_count == 1
        ), f"Data field should have single extension, but found {dot_count} dots: {data_field}"


class TestPipelineMetadata:
    """Tests for metadata supplied by the live pipeline."""

    def test_returns_metadata_block(
        self, backend_with_extension, temp_project_with_collectra
    ):
        backend = backend_with_extension
        backend._parent_folder = str(temp_project_with_collectra)
        backend._pipeline.pipeline_metadata.update(
            logo="BrandMark.png", theme="Palette.css", ext="demo"
        )
        meta = backend._pipeline_metadata()
        assert meta["logo"] == "BrandMark.png"
        assert meta["theme"] == "Palette.css"
        assert meta["ext"] == "demo"

    def test_returns_core_pipeline_metadata(self, backend_with_extension):
        assert backend_with_extension._pipeline_metadata() == {
            "name": "test",
            "ext": "collectra",
            "version": "1.0",
        }


class TestGetLogo:
    """Tests for GUIBackend.get_logo (PNG + SVG, name-from-metadata)."""

    def test_loads_png_named_in_metadata(
        self, backend_with_extension, temp_project_with_collectra
    ):
        backend = backend_with_extension
        backend._parent_folder = str(temp_project_with_collectra)
        backend._pipeline.pipeline_metadata["logo"] = "BrandMark.png"
        result = backend.get_logo()
        assert result["success"] is True
        assert result["data"].startswith("data:image/png;base64,")

    def test_loads_svg_by_glob_fallback(self, backend_with_extension, tmp_path):
        # .collectra folder with an SVG but no metadata logo name
        collectra = tmp_path / "proj.collectra"
        collectra.mkdir()
        (collectra / "brand-logo.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        )
        backend = backend_with_extension
        backend._parent_folder = str(tmp_path)
        result = backend.get_logo()
        assert result["success"] is True
        assert result["data"].startswith("data:image/svg+xml;base64,")

    def test_falls_back_to_bundled_default_when_no_logo(
        self, backend_with_extension, tmp_path
    ):
        # A project with no logo anywhere must still succeed with Collectra's
        # own bundled logo — otherwise the frontend's <img> keeps showing
        # whichever project's logo loaded last, rather than resetting.
        backend = backend_with_extension
        backend._parent_folder = str(tmp_path)
        result = backend.get_logo()
        assert result["success"] is True
        assert result["data"].startswith("data:image/svg+xml;base64,")


class TestGetThemeCss:
    """Tests for GUIBackend.get_theme_css name-from-metadata preference."""

    def test_loads_theme_named_in_metadata(
        self, backend_with_extension, temp_project_with_collectra
    ):
        backend = backend_with_extension
        backend._parent_folder = str(temp_project_with_collectra)
        backend._pipeline.pipeline_metadata["theme"] = "Palette.css"
        result = backend.get_theme_css()
        assert result["success"] is True
        assert result["path"].endswith("Palette.css")
        assert "--dag-bg" in result["css"]

    def test_prefers_metadata_named_file_over_literal_theme_css(
        self, backend_with_extension, tmp_path
    ):
        # Folder has BOTH a decoy theme.css and the file actually named in metadata.
        # A selection that just grabs whatever is named "theme.css" would return the
        # decoy; the metadata name must win.
        collectra = tmp_path / "proj.collectra"
        collectra.mkdir()
        (collectra / "theme.css").write_text("/* decoy */ body { color: red; }")
        (collectra / "grapto.css").write_text("/* correct */ body { color: blue; }")

        backend = backend_with_extension
        backend._parent_folder = str(tmp_path)
        backend._pipeline.pipeline_metadata["theme"] = "grapto.css"
        result = backend.get_theme_css()

        assert result["success"] is True
        assert result["path"].endswith("grapto.css")
        assert "blue" in result["css"]


class TestGetActiveNodeIds:
    """Tests for GUIBackend.get_active_node_ids."""

    def test_returns_node_ids_for_page(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        backend = backend_with_extension
        backend._scan_parent_folder(temp_parent_folder_with_subfolders)
        assert len(backend._collectra_folders) > 0
        result = backend.get_active_node_ids(0)
        assert result["success"] is True
        # active_ids must be labels (matching pipeline.yaml keys), not results.yaml's
        # internal per-instance ids (e.g. "img_001") — those are different strings.
        assert set(result["active_ids"]) == {"image_label", "crop_label", "text_label"}
        assert result["node_types"] == {
            "image_label": "collectra.Image",
            "crop_label": "collectra.ImageCrop",
            "text_label": "collectra.Text",
        }
        # node_ids carries the real results.yaml id(s) (distinct from the
        # label) for each label — a list, since a detector-type node can
        # produce multiple real instances under one pipeline label.
        assert result["node_ids"]["image_label"] == ["img_001"]

    def test_groups_multiple_real_ids_under_one_label(
        self,
        backend_with_extension,
        temp_parent_folder_with_subfolders,
        sample_yaml_data,
    ):
        # A detector node (e.g. collectra.ObjectDetectionYOLO) declares one
        # output label in pipeline.yaml but can produce any number of real
        # instances per page — all of them must be preserved, not just the
        # last one processed.
        import yaml

        backend = backend_with_extension
        backend._scan_parent_folder(temp_parent_folder_with_subfolders)
        folder = backend._collectra_folders[0]
        data = dict(sample_yaml_data)
        data["crop_label"] = [
            {**data["crop_label"][0], "id": "crop_label-aaa"},
            {**data["crop_label"][0], "id": "crop_label-bbb"},
        ]
        with open(folder["yaml_path"], "w") as f:
            yaml.dump(data, f)

        result = backend.get_active_node_ids(0)

        assert result["success"] is True
        assert set(result["node_ids"]["crop_label"]) == {
            "crop_label-aaa",
            "crop_label-bbb",
        }

    def test_invalid_index_returns_error(
        self, backend_with_extension, temp_parent_folder_with_subfolders
    ):
        backend = backend_with_extension
        backend._scan_parent_folder(temp_parent_folder_with_subfolders)
        result = backend.get_active_node_ids(999)
        assert result["success"] is False

    def test_no_folders_returns_error(self, backend_with_extension):
        result = backend_with_extension.get_active_node_ids(0)
        assert result["success"] is False


class TestUpdatePageText:
    """Tests for GUIBackend.update_page_text."""

    def test_writes_markdown_file(self, backend_with_extension, temp_yaml_file):
        backend = backend_with_extension
        backend._yaml_path = str(temp_yaml_file)

        result = backend.update_page_text("markdown", "# Updated heading")

        assert result["success"] is True
        written = (temp_yaml_file.parent / "markdown.md").read_text(encoding="utf-8")
        assert written == "# Updated heading"

    def test_writes_tei_file(self, backend_with_extension, temp_yaml_file):
        backend = backend_with_extension
        backend._yaml_path = str(temp_yaml_file)

        result = backend.update_page_text("tei", "<TEI>updated</TEI>")

        assert result["success"] is True
        written = (temp_yaml_file.parent / "tei.xml").read_text(encoding="utf-8")
        assert written == "<TEI>updated</TEI>"

    def test_overwrites_existing_file(self, backend_with_extension, temp_yaml_file):
        backend = backend_with_extension
        backend._yaml_path = str(temp_yaml_file)
        (temp_yaml_file.parent / "markdown.md").write_text(
            "old content", encoding="utf-8"
        )

        result = backend.update_page_text("markdown", "new content")

        assert result["success"] is True
        written = (temp_yaml_file.parent / "markdown.md").read_text(encoding="utf-8")
        assert written == "new content"

    def test_returns_error_for_unknown_kind(
        self, backend_with_extension, temp_yaml_file
    ):
        backend = backend_with_extension
        backend._yaml_path = str(temp_yaml_file)

        result = backend.update_page_text("bogus", "content")

        assert result["success"] is False

    def test_returns_error_when_no_page_loaded(self, backend_with_extension):
        backend = backend_with_extension
        backend._yaml_path = None

        result = backend.update_page_text("markdown", "content")

        assert result["success"] is False
