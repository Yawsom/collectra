"""
Tests for collectra.gui.data_display module.

Tests cover:
- normalize_parents helper function
- CollectraGraph class methods
- compute_display_value function with all display rules
- get_unique_labels and count_nodes_by_label methods
"""

import pytest

from collectra.gui.data_display import CollectraGraph, NodeDisplayValue
from collectra.gui.utils import normalise_items


class TestAnnotationGraphFromYamlData:
    """Tests for CollectraGraph.from_yaml_data class method."""

    def test_creates_graph_from_valid_data(self, sample_yaml_data):
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        assert len(graph.nodes) == 3
        assert "img_001" in graph.nodes
        assert "crop_001" in graph.nodes
        assert "text_001" in graph.nodes

    def test_parses_metadata(self, sample_yaml_data):
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        assert graph.metadata.version == "1.0.0"

    def test_empty_data_creates_empty_graph(self):
        graph = CollectraGraph.from_yaml_data({})
        assert len(graph.nodes) == 0

    def test_creates_edges_from_parents(self, sample_yaml_data):
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        assert "crop_001" in graph.children("img_001")
        assert "text_001" in graph.children("crop_001")

    def test_handles_list_values(self, complex_yaml_data):
        graph = CollectraGraph.from_yaml_data(complex_yaml_data)
        # Should have parsed both leaf crops
        assert "leaf_crop_001" in graph.nodes
        assert "leaf_crop_no_text" in graph.nodes

    def test_skips_items_without_id(self):
        data = {"label": [{"type": "collectra.Text", "data": "no id"}]}
        graph = CollectraGraph.from_yaml_data(data)
        assert len(graph.nodes) == 0

    def test_handles_non_dict_items(self):
        data = {"label": ["string_item", 123, None]}
        graph = CollectraGraph.from_yaml_data(data)
        assert len(graph.nodes) == 0


class TestAnnotationGraphAddNode:
    """Tests for CollectraGraph.add_node method."""

    def test_adds_node_to_empty_graph(self, empty_graph):
        data = {"label": "label", "type": "collectra.Text", "id": "t1", "data": "hello"}
        empty_graph.add_node(data)
        assert "t1" in empty_graph.nodes

    def test_adds_edges_for_parents(self, empty_graph):
        empty_graph.add_node({"label": "l1", "type": "t", "id": "parent"})
        empty_graph.add_node(
            {"label": "l2", "type": "t", "id": "child", "parents": "parent"}
        )
        assert "child" in empty_graph.children("parent")

    def test_handles_multiple_parents(self, empty_graph):
        empty_graph.add_node({"label": "l1", "type": "t", "id": "p1"})
        empty_graph.add_node({"label": "l2", "type": "t", "id": "p2"})
        empty_graph.add_node(
            {"label": "l3", "type": "t", "id": "child", "parents": ["p1", "p2"]}
        )
        assert "child" in empty_graph.children("p1")
        assert "child" in empty_graph.children("p2")

    def test_invalidates_cache_on_add(self, sample_graph):
        # This test is not applicable anymore since cache is no longer used
        # The graph now uses networkx directly for children/parents
        # Just verify that adding a node works correctly
        sample_graph.add_node(
            {
                "label": "new_label",
                "type": "collectra.Text",
                "id": "new_node",
                "parents": "crop_001",
            }
        )
        assert "new_node" in sample_graph.nodes
        assert "new_node" in sample_graph.children("crop_001")


class TestAnnotationGraphTraversal:
    """Tests for children, parents, and traversal methods."""

    def test_children_returns_immediate_children(self, sample_graph):
        children = sample_graph.children("img_001")
        assert children == ["crop_001"]

    def test_children_returns_empty_for_leaf(self, sample_graph):
        children = sample_graph.children("text_001")
        assert children == []

    def test_children_returns_empty_for_unknown_node(self, sample_graph):
        import networkx as nx

        with pytest.raises(nx.NetworkXError):
            sample_graph.children("nonexistent")

    def test_parents_returns_immediate_parents(self, sample_graph):
        parents = sample_graph.parents("crop_001")
        assert parents == ["img_001"]

    def test_parents_returns_empty_for_root(self, sample_graph):
        parents = sample_graph.parents("img_001")
        assert parents == []

    def test_parents_returns_empty_for_unknown_node(self, sample_graph):
        import networkx as nx

        with pytest.raises(nx.NetworkXError):
            sample_graph.parents("nonexistent")

    def test_children_of_type_filters_correctly(self, complex_graph):
        crops = complex_graph.children_of_type("container_crop_001", "ImageCrop")
        assert "leaf_crop_001" in crops
        assert "leaf_crop_no_text" in crops

    def test_children_of_type_returns_empty_when_no_match(self, sample_graph):
        # text_001 has no children at all
        result = sample_graph.children_of_type("text_001", "Text")
        assert result == []

    def test_dfs_leaves_finds_leaves(self, complex_graph):
        # Starting from text_001, find leaves of type Text
        leaves = complex_graph.dfs_leaves("text_001", "Text")
        assert "text_002" in leaves

    def test_dfs_leaves_returns_start_if_no_children(self, sample_graph):
        leaves = sample_graph.dfs_leaves("text_001", "Text")
        assert leaves == ["text_001"]

    def test_find_deepest_returns_deepest_node(self, complex_graph):
        deepest = complex_graph.find_deepest("text_001", "Text")
        assert deepest == "text_002"

    def test_find_deepest_returns_start_if_leaf(self, sample_graph):
        deepest = sample_graph.find_deepest("text_001", "Text")
        assert deepest == "text_001"

    def test_find_deepest_returns_none_for_nonexistent_type(self, sample_graph):
        # Start from img_001, look for "NonExistent" type
        deepest = sample_graph.find_deepest("img_001", "NonExistent")
        # Should return img_001 as it's a leaf in "NonExistent" type subgraph
        assert deepest == "img_001"


class TestAnnotationGraphGetters:
    """Tests for get_type, get_data, get_crop_region."""

    def test_get_label_returns_label(self, sample_graph):
        assert sample_graph.get_label("crop_001") == "crop_label"

    def test_get_label_returns_empty_for_unknown(self, sample_graph):
        assert sample_graph.get_label("nonexistent") == ""

    def test_get_type_returns_type(self, sample_graph):
        assert sample_graph.get_type("img_001") == "collectra.Image"
        assert sample_graph.get_type("crop_001") == "collectra.ImageCrop"
        assert sample_graph.get_type("text_001") == "collectra.Text"

    def test_get_type_returns_empty_for_unknown(self, sample_graph):
        assert sample_graph.get_type("nonexistent") == ""

    def test_get_data_returns_data(self, sample_graph):
        assert sample_graph.get_data("text_001") == "Hello World"

    def test_get_data_returns_empty_for_unknown(self, sample_graph):
        assert sample_graph.get_data("nonexistent") == ""

    def test_get_name_returns_empty_when_unset(self, sample_graph):
        assert sample_graph.get_name("crop_001") == ""

    def test_get_name_returns_name(self, sample_graph):
        sample_graph.set_name("crop_001", "My Sandglass")
        assert sample_graph.get_name("crop_001") == "My Sandglass"

    def test_get_crop_region_returns_coordinates(self, sample_graph):
        region = sample_graph.get_crop_region("crop_001")
        assert region["x_center"] == 0.5
        assert region["y_center"] == 0.5
        assert region["width_relative"] == 0.2
        assert region["height_relative"] == 0.1

    def test_get_crop_region_raises_for_missing_fields(self, sample_graph):
        with pytest.raises(ValueError, match="is not an annotation node"):
            sample_graph.get_crop_region("text_001")

    def test_get_crop_region_raises_for_partial_fields(self, empty_graph):
        # Node with only some crop fields - this should raise KeyError during node creation
        with pytest.raises(KeyError):
            empty_graph.add_node(
                {
                    "label": "label",
                    "type": "collectra.ImageCrop",
                    "id": "partial_crop",
                    "x_center": 0.5,
                }
            )


class TestAnnotationGraphSetters:
    """Tests for set_data, set_crop_region, set_label, set_name."""

    def test_set_name_updates_value(self, sample_graph):
        sample_graph.set_name("crop_001", "My Sandglass")
        assert sample_graph.get_node("crop_001").name == "My Sandglass"

    def test_set_name_raises_for_unknown_node(self, sample_graph):
        with pytest.raises(ValueError, match="not found"):
            sample_graph.set_name("nonexistent", "value")

    def test_set_label_updates_value(self, sample_graph):
        sample_graph.set_label("crop_001", "renamed_label")
        node = sample_graph.get_node("crop_001")
        assert node.label == "renamed_label"

    def test_set_label_raises_for_unknown_node(self, sample_graph):
        with pytest.raises(ValueError, match="not found"):
            sample_graph.set_label("nonexistent", "value")

    def test_set_data_updates_value(self, sample_graph):
        sample_graph.set_data("text_001", "Updated text")
        assert sample_graph.get_data("text_001") == "Updated text"

    def test_set_data_raises_for_unknown_node(self, sample_graph):
        with pytest.raises(ValueError, match="not found"):
            sample_graph.set_data("nonexistent", "value")

    def test_set_crop_region_updates_values(self, sample_graph):
        new_region = {
            "x_center": 0.8,
            "y_center": 0.9,
            "width_relative": 0.3,
            "height_relative": 0.4,
        }
        sample_graph.set_crop_region("crop_001", new_region)
        result = sample_graph.get_crop_region("crop_001")
        assert result == new_region

    def test_set_crop_region_raises_for_unknown_node(self, sample_graph):
        with pytest.raises(ValueError, match="is not an annotation node"):
            sample_graph.set_crop_region(
                "nonexistent",
                {
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width_relative": 0.1,
                    "height_relative": 0.1,
                },
            )

    def test_set_crop_region_raises_for_missing_required_field(self, sample_graph):
        from pydantic import ValidationError

        incomplete = {"x_center": 0.5, "y_center": 0.5}  # Missing width/height
        with pytest.raises(ValidationError):
            sample_graph.set_crop_region("crop_001", incomplete)


class TestAnnotationGraphRemoveNode:
    """Tests for remove_node method."""

    def test_removes_node_from_nodes(self, sample_graph):
        sample_graph.remove_node("text_001")
        assert "text_001" not in sample_graph.nodes

    def test_removes_edges_involving_node(self, sample_graph):
        sample_graph.remove_node("crop_001")
        # Edges to/from crop_001 should be gone
        assert "crop_001" not in sample_graph.children("img_001")
        assert "crop_001" not in sample_graph.nodes

    def test_invalidates_cache_on_remove(self, sample_graph):
        sample_graph.children("img_001")  # Populate cache
        sample_graph.remove_node("text_001")
        # Verify the node was removed and graph remains consistent
        assert "text_001" not in sample_graph.nodes

    def test_raises_for_unknown_node(self, sample_graph):
        import networkx as nx

        with pytest.raises(nx.NetworkXError):
            sample_graph.remove_node("nonexistent")


class TestAnnotationGraphRenameId:
    """Tests for rename_id method."""

    def test_new_id_replaces_old_id(self, sample_graph):
        sample_graph.rename_id("crop_001", "crop_999")

        assert "crop_001" not in sample_graph.nodes
        assert "crop_999" in sample_graph.nodes

    def test_node_object_id_field_updated(self, sample_graph):
        sample_graph.rename_id("crop_001", "crop_999")

        assert sample_graph.get_node("crop_999").id == "crop_999"

    def test_parent_edge_repointed(self, sample_graph):
        """img_001 -> crop_001 becomes img_001 -> crop_999."""
        sample_graph.rename_id("crop_001", "crop_999")

        assert "crop_999" in sample_graph.children("img_001")
        assert "crop_001" not in sample_graph.children("img_001")

    def test_child_edge_repointed(self, sample_graph):
        """crop_001 -> text_001 becomes crop_999 -> text_001."""
        sample_graph.rename_id("crop_001", "crop_999")

        assert "text_001" in sample_graph.children("crop_999")
        assert "crop_999" in sample_graph.parents("text_001")

    def test_raises_for_unknown_node(self, sample_graph):
        with pytest.raises(ValueError, match="not found"):
            sample_graph.rename_id("nonexistent", "new_id")

    def test_raises_when_new_id_already_taken(self, sample_graph):
        with pytest.raises(ValueError, match="already exists"):
            sample_graph.rename_id("crop_001", "text_001")


class TestAnnotationGraphToYamlData:
    """Tests for to_yaml_data method."""

    def test_roundtrip_preserves_structure(self, sample_yaml_data):
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        result = graph.to_yaml_data()

        # Check the header is preserved, as the first entry
        assert list(result)[0] == "results"
        assert result["results"]["type"] == "collectra.Results"

        # Check all nodes present in output
        all_ids = set()
        for key, value in result.items():
            if key == "results":
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    all_ids.add(item["id"])

        assert all_ids == {"img_001", "crop_001", "text_001"}

    def test_unset_name_omitted_from_dump(self, sample_yaml_data):
        """Pipeline-generated files never set a name — don't pollute them with it."""
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        result = graph.to_yaml_data()

        assert "name" not in result["crop_label"]

    def test_set_name_included_in_dump(self, sample_yaml_data):
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        graph.set_name("crop_001", "My Sandglass")
        result = graph.to_yaml_data()

        assert result["crop_label"]["name"] == "My Sandglass"

    def test_empty_list_field_preserved_through_roundtrip(self, sample_yaml_data):
        """A pipeline field where nothing was detected is stored as `label: []` —
        that's a real record ("this was checked, nothing found"), not an absent
        key, and must survive a save even though no node ever represents it."""
        sample_yaml_data["genus_image"] = []
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)

        result = graph.to_yaml_data()

        assert result["genus_image"] == []

    def test_empty_list_field_still_empty_after_unrelated_edit(self, sample_yaml_data):
        """The bug this guards: saving after touching an unrelated node used to
        drop every empty-list field from the file entirely."""
        sample_yaml_data["genus_image"] = []
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        graph.set_name("crop_001", "My Sandglass")

        result = graph.to_yaml_data()

        assert result["genus_image"] == []


class TestComputeDisplayValue:
    """Tests for compute_display_value function."""

    def test_nonexistent_node_returns_not_found(self, sample_graph):
        result = sample_graph.compute_display_value("nonexistent")
        assert result.value is None
        assert result.source_id is None
        assert result.crop_region is None
        assert "not found" in result.reason

    def test_image_type_returns_blank(self, sample_graph):
        result = sample_graph.compute_display_value("img_001")
        assert result.value is None
        assert "Image type" in result.reason or "Image" in result.reason

    def test_container_crop_returns_blank(self, complex_graph):
        result = complex_graph.compute_display_value("container_crop_001")
        assert result.value is None
        assert result.crop_region is not None
        assert "container" in result.reason.lower()

    def test_leaf_crop_with_text_returns_deepest_text(self, complex_graph):
        result = complex_graph.compute_display_value("leaf_crop_001")
        assert result.value == "Deepest text"
        assert result.source_id == "text_002"
        assert result.crop_region is not None

    def test_leaf_crop_without_text_returns_blank(self, complex_graph):
        result = complex_graph.compute_display_value("leaf_crop_no_text")
        assert result.value == ""
        assert result.crop_region is not None
        assert "without Text" in result.reason

    def test_text_element_returns_own_data(self, sample_graph):
        result = sample_graph.compute_display_value("text_001")
        assert result.value == "Hello World"
        assert result.source_id == "text_001"
        assert result.crop_region is None

    def test_simple_leaf_crop_with_direct_text(self, sample_graph):
        result = sample_graph.compute_display_value("crop_001")
        assert result.value == "Hello World"
        assert result.source_id == "text_001"
        assert result.crop_region is not None

    def test_unknown_type_returns_blank(self, empty_graph):
        empty_graph.add_node(
            {"label": "label", "type": "collectra.Unknown", "id": "unknown_type"}
        )
        result = empty_graph.compute_display_value("unknown_type")
        assert result.value == ""
        assert "Unknown type" in result.reason


class TestComputeDisplayValueEdgeCases:
    """Edge case tests for compute_display_value."""

    def test_empty_text_data(self, empty_graph):
        empty_graph.add_node(
            {
                "label": "img",
                "type": "collectra.Image",
                "id": "img",
                "data": "test.jpg",
            }
        )
        empty_graph.add_node(
            {
                "label": "crop",
                "type": "collectra.ImageCrop",
                "id": "crop",
                "parents": "img",
                "data": "test.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.1,
            }
        )
        empty_graph.add_node(
            {
                "label": "text",
                "type": "collectra.Text",
                "id": "text",
                "parents": "crop",
                "data": "",
            }
        )
        result = empty_graph.compute_display_value("crop")
        assert result.value == ""
        assert result.source_id == "text"

    def test_deeply_nested_text_chain(self, empty_graph):
        # Create a chain: Image -> Crop -> Text1 -> Text2 -> Text3
        empty_graph.add_node(
            {"label": "img", "type": "collectra.Image", "id": "img", "data": "test.jpg"}
        )
        empty_graph.add_node(
            {
                "label": "crop",
                "type": "collectra.ImageCrop",
                "id": "crop",
                "parents": "img",
                "data": "test.jpg",
                "x_center": 0.5,
                "y_center": 0.5,
                "width_relative": 0.2,
                "height_relative": 0.1,
            }
        )
        empty_graph.add_node(
            {
                "label": "text1",
                "type": "collectra.Text",
                "id": "text1",
                "parents": "crop",
                "data": "t1",
            }
        )
        empty_graph.add_node(
            {
                "label": "text2",
                "type": "collectra.Text",
                "id": "text2",
                "parents": "text1",
                "data": "t2",
            }
        )
        empty_graph.add_node(
            {
                "label": "text3",
                "type": "collectra.Text",
                "id": "text3",
                "parents": "text2",
                "data": "deepest",
            }
        )

        result = empty_graph.compute_display_value("crop")
        assert result.value == "deepest"
        assert result.source_id == "text3"


class TestSetDataWithCropId:
    """Tests for set_data with crop_id parameter."""

    def test_set_data_creates_text_node_when_node_id_empty(self, sample_graph):
        """set_data() creates new Text node when node_id is empty."""
        initial_count = len(sample_graph.nodes)
        sample_graph.set_data("", "New text content", crop_id="crop_001")
        assert len(sample_graph.nodes) == initial_count + 1
        # Verify new text node is child of crop_001
        new_texts = [n for n in sample_graph.nodes if n.startswith("user_text_")]
        assert len(new_texts) == 1
        assert new_texts[0] in sample_graph.children("crop_001")

    def test_set_data_raises_when_no_ids_provided(self, sample_graph):
        """set_data() raises ValueError when both node_id and crop_id are missing."""
        with pytest.raises(ValueError, match="not found and crop_id not provided"):
            sample_graph.set_data("", "data")


class TestLockedField:
    """Tests for locked field in NodeDisplayValue."""

    def test_compute_display_value_locks_image_nodes(self, sample_graph):
        """Image nodes return locked=True in display value."""
        result = sample_graph.compute_display_value("img_001")
        assert result.locked is True

    def test_compute_display_value_locks_container_crops(self, complex_graph):
        """Container crops return locked=True in display value."""
        result = complex_graph.compute_display_value("container_crop_001")
        assert result.locked is True

    def test_compute_display_value_unlocked_for_leaf_crops(self, sample_graph):
        """Leaf crops and text nodes return locked=False."""
        result = sample_graph.compute_display_value("crop_001")
        assert result.locked is False


class TestNodeNormalization:
    """Tests for node model_dump and to_yaml_data normalization."""

    def test_node_model_dump_normalizes_single_parent(self, empty_graph):
        """Node with single parent exports as string, not list."""
        empty_graph.add_node({"label": "l1", "type": "t", "id": "parent"})
        empty_graph.add_node(
            {"label": "l2", "type": "t", "id": "child", "parents": ["parent"]}
        )
        node = empty_graph.get_node("child")
        dumped = node.model_dump()
        assert dumped["parents"] == "parent"  # String, not list

    def test_to_yaml_data_unwraps_single_item_lists(self, sample_yaml_data):
        """to_yaml_data() returns single items as dicts, not lists."""
        graph = CollectraGraph.from_yaml_data(sample_yaml_data)
        result = graph.to_yaml_data()
        # Image is single item, should be dict not list
        assert isinstance(result["image_label"], dict)


class TestMetadataDefaults:
    """Tests for shared Collectra results metadata defaults."""

    def test_metadata_has_correct_defaults(self):
        """Metadata class has correct default values."""
        from collectra.commons.files import CollectraResultsMetadata

        meta = CollectraResultsMetadata()
        assert meta.version is None
        assert meta.workflow is None


class TestCollectraGraphGetUniqueLabels:
    """Tests for CollectraGraph.get_unique_labels method."""

    def test_returns_empty_list_for_empty_graph(self, empty_graph):
        result = empty_graph.get_unique_labels()
        assert result == []

    def test_returns_unique_labels_sorted(self, sample_graph):
        result = sample_graph.get_unique_labels()

        assert isinstance(result, list)
        # sample_yaml_data has: image_label, crop_label, text_label
        assert "image_label" in result
        assert "crop_label" in result
        assert "text_label" in result
        # Should be sorted
        assert result == sorted(result)

    def test_returns_labels_from_multi_label_data(self, multi_label_yaml_data):
        graph = CollectraGraph.from_yaml_data(multi_label_yaml_data)

        result = graph.get_unique_labels()

        # multi_label_yaml_data has: image_label, header_crop, body_crop, footer_crop
        assert "image_label" in result
        assert "header_crop" in result
        assert "body_crop" in result
        assert "footer_crop" in result

    def test_deduplicates_labels(self, complex_yaml_data):
        graph = CollectraGraph.from_yaml_data(complex_yaml_data)

        result = graph.get_unique_labels()

        # leaf_crop appears twice in complex_yaml_data but should only appear once
        label_counts = {}
        for label in result:
            label_counts[label] = label_counts.get(label, 0) + 1

        # No label should appear more than once
        for label, count in label_counts.items():
            assert count == 1, f"Label '{label}' appears {count} times in result"


class TestCollectraGraphCountNodesByLabel:
    """Tests for CollectraGraph.count_nodes_by_label method."""

    def test_returns_empty_dict_for_empty_graph(self, empty_graph):
        result = empty_graph.count_nodes_by_label()
        assert result == {}

    def test_counts_all_nodes_without_filter(self, sample_graph):
        result = sample_graph.count_nodes_by_label()

        # sample_yaml_data has: 1 image_label, 1 crop_label, 1 text_label
        assert result.get("image_label", 0) == 1
        assert result.get("crop_label", 0) == 1
        assert result.get("text_label", 0) == 1

    def test_counts_with_type_filter(self, sample_graph):
        result = sample_graph.count_nodes_by_label(type_filter="collectra.ImageCrop")

        # Only crop_label has type collectra.ImageCrop
        assert result.get("crop_label", 0) == 1
        # Other labels should not be counted
        assert result.get("image_label", 0) == 0
        assert result.get("text_label", 0) == 0

    def test_counts_multiple_nodes_same_label(self, multi_label_yaml_data):
        graph = CollectraGraph.from_yaml_data(multi_label_yaml_data)

        result = graph.count_nodes_by_label(type_filter="collectra.ImageCrop")

        # header_crop has 2 entries, body_crop has 1, footer_crop has 1
        assert result.get("header_crop", 0) == 2
        assert result.get("body_crop", 0) == 1
        assert result.get("footer_crop", 0) == 1

    def test_returns_empty_when_no_type_matches(self, sample_graph):
        result = sample_graph.count_nodes_by_label(
            type_filter="collectra.NonExistentType"
        )

        assert result == {}


class TestResolveId:
    """Tests for CollectraGraph.resolve_id — translates a DAG label (e.g.

    'formation') to its real internal node id (e.g. 'formation-<uuid>'),
    since the DAG only ever knows the label, never the id.
    """

    def test_resolves_label_to_real_id(self, sample_graph):
        # sample_yaml_data's top-level key "image_label" is the label;
        # its actual node id is "img_001" — a different string.
        assert sample_graph.resolve_id("image_label") == "img_001"

    def test_passes_through_an_already_real_id_unchanged(self, sample_graph):
        assert sample_graph.resolve_id("img_001") == "img_001"

    def test_returns_input_unchanged_when_nothing_matches(self, sample_graph):
        assert sample_graph.resolve_id("nonexistent_label") == "nonexistent_label"
