"""
Parses YAML structure, builds lineage relationships, and computes display values
for ImageCrop annotations based on the rules:

1. Container crop (has ImageCrop children) → display blank
2. Leaf crop with Text child → traverse to deepest Text → display that text
3. Leaf crop without Text child → display blank

"""

import copy
import uuid
from dataclasses import dataclass, field

import networkx as nx
from pydantic import BaseModel, ConfigDict

from collectra.commons.files import CollectraFile, CollectraResultsMetadata
from collectra.commons.results import pop_header, put_header

from .utils import normalise_items


class CollectraNodeFactory:

    @staticmethod
    def create_node(data: dict) -> "CollectraNode":
        """Factory method to create appropriate CollectraNode subclass."""
        node_type = data.get("type", "")
        from collectra.utils import load_class_from_string
        from collectra.types.images import ImageCrop

        try:
            is_crop = issubclass(load_class_from_string(node_type), ImageCrop)
        except (ImportError, AttributeError, ValueError):
            is_crop = "ImageCrop" in node_type
        if is_crop:
            crop_region = CollectraCropRegion(
                x_center=data["x_center"],
                y_center=data["y_center"],
                width_relative=data["width_relative"],
                height_relative=data["height_relative"],
            )
            return CollectraAnnotationNode(**data, crop_region=crop_region)
        return CollectraNode(**data)


class CollectraNode(BaseModel):
    """Node in the annotation graph."""

    model_config = ConfigDict(extra="allow")

    label: str
    type: str
    id: str
    parents: list[str] = []
    data: object = ""
    embeddings: list[str] = []
    orientation: str = "north"
    name: str = ""  # user-set per-instance name, distinct from label (the category)

    def model_dump(self, *args, **kwargs) -> dict[str, str | float]:
        data = super().model_dump(*args, **kwargs)
        label = data.pop("label", None)  # Exclude label from dump
        if "path" in data and "data" not in self.model_fields_set:
            data.pop("data", None)
        if data["parents"]:
            data["parents"] = (
                data["parents"][0] if len(data["parents"]) == 1 else data["parents"]
            )
        else:
            data.pop("parents", None)
        if not data.get("name"):
            data.pop("name", None)  # Don't clutter files that never set one
        if label is None:
            raise ValueError("Label is missing from node data.")
        return data


class CollectraCropRegion(BaseModel):
    """Crop region fields for annotation nodes."""

    x_center: float
    y_center: float
    width_relative: float
    height_relative: float


class CollectraAnnotationNode(CollectraNode):
    """Annotation node in the graph."""

    crop_region: CollectraCropRegion

    @property
    def crop(self) -> dict[str, float]:
        """Return crop region as a dict."""
        return self.crop_region.model_dump()

    def model_dump(self, *args, **kwargs) -> dict[str, str | float]:
        data = super().model_dump(*args, **kwargs)
        data.pop("crop_region", None)  # Remove crop_region field
        for key, value in self.crop.items():
            data[key] = value
        return data


class NodeDisplayValue(BaseModel):
    """Display value for a node."""

    reason: str
    value: str | None = None
    source_id: str | None = None
    crop_region: dict[str, float] | None = None
    locked: bool = False


@dataclass
class CollectraGraph:
    """
    Graph structure for annotation lineage relationships.

    Provides clean interface for traversal without external dependencies.
    """

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    metadata: CollectraResultsMetadata = field(default_factory=CollectraResultsMetadata)

    # Cached indexes for O(1) lookups
    _children_index: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _parents_index: dict[str, list[str]] = field(default_factory=dict, repr=False)

    # Labels seen in the source YAML, in original order — including ones whose
    # value was `[]` (a pipeline task ran and found nothing). Those never
    # become graph nodes, so nothing else remembers they exist; without this,
    # to_data() silently drops them on every save.
    _known_labels: list[str] = field(default_factory=list, repr=False)

    @property
    def nodes(self) -> list[str]:
        """List of node IDs in the graph."""
        if self._graph is None:
            return []
        return list(self._graph.nodes)

    @classmethod
    def from_yaml_data(cls, data: dict) -> "CollectraGraph":
        """Build graph from parsed YAML data."""
        data = copy.deepcopy(data)
        metadata = pop_header(data)
        return cls._from_data(data, metadata)

    @classmethod
    def from_collectra_file(cls, collectra_file: CollectraFile) -> "CollectraGraph":
        """Build a display graph backed by a loaded Collectra file."""
        return cls._from_data(
            copy.deepcopy(collectra_file.data),
            collectra_file.collectra_results_metadata,
        )

    @classmethod
    def _from_data(
        cls, data: dict, metadata: CollectraResultsMetadata
    ) -> "CollectraGraph":
        graph = cls(metadata=metadata)

        for label, value in data.items():
            graph._known_labels.append(label)
            items = normalise_items(value)
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    item["label"] = label  # Store label for reverse lookup
                    graph.add_node(item)

        return graph

    def add_node(self, data: dict) -> None:
        """Add a node and its parent edges."""
        data["parents"] = normalise_items(data.get("parents", []))
        node = CollectraNodeFactory.create_node(data)
        self._graph.add_node(
            node.id,
            node=node,
        )

        for parent in data["parents"]:
            self._graph.add_edge(parent, node.id)

    def children(self, node_id: str) -> list[str]:
        """Get immediate children of a node."""
        return list(self._graph.successors(node_id))

    def parents(self, node_id: str) -> list[str]:
        """Get immediate parents of a node."""
        return list(self._graph.predecessors(node_id))

    def get_node(self, node_id: str) -> CollectraNode | None:
        """Get the AnnotationNode object for a given node ID."""
        node = self._graph.nodes.get(node_id, {}).get("node", None)
        return node

    def resolve_id(self, node_id: str) -> str:
        """Translate a DAG label (e.g. "formation") to its real internal node id
        (e.g. "formation-<uuid>"). The DAG only ever knows the label; nodes are
        keyed internally by id. Returns node_id unchanged if it's already a real
        id, or if no node with that label exists.
        """
        if node_id in self._graph.nodes:
            return node_id
        for candidate_id in self._graph.nodes:
            node = self.get_node(candidate_id)
            if node and node.label == node_id:
                return candidate_id
        return node_id

    def get_type(self, node_id: str) -> str:
        """Get type of a node."""
        node = self.get_node(node_id)
        return node.type if node else ""

    def get_label(self, node_id: str) -> str:
        """Get label of a node."""
        node = self.get_node(node_id)
        return node.label if node else ""

    def get_data(self, node_id: str) -> str:
        """Get data field of a node."""
        node = self.get_node(node_id)
        return node.data if node else ""

    def get_name(self, node_id: str) -> str:
        """Get name field of a node."""
        node = self.get_node(node_id)
        return node.name if node else ""

    def get_crop_region(self, node_id: str) -> dict[str, float]:
        """
        Get the crop region coordinates of a node.

        Args:
            node_id: The node ID to get crop region from

        Returns:
            dict with x_center, y_center, width_relative, height_relative

        Raises:
            ValueError: If node is missing crop region fields
        """
        node = self.get_node(node_id)
        if not isinstance(node, CollectraAnnotationNode):
            raise ValueError(f"Node {node_id} is not an annotation node.")

        return node.crop

    def get_unique_labels(self) -> list[str]:
        """Get all unique labels from nodes in the graph."""
        labels = set()
        for node_id in self._graph.nodes:
            node = self.get_node(node_id)
            if node and node.label:
                labels.add(node.label)
        return sorted(labels)

    def count_nodes_by_label(self, type_filter: str = None) -> dict[str, int]:
        """Count nodes grouped by label, optionally filtered by type."""
        counts = {}
        for node_id in self._graph.nodes:
            node = self.get_node(node_id)
            if node and node.label:
                if type_filter is None or node.type == type_filter:
                    counts[node.label] = counts.get(node.label, 0) + 1
        return counts

    def set_label(self, node_id: str, label: str) -> None:
        """Set label field of a node."""
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph.")
        node.label = label

    def set_name(self, node_id: str, name: str) -> None:
        """Set name field of a node."""
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph.")
        node.name = name

    def set_data(self, node_id: str, data: str, crop_id: str | None = None) -> None:
        """Set data field of a node."""
        if node_id:
            node = self.get_node(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found in graph.")
            node.data = data
            return
        if not crop_id:
            raise ValueError(f"Node {node_id} not found and crop_id not provided.")
        text_data = {
            "label": "user_annotation_text",
            "type": "collectra.Text",
            "id": f"user_text_{uuid.uuid4().hex[:8]}",
            "parents": crop_id,
            "data": data,
        }
        self.add_node(text_data)

    def set_crop_region(self, node_id: str, crop_region: dict[str, float]) -> None:
        """Set crop region fields of a node (x_center, y_center, width_relative, height_relative)."""
        node = self.get_node(node_id)
        crop: CollectraCropRegion = CollectraCropRegion(**crop_region)
        if not isinstance(node, CollectraAnnotationNode):
            raise ValueError(f"Node {node_id} is not an annotation node.")
        node.crop_region = crop

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges connected to it."""
        self._graph.remove_node(node_id)

    def rename_id(self, old_id: str, new_id: str) -> None:
        """Changes a node's id, re-pointing every parent/child edge to match (nx.relabel_nodes)."""
        if old_id not in self._graph.nodes:
            raise ValueError(f"Node {old_id} not found in graph.")
        if new_id in self._graph.nodes:
            raise ValueError(f"Node {new_id} already exists in graph.")
        nx.relabel_nodes(self._graph, {old_id: new_id}, copy=False)
        node = self.get_node(new_id)
        if node:
            node.id = new_id

    def children_of_type(self, node_id: str, type_substr: str) -> list[str]:
        """Get immediate children containing type_substr in their type."""
        return [
            child_id
            for child_id in self.children(node_id)
            if type_substr in self.get_type(child_id)
        ]

    def dfs_leaves(self, start: str, type_filter: str) -> list[str]:
        """
        DFS traversal following only edges to nodes matching type_filter.
        Returns leaf nodes (nodes with no children of the filtered type).
        """
        leaves = []
        visited = set()
        stack = [start]

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)

            # Get children matching the type filter
            typed_children = self.children_of_type(node_id, type_filter)

            if not typed_children:
                # This is a leaf in the filtered subgraph
                leaves.append(node_id)
            else:
                stack.extend(typed_children)

        return leaves

    def find_deepest(self, start: str, type_filter: str) -> str:
        """
        Find the deepest node reachable from start following only type_filter edges.
        Returns the first leaf found (DFS order).
        """
        leaves = self.dfs_leaves(start, type_filter)
        return leaves[0] if leaves else ""

    def to_yaml_data(self) -> dict:
        """Convert graph back to YAML data structure."""
        return put_header(self.metadata, self.to_data())

    def to_data(self) -> dict:
        """Convert graph nodes to the data held by a CollectraFile."""
        yaml_data: dict = {}

        for node_id in self._graph.nodes:
            node = self.get_node(node_id)
            if node is None:
                continue
            node_data = node.model_dump()
            yaml_data.setdefault(node.label, []).append({**node_data})

        def _collapse(value: list[dict]) -> list[dict] | dict:
            return value[0] if len(value) == 1 else value

        final_data: dict[str, list[dict] | dict] = {}

        # Known labels first, in original order, so a label with no current
        # nodes (never had any, or had all of them deleted) round-trips as
        # `[]` instead of disappearing.
        for label in self._known_labels:
            final_data[label] = _collapse(yaml_data.pop(label, []))

        # Anything left is a label created fresh this session (e.g. a new
        # user-added Text node) that wasn't in the original data.
        for key, value in yaml_data.items():
            final_data[key] = _collapse(value)

        return final_data

    def compute_display_value(self, node_id: str) -> NodeDisplayValue:
        """
        Compute display value for an annotation.

        Args:
            node_id: The node ID to compute display value for

        Returns:
            NodeDisplayValue with value, source_id, crop_region, reason
        """
        if self.get_node(node_id) is None:
            return NodeDisplayValue(reason=f"{node_id} Element not found")

        node_type = self.get_type(node_id)

        # Rule: Image type displays empty
        if "Image" in node_type and "ImageCrop" not in node_type:
            return NodeDisplayValue(
                reason=f"{node_id} is of collectra.Image type: display blank",
                locked=True,
            )

        # Rules for ImageCrop
        if "ImageCrop" in node_type:
            crop_data = self.get_crop_region(node_id)

            # Rule 1: Container crop (has ImageCrop children)
            if self.children_of_type(node_id, "ImageCrop"):
                return NodeDisplayValue(
                    crop_region=crop_data,
                    reason=f"{node_id} is a container crop: display blank",
                    locked=True,
                )

            # Rule 2: Leaf crop with Text child
            text_children = self.children_of_type(node_id, "Text")
            text_children = text_children[0] if text_children else ""
            if text_children:
                deepest_id = self.find_deepest(text_children, "Text")
                if deepest_id == "":
                    raise ValueError(
                        f"Expected to find deepest Text child for node {node_id}, but none found."
                    )
                deepest_data = self.get_data(deepest_id) if deepest_id else ""
                return NodeDisplayValue(
                    value=deepest_data,
                    source_id=deepest_id,
                    crop_region=crop_data,
                    reason=f"Leaf crop {node_id} with Text child: deepest Text is {deepest_id}",
                )

            # Rule 3: Leaf crop without Text child
            return NodeDisplayValue(
                value="",
                source_id="",
                crop_region=crop_data,
                reason=f"Leaf crop {node_id} without Text child",
            )

        # Text elements (for completeness)
        if "Text" in node_type:
            return NodeDisplayValue(
                value=self.get_data(node_id),
                source_id=node_id,
                crop_region=None,
                reason="Text element: display data",
            )

        return NodeDisplayValue(
            value="",
            source_id="",
            crop_region=None,
            reason="Unknown type",
        )
