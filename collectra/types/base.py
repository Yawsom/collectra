import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich import print

from collectra.cli import command

from ..commons import BaseEntity, Node, NodeStatus
from ..utils import (
    change_dir,
    error_msg,
    load_class_from_string,
    traceback_error,
)

__all__ = ["Artefact", "ArtefactNode"]

from ..logger import get_logger

logger = get_logger(__name__)


class ArtefactCommands:
    """CLI capabilities for a named pipeline artefact node.

    Concrete artefact classes can return a subclass from ``cli_commands`` to
    expose commands applying to every matching item in that node.
    """

    def __init__(self, node: "ArtefactNode", ext: str) -> None:
        self.node = node
        self.ext = ext

    @command
    def evaluate(
        self,
        predicted_folder: Path,
        gold_folder: Path,
        threshold: float = 0.5,
        match_by_order: bool = False,
        output: Path | None = None,
    ) -> None:
        """Evaluate this artefact against ground-truth Collectra results."""
        from collectra.evaluator.base import Evaluator

        report = Evaluator(
            predicted_folder,
            gold_folder,
            ext=self.ext,
            labels={self.node.name},
            match_by_order=match_by_order,
        ).evaluate(threshold=threshold)
        for table in [*report.aggregate_tables, *report.tables]:
            print(table)
        if output is not None:
            report.save_csv(output)


@dataclass
class Artefact(BaseEntity):
    name: str
    id: str = field(default="")
    parents: list[str] = field(default_factory=list)
    partition: str = field(default="")
    embeddings: list[str] = field(default_factory=list)
    orientation: str = field(default="north")  # e.g., "landscape" or "portrait"
    ensemble: list[str] = field(default_factory=list)
    source_file: Path | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def set_parents(self, parents: list["Artefact"]) -> None:
        self.parents = [parent.id for parent in parents]

    def serialize(self) -> dict:
        serialized = super().serialize()
        if not self.partition:
            serialized.pop("partition", None)
        if "parents" in serialized:
            if len(serialized["parents"]) == 0:
                serialized.pop("parents")
            elif len(serialized["parents"]) == 1:
                serialized["parents"] = serialized["parents"][0]
        return serialized

    def evaluate(self, gold) -> float:
        raise NotImplementedError(
            "Eval method not implemented for base Artefact class."
        )

    def display(self, context) -> dict:
        """Describe this artefact using the display protocol (see collectra.display).

        Override to return an image, text, or properties view, or supply a
        custom JavaScript module using ``context.renderer(module_path)``.
        The context resolves related artefacts and publishes previews without
        altering data. See docs/artefacts/display.rst for the renderer contract.
        """
        return {"kind": "properties", "data": self.serialize()}

    def _extract(self, path: Path) -> None:
        raise NotImplementedError(
            "Extract method not implemented for base Artefact class."
        )

    def extract(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._extract(path)

    @classmethod
    def cli_commands(cls, node: "ArtefactNode", ext: str) -> ArtefactCommands:
        """Return the CLI object representing this artefact type in ``node``."""
        return ArtefactCommands(node, ext)

    def _generate_id(self) -> str:
        # Generate a unique ID based on the name and other attributes
        return f"{self.name}-{uuid.uuid4()}"

    def __post_init__(self) -> None:
        super().__init__(self.name)  # Initialize BaseEntity
        # An automatic id is provisional: the pipeline renames it after the
        # node (``node`` or ``node-k``) when the artefact joins its node.
        self._auto_id = not self.id
        if not self.id:
            self.id = self._generate_id()

    def attributes_to_ignore(self) -> set:
        attributes = super().attributes_to_ignore()
        attributes.add("name")
        attributes.add("partition")
        attributes.add("source_file")
        attributes.add("_auto_id")
        return attributes

    @classmethod
    def all_attributes(cls):
        return set(
            [
                attribute
                for attribute in dir(cls)
                if not attribute.startswith("_") and not callable(attribute)
            ]
        )


@dataclass
class ArtefactNode(Node):

    items: dict[str, Artefact] = field(default_factory=dict)
    ensemble_items: dict[str, Artefact] = field(default_factory=dict)
    types: set[type] = field(default_factory=set)
    skip_type_check: bool = False
    ensemble: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.status = NodeStatus.READY if self.items else NodeStatus.NOT_READY

    def add_item(self, item: Artefact) -> None:
        try:
            self.items[item.id] = item
            self.status = NodeStatus.READY
        except Exception as e:
            raise RuntimeError(f"Error adding item to ArtefactNode: {str(e)}")

    def get_item(self, item_id: str) -> Artefact | None:
        return self.items.get(item_id, None)

    def add_type(self, type_: type) -> None:
        self.types.add(type_)

    def check_type(self, type_: type) -> bool:
        for t in self.types:
            if issubclass(type_, t):
                return True
        return False

    def __str__(self) -> str:
        types_str = "\n".join([t.get_class_path() for t in self.types])
        return f"{self.name}\n{types_str}"

    def evaluate(
        self, gold_items: "ArtefactNode", threshold: float = 0.5
    ) -> dict[str, int | float]:
        """
        Evaluate predicted items against gold standard items.

        This implements the Hungarian algorithm (optimal bipartite matching) that:
        1. Validates type compatibility between predicted and gold items
        2. Computes pairwise scores between all predicted and gold items
        3. Finds optimal assignment that maximizes total matching score
        4. Filters matches below threshold
        5. Computes aggregate metrics (precision, recall, F1, mean score)

        Args:
            gold_items: ArtefactNode containing ground truth items
            threshold: Minimum score for a valid match (default 0.5)
                    - For ImageCrop: IoU threshold
                    - For Text: similarity threshold

        Returns:
            dict: {
                "precision": float,      # TP / (TP + FP)
                "recall": float,         # TP / (TP + FN)
                "f1": float,             # 2 * (P * R) / (P + R)
                "mean_score": float,     # Average score of matched pairs
                "num_predicted": int,    # Total predicted items
                "num_gold": int,         # Total gold items
                "num_matched": int,      # Number of successful matches (TP)
                "num_false_positives": int,  # Predicted items not matched
                "num_false_negatives": int,  # Gold items not matched
                "matches": [            # List of matched pairs
                    {
                        "predicted_id": str,
                        "gold_id": str,
                        "score": float
                    },
                    ...
                ],
                "unmatched_predicted": [str],  # IDs of unmatched predictions
                "unmatched_gold": [str]        # IDs of unmatched gold items
            }

        Raises:
            TypeError: If predicted and gold items are of incompatible types
        """

        if threshold > 1.0 or threshold < 0.0:
            raise ValueError(f"Invalid threshold provided: {threshold}")

        # Use gold items for type detection when self is empty, so labels with
        # no predictions still appear in aggregated metrics (e.g. empty [] in YAML)
        _type_source = self.items if self.items else gold_items.items
        metrics = {
            "types": list(set([type(item).__name__ for item in _type_source.values()])),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_score": 0.0,
            "num_predicted": 0,
            "num_gold": 0,
            "num_matched": 0,
            "num_false_positives": 0,
            "num_false_negatives": 0,
            "matches": [],
            "unmatched_predicted": [],
            "unmatched_gold": [],
        }

        # Edge case: Empty inputs with explicit semantic definitions
        if len(self.items) == 0 and len(gold_items.items) == 0:
            # Vacuous truth: nothing to predict, nothing to find
            metrics.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
            return metrics

        if len(self.items) == 0:
            # No predictions: precision undefined (0/0), defaults to 0.0 for consistency
            metrics.update(
                {
                    "precision": 0.0,
                    "num_gold": len(gold_items.items),
                    "num_false_negatives": len(gold_items.items),
                    "unmatched_gold": list(gold_items.items.keys()),
                }
            )
            return metrics

        if len(gold_items.items) == 0:
            # No gold items: zero precision (all FPs), perfect recall (nothing to miss)
            metrics.update(
                {
                    "recall": 1.0,
                    "num_predicted": len(self.items),
                    "num_false_positives": len(self.items),
                    "unmatched_predicted": list(self.items.keys()),
                }
            )
            return metrics

        # Step 0: Validate type compatibility upfront
        predicted_types = set([type(value).__name__ for value in self.items.values()])
        gold_types = set([type(value).__name__ for value in gold_items.items.values()])
        compatible = predicted_types == gold_types
        if not compatible:
            raise TypeError(
                f"Cannot evaluate {predicted_types} predictions against "
                f"{gold_types} gold items. Types must match."
            )

        # Step 1: Build score matrix
        # Shape: (num_predicted, num_gold)
        predicted_ids = list(self.items.keys())
        gold_ids = list(gold_items.items.keys())
        n_pred = len(self.items.keys())
        n_true = len(gold_items.items.keys())
        score_matrix = np.zeros((n_pred, n_true))

        if n_true > n_pred:
            padding = np.full([n_true - n_pred, n_true], 1e-9)
            score_matrix = np.vstack([score_matrix, padding])
        elif n_true < n_pred:
            padding = np.full([n_pred, n_pred - n_true], 1e-9)
            score_matrix = np.hstack([score_matrix, padding])

        for pred_idx, pred_id in enumerate(predicted_ids):
            pred_item = self.items[pred_id]
            for gold_idx, gold_id in enumerate(gold_ids):
                try:
                    gold_item = gold_items.items[gold_id]
                    score = pred_item.evaluate(gold_item)
                    score_matrix[pred_idx, gold_idx] = score
                except NotImplementedError as e:
                    logger.debug(f"Skipping evaluation for {pred_id} vs {gold_id}: {e}")

        # Step 2: Hungarian algorithm for optimal bipartite matching
        # Convert to cost matrix (minimize cost = maximize score)
        # scipy.optimize.linear_sum_assignment minimizes total cost
        # If cost_matrix has a shape of 1, skip this step
        if score_matrix.size != 1:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1.0 - score_matrix

            # Note: When multiple predicted items have identical scores against the same
            # gold item, the assignment is deterministic but arbitrary (based on internal
            # algorithm traversal order). All such assignments are equally optimal.
            try:
                pred_indices, gold_indices = linear_sum_assignment(cost_matrix)
            except Exception as e:
                raise RuntimeError(f"Error during optimal assignment: {str(e)}")
        else:
            pred_indices = np.array([0])
            gold_indices = np.array([0])

        # Step 3: Filter matches by threshold and record results
        matches = []
        matched_predicted = set()
        matched_gold = set()

        for pred_idx, gold_idx in zip(pred_indices, gold_indices):
            if pred_idx >= n_pred or gold_idx >= n_true:
                continue
            score = score_matrix[pred_idx, gold_idx]
            # Only accept matches above the threshold
            if score >= threshold:
                predicted_id = predicted_ids[pred_idx]
                gold_id = gold_ids[gold_idx]
                matches.append(
                    {
                        "predicted_id": predicted_id,
                        "gold_id": gold_id,
                        "score": float(score),
                    }
                )
                matched_predicted.add(predicted_id)
                matched_gold.add(gold_id)

        # Step 4: Identify unmatched items
        unmatched_predicted = [
            predicted_id
            for predicted_id in predicted_ids
            if predicted_id not in matched_predicted
        ]

        unmatched_gold = [
            gold_id for gold_id in gold_ids if gold_id not in matched_gold
        ]

        # Step 5: Compute metrics
        num_true_positives = len(matches)
        num_false_positives = len(unmatched_predicted)
        num_false_negatives = len(unmatched_gold)

        precision = (
            num_true_positives / (num_true_positives + num_false_positives)
            if (num_true_positives + num_false_positives) > 0
            else 0.0
        )

        recall = (
            num_true_positives / (num_true_positives + num_false_negatives)
            if (num_true_positives + num_false_negatives) > 0
            else 0.0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_score = (
            sum(match["score"] for match in matches) / len(matches) if matches else 0.0
        )

        # Step 6: Return comprehensive evaluation results
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_score": mean_score,
                "num_predicted": len(self.items),
                "num_gold": len(gold_items.items),
                "num_matched": num_true_positives,
                "num_false_positives": num_false_positives,
                "num_false_negatives": num_false_negatives,
                "matches": matches,
                "unmatched_predicted": unmatched_predicted,
                "unmatched_gold": unmatched_gold,
            }
        )
        return metrics

    def _create_instance(
        self, cls_: type, *, source_file: Path | None = None, **item
    ) -> None:
        try:
            instance = cls_(**item)
            if not instance:
                raise ValueError(f"Failed to load {item} with {cls_}")
            instance.source_file = source_file
            self.add_item(instance)
        except Exception as e:
            logger.error(
                "Failed to create instance of %s: %s", cls_.__name__, e, exc_info=True
            )

    def _create_ensemble_instances(self, base_path: Path, cls_: type, **item) -> None:
        try:
            item["id"] = f"{base_path.parent}::{item['id']}"
            if "parents" in item:
                for parent_idx, _ in enumerate(item["parents"]):
                    item["parents"][
                        parent_idx
                    ] = f"{base_path.parent}::{item['parents'][parent_idx]}"
            instance = cls_(**item)
            if not instance:
                raise ValueError(f"Failed to load {item} with {cls_}")
            self.ensemble_items[instance.id] = instance
            self.status = NodeStatus.READY
        except Exception as e:
            raise RuntimeError(f"Error creating ensemble instances: {str(e)}")

    def _create_instances(self, **item) -> None:
        for cls_ in self.types:
            self._create_instance(cls_, **item)

    def process(
        self,
        key: str,
        value: str | Path | None = None,
        **kwargs,
    ) -> None:
        skip_type_check: bool = kwargs.get("skip_type_check", False)
        ensemble: bool = kwargs.get("ensemble", False)
        value = value if value else kwargs.get("file", None)
        if value and Path(value).exists() and Path(value).is_dir():
            value = Path(value)
            with change_dir(value):
                try:
                    result_file = Path("results.yaml")
                    with open(result_file, "r") as f:
                        results: dict = yaml.safe_load(f)
                        partition = results.pop(
                            "collectra_results_metadata", dict()
                        ).get("partition", None)
                        if key not in results:
                            # Normal when the task producing this node hasn't run yet.
                            logger.info(
                                "No %s yet in %s; it will be created when its task runs.",
                                key,
                                value.name,
                            )
                            return
                        data = results[key]
                        if not data:
                            # Key exists but empty — pipeline processed it, found nothing
                            self.status = NodeStatus.READY
                            return
                        data = data if isinstance(data, list) else [data]
                        for item in data:
                            primitive_type = False
                            try:
                                primitive_type = not isinstance(item, dict) or not (
                                    "type" in item
                                    and (
                                        "path" in item
                                        or "data" in item
                                        or item["type"].endswith(".Link")
                                    )
                                )
                                if primitive_type:
                                    raise ValueError(
                                        f"Item must be a complex type with 'type' and ('path' or 'data')"
                                    )
                                cls_ = load_class_from_string(item.pop("type"))
                                if not (
                                    skip_type_check or ensemble
                                ) and not self.check_type(cls_):
                                    raise TypeError(
                                        f"{cls_} is not a subclass or not defined in {self.types}"
                                    )
                                item["name"] = key
                                if (
                                    "data" not in item
                                    and "path" in item
                                    and "path" not in cls_.__dataclass_fields__
                                ):
                                    item["data"] = item.pop("path")
                                if "parents" in item and not isinstance(
                                    item["parents"], list
                                ):
                                    item["parents"] = [item["parents"]]
                                if (
                                    partition is not None
                                    and "partition" in cls_.all_attributes()
                                ):
                                    item["partition"] = partition
                                if ensemble:
                                    self._create_ensemble_instances(value, cls_, **item)
                                else:
                                    self._create_instance(
                                        cls_, source_file=value, **item
                                    )
                            except Exception as e:
                                if not primitive_type:
                                    logger.error(
                                        "Error processing item in %s: %s",
                                        self.name,
                                        e,
                                        exc_info=True,
                                    )
                                else:
                                    self._create_instances(name=key, data=str(item))

                except Exception as e:
                    logger.error(
                        "Error loading data for %s from %s: %s",
                        key,
                        value,
                        e,
                        exc_info=True,
                    )
        elif key:
            self._create_instances(name=key, data=value)

    @staticmethod
    def batch_process(
        item_file: Path,
        artefact_nodes: list["ArtefactNode"],
        *,
        include_unlabelled: bool = False,
    ) -> list[Artefact]:
        names = [artefact_node.name for artefact_node in artefact_nodes]
        with change_dir(item_file):
            try:
                data: list[Artefact] = list()
                result_file = Path("results.yaml")
                if not result_file.exists():
                    print(error_msg(f"Invalid file: {Path.cwd()}. Ignoring..."))
                    return data
                with open(result_file, "r") as f:
                    file_data: dict = yaml.safe_load(f)
                    partition = file_data.get("collectra_results_metadata", dict()).get(
                        "partition", None
                    )
                # With unlabelled samples allowed, an empty annotation list
                # means the same as no annotations at all.
                all_names_not_found = all(
                    name not in file_data
                    or (include_unlabelled and not file_data[name])
                    for name in names
                )
                # Object detection can opt into base images with no annotations
                # as negative samples. Classification must not treat an
                # unlabelled image as a training example.
                if all_names_not_found and include_unlabelled:
                    found_base = False
                    for name, values in file_data.items():
                        values = values if isinstance(values, list) else [values]
                        for item in values:
                            if (
                                not isinstance(item, dict)
                                or "type" not in item
                                and ("data" not in item or "path" not in item)
                            ):
                                continue
                            cls_str = item.pop("type", "")
                            if cls_str == "collectra.Image":
                                cls_ = load_class_from_string(cls_str)
                                try:
                                    item["name"] = name
                                    item["data"] = (
                                        item.pop("path")
                                        if "path" in item
                                        else item["data"]
                                    )
                                    if partition is not None:
                                        item["partition"] = partition
                                    instance = cls_(**item)
                                    if not instance:
                                        raise Warning(
                                            f"Failed to load {item} with {cls_}"
                                        )
                                    found_base = True
                                    data.append(instance)
                                    break
                                except Exception as e:
                                    traceback_error(
                                        e,
                                        f"Failed to load data item {name} from {item.get('data', '')}",
                                        verbose=True,
                                    )
                        if found_base:
                            break
                elif all_names_not_found:
                    return data
                for i, name in enumerate(names):
                    if name not in file_data:
                        continue
                    value = file_data[name]
                    if not value and include_unlabelled:
                        continue
                    if not value:
                        raise Warning(
                            f"Artefact seems to be empty for {name} in {item_file}. Provided: {value}"
                        )
                    value = value if isinstance(value, list) else [value]
                    for item in value:
                        if not isinstance(item, dict) or not (
                            "type" in item
                            and (
                                "data" in item
                                or "path" in item
                                or item["type"].endswith(".Link")
                            )
                        ):
                            continue
                        cls_ = load_class_from_string(item.pop("type"))
                        match = False
                        for type_ in artefact_nodes[i].types:
                            if issubclass(cls_, type_) or cls_ == type_:
                                match = True
                                break
                        if not match:
                            continue
                        item["name"] = name
                        if "path" in item and "path" not in cls_.__dataclass_fields__:
                            item["data"] = item.pop("path")
                        if partition is not None:
                            item["partition"] = partition
                        try:
                            instance = cls_(**item)
                            if instance:
                                instance.source_file = item_file
                                data.append(instance)
                        except Exception as e:
                            traceback_error(
                                e,
                                f"Failed to load data item {name} from {item.get('data', '')}: {e}",
                            )
                return data
            except Exception as e:
                traceback_error(e, verbose=True)
                return list()


def load_artefact(item: dict[str, Any]):
    artefact_class = load_class_from_string(item.pop("type"))

    return artefact_class(**item)
