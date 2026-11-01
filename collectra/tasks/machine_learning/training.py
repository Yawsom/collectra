"""Opt-in utilities for training task implementations."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from rich.console import Console
from rich.table import Table

from collectra.types.base import ArtefactNode
from collectra.types.images import Image, ImageCrop
from collectra.commons.results import peek_header
from collectra.utils import change_dir

from ...logger import get_logger

logger = get_logger(__name__)


def print_distribution_table(
    title: str,
    classes: list[str],
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
    train_images: tuple[int, int] | None = None,
    validation_images: tuple[int, int] | None = None,
) -> None:
    """Print class counts and each class's held-out validation percentage.

    ``train_images`` and ``validation_images`` are optional
    ``(with objects, background)`` image counts. Background images have no
    objects but are still used as negative training examples.
    """
    table = Table(title=title, show_lines=True)
    table.add_column(
        "Class Name", justify="left", style="green", header_style="bold green"
    )
    table.add_column(
        "Train Count", justify="right", style="red", header_style="bold red"
    )
    table.add_column(
        "Validation Count", justify="right", style="blue", header_style="bold blue"
    )
    table.add_column("Validation %", justify="right", header_style="bold")

    total_train = 0
    total_validation = 0
    for class_name in classes:
        train_count = train_counts[class_name]
        validation_count = validation_counts[class_name]
        total = train_count + validation_count
        percentage = 100 * validation_count / total if total else 0.0
        table.add_row(
            class_name,
            str(train_count),
            str(validation_count),
            f"{percentage:.1f}%",
        )
        total_train += train_count
        total_validation += validation_count

    grand_total = total_train + total_validation
    total_percentage = 100 * total_validation / grand_total if grand_total else 0.0
    table.add_row(
        "Total",
        str(total_train),
        str(total_validation),
        f"{total_percentage:.1f}%",
        style="bold",
    )
    if train_images is not None and validation_images is not None:
        rows = [
            ("Images with objects", train_images[0], validation_images[0]),
            ("Background images (no objects)", train_images[1], validation_images[1]),
            ("Total images", sum(train_images), sum(validation_images)),
        ]
        for label, train_count, validation_count in rows:
            total = train_count + validation_count
            percentage = 100 * validation_count / total if total else 0.0
            table.add_row(
                label,
                str(train_count),
                str(validation_count),
                f"{percentage:.1f}%",
                style="bold" if label == "Total images" else None,
            )
    Console().print(table)


def prepare_object_detection_inputs(
    processed_inputs: list,
    processed_parents: list,
    input_maps: dict[str, list],
    parent_input_maps: dict[str, list],
) -> list:
    """Attach source images to detection crops and return the task inputs."""
    for item in processed_inputs:
        if type(item) is Image:
            continue
        if type(item) is not ImageCrop:
            raise TypeError(f"Input {item.id} is not an ImageCrop")
        if isinstance(item.parents, str):
            parent_id = item.parents
        elif isinstance(item.parents, list) and len(item.parents) == 1:
            parent_id = item.parents[0]
        else:
            continue
        parent = next(
            (
                candidate
                for candidate in processed_parents
                if candidate.id == parent_id and item.data == candidate.data
            ),
            None,
        )
        if not isinstance(parent, Image):
            raise ValueError(
                f"Image parent {parent_id!r} for input {item.id!r} was not found"
            )
        item.add_source_parent(parent)
    return processed_inputs


def _training_files(task, inputs: list[str]) -> list[Path]:
    pipeline = _pipeline_for(task)
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(path.glob(f"*.{pipeline.ext}"))
        elif path.suffix == f".{pipeline.ext}":
            files.append(path)
    return files


def _file_partition(item_file: Path) -> str:
    result_file = item_file / "results.yaml"
    if not result_file.exists():
        return ""
    with open(result_file, "r") as f:
        file_data = yaml.safe_load(f) or {}
    return peek_header(file_data, source=result_file).partition or ""


def _limit_training_files(
    files: list[Path], max_items: int, validation: str = "", exclude: str = ""
) -> list[Path]:
    """Keep the first ``max_items`` training and ``max_items`` validation files.

    Only the partition metadata is read, and reading stops once both
    quotas are filled, so large datasets are not loaded for quick tests.
    """
    selected = []
    counts = {"train": 0, "validation": 0}
    for item_file in sorted(files):
        partition = _file_partition(item_file)
        if exclude and partition == exclude:
            continue
        split = "validation" if validation and partition == validation else "train"
        if counts[split] < max_items:
            selected.append(item_file)
            counts[split] += 1
        if counts["train"] >= max_items and (
            not validation or counts["validation"] >= max_items
        ):
            break
    logger.info(
        "max_items=%d: using %d training and %d validation files",
        max_items,
        counts["train"],
        counts["validation"],
    )
    return selected


def _prepare_data(
    task,
    files: list[Path],
    nodes: list[ArtefactNode],
    *,
    include_unlabelled: bool = False,
):
    processed = []
    input_maps = {}
    for item_file in files:
        input_maps[item_file.name] = ArtefactNode.batch_process(
            item_file,
            nodes,
            include_unlabelled=include_unlabelled,
        )
        processed.extend(input_maps[item_file.name])
    return processed, input_maps


def _pipeline_for(task):
    if task.pipeline is None:
        raise RuntimeError(f"Task {task.name!r} is not attached to a pipeline")
    return task.pipeline


def save_training_result(task, results, *, log: str, **kwargs) -> None:
    """Copy the best model and optional class metadata into the pipeline."""
    pipeline = _pipeline_for(task)
    best_model_path = results.save_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model file not found at {best_model_path}")
    logger.info("Training metrics: %s", results.results_dict)
    current_model = pipeline.data.get(task.name, {}).get("model", "")
    if best_model_path.name == current_model:
        return
    new_model_path = f"{task.name}-{Path(log).name}.pt"
    shutil.copy(best_model_path, new_model_path)
    pipeline.data[task.name]["model"] = new_model_path
    classes_src = best_model_path.parent / "classes.json"
    if classes_src.exists():
        shutil.copy(classes_src, f"{Path(new_model_path).stem}.classes.json")


def train_from_files(
    task,
    inputs: list[str],
    trainer: Callable,
    *,
    prepare_inputs: Callable | None = None,
    include_unlabelled: bool = False,
    **kwargs,
):
    """Resolve pipeline artefacts and invoke a concrete backend trainer."""
    pipeline = _pipeline_for(task)
    task_node = pipeline.node_manager.resolve_node(task.name)
    children = pipeline.node_manager.get_children_artefact(task_node)
    parents = pipeline.node_manager.get_ancestor_artefacts(task_node)
    kwargs["classes"] = kwargs.get("classes", [child.name for child in children])
    kwargs = pipeline._merge_task_params(task.name, kwargs)
    files = _training_files(task, inputs)
    max_items = int(kwargs.get("max_items", 0) or 0)
    if max_items > 0:
        files = _limit_training_files(
            files,
            max_items,
            validation=kwargs.get("validation", ""),
            exclude=kwargs.get("exclude", ""),
        )
    processed_inputs, input_maps = _prepare_data(
        task,
        files,
        children,
        include_unlabelled=include_unlabelled,
    )
    processed_parents, parent_input_maps = _prepare_data(task, files, parents)
    if prepare_inputs is not None:
        processed_inputs = prepare_inputs(
            processed_inputs,
            processed_parents,
            input_maps,
            parent_input_maps,
        )
    with change_dir(pipeline.path):
        results = trainer(*processed_inputs, **kwargs)
        save_training_result(task, results, **kwargs)
    return results


def run_training_command(
    task,
    inputs: list[str],
    trainer: Callable,
    *,
    keep_log: bool,
    output: Path | None,
    validation: str,
    exclude: str,
    prepare_inputs: Callable | None = None,
    include_unlabelled: bool = False,
    **kwargs,
):
    """Run the common filesystem lifecycle requested by a concrete command."""
    pipeline = _pipeline_for(task)
    log = (
        Path(output).expanduser().resolve()
        if output is not None
        else Path.cwd() / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results = train_from_files(
        task,
        inputs,
        trainer,
        prepare_inputs=prepare_inputs,
        include_unlabelled=include_unlabelled,
        project=f"{pipeline.name}-{task.name}",
        log=log,
        base_folder=Path.cwd(),
        validation=validation,
        exclude=exclude,
        **kwargs,
    )
    pipeline.save(task.name)
    if not keep_log:
        shutil.rmtree(log, ignore_errors=True)
    return results
