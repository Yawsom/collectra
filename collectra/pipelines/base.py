"""Core pipeline execution and workflow management for Collectra.

This module contains the main Collectra workflow class.

It provides the core functionality for defining, excuting and managing
workflows with task dependencies.

The module supports:
    - Workflow definition and configuration management
    - Task dependency graph construction and execution
    - Workflow visualization and rendering
    - Training and inference execution modes

Classes:
    Collectra: Main workflow management class
"""

from __future__ import annotations

__all__ = ["Collectra"]

import copy
import tempfile
from contextlib import nullcontext
import datetime
import json
import shlex
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Annotated

from cappa import Arg
import graphviz
import networkx as nx
import yaml
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)

from collectra.utils import change_dir, load_class_from_string, write_yaml
from collectra.cli import command, group
from utils.get_types import get_param_types, get_return_type, unpack_types

from ..commons.base import TaskContext
from ..logger import get_logger
from ..tasks.base import (
    Task,
    TaskNode,
)
from ..types.base import (
    Artefact,
    ArtefactNode,
    NodeStatus,
)
from ..types.images import Image, ImageCrop
from ..types.links import Link
from ..types.texts import Text
from .node_graph_manager import NodeGraphManager

logger = get_logger(__name__)


class PreviousOutputs:
    """What a task produced before a forced rerun, kept until the rerun succeeds.

    Output items are emptied and the node's saved files are moved aside so the
    rerun starts clean. If the rerun produces nothing, everything is put back.
    """

    def __init__(self, snapshots: list, moved: list, backup: Path | None):
        self.snapshots = snapshots
        self.moved = moved
        self.backup = backup

    def restore(self) -> None:
        for output, items in self.snapshots:
            output.items.update(items)
        for kept, original in self.moved:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(kept), str(original))
        self.discard()

    def discard(self) -> None:
        if self.backup is not None:
            shutil.rmtree(self.backup, ignore_errors=True)
            self.backup = None


class RunProgress:
    """One overall bar for a multi-file run, with the file it is up to."""

    def __init__(self, progress, label: str, total: int):
        self.progress = progress
        self.label = label
        self.task = progress.add_task(escape(label), total=total)

    def status(self, text: str) -> None:
        self.progress.update(
            self.task, description=f"{escape(self.label)} · {escape(text)}"
        )

    def file_done(self) -> None:
        self.progress.advance(self.task)


class StageGate:
    """Holds files back at batchable tasks so a whole chunk can prefetch together."""

    def __init__(self, batchable: set[str]):
        self.batchable = batchable
        self.current: set[str] = set()
        self.deferred: dict[str, list[TaskNode]] = {}

    def defer(self, node: TaskNode, value) -> bool:
        """Hold ``node`` for the file at ``value``; False if it should run now."""
        if node.name not in self.batchable or node.name in self.current:
            return False
        held = self.deferred.setdefault(str(value), [])
        if node not in held:
            held.append(node)
        return True


class Collectra:
    """Main workflow management class for Collectra pipelines.

    This class orchestrates the execution of complex data processing workflows
    by managing task dependencies, data flow, and execution state.
    """

    # Constants
    RESULTS_FILE = "results.yaml"
    METADATA_KEY = "collectra_results_metadata"
    PIPELINE_FILE = "pipeline.yaml"
    PIPELINE_METADATA_KEY = "collectra_pipeline_metadata"

    # Graph visualization colors
    COLOR_PROCESSING = "orange"
    _stage_gate: StageGate | None = None
    _run_progress: RunProgress | None = None
    COLOR_SUCCESS = "green"
    COLOR_FAILURE = "red"
    COLOR_FONT = "black"

    def __init__(
        self,
        name: str,
        ext: str,
        version: str,
        path: str | Path = "",
        pipeline_metadata: dict | None = None,
        **kwargs,
    ):
        """Initialize the Collectra workflow.

        Args:
            name (str): Name of the workflow.
            ext (str): File extension that the workflow accepts.
            version (str): Version of the workflow.
            path (str, optional): Path to the workflow files. Defaults to name.
            **kwargs: Additional configuration parameters.
        """
        self.name: str = name
        self.ext: str = ext
        self.version: str = version
        self.pipeline_metadata = {
            **(pipeline_metadata or {}),
            "name": name,
            "ext": ext,
            "version": version,
        }
        self.node_manager: NodeGraphManager = NodeGraphManager()
        self.path: Path = Path.cwd() / name if not path else Path(path)
        self.data: dict = kwargs

    def __call__(self, task_name: str, **kwargs):
        """Allow instance to be called directly to run workflow."""
        self.run(task_name, **kwargs)

    @classmethod
    def from_file(cls, pipeline_path: str | Path) -> "Collectra":
        """Load a pipeline from a directory or a pipeline YAML file."""
        path = Path(pipeline_path).expanduser().resolve()
        config_path = path / cls.PIPELINE_FILE if path.is_dir() else path
        with config_path.open() as stream:
            metadata = yaml.safe_load(stream) or {}
        initials = metadata.pop(cls.PIPELINE_METADATA_KEY)
        return cls(
            initials["name"],
            initials["ext"],
            initials["version"],
            path=config_path.parent,
            pipeline_metadata=initials,
            **metadata,
        )

    @command(name="run")
    def cli_run(
        self,
        inputs: list[str],
        task: str = "",
        output: Path | None = None,
        single: bool = False,
        verbose: bool = False,
        usage: bool = False,
        render: bool = False,
        force: Annotated[
            bool,
            Arg(
                help="Rerun tasks whose outputs already exist, replacing those outputs. By default such tasks are skipped."
            ),
        ] = False,
    ) -> None:
        """Run the complete pipeline or start at a named task."""
        from collectra.logger import setup_logging
        from collectra.utils import resolve_files, valid_raw_files

        setup_logging(verbose=verbose)
        files = resolve_files(inputs, [f".{self.ext}", *valid_raw_files()])
        if not files:
            logger.warning("No supported input files found for: %s", ", ".join(inputs))
            return
        self.run(
            task,
            files=files,
            output=str(output) if output else None,
            single=single,
            usage=usage,
            render=render,
            force=force,
        )

    @command(name="path")
    def cli_path(self) -> None:
        """Print the folder this pipeline lives in."""
        print(self.path.resolve())

    @command(name="code", available=lambda: shutil.which("code"))
    def cli_code(self) -> None:
        """Open this pipeline's folder in VS Code."""
        import subprocess

        code = shutil.which("code")
        if code is None:
            raise SystemExit("VS Code's `code` command was not found on your PATH")
        result = subprocess.run([code, str(self.path.resolve())])
        if result.returncode:
            raise SystemExit(f"Could not open VS Code (exit code {result.returncode})")

    @command(name="gui")
    def launch_gui(
        self, *inputs: Path, debug: bool = False,
        max_pixels: Annotated[int, Arg(help="Maximum image and mask preview dimension in pixels (longest edge). Originals remain full resolution.")] = 4096,
    ) -> None:
        """Open the GUI with optional result folders, results.yaml files, or parent directories."""
        from collectra.gui.backend import start

        if max_pixels < 1:
            raise ValueError("--max-pixels must be positive")
        self._ensure_workflow_connected()
        start(pipeline=self, debug=debug, inputs=inputs, max_pixels=max_pixels)

    @command(name="partition")
    def cli_partition(
        self,
        inputs: Annotated[
            list[str],
            Arg(help="Result folders, results.yaml files, or parent directories."),
        ],
        train: Annotated[
            str,
            Arg(help="Share of results labelled 'train' (e.g. 80%, 0.8, or 100)."),
        ] = "",
        validation: Annotated[
            str,
            Arg(help="Share of results labelled 'validation' (e.g. 20%, 0.2, or 25)."),
        ] = "",
        val: Annotated[
            str, Arg(help="Share of results labelled 'val' (e.g. 20%, 0.2, or 25).")
        ] = "",
        test: Annotated[
            str, Arg(help="Share of results labelled 'test' (e.g. 10%, 0.1, or 10).")
        ] = "",
        seed: Annotated[
            int, Arg(help="Random seed used to shuffle results before splitting.")
        ] = 42,
        partition: Annotated[
            list[str] | None,
            Arg(
                help="Custom partition as NAME=VALUE (e.g. holdout=5%); may be repeated."
            ),
        ] = None,
        force: Annotated[
            bool,
            Arg(
                help="Override existing partitions instead of refusing to change them."
            ),
        ] = False,
    ) -> None:
        """Assign result partitions using percentages, fractions, or file counts.

        For example: partition results --train 80% --validation 20%.
        Use repeated --partition NAME=VALUE options for custom labels.
        All values must use the same form (percentages, fractions, or counts).
        Fails if any result already has a partition unless --force is given.
        Unassigned results retain their existing partition.
        """
        from collectra.partition import partition_results

        partitions = {
            label: value
            for label, value in (
                ("train", train),
                ("validation", validation),
                ("val", val),
                ("test", test),
            )
            if value != ""
        }
        for specification in partition or []:
            label, separator, value = specification.partition("=")
            if not separator or not label.strip() or not value.strip():
                raise ValueError("Custom partitions must use NAME=VALUE")
            if label in partitions:
                raise ValueError(f"Partition {label!r} was specified more than once")
            partitions[label] = value
        partition_results(inputs, self.ext, partitions, seed, force=force)

    @command(name="install")
    def cli_install(self, name: str, bin_dir: Path | None = None) -> None:
        """Install this pipeline as a standalone command."""
        destination = (bin_dir or Path.home() / ".local" / "bin").expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        launcher = destination / name
        pipeline_path = (self.path / self.PIPELINE_FILE).resolve()
        code = (
            "from pathlib import Path\n"
            "from collectra.main import main\n"
            f"main(pipeline_path=Path({str(pipeline_path)!r}), program_name={name!r})\n"
        )
        # Kernel shebang parsing cannot quote an interpreter path containing
        # spaces. Let the shell invoke the exact interpreter as one argument.
        launcher.write_text(
            "#!/bin/sh\n"
            f'exec {shlex.join([sys.executable, "-c", code])} "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(
            launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        print(f"Installed {name!r} at {launcher}")

    @command(name="install-app")
    def cli_install_app(
        self,
        *inputs: Path,
        name: str = "Collectra",
        app_dir: Path | None = None,
        icon: Path | None = None,
        force: bool = False,
    ) -> Path:
        """Create a macOS Applications launcher for this pipeline's GUI.

        Optional inputs are result folders or files opened when the app starts.
        The default destination is ~/Applications; icon accepts an .icns or PNG file. Pipeline metadata supplies the default icon.
        """
        from .app_launcher import install_app

        pipeline_icon = self.pipeline_metadata.get("icon") or self.pipeline_metadata.get("logo")
        if icon is None and pipeline_icon:
            icon = Path(pipeline_icon).expanduser()
            if not icon.is_absolute():
                icon = self.path / icon
        bundle = install_app(
            self.path / self.PIPELINE_FILE,
            name=name,
            app_dir=app_dir,
            icon=icon,
            inputs=inputs,
            force=force,
        )
        print(f"Installed {name!r} at {bundle}")
        return bundle

    @group(name="task")
    def cli_tasks(self) -> dict[str, Task]:
        """Operate on one of the tasks in this pipeline."""
        self._ensure_workflow_connected()
        return {
            node.name: node.get_task() for node in self.node_manager.get_task_nodes()
        }

    @group(name="artefact")
    def cli_artefacts(self) -> dict[str, object]:
        """Operate on one of the artefacts in this pipeline."""
        self._ensure_workflow_connected()
        artefacts: dict[str, object] = {}
        for node in self.node_manager.get_artefact_nodes():
            artefact_types = [
                type_ for type_ in node.types if issubclass(type_, Artefact)
            ]
            if not artefact_types:
                continue
            # IO inference can add a base and a more-specific type to a node.
            # Prefer the most specific class so its CLI capabilities win.
            artefact_type = max(
                artefact_types,
                key=lambda type_: len(type_.mro()),
            )
            artefacts[node.name] = artefact_type.cli_commands(node, self.ext)
        return artefacts

    def task(self, task_name: str) -> dict:
        """Get a task from the workflow by name.

        Args:
            task_name (str): Name of the task to retrieve.

        Returns:
            dict | None: Task configuration dictionary or None if not found.
        """
        data = self.data.get(task_name, None)
        if not data:
            raise ValueError(f"Task {task_name} not found in workflow")
        data["name"] = task_name
        return data

    def _check_is_root(self, node_name: str) -> bool:
        """Check if a node is a root node (no parents or only data parents)."""
        node = self.node_manager.resolve_node(node_name)
        parents = self.node_manager.get_parents(node)
        for parent in parents:
            grandparents = self.node_manager.get_parents(parent)
            if len(grandparents) > 0:
                return False
            if isinstance(parent, TaskNode):
                return False
        return True

    def run(self, task_name: str = "", **kwargs):
        """Run the workflow from the specified task or from all root tasks.

        Args:
            task_name: Name of the task to start from. If empty, starts from
                all root tasks in the workflow.
            **kwargs: Run options including:
                - single: Run only the starting task without propagating.
                - files: List of file paths to process.
                - usage: Whether to track usage metrics.
                - render: Whether to render workflow visualization.
                - force: Rerun tasks whose outputs already exist.
                - output: Output directory for results.
        """
        try:
            self._ensure_workflow_connected()
            starting_nodes = self._get_starting_nodes(task_name)
            run_options = self._extract_run_options(kwargs)
            self._process_files(starting_nodes, run_options)
        except Exception as e:
            logger.error("Workflow run failed: %s", e, exc_info=True)

    def _ensure_workflow_connected(self):
        """Ensure the workflow graph is connected before execution.

        If the workflow graph has no nodes, calls connect() to build
        the graph from the workflow configuration.
        """
        if self.node_manager.is_empty():
            self.connect()

    def _get_starting_nodes(self, task_name: str) -> list[TaskNode]:
        """Get the starting nodes for workflow execution.

        Args:
            task_name: Specific task to start from, or empty for all root tasks.

        Returns:
            List of starting task nodes.
        """
        if task_name:
            return self._get_specific_starting_node(task_name)
        return self._get_root_starting_nodes()

    def _get_specific_starting_node(self, task_name: str) -> list[TaskNode]:
        """Get a specific task node as the starting point."""
        task_node = self.node_manager.resolve_node(task_name)
        if not isinstance(task_node, TaskNode):
            raise TypeError(f"Task {task_name} not found in workflow")
        return [task_node]

    def _get_root_starting_nodes(self) -> list[TaskNode]:
        """Get all root task nodes as starting points."""
        starting_nodes = []
        task_nodes = self.node_manager.get_task_nodes()
        for task_node in task_nodes:
            if self._check_is_root(task_node.name):
                starting_nodes.append(task_node)
        return starting_nodes

    def _extract_run_options(self, kwargs: dict) -> dict:
        """Extract and return run options from kwargs."""
        options = {
            "single": kwargs.pop("single", False),
            "files": kwargs.pop("files", []),
            "usage": kwargs.pop("usage", False),
            "render": kwargs.pop("render", False),
            "force": kwargs.pop("force", False),
            "output": kwargs.pop("output", None),
        }
        options["task_overrides"] = kwargs.copy()
        return options

    def _get_root_input(self) -> str:
        """Get the name of the root input artefact node for the workflow."""
        for artefact_node in self.node_manager.get_artefact_nodes():
            if self._check_is_root(artefact_node.name):
                return artefact_node.name
        raise ValueError("No root input artefact node found in the workflow")

    def _process_files(self, starting_nodes: list[TaskNode], options: dict):
        """Process each file in the workflow.

        Files are processed in chunks sized to the largest ``batch_size`` of any
        task that can batch its work, so those tasks see a whole chunk at once.

        Args:
            starting_nodes: List of nodes to start execution from.
            options: Execution options (single, files, usage, render, output).
        """
        files = options["files"]
        downstream = {} if options["single"] else self._batchable_tasks(starting_nodes)
        batch_size = self._starting_batch_size(starting_nodes)
        batch_size = max([batch_size, *(t.batch_size for t in downstream.values())])
        overall = (
            Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            if len(files) > 1
            else None
        )
        with overall if overall is not None else nullcontext():
            if overall is not None:
                label = ", ".join(node.name for node in starting_nodes) or self.name
                self._run_progress = RunProgress(overall, label, len(files))
            try:
                for start in range(0, len(files), batch_size):
                    chunk = files[start : start + batch_size]
                    self._process_chunk(starting_nodes, chunk, options, downstream)
            finally:
                self._run_progress = None

    def _process_chunk(
        self,
        starting_nodes: list[TaskNode],
        chunk: list,
        options: dict,
        downstream: dict[str, Task],
    ):
        """Run a chunk of files, pausing each at batchable tasks until all arrive.

        Every file runs until it reaches a batchable task further down the DAG
        and is saved there. Once the whole chunk has arrived the task prefetches
        all of its inputs together, and each file resumes from its saved results.
        """
        stage = [(file_path, starting_nodes) for file_path in chunk]
        stage_options = options
        prefetch = self._starting_batch_size(starting_nodes) > 1
        self._stage_gate = StageGate(set(downstream)) if downstream else None
        try:
            while stage:
                if prefetch:
                    self._announce_batch(stage)
                    self._prefetch_stage(stage, stage_options)
                deferred: list[tuple[str, list[TaskNode]]] = []
                for file_path, nodes in stage:
                    if self._run_progress:
                        self._run_progress.status(Path(str(file_path)).name)
                    if self._stage_gate:
                        self._stage_gate.current = {node.name for node in nodes}
                        self._stage_gate.deferred = {}
                    try:
                        self._process_single_file(nodes, file_path, stage_options)
                    except Exception as e:
                        logger.error("Error processing file: %s", e, exc_info=True)
                    held = self._stage_gate.deferred if self._stage_gate else {}
                    deferred.extend(held.items())
                    if self._run_progress and not held:
                        self._run_progress.file_done()
                # Later stages resume from each file's saved folder, in place.
                stage = deferred
                stage_options = {**options, "output": None}
                prefetch = True
        finally:
            self._stage_gate = None
            self._clear_prefetch(
                [*starting_nodes, *self._task_nodes_by_name(downstream)]
            )

    def _announce_batch(self, stage: list[tuple[str, list[TaskNode]]]) -> None:
        """Say on the overall bar that a batch is being prepared with the model."""
        if not self._run_progress:
            return
        names = sorted(
            {
                node.name
                for _, nodes in stage
                for node in nodes
                if node.get_task().batch_size > 1
            }
        )
        if names:
            count = len(stage)
            self._run_progress.status(
                f"waiting on batch of {count} file{'s' if count != 1 else ''} "
                f"({', '.join(names)})"
            )

    def _task_nodes_by_name(self, tasks: dict[str, Task]) -> list[TaskNode]:
        return [self.node_manager.resolve_node(name) for name in tasks]

    def _starting_batch_size(self, starting_nodes: list[TaskNode]) -> int:
        """Number of files to run through the starting tasks per batch."""
        sizes = [
            int(getattr(node.get_task(), "batch_size", 1) or 1)
            for node in starting_nodes
            if isinstance(node, TaskNode)
        ]
        return max([1, *sizes])

    def _batchable_tasks(self, starting_nodes: list[TaskNode]) -> dict[str, Task]:
        """Tasks below ``starting_nodes`` that batch their work via ``prefetch``."""
        start_names = {node.name for node in starting_nodes}
        found: dict[str, Task] = {}
        seen: set[str] = set()
        queue = list(starting_nodes)
        while queue:
            node = queue.pop()
            if node.name in seen:
                continue
            seen.add(node.name)
            queue.extend(self.node_manager.get_children(node))
            if not isinstance(node, TaskNode) or node.name in start_names:
                continue
            task = node.get_task()
            if (
                isinstance(task, Task)
                and task.batch_size > 1
                and type(task).prefetch is not Task.prefetch
            ):
                found[node.name] = task
        return found

    def _prefetch_stage(self, stage: list[tuple[str, list[TaskNode]]], options: dict):
        """Let each file's pending tasks batch their model work up front.

        The files are then processed one at a time as usual, with each task
        reusing what it cached, so results are saved after every batch.
        """
        gathered: dict[str, list] = {}
        nodes_by_name: dict[str, TaskNode] = {}
        for file_path, nodes in stage:
            try:
                self._setup_file(nodes, file_path, options)
                for node in nodes:
                    task = node.get_task()
                    if not isinstance(task, Task) or task.batch_size <= 1:
                        continue
                    if not options["force"] and self._has_output(node):
                        continue
                    self._apply_task_run_overrides(task, options["task_overrides"])
                    parents = self.node_manager.get_parents_artefact(node)
                    nodes_by_name[node.name] = node
                    gathered.setdefault(node.name, []).extend(
                        list(entry) for entry in task.prepare_inputs(parents)
                    )
            except Exception as e:
                logger.warning("Could not prepare %s for batching: %s", file_path, e)
        for name, node in nodes_by_name.items():
            try:
                with change_dir(self.path):
                    node.get_task().prefetch(gathered[name])
            except Exception as e:
                logger.warning(
                    "Batched run of %s failed, running files individually: %s",
                    name,
                    e,
                    exc_info=True,
                )
                node.get_task().clear_prefetch()

    def _clear_prefetch(self, starting_nodes: list[TaskNode]):
        for node in starting_nodes:
            task = node.get_task()
            if isinstance(task, Task):
                task.clear_prefetch()

    def _setup_file(
        self, starting_nodes: list[TaskNode], file_path: str, options: dict
    ) -> tuple[str, str | Path]:
        """Create the collectra file for ``file_path`` and load it into the graph."""
        key = (
            "file"
            if Path(file_path).suffix == f".{self.ext}"
            else self._get_root_input()
        )
        key, value = self._create_collectra_file(key, file_path, options["output"])

        # Create task context for this file
        context = TaskContext(
            usage_file=(
                Path.cwd() / str(Path(value) / "usage.yaml")
                if options["usage"]
                else None
            ),
            render=options["render"],
            file_path=Path(value),
            single_run=options["single"],
        )

        # Set context for all tasks
        self._set_task_contexts(context)

        self._init_input_data(starting_nodes, **{key: value})
        return key, value

    def _process_single_file(
        self, starting_nodes: list[TaskNode], file_path: str, options: dict
    ):
        """Process a single file through the workflow.

        Args:
            starting_nodes: Nodes to start from.
            file_path: Path to file to process.
            options: Execution options.
        """
        key, value = self._setup_file(starting_nodes, file_path, options)
        self.render("workflow_run", file=str(value), render=options["render"])

        self._run_nodes(
            starting_nodes,
            single=options["single"],
            key=key,
            value=value,
            render=options["render"],
            force=options["force"],
            task_overrides=options["task_overrides"],
        )
        self._warn_unready_starting_tasks(starting_nodes, value)

    def _warn_unready_starting_tasks(self, starting_nodes: list[TaskNode], value):
        """Explain why a task the user asked for never ran for this file.

        Tasks reached part-way through the DAG may legitimately wait for their
        other inputs, so only the tasks the run started from are reported.
        """
        for node in starting_nodes:
            if node.status == NodeStatus.READY:
                continue
            missing = [
                parent.name
                for parent in self.node_manager.get_parents_artefact(node)
                if parent.status != NodeStatus.READY
            ]
            lines = [
                f"  • [bold red]{escape(name)}[/bold red]" for name in missing[:10]
            ]
            if len(missing) > 10:
                lines.append(f"  … and {len(missing) - 10} more")
            logger.warning(
                "Task '%s' did not run for %s. "
                "A task only runs once all of its inputs exist.\nMissing input data:\n%s",
                node.name,
                Path(str(value)).name,
                "\n".join(lines) or "  • its inputs",
            )

    def _create_collectra_file(
        self, key: str, value: str | Path, output_dir: str | Path | None = None
    ) -> tuple[str, str | Path]:
        """Create or prepare a collectra file structure for processing.

        Handles three cases:
        - Existing collectra folder with results.yaml
        - Single image file
        - Directory without results.yaml
        """
        self._validate_file_inputs(key, value)
        savef = Path(value)

        # Case A: Existing collectra folder with results.yaml
        if self._is_existing_collectra_folder(savef):
            return key, self._handle_existing_folder(savef, output_dir)

        # Case B: Single image file
        if self._is_image_file(savef):
            savef = self._handle_image_file(savef, output_dir)

        # Case C: Directory without results.yaml - create results file
        if savef.is_dir():
            self._create_results_file(savef, key, value)

        return "file", savef

    def _validate_file_inputs(self, key: str, value: str | Path):
        """Validate input parameters for file creation."""
        if not key or not value:
            raise ValueError("Both key and value must be provided for input data.")

        savef = Path(value)
        if not savef.exists():
            raise FileNotFoundError(f"Path {savef} does not exist.")

    def _is_existing_collectra_folder(self, path: Path) -> bool:
        """Check if path is an existing collectra folder with results."""
        return (
            path.is_dir()
            and path.suffix == f".{self.ext}"
            and (path / self.RESULTS_FILE).exists()
        )

    def _is_image_file(self, path: Path) -> bool:
        """Check if path is an image file."""
        return path.is_file() and path.suffix.lower() in Image.image_types()

    def _handle_existing_folder(
        self, savef: Path, output_dir: str | Path | None
    ) -> Path:
        """Handle existing collectra folder, optionally copying to output."""
        if output_dir:
            output_path = Path(output_dir) / savef.name
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(savef, output_path, dirs_exist_ok=True)
            savef = output_path

        return savef

    def _handle_image_file(
        self, file_path: Path, output_dir: str | Path | None
    ) -> Path:
        """Handle single image file by creating collectra folder structure."""
        folder_name = file_path.name.replace(file_path.suffix, f".{self.ext}")
        savef = (
            Path(output_dir) / folder_name
            if output_dir
            else file_path.parent / folder_name
        )
        savef.mkdir(parents=True, exist_ok=True)

        # Keep the file exactly as supplied, EXIF orientation included: every
        # reader in Collectra applies it rather than stripping it.
        shutil.copy2(file_path, savef / file_path.name)
        return savef

    def _create_results_file(self, directory: Path, key: str, value: str | Path):
        """Create a new results.yaml file in the given directory."""
        with change_dir(directory):
            results = {
                self.METADATA_KEY: {
                    "workflow": self.name,
                    "version": self.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                key: {
                    "type": "collectra.Image",
                    "id": f"{key}",
                    "data": value.name if isinstance(value, Path) else str(value),
                },
            }
            self._write_results_file(results)

    def _populate_active_paths(
        self,
        nodes: list[TaskNode | ArtefactNode] | list[TaskNode] | list[ArtefactNode],
        **kwargs,
    ):
        """Recursively populate active paths in the workflow with input data.

        Traverses the workflow graph from the given nodes, processing artefact nodes
        and propagating values to their children.

        Args:
            nodes: List of nodes to start populating from.
            **kwargs: Key-value pairs of input data to populate into matching nodes.
        """
        for node in nodes:
            if isinstance(node, ArtefactNode):
                value = kwargs.get(node.name, None)
                node.process(node.name, value, **kwargs)

            children = self.node_manager.get_children(node)
            self._populate_active_paths(children, **kwargs)

    def _init_input_data(
        self,
        starting_nodes: (
            list[TaskNode | ArtefactNode] | list[TaskNode] | list[ArtefactNode]
        ),
        **kwargs,
    ):
        """Initialize input data for workflow execution.

        Resets all nodes, reconnects the workflow, and populates parent data
        nodes with the provided input values.

        Args:
            starting_nodes: List of nodes that will begin workflow execution.
            **kwargs: Key-value pairs of input data to initialize.
        """
        self._reset_nodes()
        self.connect()
        for node in starting_nodes:
            parents = self.node_manager.get_parents_artefact(node)
            for parent in parents:
                value = kwargs.get(parent.name, None)
                parent.process(parent.name, value, **kwargs)
        self._populate_active_paths(starting_nodes, **kwargs)
        self._resolve_links()

    def _resolve_links(self) -> None:
        """Bind loaded links to artefacts in the active pipeline graph."""
        artefacts = {
            item.id: item
            for node in self.node_manager.get_artefact_nodes()
            for item in node.items.values()
        }
        for node in self.node_manager.get_artefact_nodes():
            for item in node.items.values():
                if type(item) is Link:
                    item.bind(artefacts)

    def _set_task_contexts(self, context: TaskContext):
        """Set context for all tasks in the workflow.

        Args:
            context: TaskContext to apply to all tasks.
        """
        for task_node in self.node_manager.get_task_nodes():
            task = task_node.get_task()
            if isinstance(task, Task):
                task.context = context

    def _reset_nodes(self):
        """Reset all nodes in the workflow to their initial state.

        Clears items and ensemble items from artefact nodes and sets all node
        statuses to NOT_READY.
        """
        self.node_manager.reset_all_nodes()

    def save_run(
        self, key: str, value: str | Path, artefact_node: ArtefactNode | None = None
    ):
        """Save workflow execution results to file.

        Args:
            key: Key identifier for the data.
            value: Path where results should be saved.
            artefact_node: Specific artefact node to save, or None for all nodes.
        """
        if not self._should_save(key, value, artefact_node):
            return

        savef = Path(value)
        if not savef.is_dir():
            return

        artefact_nodes = self._get_artefact_nodes_to_save(artefact_node)
        self._save_results_to_directory(savef, key, value, artefact_nodes)
        self._log_save_success(savef, artefact_node)

    def _should_save(
        self, key: str, value: str | Path, artefact_node: ArtefactNode | None
    ) -> bool:
        """Check if saving should proceed."""
        if not key or not value:
            logger.info("No save location specified, skipping save.")
            return False

        if artefact_node is not None and artefact_node.status != NodeStatus.READY:
            logger.info("No new data for %s, skipping save.", artefact_node.name)
            return False

        return True

    def _get_artefact_nodes_to_save(
        self, artefact_node: ArtefactNode | None = None
    ) -> list[ArtefactNode]:
        """Get list of artefact nodes to save."""
        if artefact_node is not None:
            return [artefact_node]

        return self.node_manager.get_artefact_nodes()

    def _save_results_to_directory(
        self,
        directory: Path,
        key: str,
        value: str | Path,
        artefact_nodes: list[ArtefactNode],
    ):
        """Save results to directory with proper formatting."""
        with change_dir(directory):
            results = self._load_or_create_results(key, value)
            results = self._collect_node_data(results, artefact_nodes)
            self._write_results_file(results)

    def _load_or_create_results(self, key: str, value: str | Path) -> dict:
        """Load existing results file or create new results dictionary."""
        resultsf = Path(self.RESULTS_FILE)

        if resultsf.exists():
            with open(resultsf, "r") as f:
                results = yaml.safe_load(f)
            results[self.METADATA_KEY][
                "timestamp"
            ] = datetime.datetime.now().isoformat()
        else:
            results = {
                self.METADATA_KEY: {
                    "workflow": self.name,
                    "version": self.version,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                key: {
                    "type": "collectra.Image",
                    "data": value.name if isinstance(value, Path) else str(value),
                },
            }

        return results

    def _collect_node_data(
        self, results: dict, artefact_nodes: list[ArtefactNode]
    ) -> dict:
        """Collect data from all ready nodes and add to results."""
        for artefact_node in artefact_nodes:
            if artefact_node.status != NodeStatus.READY:
                continue

            name = artefact_node.name
            results[name] = self._serialize_node_items(artefact_node)

        return results

    def _serialize_node_items(self, artefact_node: ArtefactNode) -> dict | list:
        """Serialize all items in an artefact node."""
        serialized_items = [item.serialize() for item in artefact_node.items.values()]
        return serialized_items[0] if len(serialized_items) == 1 else serialized_items

    def _write_results_file(self, results: dict):
        """Write results dictionary to YAML file."""
        write_yaml(results, self.RESULTS_FILE)

    def _log_save_success(self, savef: Path, artefact_node: ArtefactNode | None):
        """Log successful save operation."""
        if artefact_node:
            logger.info("Results saved to %s for %s", savef, artefact_node.name)
        else:
            logger.info("All results saved to %s", savef)

    def _run_nodes(
        self,
        nodes: list[TaskNode | ArtefactNode] | list[TaskNode] | list[ArtefactNode],
        *args,
        **kwargs,
    ):
        """Execute a list of nodes (tasks or artefact nodes) in the workflow.

        This method recursively processes nodes, handling both task execution
        and artefact node updates. It maintains visualization state and manages
        child node execution.
        """
        single_run = kwargs.get("single", False)
        render = kwargs.get("render", False)

        self._mark_nodes_as_processing(nodes)
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        child_tasks = self._execute_nodes(nodes, args, kwargs, render)

        if child_tasks and not single_run:
            self._run_nodes(child_tasks, *args, **kwargs)

    def _mark_nodes_as_processing(
        self, nodes: list[TaskNode | ArtefactNode] | list[TaskNode] | list[ArtefactNode]
    ):
        """Mark nodes as currently being processed in the graph."""
        for node in nodes:
            self.node_manager.set_node_attr(
                str(node.name), color=self.COLOR_PROCESSING, fontcolor=self.COLOR_FONT
            )

    def _execute_nodes(
        self,
        nodes: list[TaskNode | ArtefactNode] | list[TaskNode] | list[ArtefactNode],
        args: tuple,
        kwargs: dict,
        render: bool,
    ) -> list[TaskNode]:
        """Execute all nodes and collect child tasks.

        Returns:
            List of child tasks that are ready to execute.
        """
        child_tasks_of_artefact_nodes = dict()

        for node in nodes:
            if isinstance(node, TaskNode):
                self._execute_task_node(node, args, kwargs, render)
            elif isinstance(node, ArtefactNode):
                child_tasks = self._execute_artefact_node(node, args, kwargs, render)
                child_tasks_of_artefact_nodes.update(child_tasks)

        return self._check_tasks_ready(list(child_tasks_of_artefact_nodes.values()))

    def _execute_task_node(
        self, node: TaskNode, args: tuple, kwargs: dict, render: bool
    ):
        """Execute a single task node."""
        if self._check_task_ready(node) != NodeStatus.READY:
            logger.info(f"[yellow]Task {node.name} is not ready, skipping.[/yellow]")
            return

        children = self._get_node_children(node)
        force = kwargs.get("force", False)
        if not force and self._has_output(node):
            existing = [
                f"{output.name} ({len(output.items)})"
                for output in self._output_nodes(node)
                if output.items
            ]
            logger.warning(
                "Skipping task '%s' for %s: output already exists in %s. "
                "Use --force to rerun it and replace that output.",
                node.name,
                Path(str(kwargs.get("value", ""))).name or "this file",
                ", ".join(existing),
            )
            self._update_node_status(node, True)
            self._run_nodes(children, **kwargs)
            return

        if self._stage_gate and self._stage_gate.defer(node, kwargs.get("value", "")):
            return

        previous = self._take_outputs(node, kwargs.get("value")) if force else None
        try:
            results = self._run_task(node, **kwargs)
        except BaseException:
            if previous is not None:
                previous.restore()
            raise
        if previous is not None:
            # A failed rerun must not destroy the output it was meant to replace.
            previous.discard() if results else previous.restore()

        self._update_node_status(node, bool(results))
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        self._run_nodes(children, *results, **kwargs)

    def _output_nodes(self, node: TaskNode) -> list[ArtefactNode]:
        return [
            child
            for child in self.node_manager.get_children(node)
            if isinstance(child, ArtefactNode)
        ]

    def _has_output(self, node: TaskNode) -> bool:
        """Whether any of a task's output nodes already holds items."""
        return any(output.items for output in self._output_nodes(node))

    def _take_outputs(self, node: TaskNode, folder) -> PreviousOutputs:
        """Empty a task's output nodes before a forced rerun replaces them.

        Each artefact is dropped from its node. If it keeps its content in a
        file, that exact file, as the artefact records it, is moved aside until
        the rerun succeeds. Nothing is inferred from folder names, so results
        written by hand or by older versions are handled the same way.
        """
        outputs = self._output_nodes(node)
        snapshots = [(output, dict(output.items)) for output in outputs]
        moved: list[tuple[Path, Path]] = []
        backup = None
        root = Path(str(folder)).resolve() if folder else None
        for output in outputs:
            for item in output.items.values():
                for file in item.stored_files(root) if root else []:
                    if not file.resolve().is_relative_to(root):
                        logger.warning(
                            "Not deleting %s: it is outside the results folder %s",
                            file,
                            root,
                        )
                        continue
                    already = any(file == original for _, original in moved)
                    if already or not file.is_file():
                        continue
                    if backup is None:
                        backup = Path(tempfile.mkdtemp(prefix="collectra-previous-"))
                    kept = backup / str(len(moved))
                    shutil.move(str(file), str(kept))
                    moved.append((kept, file))
        for output, _ in snapshots:
            output.items.clear()
        return PreviousOutputs(snapshots, moved, backup)

    def _execute_artefact_node(
        self, node: ArtefactNode, args: tuple, kwargs: dict, render: bool
    ) -> dict:
        """Execute a single artefact node and return child tasks.

        Returns:
            Dictionary of child task names to TaskNode objects.
        """
        # Process incoming data
        for arg in args:
            if (
                isinstance(arg, Artefact)
                and node.name == arg.get_name()
                and node.check_type(type(arg))
            ):
                self._check_existing(node, arg)
        self._name_new_items(node, args)
        folder = kwargs.get("value")
        if folder:
            for item in node.items.values():
                item.finalise(Path(str(folder)), node.name)

        # Mark as READY even if no items matched — parent task has completed
        if node.status == NodeStatus.NOT_READY:
            node.status = NodeStatus.READY

        # Save results
        self.save_run(
            kwargs.get("key", ""), kwargs.get("value", ""), artefact_node=node
        )

        # Update visualization
        self._update_node_status(node, True)
        self.render("workflow_run", file=kwargs.get("value", ""), render=render)

        # Collect child tasks
        children = self._get_node_children(node)
        return {child.name: child for child in children if isinstance(child, TaskNode)}

    def _get_node_children(
        self, node: TaskNode | ArtefactNode
    ) -> list[TaskNode | ArtefactNode]:
        """Get all children of a node."""
        return self.node_manager.get_children(node)

    def _update_node_status(self, node: TaskNode | ArtefactNode, success: bool):
        """Update node visualization status based on execution result."""
        color = self.COLOR_SUCCESS if success else self.COLOR_FAILURE
        self.node_manager.set_node_attr(
            str(node.name), color=color, fontcolor=self.COLOR_FONT
        )

    def _check_task_ready(self, task_node: TaskNode) -> NodeStatus:
        """Check if a task node is ready to execute.

        A task is ready when all its parent artefact nodes have READY status.

        Args:
            task_node: The task node to check.

        Returns:
            The updated status of the task node.
        """
        parents = self.node_manager.get_parents_artefact(task_node)
        if all(parent.status == NodeStatus.READY for parent in parents):
            task_node.status = NodeStatus.READY
        return task_node.status

    def _check_tasks_ready(self, task_nodes: list[TaskNode]) -> list[TaskNode]:
        """Filter task nodes to return only those that are ready to execute.

        Args:
            task_nodes: List of task nodes to check.

        Returns:
            List of task nodes that have READY status.
        """
        ready_tasks: list[TaskNode] = list()
        ready_tasks = [
            task_node
            for task_node in task_nodes
            if self._check_task_ready(task_node) == NodeStatus.READY
        ]
        return ready_tasks

    def _name_new_items(self, node: ArtefactNode, results: tuple) -> None:
        """Give automatically identified items in ``node`` their final ids.

        A lone item takes the node's name; otherwise items are numbered
        ``node-1``, ``node-2``... skipping ids already in use. Ids that were
        loaded or chosen explicitly never change, so existing references hold.
        ``results`` is everything the task produced, so references to a renamed
        item from its siblings are updated too.
        """
        fresh = [
            item for item in node.items.values() if getattr(item, "_auto_id", False)
        ]
        if not fresh:
            return
        fresh_objects = {id(item) for item in fresh}
        taken = {
            item.id for item in node.items.values() if id(item) not in fresh_objects
        }
        renamed: dict[str, str] = {}
        number = 1
        for item in fresh:
            if len(node.items) == 1:
                new_id = node.name
            else:
                while f"{node.name}-{number}" in taken:
                    number += 1
                new_id = f"{node.name}-{number}"
            taken.add(new_id)
            renamed[item.id] = new_id
        for result in results:
            if not isinstance(result, Artefact):
                continue
            if isinstance(result.parents, str):
                result.parents = renamed.get(result.parents, result.parents)
            else:
                result.parents = [
                    renamed.get(parent, parent) for parent in result.parents
                ]
        ordered = list(node.items.values())
        for item in fresh:
            item.id = renamed[item.id]
            item._auto_id = False
        node.items.clear()
        node.items.update({item.id: item for item in ordered})

    def _check_existing(self, node: ArtefactNode, new_item: Artefact) -> None:
        """Check if an identical item already exists in the artefact node.

        If an identical item is found, update the existing item with the new item's ID.
        If no identical item is found, add the new item to the artefact node.

        Args:
            node: The artefact node to check.
            new_item: The new data item to check or add.
        """
        items = node.items
        for existing_id, existing_item in items.items():
            if self._is_identical_item(existing_item, new_item):
                new_item.id = existing_item.id
                # Still provisional if the replaced item had not been named yet.
                new_item._auto_id = getattr(existing_item, "_auto_id", False)
                items[existing_id] = new_item
                return

        # No identical item found, add as new
        node.add_item(new_item)

    def _is_identical_item(self, original_item: Artefact, new_item: Artefact) -> bool:
        """Check if two data items are identical.

        Compares both parent relationships and serialized data attributes.
        """
        if set(new_item.parents) != set(original_item.parents):
            return False

        return self._compare_serialized_data(original_item, new_item)

    def _compare_serialized_data(
        self, original_item: Artefact, new_item: Artefact
    ) -> bool:
        """Compare serialized data of two items, ignoring IDs and parents."""
        temp_original = self._prepare_for_comparison(copy.deepcopy(original_item))
        temp_new = self._prepare_for_comparison(copy.deepcopy(new_item))

        return json.dumps(temp_original, sort_keys=True) == json.dumps(
            temp_new, sort_keys=True
        )

    def _prepare_for_comparison(self, item: Artefact) -> dict:
        """Prepare data item for comparison by normalizing and removing metadata."""
        serialized = item.serialize()
        serialized.pop("parents", None)
        serialized.pop("id", None)

        # Normalize string data for comparison
        if isinstance(serialized.get("data"), str):
            serialized["data"] = serialized["data"].strip().lower()

        return serialized

    def _apply_task_run_overrides(self, task: Task, overrides: dict) -> None:
        """Apply runtime CLI overrides to a task before execution."""
        if not overrides:
            return
        merged_overrides = self._merge_task_params(task.name, dict(overrides))
        for key, value in merged_overrides.items():
            setattr(task, key, value)

    def _run_task(self, task_node: TaskNode, **kwargs) -> list:
        """Execute a task node and return its output results.

        Prepares inputs from parent artefact nodes, runs the task, and collects
        all output data items.

        Args:
            task_node: The task node to execute.
            **kwargs: Additional execution parameters.

        Returns:
            List of Artefact objects produced by the task execution.

        Raises:
            TypeError: If the node does not contain a valid Task.
        """
        task = task_node.get_task()
        if not isinstance(task, Task):
            raise TypeError(f"Node {task_node.name} is not a Task")
        logger.info("Attempting to run task: %s", task.name)
        self._apply_task_run_overrides(task, kwargs.get("task_overrides", {}))
        parents = self.node_manager.get_parents_artefact(task_node)
        entries = task.prepare_inputs(parents)
        results = list()
        if len(entries) == 0:
            logger.warning("No input data found for task %s, skipping.", task.name)
            return results
        # Inside an overall bar a per-task bar would only add noise.
        listed = (
            entries
            if self._run_progress
            else track(
                entries,
                total=len(entries),
                description=f"Running {task.name}",
            )
        )
        with change_dir(self.path):
            for entry in listed:
                if not isinstance(entry, list):
                    entry = list(entry)
                entry_result = self._execute_entries(entry, task)
                results.extend(entry_result)
        return results

    def _execute_entries(self, entries: list[Artefact], task: Task) -> list[Artefact]:
        """Execute a task with the given input entries.

        Runs the task with the provided data entries, sets parent relationships
        on output items, and handles any execution errors.

        Args:
            entries: List of input Artefact objects for the task.
            task: The task to execute.

        Returns:
            List of Artefact objects produced by the task, with parent relationships set.
        """
        output: Artefact | list[Artefact] | None = task.run(*entries)
        try:
            if output is None:
                return list()
            if not isinstance(output, list):
                output = [output]
            for result in output:
                result.set_parents(entries)
            return output
        except Exception as e:
            logger.error(
                "Error executing entries for task %s: %s (entries: %s)",
                task.name,
                e,
                entries,
                exc_info=True,
            )
            return list()

    def _merge_task_params(self, task_name: str, kwargs: dict) -> dict:
        """Merge task-specific parameters with provided kwargs."""
        task_params = self.data.get(task_name, dict()).get("params", dict())
        return task_params | kwargs

    def ensemble(self, folder: list[Path], output: str | Path):
        """Create ensembled files from source folders.

        Delegates to EnsembleProcessor for the actual ensemble logic.

        Args:
            folder: List of source folders containing collectra results
            output: Output folder path for ensembled results
        """
        from collectra.pipelines.ensemble import EnsembleProcessor

        self._ensure_workflow_connected()
        processor = EnsembleProcessor(
            node_manager=self.node_manager,
            ext=self.ext,
            name=self.name,
            version=self.version,
        )
        processor.ensemble(folder, Path(output))

    @property
    def metadata(self) -> dict:
        """Get pipeline metadata dictionary.

        Returns:
            Dictionary containing pipeline name, extension, and version
            under the PIPELINE_METADATA_KEY.
        """
        return {self.PIPELINE_METADATA_KEY: self.pipeline_metadata.copy()}

    def gui_graph(self) -> dict:
        """Return the live workflow graph in the shape expected by the GUI."""
        self._ensure_workflow_connected()
        nodes = []
        for name in self.node_manager.get_node_names():
            node = self.node_manager.resolve_node(name)
            inputs = list(self.node_manager.flow.predecessors(name))
            outputs = list(self.node_manager.flow.successors(name))
            params = {}

            if isinstance(node, TaskNode):
                task = node.get_task()
                detail = task.get_class_path()
                params = getattr(task, "params", {})
                if "LLM" in detail:
                    node_type = "llm"
                elif "ObjectDetection" in detail or "ImageClassifier" in detail:
                    node_type = "detector"
                else:
                    node_type = "processor"
            else:
                type_names = sorted(type_.get_class_path() for type_ in node.types)
                detail = ", ".join(type_names)
                node_type = "data" if node.items else "implicit"

            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "nodeType": node_type,
                    "detail": detail,
                    "inputs": inputs,
                    "outputs": outputs,
                    "params": params,
                }
            )

        edges = [
            {"source": source, "target": target}
            for source, target in self.node_manager.flow.edges
        ]
        return {"nodes": nodes, "edges": edges, "ext": self.ext}

    def save(self, task_name: str | None = None):
        """Save the workflow configuration to pipeline.yaml."""
        with change_dir(self.path):
            # Pull updated data from file if it exists to ensure we don't overwrite changes
            data_diff: dict | None = None
            if Path(self.PIPELINE_FILE).exists():
                with open(self.PIPELINE_FILE, "r") as f:
                    file_data = yaml.safe_load(f) or dict()
                    file_data.pop(self.PIPELINE_METADATA_KEY, None)
                # compare diffs between current data and file data, prioritizing file data for any keys that exist in both
                data_diff = dict()
                for key, data in file_data.items():
                    if key != task_name and file_data[key] != self.data.get(key, None):
                        data_diff[key] = data
            data_to_save = (
                self.data | data_diff if data_diff and len(data_diff) > 0 else self.data
            )
            config = self.metadata | data_to_save
            self._write_pipeline_config(config)

    def _write_pipeline_config(self, config: dict):
        """Write pipeline configuration to YAML file."""
        write_yaml(config, self.PIPELINE_FILE)

    def _get_io_list(self, io: list | str) -> list:
        """Convert an IO specification to a list format.

        Args:
            io: Either a single string or a list of IO identifiers.

        Returns:
            List containing the IO identifier(s).
        """
        return [io] if isinstance(io, str) else io

    def _check_task_io(self, io_type: str, task: Task, task_config: dict) -> list:
        """Extract IO node information from task configuration.

        Args:
            io_type: Either 'input' or 'output'.
            task: The task object.
            task_config: Configuration dictionary for the task.

        Returns:
            List of tuples containing (node_name, types).
        """
        if io_type not in task_config:
            return list()

        param_types = list(
            unpack_types(
                task.run, get_param_types if io_type == "input" else get_return_type
            ).items()
        )
        io = self._get_io_list(task_config.get(io_type, []))
        nodes: list = list()
        for item in io:
            for _, types in param_types:
                if not isinstance(types, tuple):
                    types = (types,)
                nodes.append((item, types))
        return nodes

    def _check_data_assignment(self, collectra_data: dict, data: dict) -> dict:
        """Check and resolve data references in configuration.

        If a value references another key in collectra_data, replace it with
        the actual value.

        Args:
            collectra_data: The main workflow data dictionary.
            data: The dictionary to check for references.

        Returns:
            Updated data dictionary with resolved references.
        """
        for key, value in data.items():
            if isinstance(value, str) and value in collectra_data:
                data[key] = collectra_data[value]
        return data

    def connect(self):
        """Initialise the DAG representing the workflow. Preparing it for execution."""
        relations: dict[str, dict] = dict()
        with change_dir(self.path):
            self._build_nodes_and_relations(relations)
            self._create_graph_edges(relations)

    def _build_nodes_and_relations(self, relations: dict):
        """Build workflow nodes and track their relationships."""
        for name, value in self.data.items():
            data = copy.deepcopy(value)
            if not isinstance(data, dict) or "type" not in data:
                continue

            cls_ = load_class_from_string(data.pop("type"))
            data = self._check_data_assignment(self.data, data)
            obj = self._create_or_reuse_object(name, cls_, data)

            if isinstance(obj, Task):
                self._process_task_node(obj, data, relations)
            else:
                self.node_manager.add_artefact_node(name, obj=obj)

    def _create_or_reuse_object(self, name: str, cls_: type, data: dict):
        """Create new object or reuse existing task from graph."""
        existing_task = self.node_manager.try_resolve_existing_task(name, cls_)
        if existing_task is not None:
            return existing_task
        return cls_(name, **data)

    def _process_task_node(self, task: Task, data: dict, relations: dict):
        """Process a task node by adding it to the graph and tracking IO."""
        task_key = task.name
        task.pipeline = self
        relations[task_key] = {
            "input": self._get_io_list(data.get("input", [])),
            "output": self._get_io_list(data.get("output", [])),
        }
        self.node_manager.add_task_node(task_key, task)

        # Add artefact nodes for task inputs and outputs
        ios = self._extract_task_io_nodes(task, data)
        for io_key, io_types in ios:
            self.node_manager.add_artefact_node(io_key, types=list(io_types))

    def _extract_task_io_nodes(
        self, task: Task, task_config: dict
    ) -> list[tuple[str, tuple]]:
        """Extract input and output node information from task configuration."""
        ios = []
        ios.extend(self._check_task_io("input", task, task_config))
        ios.extend(self._check_task_io("output", task, task_config))
        return ios

    def _create_graph_edges(self, relations: dict):
        """Create edges in the workflow graph based on task relationships."""
        for task_name, io in relations.items():
            inputs = io.get("input", [])
            outputs = io.get("output", [])

            for input_node in inputs:
                if self.node_manager.has_node(input_node):
                    self.node_manager.add_edge(input_node, task_name)

            for output_node in self._get_io_list(outputs):
                if self.node_manager.has_node(output_node):
                    self.node_manager.add_edge(task_name, output_node)

    def render(self, dest: str | Path = "", file="", render=False) -> None:
        """Render the workflow graph as an SVG visualization.

        Args:
            dest: Destination path for the rendered file.
            file: Optional file being processed (shown in title).
            render: Whether to actually render (allows conditional rendering).
        """
        if not render:
            return

        self._ensure_workflow_connected()
        visual_graph = self._prepare_graph_for_visualization()
        dot_str = self._generate_dot_string(visual_graph, file)

        dest_path = Path(dest) if dest else self.path / "workflow"
        graphviz.Source(dot_str).render(filename=dest_path, format="svg", cleanup=True)
        time.sleep(0.1)  # Brief pause for file system

    def _prepare_graph_for_visualization(self) -> nx.DiGraph:
        """Prepare a copy of the graph for visualization.

        Removes internal node objects that shouldn't be visualized.
        """
        visual_graph = self.node_manager.copy_graph()
        for node_id in visual_graph.nodes:
            node_data = visual_graph.nodes[node_id]
            if "node" in node_data:
                del node_data["node"]
        return visual_graph

    def _generate_dot_string(self, graph: nx.DiGraph, file: str = "") -> str:
        """Generate DOT format string from graph.

        Args:
            graph: The networkx graph to convert.
            file: Optional file name to include in graph title.

        Returns:
            DOT format string representation of the graph.
        """
        dot_str = nx.nx_pydot.to_pydot(graph).to_string()

        if file:
            title = f"{self.name} Workflow - Processing: {file}"
            dot_str = dot_str.replace(
                "{", f'{{\nlabel="{title}";\nlabelloc="t";\nfontsize=20;\n\n', 1
            )

        return dot_str
