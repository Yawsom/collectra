"""Collectra command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .cli import invoke

if TYPE_CHECKING:
    from .pipelines.base import Pipeline


def find_pipeline(argv: list[str]) -> tuple[Path, list[str]]:
    """Extract ``--pipeline PATH`` while preserving all other arguments."""
    remaining = list(argv)
    for index, argument in enumerate(remaining):
        if argument == "--pipeline":
            if index + 1 >= len(remaining):
                raise SystemExit("--pipeline requires a path")
            path = Path(remaining[index + 1])
            del remaining[index : index + 2]
            return path, remaining
        if argument.startswith("--pipeline="):
            value = argument.partition("=")[2]
            if not value:
                raise SystemExit("--pipeline requires a path")
            del remaining[index]
            return Path(value), remaining
    raise SystemExit("Missing required option: --pipeline PATH")


def resolve_workflow_path(workflow: Path) -> "Pipeline":
    """Backward-compatible name for loading a pipeline."""
    from .pipelines.base import Pipeline

    return Pipeline.from_file(workflow)


def main(
    argv: list[str] | None = None,
    *,
    pipeline_path: Path | None = None,
    program_name: str = "collectra",
):
    """Load the selected pipeline and delegate its runtime object to Cappa."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if pipeline_path is None:
        pipeline_path, arguments = find_pipeline(arguments)
    from .pipelines.base import Pipeline

    pipeline = Pipeline.from_file(pipeline_path)
    return invoke(pipeline, arguments, name=program_name)


if __name__ == "__main__":
    main()
