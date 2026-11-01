#!/usr/bin/env python3
"""Bring Collectra results files up to date with the current header form.

A results file used to begin with a ``collectra_results_metadata:`` mapping.
The header is now an entry like any other, declared by its ``type``:

    results:
      type: collectra.Results
      partition: validation
      timestamp: '2026-02-13T17:36:26.080812'

Collectra still reads the old form, with a warning, and updates a file the next
time it saves it. This script updates the files you choose in one go, so the
warning goes away. Values, timestamp included, are kept as they are, and
nothing else in the file changes. A file that already has the new header, or has
no header, is not touched, so running the script again does nothing.

Examples:
    scripts/upgrade_results_files.py Page022.livy --dry-run
    scripts/upgrade_results_files.py /path/to/output

Exit status: 0 on success (including nothing to do), 2 when no results files
were found.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console

from collectra.commons.results import upgrade_header
from collectra.utils import write_yaml


def find_results_files(paths: list[Path]) -> list[Path]:
    """Expand files, result folders and parent folders into results.yaml files."""
    found: list[Path] = []
    for path in map(Path, paths):
        if path.is_file():
            found.append(path)
        elif (path / "results.yaml").is_file():
            found.append(path / "results.yaml")
        elif path.is_dir():
            found.extend(sorted(path.rglob("results.yaml")))
    return found


def process_file(results_file: Path, dry_run: bool) -> tuple[bool, bool]:
    """Upgrade one file. Returns ``(needs_upgrade, written)``."""
    with open(results_file) as stream:
        data = yaml.safe_load(stream)
    upgraded = upgrade_header(data) if isinstance(data, dict) else None
    if upgraded is None:
        return False, False
    if dry_run:
        return True, False
    write_yaml(upgraded, results_file)
    return True, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace the old 'collectra_results_metadata' header of "
        "results files with a 'results' entry of type collectra.Results.",
        epilog="Exit status: 0 = ok, 2 = no results files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="results.yaml files, result folders, or folders containing them",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would change, write nothing"
    )
    parser.add_argument("--json", action="store_true", help="print JSON, not text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_files = find_results_files(args.paths)
    if not results_files:
        print("No results.yaml files found.", file=sys.stderr)
        return 2

    changed: list[tuple[Path, bool]] = []
    for results_file in results_files:
        needs_upgrade, written = process_file(results_file, args.dry_run)
        if needs_upgrade:
            changed.append((results_file, written))

    if args.json:
        print(
            json.dumps(
                [{"file": str(f), "applied": written} for f, written in changed],
                indent=2,
            )
        )
        return 0

    console = Console(width=None if sys.stdout.isatty() else 160)
    if args.dry_run and changed:
        console.print(
            f"Would upgrade {len(changed)} of {len(results_files)} results file(s):"
        )
        for results_file, _ in changed:
            console.print(f"  {results_file.parent.name}")
    elif changed:
        console.print(
            f"Upgraded {len(changed)} of {len(results_files)} results file(s)."
        )
    else:
        console.print(f"Nothing to upgrade in {len(results_files)} results file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
