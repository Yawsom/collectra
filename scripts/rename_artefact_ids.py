#!/usr/bin/env python3
"""Rename artefact ids in Collectra results files after their node.

An artefact that is alone in its node takes the node's name; if a node holds
several, they become ``{node}-1``, ``{node}-2``... in the order they appear in
the file. This is the same rule Collectra applies to new artefacts, so older
results (with ids such as ``tei-3f2a...``) can be brought into line.

References are rewritten too. Every ``parents`` and ``ensemble`` entry that
names a renamed artefact is updated, so lineage stays intact. References to ids
that are not in the file (for example ``other/results::id``) are left alone.

Nothing outside ``results.yaml`` is touched. In particular, files that a text
artefact points to (``path: tei/tei-3f2a.xml``) keep their names, since the
record refers to them by path. The file is rewritten with Collectra's own YAML
writer, and a file that is already in the right form is not rewritten at all,
so running the script again changes nothing.

A file is skipped, and reported, rather than risk wrong references when:
- the same id is used by more than one artefact, so a reference could mean either, or
- the new ids would collide (for example a node ``a`` with two items next to a
  node called ``a-1``).

Examples:
    scripts/rename_artefact_ids.py Page022.livy --dry-run
    scripts/rename_artefact_ids.py /path/to/output
    scripts/rename_artefact_ids.py /path/to/output --node tei --node markdown

Exit status: 0 on success (including nothing to do), 1 if any file was
skipped, 2 when no results files were found.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from collectra.commons.results import is_header
from collectra.utils import write_yaml

METADATA_KEY = "collectra_results_metadata"


def _is_header(key: str, value) -> bool:
    """The file's own header (either form) is not an artefact to rename."""
    return key == METADATA_KEY or is_header(value)


class Rename(BaseModel):
    """One artefact whose id changes (``old`` is None if it had no id)."""

    node: str
    old: str | None
    new: str
    position: int = Field(exclude=True)  # 1-based place among the node's records


class Skipped(Exception):
    """A file that cannot be renamed safely; the message says why."""


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


def _records(value) -> list[dict]:
    """The artefact records held by one node (a single mapping or a list)."""
    entries = value if isinstance(value, list) else [value]
    return [e for e in entries if isinstance(e, dict) and "type" in e]


def plan_renames(data: dict, nodes: list[str] | None = None) -> list[Rename]:
    """The id each artefact should have, for those whose id would change.

    Raises ``Skipped`` if the file cannot be renamed without making references
    ambiguous or ids collide.
    """
    renames: list[Rename] = []
    final: list[str] = []
    for key, value in data.items():
        if _is_header(key, value):
            continue
        records = _records(value)
        selected = nodes is None or key in nodes
        for position, record in enumerate(records, start=1):
            old = record.get("id")
            new = (
                (key if len(records) == 1 else f"{key}-{position}") if selected else old
            )
            if new is not None:
                final.append(str(new))
            if selected and old != new:
                renames.append(Rename(node=key, old=old, new=new, position=position))

    old_ids = [
        str(r["id"])
        for k, v in data.items()
        if not _is_header(k, v)
        for r in _records(v)
        if "id" in r
    ]
    repeated = sorted({i for i in old_ids if old_ids.count(i) > 1})
    if repeated and renames:
        raise Skipped(
            "id(s) used by more than one artefact, so references are ambiguous: "
            + ", ".join(repeated)
        )
    clashes = sorted({i for i in final if final.count(i) > 1})
    if clashes:
        raise Skipped("new ids would collide: " + ", ".join(clashes))
    return renames


def plan_ids(data: dict, nodes: list[str] | None = None) -> dict[str, str]:
    """``{old id: new id}`` for every artefact that has an id and is renamed."""
    return {r.old: r.new for r in plan_renames(data, nodes) if r.old is not None}


def _remap(value, mapping: dict[str, str]):
    if isinstance(value, list):
        return [mapping.get(v, v) if isinstance(v, str) else v for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def apply_renames(data: dict, renames: list[Rename]) -> None:
    """Rewrite ids, and every reference to them, in ``data`` (in place)."""
    mapping = {r.old: r.new for r in renames if r.old is not None}
    unnamed = {(r.node, r.position): r.new for r in renames if r.old is None}
    for key, value in data.items():
        if _is_header(key, value):
            continue
        for position, record in enumerate(_records(value), start=1):
            if record.get("id") is None:
                if (key, position) in unnamed:
                    record["id"] = unnamed[(key, position)]
            elif record["id"] in mapping:
                record["id"] = mapping[record["id"]]
        # References, in every entry that carries them.
        for entry in value if isinstance(value, list) else [value]:
            if not isinstance(entry, dict):
                continue
            for field in ("parents", "ensemble"):
                if field in entry:
                    entry[field] = _remap(entry[field], mapping)


def process_file(
    results_file: Path, nodes: list[str] | None, dry_run: bool
) -> tuple[list[Rename], bool]:
    """Plan and (unless ``dry_run``) apply the renames for one file.

    Returns ``(renames, applied)``. Raises ``Skipped`` for unsafe files.
    """
    with open(results_file) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        return [], False
    renames = plan_renames(data, nodes)
    if not renames or dry_run:
        return renames, False
    apply_renames(data, renames)
    write_yaml(data, results_file)
    return renames, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename artefact ids after their node: 'node' if alone, "
        "else 'node-1', 'node-2'...",
        epilog="Exit status: 0 = ok, 1 = some file skipped, 2 = no results files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="results.yaml files, result folders, or folders containing them",
    )
    parser.add_argument(
        "--node",
        action="append",
        metavar="NAME",
        help="only rename the artefacts of this node (repeatable); references "
        "to them are still updated everywhere",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would change, write nothing"
    )
    parser.add_argument("--json", action="store_true", help="print JSON, not a table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_files = find_results_files(args.paths)
    if not results_files:
        print("No results.yaml files found.", file=sys.stderr)
        return 2

    changed: list[tuple[Path, list[Rename], bool]] = []
    problems: list[tuple[Path, str]] = []
    for results_file in results_files:
        try:
            renames, applied = process_file(results_file, args.node, args.dry_run)
        except Skipped as reason:
            problems.append((results_file, str(reason)))
            continue
        if renames:
            changed.append((results_file, renames, applied))

    if args.json:
        payload = [
            {
                "file": str(f),
                "renamed": [r.model_dump() for r in renames],
                "applied": applied,
            }
            for f, renames, applied in changed
        ] + [{"file": str(f), "error": reason} for f, reason in problems]
        print(json.dumps(payload, indent=2))
        return 1 if problems else 0

    console = Console(width=None if sys.stdout.isatty() else 160)
    total = sum(len(renames) for _, renames, _ in changed)
    if args.dry_run and changed:
        table = Table(title="Would rename")
        for column in ("Results", "Node", "Old id", "New id"):
            table.add_column(column)
        for results_file, renames, _ in changed:
            for rename in renames:
                table.add_row(
                    results_file.parent.name,
                    rename.node,
                    rename.old or "(none)",
                    rename.new,
                )
        console.print(table)
        console.print(
            f"Would rename {total} artefact id(s) in {len(changed)} of "
            f"{len(results_files)} results file(s)."
        )
    elif changed:
        console.print(
            f"Renamed {total} artefact id(s) in {len(changed)} of "
            f"{len(results_files)} results file(s)."
        )
    else:
        console.print(f"Nothing to rename in {len(results_files)} results file(s).")
    for results_file, reason in problems:
        console.print(f"[red]Skipped {results_file.parent.name}:[/red] {reason}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
